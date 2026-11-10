"""Experimental ILS framework with ranked bins and state-keyed search reuse.

Original shapes and geometry rules must remain immutable during one run.
Weight is a fixed factor times piece volume. A supplied geometry search must
be deterministic; False means no candidate found by that search, not a proof
that no geometric placement exists. Existing ILS functions are untouched.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import numpy as np
from radio_bin_priority import post_placement_score


def geometry_search_config(size, strategy, density=5, axis='z', shape='cube',
                           selection_type='bounding_box', accessible=False,
                           encourage_dbl=True, select_range=None, nfv_pool=False):
    """Use one exact geometry signature in both construction and repacking."""
    return (tuple(int(v) for v in size), strategy, density, axis, shape,
            selection_type, accessible, encourage_dbl, select_range, nfv_pool)


@dataclass
class Solution:
    layout: list
    occupancy: list
    radio: list
    order: list


class RankedRepacker:
    def __init__(self, objects, container_size, container_volume, max_radio,
                 max_weight, search, rotate, translate, weight_factor=1.0,
                 rho=1.0, search_config=(), event=None):
        if min(container_volume, max_radio, max_weight, weight_factor) <= 0:
            raise ValueError('Capacities and weight factor must be positive')
        self.objects = objects
        self.size = tuple(container_size)
        self.volume, self.max_radio, self.max_weight = container_volume, max_radio, max_weight
        self.factor, self.rho = weight_factor, rho
        self.search, self.rotate, self.translate = search, rotate, translate
        self.config = tuple(search_config)
        self.cache = {}
        self.cache_origin = {}
        self.emit = event or (lambda **values: None)

    @staticmethod
    def prefix_key(layout, ids):
        # With immutable original shapes, identity + orientation + translation
        # uniquely specifies each placed shape and its occupancy.
        return tuple((int(i), str(info['orientation']), tuple(int(v) for v in info['translation']))
                     for i, info in zip(ids, layout))

    def record_ca_search(self, piece, info, layout, order, b, result, config):
        """Seed only actual failed SC queries with matching geometry settings.

        Called immediately after the CA search, before committing the current
        piece. Capacity rejections, unvisited bins and empty-bin policies are
        never imported as geometry failures. The original pieces are immutable
        and this engine/cache belongs to one instance and construction run.
        """
        if tuple(config) != self.config or isinstance(result, tuple) or not layout[b]:
            return
        orientation = info.get('orientation', 'x_0')
        key = (int(piece), orientation, int(b), self.prefix_key(layout[b], order[b]), self.config)
        self.cache[key] = None
        self.cache_origin[key] = 'CA'
        state_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        self.emit(event='CA_SEARCH_FAILED', piece=int(piece), bin=int(b)+1,
                  orientation=orientation, state_id=state_id)

    def validate(self, solution):
        ids = [i for group in solution.order for i in group]
        if sorted(ids) != list(range(len(self.objects))):
            raise ValueError('Solution must contain every piece exactly once')
        for b, group in enumerate(solution.layout):
            volume = sum(p['volume'] for p in group)
            if (volume > self.volume*self.rho or volume*self.factor > self.max_weight
                    or sum(p['radio'] for p in group) > self.max_radio):
                raise ValueError(f'Bin {b+1} exceeds a capacity')
            if np.any(solution.occupancy[b] > 1):
                raise ValueError('Overlapping pieces')

    def repack(self, reference, selected, orientations):
        old = {i:(b,info) for b,ids in enumerate(reference.order)
               for i,info in zip(ids,reference.layout[b])}
        layout = [[] for _ in reference.layout]
        occupancy = [np.zeros(self.size) for _ in layout]
        order = [[] for _ in layout]
        radio = [[] for _ in layout]
        changed = [False for _ in layout]
        volumes = [0.0 for _ in layout]

        def place(i,b,info):
            layout[b].append(dict(info)); occupancy[b] += info['array']
            order[b].append(i); radio[b].append(info['radio'])
            volumes[b] += info['volume']

        for i in range(selected):
            b,info = old[i]; place(i,b,info)
        changed[old[selected][0]] = True
        self.emit(event='REPACK_START', selected=selected,
                  reference_counts=[len(g) for g in reference.order],orientations=list(orientations))
        for i in range(selected,len(self.objects)):
            original_bin,old_info = old[i]
            orientation = orientations[i]
            if orientation != old_info['orientation']:
                changed[original_bin] = True
            piece = self.objects[i]
            eligible = []
            details = []
            for b in range(len(layout)):
                remaining = self.max_radio-sum(radio[b])-piece['radio']
                if (remaining < 0 or volumes[b]+piece['volume'] > self.volume*self.rho
                        or (volumes[b]+piece['volume'])*self.factor > self.max_weight):
                    self.emit(event='HARD_CONSTRAINT_REJECT', piece=i, bin=b+1)
                    continue
                score = post_placement_score(volumes[b],piece['volume'],self.volume,
                                              sum(radio[b]),piece['radio'],self.max_radio,
                                              min(1.,self.rho,self.max_weight/(self.factor*self.volume)))
                eligible.append((b,score))
                details.append(dict(bin=b+1,filling_rate=(volumes[b]+piece['volume'])/self.volume,
                    filling_rate_before=volumes[b]/self.volume,
                    filling_rate_after=(volumes[b]+piece['volume'])/self.volume,
                    current_radio=sum(radio[b]),piece_radio=piece['radio'],max_radio=self.max_radio,
                    radio_left_after=remaining,remaining_ratio=remaining/self.max_radio,score=score))
            ranked = sorted(eligible,key=lambda pair:(-pair[1],pair[0]))
            self.emit(event='BIN_RANK', piece=i, ranked=ranked,
                existing_bin_count=len(layout),
                details=sorted(details,key=lambda d:(-d['score'],d['bin'])))
            trial = None
            placed = False
            # A new bin is considered only after every eligible existing bin.
            for b,score in ranked + [(len(layout),None)]:
                if b == len(layout):
                    layout.append([]); occupancy.append(np.zeros(self.size))
                    order.append([]); radio.append([]); volumes.append(0.0); changed.append(True)
                    if (piece['radio'] > self.max_radio or piece['volume'] > self.volume*self.rho
                            or piece['volume']*self.factor > self.max_weight):
                        raise ValueError(f'Piece {i} cannot satisfy an empty bin capacity')
                prefix = self.prefix_key(layout[b],order[b])
                key = (i,orientation,b,prefix,self.config)
                state_id=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
                historical_prefix = tuple((j,info['orientation'],tuple(info['translation']))
                    for j,info in zip(reference.order[b],reference.layout[b]) if j<i) if b<len(reference.layout) else None
                # Reuse only matching states. Numeric bin order is irrelevant.
                if b==original_bin and orientation==old_info['orientation'] and prefix==historical_prefix:
                    info = dict(old_info)
                    self.emit(event='RESTORE_ORIGINAL',piece=i,bin=b+1)
                elif key in self.cache:
                    cached = self.cache[key]
                    if cached is None:
                        self.emit(event='SKIP_CACHED_FAILURE',piece=i,bin=b+1,changed=changed[b],
                            state_id=state_id,orientation=orientation,
                            source=self.cache_origin.get(key, 'REPACK'))
                        continue
                    info = dict(cached)
                    self.emit(event='REUSE_SUCCESS',piece=i,bin=b+1)
                else:
                    if trial is None:
                        axis,degree = orientation.split('_')
                        trial = dict(piece,orientation=orientation)
                        trial['array'] = self.rotate(piece['array'],int(degree),axis)
                    self.emit(event='SEARCH',piece=i,bin=b+1,changed=changed[b],state_id=state_id,orientation=orientation)
                    result = self.search(trial,layout,occupancy,b)
                    if not isinstance(result,tuple):
                        self.cache[key] = None
                        self.emit(event='SEARCH_FAILED',piece=i,bin=b+1,state_id=state_id,orientation=orientation)
                        continue
                    info = dict(trial,translation=tuple(result[1]))
                    info['array'] = self.translate(trial['array'],result[1])
                    self.cache[key] = dict(info)
                info['bin_position'] = b
                place(i,b,info)
                if b != original_bin:
                    changed[b] = changed[original_bin] = True
                elif (orientation != old_info['orientation'] or tuple(info['translation']) != tuple(old_info['translation'])):
                    changed[b] = True
                self.emit(event='PLACE',piece=i,bin=b+1,flags=list(changed))
                placed = True
                break
            if not placed:
                raise ValueError(f'Geometry search failed in empty bin for piece {i}')
        used = [b for b,ids in enumerate(order) if ids]
        result = Solution([layout[b] for b in used],[occupancy[b] for b in used],
                          [radio[b] for b in used],[order[b] for b in used])
        for b,group in enumerate(result.layout):
            for info in group: info['bin_position']=b
        self.validate(result)
        self.emit(event='REPACK_END',counts=[len(g) for g in result.order])
        return result


def improved_ILS_weight_radio(initial_solution, repacker, orientations,
                              objective, iterations=100, seed=13,
                              kick_trigger=20, selector=None):
    """Greedy local search plus kicks; initial CA is supplied by the caller.

    One evaluation changes one orientation; a kick changes about half the
    orientations and repacks from the earliest change. Global best is retained.
    Every piece is eligible: old CA last-bin exclusion is not assumed valid
    for ranked construction. An optional selector receives (state, rng, stage).
    """
    if iterations<1 or kick_trigger<1 or len(set(orientations))<2:
        raise ValueError('Positive budgets and at least two orientations required')
    repacker.validate(initial_solution)
    local = copy.deepcopy(initial_solution); best = copy.deepcopy(local)
    local_value = best_value = objective(local)
    rng = np.random.default_rng(seed)
    failures = 0
    history = []
    for evaluation in range(1,iterations+1):
        by_index = {i:p for ids,group in zip(local.order,local.layout) for i,p in zip(ids,group)}
        values = [by_index[i]['orientation'] for i in range(len(repacker.objects))]
        kick = failures>=kick_trigger
        if kick:
            indices = rng.choice(len(values),size=max(1,len(values)//2),replace=False)
        else:
            stage = max(0.0,1-failures/kick_trigger)
            if selector:
                selected = int(selector(local,rng,stage))
            else:
                weights = np.asarray([p['volume'] for p in repacker.objects],dtype=float)
                sizes = (weights-weights.min())/(np.ptp(weights)+1e-12)
                early = np.linspace(1,0,len(values)); late = 1-early
                weights = sizes+stage*early+(1-stage)*late+1e-12
                selected = int(rng.choice(len(values),p=weights/weights.sum()))
            indices = [selected]
        for i in indices:
            values[i] = str(rng.choice([o for o in orientations if o != values[i]]))
        candidate = repacker.repack(local,int(min(indices)),values)
        value = objective(candidate)
        accepted = kick or value<local_value
        if accepted: local,local_value,failures = candidate,value,0
        else: failures += 1
        if value<best_value: best,best_value = copy.deepcopy(candidate),value
        record = dict(evaluation=evaluation,kick=kick,accepted=accepted,objective=value,best=best_value)
        history.append(record); repacker.emit(event='EVALUATION',**record)
    return best,history
