"""GRASP with capacity-ranked bins and a run-scoped exact-state cache.

Original piece geometry is immutable within a run. Weight is volume times a
fixed factor. Cached failures mean this configured search found no position,
not that the region has no feasible position under every possible search.
Construction and descent use separate policy keys because their sampling
rules differ. The cache persists across every restart and LS evaluation.
"""
import copy
import math
import time
from collections import Counter

import numpy as np


class GlobalRankedPacking:
    """One instance per GRASP run; never share its cache across instances."""

    def __init__(self, objects, size, volume, max_radio, rho, max_weight,
                 weight_factor, search, rotate, translate, policy_keys, event=None):
        if not all(np.isfinite(v) and v > 0 for v in
                   (volume, max_radio, rho, max_weight, weight_factor)):
            raise ValueError('Capacities, rho and weight factor must be positive and finite')
        self.objects = copy.deepcopy(objects)
        self.size, self.volume = tuple(size), volume
        self.max_radio, self.rho = max_radio, rho
        self.max_weight, self.factor = max_weight, weight_factor
        self.search, self.rotate, self.translate = search, rotate, translate
        self.policy_keys = dict(policy_keys)
        self.event = event or (lambda **e: None)
        self.cache, self.stats = {}, Counter()
        for i, p in enumerate(self.objects):
            if not (np.isfinite(p['radio']) and p['radio'] >= 0
                    and np.isfinite(p['volume']) and p['volume'] > 0):
                raise ValueError(f'Piece {i}: invalid volume or radio')
            if not self.eligible(0, 0, p):
                raise ValueError(f'Piece {i} exceeds an empty bin capacity')

    def emit(self, name, **values):
        self.stats[name] += 1
        self.event(event=name, **values)

    def eligible(self, volume, radio, piece):
        return (radio + piece['radio'] <= self.max_radio
                and volume + piece['volume'] <= self.volume * self.rho
                and (volume + piece['volume']) * self.factor <= self.max_weight)

    @staticmethod
    def prefix_key(layout, ids):
        return tuple((int(i), p['orientation'], tuple(int(v) for v in p['translation']))
                     for i, p in zip(ids, layout))

    def ranked_bins(self, state, i):
        piece = self.objects[i]
        ranks = []
        for b, group in enumerate(state['layout']):
            vol = sum(p['volume'] for p in group)
            radio = sum(state['radio'][b])
            if not self.eligible(vol, radio, piece):
                self.emit('HARD_CONSTRAINT_REJECT', piece=i, bin=b)
                continue
            score = (vol / self.volume) * ((self.max_radio-radio-piece['radio']) / self.max_radio)
            ranks.append((b, score))
        ranks.sort(key=lambda x: (-x[1], x[0]))
        self.emit('BIN_RANK', piece=i, ranked=ranks)
        return [b for b, _ in ranks]

    def empty_state(self, bins=0):
        state = dict(layout=[], occupancy=[], radio=[], order=[])
        for _ in range(bins):
            self.add_bin(state)
        return state

    def add_bin(self, state):
        state['layout'].append([])
        state['occupancy'].append(np.zeros(self.size, dtype=np.uint8))
        state['radio'].append([])
        state['order'].append([])
        return len(state['layout'])-1

    def query(self, state, i, orientation, b, mode):
        # Capacity is checked on every visit, including cache hits.
        if not self.eligible(sum(p['volume'] for p in state['layout'][b]),
                             sum(state['radio'][b]), self.objects[i]):
            self.emit('HARD_CONSTRAINT_REJECT', piece=i, bin=b)
            return None
        prefix = self.prefix_key(state['layout'][b], state['order'][b])
        key = (i, orientation, b, prefix, self.policy_keys[mode])
        axis, degree = orientation.split('_')
        trial = dict(self.objects[i], orientation=orientation)
        trial['array'] = self.rotate(self.objects[i]['array'], int(degree), axis)
        if key in self.cache:
            result = self.cache[key]
            self.emit('SKIP_CACHED_FAILURE' if result is None else 'RESTORE_CACHED_SUCCESS',
                      piece=i, bin=b, mode=mode)
        else:
            self.emit('SEARCH', piece=i, bin=b, mode=mode)
            found = self.search(trial, state, b, mode, i)
            result = None
            if isinstance(found, tuple):
                score, position = found
                if not np.isfinite(score):
                    raise ValueError('Geometry search returned a nonfinite PES')
                position = tuple(int(v) for v in position)
                points = np.argwhere(trial['array'] != 0) + np.asarray(position)
                if not len(points) or np.any(points < 0) or np.any(points >= self.size):
                    raise ValueError('Geometry search returned an out-of-bounds placement')
                moved = self.translate(trial['array'], position)
                if np.any((moved != 0) & (state['occupancy'][b] != 0)):
                    raise ValueError('Geometry search returned an overlapping placement')
                # Store coordinates rather than full bin-sized voxel arrays.
                result = (float(score), position)
            self.cache[key] = result
            self.emit('SEARCH_FAILED' if result is None else 'SEARCH_SUCCEEDED',
                      piece=i, bin=b, mode=mode)
        if result is None:
            return None
        score, position = result
        info = dict(trial, array=self.translate(trial['array'], position), translation=position)
        return score, info

    def place(self, state, i, b, info):
        info = dict(info, bin_position=b)
        if not self.eligible(sum(p['volume'] for p in state['layout'][b]),
                             sum(state['radio'][b]), info):
            raise ValueError('Placement exceeds a bin capacity')
        if np.any((info['array'] != 0) & (state['occupancy'][b] != 0)):
            raise ValueError('Placement overlaps the current bin')
        state['layout'][b].append(info)
        state['occupancy'][b] += info['array'].astype(np.uint8)
        state['radio'][b].append(info['radio'])
        state['order'][b].append(i)

    def finish(self, state):
        used = [b for b, ids in enumerate(state['order']) if ids]
        state = {k: [state[k][b] for b in used] for k in state}
        for b, group in enumerate(state['layout']):
            for p in group:
                p['bin_position'] = b
        self.validate(state)
        return state

    def validate(self, state):
        ids = [i for group in state['order'] for i in group]
        if sorted(ids) != list(range(len(self.objects))):
            raise ValueError('Every piece must appear exactly once')
        for b, (group, indices) in enumerate(zip(state['layout'], state['order'])):
            if len(group) != len(indices):
                raise ValueError('Inconsistent bin records')
            vol = sum(p['volume'] for p in group)
            if (sum(state['radio'][b]) > self.max_radio or vol > self.volume*self.rho
                    or vol*self.factor > self.max_weight):
                raise ValueError('Bin exceeds a capacity')
            if np.any(state['occupancy'][b] > 1):
                raise ValueError('Bin contains overlapping voxels')

    def construct(self, orientations):
        state = self.empty_state()
        for i in range(len(self.objects)):
            placed = False
            bins = self.ranked_bins(state, i)
            bins.append(self.add_bin(state))
            for b in bins:
                choices = [self.query(state, i, o, b, 'CA') for o in orientations]
                choices = [x for x in choices if x is not None]
                if choices:
                    _, info = min(choices, key=lambda x: x[0])
                    self.place(state, i, b, info)
                    placed = True
                    break
            if not placed:
                raise ValueError(f'No construction orientation fits piece {i}')
            # Remove the unused new bin; keep numeric IDs stable for cache keys.
            if not state['order'][-1]:
                for values in state.values():
                    values.pop()
        return self.finish(state)

    def repack(self, reference, selected, orientations):
        state = self.empty_state(len(reference['layout']))
        old = {i: (b,p) for b, (group,ids) in enumerate(zip(reference['layout'],reference['order']))
               for i,p in zip(ids,group)}
        for i in range(selected):
            b,p = old[i]
            self.place(state, i, b, copy.deepcopy(p))
        for i in range(selected, len(self.objects)):
            bins = self.ranked_bins(state, i)
            bins.append(self.add_bin(state))
            placed = False
            for b in bins:
                # Exact successful reference state can restore its placement.
                old_b, info = old[i]
                prefix = self.prefix_key(state['layout'][b], state['order'][b])
                historic = tuple((j,p['orientation'],tuple(p['translation']))
                                 for j,p in zip(reference['order'][old_b],reference['layout'][old_b]) if j < i)
                if (b == old_b and orientations[i] == info['orientation'] and prefix == historic):
                    self.emit('RESTORE_REFERENCE_SUCCESS', piece=i, bin=b)
                    self.place(state, i, b, copy.deepcopy(info))
                    placed = True
                    break
                found = self.query(state, i, orientations[i], b, 'LS')
                if found is not None:
                    self.place(state, i, b, found[1])
                    placed = True
                    break
            if not placed:
                raise ValueError(f'No descent placement fits piece {i}')
            if not state['order'][-1]:
                for values in state.values():
                    values.pop()
        return self.finish(state)


def _geometry_search(nfv_pool, ifv_pool, size, shape, strategy, selection_type,
                     selection_range, accessible, flag_nfv, tolerance, rng, observer):
    from packing_iter_ls import SC_heuristic
    from SCH_iter_ls import score_position_pes

    def search(info, state, b, mode, i):
        if not state['layout'][b]:
            if shape == 'cube':
                position = (0,0,0)
            else:
                points = np.argwhere(ifv_pool.retrieve_ifv(info, size, shape) == 1)
                if not len(points):
                    return False
                x = int(points[:,0].min())
                y = int(points[points[:,0] == x,1].min())
                position = (x,y,0)
            return score_position_pes(info['array'], state['occupancy'][b],
                                      state['layout'][b], position, strategy, size, True)
        def record(e):
            if observer:
                observer(dict(e, piece_index=i, bin_index=b, orientation=info['orientation']))
        return SC_heuristic(nfv_pool, ifv_pool, info, state['layout'], state['occupancy'], b,
                            size, strategy, 5, 'z', shape,
                            selection_type if mode == 'CA' else 'bounding_box',
                            accessible, True, selection_range, flag_nfv, False,
                            rcl_tolerance=tolerance if mode == 'CA' else None,
                            position_rng=rng, rcl_observer=record if mode == 'CA' else None)
    return search


def improved_GRASP_weight_radio(object_info_total, nfv_pool, ifv_pool, max_radio, rho,
                                orientations, ils_orientations, orientations_list,
                                packing_alg, selection_type, selection_range, accessible_check,
                                SCH_nesting_strategy, orien_evaluation, container_size,
                                container_shape, iter_limit_per_iter, time_limit, alpha,
                                flag_NFV_POOL, visualisation, _TRACE,
                                random_CA_threshold=5, restart_limit=None,
                                no_improvement_limit=None, rcl_tolerance=.10,
                                piece_seed=101, position_seed=202, rcl_observer=None,
                                max_weight=None, weight_factor=1., cache_observer=None):
    """Eleven-value GRASP API with ranked construction and ranked descent.

    max_weight is an absolute capacity; by default it is rho*bin_volume*
    weight_factor, matching the existing weight-radio ILS experiment.
    Cached CA successes reuse the first RCL draw for that exact query state;
    cache hits do not redraw or emit a new RCL event. CA and greedy LS policies
    have distinct keys. All pieces are eligible for LS under ranked packing.
    Legacy alpha, orien_evaluation and random_CA_threshold are compatibility
    parameters; PES and rcl_tolerance govern construction in this function.
    """
    from packing_iter_ls import rotate_voxel, translate_voxel, visualize_voxel_model
    from new_ILS_southampton import get_final_performance, improve_pieces_selection_ls

    if packing_alg != 'SCH' or container_shape not in ('cube','cylinder'):
        raise ValueError('SCH and cube/cylinder are supported')
    if (not object_info_total or not orientations_list or iter_limit_per_iter <= 0
            or not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('Objects, orientations and positive budgets are required')
    if restart_limit is not None and restart_limit <= 0:
        raise ValueError('restart_limit must be positive')
    if rcl_tolerance is None or not np.isfinite(rcl_tolerance) or rcl_tolerance < 0:
        raise ValueError('A finite nonnegative RCL tolerance is required')
    stall = iter_limit_per_iter if no_improvement_limit is None else no_improvement_limit
    if stall <= 0:
        raise ValueError('no_improvement_limit must be positive')
    volume = (np.prod(container_size) if container_shape == 'cube' else
              math.pi*(container_size[0]/2)**2*container_size[2])
    if max_weight is None:
        max_weight = volume*rho*weight_factor
    piece_rng, position_rng = np.random.default_rng(piece_seed), np.random.default_rng(position_seed)
    restarted = [0]
    observer = (lambda e: rcl_observer(dict(e,restart=restarted[0]))) if rcl_observer else None
    geometry = _geometry_search(nfv_pool, ifv_pool, container_size, container_shape,
                                SCH_nesting_strategy, selection_type, selection_range,
                                accessible_check, flag_NFV_POOL, rcl_tolerance, position_rng, observer)
    config = (tuple(container_size), container_shape, SCH_nesting_strategy, selection_range,
              accessible_check, flag_NFV_POOL, 5, 'z', True)
    engine = GlobalRankedPacking(object_info_total, container_size, volume, max_radio, rho,
                                 max_weight, weight_factor, geometry, rotate_voxel, translate_voxel,
                                 {'CA': config+('CA',selection_type,rcl_tolerance),
                                  'LS': config+('LS','bounding_box',None)}, cache_observer)
    choices = list(dict.fromkeys(orientations_list))
    ca_choices = list(dict.fromkeys('x_0' if d == 0 else f'{ax}_{d}'
                                  for d in orientations[0] for ax in orientations[1]))
    started = time.perf_counter()
    original = best = None
    def evaluate(state):
        n,u,us = get_final_performance(object_info_total, container_size, container_shape, state['layout'])
        us = u if us is None else us
        state['metrics'], state['score'] = (n,u,us), n-us
        return state
    while original is None or (time.perf_counter()-started < time_limit
                               and (restart_limit is None or restarted[0] < restart_limit)):
        restarted[0] += 1
        current = evaluate(engine.construct(ca_choices))
        if original is None:
            original = copy.deepcopy(current)
        if best is None or current['score'] < best['score']:
            best = copy.deepcopy(current)
        attempts = failures = selections = 0
        while (attempts < iter_limit_per_iter and failures < stall
               and time.perf_counter()-started < time_limit and len(current['layout']) > 1):
            info = [None]*len(object_info_total)
            for group,ids in zip(current['layout'],current['order']):
                for p,i in zip(group,ids): info[i] = p
            selected = improve_pieces_selection_ls(info, selections, (stall-failures)/stall,
                       len(current['layout']), exclude_last_bin=False, rng=piece_rng)
            selections += 1
            values = [p['orientation'] for p in info]
            neighbors = [o for o in choices if o != values[selected]]
            if not neighbors:
                break
            for orientation in neighbors:
                if attempts >= iter_limit_per_iter or failures >= stall or time.perf_counter()-started >= time_limit:
                    break
                trial = list(values); trial[selected] = orientation
                candidate = evaluate(engine.repack(current, selected, trial))
                attempts += 1
                if candidate['score'] < best['score']:
                    best = copy.deepcopy(candidate)
                if candidate['score'] < current['score']:
                    current = candidate
                    failures = selections = 0
                    break
                failures += 1
    if visualisation:
        visualize_voxel_model(best['layout'], best['order'], container_size, container_shape)
    engine.emit('RUN_END', cache_entries=len(engine.cache), restarts=restarted[0])
    return (*best['metrics'], *original['metrics'], best['layout'], original['layout'],
            best['occupancy'], best['order'], time.perf_counter()-started)
