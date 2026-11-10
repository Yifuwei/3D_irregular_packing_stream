"""Shared post-placement bin priority and integration for three controllers."""
import functools
import inspect
import math
import types
import time
import hashlib
import json


def post_placement_score(current_volume, piece_volume, bin_volume,
                         current_radio, piece_radio, max_radio, max_utilization=1.):
    """Larger scores are visited first, after hard-capacity filtering."""
    if bin_volume <= 0 or max_radio <= 0 or max_utilization <= 0:
        raise ValueError('Bin volume and radio capacity must be positive')
    return ((current_volume + piece_volume) / (bin_volume * max_utilization)
            + (current_radio + piece_radio) / max_radio)


def with_radio_bin_priority(function):
    """Keep controller state transitions, replace only packing and selection.

    Packing uses a run-scoped exact-state cache. Each call gets a private
    globals dictionary; no monkey-patching is applied to the imported module.
    Set radio_priority=False for first-fit bin order with the same cache and RCL.
    cache_first_fit=False explicitly selects the legacy packing implementation.
    """
    original_signature = inspect.signature(function)

    @functools.wraps(function)
    def run(*args, radio_priority=True, max_weight=None, weight_factor=1.,
            cache_observer=None, cache_first_fit=True, packing_observer=None,
            rcl_top_k=None, **kwargs):
        if not radio_priority and not cache_first_fit:
            return function(*args, **kwargs)
        wrapper_started=time.perf_counter()
        import numpy as np
        from improved_GRASP_weight_radio import GlobalRankedPacking, _geometry_search
        from packing_iter_ls import rotate_voxel, translate_voxel
        from orientation_rcl import choose_orientation

        bound = original_signature.bind(*args, **kwargs)
        bound.apply_defaults(); p = bound.arguments
        if p['packing_alg'] != 'SCH':
            raise ValueError('Radiation-ranked packing currently requires SCH')
        size, shape = p['container_size'], p['container_shape']
        volume = (math.prod(size) if shape == 'cube' else math.pi*(size[0]/2)**2*size[2])
        if max_weight is None:
            max_weight = volume*p['rho']*weight_factor
        ctx = dict(observer=None, search_observer=None,
                   rng=np.random.default_rng(p.get('position_seed',202)), tolerance=None)
        base = (tuple(size),shape,p['SCH_nesting_strategy'],p['selection_range'],
                p['accessible_check'],p['flag_NFV_POOL'],5,'z',True)
        engine = None
        def geometry(info, state, b, mode, i):
            search = _geometry_search(p['nfv_pool'],p['ifv_pool'],size,shape,
                p['SCH_nesting_strategy'],p['selection_type'],p['selection_range'],
                p['accessible_check'],p['flag_NFV_POOL'],
                'candidates' if mode=='CA' and ctx['tolerance'] is not None else None,
                ctx['rng'],ctx['observer'])
            result = search(info,state,b,mode,i)
            if mode == 'CA' and ctx['search_observer']:
                from improved_ILS_weight_radio import geometry_search_config
                config=geometry_search_config(size,p['SCH_nesting_strategy'],shape=shape,
                    selection_type=p['selection_type'],accessible=p['accessible_check'],
                    select_range=p['selection_range'],nfv_pool=p['flag_NFV_POOL'])
                ctx['search_observer'](i,info,state['layout'],state['order'],b,result,config)
            return result
        radio = p['max_radio']
        if radio == 0 and all(o['radio']==0 for o in p['object_info_total']):
            radio = 1.  # Equivalent unconstrained remaining ratio for all-zero radiation.
        def orientation_selector(choices):
            def record(event):
                event=dict(event,level='pooled')
                engine.emit('POOLED_RCL',**event)
                if ctx['observer']:ctx['observer'](event)
            return choose_orientation(choices,ctx['tolerance'],ctx['rng'],record,top_k=rcl_top_k)
        def random_state_key():
            if ctx['tolerance'] is None:
                return 'deterministic'
            return hashlib.sha256(json.dumps(ctx['rng'].bit_generator.state,sort_keys=True).encode()).hexdigest()
        engine = GlobalRankedPacking(p['object_info_total'],size,volume,radio,p['rho'],
            max_weight,weight_factor,geometry,rotate_voxel,translate_voxel,
            {'CA':base+('CA',), 'LS':base+('LS','bounding_box',None)},cache_observer,
            bin_priority='radio' if radio_priority else 'first_fit',
            orientation_selector=orientation_selector,random_state_key=random_state_key,
            pooled_selector=orientation_selector)

        def packed(state):
            return state['layout'],state['occupancy'],state['radio'],state['order']

        def observe(kind,state):
            result=packed(state)
            if packing_observer:
                packing_observer(kind,result)
            return result

        def construct(*a, **kw):
            ctx.update(observer=kw.get('rcl_observer'),search_observer=kw.get('search_observer'),
                       rng=kw.get('position_rng',ctx['rng']),
                       tolerance=(0. if rcl_top_k is not None else kw.get('rcl_tolerance'))
                           if kw.get('random_CA') else None)
            engine.policy_keys['CA']=base+('CA',p['selection_type'],'pooled_extrema')
            engine.pooled_selector=orientation_selector if ctx['tolerance'] is not None else None
            orientations=list(dict.fromkeys('x_0' if d==0 else f'{axis}_{d}'
                              for d in a[3][0] for axis in a[3][1]))
            return observe('construction',engine.construct(orientations))

        def repack(*a, **kw):
            selected,orientation=a[1]
            state=a[5][a[6]]
            reference=dict(layout=state['bin_real_layout'],occupancy=state['bin_topos_layout'],
                           radio=state['radio_layout'],order=state['pieces_order'])
            orientations=list(a[4]);orientations[selected]=orientation
            return observe('ls',engine.repack(reference,selected,orientations))

        def kick(*a, **kw):
            # Reconstruct with the controller's perturbed orientation for each piece.
            # Greedy LS keys are used rather than random CA keys.
            return observe('kick',engine.construct([],piece_orientations=list(a[4]),mode='LS'))

        source_globals=function.__globals__
        original_selector=source_globals['improve_pieces_selection_ls']
        def select(*a, **kw):
            # Ranked construction invalidates the old first-fit last-bin exclusion.
            if radio_priority:
                kw['exclude_last_bin']=False
            return original_selector(*a, **kw)
        scope=dict(source_globals,packing_3D_voxel_lookback=construct,improved_repack=repack,
                   kick_repacking=kick,improve_pieces_selection_ls=select)
        # TRACE is controller-local in the private namespace too.
        trace=source_globals.get('trace')
        if isinstance(trace,types.FunctionType) and trace.__globals__ is source_globals:
            scope['trace']=types.FunctionType(trace.__code__,scope,trace.__name__,trace.__defaults__,trace.__closure__)
        controller=types.FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
        controller.__kwdefaults__=function.__kwdefaults__
        # Account for wrapper setup/copies within the algorithm's wall budget.
        limit=p.get('time_limit')
        if limit is not None and math.isfinite(limit):
            bound.arguments['time_limit']=max(float(np.nextafter(0.,1.)),limit-(time.perf_counter()-wrapper_started))
        result=controller(*bound.args, **bound.kwargs)
        engine.emit('RUN_END',algorithm=function.__name__,metric='used_radio_plus_used_utilization',
                    cache_entries=len(engine.cache))
        return (*result[:-1],time.perf_counter()-wrapper_started)

    extra=[inspect.Parameter(name,inspect.Parameter.KEYWORD_ONLY,default=value)
           for name,value in [('radio_priority',True),('max_weight',None),
                              ('weight_factor',1.),('cache_observer',None),
                              ('cache_first_fit',True),('packing_observer',None),('rcl_top_k',None)]]
    run.__signature__=original_signature.replace(parameters=[*original_signature.parameters.values(),*extra])
    return run
