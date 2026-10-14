"""Paired chess capacity grid. Keep the production improved_ILS controller.

Only its repacker is replaced in a private execution namespace. Each worker
has a fresh process, CUDA warmup, pools, seed and cache. Never edits production.
"""
import argparse
import collections
import copy
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(a):
    import numpy as np
    project = a.project.resolve()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(project/'src'), str(project/'test')]
    os.chdir(project/'test')
    import code_launcher as launcher
    import packing_iter_ls as packing
    import new_ILS_southampton as ils
    from function_lib import NFV_POOL, IFV_POOL
    from ca_parameters import best_ca_parameters
    from improved_ILS_weight_radio import Solution, RankedRepacker, geometry_search_config
    from numba import cuda
    assert cuda.is_available()
    shape, size, _, _, objects = launcher.load_instance(
        launcher.make_instance_id('chess', a.seed, 'cube', 999999, 1))
    volume = int(np.prod(size))
    strategy, selection_range = best_ca_parameters(shape)
    orientations = ([0,90,180,270], ['x','y','z'])
    values = ['x_0']+[f'{ax}_{d}' for d in [90,180,270] for ax in ['x','y','z']]

    def run(function, alg, nfv, ifv):
        return function(copy.deepcopy(objects), nfv, ifv, a.radio, a.rho,
            orientations, orientations, values, 'SCH', 'bounding_box',
            selection_range, False, strategy, None, alg, size, shape,
            iteration_limit=a.iterations, time_limit=None,
            kick_trigger_time=a.iterations+1, kick_level='medium',
            visualisation=False, _TRACE=False,
            ca_search_observer=(engine.record_ca_search
                if alg=='ILS' and a.method=='improved_ILS_weight_radio' else None))

    # Exclude initial CUDA compilation from formal timing.
    np.random.seed(a.seed)
    run(ils.improved_ILS, 'fixed_CA', NFV_POOL(), IFV_POOL())
    cuda.synchronize()
    nfv, ifv = NFV_POOL(), IFV_POOL()
    events = collections.Counter(); candidates = []; repack_seconds = 0.0
    trace = (out/'events.jsonl').open('w', encoding='utf-8', buffering=1)
    def emit(**e):
        events[e['event']] += 1
        trace.write(json.dumps(dict(evaluation=len(candidates), **e))+'\n')

    def geometry(info, layout, occupancy, b):
        if not layout[b]:
            occupied = np.argwhere(info['array'] == 1)
            if not len(occupied) or np.any(occupied.max(axis=0) >= size):
                return False
            return 0, (0,0,0)
        return packing.SC_heuristic(nfv, ifv, info, layout, occupancy, b,
            size, strategy, 5, 'z', shape, 'bounding_box', False, True,
            selection_range, False, False)

    engine = RankedRepacker(objects, size, volume, a.radio, a.rho*volume,
        geometry, packing.rotate_voxel, packing.translate_voxel, rho=a.rho,
        search_config=geometry_search_config(size,strategy,select_range=selection_range,shape=shape), event=emit)
    original = ils.improved_repack
    if a.method == 'improved_ILS':
        # Count the actual native skip/restore branches without changing decisions.
        source = inspect.getsource(original)
        replacements = {
            '                    continue\n': '                    emit(event="SKIP_UNCHANGED_EARLIER")\n                    continue\n',
            '                    place(num_piece, each_bin, best_current_info)':
                '                    emit(event="RESTORE_UNCHANGED_ORIGINAL")\n                    place(num_piece, each_bin, best_current_info)',
            '                result = search(': '                emit(event="SEARCH")\n                result = search(',
        }
        for old, new in replacements.items():
            assert old in source, old
            source = source.replace(old,new)
        scope = dict(packing.__dict__, emit=emit)
        exec(compile(source, 'grid_native_repack.py', 'exec'), scope)
        native = scope['improved_repack']

    def repack(*args, **kwargs):
        nonlocal repack_seconds
        candidates.append(list(args[1]))
        start = time.perf_counter()
        if a.method == 'improved_ILS':
            result = native(*args, **kwargs)
        else:
            state = args[5][args[6]]
            reference = Solution(state['bin_real_layout'], state['bin_topos_layout'],
                state['radio_layout'], state['pieces_order'])
            candidate_orientations = list(args[4])
            candidate_orientations[args[1][0]] = args[1][1]
            packed = engine.repack(reference, args[1][0], candidate_orientations)
            # Production routines expect bin_position in every placement record.
            for b, group in enumerate(packed.layout):
                for info in group: info['bin_position'] = b
            result = packed.layout, packed.occupancy, packed.radio, packed.order
        repack_seconds += time.perf_counter()-start
        if len(candidates)%10 == 0:
            print(f'{a.method}: {len(candidates)}/{a.iterations}', flush=True)
        return result

    scope = dict(ils.__dict__, improved_repack=repack)
    exec(compile(inspect.getsource(ils.improved_ILS),'grid_controller.py','exec'),scope)
    np.random.seed(a.seed)
    started = time.perf_counter()
    result = run(scope['improved_ILS'], 'ILS', nfv, ifv)
    cuda.synchronize(); elapsed = time.perf_counter()-started
    trace.close()
    assert len(candidates) == a.iterations, len(candidates)
    best = Solution(result[6], result[8],
        [[p['radio'] for p in g] for g in result[6]], result[9])
    engine.validate(best)
    initial = result[7]
    record = dict(method=a.method, dataset='chess', seed=a.seed, pieces=len(objects),
        iterations=len(candidates), kicks=0, max_radio=a.radio, rho=a.rho,
        initial_bins=int(result[3]), initial_U_star=float(result[5]),
        bins=int(result[0]), U_star=float(result[2]), seconds=elapsed,
        repack_seconds=repack_seconds, non_repack_seconds=elapsed-repack_seconds,
        initial_counts=[len(g) for g in initial], counts=[len(g) for g in best.layout],
        filling_rates=[sum(p['volume'] for p in g)/volume for g in best.layout],
        radiation=[sum(p['radio'] for p in g) for g in best.layout],
        initial_signature=[[(p['orientation'], list(map(int,p['translation']))) for p in g] for g in initial],
        events=dict(events), candidates=candidates, validated=True,
        ca_failure_cache_enabled=a.method=='improved_ILS_weight_radio',
        source_hashes={name:hashlib.sha256((project/'src'/name).read_bytes()).hexdigest()
            for name in ['packing_iter_ls.py','new_ILS_southampton.py','improved_ILS_weight_radio.py']})
    (out/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps({k:record[k] for k in ['method','bins','U_star','seconds']}),flush=True)


def collect(out):
    records = [json.loads(p.read_text()) for p in sorted(out.glob('*/result.json'))]
    rows = []
    for radio in [2000,3000,4000]:
        for rho in [.2,.3,.4]:
            pair = {r['method']:r for r in records if r['max_radio']==radio and r['rho']==rho}
            if len(pair)!=2: continue
            old,new = pair['improved_ILS'],pair['improved_ILS_weight_radio']
            assert old['initial_signature']==new['initial_signature']
            assert old['source_hashes']==new['source_hashes']
            rows.append(dict(max_radio=radio,rho=rho,initial_bins=old['initial_bins'],
                improved_ILS_bins=old['bins'],weight_radio_bins=new['bins'],
                improved_ILS_U_star=old['U_star'],weight_radio_U_star=new['U_star'],
                improved_ILS_seconds=old['seconds'],weight_radio_seconds=new['seconds'],
                time_reduction_percent=100*(old['seconds']-new['seconds'])/old['seconds'],
                native_skips=old['events'].get('SKIP_UNCHANGED_EARLIER',0),
                ranked_cached_failure_skips=new['events'].get('SKIP_CACHED_FAILURE',0),
                same_candidate_tape=old['candidates']==new['candidates']))
    if rows:
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Chess: improved_ILS versus improved_ILS_weight_radio', '',
        f'Completed: {len(records)}/18 runs; {len(rows)}/9 paired settings.', '',
        'Seed 13; 100 evaluated candidates per method. Same production improved_ILS selection, acceptance and stopping controller; only repacking differs. Kick trigger 101 (zero kicks within this budget). Each method uses the same fixed CA initial layout, hard radiation and volume limits. Weight equals volume, so its limit is rho × bin volume and is redundant with the filling constraint. Higher U* is better. Positive time reduction means weight/radio is faster.', '',
        'Fresh sequential processes, untimed CUDA/CA warmup, fresh formal pools, alternating method order. Wall time includes formal CA and repacking, excludes instance loading and warmup. Same seed/policy need not produce identical candidates after accepted layouts diverge. Single-seed exploratory comparison; timings include event logging and do not estimate seed variance.', '',
        '| Radiation | Max filling | Initial bins | improved_ILS bins | Weight/radio bins | improved_ILS U* | Weight/radio U* | improved_ILS s | Weight/radio s | Time reduction |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['max_radio']} | {r['rho']:.1f} | {r['initial_bins']} | {r['improved_ILS_bins']} | {r['weight_radio_bins']} | {r['improved_ILS_U_star']:.4f} | {r['weight_radio_U_star']:.4f} | {r['improved_ILS_seconds']:.2f} | {r['weight_radio_seconds']:.2f} | {r['time_reduction_percent']:.2f}% |")
    lines.extend(['', 'Every completed best solution passed: all 32 pieces exactly once, no overlapping voxels, radiation ≤ specified maximum and filling ≤ rho. Paired initial layouts and production source hashes match. See each run’s result.json and events.jsonl for bin counts, radiation, filling, candidate tape and executed skip/search branches.'])
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--seed',type=int,default=13)
    parser.add_argument('--method',choices=['improved_ILS','improved_ILS_weight_radio'])
    parser.add_argument('--radio',type=float)
    parser.add_argument('--rho',type=float)
    parser.add_argument('--collect',action='store_true')
    a=parser.parse_args()
    if a.method: worker(a); return
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if a.collect: collect(a.output);return
    settings=[(r,f) for r in [2000,3000,4000] for f in [.2,.3,.4]]
    for index,(radio,rho) in enumerate(settings):
        methods=['improved_ILS','improved_ILS_weight_radio']
        if index%2: methods.reverse()
        for method in methods:
            out=a.output/f'r{radio}_rho{rho}_{method}'
            if (out/'result.json').exists(): continue
            out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(Path(__file__).resolve()),'--project',str(a.project.resolve()),
                '--output',str(out),'--iterations',str(a.iterations),'--seed',str(a.seed),
                '--method',method,'--radio',str(radio),'--rho',str(rho)]
            with (out/'run.log').open('w',encoding='utf-8') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                collect(a.output)
                raise RuntimeError(f'Worker failed: {out}/run.log')
            collect(a.output)
            print(f'Finished radiation={radio}, rho={rho}, {method}',flush=True)
    collect(a.output)


if __name__=='__main__': main()
