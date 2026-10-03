"""Compare old ILS and improved ILS at exactly 500 evaluations, including kicks.

Original search code receives minimal correctness fixes and exact-budget
guards in memory. No production source is changed. Each algorithm runs in a fresh process with untimed warm-up,
fresh pools, identical input/seed/settings and external wall-clock timing.
"""
import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LOCAL = Path(r'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='chess')
parser.add_argument('--worker', choices=['original', 'improved'])
parser.add_argument('--evaluations', type=int, default=500)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--kick-trigger', type=int, default=100)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'logs' / 'compare_ils_500_minfixed')
args = parser.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)


def worker():
    import copy
    import numpy as np
    sys.path.insert(0, str(LOCAL / 'src'))
    sys.path.insert(0, str(LOCAL / 'test'))
    os.chdir(LOCAL / 'test')
    import code_launcher as launcher
    import new_ILS_southampton as ils
    from function_lib import NFV_POOL, IFV_POOL
    from numba import cuda
    assert cuda.is_available()
    instance = launcher.make_instance_id(args.dataset, args.seed, 'cube', 999999, 1)
    shape, size, max_radio, rho, objects = launcher.load_instance(instance)
    orientations = ([0, 90, 180, 270], ['x', 'y', 'z'])
    values = ['x_0'] + [f'{ax}_{d}' for d in [90, 180, 270] for ax in ['x', 'y', 'z']]
    base = [objects, None, None, max_radio, rho, orientations, orientations,
            values, 'SCH', 'bounding_box', 'bottom', False,
            'minimum_aabb_volume', None, 'ILS', size, shape]
    print('SETTINGS', json.dumps(dict(algorithm=args.worker, seed=args.seed,
          evaluations=args.evaluations, kick_trigger=args.kick_trigger,
          kick_level='medium', dataset=args.dataset, piece_count=len(objects))), flush=True)

    # Warm construction/GPU/JIT once in each independent process; exclude it.
    np.random.seed(args.seed)
    warm = list(base)
    warm[0], warm[1], warm[2], warm[14] = copy.deepcopy(objects), NFV_POOL(), IFV_POOL(), 'fixed_CA'
    ils.improved_ILS(*warm, visualisation=False, _TRACE=False)
    print('WARMUP COMPLETE (excluded from timing)', flush=True)
    budget = dict(count=0, regular=0, kicks=0)

    def instrument(function, kind):
        def wrapped(*a, **kw):
            assert budget['count'] < args.evaluations, 'Candidate budget exceeded'
            result = function(*a, **kw)
            budget['count'] += 1
            budget[kind] += 1
            print(f"EVALUATION {budget['count']} kind={kind}", flush=True)
            return result
        return wrapped

    source_path = LOCAL / 'src' / 'new_ILS_southampton.py'
    source = source_path.read_text(encoding='utf-8')
    if args.worker == 'original':
        old = inspect.getsource(ils.ILS_from_the_first_piece)
        adapted = old.replace('while nonstop:', 'while nonstop and _test_budget["count"] < _test_limit:', 1)
        adapted = adapted.replace('        for each_orientation in other_orientations:',
            '        for each_orientation in other_orientations:\n'
            '            if _test_budget["count"] >= _test_limit:\n'
            '                nonstop = False\n'
            '                break', 1)
        adapted = adapted.replace('if no_change_time >= kick_trigger_time:',
            'if no_change_time >= kick_trigger_time and _test_budget["count"] < _test_limit:', 1)
        # Correct shared-list state without changing the original search policy.
        adapted = adapted.replace('local_best_orientations_list_no_bin = initial_orientations_list_no_bin',
                                  'local_best_orientations_list_no_bin = list(initial_orientations_list_no_bin)')
        adapted = adapted.replace('global_best_orientations_list_no_bin = local_best_orientations_list_no_bin',
                                  'global_best_orientations_list_no_bin = list(local_best_orientations_list_no_bin)')
        adapted = adapted.replace('global_best_orientations_list_no_bin = tem',
                                  'global_best_orientations_list_no_bin = list(tem)')
        # The post-kick timestamp must be an end time, not a new start time.
        import re
        adapted, timing_fixes = re.subn(
            r'(?m)^(\s*)start = time\.time\(\)(\s*\n\s*overall_time_cost \+= \(end-start\))',
            r'\1end = time.time()\2', adapted)
        assert timing_fixes == 1
        # Rebuild orientation candidates after a kick changes the local state.
        adapted = adapted.replace('                n_kick += 1',
                                  '                n_kick += 1\n                break', 1)
        ast.parse(adapted)
        (args.output / 'original_budget_adapter.py').write_text(adapted, encoding='utf-8')
        scope = dict(ils.__dict__)
        scope.update(_test_budget=budget, _test_limit=args.evaluations,
                     repacking_new_ILS=instrument(ils.repacking_new_ILS, 'regular'),
                     kick_repacking=instrument(ils.kick_repacking, 'kicks'))
        exec(compile(adapted, 'original_budget_adapter.py', 'exec'), scope)
        algorithm = scope['ILS_from_the_first_piece']
        settings = dict(iteration_limit=float('inf'), time_limit=float('inf'), alpha=1)
    else:
        ils.improved_repack = instrument(ils.improved_repack, 'regular')
        ils.kick_repacking = instrument(ils.kick_repacking, 'kicks')
        algorithm = ils.improved_ILS
        settings = dict(iteration_limit=args.evaluations, time_limit=None)

    np.random.seed(args.seed)
    base[0], base[1], base[2] = copy.deepcopy(objects), NFV_POOL(), IFV_POOL()
    start = time.perf_counter()
    result = algorithm(*base, **settings, kick_trigger_time=args.kick_trigger,
                       kick_level='medium', flag_NFV_POOL=False,
                       visualisation=False, _TRACE=False)
    elapsed = time.perf_counter() - start
    assert budget['count'] == args.evaluations, budget
    assert sorted(i for ids in result[9] for i in ids) == list(range(len(objects)))
    assert all(np.max(array) <= 1 for array in result[8])
    report = dict(algorithm=args.worker, dataset=args.dataset, seed=args.seed,
                  evaluations=budget['count'], regular=budget['regular'], kicks=budget['kicks'],
                  kick_trigger=args.kick_trigger, wall_seconds=elapsed,
                  seconds_per_evaluation=elapsed/args.evaluations,
                  best_N=result[0], best_U=result[1], best_U_star=result[2],
                  origin_N=result[3], origin_U=result[4], origin_U_star=result[5],
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  python=sys.executable,
                  original_budget_adapter=args.worker == 'original',
                  original_fixes=['exact candidate budget', 'independent orientation lists',
                                  'post-kick end timestamp', 'restart neighborhood after kick'] if args.worker == 'original' else [],
                  note='Wall time includes timed construction and identical per-candidate logging; excludes warm-up/imports. Original piece selection, acceptance, repacking and perturbation policies retained.')
    (args.output / f'{args.worker}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS', json.dumps(report), flush=True)


if args.worker:
    worker()
else:
    reports = []
    for name in ['original', 'improved']:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', name,
                   '--dataset', args.dataset, '--evaluations', str(args.evaluations), '--seed', str(args.seed),
                   '--kick-trigger', str(args.kick_trigger), '--output', str(args.output)]
        print('START', name, flush=True)
        with (args.output / f'{name}.log').open('w', encoding='utf-8') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        reports.append(json.loads((args.output / f'{name}.json').read_text(encoding='utf-8')))
        print('DONE', name, reports[-1]['wall_seconds'], flush=True)
    assert reports[0]['origin_N'] == reports[1]['origin_N']
    assert abs(reports[0]['origin_U_star'] - reports[1]['origin_U_star']) < 1e-12
    fields = ['algorithm', 'evaluations', 'regular', 'kicks', 'wall_seconds',
              'seconds_per_evaluation', 'best_N', 'best_U_star', 'origin_N', 'origin_U_star']
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(reports)
    ratio = reports[0]['wall_seconds']/reports[1]['wall_seconds']
    print(f'Original/improved wall-time ratio: {ratio:.4f}', flush=True)
    print('Results:', args.output, flush=True)
