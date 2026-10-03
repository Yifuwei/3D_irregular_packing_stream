"""Compare old ILS and improved ILS at exactly 100 evaluations by default, including kicks.

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


parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--dataset')
parser.add_argument('--task-index', type=int)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--list-datasets', action='store_true')
parser.add_argument('--manifest', type=Path)
parser.add_argument('--write-manifest', type=Path)
parser.add_argument('--progress', action='store_true')
parser.add_argument('--worker', choices=['original', 'improved'])
parser.add_argument('--evaluations', type=int, default=100)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--kick-trigger', type=int, default=100)
parser.add_argument('--output', type=Path, default=Path('ils_hpc_results'))
args = parser.parse_args()
LOCAL = args.project.resolve()
if args.evaluations < 1 or args.repeats < 1:
    parser.error('evaluations and repeats must be positive')
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)


def worker():
    import copy
    import random
    import platform
    import socket
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
    random.seed(args.seed)
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
            if args.progress:
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
        native_performance = ils.get_final_performance
        def benchmark_performance(*a, **kw):
            n, u, us = native_performance(*a, **kw)
            return n, u, u if us is None else us
        scope['get_final_performance'] = benchmark_performance
        scope.update(_test_budget=budget, _test_limit=args.evaluations,
                     repacking_new_ILS=instrument(ils.repacking_new_ILS, 'regular'),
                     kick_repacking=instrument(ils.kick_repacking, 'kicks'))
        exec(compile(adapted, 'original_budget_adapter.py', 'exec'), scope)
        algorithm = scope['ILS_from_the_first_piece']
        settings = dict(iteration_limit=float('inf'), time_limit=float('inf'), alpha=1)
    else:
        ils.improved_repack = instrument(ils.improved_repack, 'regular')
        ils.kick_repacking = instrument(ils.kick_repacking, 'kicks')
        # Benchmark-only: continue after one bin to reach the exact budget.
        # In that case every piece is eligible; production stopping is untouched.
        adapted = inspect.getsource(ils.improved_ILS)
        assert 'nonstop = ALG == "ILS" and N > 1' in adapted
        adapted = adapted.replace('nonstop = ALG == "ILS" and N > 1', 'nonstop = ALG == "ILS"')
        assert adapted.count('if N == 1 or budget_exhausted():') == 2
        adapted = adapted.replace('if N == 1 or budget_exhausted():', 'if budget_exhausted():')
        selector = ils.improve_pieces_selection_ls
        def benchmark_selector(info, iteration, stage, bins, **kw):
            return selector(info, iteration, stage, 2 if bins == 1 else bins, **kw)
        scope = dict(ils.__dict__, improve_pieces_selection_ls=benchmark_selector)
        exec(compile(adapted, 'improved_budget_adapter.py', 'exec'), scope)
        (args.output / 'improved_budget_adapter.py').write_text(adapted, encoding='utf-8')
        algorithm = scope['improved_ILS']
        settings = dict(iteration_limit=args.evaluations, time_limit=None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    base[0], base[1], base[2] = copy.deepcopy(objects), NFV_POOL(), IFV_POOL()
    cuda.synchronize()
    cpu_start = time.process_time()
    start = time.perf_counter()
    result = algorithm(*base, **settings, kick_trigger_time=args.kick_trigger,
                       kick_level='medium', flag_NFV_POOL=False,
                       visualisation=False, _TRACE=False)
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    assert budget['count'] == args.evaluations, budget
    assert sorted(i for ids in result[9] for i in ids) == list(range(len(objects)))
    assert all(np.max(array) <= 1 for array in result[8])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (LOCAL / 'src').glob('*.py')}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    report = dict(cpu_seconds=cpu_seconds, hostname=socket.gethostname(), gpu=gpu, numpy_version=np.__version__, platform=platform.platform(), source_hashes=hashes, single_bin_policy='benchmark-only continuation; all pieces eligible at one bin', algorithm=args.worker, dataset=args.dataset, seed=args.seed,
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
                  note='Wall time includes construction and search; excludes warm-up/imports. CUDA synchronized at timing boundaries. Exact budget includes kicks. Detailed event tracing disabled. One-bin continuation is a benchmark-only adaptation.')
    (args.output / f'{args.worker}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS', json.dumps(report), flush=True)


def datasets():
    suffix = f'_seq{args.seed}_cube_999999_1.txt'
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (LOCAL / 'src').glob('*.py')}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if manifest['seed'] != args.seed or manifest['source_hashes'] != hashes:
            raise ValueError('Seed/source changed after submission; use the submitted source snapshot')
        return manifest['datasets']
    names = sorted(path.name.removesuffix(suffix) for path in (LOCAL / 'instances').glob('*' + suffix))
    if args.write_manifest:
        args.write_manifest.write_text(json.dumps(dict(datasets=names, seed=args.seed, source_hashes=hashes), indent=2))
    return names


def controller():
    names = datasets()
    if args.list_datasets:
        print('\n'.join(f'{i}: {name}' for i, name in enumerate(names)))
        return
    if not names:
        parser.error('No matching cube instances in project/instances')
    if args.task_index is not None:
        if not 0 <= args.task_index < len(names):
            parser.error('task-index outside dataset list')
        names = [names[args.task_index]]
    elif args.dataset:
        if args.dataset not in names:
            parser.error('Dataset instance unavailable')
        names = [args.dataset]
    rows = []
    failed = False
    for dataset in names:
        for repeat in range(args.repeats):
            output = args.output / dataset / f'repeat_{repeat + 1}'
            output.mkdir(parents=True, exist_ok=True)
            if any((output / f'{name}.json').exists() for name in ['original', 'improved']):
                parser.error(f'Results already exist: {output}; choose a fresh --output')
            # Alternate order without changing seed or input.
            order = ['original', 'improved'] if repeat % 2 == 0 else ['improved', 'original']
            reports = {}
            for name in order:
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', name,
                    '--project', str(LOCAL), '--dataset', dataset, '--evaluations', str(args.evaluations),
                    '--seed', str(args.seed), '--kick-trigger', str(args.kick_trigger), '--output', str(output)]
                if args.progress:
                    command.append('--progress')
                print('START', dataset, repeat + 1, name, flush=True)
                with (output / f'{name}.log').open('w', encoding='utf-8') as log:
                    status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if status.returncode:
                    failed = True
                    print('FAILED; inspect', output / f'{name}.log', flush=True)
                    continue
                reports[name] = json.loads((output / f'{name}.json').read_text())
            row = dict(dataset=dataset, repeat=repeat + 1, seed=args.seed,
                       status='passed' if len(reports) == 2 else 'failed')
            for name, report in reports.items():
                row[name + '_seconds'] = report['wall_seconds']
                row[name + '_N'] = report['best_N']
                row[name + '_U_star'] = report['best_U_star']
                row[name + '_evaluations'] = report['evaluations']
                row[name + '_kicks'] = report['kicks']
            if len(reports) == 2:
                old, new = reports['original'], reports['improved']
                assert old['origin_N'] == new['origin_N']
                assert abs(old['origin_U_star'] - new['origin_U_star']) < 1e-12
                row['reduction_percent'] = 100 * (1 - new['wall_seconds'] / old['wall_seconds'])
            rows.append(row)
            fields = ['dataset', 'repeat', 'seed', 'status', 'original_seconds', 'improved_seconds',
                'reduction_percent', 'original_N', 'improved_N', 'original_U_star', 'improved_U_star',
                'original_evaluations', 'improved_evaluations', 'original_kicks', 'improved_kicks']
            # Each Slurm array task owns a distinct file.
            label = f'task_{args.task_index}' if args.task_index is not None else 'comparison'
            with (args.output / f'{label}.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    if failed:
        raise SystemExit(1)


if args.worker:
    if not args.dataset:
        parser.error('worker requires --dataset')
    worker()
else:
    controller()
