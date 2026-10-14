"""Matched GPU comparison through code_launcher.testing, in isolated workers."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    from numba import cuda
    import code_launcher as launcher
    import new_ILS_southampton as algorithms
    import packing_iter_ls as packing
    from function_lib import get_nfv_gpu
    if not cuda.is_available():
        raise RuntimeError('CUDA is required for this comparison')
    # Warm up the CUDA kernel equally in both isolated workers, before timing.
    small = np.zeros((4, 4, 4), dtype=np.uint8)
    small[0, 0, 0] = 1
    get_nfv_gpu(small, small, small.shape, (4, 4, 4))
    counts = dict(constructions=0, repacks=0)
    captured = {}
    for name, kind in [('packing_3D_voxel_lookback', 'constructions'),
                       ('repacking_new_ILS', 'repacks'), ('improved_repack', 'repacks')]:
        original = getattr(algorithms, name)
        def counted(*a, _original=original, _kind=kind, **kw):
            counts[_kind] += 1
            return _original(*a, **kw)
        setattr(algorithms, name, counted)
    original_algorithm = getattr(launcher, args.worker)
    def measured(*a, **kw):
        begin = time.perf_counter()
        result = original_algorithm(*a, **kw)
        wall = time.perf_counter() - begin
        layout, occupancy, order = result[6], result[8], result[9]
        errors = []
        indices = [i for ids in order for i in ids]
        if sorted(indices) != list(range(len(a[0]))):
            errors.append('Missing or duplicate piece IDs')
        if not (len(layout) == len(occupancy) == len(order)):
            errors.append('Inconsistent bin records')
        size = a[14]  # Public GRASP signature: container_size.
        for b, (objects, ids) in enumerate(zip(layout, order)):
            rebuilt = np.zeros(size, dtype=np.uint16)
            for info, index in zip(objects, ids):
                axis, degree = info['orientation'].split('_')
                rotated = packing.rotate_voxel(a[0][index]['array'], int(degree), axis)
                coordinates = np.argwhere(rotated == 1) + np.asarray(info['translation'])
                if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(size)):
                    errors.append(f'Piece {index} out of bounds')
                    continue
                expected = np.zeros(size, dtype=np.uint8)
                expected[tuple(coordinates.T)] = 1
                if not np.array_equal(expected, info['array']):
                    errors.append(f'Piece {index} placement mismatch')
                rebuilt += expected
            if np.any(rebuilt > 1):
                errors.append(f'Bin {b} has overlapping voxels')
            if not np.array_equal(rebuilt, occupancy[b]):
                errors.append(f'Bin {b} occupancy mismatch')
        captured.update(best_N=result[0], best_U=result[1], best_U_star=result[2],
                        origin_N=result[3], origin_U=result[4], origin_U_star=result[5],
                        objective=result[0]-result[2], wall_seconds=wall,
                        algorithm_seconds=result[-1], validation_ok=not errors,
                        validation_errors=errors)
        return result
    setattr(launcher, args.worker, measured)
    rows = launcher.testing([args.dataset], [args.seed], [args.worker], ['cube'],
                            experiment_iterations=args.iterations,
                            experiment_time=args.time_limit,
                            selection_ranges=['bottom_top'],
                            nesting_strategies=[args.pes], rcl_tolerance=.10)
    captured.update(dataset=args.dataset, seed=args.seed, algorithm=args.worker,
                    time_limit=args.time_limit, ls_limit=args.iterations, piece_seed=101, position_seed=202,
                    pes=args.pes, selection_range='bottom_top', counts=counts,
                    launcher_row=rows[0], cuda_device=str(cuda.get_current_device()))
    Path(args.output).write_text(json.dumps(captured, indent=2), encoding='utf-8')
    if not captured['validation_ok']:
        raise RuntimeError(captured['validation_errors'])


def parent(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in args.seeds:
        for algorithm in ('GRASP', 'improved_GRASP'):
            name = f'{args.dataset}_seed{seed}_{algorithm}'
            result = out / (name + '.json')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', algorithm,
                   '--dataset', args.dataset, '--seed', str(seed), '--time-limit', str(args.time_limit),
                   '--iterations', str(args.iterations), '--pes', args.pes, '--output', str(result)]
            print(f'Running {name}', flush=True)
            with (out / (name + '.log')).open('w', encoding='utf-8') as log:
                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            if process.returncode or not result.exists():
                print((out / (name + '.log')).read_text(encoding='utf-8')[-4000:], flush=True)
                raise RuntimeError(f'{name} failed; see its log')
            data = json.loads(result.read_text(encoding='utf-8'))
            results.append(data)
            print(f"{name}: N={data['best_N']}, U*={data['best_U_star']:.6f}, objective={data['objective']:.6f}, wall={data['wall_seconds']:.2f}s", flush=True)
    columns = ['dataset', 'seed', 'algorithm', 'best_N', 'best_U_star', 'objective',
               'origin_N', 'origin_U_star', 'wall_seconds', 'algorithm_seconds', 'validation_ok']
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    report = ['# GRASP comparison', '',
              f'Dataset: {args.dataset}; seeds: {args.seeds}; internal time budget: {args.time_limit}s; LS evaluations per restart: {args.iterations}.',
              f'Piece seed is fixed at 101; position seed is fixed at 202. Both use cube instances, radio limit 999999, rho 1, PES `{args.pes}`, and `bottom_top`. Improved position RCL tolerance: 10%.',
              'Runs use separate processes, the same initial RNG seed per pair, and equal CUDA-kernel warmup. RNG streams diverge as algorithms consume different draws.',
              'Original GRASP counts packing-call time; improved GRASP counts wall time including copies and evaluation. Budgets are internal and non-preemptive, so completed calls may overrun. Wall times are recorded separately.',
              '', '| Seed | Algorithm | N | U* | N-U* | Wall seconds | CA calls | LS calls | Valid |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for d in results:
        report.append(f"| {d['seed']} | {d['algorithm']} | {d['best_N']} | {d['best_U_star']:.6f} | {d['objective']:.6f} | {d['wall_seconds']:.2f} | {d['counts']['constructions']} | {d['counts']['repacks']} | {d['validation_ok']} |")
    wins = ties = losses = 0
    for seed in args.seeds:
        old, new = [d for d in results if d['seed'] == seed]
        delta = new['objective'] - old['objective']
        wins += delta < -1e-10
        losses += delta > 1e-10
        ties += abs(delta) <= 1e-10
    report += ['', f'Improved GRASP: {wins} wins, {ties} ties, {losses} losses by N-U* (lower is better).',
               'This is a short single-dataset comparison, not evidence of superiority across datasets. Several changes are compared together; their individual effects are not isolated.',
               'Validation reconstructs every placed piece without modulo wrapping and checks piece counts, cube bounds, overlaps, and occupancy consistency. It does not validate full accessibility trajectories.']
    (out / 'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('Report:', out / 'report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['GRASP', 'improved_GRASP'])
    parser.add_argument('--dataset', default='chess')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--seeds', nargs='+', type=int, default=[3, 7, 13])
    parser.add_argument('--time-limit', type=float, default=30)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--pes', default='maximal_residual_box')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    worker(args) if args.worker else parent(args)
