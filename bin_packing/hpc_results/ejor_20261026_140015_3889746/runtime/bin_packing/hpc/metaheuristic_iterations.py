"""Prepare, run and collect the paired nine-configuration iteration experiment.

Preparation/collection use only the Python standard library. GPU imports occur
only in run_task. Each array task runs one unchanged improved algorithm.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import platform
import socket
import statistics
import sys
import time
import traceback
from collections import defaultdict

DEFAULT_DATASETS = ['shapesnew', 'Merged4_normal', 'st04_Example5_normal']


def configurations():
    result = []
    for limit in (100, 125, 150):
        result.append(dict(config_id=f'G{limit}', algorithm='improved_GRASP',
                           ls_limit=limit, kick_trigger=None))
        result.append(dict(config_id=f'I{limit}', algorithm='improved_ILS',
                           ls_limit=None, kick_trigger=limit))
        result.append(dict(config_id=f'GI300_K{limit}', algorithm='improved_GRASP_ILS',
                           ls_limit=300, kick_trigger=limit))
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(project):
    paths = list((project / 'src').glob('*.py'))
    paths += [project / 'test' / 'code_launcher.py', Path(__file__).resolve()]
    return {p.relative_to(project).as_posix(): sha(p) for p in paths}


def geometry_path(project, reference):
    # Mirror code_launcher.read_binvox_file: strip one leading ../ and use data/.
    reference = reference.strip().replace('\\', '/')
    if reference.startswith('../'):
        reference = reference[3:]
    return (project / 'data' / reference).resolve()


def prepare(project, datasets, shapes, seeds, time_limit=3600,
            pes='minimum_aabb_volume', selection_range='bottom_top'):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time-limit must be finite and positive')
    for name, values in [('datasets', datasets), ('shapes', shapes), ('seeds', seeds)]:
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be nonempty and distinct')
    tasks, geometries = [], {}
    for dataset in datasets:
        if Path(dataset).name != dataset or '/' in dataset or '\\' in dataset:
            raise ValueError('Dataset must be a basename')
        for shape in shapes:
            if shape not in ('cube', 'cylinder'):
                raise ValueError('Unknown shape')
            for seed in seeds:
                instance = project / 'instances' / f'{dataset}_seq{seed}_{shape}_999999_1.txt'
                lines = instance.read_text(encoding='utf-8-sig').splitlines()
                if lines[0].split('\t')[0] != shape:
                    raise ValueError(f'Shape mismatch: {instance}')
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    path = geometry_path(project, line.split('\t')[0])
                    if str(path) not in geometries:
                        geometries[str(path)] = dict(path=path.relative_to(project).as_posix(), sha256=sha(path))
                for config in configurations():
                    tasks.append(dict(task_id=len(tasks), dataset=dataset, shape=shape,
                                      seed=seed, instance=instance.name,
                                      instance_sha256=sha(instance), **config))
    return dict(schema_version=1, time_limit=time_limit, datasets=datasets,
                shapes=shapes, seeds=seeds, tasks=tasks,
                source_hashes=source_hashes(project), geometries=list(geometries.values()),
                grasp_restart_policy='consecutive_rejected_LS_evaluations; improvements reset counter',
                common=dict(pes=pes, selection_range=selection_range,
                            kick_level='medium', rcl_tolerance=.10,
                            random_CA_threshold=5, restart_limit=None,
                            no_improvement_limit=None, accessible_check=True,
                            flag_NFV_POOL=False, total_iteration_limit=None),
                internal_seed_policy='time_ns mixed with process, host and task identity; actual seeds saved per run',
                selection_basis='Default datasets are min/median/max of existing CA mean runtime; override explicitly if desired.')


def internal_seeds(task_id):
    """Fresh time-based streams, independent of the input order seed."""
    timestamp = time.time_ns()
    material = f'{timestamp}:{os.getpid()}:{socket.gethostname()}:{task_id}'.encode()
    digest = hashlib.sha256(material).digest()
    seeds = {name:int.from_bytes(digest[i*4:i*4+4], 'big')
             for i,name in enumerate(('piece','position','kick','global'))}
    return timestamp, seeds


def algorithm_kwargs(task, manifest, search_seeds):
    settings = manifest['common']
    kw = dict(time_limit=manifest['time_limit'], flag_NFV_POOL=False,
              visualisation=False, _TRACE=False,
              piece_seed=search_seeds['piece'], position_seed=search_seeds['position'])
    if task['algorithm'] == 'improved_ILS':
        kw.update(ALG='ILS', iteration_limit=None,
                  kick_trigger_time=task['kick_trigger'], kick_level=settings['kick_level'])
    else:
        kw.update(iter_limit_per_iter=task['ls_limit'], alpha=1,
                  restart_limit=None, rcl_tolerance=settings['rcl_tolerance'],
                  random_CA_threshold=settings['random_CA_threshold'])
        if task['algorithm'] == 'improved_GRASP':
            kw['no_improvement_limit'] = task['ls_limit']
            kw['ls_evaluation_limit'] = None
        else:
            kw.update(kick_trigger_time=task['kick_trigger'],
                      kick_level=settings['kick_level'], kick_seed=search_seeds['kick'])
    return kw


def validate_result(result, objects, size, shape):
    """Independent piece completeness, voxel bounds/overlap and metric check."""
    import numpy as np
    layout, topos, order = result[6], result[8], result[9]
    if len(layout) != len(topos) or len(layout) != len(order):
        raise ValueError('Inconsistent bin records')
    if sorted(i for ids in order for i in ids) != list(range(len(objects))):
        raise ValueError('Missing or duplicate piece')
    placements, volumes = [], []
    for bin_id, (pieces, ids, grid) in enumerate(zip(layout, order, topos)):
        if len(pieces) != len(ids):
            raise ValueError('Inconsistent piece records')
        occupied = np.zeros(size, dtype=bool)
        volume = 0
        for item, index in zip(pieces, ids):
            axis, angle = item['orientation'].split('_')
            source = np.rot90(objects[index]['array'], int(angle)//90,
                              axes={'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}[axis])
            points = np.argwhere(source != 0)
            points = points-points.min(0)+np.asarray(item['translation'])
            if np.any(points < 0) or np.any(points >= size):
                raise ValueError('Piece crosses bin boundary')
            if shape == 'cylinder' and np.any(
                    (points[:, 0]-size[0]/2)**2+(points[:, 1]-size[1]/2)**2 > (size[0]/2)**2):
                raise ValueError('Piece outside cylinder')
            coords = tuple(points.T)
            if occupied[coords].any():
                raise ValueError('Overlapping pieces')
            occupied[coords] = True
            array = item['array'] != 0
            if array.shape != tuple(size) or np.count_nonzero(array) != len(points) or not array[coords].all():
                raise ValueError('Layout disagrees with placement')
            if item['volume'] != objects[index]['volume']:
                raise ValueError('Piece volume changed')
            volume += objects[index]['volume']
            placements.append(dict(bin=bin_id, piece_index=index,
                                   orientation=item['orientation'],
                                   translation=[int(v) for v in item['translation']]))
        if grid.shape != tuple(size) or np.any(grid > 1) or not np.array_equal(grid != 0, occupied):
            raise ValueError('Occupancy grid disagrees with layout')
        volumes.append(volume)
    capacity = math.prod(size) if shape == 'cube' else math.pi*(size[0]/2)**2*size[2]
    rates = [v/capacity for v in volumes]
    n = len(rates)
    u = sum(p['volume'] for p in objects)/(capacity*n)
    us = (sum(rates)-min(rates))/(n-1) if n > 1 else u
    if result[0] != n or abs(result[1]-u) > 1e-9 or abs(result[2]-us) > 1e-9:
        raise ValueError('Reported N/U/U* disagree with independent calculation')
    return dict(ok=True, scope='piece identity, geometry, occupancy and N/U/U*; accessibility enforced by engine, not independently revalidated',
                bin_volumes=volumes, bin_filling_rates=rates, placements=placements)


def run_task(project, manifest, index, output):
    if not 0 <= index < len(manifest['tasks']):
        raise ValueError('Task index outside manifest')
    task = manifest['tasks'][index]
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'task_{index}.json'
    if target.exists():
        raise FileExistsError(f'Result already exists: {target}; use a fresh output directory')
    seed_time_ns, search_seeds = internal_seeds(index)
    report = dict(task, settings=algorithm_kwargs(task, manifest, search_seeds), common=manifest['common'],
                  seeds=dict(order=task['seed'], **search_seeds), seed_time_ns=seed_time_ns,
                  internal_seed_policy='time_ns+pid+hostname+task_id', status='failed')
    report.update(hostname=socket.gethostname(), platform=platform.platform(), python=sys.executable,
                  python_version=platform.python_version(), source_hashes=manifest['source_hashes'])
    overall_start = time.perf_counter()
    try:
        if manifest['source_hashes'] != source_hashes(project):
            raise ValueError('Source changed after preparation')
        instance = project / 'instances' / task['instance']
        if sha(instance) != task['instance_sha256']:
            raise ValueError('Instance changed after preparation')
        references = {line.split('\t')[0] for line in instance.read_text(encoding='utf-8-sig').splitlines()[1:] if line.strip()}
        hashes = {g['path']: g['sha256'] for g in manifest['geometries']}
        for reference in references:
            path = geometry_path(project, reference)
            if sha(path) != hashes[path.relative_to(project).as_posix()]:
                raise ValueError('Geometry changed after preparation')
        import copy
        import numpy as np
        from numba import cuda
        report['numpy_version'] = np.__version__
        if not cuda.is_available():
            raise RuntimeError('CUDA unavailable; run in a GPU allocation')
        sys.path.insert(0, str(project / 'src'))
        sys.path.insert(0, str(project / 'test'))
        os.chdir(project / 'test')
        import code_launcher as launcher
        import new_ILS_southampton as algorithms
        from function_lib import NFV_POOL, IFV_POOL
        shape, size, radio, rho, objects = launcher.load_instance(instance.stem)
        orientations = ([0, 90, 180, 270], ['x', 'y', 'z'])
        labels = ['x_0']+[f'{a}_{d}' for d in (90, 180, 270) for a in ('x', 'y', 'z')]
        common = dict(object_info_total=copy.deepcopy(objects), nfv_pool=NFV_POOL(), ifv_pool=IFV_POOL(),
                      max_radio=radio, rho=rho, orientations=orientations, ils_orientations=orientations,
                      orientations_list=labels, packing_alg='SCH', selection_type='bounding_box',
                      selection_range=manifest['common']['selection_range'], accessible_check=True,
                      SCH_nesting_strategy=manifest['common']['pes'], orien_evaluation=manifest['common']['pes'],
                      container_size=size, container_shape=shape)
        # Same untimed construction warm-up for all nine configurations.
        algorithms.improved_ILS(**common, ALG='fixed_CA', visualisation=False, _TRACE=False,
                                piece_seed=search_seeds['piece'], position_seed=search_seeds['position'])
        common.update(object_info_total=copy.deepcopy(objects), nfv_pool=NFV_POOL(), ifv_pool=IFV_POOL())
        random.seed(search_seeds['global'])
        np.random.seed(search_seeds['kick'])
        counts = dict(construction_count=0, ls_evaluation_count=0, kick_count=0)
        seconds = dict(construction_s=0., ls_s=0., kick_s=0.)
        best = [math.inf]
        best_time = [None]
        start = time.perf_counter()
        events_path = output / f'task_{index}_events.jsonl'
        with events_path.open('x', encoding='utf-8') as events:
            def instrument(function, count_key, time_key):
                def wrapped(*a, **kw):
                    begin = time.perf_counter()
                    packed = function(*a, **kw)
                    seconds[time_key] += time.perf_counter()-begin
                    counts[count_key] += 1
                    n, u, us = algorithms.get_final_performance(objects, size, shape, packed[0])
                    us = u if us is None else us
                    score = n-us
                    # Observation of completed candidates; no acceptance-policy changes.
                    if score < best[0]:
                        best[0] = score
                        best_time[0] = time.perf_counter()-start
                        events.write(json.dumps(dict(time_s=best_time[0], event=count_key,
                                                     N=n, U_star=us, objective=score, **counts))+'\n')
                        events.flush()
                    return packed
                return wrapped
            for name, counter, timing in [('packing_3D_voxel_lookback', 'construction_count', 'construction_s'),
                                          ('improved_repack', 'ls_evaluation_count', 'ls_s'),
                                          ('kick_repacking', 'kick_count', 'kick_s')]:
                setattr(algorithms, name, instrument(getattr(algorithms, name), counter, timing))
            cuda.synchronize()
            start = time.perf_counter()
            result = getattr(algorithms, task['algorithm'])(**common, **report['settings'])
            cuda.synchronize()
            elapsed = time.perf_counter()-start
        validation = validate_result(result, objects, size, shape)
        if abs((result[0]-result[2])-best[0]) > 1e-9:
            raise ValueError('Returned best differs from observed candidate best')
        report.update(status='success', validation_ok=True, validation=validation,
                      best_N=result[0], best_U=result[1], best_U_star=result[2],
                      best_N_minus_U_star=result[0]-result[2], origin_N=result[3],
                      origin_U=result[4], origin_U_star=result[5], algorithm_wall_s=result[-1],
                      measured_search_s=elapsed, setup_s=start-overall_start,
                      piece_count=len(objects), best_found_at_s=best_time[0], **counts, **seconds)
        report['total_iteration_count'] = counts['ls_evaluation_count']+counts['kick_count']
        report['source_hashes'] = manifest['source_hashes']
        report['stop_reason'] = ('time_limit' if result[-1] >= manifest['time_limit'] else
                                 'algorithm_early_stop')
    except Exception:
        report.update(error=traceback.format_exc(), validation_ok=False)
    report['elapsed_s'] = time.perf_counter()-overall_start
    target.write_text(json.dumps(report, indent=2, default=lambda x: x.item()), encoding='utf-8')
    print(json.dumps({k:report.get(k) for k in ('task_id','config_id','status','best_N','best_U_star','error')}), flush=True)
    return 0 if report['status']=='success' else 1


def collect(manifest, output):
    rows = []
    for task in manifest['tasks']:
        path = output / f"task_{task['task_id']}.json"
        row = dict(task, status='missing')
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                if any(report.get(k) != task[k] for k in task):
                    raise ValueError('Report identity differs from manifest')
                row.update({k:v for k,v in report.items() if not isinstance(v,(dict,list))})
            except (ValueError, KeyError) as error:
                row.update(status='unreadable', error=str(error))
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['shape'], row['config_id'])].append(row)
    summaries = []
    for (dataset, shape, config), members in sorted(grouped.items()):
        good = [r for r in members if r['status']=='success' and r.get('validation_ok') is True]
        summary = dict(dataset=dataset, shape=shape, config_id=config,
                       expected_count=len(members), success_count=len(good), complete=len(good)==len(members))
        for metric in ('best_N','best_U_star','best_N_minus_U_star','algorithm_wall_s',
                       'ls_evaluation_count','kick_count','construction_count','best_found_at_s'):
            vals = [r[metric] for r in good if isinstance(r.get(metric),(int,float)) and math.isfinite(r[metric])]
            summary[metric+'_mean'] = statistics.mean(vals) if vals else None
            summary[metric+'_std'] = statistics.stdev(vals) if len(vals)>1 else None
        summaries.append(summary)
    def write(path, records):
        fields = list(dict.fromkeys(k for r in records for k in r))
        with path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    write(output/'iteration_runs.csv', rows)
    write(output/'iteration_summary.csv', summaries)
    # Rank only instances with all nine configurations and all paired seeds.
    byinstance = defaultdict(list)
    for summary in summaries:
        byinstance[(summary['dataset'], summary['shape'])].append(summary)
    ranks = defaultdict(list)
    eligible = 0
    for (dataset, shape), members in byinstance.items():
        if len(members)!=9 or not all(m['complete'] for m in members):
            continue
        eligible += 1
        values = sorted(members,key=lambda m:m['best_N_minus_U_star_mean'])
        i=0
        while i<len(values):
            j=i+1
            while j<len(values) and abs(values[j]['best_N_minus_U_star_mean']-values[i]['best_N_minus_U_star_mean'])<=1e-9:
                j+=1
            rank=(i+1+j)/2
            for member in values[i:j]:
                ranks[(shape,member['config_id'])].append(rank)
                ranks[('Overall',member['config_id'])].append(rank)
            i=j
    ranking = [dict(shape=shape,config_id=config,eligible_instance_count=len(values),
                    mean_rank=statistics.mean(values)) for (shape,config),values in sorted(ranks.items())]
    if not ranking:
        ranking=[dict(shape='Overall',config_id='unavailable',eligible_instance_count=0,mean_rank=None)]
    write(output/'iteration_ranking.csv',ranking)
    print(f"Collected {len(rows)} tasks; {sum(r['status']=='success' for r in rows)} successful")
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','run','collect'])
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--shapes', nargs='+', choices=['cube','cylinder'], default=['cube','cylinder'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[3,7,13])
    parser.add_argument('--time-limit', type=float, default=3600)
    parser.add_argument('--pes', default='minimum_aabb_volume', choices=['minimum_aabb_volume','minimum_aabb_edges_len','maximal_residual_box','overlap_distance'])
    parser.add_argument('--range', default='bottom_top', choices=['bottom','bottom_top','all'])
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    if args.mode == 'prepare':
        if args.manifest.exists():
            parser.error('Manifest already exists; use a fresh path')
        manifest = prepare(project, args.datasets, args.shapes, args.seeds, args.time_limit, args.pes, args.range)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(len(manifest['tasks']))
        return 0
    if args.output is None or (args.mode=='run' and args.task_index is None):
        parser.error('run/collect requires --output; run also requires --task-index')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.mode=='run':
        return run_task(project,manifest,args.task_index,args.output.resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    collect(manifest,args.output)
    return 0


if __name__=='__main__':
    sys.exit(main())
