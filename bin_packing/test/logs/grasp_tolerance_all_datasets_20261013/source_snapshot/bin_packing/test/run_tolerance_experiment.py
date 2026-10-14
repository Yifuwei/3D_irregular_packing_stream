"""Paired two-shape tolerance experiment through code_launcher.testing."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

TEST=Path(__file__).resolve().parent
ROOT=TEST.parents[1]
PES={'cube':'maximal_residual_box','cylinder':'minimum_aabb_edges_len'}
SOURCE=['bin_packing/src/new_ILS_southampton.py','bin_packing/src/packing_iter_ls.py',
        'bin_packing/src/SCH_iter_ls.py','bin_packing/src/function_lib.py',
        'bin_packing/src/ca_parameters.py','bin_packing/test/code_launcher.py',
        'bin_packing/test/run_tolerance_experiment.py']


def save_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=lambda v:v.item()),encoding='utf-8')
    temp.replace(path)


def write_csv(path,rows,fields=None):
    if fields is None:fields=list(rows[0]) if rows else []
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def validate(result,original,size,shape):
    import numpy as np
    import packing_iter_ls as packing
    errors=[];layout,occupancy,order=result[6],result[8],result[9]
    indices=[i for ids in order for i in ids]
    if sorted(indices)!=list(range(len(original))):errors.append('Missing/duplicate piece IDs')
    if not len(layout)==len(occupancy)==len(order):errors.append('Inconsistent bin counts')
    for b,(objects,ids) in enumerate(zip(layout,order)):
        if len(objects)!=len(ids):errors.append(f'Bin {b}: inconsistent piece records')
        rebuilt=np.zeros(size,dtype=np.uint16)
        for info,index in zip(objects,ids):
            axis,degree=info['orientation'].split('_')
            rotated=packing.rotate_voxel(original[index]['array'],int(degree),axis)
            points=np.argwhere(rotated==1)+np.asarray(info['translation'])
            if np.any(points<0) or np.any(points>=np.asarray(size)):
                errors.append(f'Piece {index}: out of array bounds');continue
            if shape=='cylinder':
                # Match the production voxel-cylinder center and boundary convention.
                radial=(points[:,0]-size[0]/2)**2+(points[:,1]-size[1]/2)**2
                if np.any(radial>(size[0]/2)**2):errors.append(f'Piece {index}: outside cylinder')
            expected=np.zeros(size,dtype=np.uint8);expected[tuple(points.T)]=1
            if not np.array_equal(expected,info['array']):errors.append(f'Piece {index}: placement mismatch')
            rebuilt+=expected
        if np.any(rebuilt>1):errors.append(f'Bin {b}: overlap')
        if not np.array_equal(rebuilt,occupancy[b]):errors.append(f'Bin {b}: occupancy mismatch')
    return errors


def worker(args):
    import numpy as np
    from numba import cuda
    import code_launcher as launcher
    import new_ILS_southampton as algorithms
    from function_lib import get_nfv_gpu,njit_get_ifv_cylinder,get_bounding_box_njit_translation
    case=dict(dataset=args.dataset,shape=args.shape,order_seed=args.seed,tolerance=args.tolerance,
              piece_seed=101,position_seed=202,pes=PES[args.shape],selection_range='bottom_top',
              time_limit=args.time_limit,ls_limit=args.iterations)
    try:
        if not cuda.is_available():raise RuntimeError('CUDA unavailable')
        instance=launcher.make_instance_id(args.dataset,args.seed,args.shape,999999,1)
        _,size,_,_,objects=launcher.load_instance(instance)
        # Equal warmup for every independent worker, before the algorithm budget.
        small=np.zeros((4,4,4),dtype=objects[0]['array'].dtype);small[0,0,0]=1
        fixed=np.zeros((4,4,4));fixed[0,0,0]=1
        get_nfv_gpu(fixed,small,small.shape,(4,4,4))
        get_bounding_box_njit_translation(fixed)
        njit_get_ifv_cylinder(small,1,1,1,small.shape)
        counts=Counter();times=Counter();events=[];history=[];capture={};clock=[None]
        def observer(e):
            if e['best_pes'] is not None:
                events.append(dict(restart=e['restart'],piece_index=e['piece_index'],
                    bin_index=e['bin_index'],orientation=e['orientation'],
                    candidate_count=len(e['candidates']),rcl_size=e['active_rcl_size'],
                    best_pes=e['best_pes'],selected_pes=e['selected_pes']))
        for name,kind in [('packing_3D_voxel_lookback','construction'),('improved_repack','ls')]:
            function=getattr(algorithms,name)
            def counted(*a,_function=function,_kind=kind,**kw):
                start=time.perf_counter();result=_function(*a,**kw)
                counts[_kind]+=1;times[_kind]+=time.perf_counter()-start
                N,U,Ustar=algorithms.get_final_performance(objects,size,args.shape,result[0])
                score=N-(U if Ustar is None else Ustar)
                history.append(dict(seconds=time.perf_counter()-clock[0],kind=_kind,F=score))
                return result
            setattr(algorithms,name,counted)
        original=launcher.improved_GRASP
        def measured(*a,**kw):
            clock[0]=time.perf_counter();result=original(*a,**kw)
            elapsed=time.perf_counter()-clock[0]
            errors=validate(result,objects,size,args.shape)
            capture.update(best_N=result[0],best_U_star=result[2],F=result[0]-result[2],
                origin_N=result[3],origin_U_star=result[5],origin_F=result[3]-result[5],
                wall_seconds=elapsed,algorithm_seconds=result[-1],validation_errors=errors,
                constructions=counts['construction'],ls_calls=counts['ls'],
                construction_seconds=times['construction'],ls_seconds=times['ls'],
                cuda_device=str(cuda.get_current_device()),dimensions=list(size))
            return result
        launcher.improved_GRASP=measured
        rows=launcher.testing([args.dataset],[args.seed],['improved_GRASP'],[args.shape],
            experiment_iterations=args.iterations,experiment_time=args.time_limit,
            selection_ranges=['bottom_top'],nesting_strategies=[PES[args.shape]],
            rcl_tolerance=args.tolerance,piece_seed=101,position_seed=202,rcl_observer=observer)
        case.update(capture,status='success' if not capture['validation_errors'] else 'invalid',
                    launcher_row=rows[0],rcl_histogram=dict(Counter(e['rcl_size'] for e in events)),
                    rcl_calls=len(events),rcl_mean=sum(e['rcl_size'] for e in events)/len(events) if events else None,
                    history=history)
        for checkpoint in (60,120,180):
            known=[h['F'] for h in history if h['seconds']<=checkpoint]
            case[f'F_at_{checkpoint}s']=min(known) if known else None
        write_csv(Path(args.output).with_suffix('.rcl.csv'),events,
                  ['restart','piece_index','bin_index','orientation','candidate_count','rcl_size','best_pes','selected_pes'])
    except Exception:
        case.update(status='failed',traceback=traceback.format_exc())
        print(case['traceback'],flush=True)
    save_json(args.output,case)
    print(json.dumps({k:v for k,v in case.items() if k in ('dataset','shape','order_seed','tolerance','status','F','best_N','constructions','ls_calls')},default=lambda v:v.item()),flush=True)


def summarize(out,plan,results,running=None):
    import statistics
    successes=[r for r in results if r['status']=='success']
    fields=['dataset','shape','order_seed','tolerance','status','best_N','best_U_star','F',
            'origin_N','origin_U_star','origin_F','wall_seconds','constructions','ls_calls',
            'rcl_calls','rcl_mean','F_at_60s','F_at_120s','F_at_180s']
    write_csv(out/'results.csv',results,fields)
    pairs=[]
    for dataset in dict.fromkeys(c['dataset'] for c in plan):
        for shape in dict.fromkeys(c['shape'] for c in plan):
            for seed in dict.fromkeys(c['order_seed'] for c in plan):
                matched={r['tolerance']:r for r in successes if (r['dataset'],r['shape'],r['order_seed'])==(dataset,shape,seed)}
                if len(matched)!=3:continue
                base=matched[.1]['F']
                pairs.append(dict(dataset=dataset,shape=shape,order_seed=seed,F10=base,
                    F15=matched[.15]['F'],F20=matched[.2]['F'],
                    delta15=matched[.15]['F']-base,delta20=matched[.2]['F']-base))
    write_csv(out/'paired_comparison.csv',pairs,['dataset','shape','order_seed','F10','F15','F20','delta15','delta20'])
    aggregate=[]
    for dataset in dict.fromkeys(c['dataset'] for c in plan):
        for shape in dict.fromkeys(c['shape'] for c in plan):
            for rate in (.1,.15,.2):
                group=[r for r in successes if (r['dataset'],r['shape'],r['tolerance'])==(dataset,shape,rate)]
                if not group:continue
                values=[r['F'] for r in group]
                aggregate.append(dict(dataset=dataset,shape=shape,tolerance=rate,successes=len(group),
                    mean_F=statistics.mean(values),median_F=statistics.median(values),
                    std_F=statistics.stdev(values) if len(values)>1 else None,
                    mean_N=statistics.mean(r['best_N'] for r in group),
                    mean_constructions=statistics.mean(r['constructions'] for r in group),
                    mean_ls_calls=statistics.mean(r['ls_calls'] for r in group)))
    write_csv(out/'summary_by_dataset.csv',aggregate,
              ['dataset','shape','tolerance','successes','mean_F','median_F','std_F','mean_N','mean_constructions','mean_ls_calls'])
    frequency=[]
    for shape in ('cube','cylinder'):
        for rate in (.1,.15,.2):
            histogram=Counter()
            for r in successes:
                if (r['shape'],r['tolerance'])==(shape,rate):
                    histogram.update({int(k):v for k,v in r['rcl_histogram'].items()})
            total=sum(histogram.values())
            for size,count in sorted(histogram.items()):
                frequency.append(dict(shape=shape,tolerance=rate,rcl_size=size,count=count,
                                      percent=100*count/total,total_calls=total))
    write_csv(out/'rcl_frequency.csv',frequency,['shape','tolerance','rcl_size','count','percent','total_calls'])
    report=['# GRASP tolerance comparison: cube and cylinder','',
        f'Completed: {len(results)}/{len(plan)}; valid successes: {len(successes)}; failed/invalid: {len(results)-len(successes)}.',
        f'Currently running: {running or "none"}.',
        'Only improved_GRASP is compared, at position tolerances 10%, 15%, and 20%. Order seeds 3/7/13; piece seed 101; position seed 202. Each run has a 180-second wall budget and 20 LS evaluations per restart. All runs use bottom/top candidates and original instance dimensions.',
        'Cube PES: maximal_residual_box. Cylinder PES: minimum_aabb_edges_len. Shape-specific results are kept separate. Radio limit 999999; rho 1. Workers run sequentially with identical kernel/IFV warmup outside the algorithm budget.',
        'A running packing call may overrun the time limit. Actual elapsed time and search-call counts are recorded. Invalid/failed results are excluded from quality averages and paired comparisons, never filled with zeros.',
        '', '| Dataset | Shape | Tolerance | Valid runs | Mean N-U* | Std | Mean N | Mean CA calls | Mean LS calls |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for a in aggregate:
        std='' if a['std_F'] is None else f"{a['std_F']:.6f}"
        report.append(f"| {a['dataset']} | {a['shape']} | {100*a['tolerance']:.0f}% | {a['successes']} | {a['mean_F']:.6f} | {std} | {a['mean_N']:.2f} | {a['mean_constructions']:.2f} | {a['mean_ls_calls']:.2f} |")
    report += ['', '## Paired outcomes against 10%', '',
               '| Shape | Challenger | Complete pairs | Wins | Ties | Losses |',
               '| --- | --- | --- | --- | --- | --- |']
    for shape in ('cube','cylinder'):
        group=[p for p in pairs if p['shape']==shape]
        for rate,key in ((15,'delta15'),(20,'delta20')):
            wins=sum(p[key]<-1e-10 for p in group);ties=sum(abs(p[key])<=1e-10 for p in group)
            report.append(f'| {shape} | {rate}% | {len(group)} | {wins} | {ties} | {len(group)-wins-ties} |')
    report += ['', 'Negative paired delta means lower (better) N-U*. Different datasets are not pooled by raw objective means. RCL frequencies describe each tolerance\'s actual construction trajectory, including evaluated orientations that were not selected.',
        'This pilot uses three order seeds and fixed algorithm RNG streams. It does not establish performance across other algorithm seeds, all datasets, or full accessibility trajectories. Fewer than two constructions indicate limited evidence about repeated GRASP diversification.',
        'Validation reconstructs placements without modulo wrapping and checks all piece IDs, array bounds, cylinder mask membership, overlap, and occupancy consistency. Full insertion trajectories are not validated.',
        'Files: results.csv; paired_comparison.csv; summary_by_dataset.csv; rcl_frequency.csv; per-run JSON, RCL CSV, and logs; source_snapshot/; manifest.json.']
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    save_json(out/'progress.json',dict(completed=len(results),total=len(plan),successes=len(successes),running=running))


def parent(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    hashes={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in SOURCE}
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if manifest['hashes']!=hashes:raise RuntimeError('Source changed; use a fresh output directory')
        plan=manifest['plan']
    else:
        plan=[];pair_index=0
        for dataset in args.datasets:
            for seed in args.seeds:
                for shape in args.shapes:
                    instance=f'{dataset}_seq{seed}_{shape}_999999_1'
                    if not (ROOT/'bin_packing/instances'/f'{instance}.txt').exists():raise FileNotFoundError(instance)
                    rates=[.1,.15,.2];offset=pair_index%3;rates=rates[offset:]+rates[:offset]
                    for rate in rates:
                        plan.append(dict(dataset=dataset,order_seed=seed,shape=shape,tolerance=rate,
                                         time_limit=args.time_limit,iterations=args.iterations))
                    pair_index+=1
        manifest=dict(hashes=hashes,plan=plan,created=time.strftime('%Y-%m-%dT%H:%M:%S'))
        save_json(out/'manifest.json',manifest)
        for rel in SOURCE:
            destination=out/'source_snapshot'/rel;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/rel,destination)
    results=[]
    for index,case in enumerate(plan):
        label=f"{case['dataset']}_{case['shape']}_seed{case['order_seed']}_t{int(case['tolerance']*100)}"
        result=out/(label+'.json')
        if result.exists():
            results.append(json.loads(result.read_text(encoding='utf-8')));continue
        if any(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h for rel,h in hashes.items()):
            summarize(out,plan,results);raise RuntimeError('Source changed during experiment')
        summarize(out,plan,results,running=label)
        print(f'[{index+1}/{len(plan)}] {label}',flush=True)
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--dataset',case['dataset'],
             '--shape',case['shape'],'--seed',str(case['order_seed']),'--tolerance',str(case['tolerance']),
             '--time-limit',str(case['time_limit']),'--iterations',str(case['iterations']),'--output',str(result)]
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            try:
                completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
                    timeout=max(1800,case['time_limit']*5),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                if not result.exists():save_json(result,dict(case,status='failed',reason=f'Worker exit {completed.returncode}'))
            except subprocess.TimeoutExpired:
                save_json(result,dict(case,status='failed',reason='Process safety timeout'))
        data=json.loads(result.read_text(encoding='utf-8'));results.append(data)
        summarize(out,plan,results)
        print(f"  {data['status']}: F={data.get('F')}; CA={data.get('constructions')}; LS={data.get('ls_calls')}",flush=True)
    summarize(out,plan,results)
    print('FINAL REPORT:',out/'report.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dataset')
    parser.add_argument('--shape',choices=['cube','cylinder'])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--tolerance',type=float)
    parser.add_argument('--datasets',nargs='+',default=['chess','engine','Merged2_normal'])
    parser.add_argument('--shapes',nargs='+',default=['cube','cylinder'],choices=['cube','cylinder'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[3,7,13])
    parser.add_argument('--time-limit',type=float,default=180)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();os.chdir(TEST)
    worker(args) if args.worker else parent(args)
