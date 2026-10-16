"""All-dataset RCL experiment: one complete CA followed by 50 LS evaluations."""
import argparse
from collections import Counter
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

TEST = Path(__file__).resolve().parent
ROOT = TEST.parents[1]
RATES = (.10, .15, .20)
SEEDS = (3, 7, 13)
SHAPES = ('cube', 'cylinder')
PES = {'cube': 'maximal_residual_box', 'cylinder': 'minimum_aabb_edges_len'}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_json(p, value):
    p = Path(p); temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False,
                    default=lambda v: v.item()), encoding='utf-8')
    temp.replace(p)


def write_csv(p, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with Path(p).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def label(c):
    return f"{c['dataset']}_{c['shape']}_seed{c['order_seed']}_t{round(c['tolerance']*100)}"


def fixed_work_call(function, *args, **kwargs):
    """Bind positional launcher arguments before overriding work budgets."""
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.arguments.update(iter_limit_per_iter=50, time_limit=float('inf'),
                           restart_limit=1, no_improvement_limit=50)
    if 'ls_evaluation_limit' in inspect.signature(function).parameters:
        bound.arguments['ls_evaluation_limit'] = 50
    return function(*bound.args, **bound.kwargs)


def assert_work_counts(ca_calls, ls_calls, best_n):
    if ca_calls != 1:
        raise ValueError(f'Expected one construction, got {ca_calls}')
    if not 0 <= ls_calls <= 50:
        raise ValueError(f'Invalid LS count: {ls_calls}')
    if ls_calls != 50 and best_n != 1:
        raise ValueError(f'Only {ls_calls}/50 LS evaluations without a single-bin stop')
    return '50_evaluations_completed' if ls_calls == 50 else 'single_bin_reached'


def worker(a):
    case = dict(dataset=a.dataset, shape=a.shape, order_seed=a.seed,
                tolerance=a.tolerance, piece_seed=101, position_seed=202,
                pes=PES[a.shape], selection_range='bottom_top',
                target_ls_evaluations=50, wall_time_limit=None,
                restart_limit=1, no_improvement_limit=50,
                algorithm='improved_GRASP', protocol='one_CA_plus_50_LS')
    result_path = Path(a.output)
    status_path = result_path.with_suffix('.status.json')
    def status(phase, **values):
        save_json(status_path, dict(case=label(case), phase=phase, **values))
    try:
        import numpy as np
        from numba import cuda
        import code_launcher as launcher
        import new_ILS_southampton as alg
        from run_tolerance_experiment import validate
        from function_lib import get_nfv_gpu, njit_get_ifv_cylinder, get_bounding_box_njit_translation
        if not cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        status('loading')
        instance = launcher.make_instance_id(a.dataset, a.seed, a.shape, 999999, 1)
        _, size, _, _, objects = launcher.load_instance(instance)
        small = np.zeros((4,4,4), dtype=objects[0]['array'].dtype); small[0,0,0] = 1
        fixed = np.zeros((4,4,4)); fixed[0,0,0] = 1
        get_nfv_gpu(fixed, small, small.shape, (4,4,4))
        get_bounding_box_njit_translation(fixed)
        njit_get_ifv_cylinder(small, 1, 1, 1, small.shape)
        cuda.synchronize()
        events, history, capture = [], [], {}
        counts, timings = Counter(), Counter()
        clock = [None]
        def observer(e):
            if e['best_pes'] is not None:
                events.append(dict(restart=e['restart'], piece_index=e['piece_index'],
                    bin_index=e['bin_index'], orientation=e['orientation'],
                    candidate_count=len(e['candidates']), rcl_size=e['active_rcl_size'],
                    best_pes=e['best_pes'], selected_pes=e['selected_pes']))
        for name, kind in [('packing_3D_voxel_lookback', 'construction'), ('improved_repack', 'ls')]:
            function = getattr(alg, name)
            def counted(*args, _f=function, _kind=kind, **kwargs):
                status(_kind, completed_ls=counts['ls'],
                       next_ls=counts['ls']+1 if _kind == 'ls' else None)
                start = time.perf_counter(); packed = _f(*args, **kwargs)
                counts[_kind] += 1; timings[_kind] += time.perf_counter()-start
                n,u,us = alg.get_final_performance(objects, size, a.shape, packed[0])
                us = u if us is None else us
                f = n-us
                history.append(dict(evaluation=counts['ls'], kind=_kind,
                                    seconds=time.perf_counter()-clock[0], N=n, U_star=us, F=f))
                if _kind == 'construction':
                    dummy = [None]*11; dummy[6],dummy[8],dummy[9] = packed[0],packed[1],packed[3]
                    errors = validate(dummy, objects, size, a.shape)
                    capture['ca_validation_errors'] = errors
                    if errors:
                        raise ValueError('CA geometry validation failed: '+repr(errors))
                print(f"{_kind}: CA={counts['construction']} LS={counts['ls']}/50 F={f:.9f}", flush=True)
                return packed
            setattr(alg, name, counted)
        original = launcher.improved_GRASP
        def measured(*args, **kwargs):
            clock[0] = time.perf_counter()
            result = fixed_work_call(original, *args, **kwargs)
            elapsed = time.perf_counter()-clock[0]
            reason = assert_work_counts(counts['construction'], counts['ls'], result[0])
            errors = validate(result, objects, size, a.shape)
            capture.update(best_N=result[0], best_U_star=result[2], F=result[0]-result[2],
                origin_N=result[3], origin_U_star=result[5], origin_F=result[3]-result[5],
                improvement_F=(result[3]-result[5])-(result[0]-result[2]),
                wall_seconds=elapsed, algorithm_seconds=result[-1],
                construction_seconds=timings['construction'], ls_seconds=timings['ls'],
                constructions=counts['construction'], ls_calls=counts['ls'],
                stop_reason=reason, validation_errors=errors, dimensions=list(size),
                cuda_device=str(cuda.get_current_device()))
            return result
        launcher.improved_GRASP = measured
        rows = launcher.testing([a.dataset], [a.seed], ['improved_GRASP'], [a.shape],
            experiment_iterations=50, experiment_time=float('inf'),
            selection_ranges=['bottom_top'], nesting_strategies=[PES[a.shape]],
            rcl_tolerance=a.tolerance, piece_seed=101, position_seed=202, rcl_observer=observer)
        case.update(capture, status='success' if not capture['validation_errors'] else 'invalid',
            launcher_row=rows[0], history=history,
            rcl_histogram=dict(Counter(e['rcl_size'] for e in events)), rcl_calls=len(events),
            rcl_mean=sum(e['rcl_size'] for e in events)/len(events) if events else None)
        for checkpoint in (10,20,50):
            known=[h['F'] for h in history if h['kind']=='construction' or h['evaluation']<=checkpoint]
            case[f'F_after_{checkpoint}_LS'] = min(known)
        write_csv(result_path.with_suffix('.rcl.csv'), events,
                  ['restart','piece_index','bin_index','orientation','candidate_count','rcl_size','best_pes','selected_pes'])
        write_csv(result_path.with_suffix('.ls.csv'), history,
                  ['evaluation','kind','seconds','N','U_star','F'])
    except Exception:
        case.update(status='failed', traceback=traceback.format_exc())
        print(case['traceback'], flush=True)
    save_json(result_path, case)
    status('finished', status=case['status'], ls_calls=case.get('ls_calls'))
    print(json.dumps({k:case.get(k) for k in ['dataset','shape','order_seed','tolerance','status','F','ls_calls','stop_reason']},
                     default=lambda v:v.item()), flush=True)


def outcomes(pairs, a, b):
    d=[x[a]-x[b] for x in pairs]
    return sum(x < -1e-10 for x in d),sum(abs(x)<=1e-10 for x in d),sum(x > 1e-10 for x in d)


def summarize(out, plan, results, running=None):
    valid=[r for r in results if r['status']=='success']
    fields=['dataset','shape','order_seed','tolerance','status','origin_N','origin_U_star','origin_F',
            'best_N','best_U_star','F','improvement_F','constructions','ls_calls','stop_reason',
            'construction_seconds','ls_seconds','wall_seconds','rcl_calls','rcl_mean',
            'F_after_10_LS','F_after_20_LS','F_after_50_LS']
    write_csv(out/'results.csv',results,fields)
    pairs=[];table=[];aggregate=[]
    for ds in dict.fromkeys(c['dataset'] for c in plan):
        for sh in SHAPES:
            group=[r for r in valid if (r['dataset'],r['shape'])==(ds,sh)]
            matched=[]
            for seed in SEEDS:
                byrate={r['tolerance']:r for r in group if r['order_seed']==seed}
                if len(byrate)!=3:continue
                p=dict(dataset=ds,shape=sh,order_seed=seed)
                for rate in RATES:
                    tag=str(round(rate*100));p['F'+tag]=byrate[rate]['F'];p['CA_F'+tag]=byrate[rate]['origin_F']
                p['delta15_vs10']=p['F15']-p['F10'];p['delta20_vs10']=p['F20']-p['F10']
                p['delta15_vs20']=p['F15']-p['F20'];pairs.append(p);matched.append(p)
            row=dict(dataset=ds,shape=sh)
            means={};ca_means={}
            for rate in RATES:
                tag=str(round(rate*100));g=[r for r in group if r['tolerance']==rate]
                row['valid_'+tag]=len(g)
                for name,key in [('mean_CA_F','origin_F'),('mean_F','F'),('mean_improvement','improvement_F'),
                                 ('mean_wall_seconds','wall_seconds'),('mean_ls_calls','ls_calls')]:
                    row[name+'_'+tag]=statistics.mean(r[key] for r in g) if g else None
                row['std_F_'+tag]=statistics.stdev(r['F'] for r in g) if len(g)>1 else None
                row['early_single_bin_'+tag]=sum(r['ls_calls']<50 for r in g)
                h=Counter()
                for r in g:h.update({int(k):v for k,v in r['rcl_histogram'].items()})
                calls=sum(h.values())
                row['mean_RCL_'+tag]=sum(k*v for k,v in h.items())/calls if calls else None
                row['RCL_3_4_percent_'+tag]=100*(h[3]+h[4])/calls if calls else None
                means[rate]=row['mean_F_'+tag];ca_means[rate]=row['mean_CA_F_'+tag]
                if g:
                    aggregate.append(dict(dataset=ds,shape=sh,tolerance=rate,valid_runs=len(g),
                        mean_CA_F=row['mean_CA_F_'+tag],mean_final_F=row['mean_F_'+tag],
                        std_final_F=row['std_F_'+tag],mean_improvement=row['mean_improvement_'+tag],
                        mean_ls_calls=row['mean_ls_calls_'+tag],mean_wall_seconds=row['mean_wall_seconds_'+tag]))
            complete=all(row['valid_'+str(round(t*100))]==3 for t in RATES)
            row['status']='complete' if complete else 'partial'
            row['best_tolerance_percent']='';row['best_CA_tolerance_percent']=''
            if complete:
                row['best_tolerance_percent']='/'.join(str(round(t*100)) for t in RATES if abs(means[t]-min(means.values()))<=1e-10)
                row['best_CA_tolerance_percent']='/'.join(str(round(t*100)) for t in RATES if abs(ca_means[t]-min(ca_means.values()))<=1e-10)
            for tag,a,b in [('15_vs_10','F15','F10'),('20_vs_10','F20','F10'),('15_vs_20','F15','F20')]:
                for k,v in zip(['wins','ties','losses'],outcomes(matched,a,b)):row[tag+'_'+k]=v
            table.append(row)
    write_csv(out/'best_tolerance_by_dataset.csv',table)
    write_csv(out/'summary_by_dataset.csv',aggregate,
              ['dataset','shape','tolerance','valid_runs','mean_CA_F','mean_final_F','std_final_F','mean_improvement','mean_ls_calls','mean_wall_seconds'])
    write_csv(out/'paired_comparison.csv',pairs,
              ['dataset','shape','order_seed','F10','CA_F10','F15','CA_F15','F20','CA_F20',
               'delta15_vs10','delta20_vs10','delta15_vs20'])
    frequencies=[]
    for sh in SHAPES:
        for rate in RATES:
            h=Counter()
            for r in valid:
                if (r['shape'],r['tolerance'])==(sh,rate):h.update({int(k):v for k,v in r['rcl_histogram'].items()})
            for size,count in sorted(h.items()):
                frequencies.append(dict(shape=sh,tolerance=rate,rcl_size=size,count=count,
                                        percent=100*count/sum(h.values()),total_calls=sum(h.values())))
    write_csv(out/'rcl_frequency.csv',frequencies,['shape','tolerance','rcl_size','count','percent','total_calls'])
    lines=['# RCL tolerance experiment: one CA plus 50 LS evaluations','',
           f'Completed: {len(results)}/{len(plan)}; successful: {len(valid)}; failed/invalid: {len(results)-len(valid)}.',
           f'Current task: {running or "none"}.','',
           'Each run completes exactly one randomized construction, then 50 evaluated LS neighbors. '
           'There is no wall-time cutoff, no restart and no stagnation exit before the evaluation budget. '
           'A single-bin solution may stop early; its actual LS count and reason are recorded. '
           'LS begins at its own construction and continues from the accepted solution after improvements.',
           'Order seeds 3/7/13; piece_seed 101; position_seed 202; tolerances 10%/15%/20%. '
           'Cube PES maximal_residual_box; cylinder PES minimum_aabb_edges_len; bottom_top. '
           'Radio 999999 and rho 1. This evaluates the original improved_GRASP, not the separate weight-radio variant.',
           '', '| Dataset | Shape | Valid 10/15/20 | CA F 10/15/20 | Final F 10% | Final F 15% | Final F 20% | Best final rate | 15 vs 20 W/T/L |',
           '|---|---|---|---|---:|---:|---:|---|---|']
    fmt=lambda v:'' if v is None else f'{v:.6f}'
    for r in table:
        tags=['10','15','20'];counts='/'.join(str(r['valid_'+t]) for t in tags)
        ca='/'.join(fmt(r['mean_CA_F_'+t]) or 'pending' for t in tags)
        finals=[fmt(r['mean_F_'+t]) for t in tags]
        wtl='/'.join(str(r['15_vs_20_'+k]) for k in ['wins','ties','losses'])
        lines.append('| '+' | '.join([r['dataset'],r['shape'],counts,ca,*finals,r['best_tolerance_percent'] or 'pending',wtl])+' |')
    lines+=['','## Interpretation','',
            'Lower N-U* is better. Best rates are assigned only after all nine cases in a dataset/shape group succeed. '
            'Ties retain all tied rates. Raw objectives are not pooled across datasets. Failures/missing cases are never zero-filled. '
            'CA and final quality, improvement, runtime, standard deviation, actual LS counts, RCL means/frequencies and paired outcomes are saved in CSV.',
            'These are parameter-selection results on three order seeds. Validate selected rates with held-out seeds before generalization claims. '
            'Equal search work is not equal wall time. Geometry validation checks IDs, bounds, cylinder mask, overlap and occupancy, not full insertion trajectories.',
            'Files: results.csv, best_tolerance_by_dataset.csv, summary_by_dataset.csv, paired_comparison.csv, '
            'rcl_frequency.csv, per-case .json/.status.json/.log/.rcl.csv/.ls.csv, manifest.json and frozen_runtime/.']
    report='\n'.join(lines)+'\n'
    (out/'report.md').write_text(report,encoding='utf-8')
    (out/'best_tolerance_by_dataset.md').write_text(report,encoding='utf-8')
    save_json(out/'progress.json',dict(completed=len(results),total=len(plan),successes=len(valid),
                                    failed_or_invalid=len(results)-len(valid),running=running,protocol='one_CA_plus_50_LS'))
    if len(results)==len(plan):
        audit=[]
        for r in results:
            if r['status']!='success':audit.append(dict(case=label(r),error=r.get('traceback','failed/invalid')));continue
            try:
                assert_work_counts(r['constructions'],r['ls_calls'],r['best_N'])
                assert not r['validation_errors'] and not r['ca_validation_errors']
                assert abs(r['F']-(r['best_N']-r['best_U_star']))<1e-10
                assert len([h for h in r['history'] if h['kind']=='ls'])==r['ls_calls']
                with (out/(label(r)+'.rcl.csv')).open(newline='',encoding='utf-8') as f:
                    h=Counter(int(x['rcl_size']) for x in csv.DictReader(f))
                assert dict(h)=={int(k):v for k,v in r['rcl_histogram'].items()}
            except Exception as e:audit.append(dict(case=label(r),error=repr(e)))
        save_json(out/'final_audit.json',dict(total=len(plan),valid=len(plan)-len(audit),errors=audit))
        (out/'final_report.md').write_text(report+f'\nFinal audit: {len(plan)-len(audit)}/{len(plan)} valid cases.\n',encoding='utf-8')


def prepare(out, baseline):
    old=json.loads((baseline/'manifest.json').read_text())
    hashes={rel:digest(ROOT/rel) for rel in old['hashes']}
    if hashes!=old['hashes']:raise RuntimeError('Frozen original sources changed')
    inputs={rel:digest(ROOT/rel) for rel in old['input_hashes']}
    if inputs!=old['input_hashes']:raise RuntimeError('Frozen inputs changed')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        m=json.loads((out/'manifest.json').read_text())
        if m['hashes']!=hashes or m['runner_hash']!=digest(__file__) or m['input_hashes']!=inputs:
            raise RuntimeError('Experiment runtime changed; refusing mixed-source resume')
        return m
    plan=[dict(dataset=c['dataset'],shape=c['shape'],order_seed=c['order_seed'],
               tolerance=c['tolerance'],ls_evaluations=50) for c in old['plan']]
    m=dict(protocol='one_CA_plus_50_LS',plan=plan,hashes=hashes,input_hashes=inputs,
           runner_hash=digest(__file__),baseline_180s_archived=str(baseline),
           runtime=str(ROOT),configuration=dict(one_CA=True,target_LS=50,time_limit=None,restarts=1,
                    seeds=list(SEEDS),piece_seed=101,position_seed=202,rates=list(RATES),shapes=list(SHAPES),
                    early_stop='single_bin_reached_only'))
    save_json(out/'manifest.json',m)
    return m


def parent(a):
    out=Path(a.output).resolve();m=prepare(out,Path(a.baseline).resolve());plan=m['plan']
    results=[]
    for c in plan:
        f=out/(label(c)+'.json')
        if f.exists():results.append(json.loads(f.read_text()))
    summarize(out,plan,results)
    if a.prepare_only:return
    for index,c in enumerate(plan):
        f=out/(label(c)+'.json')
        if f.exists():continue
        for rel,h in m['hashes'].items():
            if digest(ROOT/rel)!=h:raise RuntimeError('Frozen source changed: '+rel)
        if digest(__file__)!=m['runner_hash']:raise RuntimeError('Runner changed')
        summarize(out,plan,results,running=label(c))
        print(f'[{index+1}/{len(plan)}] {label(c)}',flush=True)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
                 '--dataset',c['dataset'],'--shape',c['shape'],'--seed',str(c['order_seed']),
                 '--tolerance',str(c['tolerance']),'--output',str(f)]
        with (out/(label(c)+'.log')).open('w',encoding='utf-8') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if not f.exists():save_json(f,dict(c,status='failed',reason=f'Worker exit {completed.returncode}'))
        r=json.loads(f.read_text());results.append(r);summarize(out,plan,results)
        print(f"  {r['status']}: CA F={r.get('origin_F')} final F={r.get('F')} LS={r.get('ls_calls')}/50",flush=True)
    summarize(out,plan,results)
    print('FINAL REPORT:',out/'final_report.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dataset')
    parser.add_argument('--shape',choices=SHAPES)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--tolerance',type=float)
    parser.add_argument('--output',required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--prepare-only',action='store_true')
    a=parser.parse_args();os.chdir(TEST)
    worker(a) if a.worker else parent(a)
