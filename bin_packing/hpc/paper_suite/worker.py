"""GPU worker for one frozen protocol case; no discarded CA warm-up."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
import traceback
from collections import Counter


def task_seeds(task):
    timestamp=time.time_ns()
    digest=hashlib.sha256(f'{timestamp}:{os.getpid()}:{socket.gethostname()}:{task}'.encode()).digest()
    return timestamp,{key:int.from_bytes(digest[i*4:i*4+4],'big')
                      for i,key in enumerate(('piece','position','kick','global'))}


def restore_radio_precision(objects,instance):
    """Match historical capacity calibration; legacy loader rounds to 6 places."""
    lines=[re.split(r'\t+',line.strip()) for line in
           instance.read_text(encoding='utf-8-sig').splitlines()[1:] if line.strip()]
    if len(lines)!=len(objects):raise ValueError('Loaded instance piece count changed')
    for info,fields in zip(objects,lines):
        if info['piece_type']!=int(fields[3]) or info['volume']!=float(fields[1]):
            raise ValueError('Loaded instance order/identity changed')
        radio=float(fields[2])
        if not math.isfinite(radio) or radio<0:raise ValueError('Invalid instance radiation')
        info['radio']=radio


def settings_for(case,profile,selections):
    stage=case['stage']
    if stage in ('calibration','sensitivity'):
        return 'improved_ILS',dict(ALG='fixed_CA',iteration_limit=0,time_limit=None)
    if stage=='rcl':
        return 'improved_GRASP',dict(iter_limit_per_iter=1,
            ls_evaluation_limit=0,restart_limit=1,rcl_tolerance=case['tolerance']
                if case['tolerance'] is not None else 0.,rcl_top_k=case['top_k'],
            time_limit=float('inf'),alpha=1,ls_policy='systematic')
    chosen=case if stage in ('tuning','memory') else selections['tuning'][case['algorithm']]
    algorithm=case['algorithm']
    kw=dict(time_limit=profile['tuning_time_limit'] if stage=='tuning' else profile['time_limit'])
    if algorithm=='improved_ILS':
        kw.update(ALG='ILS',iteration_limit=None,kick_trigger_time='auto',kick_level=chosen['kick_level'])
    else:
        rcl=(dict(tolerance=profile['resource_probe']['tolerance'],top_k=None) if stage=='memory'
             else selections['rcl']['global_rcl'])
        kw.update(alpha=1,restart_limit=None,rcl_tolerance=rcl['tolerance']
                  if rcl['tolerance'] is not None else 0.,rcl_top_k=rcl['top_k'])
        if algorithm=='improved_GRASP':
            kw.update(iter_limit_per_iter=1,ls_evaluation_limit=None,ls_policy='systematic')
        else:
            kw.update(iter_limit_per_iter=None,kick_trigger_time='auto',kick_level=chosen['kick_level'])
    return algorithm,kw


def read_selections(bundle,case):
    selections={}
    if case['stage']=='full':
        selections['rcl']=json.loads((bundle/'results/rcl/selection.json').read_text())
        selections['tuning']=json.loads((bundle/'results/tuning/selection.json').read_text())
    if case['stage'] in ('sensitivity','rcl','tuning','full'):
        selections['calibration']=json.loads((bundle/'results/calibration/selection.json').read_text())
    return selections


def verify_runtime(bundle,manifest,case):
    required=[name for name in manifest['files_sha256'] if name.startswith('driver/') or
              '/src/' in name or name.endswith('/code_launcher.py') or name.endswith('/metaheuristic_iterations.py')]
    required.append(f"runtime/bin_packing/instances/{case['instance']}")
    required+=manifest['instance_geometry'][case['instance']]
    for name in set(required):
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=manifest['files_sha256'][name]:
            raise ValueError('Frozen runtime/input changed: '+name)


def valid_saved(path,case,bundle_id):
    try:
        result=json.loads(path.read_text())
        return (result.get('status')=='success' and result.get('validation_ok') is True
                and result.get('case')==case and result.get('bundle_id')==bundle_id)
    except (OSError,ValueError):
        return False


def run(bundle,stage,index):
    bundle=bundle.resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    profile=manifest['profile']
    from protocol import VERSION
    if profile['protocol'] not in (VERSION,VERSION+'-memory-only'):
        raise ValueError('Use the frozen driver belonging to this bundle; protocol mismatch')
    case=profile['plans'][stage][index]
    out=bundle/'results'/stage
    out.mkdir(parents=True,exist_ok=True)
    path=out/f'task_{index}.json'
    if valid_saved(path,case,manifest['bundle_id']):
        print('Already completed; skipped',stage,index)
        return 0
    lock=out/f'task_{index}.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps(dict(pid=os.getpid(),host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'))).encode())
    os.close(fd)
    report=dict(case=case,bundle_id=manifest['bundle_id'],status='failed',validation_ok=False)
    started=time.perf_counter()
    try:
        stamp,seeds=task_seeds(f'{stage}/{index}')
        report.update(seed_time_ns=stamp,internal_seeds=seeds,order_seed=case['seed'])
        if path.exists():
            archive=out/'archive'/str(stamp)
            archive.mkdir(parents=True)
            path.rename(archive/path.name)
            events=out/f'task_{index}_events.jsonl'
            if events.exists():events.rename(archive/events.name)
        verify_runtime(bundle,manifest,case)
        import numpy as np
        from numba import cuda
        if not cuda.is_available():raise RuntimeError('CUDA is unavailable')
        project=bundle/'runtime/bin_packing'
        sys.path[:0]=[str(project/'src'),str(project/'test'),str(project/'hpc')]
        os.chdir(project/'test')
        import code_launcher as loader
        import new_ILS_southampton as algorithms
        if not hasattr(algorithms.improved_GRASP,'__wrapped__'):
            raise RuntimeError('Frozen controller lacks the shared packing integration')
        from function_lib import NFV_POOL,IFV_POOL
        from metaheuristic_iterations import validate_result
        shape,size,_,_,objects=loader.load_instance(Path(case['instance']).stem)
        restore_radio_precision(objects,project/'instances'/case['instance'])
        if not objects:raise ValueError('Empty instance')
        cfg=manifest['ca_parameters'][f"{case['dataset']}|{shape}"]
        selections=read_selections(bundle,case)
        algorithm,search=settings_for(case,profile,selections)
        capacity=math.prod(size) if shape=='cube' else math.pi*(size[0]/2)**2*size[2]
        radio_limit=sum(o['radio'] for o in objects)+1.
        max_weight=capacity
        constrained=stage in ('sensitivity','rcl','tuning','full')
        if constrained:
            calibrated=selections['calibration'][f"{case['dataset']}|{shape}"]
            max_weight=calibrated['weight_capacity'][case['weight_level']]
            radio_limit=calibrated['radio_capacity'][case['radio_level']]
        orientations=([0,90,180,270],['x','y','z'])
        labels=['x_0']+[f'{axis}_{degree}' for degree in (90,180,270) for axis in ('x','y','z')]
        counts=Counter()
        cache_counts=Counter()
        rcl_counts={'pooled':Counter()}
        history=[]
        ls_end_events=[]
        best=[float('inf')]
        clock=[time.perf_counter()]
        event_path=out/f'task_{index}_events.jsonl'
        # Preserve interrupted events as well as failure reports.
        if event_path.exists():
            archive=out/'archive'/str(stamp)
            archive.mkdir(parents=True,exist_ok=True)
            event_path.rename(archive/event_path.name)
        with event_path.open('x',encoding='utf-8',buffering=1) as event_file:
            def cache_event(**event):
                cache_counts[event['event']]+=1
            def rcl_event(event):
                rcl_counts['pooled'][event['active_rcl_size']]+=1
            def observe(kind,packed):
                counts[kind]+=1
                n,u,us=algorithms.get_final_performance(objects,size,shape,packed[0])
                us=u if us is None else us
                score=n-us
                event=dict(kind=kind,time_s=time.perf_counter()-clock[0],N=n,U=u,U_star=us,
                           objective=score,ls_evaluations=counts['ls'],kicks=counts['kick'],
                           constructions=counts['construction'])
                if kind=='construction' or score<best[0] or (kind=='ls' and counts['ls'] in (10,20,50)):
                    history.append(event)
                if score<best[0]:
                    best[0]=score
                    event['new_best']=True
                event_file.write(json.dumps(event)+'\n')
            random.seed(seeds['global'])
            np.random.seed(seeds['kick'])
            common=dict(object_info_total=objects,nfv_pool=NFV_POOL(),ifv_pool=IFV_POOL(),
                max_radio=radio_limit,rho=1.,orientations=orientations,ils_orientations=orientations,
                orientations_list=labels,packing_alg='SCH',selection_type='bounding_box',
                selection_range=cfg['selection_range'],accessible_check=True,
                SCH_nesting_strategy=cfg['pes'],orien_evaluation=cfg['pes'],
                container_size=size,container_shape=shape,flag_NFV_POOL=False,visualisation=False,
                _TRACE=False,piece_seed=seeds['piece'],position_seed=seeds['position'],
                radio_priority=constrained,cache_first_fit=True,max_weight=max_weight,weight_factor=1.,
                packing_observer=observe,cache_observer=cache_event)
            if algorithm=='improved_GRASP_ILS':common['kick_seed']=seeds['kick']
            if algorithm!='improved_ILS':common['rcl_observer']=rcl_event
            if algorithm=='improved_GRASP':
                def ls_end(event):
                    ls_end_events.append(dict(event))
                    event_file.write(json.dumps(dict(event,kind='ls_end',time_s=time.perf_counter()-clock[0]))+'\n')
                common['ls_observer']=ls_end
            cuda.synchronize()
            clock[0]=time.perf_counter()
            result=getattr(algorithms,algorithm)(**common,**search)
            cuda.synchronize()
            algorithm_s=time.perf_counter()-clock[0]
        validation=validate_result(result,objects,size,shape)
        for group in result[6]:
            if sum(p['radio'] for p in group)>radio_limit+1e-8 or sum(p['volume'] for p in group)>max_weight+1e-8:
                raise ValueError('Returned layout violates calibrated weight/radio limit')
        if stage in ('rcl','sensitivity','calibration') and (
                counts['construction']!=1 or counts['ls'] or counts['kick']):
            raise ValueError('CA-only work count violated')
        if abs(result[0]-result[2]-best[0])>1e-8:
            raise ValueError('Returned solution differs from observed best')
        volumes=validation['bin_volumes']
        emptiest=min(range(len(volumes)),key=lambda b:volumes[b])
        nstar=(len(objects)-len(result[9][emptiest]))/(len(volumes)-1) if len(volumes)>1 else len(objects)
        report.update(status='success',validation_ok=True,algorithm=algorithm,search_settings=search,
            method=dict(cfg,radio_priority=constrained,cache='run_scoped',rcl='pooled_orientation_position',
                        warmup='none'),best_N=result[0],best_U=result[1],best_U_star=result[2],
            objective=result[0]-result[2],origin_N=result[3],origin_U_star=result[5],
            origin_objective=result[3]-result[5],algorithm_wall_s=algorithm_s,
            reported_algorithm_wall_s=result[-1],counts=dict(counts),cache_counts=dict(cache_counts),
            rcl_histograms={k:dict(v) for k,v in rcl_counts.items()},history=history,ls_end_events=ls_end_events,
            validation=validation,mean_piece_count_without_emptiest=nstar,bin_capacity=capacity,
            max_piece_volume=max(p['volume'] for p in objects),max_piece_radio=max(p['radio'] for p in objects),
            weight_capacity=max_weight,radio_capacity=radio_limit,piece_count=len(objects),
            hostname=socket.gethostname(),python=sys.version,
            resolved_limits=dict(piece_count=len(objects),kick_failed_piece_threshold=math.ceil(len(objects)/2)
                if algorithm in ('improved_ILS','improved_GRASP_ILS') and stage not in ('calibration','sensitivity') else None,
                ls_repacks_per_restart=9*len(objects) if algorithm=='improved_GRASP_ILS' else None,
                kick_in_ls_budget=False,deadline_policy='finish in-flight call; start no new call after expiry'))
        if stage=='rcl':report.update(tolerance=case['tolerance'],top_k=case['top_k'])
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        try:
            import resource
            report['host_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            report['host_max_rss_kib']=None
        report['total_wall_s']=time.perf_counter()-started
        # Infinity is metadata for fixed work, not an invalid JSON numeric literal.
        if 'search_settings' in report and not math.isfinite(report['search_settings'].get('time_limit') or 0):
            report['search_settings']['time_limit']=None
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2,default=lambda v:v.item(),allow_nan=False),encoding='utf-8')
        temp.replace(path)
        lock.unlink()
    print(json.dumps(dict(stage=stage,index=index,status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status']=='success' else 1
