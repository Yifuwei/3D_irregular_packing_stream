"""Freeze, submit, resume and collect the EJOR experiment graph (stdlib only)."""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import uuid
from collections import defaultdict

from protocol import make_plan, choose_ca_parameters, calibrate, select_tolerances, select_tuning, STAGES, VERSION, SEQUENCE_SEEDS, REFERENCE_SEEDS, kick_tuning_datasets
from worker import run, valid_saved, packing_settings, read_selections



def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def freeze(project,summary,output,datasets=None,tuning=None,time_limit=3600,smoke=False,
           shapes=('cube','cylinder'),ca_json_dir=None,reuse_bundle=None,plan='v3'):
    project=project.resolve()
    if plan not in ('v3','v4'):raise ValueError('Unknown experiment plan')
    if plan=='v4' and reuse_bundle is None:
        raise ValueError('The v4 plan preserves existing capacities; provide --reuse-bundle with completed calibration')
    if reuse_bundle is not None and ca_json_dir is not None:
        raise ValueError('Choose either --reuse-bundle or --ca-json-dir')
    with summary.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    datasets=datasets or sorted({r['dataset'] for r in rows})
    tuning=tuning or kick_tuning_datasets(datasets)
    builder=make_plan
    if plan=='v4':
        from protocol_v4 import make_plan as builder
    profile=builder(datasets,tuning,time_limit=time_limit,smoke=smoke,shapes=shapes,
                      **(dict(seeds=(3,),reference_seeds=REFERENCE_SEEDS if (ca_json_dir or reuse_bundle) else (3,),
                              full_seeds=(3,)) if smoke else {}))
    geometry_ca=choose_ca_parameters(rows,datasets)
    geometry_ca={key:value for key,value in geometry_ca.items() if key.split('|')[-1] in shapes}
    ca=dict(geometry_ca)
    if reuse_bundle is not None:
        prior=json.loads((reuse_bundle/'manifest.json').read_text())
        ca={key:dict(prior['ca_parameters'][key]) for key in geometry_ca}
    # Historical CA settings remain fixed for calibration and sensitivity reuse.
    # RCL/tuning/full resolve their separate per-shape parameters in the worker.
    for config in ca.values():
        config['reference_selection_range']=config['selection_range']
    if output.exists():raise ValueError('Use a new output directory: '+str(output))
    output.mkdir(parents=True)
    def copy(source,relative):
        dest=output/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
    try:
        for source in (project/'src').glob('*.py'):
            copy(source,'runtime/bin_packing/src/'+source.name)
        copy(project/'test/code_launcher.py','runtime/bin_packing/test/code_launcher.py')
        copy(project/'hpc/metaheuristic_iterations.py','runtime/bin_packing/hpc/metaheuristic_iterations.py')
        for source in Path(__file__).parent.iterdir():
            if source.suffix in ('.py','.slurm','.sh','.md'):
                copy(source,'driver/'+source.name)
        copy(summary,'inputs/ca_reference_summary.csv')
        geometry={}
        piece_counts={}
        for cases in profile['plans'].values():
            for case in cases:
                name=f"{case['dataset']}_seq{case['seed']}_{case['shape']}_999999_1.txt"
                case['instance']=name
                if name in geometry:
                    case['piece_count']=piece_counts[name]
                    continue
                source=project/'instances'/name
                lines=source.read_text(encoding='utf-8-sig').splitlines()
                case['piece_count']=piece_counts[name]=sum(bool(line.strip()) for line in lines[1:])
                if lines[0].split()[0]!=case['shape']:raise ValueError('Instance shape mismatch: '+name)
                copy(source,'runtime/bin_packing/instances/'+name)
                paths=[]
                for line in lines[1:]:
                    if not line.strip():continue
                    reference=line.split('\t')[0].strip().replace('\\','/')
                    if reference.startswith('../'):reference=reference[3:]
                    data=(project/'data'/reference).resolve()
                    relative=data.relative_to((project/'data').resolve())
                    dest='runtime/bin_packing/data/'+relative.as_posix()
                    if not (output/dest).exists():copy(data,dest)
                    paths.append(dest)
                geometry[name]=sorted(set(paths))
        save(output/'parameter_plan.json',dict(protocol=profile['protocol'],profile=profile,ca_parameters=ca,geometry_ca_parameters=geometry_ca,
            conventions=dict(metaheuristic_packing_scope=('fixed overall best PES per shape' if plan=='v4' else 'fixed per bin shape; historical calibration unchanged'),
                kick_trigger='ceil(n/2) consecutive rejected LS repacking calls; reset on improvement',
                kick_size='ceil(fraction*n) changed orientation labels',
                hybrid_budget='n LS repacking calls per construction; never reset on improvement',
                kick_charged_to_ls_budget=False,kick_charged_to_wall_time=True,
                orientation_labels=10,alternative_labels_per_piece=9,
                rcl_pool='up to 160 orientation-position pairs in current bin; bottom/top extrema',
                score='used_radio/max_radio + used_U/Umax',
                Umax='min(1,rho,max_weight/(weight_factor*bin_volume))',
                objective='N-U_star; U_star excludes the least-filled bin by volume',
                parameter_selection='global arithmetic mean across matched cells; candidate order breaks ties',
                deadline='includes construction, LS and kicks; finish one in-flight call before stopping')))
        recovered=[]
        if ca_json_dir:
            from ca_reference import recover_selected
            recovered,capacities,indexed_count=recover_selected(dict(profile=profile,ca_parameters=ca),project,ca_json_dir)
            for record in recovered:
                copy(Path(record['source_file']),f"inputs/ca_references/task_{record['case']['task_id']}.json")
        hashes={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob('*') if p.is_file()}
        manifest=dict(bundle_id=str(uuid.uuid4()),profile=profile,ca_parameters=ca,geometry_ca_parameters=geometry_ca,
                      instance_geometry=geometry,files_sha256=hashes)
        if recovered:
            manifest['calibration_source']=dict(kind='historical_CA',indexed_results=indexed_count,
                recovered_references=len(recovered),capacity_preview=capacities)
            for record in recovered:
                case=record['case']
                source=f"inputs/ca_references/task_{case['task_id']}.json"
                report=dict(record,bundle_id=manifest['bundle_id'],status='success',validation_ok=True,
                    calibration_source='historical_CA',source_reference=source,
                    algorithm='historical_fixed_CA',search_settings=record['historical_config'],
                    method=dict(radio_priority=False,warmup='historical_not_recorded'),
                    counts=dict(construction=1,ls=0,kick=0),
                    validation=dict(ok=True,scope='historical geometry validation; counts and N/U/U* independently recovered'))
                save(output/'results/calibration'/f"task_{case['task_id']}.json",report)
        from bundle_integrity import canonicalize_manifest
        canonicalize_manifest(output,manifest)
        save(output/'manifest.json',manifest)
        if reuse_bundle is not None:
            from reuse_completed import import_completed
            imported=import_completed(output,reuse_bundle,stages=('calibration',) if plan=='v4' else ('calibration','sensitivity'))
            manifest=json.loads((output/'manifest.json').read_text())
            collect(output,'calibration')
            if plan=='v4':
                from reuse_candidates_v4 import import_candidates
                imported['sensitivity']=import_candidates(output,reuse_bundle)
                manifest=json.loads((output/'manifest.json').read_text())
            collect(output,'sensitivity')
            print(json.dumps(dict(reused=imported),indent=2))
        print(json.dumps(dict(bundle=str(output),tasks={s:len(profile['plans'][s]) for s in STAGES}),indent=2))
        return manifest
    except Exception:
        # Retain the partial directory for inspection; it has no valid manifest.
        raise


def collect(bundle,stage):
    manifest=json.loads((bundle/'manifest.json').read_text())
    from protocol_v4 import VERSION as V4_VERSION
    is_v4=manifest['profile']['protocol']==V4_VERSION
    if manifest['profile']['protocol'] not in (VERSION,V4_VERSION):
        raise ValueError('Use the frozen driver belonging to this bundle; protocol mismatch')
    cases=manifest['profile']['plans'][stage]
    records=[]
    incomplete=[]
    for case in cases:
        path=bundle/'results'/stage/f"task_{case['task_id']}.json"
        if valid_saved(path,case,manifest['bundle_id']):
            report=json.loads(path.read_text())
            if stage!='calibration':
                expected_packing=packing_settings(case,manifest,read_selections(bundle,case))
                if any(report.get('method',{}).get(k)!=expected_packing[k] for k in ('pes','selection_range')):
                    incomplete.append(case['task_id'])
                    continue
            records.append(dict(report,**case))
        else:incomplete.append(case['task_id'])
    root=bundle/'results'/stage
    root.mkdir(parents=True,exist_ok=True)
    fields=['task_id','dataset','shape','seed','algorithm','config_id','weight_level','radio_level',
            'pes','selection_range','best_N','best_U','best_U_star','objective','algorithm_wall_s','total_wall_s']
    with (root/'runs.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    save(root/'completion.json',dict(expected=len(cases),success=len(records),incomplete=incomplete))
    if incomplete:
        for stale in ('selection.json','summary.csv'):
            (root/stale).unlink(missing_ok=True)
        print(f'{stage}: {len(records)}/{len(cases)} successful; selection withheld; missing/failed {incomplete[:20]}')
        return 1
    if stage in ('rcl','tuning'):
        expected=set(manifest['profile']['tuning_datasets'] if stage=='tuning' else manifest['profile']['datasets'])
        if {r['dataset'] for r in records}!=expected:
            raise ValueError('Parameter selection requires every dataset in its configured population')
    if stage=='calibration':
        groups=defaultdict(list)
        for r in records:groups[f"{r['dataset']}|{r['shape']}"].append(r)
        selection={key:calibrate(values) for key,values in groups.items()}
    elif stage=='sensitivity' and is_v4:
        from protocol_v4 import select_sensitivity
        selection=select_sensitivity(records,manifest['profile'])
        rows=selection['summary']
        with (root/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    elif stage=='rcl':selection=select_tolerances(records)
    elif stage=='tuning':selection=select_tuning(records)
    else:
        grouped=defaultdict(list)
        for r in records:
            grouped[(r['dataset'],r['shape'],r['weight_level'],r['radio_level'],r['algorithm'])].append(r)
        selection=[]
        for key,values in sorted(grouped.items()):
            row=dict(zip(('dataset','shape','weight_level','radio_level','algorithm'),key))
            row['repetitions']=len(values)
            for field in ('best_N','best_U','best_U_star','objective','algorithm_wall_s'):
                samples=[r[field] for r in values]
                row[field+'_mean']=statistics.mean(samples)
                row[field+'_std']=statistics.stdev(samples) if len(samples)>1 else 0.
            if stage=='sensitivity':
                baseline=manifest.get('geometry_ca_parameters',manifest['ca_parameters'])[f'{key[0]}|{key[1]}']['reference_mean_objective']
                row['geometry_reference_objective_mean']=baseline
                row['objective_change_from_geometry']=row['objective_mean']-baseline
            selection.append(row)
        with (root/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(selection[0]));writer.writeheader();writer.writerows(selection)
    save(root/'selection.json',selection)
    print(f'{stage}: {len(records)}/{len(cases)} successful; selection saved')
    if stage=='full' and manifest['profile'].get('smoke'):
        if is_v4:
            from verify_v4 import verify_bundle as verify_smoke
        else:
            from smoke import verify_smoke
        verification=verify_smoke(bundle)
        save(bundle/'smoke_verification.json',verification)
        print('Smoke verification: '+verification['status'])
        return 0 if verification['status']=='PASS' else 1
    return 0


def memory_for(case,small=16,large=32):
    return small if case['dataset'].lower() in ('chess','shapesnew') else large


def submit(bundle,max_parallel=6,small_memory=16,large_memory=32,walltime='12:00:00',
           dry_run=False,pending=False,through='full',partition='gpu',control_partition=None,flow='all'):
    if max_parallel<1 or min(small_memory,large_memory)<1:raise ValueError('Resources must be positive')
    bundle=bundle.resolve()
    if ',' in str(bundle) or '\n' in str(bundle):raise ValueError('Bundle path cannot contain commas/newlines')
    manifest=json.loads((bundle/'manifest.json').read_text())
    from protocol_v4 import VERSION as V4_VERSION
    is_v4=manifest['profile']['protocol']==V4_VERSION
    if manifest['profile']['protocol'] not in (VERSION,V4_VERSION):
        raise ValueError('Use the frozen driver belonging to this bundle; protocol mismatch')
    if flow not in ('all','sensitivity','heuristics'):raise ValueError('Unknown submission flow')
    requested=STAGES[:STAGES.index(through)+1]
    if flow=='sensitivity':requested=tuple(s for s in requested if s in ('calibration','sensitivity'))
    if flow=='heuristics':requested=tuple(s for s in requested if s!='sensitivity')
    if flow!='all':
        # Separate flows share already-calibrated capacities, not a collector job.
        cases=manifest['profile']['plans']['calibration']
        if not (bundle/'results/calibration/selection.json').is_file() or not all(
            valid_saved(bundle/'results/calibration'/f"task_{c['task_id']}.json",c,manifest['bundle_id']) for c in cases):
            raise ValueError('Complete calibration before submitting separate flows')
        requested=tuple(s for s in requested if s!='calibration')
    if not requested:raise ValueError('No stages selected for this flow/through combination')
    if not dry_run:
        from submission_graph import active_job_ids
        names={'ejor_'+s for s in requested}|{'ejor_collect_'+s for s in requested}
        for job in active_job_ids(os.environ['USER']):
            details=subprocess.check_output(['scontrol','show','job',job],text=True)
            match=re.search(r'\bJobName=(\S+)',details)
            if 'StdOut='+str(bundle/'logs') in details:
                if match is None or match.group(1) in names:
                    raise RuntimeError('An active job already targets this submission flow: '+job)
        locks=[p for s in requested for p in (bundle/'results'/s).glob('task_*.lock')]
        if locks:raise RuntimeError('Interrupted task locks remain; verify no active jobs then use unlock: '+str(locks[:3]))
    (bundle/'logs').mkdir(exist_ok=True)
    submissions=[]
    history_path=bundle/'submission_history'/f'{uuid.uuid4()}.json'
    def sbatch(args):
        command=['sbatch','--parsable',*args]
        print(' '.join(command),flush=True)
        if dry_run:job=str(900000+len(submissions))
        else:job=subprocess.check_output(command,text=True).strip().split(';')[0]
        submissions.append(dict(job_id=job,command=command))
        if not dry_run:save(history_path,submissions)
        return job
    if is_v4:
        from protocol_v4 import PARENTS
    else:
        from submission_graph import PARENTS
    stages=requested
    grouped={}
    for stage in stages:
        groups=defaultdict(list)
        for case in manifest['profile']['plans'][stage]:
            path=bundle/'results'/stage/f"task_{case['task_id']}.json"
            done=valid_saved(path,case,manifest['bundle_id'])
            if done and stage!='calibration':
                report=json.loads(path.read_text())
                try:
                    cfg=packing_settings(case,manifest,read_selections(bundle,case))
                    done=all(report.get('method',{}).get(k)==cfg[k] for k in ('pes','selection_range'))
                except (OSError,ValueError,KeyError):done=False
            if not done:groups[memory_for(case,small_memory,large_memory)].append(case['task_id'])
        grouped[stage]=groups
    collectors={}
    for stage in stages:
        parent_jobs=[collectors[p] for p in PARENTS[stage] if p in collectors]
        gpu_jobs=[];last_group=None
        env=f'ALL,EJOR_BUNDLE={bundle},EJOR_STAGE={stage}'
        for memory,indices in sorted(grouped[stage].items()):
            for offset in range(0,len(indices),900):
                chunk=indices[offset:offset+900]
                task_map=bundle/'submission_history'/f'{stage}_{uuid.uuid4()}.json'
                if not dry_run:save(task_map,chunk)
                dependencies=[]
                if parent_jobs:dependencies.append('afterok:'+':'.join(parent_jobs))
                if last_group:dependencies.append('afterany:'+last_group)
                args=[f'--partition={partition}','--gres=gpu:1','--cpus-per-task=4',f'--mem={memory}G',
                      '--time='+walltime,'--array=0-'+str(len(chunk)-1)+'%'+str(max_parallel),
                      '--job-name=ejor_'+stage,'--output='+str(bundle/'logs'/f'{stage}_%A_%a.out'),
                      '--export='+env+',EJOR_TASK_MAP='+str(task_map)]
                if dependencies:args.append('--dependency='+','.join(dependencies))
                last_group=sbatch(args+[str(bundle/'driver/run_array.slurm')]);gpu_jobs.append(last_group)
        dependencies=['afterany:'+':'.join(gpu_jobs)] if gpu_jobs else []
        if parent_jobs:dependencies.append('afterok:'+':'.join(parent_jobs))
        args=['--cpus-per-task=1','--mem=2G','--time=00:30:00','--job-name=ejor_collect_'+stage,
              '--output='+str(bundle/'logs'/f'collect_{stage}_%j.out'),'--export='+env]
        if control_partition:args.append('--partition='+control_partition)
        if dependencies:args.append('--dependency='+','.join(dependencies))
        collectors[stage]=sbatch(args+[str(bundle/'driver/run_collect.slurm')])
    return submissions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    prepare.add_argument('--ca-summary',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    prepare.add_argument('--datasets',nargs='+');prepare.add_argument('--tuning-datasets',nargs='+')
    prepare.add_argument('--time-limit',type=float,default=3600)
    prepare.add_argument('--smoke',action='store_true')
    prepare.add_argument('--plan',choices=('v3','v4'),default='v4')
    prepare.add_argument('--shapes',nargs='+',choices=('cube','cylinder'),default=('cube','cylinder'))
    prepare.add_argument('--ca-json-dir',type=Path,help='Reuse validated historical CA placements; no calibration GPU tasks')
    prepare.add_argument('--reuse-bundle',type=Path,help='Import validated calibration and matching sensitivity results; never rerun completed matching cases')
    for name in ('collect','run','submit-all','unlock','verify-smoke'):
        p=sub.add_parser(name);p.add_argument('--bundle',type=Path,required=True)
        if name in ('collect','run'):p.add_argument('--stage',choices=STAGES+('memory',) if name=='run' else STAGES,required=True)
        if name=='run':
            p.add_argument('--index',type=int,required=True)
            p.add_argument('--task-map',type=Path)
        if name=='submit-all':
            p.add_argument('--max-parallel',type=int,default=6)
            p.add_argument('--small-memory',type=int,default=16);p.add_argument('--large-memory',type=int,default=32)
            p.add_argument('--walltime',default='12:00:00');p.add_argument('--dry-run',action='store_true')
            p.add_argument('--pending',action='store_true');p.add_argument('--through',choices=STAGES,default='full')
            p.add_argument('--partition',default='gpu');p.add_argument('--control-partition')
            p.add_argument('--flow',choices=('all','sensitivity','heuristics'),default='all')
    args=parser.parse_args()
    if args.command=='prepare':
        freeze(args.project,args.ca_summary,args.output,args.datasets,args.tuning_datasets,args.time_limit,args.smoke,
               args.shapes,args.ca_json_dir,args.reuse_bundle,args.plan)
        return 0
    if args.command=='run':
        index=json.loads(args.task_map.read_text())[args.index] if args.task_map else args.index
        return run(args.bundle,args.stage,index)
    if args.command=='collect':return collect(args.bundle,args.stage)
    if args.command=='verify-smoke':
        profile=json.loads((args.bundle/'manifest.json').read_text())['profile']
        from protocol_v4 import VERSION as V4_VERSION
        if profile['protocol']==V4_VERSION:
            from verify_v4 import verify_bundle as verify_smoke
        else:
            from smoke import verify_smoke
        verification=verify_smoke(args.bundle)
        save(args.bundle/'smoke_verification.json',verification)
        print(json.dumps(verification,indent=2))
        return 0 if verification['status']=='PASS' else 1
    if args.command=='submit-all':
        submit(args.bundle,args.max_parallel,args.small_memory,args.large_memory,args.walltime,
               args.dry_run,args.pending,args.through,args.partition,args.control_partition,args.flow)
        return 0
    # An explicit unlock is allowed only after Slurm confirms the bundle is idle.
    from submission_graph import active_job_ids
    jobs=active_job_ids(os.environ['USER'])
    for job in set(jobs):
        details=subprocess.check_output(['scontrol','show','job',job],text=True)
        if 'StdOut='+str(args.bundle.resolve()/'logs') in details:
            raise RuntimeError('Bundle still has active jobs: '+job)
    for lock in (args.bundle/'results').glob('*/task_*.lock'):lock.unlink()
    return 0


if __name__=='__main__':
    sys.exit(main())
