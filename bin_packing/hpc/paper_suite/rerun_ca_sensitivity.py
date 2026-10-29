"""Prepare an existing bundle for fixed-by-shape sensitivity and global tuning."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from protocol import make_plan,choose_ca_parameters,kick_tuning_datasets
from worker import valid_saved,packing_settings
from suite import collect,save


def ensure_idle(bundle):
    if shutil.which('squeue') and os.environ.get('USER'):
        jobs=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%A'],text=True).split()
        for job in set(jobs):
            details=subprocess.check_output(['scontrol','show','job',job],text=True)
            if 'StdOut='+str(bundle/'logs') in details:
                raise RuntimeError('Stop old jobs for this bundle before updating: '+job)
    if list((bundle/'results').glob('*/task_*.lock')):
        raise RuntimeError('Task locks remain; verify jobs stopped before using the suite unlock command')


def prepare(bundle):
    bundle=Path(bundle).resolve()
    ensure_idle(bundle)
    old=json.loads((bundle/'manifest.json').read_text())
    stamp='before_global_parameters_'+uuid.uuid4().hex[:12]
    backup=bundle/'configuration_history'/stamp
    backup.mkdir(parents=True)
    for name in ('manifest.json','parameter_plan.json'):
        shutil.copyfile(bundle/name,backup/name)
    before=old['profile']
    profile=make_plan(before['datasets'],kick_tuning_datasets(before['datasets']),seeds=tuple(before['seeds']),
        reference_seeds=tuple(before['reference_seeds']),full_seeds=tuple(before['full_seeds']),
        time_limit=before['time_limit'],smoke=before.get('smoke',False),shapes=tuple(before['shapes']))
    meta={c['instance']:c for cases in before['plans'].values() for c in cases}
    for cases in profile['plans'].values():
        for c in cases:
            c['instance']=f"{c['dataset']}_seq{c['seed']}_{c['shape']}_999999_1.txt"
            c['piece_count']=meta[c['instance']]['piece_count']
    with (bundle/'inputs/ca_reference_summary.csv').open(encoding='utf-8-sig',newline='') as f:
        geometry=choose_ca_parameters(list(csv.DictReader(f)),profile['datasets'])
    geometry={k:v for k,v in geometry.items() if k.split('|')[1] in profile['shapes']}
    manifest=dict(old,profile=profile,geometry_ca_parameters=geometry)
    capacities=json.loads((bundle/'results/calibration/selection.json').read_text())
    kept={};archived={}
    def archive(path):
        if not path.exists():return
        relative=path.relative_to(bundle)
        target=bundle/'results/archive'/stamp/relative
        path.resolve().relative_to(bundle)
        target.resolve().relative_to(bundle)
        target.parent.mkdir(parents=True,exist_ok=True)
        path.replace(target)
    for stage in ('sensitivity','rcl','tuning','full'):
        kept[stage]=archived[stage]=0
        wanted={c['task_id']:c for c in profile['plans'][stage]}
        folder=bundle/'results'/stage
        for path in list(folder.glob('task_*.json')):
            if not path.stem[5:].isdigit():continue
            case=wanted.get(int(path.stem[5:]))
            try:report=json.loads(path.read_text())
            except (ValueError,OSError):report={}
            compatible=False
            if case and report.get('status')=='success' and report.get('validation_ok') is True:
                cfg=packing_settings(case,manifest)
                compatible=all(report.get('method',{}).get(k)==cfg[k] for k in ('pes','selection_range'))
                previous=report.get('case',{})
                compatible=compatible and all(previous.get(k)==case.get(k) for k in
                    ('dataset','shape','seed','weight_level','radio_level','algorithm','config_id'))
                if stage=='sensitivity':
                    limits=capacities[f"{case['dataset']}|{case['shape']}"]
                    from reuse_completed import _same
                    compatible=compatible and all(_same(report.get(field),limits[field][case[level]])
                        for field,level in (('weight_capacity','weight_level'),('radio_capacity','radio_level')))
                    counts=report.get('counts',{})
                    compatible=compatible and counts.get('construction')==1 and not counts.get('ls',0) and not counts.get('kick',0)
                else:
                    compatible=compatible and valid_saved(path,case,manifest['bundle_id'])
            if compatible:
                if report['case']!=case:
                    target=backup/'accepted_reports'/stage/path.name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(path,target)
                    report['parameter_revalidation']=dict(previous_case=report['case'],scope='global_by_shape',
                        note='Actual recorded PES/range and fixed capacities verified; no rerun')
                    report['case']=case;save(path,report)
                kept[stage]+=1
            else:
                archive(path)
                archive(path.with_name(path.stem+'_events.jsonl'))
                archived[stage]+=1
        if before['plans'][stage]!=profile['plans'][stage] or archived[stage]:
            for name in ('completion.json','runs.csv','summary.csv','selection.json'):
                archive(folder/name)
    # Preserve historical calibration and original fixed capacities without recomputation from new CA choices.
    parameter=json.loads((bundle/'parameter_plan.json').read_text())
    parameter.update(profile=profile,geometry_ca_parameters=geometry)
    parameter['conventions'].update(parameter_selection='RCL aggregates 15 datasets; kick strength aggregates the confirmed three representatives; never per-dataset winners',
        sensitivity_packing='fixed global-by-shape PES/range; original capacity calibration retained')
    save(bundle/'parameter_plan.json',parameter)
    manifest['files_sha256']['parameter_plan.json']=hashlib.sha256((bundle/'parameter_plan.json').read_bytes()).hexdigest()
    driver=Path(__file__).resolve().parent
    for path in driver.iterdir():
        if path.suffix in ('.py','.sh','.slurm','.md'):
            dest=bundle/'driver'/path.name
            if path.resolve()!=dest.resolve():shutil.copyfile(path,dest)
            manifest['files_sha256']['driver/'+path.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    for name in ('WORK_PLAN.md',):
        if (driver/name).exists():shutil.copyfile(driver/name,bundle/name)
    save(bundle/'manifest.json',manifest)
    collect(bundle,'sensitivity')
    report=dict(status='PREPARED',kept=kept,archived=archived,
        sensitivity_pending=len(profile['plans']['sensitivity'])-kept['sensitivity'],
        tuning_tasks=len(profile['plans']['tuning']),selection_datasets=profile['datasets'],
        selection_dataset_count=len(profile['datasets']),kick_selection_datasets=profile['tuning_datasets'],
        kick_selection_dataset_count=len(profile['tuning_datasets']),parameters=profile['metaheuristic_parameters'],
        calibration_preserved=True,configuration_backup=str(backup),submitted=False)
    save(bundle/'sensitivity_rerun_plan.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.bundle),indent=2))
