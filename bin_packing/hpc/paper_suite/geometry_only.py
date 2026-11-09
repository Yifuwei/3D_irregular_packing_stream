"""Freeze a geometry-only final comparison using the selected constrained parameters."""
import argparse
import hashlib
import json
from pathlib import Path
from suite import freeze, save

def prepare(source, project, output):
    source=source.resolve()
    prior=json.loads((source/'manifest.json').read_text())
    selections={}
    for stage in ('rcl','tuning'):
        path=source/'results'/stage/'selection.json'
        selections[stage]=json.loads(path.read_text())
    if selections['rcl']['global_rcl']['top_k'] != 10:
        raise ValueError('Expected the confirmed Top10 RCL')
    if selections['tuning']['improved_ILS']['kick_fraction'] != .5:
        raise ValueError('Expected the confirmed 50% kick strength')
    freeze(project,source/'inputs/ca_reference_summary.csv',output,
           datasets=prior['profile']['datasets'],
           tuning=prior['profile']['tuning_datasets'],reuse_bundle=source,plan='v4')
    manifest=json.loads((output/'manifest.json').read_text())
    profile=manifest['profile']
    profile['constraint_mode']='geometry_only'
    profile['bin_priority']='first_fit'
    # One run per dataset, shape, order seed and algorithm.
    cases=[dict(c) for c in profile['plans']['full']
           if c['weight_level']=='weak' and c['radio_level']=='weak']
    for index,case in enumerate(cases):
        case.update(task_id=index,weight_level='none',radio_level='none',
                    constraint_mode='geometry_only',
                    config_id=case['algorithm']+'_geometry_only')
    profile['plans']['full']=cases
    profile['plans']['rcl']=[]
    profile['plans']['tuning']=[]
    profile['submission_policy']='full only; selected RCL/kick inherited from source'
    profile['parameter_source']=dict(bundle_id=prior['bundle_id'],bundle=str(source))
    for stage,value in selections.items():
        path=output/'results'/stage/'selection.json'
        save(path,value)
        relative=path.relative_to(output).as_posix()
        manifest['files_sha256'][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(output/'manifest.json',manifest)
    plan=json.loads((output/'parameter_plan.json').read_text())
    plan['profile']=profile
    plan['conventions'].update(metaheuristic_packing_scope='fixed per shape',
                               bin_priority='first_fit',constraints='geometry_only',
                               parameter_selection='inherit Top10 and 50% from constrained experiments')
    save(output/'parameter_plan.json',plan)
    print(json.dumps(dict(bundle=str(output.resolve()),full_tasks=len(cases),
                          constraint_mode='geometry_only',bin_priority='first_fit'),indent=2))
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.source,args.project,args.output)
