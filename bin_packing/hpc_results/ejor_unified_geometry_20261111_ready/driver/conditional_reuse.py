"""Reuse validated Cylinder results only when effective runtime parameters match."""
import hashlib
import json
from pathlib import Path
from bundle_integrity import verified_bytes, normalized
from worker import valid_saved, packing_settings, read_selections, settings_for

def identity(case):
    return tuple(case.get(k) for k in
                 ('dataset','shape','seed','weight_level','radio_level','algorithm','config_id'))

def clean_report(report):
    report=dict(report)
    report.pop('history',None)
    report.pop('ls_end_events',None)
    return report

def stage_candidates(bundle, source, stages=('rcl','tuning','full')):
    bundle,source=Path(bundle),Path(source)
    manifest=json.loads((bundle/'manifest.json').read_text())
    previous=json.loads((source/'manifest.json').read_text())
    staged={}
    checked=set()
    for stage in stages:
        targets={identity(c):c for c in manifest['profile']['plans'][stage] if c['shape']=='cylinder'}
        count=0
        for old in previous['profile']['plans'][stage]:
            case=targets.get(identity(old))
            if case is None:continue
            path=source/'results'/stage/f"task_{old['task_id']}.json"
            if not valid_saved(path,old,previous['bundle_id']):continue
            report=json.loads(path.read_text())
            cfg=packing_settings(case,manifest)
            if any(report['method'].get(k)!=cfg[k] for k in ('pes','selection_range')):continue
            geometry_only=manifest['profile'].get('constraint_mode')=='geometry_only'
            if report['method'].get('radio_priority') != (not geometry_only):continue
            names=['runtime/bin_packing/instances/'+case['instance']]
            names+=manifest['instance_geometry'][case['instance']]
            # Explicit PES settings may change defaults; executable search logic must match.
            names += [name for name in manifest['files_sha256']
                      if name.startswith('runtime/bin_packing/src/') and not name.endswith('/ca_parameters.py')]
            for name in names:
                if name in checked:continue
                if verified_bytes(bundle,manifest,name)!=verified_bytes(source,previous,name):
                    raise ValueError('Reuse source input or algorithm differs: '+name)
                checked.add(name)
            target=bundle/'inputs/conditional_reuse'/stage/f"task_{case['task_id']}.json"
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(dict(source_bundle_id=previous['bundle_id'],
                                              source_case=old,report=clean_report(report))),encoding='utf-8')
            relative=target.relative_to(bundle).as_posix()
            manifest['files_sha256'][relative]=hashlib.sha256(normalized(target.read_bytes(),target)).hexdigest()
            count+=1
        staged[stage]=count
    from suite import save
    manifest['conditional_reuse']=staged
    save(bundle/'manifest.json',manifest)
    return staged

def try_reuse(bundle,case,manifest):
    relative=f"inputs/conditional_reuse/{case['stage']}/task_{case['task_id']}.json"
    if relative not in manifest['files_sha256']:return False
    candidate=json.loads(verified_bytes(bundle,manifest,relative))
    report=candidate['report']
    cfg=packing_settings(case,manifest)
    if any(report['method'].get(k)!=cfg[k] for k in ('pes','selection_range')):return False
    expected_priority=manifest['profile'].get('constraint_mode')!='geometry_only'
    if report['method'].get('radio_priority')!=expected_priority:return False
    try:
        selections=read_selections(bundle,case)
        algorithm,settings=settings_for(case,manifest['profile'],selections)
    except FileNotFoundError:
        return False  # Full-stage parameter selection is not available yet.
    if settings.get('time_limit')==float('inf'):settings['time_limit']=None
    if report.get('algorithm')!=algorithm or report.get('search_settings')!=settings:return False
    if manifest['profile'].get('constraint_mode')!='geometry_only':
        cap=selections['calibration'][case['dataset']+'|'+case['shape']]
        for field,level in (('weight_capacity','weight_level'),('radio_capacity','radio_level')):
            if abs(report[field]-cap[field][case[level]])>1e-8:return False
    from suite import save
    report.update(case=case,bundle_id=manifest['bundle_id'],
                  reused_from=dict(bundle_id=candidate['source_bundle_id'],
                                   case=candidate['source_case'],source_report=relative,
                                   validation='Input, executable algorithm, PES, limits and effective search parameters matched'))
    save(bundle/'results'/case['stage']/f"task_{case['task_id']}.json",report)
    return True
