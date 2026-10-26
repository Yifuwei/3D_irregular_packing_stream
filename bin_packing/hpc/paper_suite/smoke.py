"""Verify actual smoke reports and parameter handoffs without GPU imports."""
import hashlib
import json
import math

from protocol import calibrate,select_tolerances,select_tuning
from worker import valid_saved,read_selections,settings_for


def verify_smoke(bundle):
    manifest=json.loads((bundle/'manifest.json').read_text())
    profile=manifest['profile']
    problems=[]
    counts={}
    records={}
    if profile.get('smoke') is not True:problems.append('Not a smoke bundle')
    for name,digest in manifest['files_sha256'].items():
        path=bundle/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            problems.append('Frozen file missing/changed: '+name)
    for stage,cases in profile['plans'].items():
        records[stage]=[]
        for case in cases:
            path=bundle/'results'/stage/f"task_{case['task_id']}.json"
            if not valid_saved(path,case,manifest['bundle_id']):
                problems.append(f"Missing/failed result: {stage}/{case['task_id']}")
                continue
            records[stage].append(dict(json.loads(path.read_text()),**case))
        counts[stage]=dict(expected=len(cases),success=len(records[stage]))
    # Do not accept parameters derived from partial stages.
    if problems:return dict(status='FAIL',bundle_id=manifest['bundle_id'],counts=counts,problems=problems)
    try:
        for stage in profile['plans']:
            root=bundle/'results'/stage
            completion=json.loads((root/'completion.json').read_text())
            if completion!=dict(expected=counts[stage]['expected'],success=counts[stage]['success'],incomplete=[]):
                problems.append('Invalid completion metadata: '+stage)
            if not (root/'runs.csv').is_file():problems.append('Missing runs.csv: '+stage)
        calibration={}
        for case in records['calibration']:
            calibration.setdefault(f"{case['dataset']}|{case['shape']}",[]).append(case)
        expected={
            'calibration':{key:calibrate(values) for key,values in calibration.items()},
            'rcl':select_tolerances(records['rcl']),
            'tuning':select_tuning(records['tuning'])}
        for stage,selection in expected.items():
            saved=json.loads((bundle/'results'/stage/'selection.json').read_text())
            if saved!=selection:problems.append('Selected parameters disagree with reports: '+stage)
        for stage,group in records.items():
            for report in group:
                case=report['case']
                name=f"{stage}/{case['task_id']}"
                if stage=='calibration' and report.get('calibration_source')=='historical_CA':
                    from ca_reference import recover
                    source=report['source_reference']
                    if source not in manifest['files_sha256']:
                        problems.append('Historical source not frozen: '+name)
                        continue
                    recovered=recover(json.loads((bundle/source).read_text(encoding='utf-8-sig')),
                                      bundle/'runtime/bin_packing/instances'/case['instance'])
                    for key,value in recovered.items():
                        if report.get(key)!=value:problems.append('Historical statistics mismatch: '+name+'/'+key)
                    continue
                selections=read_selections(bundle,case)
                algorithm,settings=settings_for(case,profile,selections)
                if settings.get('time_limit') is not None and not math.isfinite(settings['time_limit']):
                    settings['time_limit']=None
                if report.get('search_settings')!=settings or report.get('algorithm')!=algorithm:
                    problems.append('Runtime parameters disagree with selected values: '+name)
                method=report.get('method',{})
                if method.get('radio_priority')!=(stage in ('sensitivity','rcl','tuning','full')) or method.get('warmup')!='none':
                    problems.append('Wrong packing mode/warmup: '+name)
                actual=report.get('counts',{})
                if stage in ('calibration','sensitivity') and (actual.get('construction')!=1 or actual.get('ls',0) or actual.get('kick',0)):
                    problems.append('Reference CA work count mismatch: '+name)
                if stage=='rcl':
                    ls=actual.get('ls',0)
                    if (actual.get('construction')!=1 or actual.get('kick',0) or ls!=0):
                        problems.append('RCL fixed-work count mismatch: '+name)
                if stage in ('sensitivity','rcl','tuning','full'):
                    limits=selections['calibration'][f"{case['dataset']}|{case['shape']}"]
                    for field,level in (('weight_capacity','weight_level'),('radio_capacity','radio_level')):
                        if not math.isclose(report[field],limits[field][case[level]],rel_tol=1e-12):
                            problems.append('Wrong calibrated capacity: '+name+'/'+field)
        if not (bundle/'results/full/summary.csv').is_file():problems.append('Missing final summary.csv')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        problems.append('Verification error: '+str(exc))
    return dict(status='FAIL' if problems else 'PASS',bundle_id=manifest['bundle_id'],
                counts=counts,problems=problems,
                checks=['frozen inputs/code','all task identities and validation flags','stage completeness',
                        'selected parameters recomputed','runtime parameter handoffs',
                        'packing modes and fixed work counts','full capacities','final CSV exists'])
