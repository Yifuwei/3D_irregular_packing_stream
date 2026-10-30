"""Verify the new complete experiment graph and recompute selected parameters."""
import json
from collections import defaultdict
from protocol import STAGES,select_tolerances,select_tuning
from protocol_v4 import select_sensitivity,VERSION
from worker import valid_saved,packing_settings,read_selections,settings_for
from bundle_integrity import verified_bytes


def verify_bundle(bundle):
    manifest=json.loads((bundle/'manifest.json').read_text());profile=manifest['profile'];problems=[];records={}
    if profile['protocol']!=VERSION:problems.append('Wrong protocol')
    for name in manifest['files_sha256']:
        try:verified_bytes(bundle,manifest,name)
        except (OSError,ValueError):problems.append('Frozen input/code changed: '+name)
    counts={}
    for stage in STAGES:
        records[stage]=[]
        for c in profile['plans'][stage]:
            path=bundle/'results'/stage/f"task_{c['task_id']}.json"
            if not valid_saved(path,c,manifest['bundle_id']):problems.append(f'Missing/failed {stage}/{c["task_id"]}');continue
            report=json.loads(path.read_text());records[stage].append(dict(report,**c))
        counts[stage]=dict(expected=len(profile['plans'][stage]),success=len(records[stage]))
    if problems:return dict(status='FAIL',counts=counts,problems=problems)
    try:
        expected=dict(sensitivity=select_sensitivity(records['sensitivity'],profile),
                      rcl=select_tolerances(records['rcl']),tuning=select_tuning(records['tuning']))
        for stage,value in expected.items():
            actual=json.loads((bundle/'results'/stage/'selection.json').read_text())
            if actual!=value:problems.append('Selected parameters disagree with complete reports: '+stage)
        for stage in ('sensitivity','rcl','tuning','full'):
            for r in records[stage]:
                c=r['case'];selections=read_selections(bundle,c);cfg=packing_settings(c,manifest,selections)
                if any(r.get('method',{}).get(k)!=cfg[k] for k in ('pes','selection_range')):problems.append('Wrong selected PES/range')
                algorithm,search=settings_for(c,profile,selections)
                if search.get('time_limit')==float('inf'):search['time_limit']=None
                if r.get('search_settings')!=search or r.get('algorithm')!=algorithm:problems.append('Wrong algorithm parameters')
                fixed=selections['calibration'][f"{c['dataset']}|{c['shape']}"]
                if r.get('weight_capacity')!=fixed['weight_capacity'][c['weight_level']] or r.get('radio_capacity')!=fixed['radio_capacity'][c['radio_level']]:
                    problems.append('Wrong fixed capacity')
    except (KeyError,ValueError,OSError) as exc:problems.append(str(exc))
    return dict(status='FAIL' if problems else 'PASS',counts=counts,problems=problems,
                scope='frozen files, complete cases, shared parameter selection and runtime handoff')
