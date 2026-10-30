"""Reuse only identical historical constrained CA runs in the v4 fixed-PES grid."""
import hashlib
import json
from pathlib import Path
import shutil

from bundle_integrity import verified_bytes,canonicalize_manifest
from reuse_completed import _same
from worker import valid_saved


def key(case,pes,ran):
    return tuple(case.get(k) for k in ('dataset','shape','seed','weight_level','radio_level'))+(pes,ran)


def import_candidates(bundle,source):
    bundle,source=Path(bundle).resolve(),Path(source).resolve()
    m=json.loads((bundle/'manifest.json').read_text());old=json.loads((source/'manifest.json').read_text())
    capacities=json.loads((bundle/'results/calibration/selection.json').read_text())
    wanted={key(c,c['pes'],c['selection_range']):c for c in m['profile']['plans']['sensitivity']}
    pending=[];checked=set();seen=set()
    for previous in old['profile']['plans']['sensitivity']:
        if previous['seed'] not in m['profile']['full_seeds'] or previous['weight_level'] not in ('strong','weak') or previous['radio_level'] not in ('strong','weak'):continue
        path=source/'results/sensitivity'/f"task_{previous['task_id']}.json"
        if not valid_saved(path,previous,old['bundle_id']):continue
        r=json.loads(path.read_text());method=r.get('method',{})
        identity=key(previous,method.get('pes'),method.get('selection_range'))
        case=wanted.get(identity)
        if case is None:continue
        if identity in seen:raise ValueError('Duplicate historical CA candidate')
        seen.add(identity)
        counts=r.get('counts',{})
        if counts.get('construction')!=1 or counts.get('ls',0) or counts.get('kick',0):raise ValueError('Historical sensitivity is not one fixed CA construction')
        if r.get('search_settings')!=dict(ALG='fixed_CA',iteration_limit=0,time_limit=None):raise ValueError('Historical CA search settings differ')
        limits=capacities[f"{case['dataset']}|{case['shape']}"]
        for field,level in (('weight_capacity','weight_level'),('radio_capacity','radio_level')):
            if not _same(r[field],limits[field][case[level]]):raise ValueError('Historical sensitivity uses different fixed capacities')
        names=['runtime/bin_packing/instances/'+case['instance']]+m['instance_geometry'][case['instance']]
        for name in names:
            if name in checked:continue
            if verified_bytes(bundle,m,name)!=verified_bytes(source,old,name):raise ValueError('Historical input/geometry differs: '+name)
            checked.add(name)
        pending.append((case,path,r))
    from suite import save
    for case,path,r in pending:
        relative=f"inputs/reused_v4/sensitivity/task_{case['task_id']}.json"
        dest=bundle/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        digest=hashlib.sha256(dest.read_bytes()).hexdigest();m['files_sha256'][relative]=digest
        r['reused_from']=dict(bundle_id=old['bundle_id'],case=r['case'],source_report=relative,source_sha256=digest,
                             validation='Actual PES/range, capacities, geometry, sequence and CA-only settings match v4 candidate')
        r.update(case=case,bundle_id=m['bundle_id'])
        save(bundle/'results/sensitivity'/f"task_{case['task_id']}.json",r)
    m.setdefault('completed_stage_source',{}).setdefault('counts',{})['sensitivity']=len(pending)
    canonicalize_manifest(bundle,m);save(bundle/'manifest.json',m)
    return len(pending)
