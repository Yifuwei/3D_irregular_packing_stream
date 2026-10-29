"""Import completed CA stages into a new runtime without rerunning experiments."""
import hashlib
import json
import math
from pathlib import Path
import shutil
from collections import defaultdict

from protocol import calibrate
from worker import valid_saved, settings_for


def _identity(case):
    return tuple(case.get(k) for k in ('stage','dataset','shape','seed','weight_level','radio_level','config_id'))


def _file(root,name):
    path=(root/name).resolve()
    path.relative_to(root.resolve())
    return path


def _same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(_same(a[k],b[k]) for k in a)
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
    return a==b


def _verified_bytes(root,manifest,name,text=False):
    data=_file(root,name).read_bytes()
    expected=manifest['files_sha256'].get(name)
    if hashlib.sha256(data).hexdigest()==expected:
        return data.replace(b'\r\n',b'\n') if text else data
    if text and hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==expected:
        return data.replace(b'\r\n',b'\n')
    raise ValueError('Frozen input changed: '+name)


def import_completed(bundle,source):
    """Validate all matching cases before saving; preserve original provenance."""
    bundle,source=Path(bundle).resolve(),Path(source).resolve()
    if bundle==source:raise ValueError('Reuse requires a new bundle; source is read-only')
    manifest=json.loads((bundle/'manifest.json').read_text())
    old=json.loads((source/'manifest.json').read_text())
    capacities=json.loads((source/'results/calibration/selection.json').read_text())
    pending=[];groups=defaultdict(list);checked=set()
    for stage in ('calibration','sensitivity'):
        indexed={_identity(c):c for c in old['profile']['plans'][stage]}
        for case in manifest['profile']['plans'][stage]:
            previous=indexed.get(_identity(case))
            if previous is None:raise ValueError('Missing source case: '+str(_identity(case)))
            path=source/'results'/stage/f"task_{previous['task_id']}.json"
            if not valid_saved(path,previous,old['bundle_id']):raise ValueError('Invalid source report: '+str(path))
            report=json.loads(path.read_text())
            key=f"{case['dataset']}|{case['shape']}"
            for field in ('pes','selection_range'):
                if manifest['ca_parameters'][key][field]!=old['ca_parameters'][key][field]:
                    raise ValueError('CA parameters changed: '+key+'/'+field)
            name='runtime/bin_packing/instances/'+case['instance']
            if name not in checked:
                if _verified_bytes(bundle,manifest,name,text=True)!=_verified_bytes(source,old,name,text=True):
                    raise ValueError('Instance changed: '+name)
                checked.add(name)
            for geometry in manifest['instance_geometry'][case['instance']]:
                if geometry not in checked:
                    if _verified_bytes(bundle,manifest,geometry)!=_verified_bytes(source,old,geometry):
                        raise ValueError('Geometry changed: '+geometry)
                    checked.add(geometry)
            counts=report.get('counts',{})
            if counts.get('construction')!=1 or counts.get('ls',0) or counts.get('kick',0):
                raise ValueError('Source is not a single CA construction')
            if stage=='sensitivity':
                for field,level in (('weight_capacity','weight_level'),('radio_capacity','radio_level')):
                    if not _same(report[field],capacities[key][field][case[level]]):
                        raise ValueError('Sensitivity capacity mismatch: '+key+'/'+field)
                algorithm,settings=settings_for(case,manifest['profile'],{})
                if report['algorithm']!=algorithm or report['search_settings']!=settings:
                    raise ValueError('Sensitivity CA settings changed')
            else:groups[key].append(report)
            pending.append((stage,case,path,report))
    for key,records in groups.items():
        if not _same(calibrate(records),capacities[key]):
            raise ValueError('Reference seeds do not reproduce original capacities: '+key)
    from suite import save
    totals=dict(calibration=0,sensitivity=0)
    for stage,case,path,report in pending:
        relative=f"inputs/reused/{stage}/task_{case['task_id']}.json"
        dest=bundle/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        manifest['files_sha256'][relative]=digest
        raw=report.get('source_reference') if stage=='calibration' else None
        if raw:
            raw_relative=f"inputs/reused/reference_{case['task_id']}.json"
            raw_source=_file(source,raw)
            _verified_bytes(source,old,raw,text=True)
            shutil.copyfile(raw_source,bundle/raw_relative)
            manifest['files_sha256'][raw_relative]=hashlib.sha256((bundle/raw_relative).read_bytes()).hexdigest()
            report['source_reference']=raw_relative
        report.update(case=case,bundle_id=manifest['bundle_id'],
            reused_from=dict(bundle_id=old['bundle_id'],case=json.loads(path.read_text())['case'],
                             source_report=relative,source_sha256=digest))
        save(bundle/'results'/stage/f"task_{case['task_id']}.json",report)
        totals[stage]+=1
    manifest['completed_stage_source']=dict(bundle=str(source),bundle_id=old['bundle_id'],counts=totals,
        note='Original five-seed capacities; experimental seeds filtered by new plan; no new CA execution; text LF/CRLF equivalence allowed only after verifying historical normalized hash')
    save(bundle/'manifest.json',manifest)
    return totals
