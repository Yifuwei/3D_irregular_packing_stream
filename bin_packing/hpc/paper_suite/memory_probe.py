"""Prepare three independent 3600s resource probes using the current worker."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import contextlib
import io
from suite import freeze,save
from protocol import VERSION
from worker import valid_saved


def prepare(project,summary,output,time_limit=3600):
    with contextlib.redirect_stdout(io.StringIO()):
        m=freeze(project,summary,output,['st04_Example5_normal'],['st04_Example5_normal'],time_limit,
                 True,('cylinder',))
    template=m['profile']['plans']['tuning'][0]
    chosen={'G_NI100','I_K100','GI_B300_K100'}
    cases=[]
    for case in m['profile']['plans']['tuning']:
        if case['config_id'] in chosen and case['shape']=='cylinder' and case['seed']==3:
            cases.append(dict(case,task_id=len(cases),stage='memory',protocol=VERSION+'-memory-only'))
    if len(cases)!=3:raise ValueError('Memory probe must contain exactly three algorithms')
    m['profile']['plans']={'memory':cases}
    m['profile']['protocol']=VERSION+'-memory-only'
    m['profile']['resource_probe']=dict(tolerance=.15,constraint_mode='geometry_only',
        note='Resource-only configurations, not selected performance optima; no discarded CA warmup',
        dataset=template['dataset'],shape='cylinder',seed=3,time_limit=time_limit)
    save(output/'memory_task_map.json',list(range(len(cases))))
    m['files_sha256']['memory_task_map.json']=hashlib.sha256((output/'memory_task_map.json').read_bytes()).hexdigest()
    save(output/'manifest.json',m)
    print('FINAL memory probe plan: 3 GPU tasks, 3600s each by default; no calibration/RCL/tuning submission')
    return m


def check(bundle):
    m=json.loads((bundle/'manifest.json').read_text())
    rows=[]
    for case in m['profile']['plans']['memory']:
        path=bundle/'results/memory'/f"task_{case['task_id']}.json"
        report=json.loads(path.read_text()) if path.exists() else {}
        peak=report.get('host_max_rss_kib')
        rows.append(dict(algorithm=case['algorithm'],status=report.get('status','missing'),
            valid=valid_saved(path,case,m['bundle_id']),max_rss_GiB=peak/1024**2 if peak else None,
            algorithm_wall_s=report.get('algorithm_wall_s'),counts=report.get('counts'),error=report.get('error')))
    result=dict(scope='Observed Python process RSS; check Slurm sacct for OOM/job-step peak',tasks=rows)
    save(bundle/'memory_summary.json',result)
    print(json.dumps(result,indent=2))
    return 0 if all(r['valid'] for r in rows) else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--project',type=Path,required=True)
    p.add_argument('--ca-summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--time-limit',type=float,default=3600)
    p=sub.add_parser('check');p.add_argument('--bundle',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.project,a.ca_summary,a.output,a.time_limit)
    else:sys.exit(check(a.bundle))
