"""Portable Slurm array worker and result validator; stdlib-only dry runs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SMALL_DATASETS = frozenset(('chess', 'shapesnew'))


def memory_gb(case):
    """Conservative dataset policy, not a measured peak-RSS guarantee."""
    return 16 if case['dataset'] in SMALL_DATASETS else 32


def load(bundle):
    return json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))


def label(case):
    return f"{case['dataset']}_{case['shape']}_seed{case['order_seed']}_t{round(case['tolerance']*100)}"


def verify_bundle(bundle, manifest):
    for rel, expected in manifest['files_sha256'].items():
        p = (bundle/rel).resolve()
        if not p.is_relative_to(bundle.resolve()):
            raise ValueError('Manifest path escapes bundle: '+rel)
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Bundle file changed: '+rel)


def validate_result(result, case):
    if result.get('status') != 'success':
        return False
    required = dict(dataset=case['dataset'],shape=case['shape'],order_seed=case['order_seed'],
                    tolerance=case['tolerance'],piece_seed=101,position_seed=202,
                    target_ls_evaluations=50,wall_time_limit=None,restart_limit=1,
                    no_improvement_limit=50,selection_range='bottom_top',
                    protocol='one_CA_plus_50_LS',algorithm='improved_GRASP',
                    pes='maximal_residual_box' if case['shape']=='cube' else 'minimum_aabb_edges_len')
    if any(k not in result or result[k] != v for k,v in required.items()):
        return False
    if result.get('validation_errors') != [] or result.get('ca_validation_errors') != []:
        return False
    if result.get('constructions') != 1:
        return False
    calls = result.get('ls_calls',-1)
    if calls == 50:
        if result.get('stop_reason') != '50_evaluations_completed':return False
    elif not (isinstance(calls,int) and 0 <= calls < 50 and result.get('best_N')==1
              and result.get('stop_reason')=='single_bin_reached'):
        return False
    try:
        if not all(math.isfinite(result[k]) for k in ['F','best_N','best_U_star']):return False
        if abs(result['F']-(result['best_N']-result['best_U_star'])) > 1e-10:return False
        if len([x for x in result['history'] if x['kind']=='ls']) != calls:return False
    except (KeyError,TypeError):return False
    return True


def existing_success(path, case):
    try:
        result=json.loads(path.read_text(encoding='utf-8'))
        if not validate_result(result,case):return False
        with path.with_suffix('.rcl.csv').open(newline='',encoding='utf-8') as f:
            hist=Counter(int(x['rcl_size']) for x in csv.DictReader(f))
        if dict(hist)!={int(k):v for k,v in result['rcl_histogram'].items()}:return False
        if sum(hist.values())!=result['rcl_calls']:return False
        with path.with_suffix('.ls.csv').open(newline='',encoding='utf-8') as f:
            ls=[x for x in csv.DictReader(f) if x['kind']=='ls']
        return len(ls)==result['ls_calls']
    except (OSError,ValueError,TypeError,KeyError):
        return False


def array_spec(bundle, results, pending, memory=None):
    manifest=load(bundle)
    ids=[str(i) for i,c in enumerate(manifest['plan'])
         if (memory is None or memory_gb(c)==memory)
         and (not pending or not existing_success(results/(label(c)+'.json'),c))]
    if not ids:return ''
    return f'0-{len(ids)-1}' if not pending and memory is None else ','.join(ids)


def run_task(bundle, results, index, dry_run=False):
    manifest=load(bundle)
    if index < 0 or index >= len(manifest['plan']):raise ValueError('Array index out of range')
    verify_bundle(bundle,manifest)
    case=manifest['plan'][index]
    result=results/(label(case)+'.json')
    worker=bundle/'runtime/bin_packing/test/run_ca50_tolerance_experiment.py'
    command=[sys.executable,'-u',str(worker),'--worker','--dataset',case['dataset'],
             '--shape',case['shape'],'--seed',str(case['order_seed']),
             '--tolerance',str(case['tolerance']),'--output',str(result)]
    if dry_run:
        print(json.dumps(dict(index=index,case=case,memory_gb=memory_gb(case),command=command,already_success=existing_success(result,case)),indent=2))
        return 0
    results.mkdir(parents=True,exist_ok=True)
    locks=results/'locks';locks.mkdir(exist_ok=True)
    lock=locks/(label(case)+'.lock')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(dict(pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
                           array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID')),f)
        if existing_success(result,case):
            print('Already successful; skipping:',label(case),flush=True)
            return 0
        # Preserve prior failed/mismatched attempts instead of overwriting evidence.
        suffixes=('.json','.log','.status.json','.rcl.csv','.ls.csv')
        if any((results/(label(case)+suffix)).exists() for suffix in suffixes):
            attempts=results/'previous_attempts'/f'{label(case)}_{time.time_ns()}'
            attempts.mkdir(parents=True)
            for suffix in suffixes:
                p=results/(label(case)+suffix)
                if p.exists():p.rename(attempts/p.name)
        log=results/(label(case)+'.log')
        print(f'Array index {index}: {label(case)}',flush=True)
        with log.open('w',encoding='utf-8') as f:
            completed=subprocess.run(command,cwd=worker.parent,stdout=f,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if completed.returncode:
            print('Worker exited:',completed.returncode,'log:',log,flush=True)
            return completed.returncode if completed.returncode>0 else 1
        if not existing_success(result,case):
            print('Worker result failed configuration/work/geometry validation:',result,flush=True)
            return 1
        print('Success:',result,flush=True)
        return 0
    finally:
        lock.unlink()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--results',type=Path)
    parser.add_argument('--index',type=int)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--array-spec',action='store_true')
    parser.add_argument('--pending',action='store_true')
    parser.add_argument('--memory-gb',type=int,choices=(16,32))
    a=parser.parse_args();bundle=a.bundle.resolve()
    results=a.results.resolve() if a.results else bundle/'results'
    if a.array_spec:
        print(array_spec(bundle,results,a.pending,a.memory_gb));return
    if a.index is None:parser.error('--index is required for a worker')
    raise SystemExit(run_task(bundle,results,a.index,a.dry_run))


if __name__=='__main__':main()
