"""Independent experiment branches with a shared, dynamically released GPU budget."""
import json
import subprocess
import time
from pathlib import Path

PARENTS={
    'calibration':(),
    'sensitivity':('calibration',),
    'rcl':('calibration',),
    'tuning':('calibration',),
    'full':('calibration','rcl','tuning'),
}


def initial_state(stages,pending,budget):
    roots=[s for s in ('sensitivity','rcl','tuning') if s in stages and pending[s]]
    finished={s:not bool(pending[s]) for s in stages}
    return dict(stages=list(stages),budget=budget,finished=finished,passed=dict(finished),
                serial_resources=budget<len(roots),arrays={s:[] for s in stages},quotas={})


def allocations(state):
    budget=state['budget'];stages=state['stages']
    if state['serial_resources']:return {s:budget for s in stages}
    roots=[s for s in ('sensitivity','rcl','tuning') if s in stages and not state['finished'][s]]
    root_share=max(1,budget//max(1,len(roots)))
    quota={s:(budget if s=='calibration' else max(1,budget//2) if s=='full' else root_share) for s in stages}
    active=[s for s in stages if not state['finished'][s] and
            all(state['passed'].get(p,False) for p in PARENTS[s])]
    if active:
        share,remainder=divmod(budget,len(active))
        for i,s in enumerate(active):quota[s]=share+(i<remainder)
    return quota


def array_active(job):
    result=subprocess.run(['squeue','-h','-j',job,'-o','%A'],capture_output=True,text=True)
    if result.returncode:
        if 'Invalid job id' in result.stderr:return False
        raise RuntimeError('Cannot verify array completion: '+result.stderr)
    return bool(result.stdout.strip())


def rebalance(bundle,stage=None,success=True):
    """Only completed GPU branches release slots; no live branch is reduced."""
    bundle=Path(bundle);path=bundle/'submission_history/concurrency.json'
    if not path.exists():return
    lock=path.with_suffix('.lock');deadline=time.monotonic()+30
    while True:
        try:lock.mkdir();break
        except FileExistsError:
            if time.monotonic()>deadline:raise RuntimeError('Concurrency state lock is busy: '+str(lock))
            time.sleep(.05)
    try:
        state=json.loads(path.read_text())
        if stage is not None:
            if stage not in state['stages']:return
            # Manual collection while arrays are running must not release slots.
            if any(array_active(job) for job in state['arrays'][stage]):return
            state['finished'][stage]=True;state['passed'][stage]=success
        else:
            # Catch a very short stage whose collector finished before submission ended.
            manifest=json.loads((bundle/'manifest.json').read_text())
            from worker import valid_saved
            for candidate in state['stages']:
                if state['finished'][candidate]:continue
                folder=bundle/'results'/candidate;comp=folder/'completion.json'
                if not comp.exists() or not (folder/'selection.json').exists():continue
                completion=json.loads(comp.read_text());cases=manifest['profile']['plans'][candidate]
                if completion.get('success')!=len(cases) or completion.get('incomplete'):continue
                if not all(valid_saved(folder/f"task_{c['task_id']}.json",c,manifest['bundle_id']) for c in cases):continue
                if any(array_active(job) for job in state['arrays'][candidate]):continue
                state['finished'][candidate]=state['passed'][candidate]=True
        quota=allocations(state)
        if not state['serial_resources']:
            for candidate in state['stages']:
                if state['finished'][candidate] or quota[candidate]==state['quotas'][candidate]:continue
                # Slot transfers occur only after a branch has zero running tasks.
                if quota[candidate]<state['quotas'][candidate] and any(array_active(j) for j in state['arrays'][candidate]):
                    raise RuntimeError('Refusing to lower a live branch while transferring GPU slots')
                for job in state['arrays'][candidate]:
                    if array_active(job):
                        subprocess.run(['scontrol','update',f'JobId={job}',f'ArrayTaskThrottle={quota[candidate]}'],
                                       check=True,capture_output=True,text=True)
        state['quotas']=quota
        from suite import save
        save(path,state)
    finally:lock.rmdir()
