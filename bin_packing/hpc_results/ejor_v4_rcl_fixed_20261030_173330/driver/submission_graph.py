"""Simple independent submission flows and conservative scheduler-state checks."""
import subprocess

PARENTS={
    'calibration':(),
    'sensitivity':('calibration',),
    'rcl':('calibration',),
    'tuning':('rcl',),
    'full':('tuning',),
}

TERMINAL_STATES={'BOOT_FAIL','CANCELLED','COMPLETED','DEADLINE','FAILED','NODE_FAIL',
                 'OUT_OF_MEMORY','PREEMPTED','TIMEOUT','REVOKED'}


def terminal_state(state):
    return state.strip().split('+',1)[0].split()[0] in TERMINAL_STATES if state.strip() else False


def active_job_ids(user):
    output=subprocess.check_output(['squeue','-h','-u',user,'-o','%A|%T','--states=all'],text=True)
    active=set()
    for line in output.splitlines():
        if not line.strip():continue
        job,separator,state=line.strip().partition('|')
        if not separator:raise RuntimeError('Cannot parse scheduler state: '+line)
        if not terminal_state(state):active.add(job.strip())
    return sorted(active)


def array_active(job):
    result=subprocess.run(['squeue','-h','-j',job,'-o','%T','--states=all'],capture_output=True,text=True)
    if result.returncode:
        if 'Invalid job id' in result.stderr:return False
        raise RuntimeError('Cannot verify array completion: '+result.stderr)
    return any(not terminal_state(state) for state in result.stdout.splitlines() if state.strip())
