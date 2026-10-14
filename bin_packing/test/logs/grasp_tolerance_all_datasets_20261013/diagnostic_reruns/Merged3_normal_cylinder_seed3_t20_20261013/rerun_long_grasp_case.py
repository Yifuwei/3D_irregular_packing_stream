"""Run one diagnostic worker between batch workers; always resume the parent."""
import ctypes
from ctypes import wintypes
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('C:/work/paper_repo/3D_irregular_packing_stream')
BATCH = ROOT/'bin_packing/test/logs/grasp_tolerance_all_datasets_20261013'
OUT = BATCH/'diagnostic_reruns/Merged3_normal_cylinder_seed3_t20_20261013'
LABEL = 'Merged3_normal_cylinder_seed3_t20'

k32=ctypes.WinDLL('kernel32',use_last_error=True)
nt=ctypes.WinDLL('ntdll')
k32.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
k32.OpenProcess.restype=wintypes.HANDLE
k32.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
k32.WaitForSingleObject.restype=wintypes.DWORD
k32.CloseHandle.argtypes=(wintypes.HANDLE,)
nt.NtSuspendProcess.argtypes=(wintypes.HANDLE,)
nt.NtSuspendProcess.restype=ctypes.c_long
nt.NtResumeProcess.argtypes=(wintypes.HANDLE,)
nt.NtResumeProcess.restype=ctypes.c_long


def save(phase,**values):
    data=dict(phase=phase,parent_pid=37224,current_worker_pid=45456,
              timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
    temp=OUT/'status.json.tmp';temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temp.replace(OUT/'status.json')
    print(json.dumps(data),flush=True)


def process(pid):
    cmd=f'Get-CimInstance Win32_Process -Filter "ProcessId={pid}" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress'
    r=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',cmd],capture_output=True,text=True,check=True)
    return json.loads(r.stdout) if r.stdout.strip() else None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'status.json').exists():raise RuntimeError('Diagnostic already initialized; refusing duplicate')
    manifest=json.loads((BATCH/'manifest.json').read_text())
    for rel,h in manifest['hashes'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,'Source mismatch: '+rel
    parent=process(37224);child=process(45456)
    assert parent and 'run_all_dataset_tolerance.py' in parent['CommandLine']
    assert str(BATCH).replace('\\','/') in parent['CommandLine'].replace('\\','/')
    assert child and child['ParentProcessId']==37224 and 'run_tolerance_experiment.py' in child['CommandLine']
    assert 'Merged3_normal_cylinder_seed3_t10.json' in child['CommandLine']
    original=json.loads((BATCH/(LABEL+'.json')).read_text())
    for suffix in ('.json','.log','.rcl.csv'):
        shutil.copy2(BATCH/(LABEL+suffix),OUT/('original'+suffix))
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    parent_handle=k32.OpenProcess(0x0800|0x100000,False,37224)
    child_handle=k32.OpenProcess(0x100000,False,45456)
    if not parent_handle or not child_handle:raise ctypes.WinError(ctypes.get_last_error())
    suspended=False
    try:
        status=nt.NtSuspendProcess(parent_handle)
        if status!=0:raise RuntimeError(f'NtSuspendProcess status {status}')
        suspended=True
        save('waiting_current_worker',original_wall_seconds=original['wall_seconds'])
        # Only the CPU-side parent is suspended; its current GPU worker finishes normally.
        while k32.WaitForSingleObject(child_handle,10000)==0x102:
            pass
        # Verify the immutable inputs again immediately before rerunning.
        for rel,h in manifest['hashes'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,'Source mismatch: '+rel
        for rel,h in manifest['input_hashes'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,'Input mismatch: '+rel
        result=OUT/'rerun.json'
        command=[sys.executable,'-u',str(ROOT/'bin_packing/test/run_tolerance_experiment.py'),
                 '--worker','--dataset','Merged3_normal','--shape','cylinder','--seed','3',
                 '--tolerance','0.2','--time-limit','180','--iterations','20','--output',str(result)]
        started=time.perf_counter()
        with (OUT/'rerun.log').open('w',encoding='utf-8') as log:
            worker=subprocess.Popen(command,cwd=ROOT/'bin_packing/test',stdout=log,stderr=subprocess.STDOUT)
            save('rerunning',rerun_pid=worker.pid,command=command)
            try:
                code=worker.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                worker.kill();worker.wait()
                raise RuntimeError('Diagnostic safety timeout after two hours')
        if code or not result.exists():raise RuntimeError(f'Rerun worker failed: {code}')
        rerun=json.loads(result.read_text())
        if rerun['status']!='success' or rerun['validation_errors']:
            raise RuntimeError('Diagnostic result failed validation')
        for key in ('dataset','shape','order_seed','tolerance','piece_seed','position_seed',
                    'time_limit','ls_limit','pes','selection_range','dimensions'):
            assert rerun[key]==original[key],key
        rows=[]
        for name,j in [('original',original),('rerun',rerun)]:
            rows.append(dict(run=name,wall_seconds=j['wall_seconds'],construction_seconds=j['construction_seconds'],
                N=j['best_N'],U_star=j['best_U_star'],F=j['F'],CA_calls=j['constructions'],LS_calls=j['ls_calls'],
                RCL_calls=j['rcl_calls'],RCL_mean=j['rcl_mean'],status=j['status']))
        with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
        lines=['# Diagnostic rerun: Merged3_normal cylinder seed 3, RCL 20%','',
               'Same immutable sources, inputs, seeds, PES and search budgets. Original batch records were preserved; diagnostic rerun is not silently substituted into the 270-run results.',
               '', '| Run | Wall seconds | Construction seconds | N | U* | N-U* | CA calls | LS calls |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f"| {r['run']} | {r['wall_seconds']:.3f} | {r['construction_seconds']:.3f} | {r['N']} | {r['U_star']:.9f} | {r['F']:.9f} | {r['CA_calls']} | {r['LS_calls']} |")
        lines+=['','Both recorded geometric validations passed. The original 13315.7 seconds was almost entirely one non-preemptive construction call, with zero LS calls. A runtime difference alone does not establish its cause. Host suspension, scheduling, and geometry workload have not been isolated. Full insertion trajectories are not certified.']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        save('completed',rerun_wall_seconds=rerun['wall_seconds'],external_seconds=time.perf_counter()-started)
    except Exception as e:
        save('failed',error=repr(e))
        raise
    finally:
        if suspended:
            status=nt.NtResumeProcess(parent_handle)
            (OUT/'parent_resume.json').write_text(json.dumps(dict(parent_pid=37224,ntstatus=status)),encoding='utf-8')
            if status:print(f'PARENT RESUME FAILED: {status}',flush=True)
        k32.CloseHandle(child_handle);k32.CloseHandle(parent_handle)


if __name__=='__main__':main()
