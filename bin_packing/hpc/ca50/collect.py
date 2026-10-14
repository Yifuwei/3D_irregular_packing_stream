"""Collect asynchronous array results without importing CUDA on login nodes."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

from task import label,load,verify_bundle,validate_result


def collect(bundle,results):
    manifest=load(bundle);verify_bundle(bundle,manifest)
    runner_path=bundle/'runtime/bin_packing/test/run_ca50_tolerance_experiment.py'
    spec=importlib.util.spec_from_file_location('ca50_frozen_reporter',runner_path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    results.mkdir(parents=True,exist_ok=True)
    inventory=[];records=[];pending=[]
    for index,c in enumerate(manifest['plan']):
        p=results/(label(c)+'.json');status='missing';reason=''
        if p.exists():
            try:
                r=json.loads(p.read_text(encoding='utf-8'))
                if validate_result(r,c):
                    # Verify raw RCL counts and the evaluation CSV before quality inclusion.
                    from collections import Counter
                    with p.with_suffix('.rcl.csv').open(newline='',encoding='utf-8') as f:
                        h=Counter(int(x['rcl_size']) for x in csv.DictReader(f))
                    if dict(h)!={int(k):v for k,v in r['rcl_histogram'].items()}:
                        raise ValueError('RCL histogram mismatch')
                    with p.with_suffix('.ls.csv').open(newline='',encoding='utf-8') as f:
                        calls=[x for x in csv.DictReader(f) if x['kind']=='ls']
                    if len(calls)!=r['ls_calls']:raise ValueError('LS CSV count mismatch')
                    status='success';records.append(r)
                else:
                    status='failed' if r.get('status')=='failed' else 'invalid'
                    reason=r.get('traceback',r.get('reason','Configuration, work-count or geometry validation mismatch'))
                    records.append(dict(c,status=status,reason=reason))
            except (OSError,ValueError,KeyError,TypeError) as e:
                status='invalid';reason=repr(e);records.append(dict(c,status=status,reason=reason))
        if status!='success':pending.append(str(index))
        inventory.append(dict(index=index,**c,status=status,reason=reason))
    runner.summarize(results,manifest['plan'],records)
    runner.write_csv(results/'task_inventory.csv',inventory,['index','dataset','shape','order_seed','tolerance','ls_evaluations','status','reason'])
    (results/'pending_indices.txt').write_text(','.join(pending)+'\n',encoding='utf-8')
    print(f"Success: {sum(r['status']=='success' for r in inventory)}/{len(inventory)}; pending/retry: {len(pending)}")
    print('Result table:',results/'best_tolerance_by_dataset.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--results',type=Path)
    a=parser.parse_args();bundle=a.bundle.resolve()
    collect(bundle,a.results.resolve() if a.results else bundle/'results')
