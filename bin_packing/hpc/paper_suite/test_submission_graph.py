"""Independent branches, exact dependency edges and a shared GPU throttle."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import test_suite as workflows
from suite import submit,save
from submission_graph import PARENTS,initial_state,allocations,rebalance

STAGES=('calibration','sensitivity','rcl','tuning','full')


def active(state):
    return [s for s in state['stages'] if not state['finished'][s] and
            all(state['passed'].get(p,False) for p in PARENTS[s])]


class SubmissionGraphTests(unittest.TestCase):
    def test_actual_submission_edges_do_not_chain_sensitivity_to_heuristics(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,_=workflows.FrozenWorkflowTests().fixture(Path(name))
            with contextlib.redirect_stdout(io.StringIO()):jobs=submit(bundle,dry_run=True,through='full')
            controls={};arrays={}
            for job in jobs:
                cmd=job['command'];label=next(a.split('=',1)[1] for a in cmd if a.startswith('--job-name='))
                if label.startswith('ejor_collect_'):controls[label[13:]]=job['job_id']
                else:arrays.setdefault(label[5:],[]).append(cmd)
            cal=controls['calibration'];sens=controls['sensitivity']
            for stage in ('sensitivity','rcl','tuning'):
                for cmd in arrays[stage]:
                    dependency=next(a for a in cmd if a.startswith('--dependency='))
                    self.assertIn('afterok:'+cal,dependency)
                    self.assertNotIn(sens,dependency)
                    self.assertTrue(any(a.endswith('%2') for a in cmd if a.startswith('--array=')))
            dep=next(a for a in arrays['full'][0] if a.startswith('--dependency='))
            self.assertIn(controls['rcl'],dep);self.assertIn(controls['tuning'],dep)
            self.assertNotIn(sens,dep)

    def test_all_completion_orders_keep_total_gpu_limit_and_only_increase_live_shares(self):
        for order in itertools.permutations(('sensitivity','rcl','tuning')):
            state=initial_state(STAGES,{s:int(s!='calibration') for s in STAGES},6)
            old=allocations(state)
            self.assertEqual([old[s] for s in ('sensitivity','rcl','tuning')],[2,2,2])
            for completed in order:
                previously_active=set(active(state))
                state['finished'][completed]=state['passed'][completed]=True
                new=allocations(state)
                self.assertLessEqual(sum(new[s] for s in active(state)),6)
                for s in previously_active & set(active(state)):
                    self.assertGreaterEqual(new[s],old[s],(order,s,old,new))
                old=new
            self.assertEqual(new['full'],6)

    def test_sensitivity_failure_does_not_block_full(self):
        state=initial_state(STAGES,{s:int(s!='calibration') for s in STAGES},6)
        for stage in ('sensitivity','rcl','tuning'):state['finished'][stage]=True
        state['passed'].update(sensitivity=False,rcl=True,tuning=True)
        self.assertEqual(active(state),['full'])
        self.assertEqual(allocations(state)['full'],6)

    def test_rebalance_updates_slurm_throttles_after_slots_are_released(self):
        with tempfile.TemporaryDirectory() as name:
            bundle=Path(name);state=initial_state(STAGES,{s:int(s!='calibration') for s in STAGES},6)
            state['arrays']={s:([str(i)] if s!='calibration' else []) for i,s in enumerate(STAGES,10)}
            state['quotas']=allocations(state)
            save(bundle/'submission_history/concurrency.json',state)
            live={'12','13','14'};updates=[]
            def command(args,**kwargs):
                if args[0]=='squeue':return types.SimpleNamespace(returncode=0,stdout=args[3] if args[3] in live else '',stderr='')
                updates.append(args);return types.SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch('submission_graph.subprocess.run',side_effect=command):
                rebalance(bundle,'sensitivity',True)
                self.assertEqual({tuple(a[-2:]) for a in updates},
                    {('JobId=12','ArrayTaskThrottle=3'),('JobId=13','ArrayTaskThrottle=3')})
                updates.clear();live.remove('12');rebalance(bundle,'rcl',True)
                self.assertIn(['scontrol','update','JobId=13','ArrayTaskThrottle=6'],updates)
                updates.clear();live.remove('13');rebalance(bundle,'tuning',True)
                self.assertIn(['scontrol','update','JobId=14','ArrayTaskThrottle=6'],updates)
            final=json.loads((bundle/'submission_history/concurrency.json').read_text())
            self.assertEqual(final['quotas']['full'],6)

    def test_manual_collect_during_running_array_cannot_release_slots(self):
        with tempfile.TemporaryDirectory() as name:
            bundle=Path(name);state=initial_state(STAGES,{s:int(s!='calibration') for s in STAGES},6)
            state['arrays']['sensitivity']=['11'];state['quotas']=allocations(state)
            save(bundle/'submission_history/concurrency.json',state)
            with patch('submission_graph.subprocess.run',return_value=types.SimpleNamespace(returncode=0,stdout='11',stderr='')):
                rebalance(bundle,'sensitivity',False)
            final=json.loads((bundle/'submission_history/concurrency.json').read_text())
            self.assertFalse(final['finished']['sensitivity'])

    def test_one_gpu_uses_resource_serialisation_without_sensitivity_success_dependency(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,_=workflows.FrozenWorkflowTests().fixture(Path(name))
            with contextlib.redirect_stdout(io.StringIO()):jobs=submit(bundle,dry_run=True,max_parallel=1,through='full')
            rcl=next(j for j in jobs if '--job-name=ejor_rcl' in j['command'])
            sens=next(j for j in jobs if '--job-name=ejor_sensitivity' in j['command'])
            collector=next(j for j in jobs if '--job-name=ejor_collect_sensitivity' in j['command'])
            dep=next(a for a in rcl['command'] if a.startswith('--dependency='))
            self.assertIn('afterany:'+sens['job_id'],dep)
            self.assertNotIn(collector['job_id'],dep)
            self.assertTrue(all(a.endswith('%1') for j in jobs for a in j['command'] if a.startswith('--array=')))


if __name__=='__main__':unittest.main()
