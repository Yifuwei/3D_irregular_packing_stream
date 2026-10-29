"""Fixed six-task independent flows, real task maps, and duplicate prevention."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import test_suite as workflows
from suite import submit,collect
from submission_graph import PARENTS,active_job_ids,array_active

STAGES=('calibration','sensitivity','rcl','tuning','full')


def calibrated(root):
    helper=workflows.FrozenWorkflowTests();bundle,manifest=helper.fixture(root)
    for case in manifest['profile']['plans']['calibration']:
        helper.success(bundle,manifest,case,best_U_star=.8,mean_piece_count_without_emptiest=10,
                       bin_capacity=100,max_piece_volume=30,max_piece_radio=1000)
    with contextlib.redirect_stdout(io.StringIO()):collect(bundle,'calibration')
    return bundle,manifest


class SubmissionGraphTests(unittest.TestCase):
    def test_one_shot_has_independent_sensitivity_and_serial_heuristics(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,_=calibrated(Path(name))
            with contextlib.redirect_stdout(io.StringIO()):jobs=submit(bundle,dry_run=True,through='full')
            controls={};arrays={}
            for job in jobs:
                cmd=job['command'];label=next(a.split('=',1)[1] for a in cmd if a.startswith('--job-name='))
                if label.startswith('ejor_collect_'):controls[label[13:]]=job['job_id']
                else:arrays.setdefault(label[5:],[]).append(cmd)
            for commands in arrays.values():
                self.assertTrue(all(a.endswith('%6') for cmd in commands for a in cmd if a.startswith('--array=')))
            for stage in ('sensitivity','rcl'):
                for cmd in arrays[stage]:
                    dep=next(a for a in cmd if a.startswith('--dependency='))
                    self.assertIn('afterok:'+controls['calibration'],dep)
                    self.assertNotIn(controls['sensitivity'],dep)
            for stage,parent in (('tuning','rcl'),('full','tuning')):
                dep=next(a for a in arrays[stage][0] if a.startswith('--dependency='))
                self.assertIn('afterok:'+controls[parent],dep)
                self.assertNotIn(controls['sensitivity'],dep)

    def test_separate_flow_dry_runs_only_include_the_requested_stages(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,_=calibrated(Path(name))
            for flow,expected in (('sensitivity',{'sensitivity'}),('heuristics',{'rcl','tuning','full'})):
                with contextlib.redirect_stdout(io.StringIO()):jobs=submit(bundle,dry_run=True,flow=flow)
                arrays=[j for j in jobs if any(a.startswith('--array=') for a in j['command'])]
                names={a.split('ejor_',1)[1] for j in arrays for a in j['command'] if a.startswith('--job-name=')}
                self.assertEqual(names,expected)
                self.assertTrue(all(a.endswith('%6') for j in arrays for a in j['command'] if a.startswith('--array=')))

    def test_non_dry_submission_writes_all_task_maps_without_a_concurrency_coordinator(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,m=workflows.FrozenWorkflowTests().fixture(Path(name));commands=[]
            def scheduler(args,**kwargs):
                if args[0]=='squeue':return ''
                if args[0]=='sbatch':commands.append(args);return str(8000+len(commands))+'\n'
                raise AssertionError(args)
            with patch.dict(os.environ,USER='test_user'),patch('suite.subprocess.check_output',side_effect=scheduler),contextlib.redirect_stdout(io.StringIO()):
                submitted=submit(bundle,dry_run=False)
            self.assertEqual(len(submitted),len(commands));indices={s:[] for s in STAGES}
            for command in commands:
                label=next(a.split('=',1)[1] for a in command if a.startswith('--job-name='))
                if label.startswith('ejor_collect_'):continue
                env=next(a for a in command if a.startswith('--export='))
                task_map=Path(env.split('EJOR_TASK_MAP=',1)[1])
                self.assertTrue(task_map.is_file());indices[label[5:]].extend(json.loads(task_map.read_text()))
            for stage,cases in m['profile']['plans'].items():self.assertEqual(sorted(indices[stage]),list(range(len(cases))),stage)
            self.assertFalse((bundle/'submission_history/concurrency.json').exists())

    def test_other_flow_can_submit_while_sensitivity_runs_and_holds_a_lock(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,_=calibrated(Path(name));folder=bundle/'results/sensitivity';folder.mkdir(exist_ok=True)
            (folder/'task_0.lock').write_text('running');commands=[]
            def scheduler(args,**kwargs):
                if args[0]=='squeue':return '123|RUNNING\n'
                if args[0]=='scontrol':return 'JobName=ejor_sensitivity StdOut='+str(bundle.resolve()/'logs/sensitivity.out')
                if args[0]=='sbatch':commands.append(args);return str(9000+len(commands))+'\n'
                raise AssertionError(args)
            with patch.dict(os.environ,USER='test_user'),patch('suite.subprocess.check_output',side_effect=scheduler),contextlib.redirect_stdout(io.StringIO()):
                jobs=submit(bundle,flow='heuristics')
            self.assertTrue(jobs)
            self.assertFalse(any('--job-name=ejor_sensitivity' in j['command'] for j in jobs))

    def test_duplicate_flow_is_rejected_even_for_suspended_jobs(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,_=calibrated(Path(name))
            def scheduler(args,**kwargs):
                if args[0]=='squeue':return '123|SUSPENDED\n'
                if args[0]=='scontrol':return 'JobName=ejor_sensitivity StdOut='+str(bundle.resolve()/'logs/sensitivity.out')
                raise AssertionError('Unexpected submission: '+str(args))
            with patch.dict(os.environ,USER='test_user'),patch('suite.subprocess.check_output',side_effect=scheduler):
                with self.assertRaisesRegex(RuntimeError,'active job'):submit(bundle,flow='sensitivity')

    def test_separate_flows_require_the_calibrated_capacities(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,_=workflows.FrozenWorkflowTests().fixture(Path(name))
            with self.assertRaisesRegex(ValueError,'Complete calibration'):submit(bundle,dry_run=True,flow='heuristics')

    def test_scheduler_includes_suspended_configuring_and_running_but_not_terminal(self):
        with patch('submission_graph.subprocess.check_output',return_value='11|SUSPENDED\n12|CONFIGURING\n13|COMPLETED\n14|CANCELLED\n15|RUNNING\n'):
            self.assertEqual(active_job_ids('user'),['11','12','15'])
        for state,expected in [('SUSPENDED',True),('CONFIGURING',True),('PENDING',True),('COMPLETED',False),('CANCELLED',False)]:
            with self.subTest(state=state),patch('submission_graph.subprocess.run',return_value=types.SimpleNamespace(returncode=0,stdout=state,stderr='')):
                self.assertEqual(array_active('11'),expected)


if __name__=='__main__':unittest.main()
