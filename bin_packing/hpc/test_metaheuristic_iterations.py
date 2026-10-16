import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('iterations', Path(__file__).with_name('metaheuristic_iterations.py'))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class IterationExperimentTests(unittest.TestCase):
    def test_nine_configurations_and_kwargs(self):
        configs = experiment.configurations()
        self.assertEqual(len({r['config_id'] for r in configs}), 9)
        common = dict(kick_level='medium', rcl_tolerance=.1, random_CA_threshold=5,
                      no_improvement_limit=150)
        for config in configs:
            kw = experiment.algorithm_kwargs(dict(config, seed=3), dict(time_limit=3600, common=common),
                                              {'piece':11,'position':22,'kick':33,'global':44})
            self.assertEqual(kw['time_limit'], 3600)
            if config['algorithm']=='improved_ILS':
                self.assertIsNone(kw['iteration_limit'])
                self.assertEqual(kw['kick_trigger_time'], config['kick_trigger'])
                self.assertNotIn('iter_limit_per_iter', kw)
            else:
                self.assertEqual(kw['iter_limit_per_iter'], config['ls_limit'])
                self.assertIsNone(kw['restart_limit'])
                if config['algorithm']=='improved_GRASP':
                    self.assertEqual(kw['no_improvement_limit'], config['ls_limit'])
                    self.assertIsNone(kw['ls_evaluation_limit'])
                else:
                    self.assertEqual(kw['kick_seed'], 33)

    def test_internal_seeds_are_time_based_and_numpy_compatible(self):
        with patch.object(experiment.time,'time_ns',side_effect=[1000,1001]):
            first_time,first=experiment.internal_seeds(0)
            second_time,second=experiment.internal_seeds(0)
        self.assertEqual((first_time,second_time),(1000,1001))
        self.assertNotEqual(first,second)
        self.assertTrue(all(0<=value<2**32 for value in first.values()))

    def fixture(self, root):
        project = root/'bin_packing'
        (project/'instances').mkdir(parents=True)
        (project/'data').mkdir()
        (project/'data'/'geometry.binvox').write_bytes(b'geometry')
        for dataset in ('small','medium','large'):
            for shape in ('cube','cylinder'):
                for seed in (3,7,13):
                    (project/'instances'/f'{dataset}_seq{seed}_{shape}_999999_1.txt').write_text(
                        f'{shape}\t(10, 10, 10)\t999999\t1\n../geometry.binvox\t1\t0\t1\n')
        return project

    def test_preparation_has_162_paired_tasks_and_geometry_hash(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment,'source_hashes',return_value={}):
            project = self.fixture(Path(tmp))
            m = experiment.prepare(project,['small','medium','large'],['cube','cylinder'],[3,7,13])
            self.assertEqual(len(m['tasks']),162)
            self.assertEqual(len(m['geometries']),1)
            self.assertEqual(len({t['task_id'] for t in m['tasks']}),162)
            self.assertEqual(sum(t['seed']==3 for t in m['tasks']),54)

    def test_missing_geometry_fails_before_submission(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment,'source_hashes',return_value={}):
            project = self.fixture(Path(tmp))
            (project/'data'/'geometry.binvox').unlink()
            with self.assertRaises(FileNotFoundError):
                experiment.prepare(project,['small'],['cube'],[3])

    def test_collection_retains_failed_missing_and_marks_partial(self):
        task = dict(task_id=0,dataset='small',shape='cube',seed=3,config_id='G100',algorithm='improved_GRASP',ls_limit=100,kick_trigger=None)
        manifest = dict(tasks=[task,dict(task,task_id=1,seed=7),dict(task,task_id=2,seed=13)])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            (output/'task_0.json').write_text(json.dumps(dict(task,status='success',validation_ok=True,
                                                            best_N=2,best_U_star=.8,best_N_minus_U_star=1.2)))
            (output/'task_1.json').write_text(json.dumps(dict(manifest['tasks'][1],status='failed',validation_ok=False,error='CUDA error')))
            rows,summary=experiment.collect(manifest,output)
            self.assertEqual([r['status'] for r in rows],['success','failed','missing'])
            self.assertFalse(summary[0]['complete'])
            self.assertEqual(summary[0]['success_count'],1)
            self.assertEqual(summary[0]['best_N_minus_U_star_mean'],1.2)

    def test_invalid_budgets_and_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            for budget in (0,-1,float('inf')):
                with self.assertRaises(ValueError):
                    experiment.prepare(Path(tmp),['small'],['cube'],[3],budget)
            with self.assertRaises(ValueError):
                experiment.prepare(Path(tmp),['small'],['cube'],[3,3])

    def test_rankings_use_only_complete_nine_configuration_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            tasks=[]
            for i,config in enumerate(experiment.configurations()):
                task=dict(task_id=i,dataset='small',shape='cube',seed=3,**config)
                tasks.append(task)
                score=1.0 if i<2 else 2.0+i/10
                (output/f'task_{i}.json').write_text(json.dumps(dict(task,status='success',
                    validation_ok=True,best_N=3,best_U_star=3-score,best_N_minus_U_star=score)))
            experiment.collect(dict(tasks=tasks),output)
            import csv
            with (output/'iteration_ranking.csv').open(encoding='utf-8-sig') as stream:
                ranks=list(csv.DictReader(stream))
            cube=[r for r in ranks if r['shape']=='cube']
            self.assertEqual(len(cube),9)
            self.assertEqual(sum(float(r['mean_rank'])==1.5 for r in cube),2)

    def test_validates_emptiest_bin_metric_and_rejects_wrong_metric(self):
        import numpy as np
        pieces=[dict(array=np.ones((1,1,1)),volume=1)]*2
        layout=[[dict(array=np.ones((1,1,1)),volume=1,orientation='x_0',translation=(0,0,0))] for _ in range(2)]
        result=[2,1,1,None,None,None,layout,None,[np.ones((1,1,1))]*2,[[0],[1]]]
        self.assertTrue(experiment.validate_result(result,pieces,(1,1,1),'cube')['ok'])
        result[2]=.5
        with self.assertRaises(ValueError):
            experiment.validate_result(result,pieces,(1,1,1),'cube')


if __name__=='__main__':
    unittest.main()
