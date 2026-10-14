"""Verify fixed-work budgets and actual original GRASP controller on CPU."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import run_ca50_tolerance_experiment as runner


BASELINE=Path('C:/work/paper_repo/3D_irregular_packing_stream/bin_packing/test/logs/grasp_tolerance_all_datasets_20261013/source_snapshot/bin_packing/src/new_ILS_southampton.py')


class FixedWorkTests(unittest.TestCase):
    def test_positional_launcher_budget_is_overridden(self):
        def f(x,iter_limit_per_iter,time_limit,restart_limit=None,no_improvement_limit=None):
            return x,iter_limit_per_iter,time_limit,restart_limit,no_improvement_limit
        self.assertEqual(runner.fixed_work_call(f,7,20,180),(7,50,float('inf'),1,50))

    def test_keyword_launcher_budget_is_overridden(self):
        def f(iter_limit_per_iter,time_limit,restart_limit=None,no_improvement_limit=None):
            return iter_limit_per_iter,time_limit,restart_limit,no_improvement_limit
        self.assertEqual(runner.fixed_work_call(f,iter_limit_per_iter=20,time_limit=180),
                         (50,float('inf'),1,50))

    def run_controller(self, single_bin=False, reaches_single=False, improve=False):
        clock=[0];counts=dict(ca=0,ls=0)
        def packed(score=3,bins=2):
            if bins==1:score=1
            groups=[[dict(score=score,orientation='x_0',array=np.ones((1,1,1)),bin_position=0)]]
            if bins==2:groups.append([dict(orientation='x_0',array=np.ones((1,1,1)),bin_position=1)])
            else:groups[0].append(dict(orientation='x_0',array=np.ones((1,1,1)),bin_position=0))
            return groups,[],[],[[0,1]] if bins==1 else [[0],[1]]
        def construct(*a,**kw):
            clock[0]+=5000;counts['ca']+=1
            return packed(bins=1 if single_bin else 2)
        def repack(*a,**kw):
            clock[0]+=100;counts['ls']+=1
            bins=1 if reaches_single and counts['ls']==5 else 2
            p=packed(score=3-counts['ls']*.001 if improve else 3,bins=bins)
            p[0][0][0]['orientation']=a[1][1]
            return p
        tree=ast.parse(BASELINE.read_text())
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='improved_GRASP']
        scope=dict(np=np,copy=copy,time=types.SimpleNamespace(perf_counter=lambda:clock[0]),
                   trace=lambda *a:None,packing_3D_voxel_lookback=construct,improved_repack=repack,
                   improve_pieces_selection_ls=lambda *a,**kw:0,
                   get_final_performance=lambda o,c,s,l:(l[0][0]['score'],0,None))
        exec(compile(tree,str(BASELINE),'exec'),scope)
        result=runner.fixed_work_call(scope['improved_GRASP'],[{},{}],None,None,999999,1,
            None,None,['x_0','x_90'],'SCH','bounding_box','bottom_top',False,
            'minimum_aabb_volume',None,(1,1,1),'cube',20,180,1,False,False,False)
        return counts,result

    def test_slow_ca_still_gets_fifty_rejected_ls_evaluations(self):
        counts,result=self.run_controller()
        self.assertEqual(counts,dict(ca=1,ls=50))
        self.assertGreater(result[-1],180)

    def test_accepting_improvements_does_not_reset_total_budget(self):
        counts,_=self.run_controller(improve=True)
        self.assertEqual(counts,dict(ca=1,ls=50))

    def test_single_bin_stops_are_recorded_without_filling_ls_counts(self):
        counts,_=self.run_controller(single_bin=True)
        self.assertEqual(counts,dict(ca=1,ls=0))
        counts,_=self.run_controller(reaches_single=True)
        self.assertEqual(counts,dict(ca=1,ls=5))
        self.assertEqual(runner.assert_work_counts(1,5,1),'single_bin_reached')
        with self.assertRaises(ValueError):runner.assert_work_counts(1,5,2)

    def test_no_second_construction_or_over_budget_count_is_valid(self):
        with self.assertRaises(ValueError):runner.assert_work_counts(2,50,2)
        with self.assertRaises(ValueError):runner.assert_work_counts(1,51,2)

    def test_partial_tables_and_failures_do_not_select_best_rate(self):
        plan=[dict(dataset='x')];results=[dict(dataset='x',shape='cube',order_seed=3,
                 tolerance=.1,status='failed',traceback='test failure')]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            self.assertTrue(Path(d).resolve().is_relative_to(Path(__file__).resolve().parent))
            p=Path(d);runner.summarize(p,plan+[dict(dataset='other')],results)
            import csv
            with (p/'best_tolerance_by_dataset.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertTrue(all(r['best_tolerance_percent']=='' for r in rows))
            self.assertTrue(all(r['valid_10']=='0' for r in rows))
            progress=json.loads((p/'progress.json').read_text())
            self.assertEqual(progress['successes'],0)
            self.assertEqual(progress['failed_or_invalid'],1)


if __name__=='__main__':unittest.main()
