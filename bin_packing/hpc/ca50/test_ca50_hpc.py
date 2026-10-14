import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import task


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.case=dict(dataset='chess',shape='cube',order_seed=3,tolerance=.1,ls_evaluations=50)
        self.result=dict(status='success',dataset='chess',shape='cube',order_seed=3,tolerance=.1,
             piece_seed=101,position_seed=202,target_ls_evaluations=50,wall_time_limit=None,
             restart_limit=1,no_improvement_limit=50,selection_range='bottom_top',
             protocol='one_CA_plus_50_LS',algorithm='improved_GRASP',pes='maximal_residual_box',
             validation_errors=[],ca_validation_errors=[],constructions=1,ls_calls=50,
             stop_reason='50_evaluations_completed',best_N=3,best_U_star=.25,F=2.75,
             history=[dict(kind='ls') for _ in range(50)],rcl_histogram={},rcl_calls=0)

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def test_full_and_single_bin_results_are_valid(self):
        self.assertTrue(task.validate_result(self.result,self.case))
        r=copy.deepcopy(self.result);r.update(ls_calls=0,best_N=1,F=.75,
                                history=[],stop_reason='single_bin_reached')
        self.assertTrue(task.validate_result(r,self.case))

    def test_wrong_rate_seed_protocol_counts_geometry_and_nan_are_rejected(self):
        for change in [dict(tolerance=.15),dict(piece_seed=7),dict(ls_calls=49),
                       dict(constructions=2),dict(validation_errors=['overlap']),
                       dict(wall_time_limit=180),dict(F=float('nan'))]:
            r=copy.deepcopy(self.result);r.update(change)
            self.assertFalse(task.validate_result(r,self.case),change)

    def test_array_mapping_and_pending_selection(self):
        with self.temporary() as d:
            p=Path(d);cases=[self.case,dict(self.case,tolerance=.15)]
            (p/'manifest.json').write_text(json.dumps(dict(plan=cases,files_sha256={})))
            results=p/'results';results.mkdir()
            self.assertEqual(task.array_spec(p,results,False),'0-1')
            (results/(task.label(self.case)+'.json')).write_text(json.dumps(self.result))
            (results/(task.label(self.case)+'.rcl.csv')).write_text('rcl_size\n')
            (results/(task.label(self.case)+'.ls.csv')).write_text('kind\n'+'ls\n'*50)
            self.assertEqual(task.array_spec(p,results,True),'1')
            (results/(task.label(self.case)+'.ls.csv')).write_text('kind\nls\n')
            self.assertEqual(task.array_spec(p,results,True),'0,1')

    def test_dry_run_with_spaces_does_not_create_result_files(self):
        with self.temporary() as d:
            p=Path(d)/'bundle with spaces';p.mkdir()
            (p/'manifest.json').write_text(json.dumps(dict(plan=[self.case],files_sha256={})))
            self.assertEqual(task.run_task(p,p/'output with spaces',0,True),0)
            self.assertFalse((p/'output with spaces').exists())
            with self.assertRaises(ValueError):task.run_task(p,p/'results',-1,True)
            with self.assertRaises(ValueError):task.run_task(p,p/'results',1,True)

    def test_modified_package_and_path_escape_are_rejected(self):
        with self.temporary() as d:
            p=Path(d);(p/'data.txt').write_text('changed')
            with self.assertRaises(ValueError):task.verify_bundle(p,dict(files_sha256={'data.txt':'wrong'}))
            with self.assertRaises(ValueError):task.verify_bundle(p,dict(files_sha256={'../escape':'wrong'}))

    def test_memory_groups_preserve_global_indices_and_pending_selection(self):
        self.assertEqual(task.memory_gb(self.case),16)
        self.assertEqual(task.memory_gb(dict(self.case,dataset='shapesnew')),16)
        self.assertEqual(task.memory_gb(dict(self.case,dataset='engine')),32)
        with self.temporary() as d:
            p=Path(d);cases=[self.case,dict(self.case,dataset='engine'),dict(self.case,dataset='shapesnew')]
            (p/'manifest.json').write_text(json.dumps(dict(plan=cases,files_sha256={})))
            self.assertEqual(task.array_spec(p,p/'results',False,16),'0,2')
            self.assertEqual(task.array_spec(p,p/'results',False,32),'1')


if __name__=='__main__':unittest.main()
