import unittest
from protocol_v4 import make_plan,packing_settings,BEST_PES,PARENTS,select_sensitivity

class FixedPesPlanTests(unittest.TestCase):
    def profile(self):
        datasets=['shapesnew','Merged4_normal','st04_Example5_normal']+[f'd{i}' for i in range(12)]
        return make_plan(datasets)
    def test_formal_grid(self):
        p=self.profile()
        self.assertEqual({k:len(v) for k,v in p['plans'].items()},dict(calibration=150,sensitivity=360,rcl=1440,tuning=216,full=1080))
        for stage in ('sensitivity','rcl','tuning','full'):
            for c in p['plans'][stage]:
                self.assertIn(c['seed'],(3,7,13))
                self.assertIn(c['weight_level'],('strong','weak'))
                self.assertIn(c['radio_level'],('strong','weak'))
                cfg=packing_settings(c,{},None)
                for key,value in BEST_PES[c['shape']].items():self.assertEqual(cfg[key],value)
    def test_sensitivity_is_independent(self):
        self.assertEqual(PARENTS['rcl'],('calibration',))
        self.assertEqual(PARENTS['sensitivity'],('calibration',))
    def test_summary_does_not_select_pes(self):
        p=self.profile();records=[dict(c,objective=1.) for c in p['plans']['sensitivity']]
        report=select_sensitivity(records,p)
        self.assertFalse(report['parameter_selection'])
        self.assertEqual(len(report['summary']),8)
        with self.assertRaises(ValueError):select_sensitivity(records[:-1],p)
