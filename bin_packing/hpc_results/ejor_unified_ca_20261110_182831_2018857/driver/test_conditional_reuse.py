import json
from pathlib import Path
import unittest
from unittest.mock import patch
from conditional_reuse import try_reuse

class ConditionalReuseTests(unittest.TestCase):
    def check(self, old_settings, new_settings, priority=True, geometry=False):
        case=dict(stage='full',task_id=0,dataset='d',shape='cylinder',
                  weight_level='weak',radio_level='weak')
        relative='inputs/conditional_reuse/full/task_0.json'
        manifest=dict(bundle_id='new',files_sha256={relative:'digest'},
                      profile=dict(constraint_mode='geometry_only' if geometry else 'weight_radiation'))
        report=dict(algorithm='improved_ILS',search_settings=old_settings,
                    method=dict(pes='p',selection_range='bottom_top',radio_priority=priority),
                    weight_capacity=10.,radio_capacity=20.)
        candidate=dict(report=report,source_bundle_id='old',source_case=case)
        selections=dict(calibration={'d|cylinder':dict(weight_capacity={'weak':10.},
                                                      radio_capacity={'weak':20.})})
        with patch('conditional_reuse.verified_bytes',return_value=json.dumps(candidate).encode()),              patch('conditional_reuse.packing_settings',return_value=dict(pes='p',selection_range='bottom_top')),              patch('conditional_reuse.read_selections',return_value=selections),              patch('conditional_reuse.settings_for',return_value=('improved_ILS',new_settings)),              patch('suite.save'):
            return try_reuse(Path('.'),case,manifest)
    def test_matching_parameters_reuse(self):
        self.assertTrue(self.check({'kick_level':'medium'},{'kick_level':'medium'}))
    def test_changed_kick_reruns(self):
        self.assertFalse(self.check({'kick_level':'medium'},{'kick_level':'high'}))
    def test_changed_rcl_reruns(self):
        self.assertFalse(self.check({'rcl_top_k':10},{'rcl_top_k':30}))
    def test_geometry_cannot_reuse_constrained_result(self):
        self.assertFalse(self.check({}, {}, priority=True,geometry=True))
    def test_geometry_reuses_first_fit(self):
        self.assertTrue(self.check({}, {}, priority=False,geometry=True))
