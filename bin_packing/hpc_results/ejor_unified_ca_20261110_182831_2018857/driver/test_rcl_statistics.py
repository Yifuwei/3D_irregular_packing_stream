import sys
from pathlib import Path
from collections import Counter
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from orientation_rcl import choose_orientation
from rcl_statistics import record_rcl_event,summarize_rcl

class RclStatisticsTests(unittest.TestCase):
    def sample(self,top_k=None,tolerance=.2):
        # Actual engine payload: (PES, (orientation, position)).
        values=[(10.,('x_0',(0,0,0))),(10.5,('x_0',(1,0,0))),
                (11.,('y_90',(0,0,0))),(14.,('z_90',(0,0,0)))]
        events=[]
        picked=choose_orientation(values,tolerance,np.random.default_rng(7),events.append,top_k=top_k)
        return values,picked,events[0]
    def test_tolerance_counts_unique_labels_after_filtering(self):
        _,_,event=self.sample()
        self.assertEqual(event['active_rcl_size'],3)
        self.assertEqual(event['rcl_orientation_count'],2)
        self.assertEqual(event['rcl_orientations'],['x_0','y_90'])
    def test_top_k_counts_only_active_list(self):
        for k,expected in ((1,1),(2,1),(3,2),(10,3),(30,3)):
            with self.subTest(k=k):
                _,_,event=self.sample(top_k=k)
                self.assertEqual(event['rcl_orientation_count'],expected)
    def test_worker_receives_event_and_saves_json_histogram(self):
        import json
        hist={'pooled':Counter(),'orientation_count':Counter()}
        for k in (2,3,10):record_rcl_event(hist,self.sample(top_k=k)[2])
        restored=json.loads(json.dumps(hist))
        self.assertEqual(restored['orientation_count'],{'1':1,'2':1,'3':1})
        self.assertEqual(summarize_rcl(restored),dict(events=3,mean_distinct_orientations=2.,min_distinct_orientations=1,max_distinct_orientations=3))
    def test_missing_or_invalid_telemetry_fails(self):
        for event in ({'active_rcl_size':3},{'active_rcl_size':3,'rcl_orientation_count':4}):
            with self.assertRaises(ValueError):record_rcl_event({'pooled':Counter(),'orientation_count':Counter()},event)
    def test_observation_does_not_change_selection(self):
        values,picked,_=self.sample()
        self.assertEqual(picked,choose_orientation(values,.2,np.random.default_rng(7)))
    def test_empty_candidates_emit_no_event(self):
        events=[]
        self.assertIsNone(choose_orientation([], .2,observer=events.append))
        self.assertEqual(events,[])
