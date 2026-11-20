"""GRASP deliberately bypasses dynamic preference selection."""
import unittest
import test_grasp_systematic as systematic

class FixedOrderGraspTests(unittest.TestCase):
    def test_fixed_order_and_complete_neighborhood(self):
        harness=systematic.SystematicGraspTests()
        _,calls,ends=harness.search()
        self.assertEqual([p for _,p,_,_ in calls],[0,0,1,1,2,2])
        self.assertTrue(ends[0]['local_optimum'])
