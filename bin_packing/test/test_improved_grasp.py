"""Exercise the real GRASP control flow without CUDA geometry dependencies."""
import ast
import copy
from pathlib import Path
import types
import unittest
import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'new_ILS_southampton.py'


class ImprovedGraspTests(unittest.TestCase):
    def run_search(self, constructions, moves=(), budget=100, orientations=None,
                   iterations=10, stall_limit=1, restarts=2, total_cap=None):
        clock = [0]
        constructions, moves = iter(constructions), iter(moves)
        baselines = []
        self.selections = []

        def result(score, orientation):
            return [[dict(score=score, orientation=orientation, bin_position=0,
                          array=np.ones((1, 1, 1)))],
                    [dict(orientation='x_0', bin_position=1, array=np.ones((1, 1, 1)))]], [], [], [[0], [1]]

        def select(info, iteration, stage, bins, **kwargs):
            self.selections.append((kwargs['exclude_last_bin'], stage,
                                    info[0]['orientation'], iteration))
            return 0

        def construct(*args, **kwargs):
            self.assertEqual(args[10], 'minimum_aabb_volume')
            self.assertEqual(args[9], args[10])
            self.assertTrue(kwargs['unified_pes'])
            self.assertEqual(kwargs['rcl_tolerance'], .10)
            self.assertTrue(kwargs['_encourage_dbl'])
            clock[0] += 1
            return result(next(constructions), 'x_0')

        def repack(*args, **kwargs):
            self.assertEqual(args[14], 'minimum_aabb_volume')
            self.assertEqual(args[13], args[14])
            self.assertTrue(kwargs['_encourage_dbl'])
            self.assertEqual(kwargs['_type'], 'bounding_box')
            self.assertNotIn('rcl_tolerance', kwargs)
            state = args[5][0]
            baselines.append((state['score'], args[4][0]))
            # The engine may mutate its inputs; stored best/original must survive.
            state['bin_real_layout'][0][0]['score'] = -100
            clock[0] += 1
            return result(next(moves), args[1][1])

        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'improved_GRASP']
        scope = dict(copy=copy, np=np, time=types.SimpleNamespace(perf_counter=lambda: clock[0]),
                     trace=lambda msg: None, packing_3D_voxel_lookback=construct,
                     improved_repack=repack,
                     improve_pieces_selection_ls=select,
                     get_final_performance=lambda o, c, s, l:
                         (l[0][0]['score'], 0, None))
        exec(compile(tree, str(SOURCE), 'exec'), scope)
        output = scope['improved_GRASP'](
            [{}, {}], None, None, 0, 0, None, None,
            orientations or ['x_0', 'x_90'], 'SCH', 'bounding_box', 1,
            True, 'minimum_aabb_volume', None, (1, 1, 1), 'cube', iterations,
            budget, 1, False, False, False, restart_limit=restarts,
            no_improvement_limit=stall_limit, ls_evaluation_limit=total_cap)
        return output, baselines

    def test_accepts_local_improvement_below_global_threshold(self):
        output, baselines = self.run_search([3, 5], [3, 4, 2, 2])
        self.assertEqual([s for s, _ in baselines], [3, 5, 4, 2])
        self.assertEqual(baselines[2][1], 'x_90')
        self.assertEqual(output[0], 2)
        self.assertEqual(output[3], 3)
        self.assertEqual(len(output), 11)
        self.assertEqual(output[7][0][0]['score'], 3)
        self.assertEqual([s[0] for s in self.selections], [True, True, False, False])

    def test_rejected_candidate_preserves_exclusion_and_changes_stage(self):
        self.run_search([3], [3, 2, 2], iterations=3, stall_limit=3, restarts=1, total_cap=3)
        self.assertEqual([s[0] for s in self.selections], [True, True, False])
        self.assertAlmostEqual(self.selections[1][1], 2 / 3)
        self.assertEqual(self.selections[2][1:], (1, 'x_90', 0))

    def test_restart_reenables_exclusion(self):
        self.run_search([5, 4], [3, 3, 2, 2], iterations=1)
        self.assertEqual([s[0] for s in self.selections], [True, False, True, False])

    def test_real_selector_allows_last_bin_when_exclusion_disabled(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'improve_pieces_selection_ls']
        scope = dict(np=np)
        exec(compile(tree, str(SOURCE), 'exec'), scope)
        select = scope['improve_pieces_selection_ls']
        objects = [{'bin_position': 0}, {'bin_position': 1}]
        self.assertEqual(select(objects, 1, 1, 2, deterministic=True), 0)
        self.assertEqual(select(objects, 1, 1, 2, deterministic=True,
                                exclude_last_bin=False), 1)

    def test_keeps_completed_candidate_at_deadline(self):
        output, _ = self.run_search([2], [1], budget=2)
        self.assertEqual(output[0], 1)

    def test_large_initial_objective(self):
        output, _ = self.run_search([1000], budget=1)
        self.assertEqual(output[0], 1000)

    def test_empty_neighborhood_terminates(self):
        output, baselines = self.run_search([3, 2], orientations=['x_0'])
        self.assertEqual(baselines, [])
        self.assertEqual(output[0], 2)

    def test_repacking_limit_per_restart(self):
        output, baselines = self.run_search([5, 4], [3, 2], iterations=1, total_cap=1)
        self.assertEqual(len(baselines), 2)
        self.assertEqual(output[0], 2)

    def test_improvements_reset_stall_and_allow_more_than_threshold(self):
        output, baselines = self.run_search([5], [5, 4, 4, 3, 3, 3],
                                           iterations=2, stall_limit=None, restarts=1)
        self.assertEqual(len(baselines), 6)
        self.assertEqual([s for s, _ in baselines], [5,5,4,4,3,3])
        self.assertEqual(output[0], 3)

    def test_one_hundred_rejections_after_improvement(self):
        output, baselines = self.run_search([5], [5]*99+[4]+[4]*100,
                                           iterations=100, stall_limit=None, restarts=1,budget=1000)
        self.assertEqual(len(baselines),200)
        self.assertEqual(output[0],4)

    def test_restart_after_consecutive_failure_threshold(self):
        _, baselines = self.run_search([5,6], [5,5,6,6],
                                      iterations=2,stall_limit=None,restarts=2)
        self.assertEqual([s for s,_ in baselines],[5,5,6,6])


if __name__ == '__main__':
    unittest.main()
