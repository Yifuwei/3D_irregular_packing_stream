"""Test hybrid search state transitions without CUDA geometry dependencies."""
import ast
import copy
import math
from pathlib import Path
import types
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'new_ILS_southampton.py'


class ImprovedGraspIlsTests(unittest.TestCase):
    def run_search(self, constructions, moves=(), kicks=(), iterations=4,
                   trigger=1, restarts=1, budget=100, **options):
        piece_count = options.pop('piece_count', 2)
        clock = [0]
        constructions, moves, kicks = map(iter, (constructions, moves, kicks))
        self.events = []
        self.kick_orientations = []

        def result(score, orientations):
            pieces = [dict(score=score, orientation=o, bin_position=0 if i==0 else 1,
                           array=np.ones((1,1,1))) for i,o in enumerate(orientations)]
            return [pieces[:1], pieces[1:]], [np.ones((1,1,1)) for _ in range(2)], [], [[0], list(range(1,piece_count))]

        def construct(*args, **kwargs):
            self.assertTrue(kwargs['random_CA'])
            self.assertTrue(kwargs['unified_pes'])
            self.assertEqual(kwargs['rcl_tolerance'], .10)
            self.assertEqual(args[9], args[10])
            clock[0] += 1
            score = next(constructions)
            self.events.append(('construct', score))
            return result(score, ['x_0'] * piece_count)

        def select(info, iteration, stage, bins, **kwargs):
            self.assertGreaterEqual(stage, 0)
            self.events.append(('select', kwargs['exclude_last_bin'], iteration))
            return 0

        def repack(*args, **kwargs):
            state = args[5][0]
            self.events.append(('move', state['score'], args[4][0]))
            self.assertEqual(kwargs['_type'], 'bounding_box')
            self.assertNotIn('rcl_tolerance', kwargs)
            self.assertEqual(args[13], args[14])
            orientations = list(args[4])
            orientations[args[1][0]] = args[1][1]
            # Mutating the engine's input must not corrupt original/best states.
            state['bin_real_layout'][0][0]['score'] = -100
            clock[0] += 1
            return result(next(moves), orientations)

        def kick(*args, **kwargs):
            self.kick_orientations.append(list(args[4]))
            self.events.append(('kick',))
            clock[0] += 1
            return result(next(kicks), args[4])

        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'improved_GRASP_ILS']
        self.assertEqual(len(tree.body), 1)
        scope = dict(copy=copy, math=math, np=np,
                     time=types.SimpleNamespace(perf_counter=lambda: clock[0]),
                     trace=lambda message: None, packing_3D_voxel_lookback=construct,
                     improved_repack=repack, kick_repacking=kick,
                     improve_pieces_selection_ls=select,
                     get_final_performance=lambda o, c, s, layout:
                         (2, 0, 2 - layout[0][0]['score']))
        exec(compile(tree, str(SOURCE), 'exec'), scope)
        orientation_labels = options.pop('orientation_labels', ['x_0', 'x_90'])
        output = scope['improved_GRASP_ILS'](
            [{} for _ in range(piece_count)], None, None, 0, 0, None, None, orientation_labels,
            'SCH', 'bounding_box', 1, True, 'minimum_aabb_volume', None,
            (1, 1, 1), 'cube', iterations, trigger, 'small', budget, 1,
            False, False, False, restart_limit=restarts, kick_seed=7, **options)
        return output

    def test_worse_kick_is_accepted_and_descent_continues(self):
        output = self.run_search([.5], [.6, .7, .4], [.8], iterations=3)
        baselines = [event[1] for event in self.events if event[0] == 'move']
        self.assertEqual(baselines, [.5, .8, .7])
        self.assertAlmostEqual(output[2], 1.6)
        self.assertEqual(output[7][0][0]['score'], .5)
        self.assertEqual(len(output), 11)
        selections = [event for event in self.events if event[0] == 'select']
        self.assertEqual([event[1] for event in selections], [True, False, False])
        self.assertEqual(selections[1][2], 0)

    def test_kicks_do_not_count_towards_restart_budget(self):
        self.run_search([.5, .6], [.7, .8, .9, 1.0], [.5, .5, .6, .6], iterations=2, restarts=2)
        operations = [event[0] for event in self.events if event[0] != 'select']
        self.assertEqual(operations, ['construct', 'move', 'kick', 'move'] * 2)
        selections = [event for event in self.events if event[0] == 'select']
        self.assertEqual([event[1] for event in selections], [True, False, True, False])

    def test_budget_end_reconstructs_without_extra_kick(self):
        output=self.run_search([.5],[.6],iterations=1)
        self.assertEqual(self.kick_orientations,[])
        self.assertAlmostEqual(output[2],1.5)

    def test_kick_uses_restart_best_and_changes_selected_orientation(self):
        self.run_search([.5], [.6, .9, 1.1], [.8, 1.0], iterations=3)
        # Both kicks must perturb the retained .5 solution (all x_0).
        self.assertEqual([sum(o != 'x_0' for o in os) for os in self.kick_orientations], [1, 1])

    def test_best_survives_worse_kick_and_restart(self):
        output = self.run_search([.5, .7], [.6, 1.0, .8, 1.1], [.9, 1.0, 1.0, 1.2],
                                 iterations=2, restarts=2)
        self.assertAlmostEqual(output[2], 1.5)
        self.assertEqual(output[6][0][0]['score'], .5)

    def test_finished_kick_evaluated_at_deadline(self):
        output = self.run_search([.5], [.6], [.3], budget=3, restarts=2)
        self.assertAlmostEqual(output[2], 1.7)
        self.assertEqual(output[-1], 3)

    def test_no_kick_started_at_deadline(self):
        self.run_search([.5], [.6], budget=2, restarts=2)
        self.assertEqual(self.kick_orientations, [])

    def test_disabled_kicks_continue_until_iteration_limit(self):
        self.run_search([.5], [.6, .7, .8], iterations=3, trigger=None)
        self.assertEqual(self.kick_orientations, [])
        self.assertEqual(sum(event[0] == 'move' for event in self.events), 3)

    def test_default_n_budget_counts_repacking_not_piece_selection(self):
        self.run_search([.5], [.6, .6], iterations=None, trigger=None,
                        orientation_labels=['x_0', 'x_90', 'x_180', 'y_90'])
        self.assertEqual(sum(e[0]=='move' for e in self.events), 2)
        self.assertEqual(sum(e[0]=='select' for e in self.events), 1)
        self.assertEqual(self.kick_orientations, [])

    def test_auto_kick_counts_failed_repackings_mid_piece(self):
        self.run_search([.5], [.6, .7], [.8], iterations=None, trigger='auto',
                        orientation_labels=['x_0', 'x_90', 'x_180'])
        self.assertEqual([e[0] for e in self.events],
                         ['construct','select','move','kick','select','move'])
        self.assertEqual(len(self.kick_orientations), 1)

    def test_improvements_do_not_reset_n_repacking_budget(self):
        self.run_search([.5], [.4, .3], iterations=None, trigger='auto')
        self.assertEqual(sum(e[0]=='move' for e in self.events), 2)
        self.assertEqual(self.kick_orientations, [])

    def test_odd_n_rounds_up_kick_threshold_and_preserves_n_budget(self):
        self.run_search([.5], [.6]*5, [.8], iterations=None, trigger='auto',
                        piece_count=5, orientation_labels=['x_0','x_90','x_180'])
        operations=[e[0] for e in self.events if e[0]!='select']
        self.assertEqual(operations, ['construct','move','move','move','kick','move','move'])

    def test_improvement_resets_repacking_failure_counter(self):
        self.run_search([.5], [.6,.4,.5,.6,.7], iterations=None,
                        trigger='auto', piece_count=5)
        self.assertEqual(sum(e[0]=='move' for e in self.events),5)
        self.assertEqual(self.kick_orientations,[])

    def test_invalid_iteration_budget(self):
        with self.assertRaises(ValueError):
            self.run_search([.5], iterations=0)

    def test_explicit_budgets_still_stop_exactly_at_ls_boundary(self):
        for threshold, expected_kicks in ((100, 2), (125, 2), (150, 1)):
            with self.subTest(threshold=threshold):
                self.run_search([.5], [.6] * 300, [.5] * expected_kicks,
                                iterations=300, trigger=threshold, budget=1000)
                self.assertEqual(sum(event[0] == 'move' for event in self.events), 300)
                self.assertEqual(len(self.kick_orientations), expected_kicks)


if __name__ == '__main__':
    unittest.main()
