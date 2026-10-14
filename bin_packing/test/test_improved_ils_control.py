"""CPU control-flow regressions; packing dependencies are scripted."""
import ast
import copy
import math
from pathlib import Path
import unittest
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'


class Clock:
    value = 0.0

    def perf_counter(self):
        self.value += 0.01
        return self.value


class ILSControlTests(unittest.TestCase):
    def run_ils(self, bins=None, plans=(), kicks=(), **limits):
        self.original = [dict(volume=v, radio=1, array=np.zeros((2, 2, 2)),
                              orientation='x_0', bin_position=None) for v in (4, 3, 2, 1)]
        self.selections, self.trials, self.kick_count = [], [], 0
        self.selection_flags = []
        self.kick_inputs = []
        plans, kicks = iter(plans), iter(kicks)
        clock = Clock()

        def packing(indices, orientations):
            layout = [[dict(self.original[i], bin_position=b, orientation=orientations[i])
                       for i in ids] for b, ids in enumerate(indices)]
            return layout, [np.zeros((2, 2, 2)) for _ in indices], [[1] * len(ids) for ids in indices], indices

        def construction(*args, **kwargs):
            # Deliberate mutation verifies that original geometry is isolated.
            args[0][0]['orientation'] = 'z_90'
            return packing(bins or [[0, 3], [1, 2]], ['x_0'] * 4)

        def select(objects, *args, **kwargs):
            self.selection_flags.append(kwargs['exclude_last_bin'])
            self.selections.append([x['bin_position'] for x in objects])
            return 0

        def repack(original, selected, nfv, ifv, orientations, pool, key, *args, **kwargs):
            self.assertEqual(original[0]['orientation'], 'x_0')
            self.trials.append(selected)
            indices, values, delay = next(plans)
            clock.value += delay
            return packing(indices, values)

        def kick(*args, **kwargs):
            self.kick_count += 1
            self.kick_inputs.append(list(args[4]))
            indices, values = next(kicks)
            return packing(indices, values)

        scope = dict(np=np, math=math, copy=copy, time=clock,
                     packing_3D_voxel_lookback=construction,
                     improved_repack=repack, kick_repacking=kick,
                     improve_pieces_selection_ls=select)
        tree = ast.parse((SRC / 'new_ILS_southampton.py').read_text(encoding='utf-8'))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                     n.name in {'trace', 'update_data', 'prune_data_pool', 'get_final_performance'}]
        exec(compile(tree, 'helpers', 'exec'), scope)
        revision = SRC / 'improved_ils_revision.py'
        tree = ast.parse(revision.read_text(encoding='utf-8') if revision.exists() else
                         (SRC / 'new_ILS_southampton.py').read_text(encoding='utf-8'))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'improved_ILS']
        exec(compile(tree, 'improved_ILS', 'exec'), scope)
        options = dict(iteration_limit=4, time_limit=None, kick_trigger_time=None)
        options.update(limits)
        return scope['improved_ILS'](
            self.original, None, None, 100, 1, None, None,
            ['x_0', 'x_90', 'x_180'], 'SCH', 'bounding_box', None,
            False, 1, None, 'ILS', (10, 1, 1), 'cube', **options)

    def test_rejected_trials_keep_initial_exclusion(self):
        original = [[0, 3], [1, 2]]
        self.run_ils(plans=[(original, ['x_0'] * 4, 0)] * 4)
        self.assertEqual(self.selection_flags, [True, True])

    def test_last_bin_becomes_selectable_when_exclusion_is_disabled(self):
        tree = ast.parse((SRC / 'new_ILS_southampton.py').read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'improve_pieces_selection_ls']
        scope = dict(np=np, get_aabb_length=lambda array: 1)
        exec(compile(tree, 'selector', 'exec'), scope)
        objects = [dict(bin_position=0, array=None), dict(bin_position=1, array=None)]
        selector = scope['improve_pieces_selection_ls']
        self.assertEqual(selector(objects, 1, 1, 2, deterministic=True,
                                  exclude_last_bin=True), 0)
        self.assertEqual(selector(objects, 1, 1, 2, deterministic=True,
                                  exclude_last_bin=False), 1)

    def test_unchanged_kick_keeps_initial_exclusion(self):
        original = [[0, 3], [1, 2]]
        self.run_ils(plans=[(original, ['x_0'] * 4, 0)] * 2,
                     kicks=[(original, ['x_0'] * 4)],
                     iteration_limit=3, kick_trigger_time=1, kick_level="small")
        self.assertEqual(self.selection_flags, [True, True])

    def test_independent_best_initial_and_accepted_selection(self):
        best = ['x_90', 'x_0', 'x_0', 'x_0']
        kicked = ['x_180', 'x_0', 'x_0', 'x_0']
        local = ['x_0'] * 4
        result = self.run_ils(
            plans=[([[0, 1, 2], [3]], best, 0),
                   ([[0, 3], [1, 2]], ['x_0'] * 4, 0),
                   ([[0, 1], [2, 3]], local, 0)],
            kicks=[([[0, 2], [1, 3]], kicked)],
            kick_trigger_time=1, kick_level='small')
        self.assertTrue(self.selection_flags[0])
        self.assertTrue(all(flag is False for flag in self.selection_flags[1:]))
        self.assertEqual(result[10], ['x_0'] * 4)
        self.assertEqual(result[11], best)
        self.assertEqual(result[9], [[0, 1, 2], [3]])
        self.assertEqual(self.selections, [[0, 1, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1]])
        self.assertEqual(self.trials[-1], (0, 'x_0'))
        self.assertEqual(self.kick_count, 1)

    def test_iteration_budget_counts_kick_without_overshoot(self):
        result = self.run_ils(
            plans=[([[0, 3], [1, 2]], ['x_90'] * 4, 0)],
            kicks=[([[0, 2], [1, 3]], ['x_180'] * 4)],
            iteration_limit=2, kick_trigger_time=1, kick_level='medium')
        self.assertEqual(len(self.trials), 1)
        self.assertEqual(self.kick_count, 1)
        self.assertEqual(result[13], [2])

    def test_kick_changes_exact_requested_number_of_labels(self):
        for level, expected in [('small', 1), ('medium', 2), ('large', 3)]:
            with self.subTest(level=level):
                self.run_ils(
                    plans=[([[0, 3], [1, 2]], ['x_90'] * 4, 0)],
                    kicks=[([[0, 2], [1, 3]], ['x_180'] * 4)],
                    iteration_limit=2, kick_trigger_time=1, kick_level=level)
                self.assertEqual(sum(value != 'x_0' for value in self.kick_inputs[0]), expected)

    def test_expired_time_still_records_completed_improvement(self):
        result = self.run_ils(plans=[([[0, 1, 2], [3]], ['x_90'] * 4, 2)], time_limit=1)
        self.assertEqual(result[9], [[0, 1, 2], [3]])
        self.assertGreater(result[-1], 2)
        self.assertEqual(len(self.trials), 1)

    def test_initial_and_discovered_one_bin_stop(self):
        result = self.run_ils(bins=[[0, 1, 2, 3]])
        self.assertEqual(result[0], 1)
        self.assertEqual(self.trials, [])
        result = self.run_ils(plans=[([[0, 1, 2, 3]], ['x_90'] * 4, 0)])
        self.assertEqual(result[0], 1)
        self.assertEqual(len(self.trials), 1)

    def test_zero_budget_returns_construction(self):
        result = self.run_ils(iteration_limit=0)
        self.assertEqual(self.trials, [])
        self.assertEqual(result[0], 2)

    def test_invalid_limits(self):
        for options in (dict(iteration_limit=-1), dict(time_limit=-1),
                        dict(kick_trigger_time=0),
                        dict(kick_trigger_time=1, kick_level='invalid')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_ils(**options)

    def test_selector_zero_weights_and_no_selectable_piece(self):
        tree = ast.parse((SRC / 'new_ILS_southampton.py').read_text(encoding='utf-8'))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'improve_pieces_selection_ls']
        scope = dict(np=np, get_aabb_length=lambda a: a)
        exec(compile(tree, 'selector', 'exec'), scope)
        selector = scope['improve_pieces_selection_ls']
        objects = [dict(bin_position=1, array=3), dict(bin_position=1, array=2),
                   dict(bin_position=0, array=1)]
        self.assertEqual(selector(objects, 0, 1, 2, seed=1), 2)
        with self.assertRaises(ValueError):
            selector([dict(bin_position=0, array=1)], 0, 1, 1)


if __name__ == '__main__':
    unittest.main()
