"""CPU tests of the position RCL and its real construction call path."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from test_sch_bounds import load_functions


SCH = load_functions('SCH_iter_ls.py', {
    'select_pes_rcl', 'selection_process_bottom_top_extrema',
    'selection_process_bottom_top_only', 'nesting_evaluation',
    'SC_heuristic', 'get_bounding_box', 'find_max_xyz', 'get_intersection',
})


class PositionRclTests(unittest.TestCase):
    def test_observation_does_not_change_draws_or_rng_progression(self):
        pairs = [((i, 0, 0), 100+i) for i in range(10)]
        baseline_rng = np.random.default_rng(202)
        observed_rng = np.random.default_rng(202)
        baseline = [SCH['select_pes_rcl'](pairs, .1, rng=baseline_rng) for _ in range(20)]
        events = []
        observed = [SCH['select_pes_rcl'](pairs, .1, rng=observed_rng,
                                        observer=events.append) for _ in range(20)]
        self.assertEqual(baseline, observed)
        self.assertEqual(baseline_rng.integers(100000), observed_rng.integers(100000))
        self.assertEqual(len(events), 20)
        self.assertTrue(all(e['active_rcl_size'] == 10 for e in events))
        self.assertTrue(all(len(e['candidates']) == 10 for e in events))

    def test_observation_reports_unique_candidates_and_empty_rcls(self):
        events = []
        SCH['select_pes_rcl']([((0, 0, 0), 100), ((0, 0, 0), 101),
                               ((1, 0, 0), 111)], .1,
                              rng=np.random.default_rng(202), observer=events.append)
        self.assertEqual(events[0]['best_pes'], 100)
        self.assertEqual(events[0]['active_rcl_size'], 1)
        self.assertEqual(len(events[0]['candidates']), 2)
        SCH['select_pes_rcl']([], .1, observer=events.append)
        self.assertEqual(events[1]['active_rcl_size'], 0)
        self.assertIsNone(events[1]['best_pes'])

    def test_relative_threshold_positive_negative_and_zero(self):
        for scores, eligible in (([100, 110, 111], 2),
                                 ([-100, -90, -89], 2), ([0, 0, 1], 2)):
            pairs = [((i, 0, 0), score) for i, score in enumerate(scores)]
            for index in range(eligible):
                with patch.object(np.random, 'randint', return_value=index) as draw:
                    score, position = SCH['select_pes_rcl'](pairs, .10)
                    draw.assert_called_once_with(eligible)
                    self.assertEqual((score, position), (scores[index], (index, 0, 0)))

    def test_duplicates_empty_and_invalid_tolerance(self):
        with patch.object(np.random, 'randint', return_value=0) as draw:
            result = SCH['select_pes_rcl']([((0, 0, 0), 10)] * 3, .1)
            draw.assert_called_once_with(1)
            self.assertEqual(result, (10, (0, 0, 0)))
        self.assertIs(SCH['select_pes_rcl']([], .1), False)
        for tolerance in (-.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                SCH['select_pes_rcl']([((0, 0, 0), 1)], tolerance)

    def test_extrema_are_feasible_and_at_most_sixteen_even_for_small_slices(self):
        region = np.zeros((6, 6, 4), dtype=np.uint8)
        region[:5, :5, 0] = region[:5, :5, 3] = 1
        points = SCH['selection_process_bottom_top_extrema'](region)
        self.assertEqual(len(points), 8)  # Four distinct rectangle corners per face.
        self.assertTrue(all(region[p] == 1 for p in points))
        region[:, :, 3] = 0
        points = SCH['selection_process_bottom_top_extrema'](region)
        self.assertEqual(len(points), 4)  # top == bottom does not duplicate points.
        self.assertEqual(SCH['selection_process_bottom_top_extrema'](np.zeros_like(region)), [])

    def test_sch_evaluates_real_pes_and_can_sample_a_nonbest_position(self):
        region = np.zeros((6, 6, 2), dtype=np.uint8)
        region[:2, :2, 0] = 1
        piece = np.zeros_like(region)
        piece[0, 0, 0] = 1
        SCH['get_feasible_boundary_rigorous_acces'] = lambda *args: region
        args = (None, None, {'array': piece}, [[]], [np.zeros_like(piece)], 0,
                piece.shape, 'minimum_aabb_edges_len', 5, 'z', 'cube',
                'bounding_box', True, True, 'bottom_top', False, False)
        # AABB edge score stays 12; origin-distance terms give 12, 13, 13, 14.
        self.assertEqual(SCH['SC_heuristic'](*args, rcl_tolerance=.05), (12, (0, 0, 0)))
        # At 10%, the two score-13 points are also eligible.
        with patch.object(np.random, 'randint', return_value=1) as draw:
            score, position = SCH['SC_heuristic'](*args, rcl_tolerance=.1)
            draw.assert_called_once_with(3)
            self.assertEqual(score, 13)
            self.assertEqual(region[position], 1)
        self.assertEqual(SCH['SC_heuristic'](*args), (12, (0, 0, 0)))

    def test_seed_reproducibility(self):
        state = np.random.get_state()
        pairs = [((i, 0, 0), 100 + i) for i in range(10)]
        try:
            np.random.seed(17)
            first = [SCH['select_pes_rcl'](pairs, .1) for _ in range(20)]
            np.random.seed(17)
            self.assertEqual(first, [SCH['select_pes_rcl'](pairs, .1) for _ in range(20)])
            self.assertGreater(len(set(first)), 1)
        finally:
            np.random.set_state(state)

    def test_construction_forwards_position_rcl_and_compares_orientations_by_pes(self):
        calls = []
        def search(*args, **kwargs):
            calls.append(kwargs['rcl_tolerance'])
            return (5, (1, 0, 0)) if args[2]['orientation'] == 'x_90' else (10, (2, 0, 0))
        scope = load_functions('packing_iter_ls.py', {'packing_3D_voxel_lookback'}, {
            'trace': lambda msg: None, 'SC_heuristic': search,
            'rotate_voxel': lambda a, *args: a.copy(),
            'translate_voxel': lambda a, shift: np.roll(a, shift, (0, 1, 2)),
            'get_volume': np.count_nonzero,
            'score_position_pes': lambda *args: (1, (0, 0, 0)),
        })
        array = np.zeros((4, 4, 4), dtype=np.uint8)
        array[0, 0, 0] = 1
        objects = [dict(array=array.copy(), volume=1, radio=0, orientation='x_0')
                   for _ in range(2)]
        # No second random draw among orientations is allowed.
        with patch.object(np.random, 'randint', side_effect=AssertionError('orientation RCL used')):
            result = scope['packing_3D_voxel_lookback'](
                copy.deepcopy(objects), None, None, ([0, 90], ['x']), array.shape,
                'cube', 1, 100, 'SCH', 'minimum_aabb_volume', 'minimum_aabb_volume',
                5, 'z', 'bounding_box', True, True, 'bottom_top', True, 5, False, False,
                unified_pes=True, rcl_tolerance=.1)
        self.assertEqual(calls, [.1, .1])
        self.assertEqual(result[0][0][1]['orientation'], 'x_90')
        self.assertEqual(result[0][0][1]['translation'], (1, 0, 0))


if __name__ == '__main__':
    unittest.main()
