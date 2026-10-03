"""CPU checks for incremental bin search and restoration."""
import ast
from pathlib import Path
import unittest
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'


def load_repack(search):
    # Load the real repacking logic without optional CUDA/UI dependencies.
    tree = ast.parse((SRC / 'packing_iter_ls.py').read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == 'improved_repack']
    # Substitute only the geometry search dependency, including empty bins.
    tree.body[0].body = [node for node in tree.body[0].body
                         if not isinstance(node, ast.FunctionDef)
                         or node.name != 'search']
    scope = dict(np=np, search=search,
                 rotate_voxel=lambda a, d, ax: a.copy(),
                 translate_voxel=lambda a, pos: np.roll(a, pos[0], axis=0))
    exec(compile(tree, 'packing_iter_ls.py', 'exec'), scope)
    return scope['improved_repack']


class ChangedBinsTests(unittest.TestCase):
    def run_case(self, placements):
        original, old = [], [[], [], []]
        order = [[0, 3], [1, 4], [2, 5]]
        for index in range(6):
            array = np.zeros((8, 1, 1))
            array[0, 0, 0] = 1
            info = dict(array=array, orientation='x_0', volume=1,
                        radio=1, piece_type=index, translation=(0, 0, 0))
            original.append(info)
        for b, indices in enumerate(order):
            for j, index in enumerate(indices):
                info = dict(original[index], bin_position=b,
                            translation=(j, 0, 0))
                info['array'] = np.roll(info['array'], j, axis=0)
                old[b].append(info)
        snapshot = dict(bin_real_layout=old, pieces_order=order)
        calls = []

        def search(info, layouts, occupancy, b):
            index = info['piece_type']
            calls.append((index, b))
            if placements.get(index) == b:
                return 0, (len(layouts[b]), 0, 0)
            return False

        result = load_repack(search)(
            original, (2, 'x_90'), None, None, ['x_0'] * 6,
            {'best': snapshot}, 'best', None, (8, 1, 1),
            'cube', 1, 100, None, None, None, 5, 'z',
            'bounding_box', False, True, None, False, False)
        self.assertEqual(sorted(i for bin_ids in result[3] for i in bin_ids),
                         list(range(6)))
        self.assertTrue(all(np.max(a) <= 1 for a in result[1]))
        self.assertEqual(order, [[0, 3], [1, 4], [2, 5]])
        self.assertEqual([len(b) for b in old], [2, 2, 2])
        return calls, result, old

    def test_failed_p_attempts_leave_earlier_bins_restorable(self):
        calls, result, old = self.run_case({2: 2, 5: 2})
        self.assertEqual(calls, [(2, 0), (2, 1), (2, 2), (5, 2)])
        for b in (0, 1):
            np.testing.assert_array_equal(result[0][b][1]['array'], old[b][1]['array'])
            self.assertEqual(result[0][b][1]['translation'], (1, 0, 0))

    def test_arrival_and_departure_propagate_changed_flags(self):
        calls, result, _ = self.run_case({2: 0, 3: 2, 5: 0})
        self.assertEqual(calls, [(2, 0), (3, 0), (3, 1), (3, 2), (4, 0), (5, 0)])
        self.assertEqual(result[3], [[0, 2, 5], [1, 4], [3]])

    def test_unchanged_later_bin_receives_piece_and_becomes_changed(self):
        calls, result, _ = self.run_case({2: 0, 3: 1, 4: 1, 5: 2})
        self.assertEqual(calls, [(2, 0), (3, 0), (3, 1),
                                 (4, 0), (4, 1), (5, 0), (5, 1), (5, 2)])
        self.assertEqual(result[3], [[0, 2], [1, 3, 4], [5]])

    def test_failed_later_search_leaves_original_bin_restorable(self):
        calls, result, _ = self.run_case({2: 0, 3: 2, 5: 2})
        self.assertIn((3, 1), calls)
        self.assertNotIn((4, 1), calls)
        self.assertEqual(result[0][1][1]['translation'], (1, 0, 0))
        # Failure in a previously changed bin does not clear its True flag.
        self.assertIn((5, 0), calls)

    def test_new_bin_is_changed_and_tracked(self):
        calls, result, _ = self.run_case({2: 3, 5: 3})
        self.assertIn((5, 3), calls)
        self.assertEqual(result[3], [[0, 3], [1, 4], [2, 5]])
        self.assertTrue(all(info['bin_position'] == b
                            for b, layout in enumerate(result[0])
                            for info in layout))


if __name__ == '__main__':
    unittest.main()
