"""Search RNG isolation, progression, and reproducibility regressions."""
import ast
from pathlib import Path
import unittest
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'


def load(filename, names, scope):
    tree = ast.parse((SRC / filename).read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(tree, filename, 'exec'), scope)
    return scope


class SearchRngTests(unittest.TestCase):
    def sequences(self, legacy_seed, extra_position_draws=0):
        selector = load('new_ILS_southampton.py', {'improve_pieces_selection_ls'},
                        dict(np=np, get_aabb_length=lambda array: array))['improve_pieces_selection_ls']
        rcl = load('SCH_iter_ls.py', {'select_pes_rcl'}, dict(np=np))['select_pes_rcl']
        objects = [dict(array=i+1, bin_position=i % 2) for i in range(6)]
        pairs = [((i, 0, 0), 100+i) for i in range(10)]
        np.random.seed(legacy_seed)
        piece_rng, position_rng = np.random.default_rng(101), np.random.default_rng(202)
        pieces, positions = [], []
        for i in range(40):
            position_rng.random(extra_position_draws)
            pieces.append(selector(objects, i, .5, 2, exclude_last_bin=False, rng=piece_rng))
            positions.append(rcl(pairs, .1, rng=position_rng)[1])
            np.random.random(10)  # Other random activity cannot perturb either stream.
        return pieces, positions

    def test_order_seed_does_not_change_search_streams_for_identical_inputs(self):
        self.assertEqual(self.sequences(3), self.sequences(13))

    def test_extra_position_draws_do_not_change_piece_selection(self):
        baseline = self.sequences(3)
        changed = self.sequences(3, extra_position_draws=11)
        self.assertEqual(baseline[0], changed[0])
        self.assertNotEqual(baseline[1], changed[1])

    def test_streams_advance_instead_of_reseeding_per_selection(self):
        pieces, positions = self.sequences(3)
        self.assertGreater(len(set(pieces)), 1)
        self.assertGreater(len(set(positions)), 1)

    def test_passing_same_generator_to_rcl_matches_successive_draws(self):
        rcl = load('SCH_iter_ls.py', {'select_pes_rcl'}, dict(np=np))['select_pes_rcl']
        rng = np.random.default_rng(202)
        expected = np.random.default_rng(202).integers(5, size=20).tolist()
        pairs = [((i, 0, 0), 100) for i in range(5)]
        actual = [rcl(pairs, .1, rng=rng)[1][0] for _ in range(20)]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
