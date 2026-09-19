"""CPU regressions: load real function bodies without optional GPU/UI imports.

Run: python -m unittest discover -s bin_packing/test -p test_sch_bounds.py -v
These checks do not exercise CUDA NFV generation or a full packing run.
"""
import ast
import math
from pathlib import Path
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"


def load_functions(filename, names, namespace=None):
    scope = {"np": np, "math": math}
    scope.update(namespace or {})
    tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in names]
    for node in tree.body:
        node.decorator_list = []
    exec(compile(tree, filename, "exec"), scope)
    return scope


SCH = load_functions("SCH_iter_ls.py", {
    "nesting_evaluation", "get_bounding_box", "find_max_xyz",
    "get_intersection", "SC_heuristic", "selection_process",
    "selection_process_bottom_only", "selection_process_bottom_top_only",
    "selection_process_bottom_left_filling", "voxel_floor",
})
LIB = load_functions("function_lib.py", {
    "IFV_POOL", "aabb_rotate", "njit_get_ifv_cylinder", "max_along_axis_2",
})


class BoundsTests(unittest.TestCase):
    def setUp(self):
        self.piece = np.zeros((6, 6, 6), dtype=np.uint8)
        self.piece[:2, :2, :2] = 1
        self.fixed = np.zeros_like(self.piece)

    def evaluate(self, positions, nesting=1):
        return SCH["nesting_evaluation"](
            self.piece, self.fixed, [], positions, nesting, (6, 6, 6), False)

    def test_clipped_candidates_rejected(self):
        self.assertIs(self.evaluate([(5, 0, 0), (-1, 0, 0)]), False)
        score, position = self.evaluate([(5, 0, 0), (4, 0, 0)])
        self.assertEqual(position, (4, 0, 0))
        self.assertEqual(score, 8)

    def test_empty_and_overlap_rejected(self):
        self.assertIs(self.evaluate([]), False)
        self.fixed[0, 0, 0] = 1
        self.assertIs(self.evaluate([(0, 0, 0)]), False)

    def test_overlap_distance_uses_array_and_actual_centers(self):
        self.fixed[0, 0, 0] = 1
        score, position = self.evaluate([(4, 0, 0)], nesting=4)
        self.assertEqual(position, (4, 0, 0))
        self.assertEqual(score, 2)

    def test_all_selection_modes_return_score_then_coordinate(self):
        region = np.zeros_like(self.piece)
        region[1:3, 1:3, 1:3] = 1
        SCH["get_feasible_boundary_rigorous_acces"] = lambda *args: region
        for mode in ("all", "bottom", "bottom_top", "bottom_left_filling"):
            for strategy in ("minimum_aabb_volume", "minimum_aabb_edges_len",
                             "maximal_residual_box", "overlap_distance"):
                with self.subTest(mode=mode, strategy=strategy):
                    result = SCH["SC_heuristic"](
                        None, None, {"array": self.piece}, [[]], [self.fixed],
                        0, (6, 6, 6), strategy, 2, "z", "cube",
                        "bounding_box", True, False, mode, False, False)
                    self.assertIsInstance(result, tuple)
                    score, position = result
                    self.assertTrue(np.isscalar(score))
                    self.assertEqual(len(position), 3)
                    self.assertEqual(region[position], 1)

    def test_cube_ifv_exact_fit_oversize_and_last_legal_position(self):
        pool = LIB["IFV_POOL"]()
        for shape, count in (((6, 6, 6), 1), ((7, 2, 2), 0), ((2, 2, 2), 125)):
            info = {"aabb": shape, "orientation": "x_0"}
            region = pool.get_ifv_cube(info, (6, 6, 6))
            self.assertEqual(region.sum(), count)
        self.assertEqual(region[4, 4, 4], 1)
        self.assertEqual(region[5, 0, 0], 0)

    def test_all_axes_sample_only_valid_voxels_including_last_layer(self):
        region = np.zeros((4, 5, 6), dtype=np.uint8)
        region[1:, 2:, 3:] = 1
        for axis in ("x", "y", "z"):
            for density in (1, 2, 20):
                with self.subTest(axis=axis, density=density):
                    positions = SCH["selection_process"](region, density, axis, "bounding_box")
                    self.assertTrue(positions)
                    self.assertTrue(all(region[p] == 1 for p in positions))
                    index = ("x", "y", "z").index(axis)
                    self.assertEqual(max(p[index] for p in positions), region.shape[index]-1)

    def test_empty_feasible_region_returns_false(self):
        SCH["get_feasible_boundary_rigorous_acces"] = lambda *args: np.zeros_like(self.piece)
        self.assertIs(SCH["SC_heuristic"](
            None, None, {"array": self.piece}, [[]], [self.fixed], 0,
            (6, 6, 6), "minimum_aabb_volume", 2, "z", "cube",
            "bounding_box", True, False, "bottom", False, False), False)

    def test_ifv_cache_is_container_specific(self):
        pool = LIB["IFV_POOL"]()
        info = {"piece_type": 1, "aabb": (2, 2, 2), "orientation": "x_0",
                "array": self.piece}
        first = pool.retrieve_ifv(info, (6, 6, 6), "cube")
        second = pool.retrieve_ifv(info, (5, 5, 5), "cube")
        cylinder = pool.retrieve_ifv(info, (6, 6, 6), "cylinder")
        self.assertEqual(second.shape, (5, 5, 5))
        self.assertFalse(np.array_equal(first, cylinder))
        self.assertIs(first, pool.retrieve_ifv(info, (6, 6, 6), "cube"))

    def test_cylinder_includes_top_and_rejects_oversize_height(self):
        calc = LIB["njit_get_ifv_cylinder"]
        region = calc(self.piece, 2, 2, 2, (6, 6, 6))
        self.assertEqual(region[2, 2, 4], 1)
        self.assertEqual(region[0, 0, 0], 0)
        self.assertFalse(calc(self.piece, 2, 2, 7, (6, 6, 6)).any())


if __name__ == "__main__":
    unittest.main()
