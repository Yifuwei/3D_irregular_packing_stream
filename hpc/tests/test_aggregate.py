import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect import aggregate


class AggregateTests(unittest.TestCase):
    def rows(self):
        return [dict(dataset="chess", bin_shape="cube", bin_size="56x56x56",
                     nesting_strategy="minimum_aabb_volume", selection_range="bottom",
                     constraint_mode="geometry_only", sequence_seed=seed,
                     status="success", validation_ok=True, best_N=i+1)
                for i, seed in enumerate((3, 7, 13, 19, 53))]

    def test_five_seeds_mean_and_sample_std(self):
        result = aggregate(self.rows())[0]
        self.assertTrue(result["complete"])
        self.assertEqual(result["best_N_n"], 5)
        self.assertEqual(result["best_N_mean"], 3)
        self.assertAlmostEqual(result["best_N_std"], 2.5**0.5)

    def test_invalid_missing_and_null_not_counted_as_zero(self):
        rows = self.rows()
        for row in rows[:4]:
            row["status"] = "invalid"
        result = aggregate(rows)[0]
        self.assertFalse(result["complete"])
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["best_N_mean"], 5)
        self.assertIsNone(result["best_N_std"])
        self.assertEqual(result["best_U_star_n"], 0)
        self.assertIsNone(result["best_U_star_mean"])

    def test_strategies_and_shapes_stay_separate(self):
        rows = self.rows()
        rows += [dict(row, bin_shape="cylinder") for row in self.rows()]
        rows += [dict(row, selection_range="all") for row in self.rows()]
        self.assertEqual(len(aggregate(rows)), 3)

    def test_duplicate_seed_is_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            aggregate(rows + [rows[0]])
