import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('collector', Path(__file__).with_name('collect_results.py'))
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def write(self, root, name, rows):
        fields = ['dataset', 'repeat', 'seed', 'status', 'original_seconds', 'improved_seconds',
                  'original_evaluations', 'improved_evaluations']
        with (root / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, repeat=1, old=100, new=80, status='passed', budget=100):
        return dict(dataset='chess', repeat=repeat, seed=13, status=status,
                    original_seconds=old, improved_seconds=new,
                    original_evaluations=100, improved_evaluations=budget)

    def test_paired_medians_failures_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, 'task_0.csv', [self.row(), self.row(2, 120, 100), self.row(3, status='failed')])
            result = collector.collect(root, ['chess', 'engine'])
            chess, engine = result
            self.assertEqual(chess['valid_pairs'], 2)
            self.assertEqual(chess['failed_pairs'], 1)
            self.assertEqual(chess['original_median_s'], 110)
            self.assertEqual(chess['improved_median_s'], 90)
            self.assertAlmostEqual(chess['paired_reduction_median_pct'], 18.3333333333333)
            self.assertEqual(engine['status'], 'missing')
            self.assertTrue((root / 'report.md').exists())

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, 'task_0.csv', [self.row()])
            self.write(root, 'task_1.csv', [self.row()])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                collector.collect(root)

    def test_unequal_budget_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, 'task_0.csv', [self.row(budget=99)])
            with self.assertRaisesRegex(ValueError, 'Unequal'):
                collector.collect(root)

    def test_empty_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'No task'):
                collector.collect(Path(directory))

    def test_expected_datasets_without_results(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = collector.collect(Path(directory), ['engine'])
            self.assertEqual(rows[0]['status'], 'missing')
            self.assertTrue((Path(directory) / 'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
