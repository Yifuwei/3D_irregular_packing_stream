import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('multiseed', Path(__file__).with_name('multiseed_ils.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MultiSeedTests(unittest.TestCase):
    def fixtures(self, root, seeds=(3, 7, 13, 19, 53)):
        (root / 'src').mkdir()
        (root / 'src' / 'algorithm.py').write_text('version=1')
        (root / 'instances').mkdir()
        for seed in seeds:
            for dataset in ['chess', 'engine']:
                (root / 'instances' / f'{dataset}_seq{seed}_cube_999999_1.txt').write_text('instance')

    def test_paired_seeds_and_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            manifest = module.prepare(root, [3, 7, 13, 19, 53])
            self.assertEqual(len(manifest['tasks']), 10)
            self.assertEqual(manifest['evaluations'], 100)
            self.assertTrue(all(t['sequence_seed'] == t['ls_seed'] for t in manifest['tasks']))

    def test_missing_instance_and_duplicate_seed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            with self.assertRaises(ValueError):
                module.prepare(root, [3, 3])
            (root / 'instances' / 'engine_seq7_cube_999999_1.txt').unlink()
            with self.assertRaises(ValueError):
                module.prepare(root, [3, 7])

    def test_source_change_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            manifest = module.prepare(root, [3])
            (root / 'src' / 'algorithm.py').write_text('version=2')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                module.run(root, manifest, 0, root)

    def test_worker_receives_both_seeds_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            manifest = module.prepare(root, [7])
            target = root / 'task_1_runs'
            target.mkdir()
            (target / 'comparison.csv').write_text('dataset,seed,ls_seed\nengine,7,7\n')
            with patch.object(module.subprocess, 'run') as run:
                run.return_value.returncode = 0
                module.run(root, manifest, 1, root)
                command = run.call_args.args[0]
                for flag, value in [('--seed', '7'), ('--ls-seed', '7'), ('--evaluations', '100')]:
                    self.assertEqual(command[command.index(flag)+1], value)
            self.assertTrue((root / 'task_1.csv').exists())
