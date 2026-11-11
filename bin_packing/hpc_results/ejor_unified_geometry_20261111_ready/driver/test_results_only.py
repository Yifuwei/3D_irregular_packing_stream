import ast
from pathlib import Path
import unittest

class ResultsOnlyTests(unittest.TestCase):
    def test_worker_has_no_unbounded_event_log(self):
        tree=ast.parse(Path(__file__).with_name('worker.py').read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        names={n.id for n in ast.walk(run) if isinstance(n,ast.Name)}
        self.assertNotIn('event_file',names)
        self.assertNotIn('history',names)
        self.assertNotIn('ls_end_events',names)
        self.assertIn('ls_stop_counts',names)
        self.assertIn('rcl_counts',names)
