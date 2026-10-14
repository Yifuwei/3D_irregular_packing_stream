import importlib.util
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('resume',Path(__file__).with_name('resume_iterations.py'))
resume=importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class ResumeTests(unittest.TestCase):
    def test_sparse_array(self):
        self.assertEqual(resume.array_spec([9,2,3,4,7]),'2-4,7,9')
        self.assertEqual(resume.array_spec([]),'')

    def test_success_skipped_and_partial_files_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            tasks=[dict(task_id=i,dataset='sample',seed=3) for i in range(4)]
            (root/'manifest.json').write_text(json.dumps(dict(tasks=tasks,time_limit=3600)))
            (root/'task_0.json').write_text(json.dumps(dict(tasks[0],status='success',validation_ok=True,settings={'time_limit':3600})))
            (root/'task_1.json').write_text(json.dumps(dict(tasks[1],status='failed')))
            (root/'task_2_events.jsonl').write_text('partial')
            (root/'task_3.json').write_text(json.dumps(dict(tasks[3],status='success',validation_ok=True,settings={'time_limit':30})))
            _,done,pending,archive=resume.plan(root)
            self.assertEqual(done,[0])
            self.assertEqual(pending,[1,2,3])
            self.assertEqual({p.name for p in archive},{'task_1.json','task_2_events.jsonl','task_3.json'})

    def test_source_check_accepts_line_endings_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'source.py'
            expected=hashlib.sha256(b'x=1\n').hexdigest()
            manifest={'source_hashes':{'source.py':expected}}
            path.write_bytes(b'x=1\r\n')
            self.assertEqual(resume.check_sources(root,manifest)['source.py'],hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(b'x=2\n')
            with self.assertRaises(ValueError):
                resume.check_sources(root,manifest)


if __name__=='__main__':
    unittest.main()
