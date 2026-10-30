"""Cross-platform hashes accept only line-ending differences, not edits."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from bundle_integrity import verified_bytes,canonicalize_manifest
import test_suite as workflows
from worker import verify_runtime


class BundleIntegrityTests(unittest.TestCase):
    def test_legacy_crlf_hash_accepts_linux_lf_and_reverse(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'module.py'
            for original,checkout in ((b'x=1\r\n',b'x=1\n'),(b'x=1\n',b'x=1\r\n')):
                path.write_bytes(checkout)
                manifest=dict(files_sha256={'module.py':hashlib.sha256(original).hexdigest()})
                self.assertEqual(verified_bytes(root,manifest,'module.py'),b'x=1\n')
                canonicalize_manifest(root,manifest)
                self.assertEqual(manifest['files_sha256']['module.py'],hashlib.sha256(b'x=1\n').hexdigest())
            path.write_bytes(b'x=2\n')
            with self.assertRaisesRegex(ValueError,'Frozen runtime/input changed'):
                verified_bytes(root,manifest,'module.py')

    def test_binary_data_never_gets_text_normalization(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'object.binvox';path.write_bytes(b'voxels\r\n')
            manifest=dict(files_sha256={'object.binvox':hashlib.sha256(path.read_bytes()).hexdigest()})
            canonicalize_manifest(root,manifest)
            path.write_bytes(b'voxels\n')
            with self.assertRaisesRegex(ValueError,'Frozen runtime/input changed'):
                verified_bytes(root,manifest,'object.binvox')

    def test_entire_frozen_fixture_remains_valid_after_linux_checkout(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,manifest=workflows.FrozenWorkflowTests().fixture(Path(name))
            for relative in manifest['files_sha256']:
                path=bundle/relative
                if path.suffix in ('.py','.json','.txt','.csv','.md','.sh','.slurm'):
                    path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
            case=manifest['profile']['plans']['rcl'][0]
            verify_runtime(bundle,manifest,case)
            bad=bundle/'runtime/bin_packing/src/module.py';bad.write_text('x=999\n')
            with self.assertRaisesRegex(ValueError,'Frozen runtime/input changed'):
                verify_runtime(bundle,manifest,case)


if __name__=='__main__':unittest.main()
