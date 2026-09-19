"""Exercise array chunking with a fake sbatch; never contacts Slurm."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SubmitTests(unittest.TestCase):
    def test_chunk_offsets_and_dependency_keep_global_concurrency(self):
        bash = shutil.which("bash")
        if not bash and Path("C:/Program Files/Git/bin/bash.exe").exists():
            bash = "C:/Program Files/Git/bin/bash.exe"
        if not bash:
            self.skipTest("Bash is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "hpc").mkdir()
            shutil.copyfile(Path(__file__).resolve().parents[1] / "submit.sh", root / "hpc/submit.sh")
            (root / "params.txt").write_text("task\n" * 5, encoding="utf-8", newline="\n")
            script = '''
export TEST_DIR="$PWD"
export PATH="/usr/bin:/bin:$PATH"
sbatch() {
    count=0
    if [[ -f "$TEST_DIR/count" ]]; then read -r count < "$TEST_DIR/count"; fi
    count=$((count+1))
    echo "$count" > "$TEST_DIR/count"
    printf '%s\\n' "$*" >> "$TEST_DIR/calls"
    echo "$((100+count))"
}
export -f sbatch
HPC_ARRAY_SIZE=2 bash hpc/submit.sh params.txt 8
'''
            result = subprocess.run([bash, "-c", script], cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(len(calls), 3)
            self.assertIn("--array=0-1%8", calls[0])
            self.assertIn("--dependency=afterany:101", calls[1])
            self.assertIn("--dependency=afterany:102", calls[2])
            self.assertIn("--array=0-0%8", calls[2])
            self.assertEqual([line.split()[-1] for line in calls], ["0", "2", "4"])


if __name__ == "__main__":
    unittest.main()
