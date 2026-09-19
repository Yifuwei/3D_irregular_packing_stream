"""Read a parameter row safely; do not evaluate it as shell code."""
import argparse
from pathlib import Path
import shlex
import sys

from run import main as run_one


def read_task(path, index):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if not 0 <= index < len(lines) or not lines[index].strip():
        raise ValueError(f"Missing/empty task row {index} in {path}")
    return shlex.split(lines[index])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("params")
    p.add_argument("index", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    command = read_task(args.params, args.index)
    if args.dry_run:
        command.append("--dry-run")
    return run_one(command)


if __name__ == "__main__":
    sys.exit(main())
