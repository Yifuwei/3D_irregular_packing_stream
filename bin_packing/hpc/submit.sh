#!/usr/bin/env bash
# Run from an environment with Python; compute jobs need the configured CUDA environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EVALUATIONS="${EVALUATIONS:-100}"
export REPEATS="${REPEATS:-1}"
export SEED="${SEED:-13}"
export KICK_TRIGGER="${KICK_TRIGGER:-100}"
export RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/hpc_results/$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$RESULT_DIR/slurm"
export MANIFEST="$RESULT_DIR/submission_manifest.json"
count="$("${HPC_SUBMIT_PYTHON:-python3}" "$SCRIPT_DIR/benchmark_ils.py" --project "$PROJECT_DIR" --seed "$SEED" --write-manifest "$MANIFEST" --list-datasets | wc -l)"
if (( count < 1 )); then echo 'No dataset instances found.' >&2; exit 1; fi
MAX_PARALLEL="${MAX_PARALLEL:-8}"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid MAX_PARALLEL' >&2; exit 1; }
sbatch --array="0-$((count - 1))%$MAX_PARALLEL" --export=ALL --chdir="$PROJECT_DIR" \
  --output="$RESULT_DIR/slurm/%A_%a.out" --error="$RESULT_DIR/slurm/%A_%a.err" \
  "$@" "$SCRIPT_DIR/run_ils.slurm"
echo "Results: $RESULT_DIR"
echo "Collect after jobs finish: $PYTHON_BIN $SCRIPT_DIR/collect_results.py $RESULT_DIR"
