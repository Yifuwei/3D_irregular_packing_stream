#!/usr/bin/env bash
# Three sequential algorithms on the largest dense-grid input; separate outputs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/hpc_results/memory_probe_$(date +%Y%m%d_%H%M%S)_$$}"
export MANIFEST="$RESULT_DIR/manifest.json"
mkdir -p "$RESULT_DIR/slurm"
"${HPC_SUBMIT_PYTHON:-python3}" "$SCRIPT_DIR/prepare_memory_probe.py" \
  --project "$PROJECT_DIR" --manifest "$MANIFEST" --seed "${SEED:-3}" \
  --time-limit "${TIME_LIMIT:-3600}" --kick-trigger "${RESOURCE_KICK_TRIGGER:-100}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  sbatch --array=0-2%1 --export=ALL --chdir="$PROJECT_DIR" \
    --output="$RESULT_DIR/slurm/%A_%a.out" --error="$RESULT_DIR/slurm/%A_%a.err" \
    --mem="${PROBE_MEMORY:-32G}" "$@" "$SCRIPT_DIR/run_iterations.slurm"
fi
echo "Resource-test results: $RESULT_DIR"
echo 'After completion: sacct -j JOBID --units=G --format=JobID,State,Elapsed,ReqMem,MaxRSS'
