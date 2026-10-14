#!/usr/bin/env bash
# Resume only missing/invalid tasks. Successful task JSONs are never overwritten.
set -euo pipefail
[[ $# -ge 1 ]] || { echo 'Usage: bash resume_iterations.sh RESULT_DIR [sbatch options]' >&2; exit 1; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export RESULT_DIR="$(cd -- "$1" && pwd)"
shift
MAX_PARALLEL="${MAX_PARALLEL:-3}"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid MAX_PARALLEL' >&2; exit 1; }
submit_python="${HPC_SUBMIT_PYTHON:-python3}"
# Do not archive/restart a task while its original process may still be writing.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  command -v squeue >/dev/null
  command -v scontrol >/dev/null
  active_ids="$(squeue -h -u "$USER" -o '%A' | sort -u)"
  while IFS= read -r job_id; do
    [[ -n "$job_id" ]] || continue
    info="$(scontrol show job "$job_id" 2>/dev/null || true)"
    if [[ "$info" == *"StdOut=$RESULT_DIR/"* ]]; then
      echo "Experiment still has active job $job_id. Cancel/wait for it, then retry." >&2
      exit 1
    fi
  done <<< "$active_ids"
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  "$submit_python" "$SCRIPT_DIR/resume_iterations.py" --project "$PROJECT_DIR" --results "$RESULT_DIR"
  exit 0
fi
export MANIFEST="$RESULT_DIR/resume_manifest_$(date +%Y%m%d_%H%M%S)_$$.json"
plan="$("$submit_python" "$SCRIPT_DIR/resume_iterations.py" --project "$PROJECT_DIR" \
  --results "$RESULT_DIR" --write-manifest "$MANIFEST" --archive-incomplete)"
echo "$plan"
array="$("$submit_python" -c 'import json,sys; print(json.loads(sys.argv[1])["array"])' "$plan")"
if [[ -z "$array" ]]; then echo 'All tasks already completed; nothing submitted.'; exit 0; fi
sbatch --array="${array}%${MAX_PARALLEL}" --export=ALL --chdir="$PROJECT_DIR" \
  --output="$RESULT_DIR/slurm/%A_%a.out" --error="$RESULT_DIR/slurm/%A_%a.err" \
  --mem=16G "$@" "$SCRIPT_DIR/run_iterations.slurm"
echo "Resumed into: $RESULT_DIR"
echo "Collect with original manifest: $PYTHON_BIN $SCRIPT_DIR/metaheuristic_iterations.py collect --manifest $RESULT_DIR/manifest.json --output $RESULT_DIR"
