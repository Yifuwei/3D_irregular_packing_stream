#!/usr/bin/env bash
# One array task per dataset/shape/seed/configuration. No submission with DRY_RUN=1.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/hpc_results/iterations_$(date +%Y%m%d_%H%M%S)_$$}"
export MANIFEST="$RESULT_DIR/manifest.json"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid MAX_PARALLEL' >&2; exit 1; }
read -r -a datasets <<< "${DATASETS:-shapesnew Merged4_normal st04_Example5_normal}"
read -r -a shapes <<< "${SHAPES:-cube cylinder}"
read -r -a seeds <<< "${SEEDS:-3 7 13}"
mkdir -p "$RESULT_DIR/slurm"
count="$("${HPC_SUBMIT_PYTHON:-python3}" "$SCRIPT_DIR/metaheuristic_iterations.py" prepare \
  --project "$PROJECT_DIR" --manifest "$MANIFEST" --datasets "${datasets[@]}" \
  --shapes "${shapes[@]}" --seeds "${seeds[@]}" --time-limit "${TIME_LIMIT:-3600}" \
  --pes "${PES:-minimum_aabb_volume}" --range "${SELECTION_RANGE:-bottom_top}")"
[[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'No valid tasks' >&2; exit 1; }
echo "Prepared $count tasks: $MANIFEST"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  sbatch --array="0-$((count-1))%$MAX_PARALLEL" --export=ALL --chdir="$PROJECT_DIR" \
    --output="$RESULT_DIR/slurm/%A_%a.out" --error="$RESULT_DIR/slurm/%A_%a.err" \
    "$@" "$SCRIPT_DIR/run_iterations.slurm"
fi
echo "Results: $RESULT_DIR"
echo "Collect: $PYTHON_BIN $SCRIPT_DIR/metaheuristic_iterations.py collect --manifest $MANIFEST --output $RESULT_DIR"
