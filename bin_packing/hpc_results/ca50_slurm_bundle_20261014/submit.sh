#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-$BUNDLE_DIR/results}"
mkdir -p "$RESULT_DIR/slurm"
RESULT_DIR="$(cd -- "$RESULT_DIR" && pwd)"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_PARALLEL must be a positive integer' >&2; exit 1; }
dry_run=0
pending=()
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --pending) pending=(--pending); shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
previous_job=""
submitted=0
for memory in 16 32; do
    spec="$("${HPC_SUBMIT_PYTHON:-python3}" "$BUNDLE_DIR/task.py" --bundle "$BUNDLE_DIR" \
            --results "$RESULT_DIR" --array-spec --memory-gb "$memory" "${pending[@]}")"
    [[ -n "$spec" ]] || continue
    dependency=()
    if [[ -n "$previous_job" ]]; then dependency=("--dependency=afterany:$previous_job"); fi
    command=(sbatch --parsable --array="$spec%$MAX_PARALLEL" --mem="${memory}G" \
        "${dependency[@]}" --export=ALL --chdir="$BUNDLE_DIR" \
        --output="$RESULT_DIR/slurm/%A_%a.out" --error="$RESULT_DIR/slurm/%A_%a.err" \
        "$@" "$BUNDLE_DIR/run.slurm" "$BUNDLE_DIR" "$RESULT_DIR")
    if (( dry_run )); then
        printf '%q ' "${command[@]}"; printf '\n'
        previous_job="DRY_RUN_${memory}G_JOB_ID"
    else
        job_response="$("${command[@]}")"
        previous_job="${job_response%%;*}"
        [[ "$previous_job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch job ID: $job_response" >&2; exit 1; }
        echo "Submitted ${memory}G array: $previous_job"
    fi
    submitted=1
done
if (( ! submitted )); then echo 'All requested cases already succeeded.'; exit 0; fi
if (( dry_run )); then exit 0; fi
echo "Results: $RESULT_DIR"
echo "Collect: ${HPC_SUBMIT_PYTHON:-python3} $BUNDLE_DIR/collect.py --results $RESULT_DIR"
