#!/bin/bash
# Memory-first: same largest instance, three algorithms, 32GB hard cap.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/../.." && pwd)"
repo="$(cd -- "$project/.." && pwd)"
bundle="${EJOR_OUTPUT:-$project/hpc_results/memory32_smoke_$(date +%Y%m%d_%H%M%S)_$$}"
summary="${CA_SUMMARY:-$project/hpc_results/CA param test result/summary_by_dataset.csv}"
if [[ ! -f "$summary" && -z "${CA_SUMMARY:-}" ]]; then summary="$repo/CA param test result/summary_by_dataset.csv"; fi
python "$script_dir/memory_probe.py" prepare --project "$project" --ca-summary "$summary" \
    --output "$bundle" --time-limit "${TIME_LIMIT:-3600}"
mkdir -p "$bundle/logs"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    job=$(sbatch --parsable --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G \
        --time=06:00:00 --array=0-2%1 --job-name=memory32 \
        --output="$bundle/logs/memory_%A_%a.out" \
        --export="ALL,EJOR_BUNDLE=$bundle,EJOR_STAGE=memory,EJOR_TASK_MAP=$bundle/memory_task_map.json" \
        "$bundle/driver/run_array.slurm")
    printf '%s\n' "$job" > "$bundle/job_id.txt"
    printf 'Job: %s\n' "$job"
fi
printf 'Memory probe bundle: %s\n' "$bundle"
