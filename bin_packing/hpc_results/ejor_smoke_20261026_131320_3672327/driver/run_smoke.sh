#!/bin/bash
# Real GPU smoke workflow, isolated from formal experiment results.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/../.." && pwd)"
repo="$(cd -- "$project/.." && pwd)"
bundle="${EJOR_OUTPUT:-$project/hpc_results/ejor_smoke_$(date +%Y%m%d_%H%M%S)_$$}"
summary="${CA_SUMMARY:-$project/hpc_results/CA param test result/summary_by_dataset.csv}"
if [[ ! -f "$summary" && -z "${CA_SUMMARY:-}" ]]; then
    summary="$repo/CA param test result/summary_by_dataset.csv"
fi
reference_dir="${CA_JSON_DIR:-$project/hpc_results/CA_jsons_results}"
if [[ ! -d "$reference_dir" ]]; then
    printf 'Existing CA results not found: %s\nSet CA_JSON_DIR to the existing CA JSON directory. No GPU calibration was submitted.\n' "$reference_dir" >&2
    exit 1
fi
reference_args=(--ca-json-dir "$reference_dir")
printf 'Reusing existing CA references: %s\n' "$reference_dir"
read -r -a smoke_shapes <<< "${SMOKE_SHAPES:-cube cylinder}"
dataset="${SMOKE_DATASET:-shapesnew}"
python "$script_dir/suite.py" prepare --project "$project" --ca-summary "$summary" \
    --output "$bundle" --datasets "$dataset" --tuning-datasets "$dataset" \
    --shapes "${smoke_shapes[@]}" "${reference_args[@]}" \
    --time-limit "${SMOKE_TIME_LIMIT:-30}" --smoke
python "$bundle/driver/suite.py" submit-all --bundle "$bundle" \
    --max-parallel "${MAX_PARALLEL:-1}" --walltime "${SMOKE_WALLTIME:-01:00:00}" --through sensitivity "$@"
printf 'Smoke bundle: %s\nCA sensitivity completion: %s/results/sensitivity/completion.json\n' "$bundle" "$bundle"
