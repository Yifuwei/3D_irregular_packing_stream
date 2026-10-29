#!/bin/bash
# Prepare a fresh bundle; stop after the CA sensitivity check by default.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/../.." && pwd)"
repo="$(cd -- "$project/.." && pwd)"
bundle="${EJOR_OUTPUT:-$project/hpc_results/ejor_$(date +%Y%m%d_%H%M%S)_$$}"
summary="${CA_SUMMARY:-$project/hpc_results/CA param test result/summary_by_dataset.csv}"
if [[ ! -f "$summary" && -z "${CA_SUMMARY:-}" ]]; then
    summary="$repo/CA param test result/summary_by_dataset.csv"
fi
reference_args=()
completed="${SENSITIVITY_BUNDLE:-$project/hpc_results/CA_sensitivity_check}"
reference_dir="${CA_JSON_DIR:-$project/hpc_results/CA_jsons_results}"
if [[ -f "$completed/manifest.json" ]]; then
    reference_args=(--reuse-bundle "$completed")
    printf 'Continuing from completed calibration/sensitivity: %s\n' "$completed"
elif [[ -n "${SENSITIVITY_BUNDLE:-}" ]]; then
    printf 'Sensitivity bundle not found: %s\n' "$completed" >&2
    exit 1
elif [[ -d "$reference_dir" || -n "${CA_JSON_DIR:-}" ]]; then
    reference_args=(--ca-json-dir "$reference_dir")
fi
python "$script_dir/suite.py" prepare --project "$project" --ca-summary "$summary" \
    --output "$bundle" "${reference_args[@]}"
python "$bundle/driver/suite.py" submit-all --bundle "$bundle" --max-parallel "${MAX_PARALLEL:-3}" --through sensitivity "$@"
printf 'Frozen experiment and results: %s\n' "$bundle"
