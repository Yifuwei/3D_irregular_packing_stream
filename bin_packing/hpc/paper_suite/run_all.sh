#!/bin/bash
# Prepare a fresh frozen bundle and submit the complete dependent experiment.
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
reference_dir="${CA_JSON_DIR:-$project/hpc_results/CA_jsons_results}"
if [[ -d "$reference_dir" || -n "${CA_JSON_DIR:-}" ]]; then
    reference_args=(--ca-json-dir "$reference_dir")
fi
python "$script_dir/suite.py" prepare --project "$project" --ca-summary "$summary" \
    --output "$bundle" "${reference_args[@]}"
python "$bundle/driver/suite.py" submit-all --bundle "$bundle" --max-parallel "${MAX_PARALLEL:-3}" "$@"
printf 'Frozen experiment and results: %s\n' "$bundle"
