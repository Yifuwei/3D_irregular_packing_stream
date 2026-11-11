#!/bin/bash
# New Word plan: 2x2 capacity levels, fixed overall best PES per shape.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/../.." && pwd)"
source_bundle="${CALIBRATION_BUNDLE:-$project/hpc_results/CA_sensitivity_check}"
if [[ ! -f "$source_bundle/results/calibration/selection.json" ]]; then
    printf 'Completed fixed-capacity calibration not found: %s\nSet CALIBRATION_BUNDLE to the original completed bundle.\n' "$source_bundle" >&2
    exit 1
fi
summary="${CA_SUMMARY:-$source_bundle/inputs/ca_reference_summary.csv}"
bundle="${EJOR_OUTPUT:-$project/hpc_results/ejor_v4_$(date +%Y%m%d_%H%M%S)_$$}"
prepare_args=()
if [[ "${EJOR_SMOKE:-0}" == "1" ]]; then
    prepare_args=(--smoke --datasets "${SMOKE_DATASET:-shapesnew}" --tuning-datasets "${SMOKE_DATASET:-shapesnew}" --time-limit "${SMOKE_TIME_LIMIT:-30}")
fi
python "$script_dir/suite.py" prepare --plan v4 --project "$project" --ca-summary "$summary"     --reuse-bundle "$source_bundle" --output "$bundle" "${prepare_args[@]}"
python "$bundle/driver/suite.py" submit-all --bundle "$bundle" --through full --flow all     --max-parallel "${MAX_PARALLEL:-6}" "$@"
printf 'New v4 bundle: %s\n' "$bundle"
