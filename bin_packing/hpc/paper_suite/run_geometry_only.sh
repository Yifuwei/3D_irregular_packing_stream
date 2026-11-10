#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/../.." && pwd)"
source_bundle="${PARAMETER_BUNDLE:-$project/hpc_results/ejor_v4_rcl_fixed_20261030_173330}"
bundle="${EJOR_OUTPUT:-$project/hpc_results/ejor_geometry_only_$(date +%Y%m%d_%H%M%S)_$$}"
reuse_args=()
if [[ -n "${GEOMETRY_REUSE_BUNDLE:-}" ]]; then
    reuse_args=(--reuse-geometry-bundle "$GEOMETRY_REUSE_BUNDLE")
fi
python "$script_dir/geometry_only.py" --source "$source_bundle" --project "$project" --output "$bundle" "${reuse_args[@]}"
python "$bundle/driver/suite.py" submit-all --bundle "$bundle" --through full --flow heuristics --max-parallel "${MAX_PARALLEL:-6}" "$@"
printf 'Geometry-only bundle: %s\n' "$bundle"
