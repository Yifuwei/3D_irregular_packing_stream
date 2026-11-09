#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/../.." && pwd)"
source_bundle="${PARAMETER_BUNDLE:-$project/hpc_results/ejor_v4_rcl_fixed_20261030_173330}"
bundle="${EJOR_OUTPUT:-$project/hpc_results/ejor_geometry_only_$(date +%Y%m%d_%H%M%S)_$$}"
python "$script_dir/geometry_only.py" --source "$source_bundle" --project "$project" --output "$bundle"
python "$bundle/driver/suite.py" submit-all --bundle "$bundle" --through full --flow heuristics --max-parallel "${MAX_PARALLEL:-6}" "$@"
printf 'Geometry-only bundle: %s\n' "$bundle"
