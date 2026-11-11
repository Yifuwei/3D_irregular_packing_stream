#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/../.." && pwd)"
export CALIBRATION_BUNDLE="${CALIBRATION_BUNDLE:-$project/hpc_results/ejor_v4_rcl_fixed_20261030_173330}"
export EJOR_OUTPUT="${EJOR_OUTPUT:-$project/hpc_results/ejor_unified_ca_$(date +%Y%m%d_%H%M%S)_$$}"
exec bash "$script_dir/run_plan_v4.sh" "$@"
