#!/bin/bash
# Stop this bundle's old jobs before preparing the rerun; default end is sensitivity.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bundle="${1:-$script_dir/../../hpc_results/ejor_continue_20261029_ready}"
if [[ $# -gt 0 ]]; then shift; fi
python "$script_dir/rerun_ca_sensitivity.py" --bundle "$bundle"
python "$bundle/driver/suite.py" submit-all --bundle "$bundle" --through sensitivity --max-parallel 6 "$@"
