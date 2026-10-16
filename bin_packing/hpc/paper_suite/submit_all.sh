#!/bin/bash
# Usage: bash submit_all.sh /scratch/path/ejor_bundle [--dry-run ...]
set -euo pipefail
bundle="$1"
shift
python "$bundle/driver/suite.py" submit-all --bundle "$bundle" "$@"
