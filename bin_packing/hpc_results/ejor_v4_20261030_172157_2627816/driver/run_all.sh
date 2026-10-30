#!/bin/bash
# Default to the latest Word plan; legacy frozen packages retain their old driver.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_plan_v4.sh" "$@"
