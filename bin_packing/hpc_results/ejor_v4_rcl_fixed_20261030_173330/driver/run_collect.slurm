#!/bin/bash
set -euo pipefail
if [[ "${HPC_MODULE:-miniforge}" != "none" ]]; then module load "${HPC_MODULE:-miniforge}"; fi
if [[ "${HPC_ENV:-packing}" != "none" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${HPC_ENV:-packing}"
fi
export PYTHONDONTWRITEBYTECODE=1
python "$EJOR_BUNDLE/driver/suite.py" collect --bundle "$EJOR_BUNDLE" --stage "$EJOR_STAGE"
