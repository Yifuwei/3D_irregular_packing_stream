#!/bin/bash
set -euo pipefail
if [[ "${HPC_MODULE:-miniforge}" != "none" ]]; then module load "${HPC_MODULE:-miniforge}"; fi
if [[ "${HPC_ENV:-packing}" != "none" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${HPC_ENV:-packing}"
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [[ "$EJOR_STAGE" == "memory" ]]; then
    exec /usr/bin/time -v -o "$EJOR_BUNDLE/logs/memory_${SLURM_ARRAY_TASK_ID}_time.txt" \
        python "$EJOR_BUNDLE/driver/suite.py" run --bundle "$EJOR_BUNDLE" --stage "$EJOR_STAGE" \
        --index "$SLURM_ARRAY_TASK_ID" --task-map "$EJOR_TASK_MAP"
else
    python "$EJOR_BUNDLE/driver/suite.py" run --bundle "$EJOR_BUNDLE" --stage "$EJOR_STAGE" \
        --index "$SLURM_ARRAY_TASK_ID" --task-map "$EJOR_TASK_MAP"
fi
