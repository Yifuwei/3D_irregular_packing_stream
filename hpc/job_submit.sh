#!/bin/bash
#SBATCH --job-name=packing_nesting_range_all_instances
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pmywe@leeds.ac.uk

# submit.sh supplies --array and absolute stdout/stderr paths before submission.
# For 1,800 experiments it creates arrays of 1,000 and 800 tasks, each with %8.

set -eo pipefail
ROOT="$1"
PARAMS="$2"
TASK_OFFSET="${3:-0}"
cd "$ROOT"

# Set HPC_MODULE=none / HPC_CONDA_ENV=none to use an already active environment.
if [[ "${HPC_MODULE:-miniforge}" != none ]]; then
    module load "${HPC_MODULE:-miniforge}"
fi
if [[ "${HPC_CONDA_ENV:-packing}" != none ]]; then
    source ~/.bashrc
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${HPC_CONDA_ENV:-packing}"
fi
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
echo "[INFO] job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} params=$PARAMS"
TASK_INDEX=$((SLURM_ARRAY_TASK_ID + TASK_OFFSET))
echo "[INFO] global task index=$TASK_INDEX"
launcher=(python -u "$ROOT/hpc/task.py" "$PARAMS" "$TASK_INDEX")
if [[ "${HPC_NUMACTL:-1}" == 1 ]]; then
    launcher=(numactl --interleave=all "${launcher[@]}")
fi
srun --exclusive -N1 -n1 "${launcher[@]}"
echo "==== [OK] Task $TASK_INDEX finished ===="
