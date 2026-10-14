#!/usr/bin/env bash
#SBATCH --job-name=grasp_ca50_rcl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -eo pipefail
if [[ "${HPC_MODULE:-miniforge}" != none ]]; then
    module load "${HPC_MODULE:-miniforge}"
fi
if [[ "${HPC_CONDA_ENV:-packing}" != none ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${HPC_CONDA_ENV:-packing}"
fi
set -u
BUNDLE_DIR="$1"
RESULT_DIR="$2"
: "${SLURM_ARRAY_TASK_ID:?Submit through submit.sh}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/grasp_ca50_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
command -v nvidia-smi >/dev/null && nvidia-smi
launcher=("${PYTHON_BIN:-python}" -u "$BUNDLE_DIR/task.py" --bundle "$BUNDLE_DIR" \
          --results "$RESULT_DIR" --index "$SLURM_ARRAY_TASK_ID")
if [[ "${HPC_NUMACTL:-0}" == 1 ]]; then
    launcher=(numactl --interleave=all "${launcher[@]}")
fi
srun --exclusive -N1 -n1 "${launcher[@]}"
