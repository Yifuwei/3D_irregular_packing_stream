#!/usr/bin/env bash
#SBATCH --job-name=meta_iterations
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
set -eo pipefail
if [[ "${HPC_MODULE:-miniforge}" != none ]]; then
  module load "${HPC_MODULE:-miniforge}"
fi
if [[ "${HPC_CONDA_ENV:-packing}" != none ]]; then
  source ~/.bashrc
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${HPC_CONDA_ENV:-packing}"
fi
set -u
: "${PROJECT_DIR:?Use submit_iterations.sh}"
: "${PYTHON_BIN:?Missing CUDA environment Python}"
: "${RESULT_DIR:?Missing result directory}"
: "${MANIFEST:?Missing manifest}"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/iterations_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
nvidia-smi
launcher=("$PYTHON_BIN" -u "$PROJECT_DIR/hpc/metaheuristic_iterations.py" run \
  --project "$PROJECT_DIR" --manifest "$MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" --output "$RESULT_DIR")
if [[ "${HPC_NUMACTL:-1}" == 1 ]]; then
  launcher=(numactl --interleave=all "${launcher[@]}")
fi
srun --exclusive -N1 -n1 "${launcher[@]}"
