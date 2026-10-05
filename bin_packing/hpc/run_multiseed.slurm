#!/usr/bin/env bash
#SBATCH --job-name=ils100_seeds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
set -eo pipefail
# Match the existing repository hpc/job_submit.sh environment conventions.
if [[ "${HPC_MODULE:-miniforge}" != none ]]; then
    module load "${HPC_MODULE:-miniforge}"
fi
if [[ "${HPC_CONDA_ENV:-packing}" != none ]]; then
    source ~/.bashrc
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${HPC_CONDA_ENV:-packing}"
fi
set -u
: "${PROJECT_DIR:?Submit using submit.sh}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the CUDA-enabled environment Python}"
: "${RESULT_DIR:?Missing output directory}"
# If your cluster requires a CUDA module, load it here before srun.
export MPLBACKEND=Agg
export PYTHONHASHSEED="${SEED:-13}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ils_numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
nvidia-smi
launcher=("$PYTHON_BIN" -u "$PROJECT_DIR/hpc/multiseed_ils.py" \
  --project "$PROJECT_DIR" --task-index "$SLURM_ARRAY_TASK_ID" \
  --manifest "$MANIFEST" --output "$RESULT_DIR")
if [[ "${HPC_NUMACTL:-1}" == 1 ]]; then
    launcher=(numactl --interleave=all "${launcher[@]}")
fi
srun --exclusive -N1 -n1 "${launcher[@]}"
