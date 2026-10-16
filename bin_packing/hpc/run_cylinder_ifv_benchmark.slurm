#!/usr/bin/env bash
# CPU boundary benchmark: no GPU resource is requested.
#SBATCH --job-name=cylinder_ifv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -eo pipefail
module load "${HPC_MODULE:-miniforge}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HPC_CONDA_ENV:-packing}"
set -u
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
output="${IFV_OUTPUT:-$PROJECT_DIR/hpc_results/cylinder_ifv_${SLURM_JOB_ID}}"
python -u "$PROJECT_DIR/hpc/benchmark_cylinder_ifv.py" --project "$PROJECT_DIR" \
  --dataset "${DATASET:-st04_Example5_normal}" --seed "${SEED:-3}" \
  --repeats "${REPEATS:-3}" --max-pieces "${MAX_PIECES:-3}" --output "$output"
