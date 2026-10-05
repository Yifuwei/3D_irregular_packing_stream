# ILS versus improved ILS: HPC benchmark

For the updated five-seed study (3, 7, 13, 19, 53), use `submit_multiseed.sh`.
See `README_multiseed.md` for submission and result collection.

This folder is independent of the existing repository-root `hpc` experiments.
Upload **the current local project** `C:/work/paper_repo/3D_irregular_packing_stream`,
including `bin_packing/src`, `bin_packing/test/code_launcher.py`, `bin_packing/data`,
`bin_packing/instances`, and this `bin_packing/hpc` directory. Keep filename case intact.
Do not upload an older checkout instead: the current improved implementation must be present.

## Submit

On the HPC login node, from the uploaded repository root:

```bash
# Existing cluster conventions: miniforge module, conda environment packing.
# Build that environment using lib_requirements.txt if it is not already available.
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate packing
python -c 'from numba import cuda; print(cuda.__version__ if hasattr(cuda, "__version__") else "Numba imported")'

bash bin_packing/hpc/submit.sh
```

Defaults: all 15 seed-13 cube datasets; 100 candidate evaluations per algorithm;
kick threshold 100; one paired run per dataset; at most eight dataset jobs at once.
Each task requests the existing `gpu` partition, one GPU, four CPU cores, 240 GB RAM
and six hours. Change resources using normal sbatch overrides:

```bash
# Example: three repeats of each 100-evaluation experiment, longer job limit.
REPEATS=3 MAX_PARALLEL=4 bash bin_packing/hpc/submit.sh --time=12:00:00
# Use an already configured environment instead of loading miniforge/packing:
HPC_MODULE=none HPC_CONDA_ENV=none HPC_NUMACTL=0 \
  PYTHON_BIN=/absolute/path/to/environment/bin/python bash bin_packing/hpc/submit.sh
# If your GPU queue differs, override --partition / --gres / --account here.
```

The submission prints the result directory and job ID. Use `squeue -u "$USER"`
to check progress and `scancel JOB_ID` to stop the array. Notifications are not sent by these scripts.
The dataset ordering and source hashes are frozen in `submission_manifest.json`;
editing production source while jobs are queued causes a clear failure.

## Collect

After every array task has finished (including failed tasks):

```bash
python bin_packing/hpc/collect_results.py /absolute/path/to/result_directory \
  --project "$PWD/bin_packing"
```

Outputs:

- `collected_runs.csv`: individual paired runs, times, budgets, kicks and solution quality.
- `summary.csv`: per-dataset medians, ranges, valid and failed paired-run counts.
- `report.md`: the combined comparison table; positive reduction means improved is faster.
- `DATASET/repeat_N/{original,improved}.log` and `.json`: execution evidence and hardware/source metadata.
- `slurm/*.out` and `.err`: scheduler/environment failures, including tasks with no result CSV.

The collector can run before completion, but missing/partial rows are not final results.
One repeat is one observation; use at least three for timing variability.
Every repeat uses seed 13 and the same input. Order alternates across repeats.

## What is timed and adapted

Each algorithm uses a separate process, fresh geometry pools, and untimed construction/GPU
warm-up. Random state resets before formal execution. CUDA is synchronized before and after
timing. Time includes initial construction and all candidate processing, but excludes imports,
input loading, warm-up and final result checks. Detailed event tracing is disabled; enable
`--progress` for candidate completion lines when running the Python entry point directly.
Both algorithms run sequentially inside the same resource allocation.

Production functions are loaded from the uploaded source. Original ILS gets the same minimal
test-only fixes used locally: exact budget guards, independent orientation-list snapshots,
post-kick end timestamp, and rebuilding the neighborhood after a kick. Executed adapter
source is saved. A test-only objective wrapper normalizes single-bin U* to U.

**Exactly 100 includes regular repacks and kicks; it does not mean 100 kicks.**
Initial construction and warm-up are outside that count. A threshold of 100 may produce no
kicks within a 100-evaluation run.

The production improved algorithm stops at one bin. To measure a fixed workload, this
benchmark alone disables that stop and makes every piece eligible when only one bin remains.
The two improved stop conditions and selector wrapper are saved in `improved_budget_adapter.py`.
Further evaluations after one bin serve timing only; they cannot reduce the bin count.
No production file is edited. This policy addresses the earlier Merged1 early-exit failure.
These are whole-algorithm comparisons, not isolated measurements of skip-bin optimization.

## Local validation

```bash
python -m unittest discover -s bin_packing/hpc -p 'test_*.py'
bash -n bin_packing/hpc/submit.sh
bash -n bin_packing/hpc/run_ils.slurm
python bin_packing/hpc/benchmark_ils.py --dataset chess --output fresh_validation_directory
```

HPC execution itself requires a GPU allocation and the cluster's available modules.
Successful local validation does not establish cluster compatibility.
