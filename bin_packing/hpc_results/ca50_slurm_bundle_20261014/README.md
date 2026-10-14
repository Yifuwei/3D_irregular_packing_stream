# Slurm: RCL tolerance selection with one CA plus 50 LS evaluations

This package runs the current fixed-work tolerance experiment as **270 independent GPU array tasks**: 15 datasets, cube/cylinder, order seeds 3/7/13, and tolerances 10%/15%/20%. Each task completes exactly one CA and 50 evaluated LS neighbors, except that a single-bin solution may stop early with its actual count recorded. There is no 180-second algorithm cutoff, stagnation restart, or second construction. Slurm's wall limit remains a separate hard process limit; a killed task has no valid final result and can be retried.

The frozen worker uses the original `improved_GRASP`, not `improved_GRASP_weight_radio` or `improved_GRASP_ILS`. Piece seed is 101, position seed is 202, radio limit is 999999, and rho is 1. Cube PES is `maximal_residual_box`; cylinder PES is `minimum_aabb_edges_len`; both use `bottom_top`. RCL events include all evaluated orientations. No local results are automatically imported into the cluster comparison, so different hardware timings are kept separate.

## Prepare locally

From the repository root, using Python 3.10 or later:

```powershell
python bin_packing/hpc/ca50/prepare_bundle.py `
  --experiment bin_packing/test/logs/grasp_tolerance_ca50_all_datasets_20261014 `
  --output bin_packing/hpc_results/ca50_slurm_bundle_20261014
```

The output folder and ZIP contain frozen algorithm modules, the CA+50 worker, all referenced instance/voxel files, a 270-case manifest with SHA256 checks, submission scripts, and collection scripts. Existing output folders are never overwritten. Runtime inputs and original algorithm hashes must match the local experiment manifest. Upload and extract the entire ZIP on a shared filesystem visible to compute nodes; the original Windows repository path is provenance only and is not required on the cluster.

## Cluster environment and dry run

The defaults follow the repository's existing cluster conventions: module `miniforge`, Conda environment `packing`, partition `gpu`, one GPU, four CPUs, and a 12-hour Slurm wall limit per case. Host memory is **32 GB per task**, reduced to **16 GB for chess and shapesnew**. These two datasets have clearly smaller voxel grids and object arrays; this is a conservative configuration policy, not a measured peak-memory guarantee. GPU VRAM is separate. Use the site's correct GPU partition/account/QOS and environment. A working NumPy/SciPy/Numba/CUDA environment must also provide the existing launcher's dependencies, including pandas, matplotlib and line_profiler. This bundle does not install or modify cluster software.

```bash
unzip ca50_slurm_bundle_20261014.zip
cd ca50_slurm_bundle_20261014
module load miniforge
conda activate packing
python task.py --index 0 --dry-run
bash submit.sh --dry-run
```

Both dry runs use only the Python standard library and do not allocate GPUs or submit jobs. The worker dry run verifies the package hashes and prints the exact selected case and command. The submission dry run prints the Slurm command.

To use a pre-activated environment, set `HPC_MODULE=none HPC_CONDA_ENV=none`. `PYTHON_BIN` controls compute-node Python; `HPC_SUBMIT_PYTHON` controls login-node manifest parsing. `HPC_NUMACTL=1` optionally enables `numactl --interleave=all` if available.

## Submit and monitor

```bash
# Up to eight independent cases run concurrently, each on its own GPU.
# 36 small cases request 16G; the remaining 234 cases request 32G.
MAX_PARALLEL=8 bash submit.sh

# Override cluster resources and account as required by the site:
MAX_PARALLEL=4 bash submit.sh -- --partition=gpu --account=YOUR_ACCOUNT --time=24:00:00

squeue -u "$USER"
```

The arrays preserve the global indices 0..269 from the JSON plan. Submission creates a 16G array for chess/shapesnew and a 32G array for the other datasets. The 32G array waits for the 16G array to finish (`afterany` dependency), so total GPU concurrency stays at MAX_PARALLEL rather than doubling. With MAX_PARALLEL=8, peak requested host memory is 128G during the small array and 256G during the standard array. Empty groups, including during pending-only submission, are skipped. If you explicitly supply `--mem=...` after `--`, that value overrides both groups. Inspect MaxRSS for pilot jobs and increase a group's allocation if jobs are killed for memory exhaustion.

Each worker has a fresh process, CUDA warmup, pools and RNG streams. OMP/MKL/OpenBLAS/Numba thread counts match allocated CPUs. Job-specific Numba cache directories avoid cross-task compilation cache contention. Results and per-case logs are isolated; no shared summary files are written by GPU tasks. Do not edit the frozen bundle files after preparation because their hashes are checked; use sbatch command-line resource overrides instead.

`RESULT_DIR=/shared/path/ca50_run` selects a separate results directory. Keep the same value when collecting or resubmitting. Do not submit overlapping arrays concurrently to the same directory. Per-case exclusive locks prevent duplicate work; if Slurm kills a task before cleanup, inspect `results/locks/*.lock` and confirm the recorded job is no longer active before manually removing that specific stale lock. Locks are never cleared automatically.

## Collect results and retry missing/failed cases

Run collection after the array completes, or during execution for a partial report. Use only one collector at a time.

```bash
python collect.py
bash submit.sh --pending --dry-run
bash submit.sh --pending
python collect.py
```

The pending list excludes parameter-matching successful results; failed, missing, or mismatched results are retried. Prior failed/mismatched attempts are retained under `results/previous_attempts/`. A timeout/OOM with no JSON remains `missing`, not a zero-score result; inspect `results/slurm/` and the case log for the cause. Collection checks geometry-validation status, work counts, final objective arithmetic, raw RCL histograms, and LS evaluation CSV counts before including quality results.

Outputs in `results/`:

- `best_tolerance_by_dataset.md` / `.csv`: mean CA and final N-U*, improvements, elapsed time, RCL statistics and best rates for each dataset and shape. A winner is assigned only after all nine cases in that group succeed; ties retain every tied rate.
- `paired_comparison.csv`: matched-seed direct comparisons, including 15% versus 20%.
- `summary_by_dataset.csv`, `results.csv`, `rcl_frequency.csv`: detailed summary and raw result tables.
- `task_inventory.csv`, `pending_indices.txt`: success/failed/invalid/missing coverage and resubmission indices.
- `final_report.md`, `final_audit.json`: generated when all 270 cases have produced records; failures remain visible and audit validity must be checked before calling the experiment complete.
- Per-case `.json`, `.status.json`, `.log`, `.rcl.csv`, `.ls.csv`: work counts, CA/final geometry validation, current phase, candidate scores and trajectory.

Lower N-U* is better. Equal numbers of LS evaluations do not imply equal wall time. Three order seeds are parameter-selection evidence rather than independent generalization validation. Geometry checks do not certify complete insertion trajectories. Test selected rates with held-out seeds before claiming generalization.

## Local script tests

```bash
python -m unittest discover -s bin_packing/hpc/ca50 -p 'test_*.py' -v
bash -n bin_packing/hpc/ca50/submit.sh
bash -n bin_packing/hpc/ca50/run.slurm
```

These tests exercise task mapping, invalid index rejection, portable paths, modified-file detection, result identity/work validation, pending selection, and partial collection. They do not submit Slurm jobs or test cluster CUDA compatibility.
