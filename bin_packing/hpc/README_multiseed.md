# Five-seed ILS comparison

Use the **updated** source with last-bin eligibility after an accepted layout change.
Upload the local `bin_packing/src/new_ILS_southampton.py` and updated `bin_packing/hpc`
files into the existing HPC project. Previously uploaded code/results use the old rule.

## Submit from the HPC repository root

```bash
bash bin_packing/hpc/submit_multiseed.sh
```

Defaults: sequence seeds **3, 7, 13, 19, 53**; the LS RNG seed equals the sequence seed.
Fifteen datasets times five paired seeds gives **75 Slurm tasks**, at most eight concurrent.
Each task sequentially evaluates original and improved ILS **100 times each** on the same
allocated GPU, using identical input sequence and RNG seed. Seeds differ across tasks.
Warm-up is excluded and RNG state resets before formal execution. Algorithm order alternates
across tasks to reduce a systematic order bias. Seeds, input hashes, budgets and code hashes
are recorded in the frozen submission manifest and individual JSON records.

The exact budget includes kicks. With the default budget 100 and kick threshold 100,
the algorithm stops before a kick can be added. Keep this setting for comparison with the
previous experiment; changing the threshold creates a different experimental setting.

Resource/environment conventions match the existing scripts: `gpu` partition, one GPU,
four CPU cores, 240 GB memory, six-hour limit, miniforge/packing and optional numactl.
Standard sbatch overrides and HPC_MODULE/HPC_CONDA_ENV/HPC_NUMACTL variables still work.

```bash
# Optional: three timing repeats per paired seed, each still 100 evaluations.
REPEATS=3 MAX_PARALLEL=4 bash bin_packing/hpc/submit_multiseed.sh --time=12:00:00
# Explicitly run only the four additional seeds:
SEEDS="3 7 19 53" bash bin_packing/hpc/submit_multiseed.sh
```

## Check and collect

```bash
squeue -u "$USER"
python bin_packing/hpc/collect_results.py /absolute/path/printed/after/Results
```

The result directory is printed at submission and defaults to `bin_packing/hpc_results/`.
Do not reuse the previous result directory. Output locations:

- `task_INDEX_runs/DATASET/repeat_N/{original,improved}.json` and `.log`: actual worker results.
- `task_INDEX.csv`: each dataset/seed paired comparison.
- `collected_runs.csv`: all runs with `seed` (sequence seed) and `ls_seed`, time, bins, U*, budget and kicks.
- `summary.csv`: per-dataset timing medians, means, standard deviations and ranges; average
  bin counts and counts of improved having fewer/equal/more bins across valid pairs.
- `report.md`: timing table. Inspect the quality columns in summary.csv alongside speed.

Do not merge the old seed-13 result into these new runs: the source changed. The included
seed-13 rerun supplies a baseline on the same code version. Only complete paired runs enter
the statistics; unfinished datasets are labeled partial/missing rather than passed.

This is a **paired-seed design**: sequence and LS randomness vary together. It evaluates
overall sensitivity but cannot attribute variability separately to the two sources.
A factorial study would instead cross sequence seeds with LS seeds. One repeat per paired
seed yields five different random conditions, not five timing repeats of the same condition.

## Validation

```bash
python -m unittest discover -s bin_packing/hpc -p 'test_*.py'
bash -n bin_packing/hpc/submit_multiseed.sh
bash -n bin_packing/hpc/run_multiseed.slurm
```

The prepare command refuses duplicate seeds, missing dataset coverage and nonpositive budgets.
Workers refuse code/instance changes after submission. The tests check paired seeds, exact
budget arguments, missing inputs, duplicate results and partial-result labeling.
Real cluster compatibility remains subject to the configured modules and GPU allocation.
