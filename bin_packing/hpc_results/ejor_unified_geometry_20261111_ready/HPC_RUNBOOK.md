# Geometry-only HPC runbook

This bundle uses first-fit, minimum AABB edge length / bottom-top for both shapes, 10% RCL tolerance and 75% kick strength. It contains 270 full tasks at 3600 seconds per run. The 36-task 16 GiB array precedes the 234-task 32 GiB array. Each runs at most six tasks concurrently.

Transfer this bundle directory to HPC, then run:

```bash
export BUNDLE="/mnt/scratch/pmywe/GPU/3D_irregular_packing_stream/bin_packing/hpc_results/ejor_unified_geometry_20261111_ready"
module load miniforge
conda activate packing
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --through full --flow heuristics --dry-run
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --through full --flow heuristics
squeue -u "$USER"
```

After the collector finishes, check:

```bash
cat "$BUNDLE/results/full/completion.json"
head "$BUNDLE/results/full/summary.csv"
```

Completion should show 270 successful runs and an empty incomplete list. Per-run results are in results/full/runs.csv and task JSON files. The worker stores final results and aggregate counters without per-event traces.
