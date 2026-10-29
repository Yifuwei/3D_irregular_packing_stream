# EJOR experiment suite: one-shot submission

See [WORK_PLAN.md](WORK_PLAN.md) for the current protocol and execution plan.

- Cube: maximal_residual_box / bottom. Cylinder: minimum_aabb_edges_len / bottom_top.
- CA/PES and RCL are shared choices across all 15 datasets. Kick strength uses the confirmed three representative datasets together, with one shared winner inherited by hybrid.
- Preserve the original five-seed capacities. Actual experiment sequence seeds: 3,7,13.
- Keep 324 matching sensitivity reports; rerun 486 mismatches. RCL has 3240 tasks, kick tuning 486, full 2430.
- One-shot DAG: calibration -> {sensitivity, RCL, kick tuning}; {RCL, kick tuning} -> full. Sensitivity does not gate the other branches.
- Entire graph maximum: 6 GPU tasks. Initially sensitivity/RCL/tuning use 2 each; completed branches release slots to the others automatically. Per task: 1 GPU, 4 CPU; RAM 16G for chess/shapesnew, otherwise 32G.
- Source and frozen scripts default to --through full. Explicit --through sensitivity remains available.

After stopping the package's old jobs and pulling the updated source, from the HPC repository root:

```bash
BUNDLE=bin_packing/hpc_results/ejor_continue_20261029_ready
bash bin_packing/hpc/paper_suite/run_ca_sensitivity_rerun.sh "$BUNDLE" --dry-run
bash bin_packing/hpc/paper_suite/run_ca_sensitivity_rerun.sh "$BUNDLE"
```

This updates the existing package and submits all remaining independent branches and their collectors. No need to wait for sensitivity before submitting later experiments. Full waits for successful RCL/tuning selection. No overlapping duplicate submission to the same package is allowed. See submission_history/concurrency.json for current allocations.
