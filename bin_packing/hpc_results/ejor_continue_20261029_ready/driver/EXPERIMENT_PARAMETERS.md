# EJOR experiment suite

The current authoritative plan is [WORK_PLAN.md](WORK_PLAN.md). CA/PES and RCL aggregate all 15 datasets; kick strength aggregates the original three representatives together. No independent per-dataset winners.

- Shared CA/PES: cube=maximal_residual_box/bottom; cylinder=minimum_aabb_edges_len/bottom_top.
- Keep original five-seed capacities. Experiments use only sequence seeds 3,7,13.
- CA sensitivity: 810 total; 324 compatible results retained, 486 require rerun with the shared settings.
- RCL: 3240 tasks. ILS strength tuning: 486 tasks across shapesnew, Merged4_normal, st04_Example5_normal. Full: 2430 tasks.
- Default maximum GPU concurrency 6; 1 GPU/4 CPU per task, 16G for chess/shapesnew, otherwise 32G.
- Complete stages gate later stages. Selected parameter files record and validate the dataset population.

Use the existing ready package and its frozen driver. Stop its old jobs before synchronizing updates.

```bash
BUNDLE=bin_packing/hpc_results/ejor_continue_20261029_ready
bash "$BUNDLE/driver/run_ca_sensitivity_rerun.sh" "$BUNDLE" --dry-run
bash "$BUNDLE/driver/run_ca_sensitivity_rerun.sh" "$BUNDLE"
# After sensitivity succeeds:
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --through full --max-parallel 6
```

For a fresh package, run_all.sh prepares a current runtime, retains original calibration references and only reuses sensitivity reports whose actual PES/range and capacities match. Pending sensitivity cases must complete before downstream stages can pass their collectors. Existing source results are not overwritten.
