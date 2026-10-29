# EJOR experiments: simple independent flows

Current plan: [WORK_PLAN.md](WORK_PLAN.md).

- Sensitivity is independent of RCL/ILS tuning/full.
- Heuristic flow is RCL -> ILS tuning -> full.
- Each flow has a fixed maximum of 6 GPU tasks; combined maximum is 12. No automatic throttle redistribution.
- Cube: maximal_residual_box/bottom. Cylinder: minimum_aabb_edges_len/bottom_top.
- Original five-seed capacities are unchanged; actual experiment seeds are 3,7,13.
- Retain 324 sensitivity reports and rerun 486. RCL: 3240 tasks across 15 datasets. Kick tuning: 486 tasks across the three confirmed representatives. Full: 2430 tasks.

After pulling the updated source and confirming the old jobs are cancelled, submit everything:

```bash
BUNDLE=bin_packing/hpc_results/ejor_continue_20261029_ready
bash bin_packing/hpc/paper_suite/run_ca_sensitivity_rerun.sh "$BUNDLE" --dry-run
bash bin_packing/hpc/paper_suite/run_ca_sensitivity_rerun.sh "$BUNDLE"
```

Alternatively prepare once, then submit the two flows consecutively without waiting:

```bash
python bin_packing/hpc/paper_suite/rerun_ca_sensitivity.py --bundle "$BUNDLE"
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --flow sensitivity --max-parallel 6
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --flow heuristics --max-parallel 6
```

Do not rerun package preparation while either flow is active. The submission guard allows disjoint flows and rejects duplicates. All existing compatible results remain reusable. See WORK_PLAN.md for details.
