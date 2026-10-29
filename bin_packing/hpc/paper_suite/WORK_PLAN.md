# Current work plan — 2026-10-29

## Parameter-selection rule

CA/PES and RCL parameters are selected from aggregate performance across all 15 datasets, never separately per dataset. Explicit exception confirmed by the user: ILS kick strength is selected by aggregating the original three representative datasets together, producing one shared winner. Cube/cylinder may use different shared settings. Smoke/resource probes are diagnostics, not parameter-selection evidence. Missing entire datasets blocks selection and full-experiment handoff.

| Bin shape | Shared PES | Selection range |
|---|---|---|
| Cube | maximal_residual_box | bottom |
| Cylinder | minimum_aabb_edges_len | bottom_top |

These explicit user-confirmed configurations apply to CA sensitivity, RCL tests, ILS tuning and all three full-test algorithms. Direction and placement use the same PES. The randomized RCL candidate pool still uses its prescribed top/bottom extremal-position mechanism.

The original five-seed capacity calibration is fixed and remains unchanged. Historical per-dataset CA choices in ca_parameters describe calibration provenance only; they must not be used to choose new algorithm parameters. geometry_ca_parameters records the geometric baseline under the shared settings for sensitivity comparison.

## Status and experiment scale

| Stage | Dataset coverage | Tasks | Current action / budget |
|---|---|---:|---|
| Capacity calibration | 15 datasets × 2 shapes × 5 reference seeds | 150 | Preserve all original references and capacities; no rerun |
| CA sensitivity | 15 × 2 × 3 experiment seeds × 9 W/R pairs | 810 | Keep 324 matching reports; archive 486 wrong-configuration reports and rerun those cases |
| RCL selection | 15 × 2 × 3 seeds × 4 RCL choices × 9 W/R pairs | 3240 | One randomized construction per run; 10%, 20%, Top10, Top30; select one global RCL across all 15 datasets |
| ILS kick-strength selection | 3 representatives × 2 × 3 seeds × 3 strengths × 9 W/R pairs | 486 | 1800s/run; compare 25/50/75%; select one shared strength across shapesnew, Merged4_normal, st04_Example5_normal |
| Full heuristic comparison | 15 × 2 × 3 seeds × 9 W/R pairs × 3 algorithms | 2430 | 3600s/run; ILS, GRASP, GRASP×ILS; inherit global RCL/kick settings |

Reference seeds: 3,7,13,19,53. Experiment seeds: 3,7,13. Kick tuning remains on shapesnew, Merged4_normal and st04_Example5_normal: 486 tasks. Total plan entries: 7116. After retaining 150 calibration and 324 sensitivity cases, 6642 new experiment tasks remain before any compatible later-stage results are considered.

Best means smallest arithmetic mean N−U* across matched cells. Equal experiment design gives each dataset equal weight. Selection records include dataset names/count, scope and metric. Full requires RCL to cover all 15 datasets and kick strength to cover the confirmed three representatives; each must produce a shared aggregate winner.

## Algorithm rules and resources

- ILS and hybrid select the highest dynamic size/order score among unvisited pieces. Improvement or kick resets the sweep and failure counter.
- One LS iteration is one completed candidate-orientation repacking call. Default kick threshold is ceil(piece_count/2) rejected LS repacking calls.
- ILS continues until its overall time budget. Hybrid uses piece_count LS repacking calls per construction; improvement does not reset this budget. Kicks are outside the LS count but inside wall time. No extra kick at the last LS budget boundary.
- GRASP uses fixed input order. First improvement restarts the scan. A complete failed scan of all pieces/orientations triggers reconstruction. Time/budget interruption does not certify local optimum.
- Default concurrency: 6. Each GPU task: 1 GPU, 4 CPU cores; chess/shapesnew use 16G RAM, other datasets 32G. CPU collectors use 1 core/2G. Default Slurm job walltime is 12h, separate from algorithm budgets.

## Sensitivity rerun on HPC

The existing package is updated in place: bin_packing/hpc_results/ejor_continue_20261029_ready. Original source bundle CA_sensitivity_check is preserved. There is no automatic remote cancellation or synchronization.

1. Stop only the active jobs targeting this package. If cancelled tasks leave locks, verify no jobs remain and use the suite unlock command.
2. Synchronize the updated existing package to HPC, including driver, manifest.json and parameter_plan.json. Additive uploads may leave old result files; the preparation script detects and archives those incompatible reports.
3. From the HPC repository root with the packing environment active:

```bash
BUNDLE=bin_packing/hpc_results/ejor_continue_20261029_ready

# Prepare/check the sensitivity-only submission (does not submit GPU jobs)
bash "$BUNDLE/driver/run_ca_sensitivity_rerun.sh" "$BUNDLE" --dry-run

# Submit only the missing/wrong-configuration sensitivity cases
bash "$BUNDLE/driver/run_ca_sensitivity_rerun.sh" "$BUNDLE"

# After sensitivity reaches 810/810 success, inspect completion.json
cat "$BUNDLE/results/sensitivity/completion.json"

# Preview then submit subsequent experiments with all-15 RCL and three-dataset kick tuning
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --through full --max-parallel 6 --dry-run
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --through full --max-parallel 6
```

The rerun wrapper defaults to --through sensitivity. It preserves valid matching results and archives incompatible reports and stale aggregate files; it is safe to repeat after confirming no active jobs. Later stages are enabled explicitly by --through full and wait for successful collectors. RCL/tuning/full reports from old parameter/case definitions do not count as new-config results. Do not manually relabel old incompatible reports.

Outputs: results/sensitivity, results/rcl, results/tuning, results/full; logs under logs/. sensitivity_rerun_plan.json states preserved/pending counts. Configuration/report backups are retained inside the package.
