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

## One-shot HPC submission

Sensitivity, RCL and ILS kick tuning are independent branches after the fixed capacity calibration. Full requires the selected RCL and kick parameters, but does not require sensitivity to succeed.

```text
Calibration (already completed)
  +-- Sensitivity (486 pending; independent)
  +-- RCL selection -------------------+
  +-- ILS kick selection --------------+--> Full
```

The per-package GPU maximum is 6 for the entire graph, not 6 for each branch. Initially three pending branches each receive 2 slots. Collectors automatically transfer released slots using Slurm ArrayTaskThrottle. At most one memory-group/chunk array within each stage runs at a time. After a branch finishes or fails, remaining eligible stages may use its slots, up to 6 in total. Full gets its allocation before its prerequisite collectors exit successfully. If sensitivity is still active when full starts, they share the six slots; sensitivity success is not a dependency. Low concurrency settings use serial resource lanes rather than allocating more GPUs than requested.

The package remains bin_packing/hpc_results/ejor_continue_20261029_ready. Source CA_sensitivity_check remains unchanged. The user has cancelled the old jobs; preparation also rejects any active jobs still targeting this package or unresolved task locks. Completed compatible reports remain reusable.

After pushing/pulling the updated paper_suite source, from the HPC repository root in the packing environment:

```bash
BUNDLE=bin_packing/hpc_results/ejor_continue_20261029_ready

# Prepare/update the existing package and preview ALL remaining stages
bash bin_packing/hpc/paper_suite/run_ca_sensitivity_rerun.sh "$BUNDLE" --dry-run

# One command submits sensitivity + RCL + kick tuning + full
bash bin_packing/hpc/paper_suite/run_ca_sensitivity_rerun.sh "$BUNDLE"
```

When using the already-synchronized package's driver directory, the equivalent entry is:

```bash
bash "$BUNDLE/driver/run_ca_sensitivity_rerun.sh" "$BUNDLE"
```

All three wrappers (submit_all.sh, run_all.sh, run_ca_sensitivity_rerun.sh) and the suite submit-all CLI now default to --through full. Explicit --through sensitivity is still available if only the diagnostic rerun is intended. Never submit a second overlapping graph while jobs from the same package are active.

Missing GPU tasks in this prepared graph: sensitivity 486, RCL 3240, tuning 486, full 2430. Check CPU collection logs and each stage's completion.json/selection.json. submission_history/concurrency.json records dynamic slot allocations. No Slurm jobs are submitted by local dry runs.

Outputs: results/sensitivity, results/rcl, results/tuning, results/full; logs under logs/. sensitivity_rerun_plan.json states retained/pending counts. Original fixed capacities and the 324 compatible sensitivity reports remain preserved.
