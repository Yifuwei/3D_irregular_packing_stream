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

## Simple submission: two independent flows, six tasks each

There is no dynamic concurrency coordinator or automatic scontrol update.

```text
Fixed capacity calibration (already completed)
  +-- Sensitivity                         [up to 6 GPU tasks]
  +-- RCL -> ILS kick tuning -> Full       [up to 6 GPU tasks]
```

The two flows may overlap, up to 12 GPU tasks combined. Within the heuristic flow, memory groups/chunks and the three experiment stages run sequentially, so its maximum remains six. Sensitivity success is not a prerequisite of the heuristic flow.

### Prepare/update once, then submit everything with one command

After the old jobs have been cancelled and the updated paper_suite source is pulled on HPC, from the repository root:

```bash
BUNDLE=bin_packing/hpc_results/ejor_continue_20261029_ready
bash bin_packing/hpc/paper_suite/run_ca_sensitivity_rerun.sh "$BUNDLE" --dry-run
bash bin_packing/hpc/paper_suite/run_ca_sensitivity_rerun.sh "$BUNDLE"
```

The wrapper prepares the existing package and defaults to --flow all --through full. It preserves compatible results. No new package is created.

### Or submit the two flows separately, without waiting

```bash
BUNDLE=bin_packing/hpc_results/ejor_continue_20261029_ready

# Update the package once while it is idle
python bin_packing/hpc/paper_suite/rerun_ca_sensitivity.py --bundle "$BUNDLE"

# Submit sensitivity first, then immediately submit the heuristic chain
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --flow sensitivity --max-parallel 6
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --flow heuristics --max-parallel 6
```

Do not prepare/update the package again between these two submissions. Different flows can share the package while running; a duplicate submission of the same flow is rejected. Only locks belonging to the requested flow block its submission. Separate flows require the existing calibration reports and fixed-capacity selection to be complete.

Remaining tasks: sensitivity 486, RCL 3240, ILS tuning 486, full 2430. RCL aggregates all 15 datasets; kick strength aggregates the original three representatives together. Fixed PES/range and original capacity calibration are unchanged.

Suspended/configuring jobs are treated as active. Frozen text hashes ignore only LF/CRLF differences across Windows/Linux, while binary files and substantive code/input changes remain strictly verified. Outputs remain under the existing bundle's results/ and logs/ directories.
