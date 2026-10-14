# GRASP position-tolerance experiment

## Objective

Compare final `F = best_N - best_U_star` for position-RCL tolerances 0.10, 0.15, and 0.20. Lower F is better. Use only `improved_GRASP`, holding construction scoring, orientation selection, local search, and constraints constant. This isolates tolerance from the other differences between original and improved GRASP.

## Fixed configuration

| Setting | Value |
| --- | --- |
| Algorithm | improved_GRASP |
| Container | cube; original instance dimensions |
| PES and orientation scoring | maximal_residual_box, including the existing distance-to-origin term |
| Candidate generation | bottom_top AABB extrema; at most 16 unique positions per orientation |
| Active RCL tolerance | 0.10, 0.15, or 0.20 for the entire run |
| Piece seed | 101 |
| Position seed | 202 |
| Radio limit / rho | 999999 / 1, as in the existing geometry-focused launcher tests |
| Wall-clock search budget | 180 seconds, including initial construction, copies, and evaluation |
| LS evaluations per restart | 20 |
| Consecutive-rejection threshold | 20, the default derived from the per-restart LS budget |
| Restart limit | None; controlled by the shared time budget |
| Visualization | Disabled |

Each run initializes the two independent search Generators once. Do not reseed at individual moves or restarts. Each paired instance uses identical order, piece, and position seeds across all three tolerances. Different candidate sets and trajectories can still produce different choices.

## Stage 1: pilot matrix

- Datasets: chess, engine, Merged2_normal.
- Order seeds: 3, 7, 13.
- Three tolerances per dataset/order pair.
- Total: 3 x 3 x 3 = 27 runs.
- Allocated search time: 81 minutes, excluding startup, equal warmup, validation, and non-preemptive-call overruns.

Use the pilot primarily to validate the implementation and budget. Record completed construction counts, LS calls, and whether the run reaches a second construction. A single construction followed by LS does not evaluate GRASP's repeated diversification.

If any tolerance usually fails to reach a second construction on a dataset, raise the common budget for that dataset to 600 seconds for all three tolerances and rerun the matched cases. Do not give more time to only the worse-performing tolerance. Record the chosen budget before the full matrix; report results within each dataset's matched budget.

## Stage 2: full matrix

Datasets: chess, engine, liu, Merged1_normal, Merged2_normal, Merged3_normal, Merged4_normal, Merged5_normal, shapesnew, St_05_Example3_normal, st04_Example2_normal, st04_Example3_normal, st04_Example4_normal, st04_Example5_normal, st05_e2.

Order seeds: 3, 7, 13, 19, 53.

Total: 15 x 5 x 3 = 225 runs. At 180 seconds each, allocated search time is 11.25 hours before overhead and overruns. Reuse pilot results only if source, configuration, budget, and warmup protocol match the final experiment exactly.

Keep cylinders and other PES strategies out of this first matrix. They need separate matched experiments; do not pool their RCL statistics with this configuration.

## Execution protocol

1. Freeze and record a source snapshot, Git revision, environment versions, and GPU model. Uncommitted edits require a snapshot in addition to a Git revision.
2. Run one worker process per configuration, with fresh NFV/IFV pools. Run GPU workers sequentially on this single-GPU machine.
3. Use the same CUDA/JIT warmup protocol before each measured algorithm run, and exclude warmup from the search budget.
4. Rotate tolerance order between dataset/order pairs: 10/15/20, then 15/20/10, then 20/10/15. This reduces systematic run-order effects without consuming the algorithm RNG streams.
5. Record actual wall time separately from the requested budget. A running packing call may finish beyond the deadline; evaluate it according to the existing improved_GRASP behavior.
6. Validate piece counts, boundaries, voxel occupancy, and overlaps. Record failures separately; do not treat failed or missing runs as zero-valued results. This validation does not certify complete accessibility trajectories.

The generated CSV plans contain concrete launcher commands. The existing CLI supports tolerance and LS/time budgets. A measurement worker should additionally wrap construction/repacking calls, as in `compare_grasp.py`, and attach `rcl_observer` through `code_launcher.testing` to collect counters and RCL frequencies. CSV commands alone produce the standard launcher metrics, not the extra counters.

## Required records

For each run, save dataset, instance ID, order_seed, piece_seed, position_seed, tolerance, PES, dimensions, requested budget, actual elapsed time, source snapshot ID, and validation status.

Primary metrics:

- Final N, U_star, and F = N - U_star.
- Initial N, U_star, and initial F, to distinguish construction quality from LS improvement.

Diagnostic metrics:

- Complete construction count and LS call count.
- Best-so-far F versus wall time, especially at 60, 120, and 180 seconds. Use only results known by each checkpoint; do not attribute a later completed call to an earlier checkpoint.
- RCL-size frequencies at the active tolerance, separately for all evaluated RCL calls and those used for placement.
- Time spent in construction, repacking, copying/evaluation, and logging where measurable.

Collect RCL frequencies during each tolerance's actual trajectory. The existing profile estimated several thresholds on a shared 10% trajectory; it cannot substitute for these full-run outcome comparisons.

## Analysis and decision rule

For each dataset/order pair, compare the three final F values. Relative to the 10% baseline, define delta_15 = F_15 - F_10 and delta_20 = F_20 - F_10. Negative deltas are improvements. Use an absolute numerical equality tolerance of 1e-10 for ties, not a substantive performance margin.

Report paired wins/ties/losses, mean and median deltas within each dataset, and whether bin count changes. Report all five order-seed outcomes rather than just the single best run.

Across datasets, give each dataset equal weight when summarizing ranks or normalized gaps. Do not choose a global tolerance by averaging raw F across differently sized datasets, which would overweight instances with larger bin counts.

Select a larger tolerance only if it improves final packing quality consistently under matched budgets. An average RCL size of 3-4 is a selectivity diagnostic, not an acceptance criterion. If 15% and 20% are indistinguishable in outcome, retain the simpler established baseline unless another measured advantage justifies changing it.

Fixed search seeds make this a controlled comparison across object orders. They do not measure variability across algorithm random streams. Before claiming broad superiority, repeat on additional search-seed pairs and preferably on held-out datasets.

## Single-run example

From the repository root, in the cutting_stable environment:

```bash
python bin_packing/test/code_launcher.py --dataset chess --algorithms improved_GRASP --seeds 13 --iterations 20 --time-limit 180 --range bottom_top --pes maximal_residual_box --rcl-tolerance 0.15 --output logs/grasp_tolerance_pilot/chess_seed13_t15.csv
```

The launcher resolves this output under its test working directory. Use a fresh output root for each frozen experiment. The CSV plans are prepared only; no batch runs have been started.
