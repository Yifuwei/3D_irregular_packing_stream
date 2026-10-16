# RCL tolerance experiment: one CA plus 50 LS evaluations

Completed: 30/270; successful: 30; failed/invalid: 0.
Current task: engine_cube_seed13_t15.

Each run completes exactly one randomized construction, then 50 evaluated LS neighbors. There is no wall-time cutoff, no restart and no stagnation exit before the evaluation budget. A single-bin solution may stop early; its actual LS count and reason are recorded. LS begins at its own construction and continues from the accepted solution after improvements.
Order seeds 3/7/13; piece_seed 101; position_seed 202; tolerances 10%/15%/20%. Cube PES maximal_residual_box; cylinder PES minimum_aabb_edges_len; bottom_top. Radio 999999 and rho 1. This evaluates the original improved_GRASP, not the separate weight-radio variant.

| Dataset | Shape | Valid 10/15/20 | CA F 10/15/20 | Final F 10% | Final F 15% | Final F 20% | Best final rate | 15 vs 20 W/T/L |
|---|---|---|---|---:|---:|---:|---|---|
| chess | cube | 3/3/3 | 2.732183/2.731542/2.741543 | 2.731512 | 2.727281 | 2.735913 | 15 | 1/1/1 |
| chess | cylinder | 3/3/3 | 2.753059/2.745149/2.753774 | 2.737379 | 2.736523 | 2.747876 | 15 | 2/1/0 |
| engine | cube | 2/2/2 | 2.779699/2.245397/2.776113 | 2.252564 | 2.245397 | 2.247857 | pending | 1/1/0 |
| engine | cylinder | 2/2/2 | 1.754841/1.757885/1.737656 | 1.754841 | 1.757885 | 1.731901 | pending | 0/0/2 |
| Merged2_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged2_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged1_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged1_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged3_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged3_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged4_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged4_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged5_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| Merged5_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| St_05_Example3_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| St_05_Example3_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| liu | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| liu | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| shapesnew | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| shapesnew | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st04_Example2_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st04_Example2_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st04_Example3_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st04_Example3_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st04_Example4_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st04_Example4_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st04_Example5_normal | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st04_Example5_normal | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st05_e2 | cube | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |
| st05_e2 | cylinder | 0/0/0 | pending/pending/pending |  |  |  | pending | 0/0/0 |

## Interpretation

Lower N-U* is better. Best rates are assigned only after all nine cases in a dataset/shape group succeed. Ties retain all tied rates. Raw objectives are not pooled across datasets. Failures/missing cases are never zero-filled. CA and final quality, improvement, runtime, standard deviation, actual LS counts, RCL means/frequencies and paired outcomes are saved in CSV.
These are parameter-selection results on three order seeds. Validate selected rates with held-out seeds before generalization claims. Equal search work is not equal wall time. Geometry validation checks IDs, bounds, cylinder mask, overlap and occupancy, not full insertion trajectories.
Files: results.csv, best_tolerance_by_dataset.csv, summary_by_dataset.csv, paired_comparison.csv, rcl_frequency.csv, per-case .json/.status.json/.log/.rcl.csv/.ls.csv, manifest.json and frozen_runtime/.
