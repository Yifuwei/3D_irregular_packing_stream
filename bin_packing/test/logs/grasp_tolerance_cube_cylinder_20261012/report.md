# GRASP tolerance comparison: cube and cylinder

Completed: 4/54; valid successes: 4; failed/invalid: 0.
Currently running: chess_cylinder_seed3_t20.
Only improved_GRASP is compared, at position tolerances 10%, 15%, and 20%. Order seeds 3/7/13; piece seed 101; position seed 202. Each run has a 180-second wall budget and 20 LS evaluations per restart. All runs use bottom/top candidates and original instance dimensions.
Cube PES: maximal_residual_box. Cylinder PES: minimum_aabb_edges_len. Shape-specific results are kept separate. Radio limit 999999; rho 1. Workers run sequentially with identical kernel/IFV warmup outside the algorithm budget.
A running packing call may overrun the time limit. Actual elapsed time and search-call counts are recorded. Invalid/failed results are excluded from quality averages and paired comparisons, never filled with zeros.

| Dataset | Shape | Tolerance | Valid runs | Mean N-U* | Std | Mean N | Mean CA calls | Mean LS calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chess | cube | 10% | 1 | 2.720524 |  | 3.00 | 5.00 | 90.00 |
| chess | cube | 15% | 1 | 2.714132 |  | 3.00 | 4.00 | 60.00 |
| chess | cube | 20% | 1 | 2.720524 |  | 3.00 | 6.00 | 100.00 |
| chess | cylinder | 15% | 1 | 2.735608 |  | 3.00 | 3.00 | 55.00 |

## Paired outcomes against 10%

| Shape | Challenger | Complete pairs | Wins | Ties | Losses |
| --- | --- | --- | --- | --- | --- |
| cube | 15% | 1 | 1 | 0 | 0 |
| cube | 20% | 1 | 0 | 1 | 0 |
| cylinder | 15% | 0 | 0 | 0 | 0 |
| cylinder | 20% | 0 | 0 | 0 | 0 |

Negative paired delta means lower (better) N-U*. Different datasets are not pooled by raw objective means. RCL frequencies describe each tolerance's actual construction trajectory, including evaluated orientations that were not selected.
This pilot uses three order seeds and fixed algorithm RNG streams. It does not establish performance across other algorithm seeds, all datasets, or full accessibility trajectories. Fewer than two constructions indicate limited evidence about repeated GRASP diversification.
Validation reconstructs placements without modulo wrapping and checks all piece IDs, array bounds, cylinder mask membership, overlap, and occupancy consistency. Full insertion trajectories are not validated.
Files: results.csv; paired_comparison.csv; summary_by_dataset.csv; rcl_frequency.csv; per-run JSON, RCL CSV, and logs; source_snapshot/; manifest.json.
