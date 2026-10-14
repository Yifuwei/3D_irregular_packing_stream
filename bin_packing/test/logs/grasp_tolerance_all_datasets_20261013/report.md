# GRASP tolerance comparison: cube and cylinder

Completed: 109/270; valid successes: 109; failed/invalid: 0.
Currently running: none.
Only improved_GRASP is compared, at position tolerances 10%, 15%, and 20%. Order seeds 3/7/13; piece seed 101; position seed 202. Each run has a 180-second wall budget and 20 LS evaluations per restart. All runs use bottom/top candidates and original instance dimensions.
Cube PES: maximal_residual_box. Cylinder PES: minimum_aabb_edges_len. Shape-specific results are kept separate. Radio limit 999999; rho 1. Workers run sequentially with identical kernel/IFV warmup outside the algorithm budget.
A running packing call may overrun the time limit. Actual elapsed time and search-call counts are recorded. Invalid/failed results are excluded from quality averages and paired comparisons, never filled with zeros.

| Dataset | Shape | Tolerance | Valid runs | Mean N-U* | Std | Mean N | Mean CA calls | Mean LS calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chess | cube | 10% | 3 | 2.720964 | 0.003197 | 3.00 | 7.33 | 139.00 |
| chess | cube | 15% | 3 | 2.722989 | 0.008857 | 3.00 | 7.33 | 131.33 |
| chess | cube | 20% | 3 | 2.724663 | 0.006245 | 3.00 | 7.67 | 135.00 |
| chess | cylinder | 10% | 3 | 2.744434 | 0.004166 | 3.00 | 4.33 | 73.00 |
| chess | cylinder | 15% | 3 | 2.737465 | 0.001820 | 3.00 | 4.00 | 67.33 |
| chess | cylinder | 20% | 3 | 2.745149 | 0.003414 | 3.00 | 4.33 | 72.67 |
| engine | cube | 10% | 3 | 2.428878 | 0.614833 | 2.67 | 1.33 | 7.00 |
| engine | cube | 15% | 3 | 2.424955 | 0.615616 | 2.67 | 1.00 | 11.33 |
| engine | cube | 20% | 3 | 2.428411 | 0.602238 | 2.67 | 1.00 | 7.33 |
| engine | cylinder | 10% | 3 | 1.762133 | 0.019219 | 2.00 | 1.00 | 0.00 |
| engine | cylinder | 15% | 3 | 1.760106 | 0.004264 | 2.00 | 1.00 | 0.00 |
| engine | cylinder | 20% | 3 | 1.750001 | 0.028205 | 2.00 | 1.00 | 0.00 |
| Merged2_normal | cube | 10% | 3 | 2.675641 | 0.003310 | 3.00 | 1.33 | 10.33 |
| Merged2_normal | cube | 15% | 3 | 2.322609 | 0.609822 | 2.67 | 1.33 | 12.00 |
| Merged2_normal | cube | 20% | 3 | 2.675641 | 0.003310 | 3.00 | 1.33 | 6.67 |
| Merged2_normal | cylinder | 10% | 3 | 1.989967 | 0.640113 | 2.33 | 1.00 | 0.00 |
| Merged2_normal | cylinder | 15% | 3 | 1.643847 | 0.024512 | 2.00 | 1.00 | 0.00 |
| Merged2_normal | cylinder | 20% | 3 | 2.009838 | 0.621044 | 2.33 | 1.00 | 0.00 |
| Merged1_normal | cube | 10% | 3 | 1.687293 | 0.009691 | 2.00 | 3.00 | 41.00 |
| Merged1_normal | cube | 15% | 3 | 1.001860 | 0.588757 | 1.33 | 3.33 | 33.67 |
| Merged1_normal | cube | 20% | 3 | 1.001860 | 0.588757 | 1.33 | 3.33 | 36.67 |
| Merged1_normal | cylinder | 10% | 3 | 1.396361 | 0.586841 | 1.67 | 1.33 | 9.00 |
| Merged1_normal | cylinder | 15% | 3 | 1.396361 | 0.586841 | 1.67 | 1.33 | 5.33 |
| Merged1_normal | cylinder | 20% | 3 | 1.396361 | 0.586841 | 1.67 | 1.33 | 9.67 |
| Merged3_normal | cube | 10% | 3 | 3.670297 | 0.003879 | 4.00 | 1.00 | 4.33 |
| Merged3_normal | cube | 15% | 3 | 3.329723 | 0.597057 | 3.67 | 1.00 | 5.00 |
| Merged3_normal | cube | 20% | 3 | 3.327941 | 0.614006 | 3.67 | 1.00 | 2.67 |
| Merged3_normal | cylinder | 10% | 3 | 2.661848 | 0.032710 | 3.00 | 1.00 | 0.00 |
| Merged3_normal | cylinder | 15% | 3 | 2.663778 | 0.044552 | 3.00 | 1.00 | 0.00 |
| Merged3_normal | cylinder | 20% | 3 | 2.672955 | 0.027576 | 3.00 | 1.00 | 0.00 |
| Merged4_normal | cube | 10% | 3 | 4.323011 | 0.595599 | 4.67 | 1.00 | 0.00 |
| Merged4_normal | cube | 15% | 3 | 3.980438 | 0.601245 | 4.33 | 1.00 | 0.00 |
| Merged4_normal | cube | 20% | 3 | 4.681285 | 0.012720 | 5.00 | 1.00 | 0.00 |
| Merged4_normal | cylinder | 10% | 3 | 3.673565 | 0.017227 | 4.00 | 1.00 | 0.00 |
| Merged4_normal | cylinder | 15% | 3 | 4.032257 | 0.598371 | 4.33 | 1.00 | 0.00 |
| Merged4_normal | cylinder | 20% | 3 | 3.676695 | 0.003784 | 4.00 | 1.00 | 0.00 |
| Merged5_normal | cube | 10% | 1 | 5.665892 |  | 6.00 | 1.00 | 0.00 |

## Paired outcomes against 10%

| Shape | Challenger | Complete pairs | Wins | Ties | Losses |
| --- | --- | --- | --- | --- | --- |
| cube | 15% | 18 | 9 | 4 | 5 |
| cube | 20% | 18 | 6 | 6 | 6 |
| cylinder | 15% | 18 | 7 | 3 | 8 |
| cylinder | 20% | 18 | 7 | 5 | 6 |

Negative paired delta means lower (better) N-U*. Different datasets are not pooled by raw objective means. RCL frequencies describe each tolerance's actual construction trajectory, including evaluated orientations that were not selected.
This pilot uses three order seeds and fixed algorithm RNG streams. It does not establish performance across other algorithm seeds, all datasets, or full accessibility trajectories. Fewer than two constructions indicate limited evidence about repeated GRASP diversification.
Validation reconstructs placements without modulo wrapping and checks all piece IDs, array bounds, cylinder mask membership, overlap, and occupancy consistency. Full insertion trajectories are not validated.
Files: results.csv; paired_comparison.csv; summary_by_dataset.csv; rcl_frequency.csv; per-run JSON, RCL CSV, and logs; source_snapshot/; manifest.json.
