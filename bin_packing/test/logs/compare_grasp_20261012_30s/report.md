# GRASP comparison

Dataset: chess; seeds: [3, 7, 13]; internal time budget: 30s; LS evaluations per restart: 100.
Both use cube instances, radio limit 999999, rho 1, PES `maximal_residual_box`, and `bottom_top`. Improved position RCL tolerance: 10%.
Runs use separate processes, the same initial RNG seed per pair, and equal CUDA-kernel warmup. RNG streams diverge as algorithms consume different draws.
Original GRASP counts packing-call time; improved GRASP counts wall time including copies and evaluation. Budgets are internal and non-preemptive, so completed calls may overrun. Wall times are recorded separately.

| Seed | Algorithm | N | U* | N-U* | Wall seconds | CA calls | LS calls | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | GRASP | 3 | 0.265639 | 2.734361 | 33.72 | 1 | 2 | True |
| 3 | improved_GRASP | 3 | 0.240795 | 2.759205 | 35.57 | 1 | 0 | True |
| 7 | GRASP | 3 | 0.256546 | 2.743454 | 33.22 | 1 | 0 | True |
| 7 | improved_GRASP | 3 | 0.256546 | 2.743454 | 32.48 | 1 | 0 | True |
| 13 | GRASP | 3 | 0.255370 | 2.744630 | 30.81 | 1 | 0 | True |
| 13 | improved_GRASP | 4 | 0.226382 | 3.773618 | 32.57 | 1 | 0 | True |

Improved GRASP: 0 wins, 1 ties, 2 losses by N-U* (lower is better).
This is a short single-dataset comparison, not evidence of superiority across datasets. Several changes are compared together; their individual effects are not isolated.
Validation reconstructs every placed piece without modulo wrapping and checks piece counts, cube bounds, overlaps, and occupancy consistency. It does not validate full accessibility trajectories.
