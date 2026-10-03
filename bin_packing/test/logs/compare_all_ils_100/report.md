Each algorithm: 100 candidate evaluations including kicks; seed 13; cube; kick trigger 100.
Independent processes; sequential execution; warm-up excluded; original fixes confined to test adapter.

| Dataset | Status | ILS (s) | Improved ILS (s) | Time reduction |
|---|---|---:|---:|---:|
| Merged1_normal | failed | 526.99 | — | — |
| Merged2_normal | passed | 862.41 | 875.32 | -1.50% |
| Merged3_normal | passed | 1155.13 | 726.58 | 37.10% |
| Merged4_normal | passed | 1530.15 | 1243.95 | 18.70% |
| Merged5_normal | running/pending | 1974.53 | — | — |
| St_05_Example3_normal | running/pending | — | — | — |
| chess | passed | 99.82 | 62.40 | 37.48% |
| engine | passed | 333.23 | 426.72 | -28.06% |
| liu | passed | 1514.84 | 1147.96 | 24.22% |
| shapesnew | running/pending | — | — | — |
| st04_Example2_normal | running/pending | — | — | — |
| st04_Example3_normal | running/pending | — | — | — |
| st04_Example4_normal | running/pending | — | — | — |
| st04_Example5_normal | running/pending | — | — | — |
| st05_e2 | running/pending | — | — | — |

Positive reduction means improved ILS is faster. These are complete-algorithm comparisons, not isolated skip-bin timings.