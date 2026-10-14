# Chess weight/radiation logic verification

Real chess radiation proxies; seed 13; 10 ILS evaluations per limit. Weight factor=1 and per-bin weight cap=80% container volume.
Each run also repeats an identical initial-state repack twice to explicitly verify failure-cache reuse; these repetitions are not ILS evaluations.

| Radio limit | Initial bins | Best bins | ILS cached-failure skips | Repeat-test skips | Metric/order queries audited |
|---|---:|---:|---:|---:|---:|
| 2000 | 3 | 3 | 0 | 19 | 283 |
| 3000 | 3 | 3 | 2 | 23 | 283 |
| 4000 | 3 | 3 | 4 | 23 | 283 |

All hard constraints, metric recomputations, attempt order and matching failure history checks passed.
Piece indices are zero-based; bin labels are one-based. This is logic validation, not evidence of global optimality or performance gains.