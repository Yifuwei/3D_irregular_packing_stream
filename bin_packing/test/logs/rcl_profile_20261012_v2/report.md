# Position RCL size profile

Dataset: chess; order seeds: [3, 7, 13, 19, 53]; PES: `maximal_residual_box`; cube bins; piece seed 101; position seed 202.
Each seed runs exactly one complete construction through code_launcher.testing with the actual position tolerance fixed at 10%. The negligible LS time budget stops after construction; no LS or restart is profiled.
All four tolerance sizes are computed from the same unique scored candidates, using best + tolerance * abs(best) and the production roundoff allowance. No extra random numbers are drawn for observation.
A call is one nonempty position RCL for one piece/orientation/bin query. All evaluated orientations are counted, including those not ultimately placed. Empty-bin starting placements and AFR queries that fail before RCL creation are not RCL calls.
Means weight each nonempty RCL call equally. used_for_placement reports only RCL calls matching final constructed placements; empty-bin placements remain excluded.

| Scope | Tolerance | Calls | Mean size | Median | Q25–Q75 | Size 1 | Size 3–4 | All candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_calls | 5% | 1227 | 2.522 | 2.0 | 1.0–3.0 | 26.4% | 18.8% | 0.4% |
| all_calls | 10% | 1227 | 2.763 | 2.0 | 2.0–3.0 | 21.2% | 24.8% | 0.7% |
| all_calls | 15% | 1227 | 2.982 | 2.0 | 2.0–4.0 | 18.2% | 30.1% | 0.8% |
| all_calls | 20% | 1227 | 3.286 | 3.0 | 2.0–4.0 | 14.3% | 29.2% | 1.0% |
| used_for_placement | 5% | 145 | 2.345 | 2.0 | 1.0–2.0 | 30.3% | 17.2% | 2.1% |
| used_for_placement | 10% | 145 | 2.483 | 2.0 | 1.0–3.0 | 26.9% | 20.7% | 2.8% |
| used_for_placement | 15% | 145 | 2.641 | 2.0 | 2.0–3.0 | 22.8% | 24.8% | 3.4% |
| used_for_placement | 20% | 145 | 2.869 | 2.0 | 2.0–4.0 | 20.0% | 26.9% | 4.1% |

A mean near 3–4 is a selectivity target, not proof of better final packing. All retained candidates meet the local PES threshold, but PES is not the final N-U* objective. The raw JSON files retain every scored candidate; per_call.csv lists all call sizes and summary.csv includes per-order-seed means.

## RCL size frequencies (all evaluated nonempty RCL calls)

| Tolerance | Size 1 | Size 2 | Size 3 | Size 4 | Size >=5 |
| --- | --- | --- | --- | --- | --- |
| 5% | 324 (26.4%) | 558 (45.5%) | 110 (9.0%) | 121 (9.9%) | 114 (9.3%) |
| 10% | 260 (21.2%) | 514 (41.9%) | 153 (12.5%) | 151 (12.3%) | 149 (12.1%) |
| 15% | 223 (18.2%) | 452 (36.8%) | 184 (15.0%) | 185 (15.1%) | 183 (14.9%) |
| 20% | 175 (14.3%) | 427 (34.8%) | 154 (12.6%) | 204 (16.6%) | 267 (21.8%) |
