# Chess: improved_ILS versus improved_ILS_weight_radio

Completed: 18/18 runs; 9/9 paired settings.

Seed 13; 100 evaluated candidates per method. Same production improved_ILS selection, acceptance and stopping controller; only repacking differs. Kick trigger 101 (zero kicks within this budget). Each method uses the same fixed CA initial layout, hard radiation and volume limits. Weight equals volume, so its limit is rho × bin volume and is redundant with the filling constraint. Higher U* is better. Positive time reduction means weight/radio is faster.

Fresh sequential processes, untimed CUDA/CA warmup, fresh formal pools, alternating method order. Wall time includes formal CA and repacking, excludes instance loading and warmup. Same seed/policy need not produce identical candidates after accepted layouts diverge. Single-seed exploratory comparison; timings include event logging and do not estimate seed variance.

| Radiation | Max filling | Initial bins | improved_ILS bins | Weight/radio bins | improved_ILS U* | Weight/radio U* | improved_ILS s | Weight/radio s | Time reduction |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 0.2 | 4 | 4 | 4 | 0.1923 | 0.1923 | 23.56 | 20.46 | 13.17% |
| 2000 | 0.3 | 3 | 3 | 3 | 0.2682 | 0.2682 | 55.46 | 42.55 | 23.29% |
| 2000 | 0.4 | 3 | 3 | 3 | 0.2682 | 0.2682 | 57.92 | 45.01 | 22.30% |
| 3000 | 0.2 | 4 | 4 | 4 | 0.1923 | 0.1923 | 26.63 | 21.28 | 20.09% |
| 3000 | 0.3 | 3 | 3 | 3 | 0.2682 | 0.2682 | 56.46 | 42.95 | 23.93% |
| 3000 | 0.4 | 3 | 3 | 3 | 0.2682 | 0.2682 | 57.34 | 45.75 | 20.20% |
| 4000 | 0.2 | 4 | 4 | 4 | 0.1923 | 0.1923 | 26.62 | 21.13 | 20.62% |
| 4000 | 0.3 | 3 | 3 | 3 | 0.2682 | 0.2682 | 55.16 | 44.22 | 19.84% |
| 4000 | 0.4 | 3 | 3 | 3 | 0.2682 | 0.2682 | 56.98 | 45.52 | 20.12% |

Every completed best solution passed: all 32 pieces exactly once, no overlapping voxels, radiation ≤ specified maximum and filling ≤ rho. Paired initial layouts and production source hashes match. See each run’s result.json and events.jsonl for bin counts, radiation, filling, candidate tape and executed skip/search branches.

Independent trace audit passed: 14028 metric/visit-order queries and 3683 failure-cache skips. Identical actual candidate tapes in 9/9 pairs.

Weight/radio produced fewer/same/more bins in 0/9/0 settings. Summed formal wall time: 416.14s versus 328.87s; reduction 20.97%. This aggregate is descriptive, not a statistical significance result.

Interpretation: all nine settings retained the CA bin count and U*. This experiment shows lower measured wall time, not better final packing quality. The replacement contains both ranked search and state-keyed reuse, so timing gains cannot be attributed to the metric alone. Chess has total radiation 2170.858673; limits 3000 and 4000 cannot reject a bin on radiation alone, although they still change score values. Since weight is proportional to volume, this experiment has no independent weight constraint. More seeds, repeated timing and tighter radiation cases are needed for a broader conclusion.

The test invokes the actual production improved_ILS controller in a private namespace and substitutes RankedRepacker for the new method. It does not compare the experimental standalone framework's different selector. Production algorithm files were not edited for this experiment.

Reproduce from bin_packing/test with cutting_stable Python:
```powershell
python compare_weight_radio_grid.py --project .. --output logs/weight_radio_grid_100_seed13 --iterations 100 --seed 13
python audit_weight_radio_grid.py logs/weight_radio_grid_100_seed13
```
Existing completed result files are reused; choose a new output directory for a fresh experiment. comparison.csv is the combined table; each parameter/method subdirectory contains result.json and events.jsonl. audit_summary.json contains independent trace verification counts.
