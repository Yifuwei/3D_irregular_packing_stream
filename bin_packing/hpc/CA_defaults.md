# Empirical CA defaults

Source: `hpc result/method_average_across_datasets.xlsx`, sheet `Method averages`.
Fifteen datasets, five seeds per configuration, geometry-only constraints.
Use the user-selected combinations with the highest mean U* for each shape.

| Shape | PES / SCH_nesting_strategy | selection_range | Mean N | Mean U* | Mean CA seconds |
|---|---|---|---:|---:|---:|
| cube | maximal_residual_box | bottom | 3.60 | 0.285685 | 73.29 |
| cylinder | minimum_aabb_edges_len | bottom_top | 2.946667 | 0.259745 | 430.16 |

These are highest average U* values among tested combinations, not a guarantee
of the best result for each dataset or for subsequent ILS.

`src/ca_parameters.py` defines the shared defaults. `test/code_launcher.py`
uses them when parameter lists are omitted; `--full-matrix` still explicitly
scans all 12 combinations. `hpc/benchmark_ils.py` passes the same defaults
to both original ILS and improved ILS and saves them in logs and result JSON.
The existing multiseed benchmark still loads cube instances only.
Direct function calls with explicitly supplied parameters continue to use those parameters.
Only PES and selection range changed. Accessibility, orientation settings,
candidate budget, and seeds stay at their existing benchmark settings;
this does not reproduce every setting of the earlier CA experiment.

Upload `src/ca_parameters.py`, `hpc/benchmark_ils.py`, and the updated launcher
before a new submission. Do not replace source files midway through a submitted
array: the multiseed manifest checks source hashes. Existing submitted jobs and
old results are not updated by these local changes.
