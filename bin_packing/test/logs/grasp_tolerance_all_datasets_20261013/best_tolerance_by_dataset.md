# Dataset-specific RCL tolerance results

Status: partial; completed 106/270.
Current task: Merged4_normal_cylinder_seed13_t10.

Each dataset and container shape is evaluated separately. Lower N-U* is better. Three order seeds (3/7/13), piece seed 101, position seed 202, 180 seconds, 20 LS evaluations per restart; shape-specific PES and bottom/top candidates.
The best rate is selected only when all nine cases in a dataset/shape group are valid. Missing or failed runs are never filled with zeros. Ties retain all tied rates.

| Dataset | Shape | Valid runs 10/15/20 | Mean F 10% | Mean F 15% | Mean F 20% | Best rate | 15 vs 20 W/T/L |
|---|---|---|---:|---:|---:|---|---|
| chess | cube | 3/3/3 | 2.720964 | 2.722989 | 2.724663 | 10 | 1/1/1 |
| chess | cylinder | 3/3/3 | 2.744434 | 2.737465 | 2.745149 | 15 | 3/0/0 |
| engine | cube | 3/3/3 | 2.428878 | 2.424955 | 2.428411 | 15 | 1/1/1 |
| engine | cylinder | 3/3/3 | 1.762133 | 1.760106 | 1.750001 | 20 | 1/1/1 |
| Merged2_normal | cube | 3/3/3 | 2.675641 | 2.322609 | 2.675641 | 15 | 1/2/0 |
| Merged2_normal | cylinder | 3/3/3 | 1.989967 | 1.643847 | 2.009838 | 15 | 2/0/1 |
| Merged1_normal | cube | 3/3/3 | 1.687293 | 1.001860 | 1.001860 | 15/20 | 0/3/0 |
| Merged1_normal | cylinder | 3/3/3 | 1.396361 | 1.396361 | 1.396361 | 10/15/20 | 0/3/0 |
| Merged3_normal | cube | 3/3/3 | 3.670297 | 3.329723 | 3.327941 | 20 | 2/0/1 |
| Merged3_normal | cylinder | 3/3/3 | 2.661848 | 2.663778 | 2.672955 | 10 | 1/0/2 |
| Merged4_normal | cube | 3/3/3 | 4.323011 | 3.980438 | 4.681285 | 15 | 3/0/0 |
| Merged4_normal | cylinder | 2/2/3 | 3.679014 | 4.214447 | 3.676695 | pending | 0/0/2 |
| Merged5_normal | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| Merged5_normal | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |
| St_05_Example3_normal | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| St_05_Example3_normal | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |
| liu | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| liu | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |
| shapesnew | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| shapesnew | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |
| st04_Example2_normal | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| st04_Example2_normal | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |
| st04_Example3_normal | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| st04_Example3_normal | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |
| st04_Example4_normal | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| st04_Example4_normal | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |
| st04_Example5_normal | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| st04_Example5_normal | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |
| st05_e2 | cube | 0/0/0 |  |  |  | pending | 0/0/0 |
| st05_e2 | cylinder | 0/0/0 |  |  |  | pending | 0/0/0 |

## Interpretation

These are parameter-selection results on the same three seeds, not independent confirmation of generalization. Validate chosen rates on held-out seeds before claiming a dataset-specific improvement. RCL means describe all evaluated orientation-level RCL calls, including orientations not ultimately packed.
Some runs may exhaust the budget during construction and execute no LS. The CSV includes zero-LS counts per rate. Non-preemptive packing calls can exceed the nominal budget. Geometry validation checks IDs, boundaries, overlap and occupancy; full accessibility trajectories are not certified.

Saved tables: best_tolerance_by_dataset.csv, summary_by_dataset.csv, paired_comparison.csv, results.csv, rcl_frequency.csv. Per-run JSON, RCL CSV, logs, manifest and source snapshots retain provenance.
