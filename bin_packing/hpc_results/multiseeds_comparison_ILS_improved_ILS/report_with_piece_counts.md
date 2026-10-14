# Five-seed ILS comparison

15 datasets; seeds 3, 7, 13, 19, 53; 100 evaluations per algorithm per seed; all 75 paired runs passed; no kicks.
Times are complete-algorithm wall times in seconds, including CA and search but excluding warmup. Mean ± sample SD across five seeds.
Bins are mean best bin counts across five seeds; decimal means are not fractional physical bins. Piece counts come from all five local cube instance files and agree across seeds.
Time reduction is 100 × (1 − Improved mean / ILS mean); negative means Improved is slower. This differs from the paired-percentage median in the earlier report.

| Dataset | Pieces | ILS time (s) | Improved time (s) | Mean time reduction | ILS bins (mean) | Improved bins (mean) |
|---|---:|---:|---:|---:|---:|---:|
| Merged1_normal | 15 | 131.80 ± 15.80 | 124.38 ± 17.33 | 5.63% | 1.60 | 1.60 |
| Merged2_normal | 30 | 301.73 ± 38.36 | 274.97 ± 64.65 | 8.87% | 2.60 | 2.60 |
| Merged3_normal | 45 | 500.15 ± 67.27 | 428.49 ± 73.17 | 14.33% | 3.60 | 3.40 |
| Merged4_normal | 60 | 760.15 ± 83.19 | 627.42 ± 85.11 | 17.46% | 4.20 | 4.20 |
| Merged5_normal | 75 | 1037.61 ± 122.92 | 800.34 ± 91.83 | 22.87% | 5.00 | 5.00 |
| St_05_Example3_normal | 25 | 1134.49 ± 148.67 | 1306.07 ± 62.11 | -15.12% | 2.00 | 2.00 |
| chess | 32 | 46.80 ± 5.17 | 46.84 ± 7.25 | -0.10% | 3.00 | 3.00 |
| engine | 97 | 235.47 ± 32.91 | 219.73 ± 40.29 | 6.69% | 2.20 | 2.20 |
| liu | 36 | 756.92 ± 43.37 | 398.93 ± 73.59 | 47.30% | 8.80 | 8.80 |
| shapesnew | 43 | 12.37 ± 1.26 | 6.01 ± 1.01 | 51.41% | 9.40 | 9.40 |
| st04_Example2_normal | 20 | 470.18 ± 72.32 | 453.32 ± 92.84 | 3.59% | 1.80 | 1.80 |
| st04_Example3_normal | 30 | 745.17 ± 80.83 | 766.88 ± 85.45 | -2.91% | 2.00 | 2.00 |
| st04_Example4_normal | 40 | 1077.44 ± 120.91 | 897.69 ± 143.47 | 16.68% | 2.60 | 3.00 |
| st04_Example5_normal | 50 | 1382.55 ± 149.90 | 1219.54 ± 96.21 | 11.79% | 3.00 | 3.00 |
| st05_e2 | 12 | 469.52 ± 49.71 | 455.65 ± 51.06 | 2.95% | 1.40 | 1.20 |

Total measured runtime: ILS 45311.78 s; Improved 40131.34 s; reduction 11.43%.
Only aggregate result files were downloaded; raw worker logs/JSON and submission manifest are unavailable for code/input/hardware provenance checks.