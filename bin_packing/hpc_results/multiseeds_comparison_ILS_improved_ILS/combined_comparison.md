| Dataset | ILS time (mean ± SD, s) | Improved time (mean ± SD, s) | Paired reduction median | ILS bins (mean) | Improved bins (mean) | ILS U* (mean) | Improved U* (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Merged1_normal | 131.80 ± 15.80 | 124.38 ± 17.33 | 5.09% | 1.60 | 1.60 | 0.3273 | 0.3259 |
| Merged2_normal | 301.73 ± 38.36 | 274.97 ± 64.65 | 14.14% | 2.60 | 2.60 | 0.3443 | 0.3410 |
| Merged3_normal | 500.15 ± 67.27 | 428.49 ± 73.17 | 8.99% | 3.60 | 3.40 | 0.3499 | 0.3595 |
| Merged4_normal | 760.15 ± 83.19 | 627.42 ± 85.11 | 14.70% | 4.20 | 4.20 | 0.3584 | 0.3592 |
| Merged5_normal | 1037.61 ± 122.92 | 800.34 ± 91.83 | 23.37% | 5.00 | 5.00 | 0.3707 | 0.3715 |
| St_05_Example3_normal | 1134.49 ± 148.67 | 1306.07 ± 62.11 | -13.11% | 2.00 | 2.00 | 0.3758 | 0.3837 |
| chess | 46.80 ± 5.17 | 46.84 ± 7.25 | -2.06% | 3.00 | 3.00 | 0.2764 | 0.2800 |
| engine | 235.47 ± 32.91 | 219.73 ± 40.29 | 4.86% | 2.20 | 2.20 | 0.2654 | 0.2659 |
| liu | 756.92 ± 43.37 | 398.93 ± 73.59 | 49.41% | 8.80 | 8.80 | 0.2264 | 0.2254 |
| shapesnew | 12.37 ± 1.26 | 6.01 ± 1.01 | 53.01% | 9.40 | 9.40 | 0.2488 | 0.2467 |
| st04_Example2_normal | 470.18 ± 72.32 | 453.32 ± 92.84 | 1.46% | 1.80 | 1.80 | 0.2066 | 0.2066 |
| st04_Example3_normal | 745.17 ± 80.83 | 766.88 ± 85.45 | -4.88% | 2.00 | 2.00 | 0.2211 | 0.2257 |
| st04_Example4_normal | 1077.44 ± 120.91 | 897.69 ± 143.47 | 13.88% | 2.60 | 3.00 | 0.2184 | 0.2082 |
| st04_Example5_normal | 1382.55 ± 149.90 | 1219.54 ± 96.21 | 10.90% | 3.00 | 3.00 | 0.2334 | 0.2339 |
| st05_e2 | 469.52 ± 49.71 | 455.65 ± 51.06 | 2.81% | 1.40 | 1.20 | 0.3247 | 0.3321 |

75 passed pairs; 15 datasets × 5 paired sequence/LS seeds; 100 evaluations per algorithm.
SD measures variation across seeds, including different sequences and LS choices; it is not solely timing noise.
Positive paired reduction means improved is faster. It is the median of five paired percentages, not a ratio of mean times.
Only aggregate CSV/MD files downloaded; raw JSON/logs and manifest unavailable for source/input/GPU audit.