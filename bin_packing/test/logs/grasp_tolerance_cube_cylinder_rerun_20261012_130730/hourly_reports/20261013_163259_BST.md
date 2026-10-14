# Final GRASP tolerance experiment report

Checked: 2026-10-13 16:32 BST (Europe/London).

54/54 planned runs completed; 54 successful, 0 failed, 0 with recorded validation errors. No active task remains.

## Verification

Exact manifest coverage: 54 unique cases. Per-run configuration, N-U* arithmetic, 18 paired comparisons, raw RCL histograms and aggregate frequencies match. All seven source snapshots match the manifest SHA256 hashes. Current source hash matches: 7/7.

Recorded geometry validation checks piece IDs, bounds, overlap and occupancy (including cylinder mask constraints). It does not certify the full accessibility path.

## Mean N-U* (lower is better)

| Dataset | Shape | 10% | 15% | 20% |
|---|---|---:|---:|---:|
| chess | cube | 2.720964 | 2.722989 | 2.724663 |
| chess | cylinder | 2.744434 | 2.737465 | 2.745149 |
| engine | cube | 2.428878 | 2.424955 | 2.428411 |
| engine | cylinder | 1.762133 | 1.760106 | 1.750001 |
| Merged2_normal | cube | 2.675641 | 2.322609 | 2.675641 |
| Merged2_normal | cylinder | 1.989967 | 1.643847 | 2.009838 |

Each mean uses three order seeds; raw objectives are not pooled across datasets.

## Paired comparison against 10%

| Shape | 15% wins/ties/losses | 20% wins/ties/losses |
|---|---:|---:|
| cube | 5/2/2 | 3/4/2 |
| cylinder | 6/0/3 | 5/2/2 |

## RCL call frequencies

These include all recorded nonempty orientation-level RCL calls, not only placements ultimately used. Means are weighted by call count.

| Shape | Tolerance | Calls | Mean size | Size 1 | Size 2 | Size 3-4 | Size 5+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| cube | 10% | 9934 | 3.165 | 18.1% | 30.7% | 30.4% | 20.8% |
| cube | 15% | 9012 | 3.504 | 12.7% | 27.5% | 33.0% | 26.8% |
| cube | 20% | 9319 | 3.749 | 10.1% | 25.9% | 32.1% | 32.0% |
| cylinder | 10% | 6650 | 2.421 | 33.1% | 30.0% | 27.3% | 9.6% |
| cylinder | 15% | 6489 | 3.091 | 19.8% | 28.4% | 31.5% | 20.2% |
| cylinder | 20% | 6753 | 3.413 | 16.8% | 24.8% | 31.8% | 26.5% |

## Interpretation and limits

15% is the strongest candidate in this pilot, including a reduction from three bins to two for Merged2_normal cube seed 7 and cylinder seed 13 compared with both 10% and 20%. This is not evidence of a universal optimum. Three order seeds and fixed piece/position streams limit generalization.

All 18 engine/Merged2_normal cylinder cases made zero LS calls: their construction exhausted the budget. These cases compare construction outcomes rather than the full construction-plus-LS algorithm. The 180-second budget is checked between non-preemptive calls, so actual runtimes may exceed 180 seconds.

The earlier 39 successful runs and resumed 15 runs are retained in the same output directory. Original data, source snapshots and interrupted logs are preserved. No default tolerance was changed.

Detailed generated report: [report.md](report.md). Raw data: [results.csv](results.csv), [paired_comparison.csv](paired_comparison.csv), [rcl_frequency.csv](rcl_frequency.csv).
