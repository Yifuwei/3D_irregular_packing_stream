# Experiment plan 2026-10-30

CA sensitivity uses the overall best geometry-only PES per bin shape, shared across all 15 datasets. It does not tune PES under constraints.

| Shape | PES | Selection range |
| --- | --- | --- |
| Cube | maximal_residual_box | bottom |
| Cylinder | minimum_aabb_edges_len | bottom_top |

Use sequence seeds 3, 7, 13 and strong/weak weight and radiation (four combinations). Preserve the original five-seed capacity calibration.

| Stage | Cases |
| --- | ---: |
| Calibration references, reused | 150 |
| Fixed-PES CA sensitivity | 360 |
| RCL, four candidates, all 15 datasets | 1440 |
| ILS kick strengths, three representative datasets | 216 |
| Full comparison, three algorithms, one hour each | 1080 |

RCL selects one aggregate winner across all 15 datasets. Kick strength selects one aggregate winner across shapesnew, Merged4_normal, st04_Example5_normal. GRASP x ILS inherits those shared settings. Existing repacking-based LS budgets remain unchanged.

Sensitivity and the RCL -> tuning -> full chain are independent. Each flow has a maximum concurrency of six; combined they can run twelve tasks. Submit both with one command: `bash bin_packing/hpc/paper_suite/run_all.sh`. Add `--dry-run` to inspect submission commands. The launcher creates a fresh v4 bundle and reuses only matching historical sensitivity reports and the original capacity calibration. Existing frozen v3 packages are not modified.

Local preparation verified 150 reused calibration reports and 144 compatible sensitivity reports; 216 sensitivity cases remain. Original calibration selection is exactly unchanged.
