# St05 controlled runtime diagnosis

St_05_Example3_normal; 25 pieces; cube; sequence and LS seed 13; 100 evaluations per completed group; no kicks.
ILS uses the improved selection policy through a test adapter. All 100 evaluated pieces AND orientations matched improved_ILS. Source hashes and initial objectives match. Accepted intermediate layouts are not asserted identical.
Initial CA bin counts: [16, 9]. CA parameters: maximal_residual_box + bottom.

| Method | Wall time (s) | CA (s) | Repack (s) | Geometry searches | Search time (s) | Best bins | Best U* |
|---|---:|---:|---:|---:|---:|---:|---:|
| improved_ILS | 4669.06 | 636.40 | 3964.71 | 1648 | 3701.00 | 2 | 0.413553 |
| ILS (improved selection policy) | 5114.20 | 684.46 | 4365.78 | 1983 | 3956.93 | 2 | 0.413553 |

improved_ILS reduces total runtime by 8.70%, repack runtime by 9.19%, and geometry search count by 16.89%.
Improved events: skips 181, restores 169, changed-bin visits 1524.

The extra replay group was stopped at user request; its partial observations are excluded.
Conclusion: the controlled local run does not reproduce the HPC slowdown. It supports a repacking benefit under matched candidates, but does not establish the historical cause.
Limitations: one seed, one timing run, instrumented timings, different hardware/current CA settings from unverified historical HPC configuration. Nested operation timings overlap and must not be summed. No production selection policy was changed.