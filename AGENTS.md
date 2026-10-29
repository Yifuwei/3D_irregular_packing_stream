# Project instructions

- Best parameters must be shared aggregate choices, never independent per-dataset winners. CA/PES and RCL use all 15 datasets. Explicit exception: ILS kick-strength tuning aggregates only shapesnew, Merged4_normal, st04_Example5_normal; select one shared best strength from those three.
- Cube and cylinder may use different globally shared parameter sets. Current user-confirmed PES/range: cube=maximal_residual_box/bottom; cylinder=minimum_aabb_edges_len/bottom_top.
- RCL covers all 15 datasets; kick strength covers the confirmed three representatives together. GRASP×ILS inherits both shared choices. Smoke/resource probes are diagnostics, not final parameter-selection evidence.
- Keep explicit selection population, metric, scope and chosen settings in frozen plans/results. Missing entire datasets must block final parameter selection.
- Reuse sensitivity only when its actual reported PES/range, inputs and fixed capacities match the new experiment. Never relabel incompatible historical results as current results.
- Preserve the original five-seed capacity calibration unless the user explicitly requests recalibration; actual experimental sequence seeds are 3,7,13.
- Never silently modify the HPC copy or cancel remote jobs. Local package changes require stopping the package's old jobs before synchronization and resubmission.
