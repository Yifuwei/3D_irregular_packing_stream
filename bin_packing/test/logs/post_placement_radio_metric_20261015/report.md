# Shared post-placement radiation bin priority

All three public main-repository controllers (improved_ILS, improved_GRASP, improved_GRASP_ILS) now use the shared post-placement metric through radio_bin_priority.py. Both existing dedicated weight/radio engines also use this same function.

Score: ((current_volume + piece_volume) / bin_volume) * ((max_radio - current_radio - piece_radio) / max_radio), descending; bin index breaks ties. Capacity checks precede ranking. Final N-U* and geometry PES are unchanged.

The shared adapter preserves each controller's acceptance, stagnation, restart, kick and evaluation-budget rules while routing CA, LS and kicks through ranked packing with a run-scoped exact-state cache. All pieces are eligible under ranking; numeric first-fit skip assumptions are not reused. Exact-state CA success reuse retains its first sampled position, which can reduce restart diversification. radio_priority=False is the explicit previous first-fit path. Separate run namespaces prevent global monkey-patching.

Verification: 70 CPU regression tests passed, including actual controller executions for all three methods, kick paths, both ranking engines, an example where the new formula reverses the old bin order, empty-bin ranking, constraints, cache invalidation and existing RCL/geometry-bound checks. Test log: unittest.log. No additional full GPU benchmark was performed.

The main-repository sources and weight/radio framework documentation were updated. Existing experiment snapshots and the already prepared/submitted HPC bundle remain unchanged; every file in that bundle still matches its manifest SHA256.
