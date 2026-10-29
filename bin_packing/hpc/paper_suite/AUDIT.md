# Unified parameter-selection audit

- Persistent rule: AGENTS.md and .cursor/rules/global-experiment-parameters.mdc.
- CA/PES/range: user-confirmed shared configurations by shape; choose_ca_parameters reads the corresponding baseline row and never selects a per-dataset winner.
- Sensitivity: worker applies the shared shape configuration; importer/collector check actual method.pes and method.selection_range. Only compatible historical results can be reused.
- Capacity calibration: original five-seed references are unchanged; legacy settings document provenance, not new parameter optimisation.
- RCL: all 15 datasets, all matched seed/shape/constraint cells; one global RCL, with selected population recorded.
- ILS strength: the confirmed three representatives (shapesnew, Merged4_normal, st04_Example5_normal), aggregated together to one shared winner. This is the explicit exception to all-15 CA/PES and RCL selection.
- GRASP×ILS: inherits global RCL and ILS strength; no dataset-specific map or fallback.
- Full: read_selections checks RCL population against all 15 datasets and kick population against the three tuning datasets before using parameters.
- Incomplete stages and mismatched PES/range block selection; stale selection.json/summary.csv are removed rather than reused.

Current experiment counts: calibration 150; sensitivity 810; RCL 3240; tuning 486; full 2430. Sensitivity rerun keeps 324 valid reports and archives 486 incompatible reports. Algorithms and repacking budgets remain as documented in WORK_PLAN.md.

One-shot graph update: sensitivity, RCL and tuning depend only on calibration; full depends on RCL and tuning. All branches share a six-GPU cap with collector-driven ArrayTaskThrottle updates; completed GPU branches release slots even if their diagnostic collector reports incomplete results. Independent-graph edges, all completion orders, failure isolation, manual-collection safety, low-concurrency fallback and Slurm throttle commands are covered by tests.
