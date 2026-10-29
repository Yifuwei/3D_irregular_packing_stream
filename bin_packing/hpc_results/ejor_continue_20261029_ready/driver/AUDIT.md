# Unified parameter-selection audit

- Persistent rule: AGENTS.md and .cursor/rules/global-experiment-parameters.mdc.
- CA/PES/range: user-confirmed shared configurations by shape; choose_ca_parameters reads the corresponding baseline row and never selects a per-dataset winner.
- Sensitivity: worker applies the shared shape configuration; importer/collector check actual method.pes and method.selection_range. Only compatible historical results can be reused.
- Capacity calibration: original five-seed references are unchanged; legacy settings document provenance, not new parameter optimisation.
- RCL: all 15 datasets, all matched seed/shape/constraint cells; one global RCL, with selected population recorded.
- ILS strength: all 15 datasets, not the former three representatives. Formal subset tuning is rejected; one global strength.
- GRASP×ILS: inherits global RCL and ILS strength; no dataset-specific map or fallback.
- Full: read_selections checks both selected populations against the full dataset list before using parameters.
- Incomplete stages and mismatched PES/range block selection; stale selection.json/summary.csv are removed rather than reused.

Current experiment counts: calibration 150; sensitivity 810; RCL 3240; tuning 2430; full 2430. Sensitivity rerun keeps 324 valid reports and archives 486 incompatible reports. Algorithms and repacking budgets remain as documented in WORK_PLAN.md.
