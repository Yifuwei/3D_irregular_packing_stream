# Current submission and parameter audit

- Sensitivity is an independent diagnostic flow, fixed at up to 6 GPU tasks.
- RCL -> ILS tuning -> full is a second flow, fixed at up to 6 GPU tasks; sensitivity success is not a dependency.
- Up to 12 GPU tasks may run across both flows. The old shared-throttle coordinator is removed.
- Source and frozen entry points can submit all remaining tasks once, or submit --flow sensitivity and --flow heuristics separately.
- Different flows can coexist in one calibrated package; duplicate-flow jobs and relevant locks are rejected.
- Suspended/configuring jobs remain active for duplicate protection.
- Windows/Linux text line endings are normalized for integrity checks; substantive changes and binary differences are rejected.
- CA/PES and RCL use aggregate shared choices across 15 datasets. ILS kick strength is the explicit exception: aggregate the confirmed three representatives, select one shared winner, and inherit it in hybrid.
- Shared shape parameters: cube=maximal_residual_box/bottom; cylinder=minimum_aabb_edges_len/bottom_top.
- Original five-seed capacity calibration is preserved; sensitivity reports are reused only when their actual PES/range and capacities match.

Task counts: sensitivity 810 (324 retained, 486 pending), RCL 3240, ILS tuning 486, full 2430. No Slurm jobs are submitted by local tests/dry runs.
