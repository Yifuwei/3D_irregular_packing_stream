# Diagnostic rerun: Merged3_normal cylinder seed 3, RCL 20%

Same immutable sources, inputs, seeds, PES and search budgets. Original batch records were preserved; diagnostic rerun is not silently substituted into the 270-run results.

| Run | Wall seconds | Construction seconds | N | U* | N-U* | CA calls | LS calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 13315.700 | 13315.337 | 3 | 0.354857934 | 2.645142066 | 1 | 0 |
| rerun | 351.454 | 351.257 | 3 | 0.354857934 | 2.645142066 | 1 | 0 |

Both recorded geometric validations passed. The original 13315.7 seconds was almost entirely one non-preemptive construction call, with zero LS calls. A runtime difference alone does not establish its cause. Host suspension, scheduling, and geometry workload have not been isolated. Full insertion trajectories are not certified.
