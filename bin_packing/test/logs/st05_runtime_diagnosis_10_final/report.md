# St05 runtime diagnosis

Same seed and selection policy do not guarantee identical candidates. Replay enforces identical piece/orientation per evaluation; accepted layouts may differ.
All runs use current CA defaults, fresh processes/pools, excluded warmup, exact evaluation budgets, and no kicks. Diagnostic timings include probe overhead.

| Mode | Wall seconds | Repack seconds | Search calls | Search seconds | Best bins | Best U* | Identical candidates |
|---|---:|---:|---:|---:|---:|---:|---|
| improved | 1390.81 | 666.81 | 300 | 625.18 | 2 | 0.401844 | True |
| shared_policy | 1381.27 | 699.34 | 307 | 629.27 | 2 | 0.401844 | True |
| replay | 1435.93 | 754.03 | 307 | 677.81 | 2 | 0.401844 | True |

Nested operation timings must not be summed. See diagnostics.json for rotation, array comparison, capacity scans and skip/restore counts. This is a current-code experiment, not proof of the historical HPC timing cause.