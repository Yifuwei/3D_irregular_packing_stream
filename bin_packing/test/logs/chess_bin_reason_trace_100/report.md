# chess runtime diagnosis

Same seed and selection policy do not guarantee identical candidates. Replay enforces identical piece/orientation per evaluation; accepted layouts may differ.
All runs use current CA defaults, fresh processes/pools, excluded warmup, exact evaluation budgets, and no kicks. Diagnostic timings include probe overhead.

| Mode | Wall seconds | Repack seconds | Search calls | Search seconds | Best bins | Best U* | Identical candidates |
|---|---:|---:|---:|---:|---:|---:|---|
| improved | 61.40 | 51.02 | 2365 | 46.84 | 3 | 0.268153 | True |
| shared_policy | 77.55 | 67.15 | 3360 | 59.28 | 3 | 0.268153 | True |
| replay | 80.20 | 70.62 | 3360 | 62.20 | 3 | 0.268153 | True |

Nested operation timings must not be summed. See diagnostics.json for rotation, array comparison, capacity scans and skip/restore counts. This is a current-code experiment, not proof of the historical HPC timing cause.