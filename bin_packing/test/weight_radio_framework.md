# improved_ILS_weight_radio framework

```mermaid
flowchart TD
    A[CA initial solution with weight and radiation limits] --> B[Select piece or kick orientations]
    B --> C[Restore sequence prefix before earliest changed piece]
    C --> D[Process next piece in fixed sequence]
    D --> E[Reject bins exceeding weight, volume or radiation limits]
    E --> F[Score eligible bins: filling rate times radiation remainder ratio]
    F --> G[Sort descending; break ties by bin index]
    G --> H[Visit next ranked bin]
    H --> I{Same orientation and exact prefix state?}
    I -->|Known failure| J[Skip geometric search]
    I -->|Known success or unchanged original placement| K[Reuse position]
    I -->|No matching result| L[Search geometry and cache outcome]
    L -->|Failure| J
    L -->|Success| K
    J --> M{More eligible existing bins?}
    M -->|Yes| H
    M -->|No| N[Try new empty bin after capacity checks]
    N -->|Success| K
    N -->|Failure| X[Report infeasible placement]
    K --> O[Update occupancy, quantities and change flags]
    O --> P{More pieces?}
    P -->|Yes| D
    P -->|No| Q[Validate candidate; accept improvement or kick; retain global best]
    Q --> R{Evaluation budget remains?}
    R -->|Yes| B
    R -->|No| S[Return global best and log]
```

The framework is independent of existing ILS. Initial CA must obey the same
limits; for constant volume-to-weight factor, weight capacity is converted
into an equivalent CA volume limit. The ranked repacker checks both limits.
Original piece shapes, radio observations and search settings remain immutable
within a cache lifetime. Failure reuse requires an identical piece, orientation,
bin index, processed prefix and search configuration. No bin-index-before-original
shortcut is used. Matching successes are reused only at their ranked turn.

The default selector is a volume/stage-weighted framework selector; it is not
claimed identical to existing improved_ILS. An injectable selector allows a
future matched-policy comparison. All pieces are eligible. Kicks change about
half the orientations, use the ranked repacker, and retain the global best.
Cache size and prefix-key overhead have not yet been optimized or benchmarked.

CPU tests cover ranking, untried unchanged bins, cache failures, invalidation,
weight constraints, kick behavior and the evaluation budget. The real chess
smoke test loads its stored radiation proxy values without changing them.
Its per-bin limits are explicit experimental choices, not physical calibration.
