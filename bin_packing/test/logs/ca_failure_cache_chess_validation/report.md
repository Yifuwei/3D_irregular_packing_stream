# CA failure cache integration verification

Chess: 32 pieces, seed 13, radiation maximum 2000, maximum filling rate 0.3.

- CA recorded 163 actual failed geometry searches.
- Ten framework ILS evaluations, including 5 kicks, reused 7 CA failures naturally.
- Example: piece 24 (zero-based), bin 2 (one-based), orientation y_270; CA failure and first repack skip share state ID `45d87c1cf2cc0b2a2d62`.
- Construction with and without the observer produced identical bin memberships, orientations, translations and voxel occupancy.
- Independently checked 283 ranked queries, actual visit order and matching failure provenance for every cache skip.
- Repeating the same query made zero new geometry searches on the second repeat.
- Eight framework/cache and eleven original ILS controller tests passed. Orientation/state changes and mismatched search parameters invalidate reuse; successful CA searches and empty-bin handling do not seed failure entries.
- Best solution: 3 bins. All pieces present exactly once, no overlaps, filling and radiation within limits.

CA now accepts optional `search_observer`; improved_ILS forwards optional `ca_search_observer`. The radio/weight runners pass `engine.record_ca_search`, and CA/repack use the same complete geometry-search signature. Cache is scoped to one immutable dataset instance/run. Ordinary callers that omit the observer retain their previous behavior.

This is functional verification, not a new 100-iteration timing comparison. Earlier timing results predate this integration.
