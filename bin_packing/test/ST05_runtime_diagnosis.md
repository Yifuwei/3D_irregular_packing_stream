# Local St05 diagnosis

Run in PowerShell (CUDA environment required):

```powershell
& 'C:\work\anaconda\envs\cutting_stable\python.exe' 'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing\test\diagnose_st05_runtime.py' --project 'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing' --evaluations 100 --seed 13 --output 'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing\test\logs\st05_diagnosis_100'
```

Use a fresh output directory. Runs execute sequentially in independent processes:

1. Improved ILS, recording every evaluated piece and orientation.
2. Original ILS with the actual improved selection function, accepted-layout
   inputs, stage weight and last-bin eligibility. Selection changes exist only
   in the generated test adapter, not production functions.
3. Original ILS replaying the improved piece/orientation tape, overriding its
   neighborhood choices to ensure exactly identical candidates.

The same seed and selection policy alone cannot guarantee identical choices:
accepted layouts, stages and orientation-loop stopping can diverge. The third
run asserts candidate equality. Accepted layouts may still differ, so replay
does not completely isolate repacking on identical input states.

Each run has excluded warmup, reset seed, fresh pools and exact budgets.
The diagnostic intentionally disables kicks with threshold budget+1.
Current CA defaults are used and recorded. Do not mix these times with older
HPC code/CA settings, different GPUs or concurrent GPU experiments.

Outputs: `report.md`, `comparison.csv`, and each mode's `run.log`, algorithm
JSON, `candidate_tape.json`, `diagnostics.json` and generated test adapters.
Diagnostics include repack-only geometry search counts/times, rotations,
translations, capacity scans, array comparisons and improved skip/restore events.
CA has a separate timer. Timings are host wall durations; nested timings overlap
and must not be added. Probes introduce overhead, so these are diagnostic
measurements rather than clean final performance benchmarks.

A ten-evaluation pilot checks the workflow on St05. Use 100 evaluations for
the full comparison; a pilot cannot establish the historical 100-evaluation cause.
