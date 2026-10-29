# October 29 work-plan check

Source: 20261029 EJOR paper update summary.pdf; latest conversation instructions take precedence.

| Work-plan item | Current script behavior | Result |
|---|---|---|
| CA sensitivity | 15 datasets x 2 shapes x 3 seeds x 9 constraint pairs = 810; tuned fixed CA, no LS | Matches |
| ILS kick-strength tuning | 3 selected datasets x 2 shapes x 3 seeds x 3 strengths x 9 pairs = 486; 1800s each | Matches |
| RCL comparison | 15 x 2 x 3 x 4 RCL choices x 9 pairs = 3240; one random CA, no LS | Matches |
| Full heuristic comparison | 15 x 2 x 3 seeds x 9 pairs x 3 algorithms = 2430; 3600s each | Matches |
| Reference calibration | 15 x 2 x 5 = 150 validated historical CA references; original fixed capacities retained | Latest explicit confirmation: five-seed calibration, three-seed experiments |
| ILS/hybrid piece selection | Dynamic size/order scores over unvisited pieces; no probability sampling in default controllers | Latest user confirmation supersedes PDF |
| Kick trigger and hybrid budget | ceil(n/2) rejected LS repacking calls; hybrid n LS repacking calls per construction | Latest repacking-count request supersedes PDF piece counts and x9 factor |
| GRASP local search | Fixed input order; first improvement restarts scan; complete unsuccessful neighborhood scan before reconstruction | Matches |
| Parameter choice | Global RCL choice and ILS kick strength, inherited by hybrid | Global option allowed in plan |
| Submission | Default stops after sensitivity; --through full enables later stages; completion gates required | Intentional default retained |

Experimental sequence seeds are restricted to 3, 7, 13. Calibration retains all five reference seeds. Smoke uses seed3 for experiments and all five references when importing. Existing five-seed CA parameter-test results and frozen bundles are not rewritten.

Verification: 24 HPC tests passed, including formal freeze with five-seed references and three-seed experiment tasks, forbidden-seed rejection, all-stage counts, historical calibration recovery, submission dependencies, collection, and smoke parameter handoffs. Independently checked every generated task against the real instance files; all 30 capacity groups retain reference_count=5, and all suite Python files for syntax.

No jobs were submitted to Slurm; these are CPU planning/control-flow tests, not a new GPU performance run. Prepare a fresh v3 bundle; old frozen bundles retain their old drivers and seeds.

Continuation update: run_all.sh detects CA_sensitivity_check, validates and imports 150 calibration records and the 810 matching sensitivity records into a new runtime bundle. Only later stages need new GPU work; --through full enables them. Inputs, CA parameters, capacities and task success are checked, with original report hashes/provenance retained.

Real continuation validation: ejor_continue_20261029_ready imports all 150 calibration records and 810 matching sensitivity records. Both stages collect successfully, and the full dry-run graph contains GPU arrays only for RCL, tuning and full. All 24 HPC tests pass, including historical reference provenance and verified Linux/Windows line-ending equivalence. No Slurm jobs submitted.
