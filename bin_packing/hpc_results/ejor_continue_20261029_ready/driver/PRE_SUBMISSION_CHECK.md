# 2026-10-29 work-plan implementation audit

Protocol: ejor-three-seed-repacking-v3. The latest user confirmations take precedence over the PDF where they differ.

| Requirement | Implementation | Verification |
|---|---|---|
| Sequence seeds | References retain 3, 7, 13, 19, 53; experiment stages use only 3, 7, 13 | Protocol scale, forbidden seed and formal freeze tests |
| CA parameters | Existing four PES x three selection ranges; selected PES also evaluates orientations | Existing CA summary import and geometry tests |
| Capacities | Recovered CA U*/n* retained from the original five-seed calibration per dataset/shape; weight/radio formulas retained | Historical placement recovery and calibration tests |
| Bin priority/cache | Hard constraints, then post-placement R/Rmax + U/Umax; first fit without constraints; run-scoped success/failure cache | Ranked engine and real-controller tests |
| ILS selection | Dynamic size/order score, highest unvisited piece; resets on improvement/kick | Preference and controller tests |
| ILS kick | ceil(n/2) consecutive rejected LS repacking calls; total search continues; tune 25/50/75% strength | Counter reset, kick size and budget tests |
| GRASP RCL | Pooled top/bottom extremal positions across orientations; 10%, 20%, Top10, Top30; one construction per RCL test | Pooled RCL and worker count tests |
| GRASP LS | Fixed input order, first improvement; complete failed scan before reconstruction | Complete neighborhood and honest stop-reason tests |
| GRASP x ILS | Inherit tuned RCL/kick strength; n LS repacking calls per construction; ceil(n/2) failure trigger | Repacking budget, odd-n, reset and boundary tests |
| Budget counting | One candidate-orientation repacking is one LS iteration, including cache hits. Kicks are separate from hybrid LS budget but consume wall time | Multi-orientation mid-piece boundary tests |
| Stages | calibration -> sensitivity -> RCL -> ILS tuning -> full, with completion gates | Freeze/submit/collect and smoke-verification tests |
| Submission default | Only through sensitivity; --through full explicitly enables remaining stages | CLI and submission-graph tests |

Formal counts: calibration 150, sensitivity 810, RCL 3240, tuning 486, full 2430; total 7116. Tuning uses three selected datasets and 1800s/run; full uses all 15 datasets and 3600s/run.

The completed historical CA parameter study remains historical five-seed data; the calibration preview retains the exact original five-seed capacities. Parameter selection is global across matched cells, as permitted by the plan. Old frozen bundles must run their own driver; prepare a fresh v3 bundle for this design.

CPU tests validate planning, handoffs and algorithm control flow; they do not constitute a new GPU/Slurm execution or a full performance experiment.

Continuation update: run_all.sh detects CA_sensitivity_check, validates and imports 150 calibration records and the 810 matching sensitivity records into a new runtime bundle. Only later stages need new GPU work; --through full enables them. Inputs, CA parameters, capacities and task success are checked, with original report hashes/provenance retained.
