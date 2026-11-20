# Voxel-Based 3D Irregular Bin Packing

Code, voxel datasets and experimental results accompanying *Voxel-Based GPU-Empowered 3D Irregular Bin Packing Stream with Metaheuristics for Tokamak Fusion Reactor Decommissioning*.

The framework packs irregular voxelised pieces into cubic and cylindrical bins with accessibility, weight and radiation constraints. CUDA accelerates geometric feasibility calculations. Constructive packing is combined with ILS, GRASP and GRASP×ILS.

## Contributions

- **Voxel-based geometric feasibility.** No-Fit Voxels (NFVs) describe collision regions. An NFV pool caches reusable geometric computations and supports transformation-based reuse. Inner-Fit Voxels (IFVs) enforce container boundaries. For cylinders, the horizontal projection's convex hull is tested against the circular boundary and feasible translations are extended over valid heights.
- **Shared constructive packing.** Packing evaluation strategies (PES) score orientation–position candidates for cubic and cylindrical containers. Accessibility checks restrict feasible placements.
- **Constraint-aware repacking.** Eligible bins are prioritised by combined normalised post-placement radiation and volume loads. Successful and failed geometric searches are cached by piece, orientation, bin state and search settings.
- **Three metaheuristics.** ILS uses dynamic piece ranking and kicks. GRASP combines a pooled orientation–position RCL with systematic local descent. GRASP×ILS performs budgeted ILS phases after randomised constructions.
- **Computational evaluation.** Experiments cover CA parameter testing, capacity sensitivity, RCL selection, kick tuning and final constrained and geometry-only comparisons.

## Current algorithm configuration

Both bin shapes use **CSH / minimum AABB edge length / bottom–top**, represented by minimum_aabb_edges_len and bottom_top in the code.

| Component | ILS | GRASP | GRASP×ILS |
|---|---|---|---|
| Initial construction | Deterministic CA | Randomised CA | Randomised CA |
| RCL | — | 10% PES tolerance | 10% PES tolerance |
| LS piece selection | Dynamic ranking of eligible unvisited pieces | Fixed sequence | Dynamic ranking of eligible unvisited pieces |
| Accepted move | First strict improvement over current solution | First strict improvement, then restart scan | First strict improvement over current solution |
| Kick threshold | ceil(n/2) failed repacking calls | — | ceil(n/2) failed repacking calls |
| Kick strength | Change ceil(0.75n) orientation labels | — | Change ceil(0.75n) orientation labels |
| New construction | During kicks | Complete unsuccessful neighbourhood scan | After n LS repacking calls |
| Overall budget | 3600 seconds | 3600 seconds | 3600 seconds |

Here n is the piece count. One LS evaluation is one alternative-orientation repacking call. Dynamic scores combine AABB size and sequence position, increasingly favouring later pieces as failed evaluations accumulate. An improvement or kick resets the failure counter and visited-piece set. Some ILS/hybrid states exclude pieces in the last bin.

The hybrid's per-construction LS budget continues after improvements. Kick reconstruction is included in elapsed time. GRASP's local optimum concerns the defined orientation-and-repacking neighbourhood. A packing call already in progress may finish after the overall deadline.

The objective is **M − U***. M is the bin count and U* is mean utilisation after excluding the least-filled bin. For one bin, U* = U. The **volume-adjusted cost score (VACS)** is V(M − U*). Each dataset's own bin volume V is applied before averaging across datasets.

## Experimental protocol

| Experiment | Population and settings | Runs |
|---|---|---:|
| Capacity calibration | 15 datasets, 2 shapes, 5 original reference seeds | 150 reused |
| CA sensitivity | 15 datasets, 2 shapes, 3 seeds, 4 weight/radiation combinations | 360 |
| RCL selection | 4 rules, 15 datasets, 2 shapes, 3 seeds, 4 constraint combinations | 1440 |
| Kick-strength tuning | 3 strengths, 3 representative datasets, 2 shapes, 3 seeds, 4 combinations | 216 |
| Final constrained comparison | 3 algorithms, 15 datasets, 2 shapes, 3 seeds, 4 combinations | 1080 |
| Geometry-only comparison | 3 algorithms, 15 datasets, 2 shapes, 3 seeds, first-fit bins | 270 |

Experimental sequence seeds are **3, 7, 13**. Original capacity references use **3, 7, 13, 19, 53**. Capacities are calibrated separately for each dataset and shape and fixed across the constrained optimisation experiments.

RCL selection aggregates all 15 datasets and both shapes. Kick strength aggregates shapesnew, Merged4_normal and st04_Example5_normal, with 1800 seconds per tuning run. The selected shared settings are 10% RCL tolerance and 75% kick strength.

Geometry-only runs use non-binding weight/radiation limits and first-fit bin order. Accessibility checking remains enabled. Internal stochastic seeds are generated per run and saved separately from sequence seeds. Reproducing a time-limited search also depends on runtime conditions.

## Final results

Means below are recomputed from the current public per-run CSV. Each constrained entry contains 180 runs and each geometry-only entry contains 45 runs. Lower objective values are better.

| Setting | Shape | ILS | GRASP | GRASP×ILS |
|---|---|---:|---:|---:|
| Weight/radiation constrained | Cube | 7.56381 | 7.56389 | **7.54658** |
| Weight/radiation constrained | Cylinder | 6.56641 | 6.57208 | **6.56628** |
| Geometry-only, first-fit | Cube | 2.97914 | 2.95582 | **2.90542** |
| Geometry-only, first-fit | Cylinder | 2.57324 | **2.52193** | 2.52195 |

Relative performance varies across constraint combinations and datasets. These are descriptive means. Statistical significance has not been established.

## Paper tables and data

Seven CSV files in [bin_packing/result](bin_packing/result/) preserve per-run values at their available precision.

| Paper table | CSV | Rows |
|---|---|---:|
| F.3 — CA parameter testing | [Table_F3_CA_parameter_testing_raw.csv](bin_packing/result/Table_F3_CA_parameter_testing_raw.csv) | 1110 |
| 6 — CA sensitivity | [Table_6_CA_sensitivity_raw.csv](bin_packing/result/Table_6_CA_sensitivity_raw.csv) | 450 |
| F.4 — Overall RCL selection | [Table_F4_RCL_overall_raw.csv](bin_packing/result/Table_F4_RCL_overall_raw.csv) | 1440 |
| F.5 — RCL selection by shape | [Table_F5_RCL_by_shape_raw.csv](bin_packing/result/Table_F5_RCL_by_shape_raw.csv) | 1440 |
| F.6 — Kick tuning | [Table_F6_ILS_kick_tuning_raw.csv](bin_packing/result/Table_F6_ILS_kick_tuning_raw.csv) | 216 |
| F.7 — VACS | [Table_F7_VACS_raw.csv](bin_packing/result/Table_F7_VACS_raw.csv) | 1350 |
| 8 — CA and metaheuristics | [Table_8_full_comparison_raw.csv](bin_packing/result/Table_8_full_comparison_raw.csv) | 1800 |

F.4 and F.5 use the same underlying runs. Orientation histograms and event counts support later aggregation. F.3 includes 1080 individual CSH runs and 30 historical DBLF dataset-level five-seed aggregates, marked by record_level and aggregate_* fields. Matching three-seed DBLF records are unavailable. DBLF used accessibility disabled, while CSH enabled it.

NFV pool workbooks are also retained in the result directory. Source paths inside CSVs record original experiment locations. Historical scripts, frozen bundles and original records are stored locally under archive directories.

## Repository layout

- [bin_packing/src](bin_packing/src/) — CA, geometry, caches and metaheuristic controllers.
- [bin_packing/test](bin_packing/test/) — launcher, validation tools and tests.
- [bin_packing/data](bin_packing/data/) — voxelised pieces.
- [bin_packing/instances](bin_packing/instances/) — instance definitions.
- [bin_packing/result](bin_packing/result/) — paper-table CSVs and NFV pool workbooks.
- [open_dimension](open_dimension/) — open-dimension packing code and data.

Directories named archive at any depth are excluded from Git. Per-event trace files and Python caches are ignored. Historical HPC submission scripts are available in the local archive.

## Environment and execution

Install dependencies in a Python environment compatible with the NVIDIA driver and CUDA runtime.

    python -m pip install -r lib_requirements.txt
    python -c "from numba import cuda; print(cuda.is_available())"

The geometry implementation uses NumPy, SciPy and Numba CUDA. Plotting and profiling dependencies are listed in lib_requirements.txt. Package versions are currently unpinned.

Inspect the launcher options before running an experiment.

    cd bin_packing/test
    python code_launcher.py --help

Shared defaults are in [ca_parameters.py](bin_packing/src/ca_parameters.py). Main controllers are in [new_ILS_southampton.py](bin_packing/src/new_ILS_southampton.py), with shared packing integration in [radio_bin_priority.py](bin_packing/src/radio_bin_priority.py).

See [LICENSE](LICENSE) for licensing terms.
