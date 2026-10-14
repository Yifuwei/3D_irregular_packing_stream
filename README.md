# 3D Irregular Packing Stream

Code and datasets for *Voxel-Based GPU-Empowered 3D Irregular Bin Packing Stream with Metaheuristics for Tokamak Fusion Reactor Decommissioning*.

## Current algorithms

- **CA:** sequential first-fit construction with orientation search, CUDA NFV/IFV calculations and candidate-position heuristics; supports cubic and cylindrical bins.
- **ILS / improved_ILS:** orientation-based local search and kicks, retaining the best solution. Evaluation budgets include kicks and exclude CA. In improved_ILS, last-bin pieces become eligible after an accepted layout change.
- **Original improved_repack:** preserves the prefix before the selected piece, which tries bins in order. Later pieces search changed bins, skip unchanged earlier bins, restore placements in unchanged original bins, and search unchanged later bins. Transfers mark both source and target bins changed.
- **Experimental improved_ILS_weight_radio:** preserves the piece sequence and ranks capacity-feasible bins by descending `F_b × (R_max − R_b − r_p) / R_max`. Weight is proportional to volume; matching weight/filling limits are equivalent.
- **Ranked-repacking cache:** reuses failed CA/repack searches only when piece identity, orientation, bin prefix and geometry-search settings match. Successful repack placements can also be reused. An unchanged flag alone is insufficient. CA recording uses `ca_search_observer=engine.record_ca_search`.

CA defaults: cube = `maximal_residual_box` / `bottom`; cylinder = `minimum_aabb_edges_len` / `bottom_top`.

## Run and test

Use a compatible CUDA environment; local experiments use `cutting_stable`.

```bash
conda activate cutting_stable
cd bin_packing/test
python code_launcher.py
python test_improved_ils_control.py
python test_weight_radio_framework.py

# Real chess validation, including CA-cache reuse and unchanged CA output.
python run_weight_radio_chess.py --project .. --output logs/ca_cache_check --max-radio 2000 --rho 0.3 --iterations 10 --cache-repeat-test --verify-ca-observer

# Radiation limits 2000/3000/4000 × filling limits 0.2/0.3/0.4.
python compare_weight_radio_grid.py --project .. --output logs/radio_grid_new --iterations 100 --seed 13
python audit_weight_radio_grid.py logs/radio_grid_new
```

The grid uses the same production improved_ILS controller for both repackers. Its kick threshold is 101, so these 100-evaluation runs contain no kicks. The standalone radio/weight framework has its own selector and allows all pieces from the start.

Outputs: `comparison.csv`, `report.md`, per-run JSON results and event logs. Completed runs are reused; choose a fresh output directory after code changes. The previous seed-13 grid found equal bin counts/U* and 20.97% lower summed wall time; those results predate CA-cache integration.

## Repository

- `bin_packing/src/`: CA, ILS, ranked repacking, geometry and shared CA parameters.
- `bin_packing/test/`: launcher, tests and experiment scripts.
- `bin_packing/data/`, `instances/`: voxel data and instance definitions.
- `open_dimension/`: strip-packing experiments.

[HPC guide](bin_packing/hpc/README.md) · [Multi-seed comparison](bin_packing/hpc/README_multiseed.md) · [Ranked-repacking flow](bin_packing/test/weight_radio_framework.md)

## Search random seeds

The launcher varies the instance-order seed independently of the search streams. Piece selection is fixed at `piece_seed=101`, and packing-position selection at `position_seed=202`. Each algorithm initializes two separate NumPy Generators once per run and reuses them across selections and GRASP restarts. Position draws therefore cannot change the piece-selection random stream. Different layouts, candidate lists, or probabilities may still produce different choices with the same seeds. Legacy ILS kick randomness remains on the global RNG.

The `testing(...)` interface accepts `piece_seed` and `position_seed` overrides; their defaults are fixed for the current comparison stage. Past comparison reports were produced before this RNG separation and retain their original results.
