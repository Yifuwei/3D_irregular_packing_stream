# 3D_irregular_packing_stream

This is the repo for the data and code for paper “Voxel-Based GPU-Empowered 3D Irregular Bin
Packing Stream with Metaheuristics for Tokamak
Fusion Reactor Decommissioning”

<img src="https://github.com/Yifuwei/3D_irregular_packing_stream/blob/main/novel_benchmark_merged.png" width="500" >

## Features

- Fast and efficient constructive algorithm based on voxel representation, considering orientation, achieving new benchmark performance. 
- Algortihms can fit 3D irregular bin packing and 3D irregular strip packing problem, and support cylinder and cubic containers. 
- Fast No-Fit Voxel (NFV) calculation in real time, empowered by parallel computation on GPU (cuda framework).
- Quick and robust morphological techniques to reduce the packing candidates by only consider touching positions.
- NFV and Inner-Fit Voxel (IFV) pool techniques to reduce NFV calculation.
- Robust and efficient accessibility check procedure.
- Efficient metaheuristics, neighbor is the novel piece orientation.

## Usage

``````
pip install -r lib_requirements.txt
python code_launcher.py
``````

## Structure

./open_dimension is for testing the algorithm in the benchmark datasets.

./bin_packing is for the 3D bin packing problem.

./open_dimension or ./bin_packing

├── src/ # source code

- binvox_rw.py - for reading binvox file into 3D binary array.
- function_lib.py - tools lib, contains Feasible Region/NFV/IFV calculation, nfv/ifv pool and visualisation.
- new_ILS_southampton.py -  main logic of ILS/GRASP/GRASP-ILS.
- packing_iter_ls - main logic of constructive algorithm.
- SCH_iter_ls - main logic of Selection of Candidates Heuristic (SCH) in packing, mainly for select packing position out of Feasible Region.

├── test/  # launch the code

- code_launcher.py     # in the main function, change the dataset/algorithm/random seed to configure the code

├── data/ 

- voxel data from ESICUP

├── result/

- numerical - for tables
- visualisation - for images

├── instances/  # instances info for all experiments

