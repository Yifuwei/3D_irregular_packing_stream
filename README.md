# 3D_irregular_packing_stream

This is the repo for the data and code for paper “Voxel-Based GPU-Empowered 3D Irregular Bin
Packing Stream with Metaheuristics for Tokamak
Fusion Reactor Decommissioning”

<img src="https://github.com/Yifuwei/3D_irregular_packing_stream/blob/main/novel_benchmark_merged.png" alt="novel_benchmark_merged" style="zoom: 33%;" />

## Features

- Fast and efficient constructive algorithm based on voxel, considering orientation
- Fast no-fit voxel calculation is empowered by parallel computation on GPU (cuda framework).
- Cylinder and Cubic containers. 
- nfv and ifv pool
- Robust and efficient accessibility check
- Efficient metaheuristics, Neighbour is the object orientation.

## Usage

``````
pip install -r lib_requirements.txt
python code_launcher.py
``````

## Structure

./open_dimension or ./bin_packing

├── src/ # source code

- binvox_rw.py - for reading binvox file into 3D binary array.
- function_lib.py - tools lib, contains Feasible Region/NFV/IFV calculation. nfv/ifv pool..
- iter_local_search.py -  main logic of ILS.
- packing_iter_ls - main logic of sequential packing, considering orientation, first fit.
- SCH_iter_ls - main logic of Selection of Candidates Heuristic (SCH) in packing, mainly for select packing position out of Feasible Region.
- summary_results.py - get the full computational results of Algorithm

├── test/ # launch the code

- code_launcher.py

├── data/ 

- voxel data from ESICUP

├── result/

- numerical - for tables
- visualisation - for images

