from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
from contextlib import contextmanager

import os
import sys
import csv
import time
import math
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from code_launcher import (
    read_binvox_file,
    voxel_volume,
    get_cubic_bin_size,
    pre_normalised_object,
    get_final_performance
)

from iter_local_search import iter_local_search
from function_lib import save_voxel_model

@contextmanager
def tqdm_joblib(tqdm_object):
    from joblib.parallel import BatchCompletionCallBack, Parallel
    old_batch_callback = BatchCompletionCallBack.__call__

    def new_batch_callback(self, *args, **kwargs):
        tqdm_object.update(n=self.batch_size)
        return old_batch_callback(self, *args, **kwargs)

    BatchCompletionCallBack.__call__ = new_batch_callback
    try:
        yield tqdm_object
    finally:
        BatchCompletionCallBack.__call__ = old_batch_callback
        tqdm_object.close()

def is_done(task, check_path):
    
    dataset, alg, shape, factor, evaluation, nest, sel_range, sel_type, seed = task
    print(f"Checking dataset {dataset}")
    experiment_id = f"{alg}_{shape}_{factor}_{evaluation}_{nest}_{sel_range}_{seed}"
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # csv_dir = os.path.join(BASE_DIR, "results")
    result_csv = os.path.join(check_path, f"{dataset}_results.csv")
    tmp_csv = os.path.join(check_path, f"{dataset}_{experiment_id}_tmp.csv")
    # return experiment_already_done(experiment_id, tmp_csv)
    return experiment_already_done(experiment_id, result_csv) or experiment_already_done(experiment_id, tmp_csv)

def experiment_already_done(experiment_id, filepath):
    # print("check", filepath)
    
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['Algorithm']}_{row['ContainerShape']}_{row['BinSizeFactor']}_{row['Evaluation']}_{row['NestingStrategy']}_{row['Selection_range']}_{row['Seed']}"
            
            print(f"key is {key}")
            print("experiment_id is", experiment_id)
            
            if key == experiment_id:
                print("PASS!!!")
                return True
            
    return False

def run_single_experiment(each_dataset, each_algorithm, container_shape,
                          each_container_factor, each_evaluation, each_nesting, each_select_range, each_select_type, each_seed):
    
    BASE_DIR = "D:/Carlos_project/3D_packing_voxel_ILS"
    csv_dir = os.path.join(BASE_DIR, "result","numerical")
    vis_dir = os.path.join(BASE_DIR, "result","visualisation")
    list_path = os.path.join(BASE_DIR, f"data/InstanceDescription/{each_dataset}.txt")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)


    experiment_id = f"{each_algorithm}_{container_shape}_{each_container_factor}_{each_evaluation}_{each_nesting}_{each_select_range}_{each_seed}"
    # final_csv_path = os.path.join(csv_dir, f"{each_dataset}_results.csv")
    tmp_csv_path = os.path.join(csv_dir, f"{each_dataset}_{experiment_id}_tmp.csv")

    
    object_info = []
    
    with open(list_path, "r", encoding="utf-8") as file:
        jump = True
        for line in file:
            if jump:
                jump = False
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                filepath, count = parts[0], int(parts[1])
                for _ in range(count):
                    voxel_data = read_binvox_file(BASE_DIR, filepath, normalise=True).astype(int)
                    volume = voxel_volume(voxel_data)
                    object_info.append((filepath, voxel_data, volume, voxel_data.shape))

    object_info.sort(key=lambda x: x[2], reverse=True)
    raw_object_total = [item[1] for item in object_info]
    container_size = get_cubic_bin_size(raw_object_total, each_container_factor, container_shape)
    object_total = [pre_normalised_object(obj, container_size) for obj in raw_object_total]

    # nesting_strategy_ILP = "minimum edges length"
    accessible_check = True
    orientations = ((0,90,180,270),('x','y','z'))
    iteration_limit = 50
    time_limit = 3600
    neighbour_type = "orientation only"
    alpha = 1
    
    # use alpha to control the tendancy to choose the large object which has larger AABB
    #==================================================================
    # to select a piece and its orientation and packing position
    # ideally
    # | alpha      | preference                |
    # | ---------- | ------------------- |
    # | 0          | random              |
    # | 1          | slightly prefer bigger aabb             |
    # | 1-10       | largerly prefer bigger aabb             |
    # | 10+        | biggest only        |
    
    rho = 1
    per_ILW = 0.2 # percentage of ILW
    threhold = 10 * per_ILW
    _pieces_radio = []
    
    for i in range(len(object_total)): 
        judge = np.random.uniform(0,10)
        if judge < threhold:
            rad = np.random.uniform(900,1100)
        elif judge >= threhold:
            rad = np.random.uniform(1,10)
        _pieces_radio.append(rad)
              
    # max_radio_list = [1100,1300,1500,1700,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3500,4000]
    max_radio = 9999999
    np.random.seed(each_seed)
    np.random.shuffle(object_total)
    
    check_1 = time.perf_counter()
    best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star, best_current_layout, origin_current_layout, best_topos_layout, origin_topos_layout, best_pieces_order = iter_local_search(object_total, _pieces_radio, max_radio, rho, orientations, 
                                                                                                                                each_algorithm, each_select_type, each_select_range, accessible_check,
                                                                                                                                each_nesting, each_evaluation,
                                                                                                                                container_size, container_shape,
                                                                                                                                iteration_limit, time_limit, alpha, neighbour_type, _TRACE = False) # switch off/on the print msg

    check_2 = time.perf_counter()
    T = check_2 - check_1

    filename = f"{each_dataset}_{each_algorithm}_{container_shape}_F{each_container_factor}_S{each_seed}_{each_evaluation}_{each_nesting}_{each_select_range}"
    save_path = os.path.join(vis_dir, filename)
    
    # N, U, U_star = get_final_performance(object_total, container_size, container_shape, best_topos_layout)
    result_list = [best_N, best_U, best_U_star,T]
    
    # save_voxel_model(best_current_layout, container_size, container_shape, result_list, best_pieces_order, save_path=save_path, save_type="png")

    no_overlap = all(np.all(each_bin <= 1.5) for each_bin in best_topos_layout)

    print(f"✅ Done: {experiment_id} | Time: {T:.2f}s | Bins: {best_N} | U*: {best_U_star} | Overlap: {'No' if no_overlap else 'Yes'}")

    with open(tmp_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.path.getsize(tmp_csv_path) == 0:
            writer.writerow(["Dataset", "Algorithm", "ContainerShape", "BinSizeFactor", "Evaluation", "NestingStrategy", "Selection_range",
                             "Seed", "Time", "Best_BinsUsed", "Best_U", "Best_U_star", "Ori_BinsUsed", "Ori_U", "Ori_U_star", "NoOverlap", "ContainerVolume", "ImageName"])
            
        volume = (container_size[0])**3 if container_shape == "cube" else (math.pi * (container_size[0]/2)**2) * container_size[0]
        
        writer.writerow([each_dataset, each_algorithm, container_shape, each_container_factor,
                         each_evaluation, each_nesting,each_select_range, each_seed, T, best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,
                         no_overlap, volume, filename + ".png"])
        
        
def merge_dataset_csv(dataset_name, container_shape):
    
    BASE_DIR = "D:/Carlos_project/3D_packing_voxel_ILS"
    csv_dir = os.path.join(BASE_DIR, "result","numerical")
    
    # if container_shape == "cube":
    #     csv_dir = os.path.join(BASE_DIR, "results")
    # elif container_shape == "cylinder": 
    #     csv_dir = os.path.join(BASE_DIR, "results")
        
    tmp_files = [f for f in os.listdir(csv_dir) if f.startswith(dataset_name) and f.endswith("_tmp.csv")]

    if not tmp_files:
        print(f"⚠️ can't find dataset={dataset_name} tem file")
        return

    df_list = [pd.read_csv(os.path.join(csv_dir, f)) for f in tmp_files]
    df_all = pd.concat(df_list, ignore_index=True)

    final_csv_path = os.path.join(csv_dir, f"{dataset_name}_results.csv")
    if os.path.exists(final_csv_path) and os.path.getsize(final_csv_path) > 0:
        df_existing = pd.read_csv(final_csv_path)
        df_all = pd.concat([df_existing, df_all], ignore_index=True)

    df_all.drop_duplicates(
        subset=["Algorithm", "ContainerShape", "BinSizeFactor", "Evaluation", "NestingStrategy", "Selection_range", "Seed"],
        keep="last",
        inplace=True
    )

    df_all.to_csv(final_csv_path, index=False)
    print(f"✅ Merge completed:{final_csv_path} has {len(df_all)} rows")
    
    # for f in tmp_files:
    #     os.remove(os.path.join(csv_dir, f))
    #     print(f"🗑️ Deleted temporary file: {f}")

def parallel_experiment():
    
    csv_dir = "D:\\Carlos_project\\3D_packing_voxel_ILS\\result\\numerical"
    
    seed_list = [3, 13, 7, 19, 53]
    
    # list_datasets_name = ["st04_Example5_normal","st05_e2"]
    
    # list_datasets_name = ["chess","engine","liu","Merged1_normal","Merged2_normal","Merged3_normal",
    #                       "Merged4_normal","Merged5_normal"]
    
    list_datasets_name = ["chess","engine","liu","Merged1_normal","Merged2_normal","Merged3_normal",
                          "Merged4_normal","Merged5_normal", "St_05_Example3_normal",
                          "st04_Example2_normal","st04_Example3_normal","st04_Example4_normal","st04_Example5_normal",
                          "st05_e2"]
    
    alg = "SCH"
    
    # packing_alg_list = ["SCH"]
    # packing_alg_list = ["quick_dblf", "SCH", "ILP"]
    
    container_shape_list = ["cube", "cylinder"]
    # =================================================================
    # decide parameter for packing algorithm
    # decide the general type constructive alorithm 
    # packing_alg  = "SCH"
    # selection_type = "bounding_box"
    # selection_range = "bottom"
    # SCH_nesting_strategy = "minimum volume of AABB"
    # _evaluation = "waste_overlap_distance"   
    
    # fyi
    # alg - nesting - orientation eva - select range - select type
    # for cubic container
    # SCH - maximum connected space - minimum_bb_volume - bottom - "bounding_box"
    # for cylindrical container
    # SCH - minimal volume of AABB - waste_overlap_distance - bottom_top - "bounding_box"
    # =================================================
    
    # _evaluation_list = ["maximum connected space", "waste_overlap_distance"]
    #_evaluation_list = ["maximum connected space", "waste_overlap_distance", "minimum_bb_volume"]
    # nesting_strategy_list = ["minimum volume of AABB","minimum length of edges of AABB"]
    #nesting_strategy_list = ["minimum volume of AABB","minimum length of edges of AABB", "maximum connected space"]
    #This is for SCH algorithm "all" is the original version
    # "bottom" and "bottom_top"    
    # select_range_list = ["bottom",'bottom_top']
    
    factor = 1.1
    # bin_size_factor_list = [1.1]

    task_list = []
    
    time_limit = 3600
    iteration_limit = 100
    
    for dataset in list_datasets_name:
        for shape in container_shape_list:
            
            if shape == "cube":
                # it is the best configuration after exp
                select_type = "bounding_box"
                select_range = "bottom"
                nest = "maximum connected space"
                evaluation = "minimum_bb_volume"
                
            elif shape == "cylinder":
                # it is the best configuration after exp
                select_type = "bounding_box"
                select_range = "bottom_top"
                nest = "minimum volume of AABB"
                evaluation = "waste_overlap_distance"     
                       
            for seed in seed_list:
                task_list.append((dataset, alg, shape, factor, evaluation, nest, select_range, select_type, seed))

    # filter the task
    filtered_task_list = [task for task in task_list if not is_done(task, csv_dir)]

    print(f"Total tasks: {len(task_list)}, already done: {len(task_list)-len(filtered_task_list)}, tasks to run: {len(filtered_task_list)}")

    # n_jobs = int(os.environ.get("NSLOTS", 4))

    with tqdm_joblib(tqdm(desc="Progress", total=len(filtered_task_list))) as progress_bar:
        Parallel(n_jobs=1)(
            delayed(run_single_experiment)(*task) for task in filtered_task_list
        )

    for dataset_name in list_datasets_name:
        for container_shape in container_shape_list: 
            merge_dataset_csv(dataset_name,container_shape)
        
if __name__ == "__main__":
    parallel_experiment()

