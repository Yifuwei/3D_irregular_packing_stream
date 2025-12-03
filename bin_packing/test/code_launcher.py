import os
import re
import sys
import csv
import time 
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import line_profiler

from scipy.ndimage import binary_fill_holes

# ========================================================\
# call function in src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import binvox_rw
from iter_local_search import iter_local_search, bin_lower_bound, pieces_selection_ls
from new_ILS_southampton import ILS_from_the_first_piece, GRASP, GRASP_ILS
from function_lib import save_voxel_model, NFV_POOL, IFV_POOL, get_bounding_box

def voxel_volume(voxel_data):
    voxel_data = binary_fill_holes(voxel_data)

    return np.sum(voxel_data)

def read_binvox_file(BASE_DIR, object_path, normalise = True):
    #BASE_DIR: D:\PhD_projects\chapter2\3D_packing_voxel
    #object_path: ../Chess/classic_bishop_extracoarse_nh.binvox
    
    #data_dir: D:\PhD_projects\chapter2\3D_packing_voxel\data
    # what I want: D:\PhD_projects\chapter2\3D_packing_voxel\data\Chess\classic_bishop_extracoarse_nh.binvox
    
    filepath = object_path.strip()  # Remove any leading/trailing whitespace or tab characters
    
    if filepath.startswith("../") or filepath.startswith("..\\"):
        filepath = filepath[3:]
        
    data_dir = os.path.join(BASE_DIR, "data")

    filepath = os.path.normpath(os.path.join(data_dir, filepath))
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f" File not found: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            # Read the binvox file as a 3D voxel array (model.data is a boolean ndarray)
            model = binvox_rw.read_as_3d_array(f)
            data = model.data.astype(int)  # Convert boolean array to int (0 or 1)
            # print(f" Successfully loaded: {filepath} | Voxel dimensions: {data.shape}")
            
            if normalise:
                
                x, y, z = np.where(data == 1)
                
                if len(x) == 0:
                    print("Empty voxel model, skipping normalization.")
                    return data

                # bounding box
                min_x, min_y, min_z = np.min(x), np.min(y), np.min(z)
                max_x, max_y, max_z = np.max(x), np.max(y), np.max(z)

                new_shape = (max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)
                new_data = np.zeros(new_shape, dtype=int)

                # translate
                for i in range(len(x)):
                    new_x, new_y, new_z = x[i] - min_x, y[i] - min_y, z[i] - min_z
                    new_data[new_x, new_y, new_z] = 1

                # print(f"Normalized to origin | New Voxel dimensions: {new_data.shape}")
                return new_data

            else:
                return data
            


    except Exception as e:
        # Raise an error if file reading or parsing fails
        raise RuntimeError(f"Error while reading binvox file: {filepath}\nDetails: {e}")

def pre_normalised_object(data, dimension):
    
      
    pad_x_before, pad_x_after = 0, max(dimension[0] - data.shape[0], 0)
    pad_y_before, pad_y_after = 0, max(dimension[1] - data.shape[1], 0)
    pad_z_before, pad_z_after = 0, max(dimension[2] - data.shape[2], 0)
    
    normal_data = np.pad(
                    data,
                    pad_width= ((pad_x_before, pad_x_after), 
                                (pad_y_before, pad_y_after), 
                                (pad_z_before, pad_z_after)),
                    mode='constant',
                    constant_values=0)
    
    # tailor the size to the shape of container

    tailored_data = normal_data[
         : dimension[0],
         : dimension[1],
         : dimension[2]
    ]

    return tailored_data
    
    
def get_final_performance(object_total, container_size, container_shape, topos_layout): 
    
    total_volume = 0
    U_list = []
    
    for each_object in object_total: 
        x, y, z = np.where(each_object == 1)
        total_volume += len(x)

        
    if container_shape == "cube":
        volume_bin = container_size[0] * container_size[1] * container_size[2]
        
    elif container_shape == "cylinder": 
        volume_bin = (math.pi * (container_size[0]/2)**2) * container_size[2]
        
        
    N = len(topos_layout)  
    U = total_volume / (volume_bin * N)

    if N < 2: 
        return N, U, None 
    
    for each_bin in topos_layout:
        x, y, z = np.where(each_bin == 1)
        U_list.append(len(x) / volume_bin)

    min_value = min(U_list)
    U_list.remove(min_value)  

    U_star = sum(U_list) / (N - 1)

    return N, U, U_star

def get_cubic_bin_size_largest_item(object_total, factor):
    max_dimension = 0
    largest_object = None
    largest_object_index = -1

    for idx, each_object in enumerate(object_total): 
        x, y, z = np.where(each_object == 1)
        
        if len(x) == 0:
            continue
        
        bb_length = max(x) - min(x) + 1
        bb_width = max(y) - min(y) + 1
        bb_height = max(z) - min(z) + 1 
        
        current_max_dim = max(bb_length, bb_width, bb_height)

        if current_max_dim > max_dimension:
            max_dimension = current_max_dim
            largest_object = each_object
            largest_object_index = idx

    dim = math.ceil(max_dimension * factor)
    # print(f"Largest dimentsion is {max_dimension}")
    return (dim, dim, dim), largest_object

def get_cubic_bin_size(object_total, factor, container_shape):
    # factor can be 1.1 1.5 2.0
    max_dimension = 0
    
    if container_shape == "cube":

        for each_object in object_total: 
            x,y,z = np.where(each_object == 1)
            
            if len(x) == 0:
                continue
            
            bb_length = max(x)-min(x) + 1
            bb_width = max(y)-min(y) + 1
            bb_height = max(z)-min(z) + 1 
            
            max_dimension = max(max_dimension, bb_length, bb_width, bb_height)
            
        dim = math.ceil(max_dimension * factor)
        
    elif container_shape == "cylinder":
        
        for each_object in object_total: 
            
            x,y,z = np.where(each_object == 1)
            
            if len(x) == 0:
                continue
            
            bb_length = max(x)-min(x) + 1
            bb_width = max(y)-min(y) + 1
            bb_height = max(z)-min(z) + 1 
            
            tem1 = math.sqrt(bb_length**2 + bb_width**2)
            tem2 = math.sqrt(bb_length**2 + bb_height**2)
            tem3 = math.sqrt(bb_height**2 + bb_width**2)
            
            max_dimension = max(max_dimension, tem1, tem2, tem3)
            
        dim = math.ceil(max_dimension * factor)
              
    
    return (dim,dim,dim)


def get_rho_list(object_total, container_size, number):
    # get maximum allowable space (rho) 
    # rho_list = []
    
    # polyons_tem = sorted_by_area(polygon_total)
    # print(polyons_tem0[0])
    # print(smallest_rho)
    smallest_rho = np.sum(object_total[0])/ (container_size[0]*container_size[1]*container_size[2])
    # print(smallest_rho)
    step = (0.7-smallest_rho)/(number-1)
    rho_list = [smallest_rho]
    
    tem = smallest_rho
    
    for each_step in range(number-1):
        tem += step
        rho_list.append(round(tem,6))
        
        
    return rho_list

def load_binvox(filepath):
    with open(filepath, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        array = model.data.astype(np.uint8) 
        return array
    
def instance_generator(amplifier=1): 

    # generate instance for all input/data combinations
    # ==================================================
    # STRUCTURE
    # instance = {  
    #     # ============================================
    #     # pieces-relative data
    #     "name": None,     
    #     "dataset": [], # a list containing 3D np array
    #     "orientations": [],
    #     "radio_list": [],
    #     "weight_list": [], 
    #     # ===========================================
    #     # container data
    #     "container_shape":"cube",
    #     "container_size": None,
    #     "max_radio":[],
    #     "max_weight": []
    # } 
    list_datasets_name = ['Merged4_normal']
    # list_datasets_name = ["shapesnew"]
    
    # list_datasets_name = ["chess", "engine", "liu", "Merged1_normal", "Merged2_normal","Merged3_normal",
    #                 "Merged4_normal","Merged5_normal","St_05_Example3_normal",
    #                 "st04_Example2_normal","st04_Example3_normal","st04_Example4_normal","st04_Example5_normal",
    #                 "st05_e2", "shapesnew"]
    
    # name = "chess"
    
    seq_seed_list = [3,7,13,19,53]
    per_ILW = 0.2 # percentage of ILW
    threhold = 10 * per_ILW
    radioactivity_seed = 19980503
    rad_generator = np.random.default_rng(radioactivity_seed)
    container_shape_list = ["cube","cylinder"]
    # orientations = [("x","y","z"),(0,90,180,270)]
    max_radio_list = [2000,3000,4000]
    max_weight_list = [0.2,0.3,0.4]

    for seq_seed in seq_seed_list:
        for each_name in list_datasets_name:
            for each_container_shape in container_shape_list:
                for each_max_radio in max_radio_list:
                    for each_max_weight in max_weight_list:

                        BASE_DIR = "../"
                        list_path = f"../data/InstanceDescription/{each_name}.txt"
                        instance_id = make_instance_id(each_name, seq_seed, each_container_shape, each_max_radio, each_max_weight)

                        instance_path = f"../instances/{instance_id}.txt"

                        if os.path.exists(instance_path):
                            print(f"Instance {instance_id} is already there")
                            continue  
                        
                        object_info  = []
                        txt_info = []

                        with open(list_path, "r", encoding="utf-8") as file:
                            
                            jump = True
              
                            type_id = 777
                            items_num = 0
                            for line in file:
                                
                                if jump:
                                    # skip the first line
                                    jump = False
                                    pass
                                
                                else:

                                    parts = line.rsplit("\t", 1)

                                    if len(parts) == 2:
                                        
                                        filepath, count = parts[0], int(parts[1])
                                        
                                        for _ in range(int(count*amplifier)):
                                            voxel_data = read_binvox_file(BASE_DIR,filepath,normalise=True).astype(int)
                                            volume = voxel_volume(voxel_data)  

                                            judge = rad_generator.uniform(0,10)

                                            if judge < threhold:
                                                rad = rad_generator.uniform(900,1100)
                                            
                                            elif judge >= threhold:
                                                rad = rad_generator.uniform(1,10)

                                            object_info.append((filepath, voxel_data, volume)) # for bin size
                                            txt_info.append((filepath, volume, rad, type_id))
                                            items_num += 1

                                        type_id += 1

                        # sort the order of object_total here
                        object_info.sort(key=lambda x: x[2], reverse=True)

                        raw_object_total = [item[1] for item in object_info ] # (total, weight, radioactivity) 
                        seq_seed = seq_seed
                        seq_generator = np.random.default_rng(seq_seed) # define the seqerated envioronment for seeds
                        seq_generator.shuffle(txt_info)

                        # Change the container size/type here

                        factor = 1.1
                        container_size = get_cubic_bin_size(raw_object_total,factor,each_container_shape) # decide the container shape
                        # instance_id = make_instance_id(each_name, seq_seed, each_container_shape, each_max_radio, each_max_weight)
        
                        with open(f"../instances/{instance_id}.txt", "w", encoding="utf-8") as f:
                            
                            # Write the container info
                            f.write(f"{each_container_shape}" + "\t" + f"{container_size}" + "\t" + f"{each_max_radio}" + "\t" + f"{each_max_weight}" +"\n") 

                            for line in txt_info:
                                content = str(line[0]) + "\t" + str(line[1]) + "\t" + str(line[2]) + "\t" + str(line[3])
                                f.write(content + "\n")
                        
                        print(f"ID: {instance_id} SAVED!")

    #
    #  return (type_id-777), items_num
                    
def make_instance_id(name, seed, container_shape, max_radio, max_weight):
    return f"{name}_seq{seed}_{container_shape}_{max_radio}_{max_weight}"


def load_instance(instance_id, input_dir="../instances"):
    # read data from the instance.txt file
    # this will create a dict for all info 

    # =======================================================================================
    # key --      value                     --    value example
    # =======================================================================================
    # "array"     -- current 3D binary array        --    np.array((0 0 0),(1,1,1)...) (np array)     
    # "translation"     -- translation (for retrieve nfv) --    (11,13,67) (tuple)
    # "orientation"     -- orientation (for retrieve nfv) --    "x_180" (string)
    # "bin_position"     -- bin position                   --    5 (6 th bin) (int)
    # "volume"     -- volume after filling holes     --    800 (int)
    # "radio"     -- radioactivity                  --    1100 (float)
    # "piece_type"     -- piece type (for retrieve nfv)  --    777 (a number represents a group of item) (int)
    # =======================================================================================

    object_info_total = []

    # object_total = []
    # volume_list = []
    # radio_list = []

    instance_dir = os.path.join(input_dir, f"{instance_id}.txt")

    with open(instance_dir, "r", encoding="utf-8") as file:

        first_line = True
        
        for line in file:
            object_info = {}

            if first_line:
                # first line is for the container info

                parts = line.rsplit("\t")

                first_line = False

                container_shape = parts[0]
                container_size = tuple(map(int, parts[1].strip("()").split(",")))
                max_radio = parts[2] # this is for line one only
                max_weight = parts[3] # this is for line one only

            else:
                # parts = line.split("\t")
                # take multiple \t as one \t
                parts = re.split(r"\t+", line.strip())

                filepath = parts[0]
                voxel_data = read_binvox_file("../",filepath,normalise=True).astype(int)
                object_info["array"] = pre_normalised_object(voxel_data,container_size)
                object_info["translation"] = (0,0,0)
                object_info["orientation"] = "x_0"
                object_info["bin_position"] = None
                object_info["volume"] = int(parts[1])
                object_info["radio"] = round(float(parts[2]),6)
                object_info["piece_type"] = int(parts[3])
                object_info["aabb"] = get_bounding_box(voxel_data)

                object_info_total.append(object_info)

                # object_total.append(pre_normalised_object(voxel_data,container_size))
                # volume_list.append(int(parts[1]))
                # radio_list.append(round(float(parts[2]),6))

        return container_shape, container_size, int(max_radio), int(max_weight), object_info_total
    
def testing(list_datasets_name, seq_seed_list, ALG_list):

    # =======================================================================================
    # key --      value                     --    value example
    # =======================================================================================
    # "array"     -- current 3D binary array        --    np.array((0 0 0),(1,1,1)...) (np array)     
    # "translation"     -- translation (for retrieve nfv) --    (11,13,67) (tuple)
    # "orientation"     -- orientation (for retrieve nfv) --    "x_180" (string)
    # "bin_position"     -- bin position                   --    5 (6 th bin) (int)
    # "volume"     -- volume after filling holes     --    800 (int)
    # "radio"     -- radioactivity                  --    1100 (float)
    # "piece_type"     -- piece type (for retrieve nfv)  --    777 (a number represents a group of item) (int)
    # =======================================================================================

    orientations = ([0,90,180,270],["x","y","z"]) # this is for constructive algorithm
    # orientations = ([0],["x"])

    ils_orientations = ([0,90,180,270],['x','y','z']) # this is for metaheuristic 
    # orientations = ([0,90],["x"])

    orientations_list = []

    for each_degree in ils_orientations[0]:

        if each_degree == 0:
            orientations_list.append("x_0")

        else: 
            for each_axis in ils_orientations[1]:
                tem_str = each_axis + "_" + str(each_degree)
                orientations_list.append(tem_str)

    name_list = list_datasets_name
    # list_datasets_name = ["chess", "engine", "liu", "Merged1_normal", "Merged2_normal","Merged3_normal",
    #                 "Merged4_normal","Merged5_normal","St_05_Example3_normal",
    #                 "st04_Example2_normal","st04_Example3_normal","st04_Example4_normal","st04_Example5_normal",
    #                 "st05_e2"]

    # seq_seed_list = [13] # for sequence
    container_shape_list = ["cylinder"]
    max_radio_list = [999999]
    max_rho_list = [1]
    result_list = []
    # ALG_list = ["BLF", "fixed_CA", "random_CA", "ILS", "GRASP", "GRASP_ILS"]
    # ALG_list = ["fixed_CA"]
    # fixed_CA: only run CA
    # random_CA: only run random CA
    # ILS: activate ILS
    # GRASP: activate GRASP algorithm
    # GRASP_ILS: activate ILS and GRASP
    U_star_overall = 0
    T_overall = 0

    for each_name in name_list: 
        for seq_seed in seq_seed_list:
            for each_ALG in ALG_list:     
                for each_container_shape in container_shape_list:
                    for each_max_rho in max_rho_list:
                        for each_max_radio in max_radio_list:

                            instance_id = make_instance_id(each_name, seq_seed, each_container_shape, each_max_radio, each_max_rho)
                            container_shape, container_size, max_radio, rho, object_info_total = load_instance(instance_id)

                            # for cube
                            if container_shape == "cube":
                                packing_alg  = "SCH"
                                selection_type = "bounding_box"
                                selection_range = "bottom"
                                SCH_nesting_strategy = "maximum connected space"
                                _evaluation = "minimum_bb_volume"   

                            # for cylinder
                            elif container_shape == 'cylinder':
                                packing_alg  = "SCH"
                                selection_type = "bounding_box"
                                selection_range = "bottom_top"
                                SCH_nesting_strategy = "minimum volume of AABB"
                                _evaluation = "waste_overlap_distance"   


                            # This can be done in a more realistic way! 
                            accessible_check = True
                            flag_NFV_POOL = False 
                            # =================================================
                            
                            # ========================================================================
                        
                            
                            # Database for all nfv
                            # this might needs to be a Class to pack all functions and shared variable. 

                            # create the object for the nfv_pool
                            nfv_pool = NFV_POOL() 
                            ifv_pool = IFV_POOL()

                            # switch for GRASP, this is for adding controllable randomness in the constructive alg
                            # random_CA = True

                            # switch for different algs, turn on only one alg or it will run fixed CA as default setting 

                            ALG = each_ALG
                            # input list
                            # fixed_CA: only run CA
                            # random_CA: only run random CA
                            # ILS: activate ILS
                            # GRASP: activate GRASP algorithm
                            # GRASP_ILS: activate ILS and GRASP
                            # random_CA = False # for constructive algorithm, select one randomly out of the 5 best packing position for each piece 
                            # random_CA_threhold = 5   # depends on the orientations


                            if ALG in ("fixed_CA","ILS","random_CA"): 
                                
                                # define the parameter/effort level of ILS algorithm
                                iteration_limit = 1000
                                time_limit = 360000
                                alpha = 1   

                                # to select a piece and its orientation and packing position
                                # ideally
                                # | alpha      | preference                |
                                # | ---------- | ------------------- |
                                # | 0          | random              |
                                # | 1          | slightly prefer bigger aabb             |
                                # | 1-10       | largerly prefer bigger aabb             |
                                # | 10+        | biggest only        |
                                #==================================================================  

                                # kick_trigger_time = len(object_info_total) * 10 / 5 # ? when objects are 300, do we need 600 iters to trigger kick???
                               
                                kick_trigger_time = 120
                                kick_level = "medium"
                                # neighbour_type = "orientation_only"

                            elif ALG == "GRASP": 
                                
                                # define the parameter/effort level of ILS algorithm
                                iter_limit_per_GRASP_iter = 100
                                time_limit = 3600
                                alpha = 1   
                                # to select a piece and its orientation and packing position
                                # ideally
                                # | alpha      | preference                |
                                # | ---------- | ------------------- |
                                # | 0          | random              |
                                # | 1          | slightly prefer bigger aabb             |
                                # | 1-10       | largerly prefer bigger aabb             |
                                # | 10+        | biggest only        |
                                #==================================================================  

                            elif ALG == "GRASP_ILS": 
                                
                                # define the parameter/effort level of ILS algorithm
                                iter_limit_per_GRASP_iter = 100
                                time_limit = 3600
                                alpha = 1   
                                # to select a piece and its orientation and packing position

                                kick_trigger_time = 50
                                kick_level = "medium"

                            elif ALG == "BLF":
                                selection_range = "bottom_left_filling"
                                _evaluation = "height_only"  

                                iteration_limit = None
                                time_limit = None
                                alpha = None   
                                kick_trigger_time = None
                                kick_level = None

                            #==================================================================

                            # use alpha to control the tendancy to choose the large object which has larger AABB
                            # alpha = 1     
                            # kick_trigger_time = len(object_info_total) * 10 / 5
                            # kick_level = "medium"
                        
                            print("======================= Alg started ======================= ")
                            print(f"Dataset is {each_name}")
                            print(f"Algorithm is {each_ALG}")
                            print(f"Rho is {rho}, max_radio is {max_radio}, {0.2 *100} % is ILW.")
                            print(f"Seed for sequence is {seq_seed}")

                            check1 = time.time()

                            # constructive algorithm is involved
                            # best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star, best_current_layout, origin_current_layout, best_topos_layout, origin_topos_layout, best_pieces_order = iter_local_search(object_total, _pieces_radio, max_radio, rho, orientations, 
                            #                                                                                                                                                                                         packing_alg, selection_type, selection_range, accessible_check,
                            #                                                                                                                                                                                         SCH_nesting_strategy, _evaluation,
                            #                                                                                                                                                                                         container_size,container_shape,
                            #                                                                                                                                                                                         iteration_limit, time_limit, alpha, neighbour_type, visualisation = False, _TRACE = True)


                            if ALG in ("BLF", "fixed_CA", "ILS", "random_CA"):
                                best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star, \
                                best_current_layout, origin_current_layout, best_topos_layout,\
                                best_pieces_order, initial_orientations_list_no_bin,\
                                best_orientations_list_no_bin, local_local_best_list, local_change_iter_list, overall_time_cost = ILS_from_the_first_piece(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations,  orientations_list,
                                                                                                                                                            packing_alg, selection_type, selection_range, accessible_check,
                                                                                                                                                            SCH_nesting_strategy, _evaluation, ALG,
                                                                                                                                                            container_size, container_shape,
                                                                                                                                                            iteration_limit= iteration_limit, time_limit=time_limit, alpha=alpha, kick_trigger_time=kick_trigger_time, kick_level=kick_level, flag_NFV_POOL=False, visualisation = False, _TRACE = True)
                                                                                                                                                                                                            
                                                                                                                                                # def ILS_from_the_first_piece(object_info, nfv_pool, max_radio, rho, orientations, ils_orientations, orientations_list,
                                                                                                                                                #             packing_alg, selection_type, selection_range, accessible_check,
                                                                                                                                                #             SCH_nesting_strategy, orien_evaluation, _GRASP, _GRASP_threhold,
                                                                                                                                                #             container_size, container_shape,
                                                                                                                                                #             iteration_limit, time_limit, alpha, neighbour_type, kick_trigger_time, kick_level, visualisation, _TRACE):
                            
                            elif ALG == "GRASP":

                                best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,  \
                                best_current_layout, origin_current_layout, best_topos_layout, best_pieces_order, overall_time_cost = GRASP(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations, orientations_list,
                                                                                                                                                        packing_alg, selection_type, selection_range, accessible_check,
                                                                                                                                                        SCH_nesting_strategy, _evaluation,
                                                                                                                                                        container_size, container_shape,
                                                                                                                                                        iter_limit_per_GRASP_iter, time_limit, alpha, flag_NFV_POOL, visualisation = False, _TRACE = True)
                                                                                                                            
                            elif ALG == "GRASP_ILS":
                                best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,  \
                                best_current_layout, origin_current_layout, best_topos_layout, best_pieces_order, overall_time_cost = GRASP_ILS(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations, orientations_list,
                                                                                                                                                packing_alg, selection_type, selection_range, accessible_check,
                                                                                                                                                SCH_nesting_strategy, _evaluation,
                                                                                                                                                container_size, container_shape,
                                                                                                                                                iter_limit_per_GRASP_iter, kick_trigger_time, kick_level, time_limit, alpha, flag_NFV_POOL, visualisation = False, _TRACE = True)

                            else:
                                raise ValueError
                            
                            check2 = time.time()

                            T = check2 - check1

                            result_list.append([each_name, len(object_info_total), seq_seed, each_ALG, best_N, best_U_star, origin_N, origin_U_star,(best_U-origin_U)/origin_U*100, (best_U_star-origin_U_star)/origin_U_star*100, overall_time_cost])
                            no_overlap = all(np.all(each_bin <= 1.5) for each_bin in best_topos_layout)

                            # print(f"The final orientations are {best_orientations_list_no_bin}")
                            
                            # print(f"Random CA is {random_CA}")
                            # print(f" GRASP IS {_GRASP}")
                            print(f"Alg applied is {ALG}" )
                            print(f"After ILS, {best_N} bins are used, U_star is {best_U_star}")   
                            print(f" U_star Improvement: {(best_U_star-origin_U_star)/origin_U_star * 100}%")
                            print(f" U Improvement: {(best_U-origin_U)/origin_U * 100}%")
                            print(f"==== {ALG} Done - Bins: {best_N} | U*: {best_U_star} | Overlap test: {'Pass' if no_overlap else 'NOT Pass!'} ====")
                            print(f"It takes {overall_time_cost} s, overall")

                            # save_path = "D:/Carlos_project/3D_packing_voxel_ILS/result/visualisation/optimised.png"
                            # save_voxel_model(best_current_layout, container_size, container_shape, result_list, best_pieces_order,  save_path=save_path, save_type="png")

                            U_star_overall += (best_U_star-origin_U_star)/origin_U_star * 100
                            T_overall += (overall_time_cost)
                            
                            # =========================================================================
                            # visualisation for the local_local_best_list, local_change_iter_list 
                            # plt.plot(local_change_iter_list, local_local_best_list, marker='o', label='N-U*')
                            # # label and legend
                            # plt.xlabel('Iterations')
                            # plt.ylabel('N-U*')
                            # plt.title('N-U* along the iterations')
                            # plt.legend()         
                            # plt.grid(True)       
                            # plt.tight_layout()   
                            # plt.savefig(f"{each_name}.png")
                            # plt.show()

                           

                            # df = pd.DataFrame(result_list,columns=columns)
                            # print(f"In average U* improvement is: {U_star_overall/len(seq_seed_list)}, T is {T_overall/len(seq_seed_list)}")
                            # df.to_csv("amplifier_test.csv", index = False)
                            # print(df)

        # # df.to_csv("chess_new_ILS.csv", index = False)

        # # draw the bar chart for the nfv
        # x_labels = [str(k) for k in nfv_pool.keys()]
        # y_values = list(nfv_pool.values())

        # plt.figure(figsize=(6, 4))
        # plt.bar(x_labels, y_values)
        # plt.xlabel("Tuple keys")
        # plt.ylabel("calculation times")
        # plt.title(f"NFV calculation times - overall {sum(y_values)} times")
        # plt.xticks(rotation=30)
        # plt.tight_layout()
        # plt.show()
    
    # df = pd.DataFrame(result_list,columns=columns)
    # df.to_csv("results_liu_st04_Example3_normal.csv", index = False)
    return result_list

def __main__():

    list_datasets_name = ["shapesnew"] 
    ALG_list = ["BLF"]
    seq_seed_list = [13]
    # instance_generator() 
    testing(list_datasets_name,seq_seed_list,ALG_list)

    # columns =  ["dataset","piece_num", "seed", "ALG", "best_N", "best_U_star", "origin_N", "origin_U_star","U improve (%)", "U_star improve (%)", "T (s)"]
    # list_datasets_name = ["chess"] 

    # result_list_overall = []
    # for amplifier in [1]:
    #     instance_generator(list_datasets_name,amplifier) 
    #     result_list = testing(list_datasets_name)

    #     result_list_overall.append(result_list[0])
    #     df = pd.DataFrame(result_list_overall, columns=columns)

    #     # print("Types num: ", types_num, "Items num: ", item_num)
    #     print(df)
    
   # df.to_csv("amplifier_test.csv", index = False)

__main__()
