import os
import csv

import time 
import numpy as np
import math
import pandas as pd

from scipy.ndimage import binary_fill_holes
# from packing_iter_ls import packing_3D_voxel_lookback, visualize_voxel_model, visualize_single_object
# from packing_iter_ls import packing_3D_voxel_lookback, visualize_voxel_model, visualize_single_object, repacking_ls_per_iteration
from function_lib import save_voxel_model, get_bounding_box

global TRACE


# decide the random seed for ls
def trace(msg):
    global TRACE
    
    if TRACE:
        print(msg)
        
def voxel_volume(array):
    
    x,y,z = np.where(array == 1)
    
    return len(x)

def get_aabb_length(shapes):
    # calculate the max(x,y,z) of a bounding box
    # print(np.shape(shapes))
    x, y, z = np.where (shapes == 1)
    
    delta_x = max(x) - min(x) + 1
    delta_y = max(y) - min(y) + 1
    delta_z = max(z) - min(z) + 1
    
    return 4*(delta_x+delta_y+delta_z)

def get_aabb_volume(object):
    
    x,y,z = get_bounding_box(object)
    volume = x*y*z
    
    return volume

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


# data structure for the local search algorithm

def update_data(data_pool, layout, topos_layout, radio_list,
                pieces_order, pieces_orientation, pieces_orientation_value_pool, 
                pieces_packing_position, pieces_packing_position_pool,
                U_star, N_U, T, selected_info, iteration, neighbour_type):
    
    # selected_info = (pieces_index, orientation, packing_position)
    
    new_row = pd.DataFrame([{
        "iteration": iteration,
        "bin_real_layout": layout,
        "bin_topos_layout": topos_layout,
        "pieces_order": pieces_order,
        "pieces_packing_position": pieces_packing_position,
        "pieces_packing_position_pool": pieces_packing_position_pool,
        "pieces_orientation": pieces_orientation,
        "pieces_orientation_value_pool": pieces_orientation_value_pool,
        "performance_U_star": U_star,
        "performance_N_U_star": N_U,    
        "time": T,
        "neighbour_type":neighbour_type,
        "selected_pool":selected_info, # We might care more about a list of orientation for all pieces 
        "radio_layout":radio_list
    }])
    
    data_pool = pd.concat([data_pool, new_row], ignore_index=True)
    
    return data_pool

def get_new_bin_position(object_total): 
    """_summary_

    To assign a ideal bin for each piece 
    for the bin-based neighbour.

    Args:
        object_total (3d binary array): _description_

    return: 
        a list to indicate each piece goes to which bin

        
    """
    new_bin_index = 0

    return new_bin_index

def bin_lower_bound(object_total, bin_size, container_shape): 
    
    """_summary_

    To get the lower bound of object 
    for the bin-based neighbour.

    Args:
        object_total (3d binary array): list for pieces
        bin_size(tuple): e.g (60,60,60)
        container_shape(str): "cube" or "cylinder"

    return: 
        the lower bin of number of bin (bin needed at least)

    """
    volume_overall = 0 
    container_volume = 0
    bounding_box_volume_overall = 0

    for each_piece in object_total:
        filled_piece = binary_fill_holes(each_piece)
        tem = voxel_volume(filled_piece)
        tem_bounding_box = get_aabb_volume(each_piece)

        bounding_box_volume_overall += tem_bounding_box
        volume_overall += tem 

    if container_shape == "cube":
        container_volume = bin_size[0]*bin_size[1]*bin_size[2]
    
    elif container_shape == "cylinder":
        container_volume = (bin_size[0]/2)**2 * math.pi * bin_size[2] 
    
    else:
        print("Error! no such bin")

    piece_volume_lower_bound = np.ceil(volume_overall/container_volume) 
    aabb_volume_lower_bound = np.ceil(bounding_box_volume_overall/container_volume) 

    return piece_volume_lower_bound, aabb_volume_lower_bound

def pieces_selection_ls(object_total, alpha, seed=None):
    # Maybe also consider the piece order, prefer to select the earlier piece
    # normalised softmax exponential sampling process, alpha is to decide the preference level for the larger object
    # select one object as aimed neighbour searching
    # use individual random seed generation

    rng = np.random.default_rng(seed) 
    value = np.array([get_aabb_length(obj) for obj in object_total])
    
    n = len(object_total)
    order = np.arange(1,n+1)[::-1]
    order_weight = order * 10 / order.sum()
    # value = np.array([get_aabb_volume(obj) for obj in object_total])
    
    v_min = np.min(value)
    v_max = np.max(value)
    v_norm = (value - v_min) / (v_max - v_min + 1e-8)  

    weights_early_item = np.exp(alpha * v_norm) + order_weight 
    probs_early_item = weights_early_item / np.sum(weights_early_item)

    # weight =  np.exp(alpha * v_norm)
    # probs = weight / sum(weight)
    
    selected_index = rng.choice(len(object_total), p=probs_early_item)
    
    # print(probs_early_item)
    # print(probs)
    # print(value/np.sum(value))
    # exit(-1)

    return selected_index

def orientation_selection_ls(orientations):
    # randomly select one orientation
    
    # orientations = ([0,90,180,270],('x','y','z')) 
    tem_orien_list=[]
    
    for each_degree in orientations[0]:
        if each_degree == 0:
            tem_orien_list.append('x_0')
        else:
            for each_axis in orientations[1]:
                _str = each_axis + "_" + str(each_degree)
                tem_orien_list.append(_str)
                
    probs = [1/len(tem_orien_list) for i in range(len(tem_orien_list))]
    selected_orien_index = np.random.choice(len(tem_orien_list),p=probs)
    selected_orien = tem_orien_list[selected_orien_index]
    
    return selected_orien

def find_element_index(nested_list, target, path=None):
    
    # find the index of a element from an irregular list
    # return a list - [a,b,...] to indicate the index of the element depending on the list structure 
    
    if path is None:
        path = []
    
    for idx, element in enumerate(nested_list):
        current_path = path + [idx]
        if element == target:
            return current_path
        
        elif isinstance(element, list):
            result = find_element_index(element, target, current_path)
            if result is not None:
                return result
    
    return None

def iter_local_search(object_total, _pieces_radio, max_radio, rho, orientations,
                      packing_alg, selection_type, selection_range, accessible_check,
                      SCH_nesting_strategy, orien_evaluation,
                      container_size,container_shape,
                      iteration_limit, time_limit, alpha, neighbour_type, visualisation, _TRACE):
    global TRACE 

    TRACE = _TRACE
    
    df = {"iteration":[], 
             "bin_real_layout":[], 
             "bin_topos_layout":[], 
             "pieces_order":[],
             "pieces_packing_position":[],
             "pieces_packing_position_pool":[],
             "pieces_orientation":[],
             "pieces_orientation_value_pool":[],
             "performance_U_star":[],
             "performance_N_U_star":[],
             "time":[],
             "neighbour_type":[],
             "selected_pool":[],
             "radio_layout":[]
             }
    
    data_pool = pd.DataFrame(df)
    
    # trace("============================================================")
    trace(f"Objects number: {len(object_total)}")
    trace(f"The container type: {container_shape}")
    trace(f"Container size is: {container_size}")
    trace(f"Packing algorithm is {packing_alg}") 
    trace(f"Nesting Strategy is: {SCH_nesting_strategy}")
    trace(f"Evaluation for orientation is {orien_evaluation}")
    trace("Accessibility check is True")
    trace("============================================================")

    # ============================================================================
    # Constructive algortithm
    trace("Solution loading..")
    
    check_1 = time.time()
    # orientations_start = ([0,90],["x"])
    layout, topos_layout, radio_list, pieces_order, pieces_orientation, pieces_orientation_value_pool, pieces_packing_position, pieces_packing_position_pool = packing_3D_voxel_lookback(object_total, orientations, container_size,
                                                                                                                                                            container_shape, rho, _pieces_radio, max_radio, 
                                                                                                                                                            packing_alg, orien_evaluation,
                                                                                                                                                            SCH_nesting_strategy = "minimum volume of AABB", density = 5, axis = 'z', 
                                                                                                                                                            _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                                                                                            _select_range = selection_range, _TRACE = False) # for SCH

    
    check_2 = time.time()   

    # print('pieces orientation pool is: ',pieces_orientation)
    # print('pieces order pool is: ',pieces_order)
    
    # this is for the ILS to select the 
    # pieces_order_no_bin = []
    # for each_piece in range(len(object_total)):
    #     index = find_element_index(pieces_order, each_piece, path=None)
    #     pieces_order_no_bin.append(pieces_orientation[index[0]][index[1]])

    if visualisation:
        visualize_voxel_model(layout, pieces_order, container_size, container_shape)
    
    N, U, U_star = get_final_performance(object_total,container_size, container_shape, topos_layout)

    T = check_2 - check_1
    
    
    # =============================================
    # ## overlap checking
    # no_overlap = True
    # for bin_idx, each_bin in enumerate(topos_layout):
    #     overlap_voxels = np.argwhere(each_bin > 1.5)
    #     if overlap_voxels.size > 0:
    #         no_overlap = False
    #         print(f"⚠️ Bin {bin_idx} finds overlapping voxels")
    #         for voxel in overlap_voxels:
    #             print(f"  - coordinate: {tuple(voxel)}, voxel value: {each_bin[tuple(voxel)]}")
    # =============================================  

    # print(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U_star is {U_star}")         
    trace(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U_star is {U_star}")
    
    no_overlap = all(np.all(each_bin <= 1.5) for each_bin in topos_layout)
    trace(f"Constructive algorithm - Overlap: {'No' if no_overlap else 'Yes'}")
    
    if U_star == None: 
        U_star = U
        
    N_U = N - U_star
    
        
    data_pool = update_data(data_pool, layout, topos_layout, radio_list,
                        pieces_order, pieces_orientation, pieces_orientation_value_pool, 
                        pieces_packing_position, pieces_packing_position_pool,
                        U_star, N_U, T, selected_info = "None", iteration="origin", neighbour_type= "None")
    
    # data_pool, layout, topos_layout, radio_list,
    # pieces_order, pieces_orientation, pieces_orientation_value_pool, 
    # pieces_packing_position, pieces_packing_position_pool,
    # U_star, N_U, T, selected_info, iteration, neighbour_type
    
    # ==========================================================================================================
    trace("Iterative Local Search started!")
    
    best_iteration = "origin"
    best_perform = N_U
    n_iter = 1
    overall_time_cost = 0 
    nonstop = True
    
    aabb_volume_list = [get_aabb_volume(i) for i in object_total]
    aabb_sorted = np.argsort(aabb_volume_list)[::-1]

    # to track the orientaions of constructive algorithm
    initial_selected_info = []
    for each_piece in range(len(object_total)):
        index = find_element_index(pieces_order,each_piece)
        initial_selected_info.append((each_piece,pieces_orientation[index[0]][index[1]]))

    while nonstop:
        trace(f"=====================Iteration {n_iter}=========================")
        # use alpha to control the tendancy to choose the large object
        # ==================================================================
        # to select a piece and its orientation and packing position
        # ideally
        # | alpha      | preference                |
        # | ---------- | ------------------- |
        # | 0          | random              |
        # | 1          | slightly prefer bigger aabb             |
        # | 1-10       | largerly prefer bigger aabb             |
        # | 10+        | biggest only        |

        not_valid_trial = True
        max_sample = 100
        trial = 0
        
        best_data = data_pool[data_pool["iteration"] == best_iteration]
        
        # read existing selected info
        seen_selected_info = set(tuple(x) for x in data_pool["selected_pool"].values)
        selected_info = 0
        
        # this is for avoiding the replicated neighbor, prefer to select the larger object to change the orientation
        while not_valid_trial:
            
            filter_data = data_pool[data_pool["iteration"] == best_iteration]
            pieces_order = filter_data["pieces_order"].iloc[0]

            # decide which piece
            selected_index = pieces_selection_ls(object_total, alpha = alpha, seed = 42)
            
            # decide which orientation
            selected_orientation = orientation_selection_ls(orientations)
            volume_rank = list(aabb_sorted).index(selected_index) 

            # To know the position of the selected piece in the sequence 
            index = find_element_index(pieces_order, selected_index)
                 
            # tem = pieces_packing_position_pool[index[0]][index[1]]
            
            # pieces_packing_position_pool
            #  # to track other packing positions sort them from best to worst [(orien, [sorted_packing_position]),(),(),...]
            # 
            # for each_group in tem:  
            #     if each_group[0] == selected_orientation:
            #         # just select the best packing position
            #         # it can be adjustable!
            #         selected_position = each_group[1][0] 
            
            selected_info = (selected_index, selected_orientation)
            
            if selected_info in seen_selected_info or selected_info in initial_selected_info:
                not_valid_trial = True
                
            else:
                trace(f"The selected piece is {selected_index}")
                trace(f"It has {volume_rank} largest aabb, it's piece {index[1]} in bin {index[0]}, in the best solution") 
                not_valid_trial= False
                
            if trial>= max_sample:
                not_valid_trial= False
                trace("Can find a valid neighbor, end ILS!")
                nonstop = False
                
            trial += 1
               
        trace(f"selected info is: {selected_info} Current best iteration is: {best_iteration}")
        
        
        start = time.time()
        
        layout, topos_layout, radio_list, pieces_order, pieces_orientation, pieces_orientation_value_pool, pieces_packing_position, pieces_packing_position_pool = repacking_ls_per_iteration(selected_info, data_pool, best_iteration, object_total, orientations, container_size,
                                                                                                                                                                                            container_shape, rho, _pieces_radio, max_radio, 
                                                                                                                                                                                            packing_alg, orien_evaluation,
                                                                                                                                                                                            SCH_nesting_strategy = "minimum volume of AABB", density = 5, axis = 'z', 
                                                                                                                                                                                            _type = "bounding_box",_accessible_check = True, _encourage_dbl = True, 
                                                                                                                                                                                            _select_range = "bottom_top", _neighbor_type = neighbour_type,  _TRACE = False)
        
        end = time.time()


        T = end - start
        overall_time_cost += T
        
        trace(f"=== Re-pack finished! Cost {T} s in this iteration, It takes {overall_time_cost} s overall ===")
        # trace(pieces_order)
        no_overlap = all(np.all(each_bin <= 1.5) for each_bin in topos_layout)
        trace(f"Repack - Overlap: {'No' if no_overlap else 'Yes'}")
        
        if overall_time_cost > time_limit:
            trace("Has reached the time limit, stop!")
            nonstop = False
        
        elif n_iter == iteration_limit and overall_time_cost < time_limit:
            
            
            N, U, U_star = get_final_performance(object_total,container_size, container_shape, topos_layout)
            
            if U_star == None: 
                U_star = U
                
            perform_this_iter = N - U_star
              
            data_pool = update_data(data_pool, layout, topos_layout,radio_list,
                        pieces_order, pieces_orientation, pieces_orientation_value_pool, 
                        pieces_packing_position, pieces_packing_position_pool,
                        U_star, perform_this_iter, T, selected_info, iteration=n_iter, neighbour_type=neighbour_type)
            
            trace("Database updated!")
            trace("Has reached the iteration limit, stop!")
            nonstop = False
        
        else:
            
            N, U, U_star = get_final_performance(object_total,container_size, container_shape, topos_layout)
            
            if U_star == None: 
                U_star = U
                
            perform_this_iter = N - U_star
                
            data_pool = update_data(data_pool, layout, topos_layout,radio_list,
                        pieces_order, pieces_orientation, pieces_orientation_value_pool, 
                        pieces_packing_position, pieces_packing_position_pool,
                        U_star, perform_this_iter, T, selected_info, iteration=n_iter, neighbour_type=neighbour_type)
            
            trace("Database updated!")

        if perform_this_iter < best_perform: 
        # smaller values are better 
            old_best = best_perform
            best_iteration = n_iter
            best_perform = perform_this_iter
            trace(f"A better result is found by iteration {n_iter}")
            trace(f"Previous best (N-U*) is {old_best}, now is {best_perform}")
            
        else:
            trace("This iteration NOT makes result better!")
            
        n_iter += 1  
        
        
    # ==================================================================================
    # read the final results
    best_data = data_pool[data_pool["iteration"] == best_iteration]
    original_data = data_pool[data_pool["iteration"] == "origin"]
    
    best_pieces_order = best_data["pieces_order"].iloc[0]
    best_topos_layout = best_data["bin_topos_layout"].iloc[0]
    origin_topos_layout = original_data["bin_topos_layout"].iloc[0]
    
    best_current_layout = best_data["bin_real_layout"].iloc[0]
    origin_current_layout = original_data["bin_real_layout"].iloc[0]
    
    # best_visual_layout = best_data["bin_real_layout"].iloc[0]
    

    best_N, best_U, best_U_star = get_final_performance(object_total,container_size, container_shape, best_topos_layout)
    origin_N, origin_U, origin_U_star = get_final_performance(object_total,container_size, container_shape, origin_topos_layout)
    
    if best_U_star == None: 
        best_U_star = best_U
    if origin_U_star == None: 
        origin_U_star = origin_U
        
    # trace("============================================================")
    # trace(f"ILS finished, cost {overall_time_cost} s, num of iterations is {n_iter}")
    # trace(f"Time limit is {time_limit} s, iteration limit is {iteration_limit}")
    # trace(f"Best iteration is {best_iteration}")
    # trace(f"Constructive algorithm, {origin_N} bins are used, U_star is {origin_U_star}")
    # trace(f"After ILS, {best_N} bins are used, U_star is {best_U_star}")
    # trace(f" U_star Improvement: {(best_U_star-origin_U_star)/origin_U_star * 100}%")
    # trace(f" U Improvement: {(best_U-origin_U)/origin_U * 100}%")
    # trace("============================================================")

    print(f"After ILS, {best_N} bins are used, U_star is {best_U_star}")   
    print(f" U_star Improvement: {(best_U_star-origin_U_star)/origin_U_star * 100}%")
    print(f" U Improvement: {(best_U-origin_U)/origin_U * 100}%")
    
    if visualisation:
        visualize_voxel_model(best_current_layout,best_pieces_order, container_size, container_shape)
    
    return best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star, best_current_layout, origin_current_layout, best_topos_layout, origin_topos_layout, best_pieces_order
    
    # df = pd.DataFrame(data_pool)
# ============================================================================