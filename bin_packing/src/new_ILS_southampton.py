import os
import csv

import time 
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import line_profiler

from scipy.ndimage import binary_fill_holes
from packing_iter_ls import packing_3D_voxel_lookback, visualize_voxel_model, visualize_single_object, repacking_new_ILS, kick_repacking, improved_repack
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

def get_final_performance(object_info_total, container_size, container_shape, layout):

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

    total_volume = 0

    for each_object_info in object_info_total: 
        total_volume += each_object_info["volume"]
    
    if container_shape == "cube":
        volume_bin = container_size[0] * container_size[1] * container_size[2]

    elif container_shape == "cylinder": 
        volume_bin = (math.pi * (container_size[0]/2)**2) * container_size[2]
        
    N = len(layout)  
    U = total_volume / (volume_bin * N)

    if N < 2: 
        return N, U, None 
    
    U_list = []
    for each_bin in layout:
        packed_volume = 0
        for each_object_info_bin in each_bin:
            packed_volume += each_object_info_bin["volume"]

        U_list.append(packed_volume / volume_bin)

    min_value = min(U_list)
    U_list.remove(min_value)  

    U_star = sum(U_list) / (N - 1)

    return N, U, U_star


# data structure for the local search algorithm

def update_data(data_pool, layout, topos_layout, radio_layout,
                pieces_order, U_star, N_U, T, selected_info, iteration):
    data_pool[iteration] = {
        "bin_real_layout": layout,
        "bin_topos_layout": topos_layout,
        "pieces_order": pieces_order,
        "performance_U_star": U_star,
        "performance_N_U_star": N_U,
        "time": T,
        "selected_pool": selected_info,
        "radio_layout": radio_layout
    }
    return data_pool

def prune_data_pool(data_pool, *keep_keys):
    keep = set(keep_keys)
    for k in list(data_pool.keys()):
        if k not in keep:
            del data_pool[k]

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

def improve_pieces_selection_ls(object_info_total, iter_time, LS_stage_var, num_used_bin, seed=None, deterministic=False):
    # • Give additional selection weight to the earlier pieces when LS is at its early stage, and vice versa. 
    # The selection_age (the average number of iteration that the piece is selected in the LS) 
    # can gradually increase along the piece order from first to last.  
    # • Early and late LS stages are defined as “kick_trigger_iter – current_no_improve_iter “, smaller the number, later the LS stage. 
    # • Or deterministically, the LS can be simply searched from the first to the last.
    # LS_stage_var - range 0 to 1 - smaller the value, later the LS stage
    # don't need to select the pieces in the last bin, as they will never improve the packing performance.

    # object_info_total = []

    # for each_object_info in _object_info_total:
    #     # get rid of pieces in the last bin
    #     if each_object_info["bin_position"] == num_used_bin - 1:
    #         pass
    #     else:
    #         object_info_total.append(each_object_info)


    n = len(object_info_total)
    
    selectable_mask = np.array([
        obj_info["bin_position"] != num_used_bin - 1
        for obj_info in object_info_total
        ])

    if not np.any(selectable_mask):
        raise ValueError("No selectable pieces outside the last bin")

    if seed != None:
        rng = np.random.default_rng(seed) 
    else:
        rng = np.random

    # if deterministic:
    #     # if the deterministic is True, the LS will select the pieces from the first to the last
    #     selected_index = iter_time % n # for avoiding out of range
    #     return selected_index

    if deterministic:
        # Search selectable pieces from first to last
        selectable_indices = np.flatnonzero(selectable_mask)
        selected_index = selectable_indices[
            iter_time % len(selectable_indices)
        ]
        return selected_index

    weight_values = np.array([get_aabb_length(obj_info["array"]) for obj_info in object_info_total])
    
    
    # order = np.arange(1,n+1)[::-1]
    # order_weight = order * 10/ order.sum()
    # value = np.array([get_aabb_volume(obj) for obj in object_total])
    
    v_min = np.min(weight_values)
    v_max = np.max(weight_values)
    
    if v_max - v_min < 1e-8:
        # Equal size -> size does not distinguish pieces
        size_weight = np.ones(n)

    else:
        size_weight = (weight_values - v_min) / (v_max - v_min)  

    if n == 1:
        # if there is only one object
        order_weight = np.ones(1)

    else:
        early_order_weight = np.linspace(1.0, 0.0, n)
        late_order_weight = np.linspace(0.0, 1.0, n)

        order_weight = (
            LS_stage_var * early_order_weight
            + (1 - LS_stage_var) * late_order_weight
        )

    overall_weight =  size_weight + order_weight 

    overall_weight[~selectable_mask] = 0.0
    total_weight = np.sum(overall_weight)
    if total_weight <= 0:
        overall_weight = selectable_mask.astype(float)
        total_weight = np.sum(overall_weight)
    probs = overall_weight / total_weight

    selected_index = rng.choice(len(object_info_total), p=probs)
    # print(probs)
    # print(value/np.sum(value))
    # exit(-1)
    return selected_index


def pieces_selection_ls(object_info_total, alpha, seed=None):
    # Maybe also consider the piece order, prefer to select the earlier piece
    # normalised softmax exponential sampling process, alpha is to decide the preference level for the larger object
    # select one object as aimed neighbour searching
    # use individual random seed generation

    if seed != None:
        rng = np.random.default_rng(seed) 
    else:
        rng = np.random

    value = np.array([get_aabb_length(obj_info["array"]) for obj_info in object_info_total])
    
    n = len(object_info_total)
    order = np.arange(1,n+1)[::-1]
    order_weight = order * 10/ order.sum()
    # value = np.array([get_aabb_volume(obj) for obj in object_total])
    
    v_min = np.min(value)
    v_max = np.max(value)
    v_norm = (value - v_min) / (v_max - v_min + 1e-8)  

    weights = np.exp(alpha * v_norm) + order_weight 
    probs = weights / np.sum(weights)
   
    selected_index = rng.choice(len(object_info_total), p=probs)
    
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


# Previous improved_ILS implementation, retained for comparison.
# def improved_ILS(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations,
#                 orientations_list, packing_alg, selection_type, selection_range, accessible_check,
#                 SCH_nesting_strategy, orien_evaluation, ALG, container_size, container_shape,
#                 iteration_limit=None, time_limit=None, kick_trigger_time=None,
#                 kick_level=None, flag_NFV_POOL=False,  visualisation=False, _TRACE=False):
# 
#     """
#     Improved LS piece selection - earlier pieces tend to be selected in the early stage of the LS, vice versa.
#     And Re-Pack Strategy - if the ealier bins of the selected bin (i.e. target bin) are not changed: then, the packing start with the selected bin,
#     and, if the same conditions hold, the current piece cannot fit in the selected bin, then all pieces' positions and orientations remain. Otherwise,
#     we need to check from the earliest bin, which is changed when repacking is triggered (the selected piece and all pieces after it are taken out),
#     to the last bin.
#     We track "local_best_orientations_list_no_bin" to update the local optimum packing orientation, and data_pool to update all other info.
# 
#     In repack, filtered_data = old_data_pool[local_best_iteration], this filtered date include topos/current real packing layout, and radio layout.
#     """
# 
#     global TRACE
#     TRACE = _TRACE
# 
#     # structure of object_info_total
#     # =======================================================================================
#     # key --      value                     --    value example
#     # =======================================================================================
#     # "array"     -- current 3D binary array        --    np.array((0 0 0),(1,1,1)...) (np array)
#     # "translation"     -- translation (for retrieve nfv) --    (11,13,67) (tuple)
#     # "orientation"     -- orientation (for retrieve nfv) --    "x_180" (string)
#     # "bin_position"     -- bin position                   --    5 (6 th bin) (int)
#     # "volume"     -- volume after filling holes     --    800 (int)
#     # "radio"     -- radioactivity                  --    1100 (float)
#     # "piece_type"     -- piece type (for retrieve nfv)  --    777 (a number represents a group of item) (int)
#     # =======================================================================================
#     original_object_info_total = copy.deepcopy(object_info_total)
# 
#     data_pool = {}
# 
#     # trace("============================================================")
#     trace(f"Algorithm is {ALG}")
#     trace(f"Objects number: {len(object_info_total)}")
#     trace(f"The container type: {container_shape}")
#     trace(f"Container size is: {container_size}")
#     trace(f"Packing algorithm is {packing_alg}")
#     trace(f"Nesting Strategy is: {SCH_nesting_strategy}")
#     trace(f"Evaluation for orientation is {orien_evaluation}")
#     trace(f"Accessibility check is {accessible_check}")
#     trace("============================================================")
# 
#     # ============================================================================
#     # Constructive algortithm
#     trace("Solution loading..")
#     # original_object_info = copy.deepcopy(object_info_total)
# 
#     if ALG == "fixed_CA" or ALG == "ILS" or ALG == "BLF":
#         random_CA = False
#         random_CA_threshold = None
# 
#     elif ALG == "random_CA":
#         random_CA = True
#         # random_CA_seed = 42 # to decide the seed of the packing positon seletion
#         random_CA_threshold = 5 # this means best five packing position is selected randomly.
# 
#     overall_time_cost = 0
# 
#     check_1 = time.time()
#     # orientations_start = ([0,90],["x"])
# 
#     layout, topos_layout, radio_layout, pieces_order = packing_3D_voxel_lookback(original_object_info_total,
#                                                                                 nfv_pool, ifv_pool, orientations, container_size,
#                                                                                 container_shape, rho, max_radio,
#                                                                                 packing_alg, orien_evaluation,
#                                                                                 SCH_nesting_strategy, density = 5, axis = 'z',
#                                                                                 _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True,
#                                                                                 _select_range = selection_range, random_CA= random_CA, random_CA_threshold = random_CA_threshold,
#                                                                                 flag_NFV_POOL=flag_NFV_POOL, _TRACE = False) # for SCH
# 
# 
# 
#     check_2 = time.time()
# 
#     # print('pieces orientation pool is: ',pieces_orientation)
#     # print('pieces order pool is: ',pieces_order)
# 
#     # this is for the ILS to select the
#     initial_orientations_list_no_bin = []
# 
#     for each_piece in range(len(object_info_total)):
#         index = find_element_index(pieces_order, each_piece, path=None)
#         each_object_info = layout[index[0]][index[1]]
#         initial_orientations_list_no_bin.append(each_object_info["orientation"])
# 
#     if visualisation:
#         visualize_voxel_model(layout, pieces_order, container_size, container_shape)
# 
#     N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
# 
#     T = check_2 - check_1
#     overall_time_cost += T
# 
#     # print(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U_star is {U_star}")
#     trace(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U is {U}, U_star is {U_star}")
# 
#     no_overlap = all(np.all(each_bin <= 1.5) for each_bin in topos_layout)
#     trace(f"Constructive algorithm - Overlap check: {'Pass' if no_overlap else 'NOT pass!'}")
# 
#     if U_star == None:
#         U_star = U
# 
#     N_U = N - U_star
# 
#     # data_pool is to track result of each iteration
#     data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
#                         pieces_order, U_star, N_U, T, selected_info = "None", iteration="origin")
# 
#     trace("Database updated!")
# 
#     # =========================================================================================================
#     # if we activate nfv_ppol, this is to count the number of replicated of NFV
#     # print("all calculation time of nfv is", nfv_pool.all_nfv_cal)
#     # print("validate nfv cal is",nfv_pool.val_nfv_cal)
#     # print("repilcated nfv cal is",nfv_pool.rep_nfv_cal)
#     # print(f"Replicated NFVs: {nfv_pool.rep_nfv_cal/nfv_pool.all_nfv_cal * 100} %")
# 
#     if ALG == "fixed_CA" or ALG == "random_CA" or ALG == "BLF" or N == 1:
#         # if only one bin is used, alg will end right away.
#         return N, U, U_star, N, U, U_star, \
#                 layout, topos_layout, topos_layout,  \
#                 pieces_order, initial_orientations_list_no_bin, \
#                 initial_orientations_list_no_bin,None,None,overall_time_cost
# 
#     # ==========================================================================================================
#     trace("Iterative Local Search started!")
#     local_best_iteration = "origin"
#     global_best_iteration = local_best_iteration
# 
#     local_best_perform = N_U
#     global_best_perform = local_best_perform
# 
#     n_iter = 1
#     n_kick = 1
# 
#     nonstop = True
# 
#     # aabb_volume_list = [get_aabb_volume(i[0]) for i in object_info]
#     # aabb_sorted = np.argsort(aabb_volume_list)[::-1]
# 
#     # to track the orientaions of constructive algorithm
#     # initial_selected_info = []
#     # for each_piece in range(len(object_total)):
#     #     index = find_element_index(pieces_order,each_piece)
#     #     initial_selected_info.append((each_piece,pieces_orientation[index[0]][index[1]]))
# 
#     local_best_orientations_list_no_bin = list(initial_orientations_list_no_bin)
#     global_best_orientations_list_no_bin = list(local_best_orientations_list_no_bin)
# 
#     # local_best_orientations_list_no_bin = initial_orientations_list_no_bin
#     # global_best_orientations_list_no_bin = local_best_orientations_list_no_bin
# 
#     # print(local_best_orientations_list_no_bin)
# 
#     local_best_list = []
#     local_change_iter_list = []
# 
#     no_change_time = 0
#     n_kick = 1
#     iter_piece_selection = 0
# 
#     original_object_info_total = copy.deepcopy(object_info_total)
# 
#     while nonstop:
# 
# 
#         # decide which piece
#         # selected_index = pieces_selection_ls(object_info_total, alpha = alpha, seed=None)
#         LS_stage_var = (kick_trigger_time - no_change_time)/kick_trigger_time # smaller the value, later the LS stage
#         num_used_bin = len(topos_layout)
#         selected_index = improve_pieces_selection_ls(object_info_total, iter_piece_selection, LS_stage_var, num_used_bin, seed=None, deterministic=False)
#         iter_piece_selection += 1
# 
#         # decide which orientation
#         # selected_orientation = orientation_selection_ls(orientations)
# 
#         other_orientations = list(orientations_list)
#         other_orientations.remove(local_best_orientations_list_no_bin[selected_index])
# 
#         for each_orientation in other_orientations:
# 
#             trace(f"===================== iteration {n_iter} =========================")
# 
#             selected_info = (selected_index, each_orientation)
# 
#             trace (f"Object: {selected_index}, orientation: {each_orientation}, original orientation: {local_best_orientations_list_no_bin[selected_index]}")
#             trace (f"Current local best iteration is {local_best_iteration}")
#             trace (f"Current global best iteration is {global_best_iteration}")
# 
#             start = time.time()
#             layout, topos_layout, radio_layout, pieces_order = improved_repack(original_object_info_total,
#                                                                                 selected_info, nfv_pool, ifv_pool,
#                                                                                 local_best_orientations_list_no_bin,
#                                                                                 data_pool, local_best_iteration,
#                                                                                 ils_orientations, container_size,
#                                                                                 container_shape, rho, max_radio,
#                                                                                 packing_alg, orien_evaluation,
#                                                                                 SCH_nesting_strategy, density = 5, axis = 'z',
#                                                                                 _type = "bounding_box",_accessible_check = accessible_check, _encourage_dbl = True,
#                                                                                 _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL,  _TRACE = False)
# 
#             # layout, topos_layout, radio_layout, pieces_order = repacking_new_ILS(original_object_info_total, selected_info, nfv_pool, ifv_pool,
#             #                                                                     local_best_orientations_list_no_bin, data_pool, local_best_iteration, ils_orientations, container_size,
#             #                                                                     container_shape, rho, max_radio,
#             #                                                                     packing_alg, orien_evaluation,
#             #                                                                     SCH_nesting_strategy, density = 5, axis = 'z',
#             #                                                                     _type = "bounding_box",_accessible_check = accessible_check, _encourage_dbl = True,
#             #                                                                     _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL,  _TRACE = False)
# 
#             end = time.time()
# 
#             # kick_flag = False
#             kick_iteration = 0
# 
#             T = end - start
#             overall_time_cost += T
# 
#             trace(f"=== Re-pack finished! Cost {T} s in this iteration, It takes {overall_time_cost} s overall ===")
#             # trace(pieces_order)
# 
#             # no_overlap = all(np.all(each_bin <= 1.5) for each_bin in layout)
#             # trace(f"Repack - Overlap: {'No' if no_overlap else 'Yes'}")
# 
#             if overall_time_cost > time_limit:
#                 trace("Has reached the time limit, stop!")
#                 nonstop = False
#                 break
# 
#             elif n_iter == iteration_limit and overall_time_cost < time_limit:
# 
# 
#                 N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
# 
#                 if U_star == None:
#                     U_star = U
# 
#                 perform_this_iter = N - U_star
# 
# 
#                 if perform_this_iter < global_best_perform:
#                     trace(f"A GLOBAL best is found in iteration - {n_iter}")
#                     no_change_time = 0
#                     old_best = global_best_perform
#                     global_best_iteration = n_iter
#                     global_best_perform = perform_this_iter
# 
#                     local_best_orientations_list_no_bin[selected_index] = each_orientation
#                     global_best_orientations_list_no_bin = local_best_orientations_list_no_bin
#                     trace("GLOBAL best updated")
# 
# 
#                 if perform_this_iter < local_best_perform:
#                     no_change_time = 0
#                     # smaller values are better
#                     old_best = local_best_perform
#                     local_best_iteration = n_iter
#                     local_best_perform = perform_this_iter
#                     local_best_list.append(local_best_perform)
#                     local_change_iter_list.append(n_iter)
#                     trace(f"Previous orientations of pieces are: {local_best_orientations_list_no_bin}")
# 
#                     local_best_orientations_list_no_bin[selected_index] = each_orientation # UPDATE THE BEST OPRIENTATION IF IT'S BETTER
# 
#                     trace(f"A better LOCAL result is found in iteration - {n_iter}")
#                     trace(f"Previous best (N-U*) is {old_best}, now is {local_best_perform}")
#                     trace(f"Now best orientations of pieces are: {local_best_orientations_list_no_bin}")
# 
#                     data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
#                             pieces_order, U_star, perform_this_iter, T, selected_info, iteration=n_iter)
#                     prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
#                     trace("Database updated!")
# 
#                 trace("Has reached the iteration limit, stop!")
#                 nonstop = False
#                 break
# 
#             else:
# 
#                 N, U, U_star = get_final_performance(object_info_total,container_size, container_shape, layout)
# 
#                 if U_star == None:
#                     U_star = U
# 
#                 perform_this_iter = N - U_star
# 
# 
#             if perform_this_iter < global_best_perform:
#                 trace(f"A GLOBAL best is found in iteration - {n_iter}")
#                 no_change_time = 0
#                 # iter_piece_selection = 0
#                 old_best = global_best_perform
#                 global_best_iteration = n_iter
#                 global_best_perform = perform_this_iter
# 
#                 local_best_orientations_list_no_bin[selected_index] = each_orientation
#                 global_best_orientations_list_no_bin = local_best_orientations_list_no_bin
#                 trace("GLOBAL best updated")
# 
#             if perform_this_iter < local_best_perform:
#                 no_change_time = 0
#                 iter_piece_selection = 0
#                 # smaller values are better
#                 old_best = local_best_perform
#                 local_best_iteration = n_iter
#                 local_best_perform = perform_this_iter
#                 local_best_list.append(local_best_perform)
#                 local_change_iter_list.append(n_iter)
#                 trace(f"Previous orientations of pieces are: {local_best_orientations_list_no_bin}")
# 
#                 local_best_orientations_list_no_bin[selected_index] = each_orientation # UPDATE THE BEST OPRIENTATION IF IT'S BETTER
# 
#                 trace(f"A better LOCAL result is found in iteration - {n_iter}")
#                 trace(f"Previous best (N-U*) is {old_best}, now is {local_best_perform}")
#                 trace(f"Now best orientations of pieces are: {local_best_orientations_list_no_bin}")
# 
#                 data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
#                             pieces_order, U_star, perform_this_iter, T, selected_info, iteration=n_iter)
#                 prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
#                 trace("Database updated!")
# 
#                 n_iter += 1
#                 break # break to the next pieces
# 
#             else:
#                 n_iter += 1
#                 no_change_time += 1
#                 trace("This iteration NOT makes result better!")
#                 trace(f"No-improvement Rep - {no_change_time}")
# 
# 
# 
#             if no_change_time >= kick_trigger_time:
#                 trace(f"===================== Kick {n_kick} =========================")
#                 trace(f"!!!!Solution no change for {no_change_time} replications, a kick is triggered!!!!")
#                 # kick_flag = True
#                 no_change_time = 0 # re-count the no change time
#                 trace("No change times return 0!")
# 
#                 kick_iteration = n_iter
# 
#                 # can be parameterised
#                 # if after an amount of time, the solution is unchanged
#                 # we give a kick to the orientations
# 
#                 tem = list(global_best_orientations_list_no_bin)
# 
#                 if kick_level == "small":
#                     change_n = int(np.ceil(len(object_info_total)/4))
#                     change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)
#                     for each_position in change_position:
#                         tem[each_position] = np.random.choice(orientations_list)
# 
#                 elif kick_level == "medium":
#                     change_n = int(np.ceil(len(object_info_total)/2))
#                     change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)
#                     for each_position in change_position:
#                         tem[each_position] = np.random.choice(orientations_list)
# 
#                 elif kick_level == "large":
#                     change_n = int(np.ceil(3*len(object_info_total)/4))
#                     change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)
#                     for each_position in change_position:
#                         tem[each_position] = np.random.choice(orientations_list)
# 
#                 local_best_orientations_list_no_bin = tem
# 
#                 trace("Kick repacking started")
#                 start = time.time()
# 
#                 layout, topos_layout, radio_layout, pieces_order = kick_repacking(original_object_info_total, nfv_pool, ifv_pool, orientations, local_best_orientations_list_no_bin, container_size,
#                                                                                 container_shape, rho, max_radio,
#                                                                                 packing_alg, orien_evaluation,
#                                                                                 SCH_nesting_strategy, density = 5, axis = 'z',
#                                                                                 _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True,
#                                                                                 _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL, _TRACE = False)
#                 start = time.time()
# 
#                 overall_time_cost += (end-start)
# 
#                 N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
# 
#                 if U_star == None:
#                     U_star = U
# 
#                 perform_this_iter = N - U_star
# 
#                 # kick data must be the local best for the new iterations
#                 data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
#                             pieces_order, U_star, perform_this_iter, "kick", selected_info = "None", iteration=n_iter)
# 
#                 local_best_list.append(local_best_perform)
#                 local_change_iter_list.append(kick_iteration-1)
# 
#                 local_best_iteration = kick_iteration
#                 prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
#                 trace("Database updated!")
# 
#                 local_best_list.append(perform_this_iter)
#                 local_change_iter_list.append(local_best_iteration)
# 
#                 trace(f"Current local best (N-U*) is {local_best_perform}, kick leads to {perform_this_iter}")
#                 trace(f"Local best iteration updated to the kick iteration {kick_iteration}!")
# 
#                 if perform_this_iter < global_best_perform:
#                     # highly unlikely
#                     old_best = global_best_perform
#                     global_best_iteration = n_iter
#                     global_best_perform = perform_this_iter
#                     global_best_orientations_list_no_bin = tem
# 
#                 local_best_perform = perform_this_iter
#                 iter_piece_selection = 0
#                 n_iter += 1
#                 n_kick += 1
# 
#         if nonstop == False:
#             break
# 
# 
# 
#     # ==================================================================================
#     # read the final results
#     best_data = data_pool[global_best_iteration]
#     original_data = data_pool["origin"]
# 
#     best_pieces_order = best_data["pieces_order"]
#     best_topos_layout = best_data["bin_topos_layout"]
#     # origin_topos_layout = original_data["bin_topos_layout"]
# 
#     best_current_layout = best_data["bin_real_layout"]
#     origin_current_layout = original_data["bin_real_layout"]
# 
# 
#     best_N, best_U, best_U_star = get_final_performance(object_info_total, container_size, container_shape, best_current_layout)
#     origin_N, origin_U, origin_U_star = get_final_performance(object_info_total, container_size, container_shape, origin_current_layout)
# 
#     if best_U_star == None:
#         best_U_star = best_U
# 
#     if origin_U_star == None:
#         origin_U_star = origin_U
# 
#     # trace("============================================================")
#     # trace(f"ILS finished, cost {overall_time_cost} s, num of iterations is {n_iter}")
#     # trace(f"Time limit is {time_limit} s, iteration limit is {iteration_limit}")
#     # trace(f"Best iteration is {best_iteration}")
#     # trace(f"Constructive algorithm, {origin_N} bins are used, U_star is {origin_U_star}")
#     # trace(f"After ILS, {best_N} bins are used, U_star is {best_U_star}")
#     # trace(f" U_star Improvement: {(best_U_star-origin_U_star)/origin_U_star * 100}%")
#     # trace(f" U Improvement: {(best_U-origin_U)/origin_U * 100}%")
#     # trace("============================================================")
# 
# 
#     if visualisation:
#         visualize_voxel_model(best_current_layout, best_pieces_order, container_size, container_shape)
# 
# 
#     # draw the bar chart for the nfv
#     # x_labels = [str(k) for k in nfv_pool.keys()]
#     # y_values = list(nfv_pool.values())
#     # plt.figure(figsize=(6, 4))
#     # plt.bar(x_labels, y_values)
#     # plt.xlabel("Tuple keys")
#     # plt.ylabel("calculation times")
#     # plt.title(f"NFV calculation times {sum(y_values)}")
#     # plt.xticks(rotation=30)
#     # plt.tight_layout()
#     # plt.show()
# 
#     return best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,  \
#             best_current_layout, origin_current_layout, best_topos_layout,  \
#             best_pieces_order, initial_orientations_list_no_bin, global_best_orientations_list_no_bin,local_best_list,local_change_iter_list,overall_time_cost

def improved_ILS(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations,
                 orientations_list, packing_alg, selection_type, selection_range, accessible_check,
                 SCH_nesting_strategy, orien_evaluation, ALG, container_size, container_shape,
                 iteration_limit=None, time_limit=None, kick_trigger_time=None,
                 kick_level=None, flag_NFV_POOL=False, visualisation=False, _TRACE=False):
    
    """Run randomized descent and kicks with independent accepted/best states.

    iteration_limit counts evaluated neighbors and kick solutions, excluding
    construction. None disables a limit. Wall time includes construction.
    A running packing call is allowed to finish and its result is evaluated
    before stopping; no further candidate is started after the budget expires.
    """

    global TRACE
    TRACE = _TRACE

    if ALG not in ("ILS", "fixed_CA", "random_CA", "BLF"):
        raise ValueError("Unsupported ALG")
    if not object_info_total:
        raise ValueError("At least one piece is required")
    if iteration_limit is not None and (not isinstance(iteration_limit, (int, np.integer)) or iteration_limit < 0):
        raise ValueError("iteration_limit must be a nonnegative integer or None")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative or None")
    if ALG == "ILS":
        if kick_trigger_time is not None and (not isinstance(kick_trigger_time, (int, np.integer)) or kick_trigger_time <= 0):
            raise ValueError("kick_trigger_time must be a positive integer or None")
        if kick_trigger_time is not None and kick_level not in ("small", "medium", "large"):
            raise ValueError("kick_level must be small, medium or large when kicks are enabled")
        if kick_trigger_time is not None and len(set(orientations_list)) < 2:
            raise ValueError("Kicks require at least two distinct orientations")
        if not orientations_list:
            raise ValueError("orientations_list must not be empty")

    overall_start = time.perf_counter()
    original_object_info_total = copy.deepcopy(object_info_total)
    data_pool = {}
    local_best_list, local_change_iter_list = [], []
    common = dict(density=5, axis='z', _accessible_check=accessible_check,
                  _encourage_dbl=True, _select_range=selection_range,
                  flag_NFV_POOL=flag_NFV_POOL, _TRACE=False)

    # Construction may modify its input; preserve unplaced original geometry.
    start = time.perf_counter()
    layout, topos_layout, radio_layout, pieces_order = packing_3D_voxel_lookback(
        copy.deepcopy(original_object_info_total), nfv_pool, ifv_pool,
        orientations, container_size, container_shape, rho, max_radio,
        packing_alg, orien_evaluation, SCH_nesting_strategy,
        _type=selection_type, random_CA=ALG == "random_CA",
        random_CA_threshold=5 if ALG == "random_CA" else None, **common)
    T = time.perf_counter() - start

    def evaluate(layout, topos_layout, pieces_order):
        # Reject invalid candidates before they can update accepted/best states.
        indices = [i for ids in pieces_order for i in ids]
        if sorted(indices) != list(range(len(object_info_total))):
            raise ValueError("Packing must contain every piece exactly once")
        if len(layout) != len(topos_layout) or len(layout) != len(pieces_order):
            raise ValueError("Inconsistent bin records")
        if any(len(objects) != len(ids) for objects, ids in zip(layout, pieces_order)):
            raise ValueError("Inconsistent piece records")
        if any(np.any(array > 1) for array in topos_layout):
            raise ValueError("Packing contains overlapping voxels")
        N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
        U_star = U if U_star is None else U_star
        return N, U, U_star, N - U_star

    def read_orientations(layout, pieces_order):
        values = [None] * len(object_info_total)
        for each_bin, indices in enumerate(pieces_order):
            for j, index in enumerate(indices):
                values[index] = layout[each_bin][j]["orientation"]
        return values

    N, U, U_star, local_best_perform = evaluate(layout, topos_layout, pieces_order)
    initial_orientations_list_no_bin = read_orientations(layout, pieces_order)
    local_best_orientations_list_no_bin = list(initial_orientations_list_no_bin)
    global_best_orientations_list_no_bin = list(initial_orientations_list_no_bin)
    global_best_perform = local_best_perform
    local_best_iteration = global_best_iteration = "origin"
    update_data(data_pool, layout, topos_layout, radio_layout, pieces_order,
                U_star, local_best_perform, T, "None", "origin")
    if visualisation:
        visualize_voxel_model(layout, pieces_order, container_size, container_shape)

    n_iter = 0
    no_change_time = iter_piece_selection = 0

    def budget_exhausted():
        return ((iteration_limit is not None and n_iter >= iteration_limit) or
                (time_limit is not None and time.perf_counter() - overall_start >= time_limit))

    def accept_candidate(layout, topos_layout, radio_layout, pieces_order, T,
                         selected_info, is_kick=False):
        
        # Store layout, actual orientations and objective as one consistent state.
        nonlocal local_best_iteration, global_best_iteration
        nonlocal local_best_perform, global_best_perform
        nonlocal local_best_orientations_list_no_bin, global_best_orientations_list_no_bin
        nonlocal no_change_time, iter_piece_selection

        N, U, U_star, perform_this_iter = evaluate(layout, topos_layout, pieces_order)
        accepted = is_kick or perform_this_iter < local_best_perform
        globally_better = perform_this_iter < global_best_perform
        
        if accepted or globally_better:
            candidate_orientations = read_orientations(layout, pieces_order)
            update_data(data_pool, layout, topos_layout, radio_layout, pieces_order,
                        U_star, perform_this_iter, T, selected_info, n_iter)
            if accepted:
                local_best_iteration = n_iter
                local_best_perform = perform_this_iter
                local_best_orientations_list_no_bin = list(candidate_orientations)
                local_best_list.append(perform_this_iter)
                local_change_iter_list.append(n_iter)
                no_change_time = iter_piece_selection = 0
            if globally_better:
                global_best_iteration = n_iter
                global_best_perform = perform_this_iter
                global_best_orientations_list_no_bin = list(candidate_orientations)
            prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
        else:
            no_change_time += 1
        trace(f"Evaluation {n_iter}: N-U*={perform_this_iter}, accepted={accepted}, kick={is_kick}")
        return accepted, N

    nonstop = ALG == "ILS" and N > 1
    while nonstop and not budget_exhausted():
        # Select from the accepted local solution, never from a rejected trial.
        filtered_data = data_pool[local_best_iteration]
        num_used_bin = len(filtered_data["bin_real_layout"])
        selection_info = [None] * len(object_info_total)
        for each_bin, indices in enumerate(filtered_data["pieces_order"]):
            for j, index in enumerate(indices):
                selection_info[index] = filtered_data["bin_real_layout"][each_bin][j]
        LS_stage_var = ((kick_trigger_time - no_change_time) / kick_trigger_time
                        if kick_trigger_time is not None else 1.0)
        selected_index = improve_pieces_selection_ls(
            selection_info, iter_piece_selection, LS_stage_var,
            num_used_bin, seed=None, deterministic=False)
        iter_piece_selection += 1
        other_orientations = [value for value in orientations_list
                              if value != local_best_orientations_list_no_bin[selected_index]]
        if not other_orientations:
            break

        for each_orientation in other_orientations:
            if budget_exhausted():
                nonstop = False
                break
            selected_info = (selected_index, each_orientation)
            start = time.perf_counter()
            layout, topos_layout, radio_layout, pieces_order = improved_repack(
                original_object_info_total, selected_info, nfv_pool, ifv_pool,
                local_best_orientations_list_no_bin, data_pool, local_best_iteration,
                ils_orientations, container_size, container_shape, rho, max_radio,
                packing_alg, orien_evaluation, SCH_nesting_strategy,
                _type="bounding_box", **common)
            T = time.perf_counter() - start
            n_iter += 1
            accepted, N = accept_candidate(layout, topos_layout, radio_layout,
                                          pieces_order, T, selected_info)
            if N == 1 or budget_exhausted():
                nonstop = False
                break
            if accepted:
                break

            if kick_trigger_time is not None and no_change_time >= kick_trigger_time:
                # Retain the existing global-best-based perturbation policy.
                tem = list(global_best_orientations_list_no_bin)
                fraction = {"small": 0.25, "medium": 0.5, "large": 0.75}[kick_level]
                change_n = int(np.ceil(len(object_info_total) * fraction))
                change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)
                for each_position in change_position:
                    # Each selected piece must change its orientation label.
                    # Remove duplicates to sample distinct alternatives uniformly.
                    other_orientations_kick = [value for value in dict.fromkeys(orientations_list)
                                               if value != tem[each_position]]
                    tem[each_position] = np.random.choice(other_orientations_kick)
                start = time.perf_counter()
                layout, topos_layout, radio_layout, pieces_order = kick_repacking(
                    copy.deepcopy(original_object_info_total), nfv_pool, ifv_pool,
                    orientations, tem, container_size, container_shape, rho, max_radio,
                    packing_alg, orien_evaluation, SCH_nesting_strategy,
                    _type=selection_type, **common)
                T = time.perf_counter() - start
                n_iter += 1
                _, N = accept_candidate(layout, topos_layout, radio_layout,
                                        pieces_order, T, "kick", is_kick=True)
                if N == 1 or budget_exhausted():
                    nonstop = False
                # Rebuild piece selection and orientation candidates after every kick.
                break

    best_data = data_pool[global_best_iteration]
    original_data = data_pool["origin"]
    best_current_layout = best_data["bin_real_layout"]
    origin_current_layout = original_data["bin_real_layout"]
    best_topos_layout = best_data["bin_topos_layout"]
    best_pieces_order = best_data["pieces_order"]
    
    best_N, best_U, best_U_star, _ = evaluate(best_current_layout, best_topos_layout, best_pieces_order)
    origin_N, origin_U, origin_U_star, _ = evaluate(
        origin_current_layout, original_data["bin_topos_layout"], original_data["pieces_order"])
    if visualisation:
        visualize_voxel_model(best_current_layout, best_pieces_order, container_size, container_shape)
    overall_time_cost = time.perf_counter() - overall_start

    return (best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,
            best_current_layout, origin_current_layout, best_topos_layout,
            best_pieces_order, list(initial_orientations_list_no_bin),
            list(global_best_orientations_list_no_bin), local_best_list,
            local_change_iter_list, overall_time_cost)
    
    # df = pd.DataFrame(data_pool)
# ============================================================================



def ILS_from_the_first_piece(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations, orientations_list,
                            packing_alg, selection_type, selection_range, accessible_check,
                            SCH_nesting_strategy, orien_evaluation, ALG,
                            container_size, container_shape,
                            iteration_limit=None, time_limit=None, alpha=None, kick_trigger_time=None, kick_level=None, flag_NFV_POOL=False,  visualisation=False, _TRACE=False):
    
    global TRACE 
    TRACE = _TRACE

    # structure of object_info_total
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
    original_object_info_total = copy.deepcopy(object_info_total)

    data_pool = {}

    # trace("============================================================")
    trace(f"Algorithm is {ALG}")
    trace(f"Objects number: {len(object_info_total)}")
    trace(f"The container type: {container_shape}")
    trace(f"Container size is: {container_size}")
    trace(f"Packing algorithm is {packing_alg}") 
    trace(f"Nesting Strategy is: {SCH_nesting_strategy}")
    trace(f"Evaluation for orientation is {orien_evaluation}")
    trace(f"Accessibility check is {accessible_check}")
    trace("============================================================")

    # ============================================================================
    # Constructive algortithm
    trace("Solution loading..")
    # original_object_info = copy.deepcopy(object_info_total)

    if ALG == "fixed_CA" or ALG == "ILS" or ALG == "BLF":
        random_CA = False
        random_CA_threshold = None

    elif ALG == "random_CA":
        random_CA = True
        random_CA_seed = 42 # to decide the seed of the packing positon seletion
        random_CA_threshold = 5 # this means best five packing position is selected randomly.
    
    overall_time_cost = 0 

    check_1 = time.time()
    # orientations_start = ([0,90],["x"])
    # 

    layout, topos_layout, radio_layout, pieces_order = packing_3D_voxel_lookback(original_object_info_total, nfv_pool, ifv_pool, orientations, container_size,
                                                                                container_shape, rho, max_radio, 
                                                                                packing_alg, orien_evaluation,
                                                                                SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                _select_range = selection_range, random_CA= random_CA, random_CA_threshold = random_CA_threshold,flag_NFV_POOL=flag_NFV_POOL, _TRACE = False) # for SCH
    
                                                                                                                                            # def packing_3D_voxel_lookback(polobject_info_total, nfv_pool, orientation, container_size, 
                                                                                                                                            #                         container_shape, rho, max_radio, 
                                                                                                                                            #                         packing_alg, _evaluation,
                                                                                                                                            #                         SCH_nesting_strategy, density, axis, 
                                                                                                                                            #                         _type, _accessible_check, _encourage_dbl,
                                                                                                                                            #                         _select_range, GRASP, GRASP_threhold, _TRACE):

    
    check_2 = time.time()   

    # print('pieces orientation pool is: ',pieces_orientation)
    # print('pieces order pool is: ',pieces_order)
    
    # this is for the ILS to select the 
    initial_orientations_list_no_bin = []

    for each_piece in range(len(object_info_total)):
        index = find_element_index(pieces_order, each_piece, path=None)
        each_object_info = layout[index[0]][index[1]]
        initial_orientations_list_no_bin.append(each_object_info["orientation"])

    if visualisation:
        visualize_voxel_model(layout, pieces_order, container_size, container_shape)
    
    N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)

    T = check_2 - check_1
    overall_time_cost += T

    # print(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U_star is {U_star}")         
    trace(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U is {U}, U_star is {U_star}")
    
    no_overlap = all(np.all(each_bin <= 1.5) for each_bin in topos_layout)
    trace(f"Constructive algorithm - Overlap check: {'Pass' if no_overlap else 'NOT pass!'}")
    
    if U_star == None: 
        U_star = U
        
    N_U = N - U_star
    
    # data_pool is to track result of each iteration
    data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                        pieces_order, U_star, N_U, T, selected_info = "None", iteration="origin")
    
    trace("Database updated!")

    # data_pool, layout, topos_layout, radio_layout,
    # pieces_order, pieces_orientation, pieces_orientation_value_pool, 
    # pieces_packing_position, pieces_packing_position_pool,
    # U_star, N_U, T, selected_info, iteration, neighbour_type

    # draw the bar chart for the nfv
    # x_labels = [str(k) for k in nfv_pool.keys()]
    # y_values = list(nfv_pool.values())
    # 
    # plt.figure(figsize=(6, 4))
    # plt.bar(x_labels, y_values)
    # plt.xlabel("Tuple keys")
    # plt.ylabel("calculation times")
    # plt.title(f"NFV calculation times {sum(y_values)}")
    # plt.xticks(rotation=30)
    # plt.tight_layout()
    # plt.show()

    # =========================================================================================================
    # if we activate nfv_ppol, this is to count the number of replicated of NFV
    # print("all calculation time of nfv is", nfv_pool.all_nfv_cal)
    # print("validate nfv cal is",nfv_pool.val_nfv_cal)
    # print("repilcated nfv cal is",nfv_pool.rep_nfv_cal)
    # print(f"Replicated NFVs: {nfv_pool.rep_nfv_cal/nfv_pool.all_nfv_cal * 100} %")
    
    if ALG == "fixed_CA" or ALG == "random_CA" or ALG == "BLF":
        return N, U, U_star, N, U, U_star, \
            layout, topos_layout, topos_layout,  \
            pieces_order, initial_orientations_list_no_bin, initial_orientations_list_no_bin,None,None,overall_time_cost

    # ==========================================================================================================
    trace("Iterative Local Search started!")
    local_best_iteration = "origin"
    global_best_iteration = local_best_iteration

    local_best_perform = N_U
    global_best_perform = local_best_perform
    
    n_iter = 1
    n_kick = 1
    
    nonstop = True
    
    # aabb_volume_list = [get_aabb_volume(i[0]) for i in object_info]
    # aabb_sorted = np.argsort(aabb_volume_list)[::-1]

    # to track the orientaions of constructive algorithm
    # initial_selected_info = []
    # for each_piece in range(len(object_total)):
    #     index = find_element_index(pieces_order,each_piece)
    #     initial_selected_info.append((each_piece,pieces_orientation[index[0]][index[1]]))
    

    local_best_orientations_list_no_bin = initial_orientations_list_no_bin
    global_best_orientations_list_no_bin = local_best_orientations_list_no_bin

    # print(local_best_orientations_list_no_bin)

    local_best_list = []
    local_change_iter_list = []
    
    no_change_time = 0
    n_kick = 1  
    # kick_flag = False
    # kick_iteration = 0
    
    original_object_info_total = copy.deepcopy(object_info_total)

    while nonstop:   
        

        # decide which piece
        selected_index = pieces_selection_ls(object_info_total, alpha = alpha, seed=None)

        # decide which orientation
        # selected_orientation = orientation_selection_ls(orientations)

        other_orientations = list(orientations_list)
        other_orientations.remove(local_best_orientations_list_no_bin[selected_index])

        for each_orientation in other_orientations: 

            trace(f"===================== iteration {n_iter} =========================")
            
            selected_info = (selected_index, each_orientation)

            trace (f"Object: {selected_index}, orientation: {each_orientation}, original orientation: {local_best_orientations_list_no_bin[selected_index]}")
            trace (f"Current local best iteration is {local_best_iteration}")
            trace (f"Current global best iteration is {global_best_iteration}")

            start = time.time()
            
            layout, topos_layout, radio_layout, pieces_order = repacking_new_ILS(original_object_info_total, selected_info, nfv_pool, ifv_pool, local_best_orientations_list_no_bin, data_pool, local_best_iteration, ils_orientations, container_size,
                                                                                container_shape, rho, max_radio, 
                                                                                packing_alg, orien_evaluation,
                                                                                SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                _type = "bounding_box",_accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL,  _TRACE = False)

            end = time.time()

            # kick_flag = False
            kick_iteration = 0

            T = end - start
            overall_time_cost += T
            
            trace(f"=== Re-pack finished! Cost {T} s in this iteration, It takes {overall_time_cost} s overall ===")
            # trace(pieces_order)
            
            # no_overlap = all(np.all(each_bin <= 1.5) for each_bin in layout)
            # trace(f"Repack - Overlap: {'No' if no_overlap else 'Yes'}")

            if overall_time_cost > time_limit:
                trace("Has reached the time limit, stop!")
                nonstop = False
                break
            
            elif n_iter == iteration_limit and overall_time_cost < time_limit:
                
                
                N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
                
                if U_star == None: 
                    U_star = U
                    
                perform_this_iter = N - U_star
                
                
                if perform_this_iter < global_best_perform: 
                    trace(f"A GLOBAL best is found in iteration - {n_iter}")
                    no_change_time = 0
                    old_best = global_best_perform
                    global_best_iteration = n_iter
                    global_best_perform = perform_this_iter

                    local_best_orientations_list_no_bin[selected_index] = each_orientation
                    global_best_orientations_list_no_bin = local_best_orientations_list_no_bin
                    trace("GLOBAL best updated") 
                    
                
                if perform_this_iter < local_best_perform: 
                    no_change_time = 0
                    # smaller values are better 
                    old_best = local_best_perform
                    local_best_iteration = n_iter
                    local_best_perform = perform_this_iter
                    local_best_list.append(local_best_perform)
                    local_change_iter_list.append(n_iter)
                    trace(f"Previous orientations of pieces are: {local_best_orientations_list_no_bin}")

                    local_best_orientations_list_no_bin[selected_index] = each_orientation # UPDATE THE BEST OPRIENTATION IF IT'S BETTER

                    trace(f"A better LOCAL result is found in iteration - {n_iter}") 
                    trace(f"Previous best (N-U*) is {old_best}, now is {local_best_perform}")
                    trace(f"Now best orientations of pieces are: {local_best_orientations_list_no_bin}")

                    data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                            pieces_order, U_star, perform_this_iter, T, selected_info, iteration=n_iter)
                    prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
                    trace("Database updated!")

                trace("Has reached the iteration limit, stop!")
                nonstop = False
                break
            
            else:
                
                N, U, U_star = get_final_performance(object_info_total,container_size, container_shape, layout)
                
                if U_star == None: 
                    U_star = U
                    
                perform_this_iter = N - U_star
                    
                # data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                #             pieces_order, U_star, perform_this_iter, T, selected_info, iteration=n_iter)
                
                # trace("Database updated!")
            
            if perform_this_iter < global_best_perform: 
                trace(f"A GLOBAL best is found in iteration - {n_iter}")
                no_change_time = 0
                old_best = global_best_perform
                global_best_iteration = n_iter
                global_best_perform = perform_this_iter

                local_best_orientations_list_no_bin[selected_index] = each_orientation
                global_best_orientations_list_no_bin = local_best_orientations_list_no_bin
                trace("GLOBAL best updated") 

            if perform_this_iter < local_best_perform: 
                no_change_time = 0
                # smaller values are better 
                old_best = local_best_perform
                local_best_iteration = n_iter
                local_best_perform = perform_this_iter
                local_best_list.append(local_best_perform)
                local_change_iter_list.append(n_iter)
                trace(f"Previous orientations of pieces are: {local_best_orientations_list_no_bin}")

                local_best_orientations_list_no_bin[selected_index] = each_orientation # UPDATE THE BEST OPRIENTATION IF IT'S BETTER

                trace(f"A better LOCAL result is found in iteration - {n_iter}") 
                trace(f"Previous best (N-U*) is {old_best}, now is {local_best_perform}")
                trace(f"Now best orientations of pieces are: {local_best_orientations_list_no_bin}")

                data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                            pieces_order, U_star, perform_this_iter, T, selected_info, iteration=n_iter)
                prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
                trace("Database updated!")

                n_iter += 1
                break # break to the next pieces
                
            else:
                n_iter += 1 
                no_change_time += 1
                trace("This iteration NOT makes result better!")     
                trace(f"No-improvement Rep - {no_change_time}")     

            

            if no_change_time >= kick_trigger_time: 
                trace(f"===================== Kick {n_kick} =========================") 
                trace(f"!!!!Solution no change for {no_change_time} replications, a kick is triggered!!!!")
                # kick_flag = True
                no_change_time = 0 # re-count the no change time 
                trace("No change times return 0!")

                kick_iteration = n_iter

                # can be parameterised
                # if after an amount of time, the solution is unchanged
                # we give a kick to the orientations

                tem = list(global_best_orientations_list_no_bin)

                if kick_level == "small":
                    change_n = int(np.ceil(len(object_info_total)/4))
                    change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)   
                    for each_position in change_position:
                        tem[each_position] = np.random.choice(orientations_list)

                elif kick_level == "medium":
                    change_n = int(np.ceil(len(object_info_total)/2))
                    change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)   
                    for each_position in change_position:
                        tem[each_position] = np.random.choice(orientations_list)

                elif kick_level == "large":
                    change_n = int(np.ceil(3*len(object_info_total)/4))
                    change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)   
                    for each_position in change_position:
                        tem[each_position] = np.random.choice(orientations_list)
                
                local_best_orientations_list_no_bin = tem

                trace("Kick repacking started")
                start = time.time()
                
                layout, topos_layout, radio_layout, pieces_order = kick_repacking(original_object_info_total, nfv_pool, ifv_pool, orientations, local_best_orientations_list_no_bin, container_size,
                                                                                container_shape, rho, max_radio, 
                                                                                packing_alg, orien_evaluation,
                                                                                SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL, _TRACE = False)
                start = time.time()
                
                overall_time_cost += (end-start)

                N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
                
                if U_star == None: 
                    U_star = U
                    
                perform_this_iter = N - U_star
                
                # kick data must be the local best for the new iterations
                data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                            pieces_order, U_star, perform_this_iter, "kick", selected_info = "None", iteration=n_iter)

                local_best_list.append(local_best_perform)
                local_change_iter_list.append(kick_iteration-1)

                local_best_iteration = kick_iteration
                prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
                trace("Database updated!")
                
                local_best_list.append(perform_this_iter)
                local_change_iter_list.append(local_best_iteration)

                trace(f"Current local best (N-U*) is {local_best_perform}, kick leads to {perform_this_iter}")
                trace(f"Local best iteration updated to the kick iteration {kick_iteration}!")

                if perform_this_iter < global_best_perform:
                    # highly unlikely
                    old_best = global_best_perform
                    global_best_iteration = n_iter
                    global_best_perform = perform_this_iter
                    global_best_orientations_list_no_bin = tem
                
                local_best_perform = perform_this_iter
                
                n_iter += 1
                n_kick += 1

        if nonstop == False:
            break
        
        
        
    # ==================================================================================
    # read the final results
    best_data = data_pool[global_best_iteration]
    original_data = data_pool["origin"]

    best_pieces_order = best_data["pieces_order"]
    best_topos_layout = best_data["bin_topos_layout"]
    # origin_topos_layout = original_data["bin_topos_layout"]

    best_current_layout = best_data["bin_real_layout"]
    origin_current_layout = original_data["bin_real_layout"]


    best_N, best_U, best_U_star = get_final_performance(object_info_total, container_size, container_shape, best_current_layout)
    origin_N, origin_U, origin_U_star = get_final_performance(object_info_total, container_size, container_shape, origin_current_layout)

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

    
    if visualisation:
        visualize_voxel_model(best_current_layout, best_pieces_order, container_size, container_shape)
    

    # draw the bar chart for the nfv
    # x_labels = [str(k) for k in nfv_pool.keys()]
    # y_values = list(nfv_pool.values())
    # plt.figure(figsize=(6, 4))
    # plt.bar(x_labels, y_values)
    # plt.xlabel("Tuple keys")
    # plt.ylabel("calculation times")
    # plt.title(f"NFV calculation times {sum(y_values)}")
    # plt.xticks(rotation=30)
    # plt.tight_layout()
    # plt.show()

    return best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,  \
            best_current_layout, origin_current_layout, best_topos_layout,  \
            best_pieces_order, initial_orientations_list_no_bin, global_best_orientations_list_no_bin,local_best_list,local_change_iter_list,overall_time_cost
    
    # df = pd.DataFrame(data_pool)
# ============================================================================



def GRASP(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations, orientations_list,
            packing_alg, selection_type, selection_range, accessible_check,
            SCH_nesting_strategy, orien_evaluation,
            container_size, container_shape,
            iter_limit_per_iter, time_limit, alpha, flag_NFV_POOL, visualisation, _TRACE):

    global TRACE 
    TRACE = _TRACE

    # GRASP: (Random Constructive Algorithm + LS) in a loop

    # structure of object_info_total
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

    

    data_pool = {}

    # trace("============================================================")
    trace(f"Objects number: {len(object_info_total)}")
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

    random_CA = True
    # random_CA_seed = 41 # to decide the seed of the packing positon seletion
    random_CA_threshold = 5 # this means best five packing position is selected randomly.

    GRASP_LOOP = True
    n_iter_GRASP = 1
    overall_n_packing = 0
    overall_time_cost = 0
    global_best_perform = 999

    while GRASP_LOOP:

        original_object_info_total = copy.deepcopy(object_info_total)

        trace(f" =========== GRASP Iteration {n_iter_GRASP}: begins! ============")
        trace(f" This is packing {overall_n_packing+1} overall! CA starts")
        check_1 = time.time()

        layout, topos_layout, radio_layout, pieces_order = packing_3D_voxel_lookback(original_object_info_total, nfv_pool, ifv_pool, orientations, container_size,
                                                                                    container_shape, rho, max_radio, 
                                                                                    packing_alg, orien_evaluation,
                                                                                    SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                    _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                    _select_range = selection_range, random_CA= random_CA, random_CA_threshold = random_CA_threshold, flag_NFV_POOL=flag_NFV_POOL, _TRACE = False) # for SCH
        
                                                                                                                                                # def packing_3D_voxel_lookback(polobject_info_total, nfv_pool, orientation, container_size, 
                                                                                                                                                #                         container_shape, rho, max_radio, 
                                                                                                                                                #                         packing_alg, _evaluation,
                                                                                                                                                #                         SCH_nesting_strategy, density, axis, 
                                                                                                                                                #                         _type, _accessible_check, _encourage_dbl,
                                                                                                                                                #                         _select_range, GRASP, GRASP_threhold, _TRACE):

        
        check_2 = time.time()   

        
        overall_n_packing += 1

        CA_iteration = overall_n_packing 
        # this is for the ILS to select the 
        initial_orientations_list_no_bin = []

        for each_piece in range(len(object_info_total)):
            index = find_element_index(pieces_order, each_piece, path=None)
            each_object_info = layout[index[0]][index[1]]
            initial_orientations_list_no_bin.append(each_object_info["orientation"])

        # if visualisation:
        #     visualize_voxel_model(layout, pieces_order, container_size, container_shape)
        
        N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)

        T = check_2 - check_1
        overall_time_cost += T # this time needs to be counted 

        # global_best_orientations_list_no_bin = [i.orien for i in re_CA_packed_pieces]
        
        # print(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U_star is {U_star}")         
        trace(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U is {U}, U_star is {U_star}")
        
        no_overlap = all(np.all(each_bin <= 1.5) for each_bin in topos_layout)
        trace(f"Constructive algorithm - Overlap check: {'Pass' if no_overlap else 'NOT pass!'}")
        
        if U_star == None: 
            U_star = U
            
        perform_this_iter = N - U_star
        # perform_this_iter = N_U
        # data_pool is to track result of each iteration

        # =========================================================================================================
        # if we activate nfv_ppol, this is to count the number of replicated of NFV
        # print("all calculation time of nfv is", nfv_pool.all_nfv_cal)
        # print("validate nfv cal is",nfv_pool.val_nfv_cal)
        # print("repilcated nfv cal is",nfv_pool.rep_nfv_cal)
        # print(f"Replicated NFVs: {nfv_pool.rep_nfv_cal/nfv_pool.all_nfv_cal * 100} %")
        
        # ==========================================================================================================

        # local_best_iteration = overall_n_packing
        # global_best_iteration = overall_n_packing

        # local_best_perform = N_U
        # global_best_perform = N_U

        if perform_this_iter < global_best_perform: 
            trace(f"A GLOBAL best is found in iteration - {overall_n_packing}")
            old_best = global_best_perform
            global_best_iteration = overall_n_packing
            global_best_perform = perform_this_iter

            # local_best_orientations_list_no_bin = each_orientation
            # global_best_orientations_list_no_bin[selected_index] = each_orientation
            global_best_orientations_list_no_bin = initial_orientations_list_no_bin

            trace(f"A new GLOBAL best result is found in iteration - {CA_iteration}") 
            trace(f"Previous best (N-U*) is {old_best}, now is {perform_this_iter}")
            # trace(f"Now best orientations of pieces are: {global_best_orientations_list_no_bin}")
            # n_iter += 1 
            data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                        pieces_order, U_star, perform_this_iter, T, selected_info = "None", iteration= CA_iteration)
            prune_data_pool(data_pool, 1, global_best_iteration, CA_iteration)
            trace("Database updated!")

            keep_this_iter = False
            # break # break to the next GRASP iteration

        else:
            # n_iter += 1
            data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                            pieces_order, U_star, perform_this_iter, T, selected_info = "None", iteration= CA_iteration)
            prune_data_pool(data_pool, 1, global_best_iteration, CA_iteration)
            global_best_orientations_list_no_bin = initial_orientations_list_no_bin
            trace("This iteration NOT makes result better!")


        if overall_time_cost > time_limit:
            trace("Has reached the time limit, stop!")
            ALG_stop = True
            break

        else:
            ALG_stop = False


        keep_this_iter = True

        n_iter = 1 # this is to track the num of iter in each GRASP iter

        original_object_info_total = copy.deepcopy(object_info_total) # re-initialise
        trace("Local Search started!")
        while keep_this_iter:

            # decide which piece
            selected_index = pieces_selection_ls(object_info_total, alpha = alpha, seed=None)

            # decide which orientation
            # selected_orientation = orientation_selection_ls(orientations)

            other_orientations = list(orientations_list)
            other_orientations.remove(global_best_orientations_list_no_bin[selected_index])     

            for each_orientation in other_orientations: 

                trace(f"===================== iteration {n_iter} =========================")
                trace(f" This is packing {overall_n_packing+1} overall!")
                selected_info = (selected_index, each_orientation)

                trace (f"Object: {selected_index}, orientation: {each_orientation}, original orientation: {global_best_orientations_list_no_bin[selected_index]}")
                # trace (f"Current local best iteration is {local_best_iteration}")
                trace (f"Current global best iteration is {global_best_iteration}")

                start = time.time()
                # this is one iteration to repack
                # this is the local search based on the orientation
                layout, topos_layout, radio_layout, pieces_order = repacking_new_ILS(original_object_info_total, selected_info, nfv_pool, ifv_pool, global_best_orientations_list_no_bin, data_pool, 
                                                                                     CA_iteration, ils_orientations, container_size,
                                                                                    container_shape, rho, max_radio, 
                                                                                    packing_alg, orien_evaluation,
                                                                                    SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                    _type = "bounding_box",_accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                    _select_range = selection_range, flag_NFV_POOL = flag_NFV_POOL,  _TRACE = False)

                end = time.time()


                T = end - start
                overall_time_cost += T
                
                trace(f"=== Re-pack finished! Cost {T} s in this iteration, It takes {overall_time_cost} s overall ===")
                overall_n_packing += 1

                # n_iter += 1
                # trace(pieces_order)


                if overall_time_cost > time_limit:
                    trace("Has reached the time limit, stop!")
                    ALG_stop = True
                    break

                elif overall_time_cost < time_limit:      
                    
                    N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
                    
                    if U_star == None: 
                        U_star = U
                        
                    perform_this_iter = N - U_star
                    
                    data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                                pieces_order, U_star, perform_this_iter, T, selected_info, iteration=overall_n_packing)
                    
                    trace("Database updated!")
                
                if perform_this_iter < global_best_perform: 
                    trace(f"A GLOBAL best is found in iteration - {overall_n_packing}")
                    old_best = global_best_perform
                    global_best_iteration = overall_n_packing
                    global_best_perform = perform_this_iter

                    # local_best_orientations_list_no_bin = each_orientation
                    global_best_orientations_list_no_bin[selected_index] = each_orientation
                    trace(f"A new GLOBAL best result is found in iteration - {overall_n_packing}") 
                    trace(f"Previous best (N-U*) is {old_best}, now is {perform_this_iter}")
                    trace(f"Now best orientations of pieces are: {global_best_orientations_list_no_bin}")
                    # n_iter += 1 

                    keep_this_iter = False

                    data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                                            pieces_order, U_star, perform_this_iter, T, selected_info, iteration= overall_n_packing)
                    
                    trace("Database updated!")

                    break # break to the next GRASP iteration
                    
                else:
                    # n_iter += 1 
                    trace("This iteration NOT makes result better!")      

                if n_iter >= iter_limit_per_iter:
                    trace("Has reached the iteration in this GRASP iter limit, move to the next GRASP iter!")
                    keep_this_iter = False
                    break # break out of this for loop

                else:
                    n_iter += 1

            if ALG_stop:
                break

        n_iter_GRASP += 1     

        if ALG_stop:
            # jump out the overall loop
            break



    # ==================================================================================
    # read the final results
    best_data = data_pool[global_best_iteration]
    original_data = data_pool[1]

    best_pieces_order = best_data["pieces_order"]
    best_topos_layout = best_data["bin_topos_layout"]
    # origin_topos_layout = original_data["bin_topos_layout"]

    best_current_layout = best_data["bin_real_layout"]
    origin_current_layout = original_data["bin_real_layout"]


    best_N, best_U, best_U_star = get_final_performance(object_info_total, container_size, container_shape, best_current_layout)
    origin_N, origin_U, origin_U_star = get_final_performance(object_info_total, container_size, container_shape, origin_current_layout)

    if best_U_star == None:
        best_U_star = best_U

    if origin_U_star == None:
        origin_U_star = origin_U


    if visualisation:
        visualize_voxel_model(best_current_layout,best_pieces_order, container_size, container_shape)


    return best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,  \
            best_current_layout, origin_current_layout, best_topos_layout,  \
            best_pieces_order,overall_time_cost


def GRASP_ILS(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations, orientations_list,
                packing_alg, selection_type, selection_range, accessible_check,
                SCH_nesting_strategy, orien_evaluation,
                container_size, container_shape,
                iter_limit_per_iter, kick_trigger_time, kick_level, time_limit, alpha, flag_NFV_POOL, visualisation, _TRACE):
    
    global TRACE 
    TRACE = _TRACE


    # GRASP: (Random Constructive Algorithm + LS) in a loop

    # structure of object_info_total
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

    

    data_pool = {}

    # trace("============================================================")
    trace(f"Objects number: {len(object_info_total)}")
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

    random_CA = True
    # random_CA_seed = 41 # to decide the seed of the packing positon seletion
    random_CA_threshold = 5 # this means best five packing position is selected randomly.

    GRASP_LOOP = True
    n_iter_GRASP = 1
    overall_n_packing = 0
    overall_time_cost = 0
    global_best_perform = 999
    local_best_perform = 999

    while GRASP_LOOP:

        original_object_info_total = copy.deepcopy(object_info_total)

        trace(f" =========== GRASP Iteration {n_iter_GRASP}: begins! ============")
        trace(f" This is packing {overall_n_packing+1} overall! CA starts")
        check_1 = time.time()

        layout, topos_layout, radio_layout, pieces_order = packing_3D_voxel_lookback(original_object_info_total, nfv_pool, ifv_pool, orientations, container_size,
                                                                                    container_shape, rho, max_radio, 
                                                                                    packing_alg, orien_evaluation,
                                                                                    SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                    _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                    _select_range = selection_range, random_CA= random_CA, random_CA_threshold = random_CA_threshold,flag_NFV_POOL = flag_NFV_POOL, _TRACE = False) # for SCH
        
                                                                                                                                                # def packing_3D_voxel_lookback(polobject_info_total, nfv_pool, orientation, container_size, 
                                                                                                                                                #                         container_shape, rho, max_radio, 
                                                                                                                                                #                         packing_alg, _evaluation,
                                                                                                                                                #                         SCH_nesting_strategy, density, axis, 
                                                                                                                                                #                         _type, _accessible_check, _encourage_dbl,
                                                                                                                                                #                         _select_range, GRASP, GRASP_threhold, _TRACE):

        check_2 = time.time()   

        overall_n_packing += 1
        no_change_time = 0

        # this is for the ILS to select the 
        initial_orientations_list_no_bin = []
        for each_piece in range(len(object_info_total)):
            index = find_element_index(pieces_order, each_piece, path=None)
            each_object_info = layout[index[0]][index[1]]
            initial_orientations_list_no_bin.append(each_object_info["orientation"])

        # if visualisation:
        #     visualize_voxel_model(layout, pieces_order, container_size, container_shape)
        
        N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)

        T = check_2 - check_1
        overall_time_cost += T # this time needs to be counted 

        # print(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U_star is {U_star}")         
        trace(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U is {U}, U_star is {U_star}")
        
        no_overlap = all(np.all(each_bin <= 1.5) for each_bin in topos_layout)
        trace(f"Constructive algorithm - Overlap check: {'Pass' if no_overlap else 'NOT pass!'}")
        
        if U_star == None: 
            U_star = U
            
        perform_this_iter = N - U_star
        # perform_this_iter = N_U
        # data_pool is to track result of each iteration
        # data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
        #                     pieces_order, U_star, perform_this_iter, T, selected_info = "None", iteration= overall_n_packing)

        # =========================================================================================================
        # if we activate nfv_ppol, this is to count the number of replicated of NFV
        # print("all calculation time of nfv is", nfv_pool.all_nfv_cal)
        # print("validate nfv cal is",nfv_pool.val_nfv_cal)
        # print("repilcated nfv cal is",nfv_pool.rep_nfv_cal)
        # print(f"Replicated NFVs: {nfv_pool.rep_nfv_cal/nfv_pool.all_nfv_cal * 100} %")
        
        # ==========================================================================================================
        trace("Local Search started!")
        # local_best_iteration = overall_n_packing
        # global_best_iteration = overall_n_packing

        # local_best_perform = N_U
        # global_best_perform = N_U

        if perform_this_iter < global_best_perform: 
            trace(f"A GLOBAL best is found in iteration - {overall_n_packing}")
            
            old_best = global_best_perform
            global_best_iteration = overall_n_packing
            global_best_perform = perform_this_iter

            # local_best_orientations_list_no_bin = each_orientation
            # global_best_orientations_list_no_bin[selected_index] = each_orientation
            local_best_orientations_list_no_bin = initial_orientations_list_no_bin
            global_best_orientations_list_no_bin = initial_orientations_list_no_bin

            trace(f"A new GLOBAL best result is found in iteration - {overall_n_packing}") 
            trace(f"Previous best (N-U*) is {old_best}, now is {perform_this_iter}")
            # trace(f"Now best orientations of pieces are: {global_best_orientations_list_no_bin}")
            # n_iter += 1 
            # break # break to the next GRASP iteration

        if perform_this_iter < local_best_perform:

            
            # smaller values are better 
            old_best = local_best_perform
            local_best_iteration = overall_n_packing
            local_best_perform = perform_this_iter
            # local_best_list.append(local_best_perform)
            # local_change_iter_list.append(n_iter)
            trace(f"Previous orientations of pieces are: {local_best_orientations_list_no_bin}")
            # local_best_orientations_list_no_bin[selected_index] = each_orientation # UPDATE THE BEST OPRIENTATION IF IT'S BETTER

            trace(f"A better LOCAL result is found in iteration - {overall_n_packing}") 
            trace(f"Previous best (N-U*) is {old_best}, now is {local_best_perform}")
            # trace(f"Now best orientations of pieces are: {local_best_orientations_list_no_bin}")
            
            data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                            pieces_order, U_star, perform_this_iter, T, selected_info = "None", iteration= overall_n_packing)
            prune_data_pool(data_pool, 1, local_best_iteration, global_best_iteration)
            trace("Database updated!")

        else:
            # n_iter += 1
            no_change_time += 1
            trace("This iteration NOT makes result better!")

        ALG_stop = False


        keep_this_iter = True

        n_iter = 1 # this is to track the num of iter in each GRASP iter

        original_object_info_total = copy.deepcopy(object_info_total) # re-initialise
        no_change_time = 0
        n_kick = 1

        while keep_this_iter:

            # decide which piece
            selected_index = pieces_selection_ls(object_info_total, alpha = alpha, seed=None)

            # decide which orientation
            # selected_orientation = orientation_selection_ls(orientations)

            other_orientations = list(orientations_list)
            other_orientations.remove(global_best_orientations_list_no_bin[selected_index])     

            for each_orientation in other_orientations: 

                trace(f"===================== iteration {n_iter} =========================")
                trace(f" This is packing {overall_n_packing+1} overall!")
                selected_info = (selected_index, each_orientation)

                trace (f"Object: {selected_index}, orientation: {each_orientation}, original orientation: {local_best_orientations_list_no_bin[selected_index]}")
                trace (f"Current local best iteration is {local_best_iteration}")
                trace (f"Current global best iteration is {global_best_iteration}")

                start = time.time()

                # this is one iteration to repack
                layout, topos_layout, radio_layout, pieces_order = repacking_new_ILS(original_object_info_total, selected_info, nfv_pool, ifv_pool, local_best_orientations_list_no_bin, data_pool, 
                                                                                     local_best_iteration, ils_orientations, container_size,
                                                                                    container_shape, rho, max_radio, 
                                                                                    packing_alg, orien_evaluation,
                                                                                    SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                    _type = "bounding_box",_accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                    _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL,  _TRACE = False)

                end = time.time()


                T = end - start
                overall_time_cost += T
                
                trace(f"=== Re-pack finished! Cost {T} s in this iteration, It takes {overall_time_cost} s overall ===")
                overall_n_packing += 1
                n_iter += 1

                # n_iter += 1
                # trace(pieces_order)


                if overall_time_cost > time_limit:
                    trace("Has reached the time limit, stop!")
                    ALG_stop = True
                    break

                elif overall_time_cost < time_limit:      
                    
                    N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
                    
                    if U_star == None: 
                        U_star = U
                        
                    trace("Database updated!")

                perform_this_iter = N - U_star

                if perform_this_iter < global_best_perform: 
                    trace(f"A GLOBAL best is found in iteration - {overall_n_packing}")
                    old_best = global_best_perform
                    global_best_iteration = overall_n_packing
                    global_best_perform = perform_this_iter
                    no_change_time = 0
                      
                    # local_best_orientations_list_no_bin = each_orientation
                    global_best_orientations_list_no_bin[selected_index] = each_orientation
                    trace(f"A new GLOBAL best result is found in iteration - {overall_n_packing}") 
                    trace(f"Previous best (N-U*) is {old_best}, now is {perform_this_iter}")
                    trace(f"Now best orientations of pieces are: {global_best_orientations_list_no_bin}")
                    # n_iter += 1 

                    # keep_this_iter = False
                    # break # break to the next GRASP iteration

                if perform_this_iter < local_best_perform:

                    # smaller values are better 
                    old_best = local_best_perform
                    local_best_iteration = overall_n_packing
                    local_best_perform = perform_this_iter
                    no_change_time = 0
                    # local_best_list.append(local_best_perform)
                    # local_change_iter_list.append(n_iter)
                    trace(f"Previous orientations of pieces are: {local_best_orientations_list_no_bin}")
                    local_best_orientations_list_no_bin[selected_index] = each_orientation # UPDATE THE BEST OPRIENTATION IF IT'S BETTER

                    trace(f"A better LOCAL result is found in iteration - {overall_n_packing}") 
                    trace(f"Previous best (N-U*) is {old_best}, now is {local_best_perform}")
                    # trace(f"Now best orientations of pieces are: {local_best_orientations_list_no_bin}")

                    data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                                pieces_order, U_star, perform_this_iter, T, selected_info, iteration=overall_n_packing)
                    prune_data_pool(data_pool, 1, local_best_iteration, global_best_iteration)
                    trace("Database updated!")

                else:
                    # n_iter += 1
                    no_change_time += 1
                    trace("This iteration NOT makes result better!")
                    trace(f"No-improvement Rep - {no_change_time}")        

                # judge if the iteration time iter_limit_per_iter
                if n_iter > iter_limit_per_iter:
                    trace("Has reached the iteration in this GRASP iter limit, move to the next GRASP iter!")
                    keep_this_iter = False
                    break # break out of this for loop


                
                if no_change_time >= kick_trigger_time: 

                    trace(f"===================== Kick {n_kick} =========================") 
                    trace(f"!!!!Solution no change for {no_change_time} replications, a kick is triggered!!!!")
                    # kick_flag = True
                    no_change_time = 0 # re-count the no change time 
                    trace("No change times return 0!")

                    kick_iteration = n_iter

                    # can be parameterised
                    # if after an amount of time, the solution is unchanged
                    # we give a kick to the orientations

                    tem = list(global_best_orientations_list_no_bin)

                    if kick_level == "small":
                        change_n = int(np.ceil(len(object_info_total)/4))
                        change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)
                        for each_position in change_position:
                            tem[each_position] = np.random.choice(orientations_list)

                    elif kick_level == "medium":
                        change_n = int(np.ceil(len(object_info_total)/2))
                        change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)
                        for each_position in change_position:
                            tem[each_position] = np.random.choice(orientations_list)

                    elif kick_level == "large":
                        change_n = int(np.ceil(3*len(object_info_total)/4))
                        change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)
                        for each_position in change_position:
                            tem[each_position] = np.random.choice(orientations_list)

                    local_best_orientations_list_no_bin = tem

                    trace("Kick repacking started")

                    layout, topos_layout, radio_layout, pieces_order = kick_repacking(original_object_info_total, nfv_pool, ifv_pool, orientations, local_best_orientations_list_no_bin, container_size,
                                                                                    container_shape, rho, max_radio, 
                                                                                    packing_alg, orien_evaluation,
                                                                                    SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                    _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                    _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL, _TRACE = False)

                    N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)

                    overall_n_packing += 1
                    n_kick += 1
                    n_iter += 1
                    no_change_time = 0

                    if U_star == None: 
                        U_star = U
                        
                    perform_this_iter = N - U_star
                    
                    data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                                pieces_order, U_star, perform_this_iter, "kick", selected_info, iteration=overall_n_packing)

                    local_best_iteration = overall_n_packing
                    prune_data_pool(data_pool, 1, local_best_iteration, global_best_iteration)
                    trace("Database updated!")
                    
                    # local_best_list.append(perform_this_iter)
                    # local_change_iter_list.append(local_best_iteration)

                    trace(f"Current local best (N-U*) is {local_best_perform}, kick leads to {perform_this_iter}")
                    trace(f"Local best iteration updated to the kick iteration {kick_iteration}!")

                    if perform_this_iter < global_best_perform:
                        # highly unlikely
                        old_best = global_best_perform
                        global_best_iteration = overall_n_packing

                        global_best_perform = perform_this_iter
                        global_best_orientations_list_no_bin = tem
                    
                    local_best_perform = perform_this_iter
                    
                    # check if it reaches iter_limit_per_iter
                    if n_iter > iter_limit_per_iter:
                        trace("Has reached the iteration in this GRASP iter limit, move to the next GRASP iter!")
                        keep_this_iter = False
                        break # break out of this for loop
                   
            if ALG_stop:
                break

        n_iter_GRASP += 1     

        if ALG_stop:
            # jump out the overall loop
            break
        
        
        
    # ==================================================================================
    # read the final results
    best_data = data_pool[global_best_iteration]
    original_data = data_pool[1]

    best_pieces_order = best_data["pieces_order"]
    best_topos_layout = best_data["bin_topos_layout"]
    # origin_topos_layout = original_data["bin_topos_layout"]

    best_current_layout = best_data["bin_real_layout"]
    origin_current_layout = original_data["bin_real_layout"]


    best_N, best_U, best_U_star = get_final_performance(object_info_total, container_size, container_shape, best_current_layout)
    origin_N, origin_U, origin_U_star = get_final_performance(object_info_total, container_size, container_shape, origin_current_layout)

    if best_U_star == None:
        best_U_star = best_U

    if origin_U_star == None:
        origin_U_star = origin_U


    if visualisation:
        visualize_voxel_model(best_current_layout,best_pieces_order, container_size, container_shape)


    return best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,  \
            best_current_layout, origin_current_layout, best_topos_layout,  \
            best_pieces_order,overall_time_cost