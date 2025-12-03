import numpy as np
import copy 
import binvox_rw
import plotly.graph_objects as go
import time 
import math
import line_profiler
# import matplotlib.pyplot as plt
# from mayavi import mlab
# from mpl_toolkits.mplot3d import Axes3D
from SCH_iter_ls import SC_heuristic
from function_lib import insert_add, get_ifv_cube_strip_packing,aabb_rotate
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

global TRACE # a switch for output 
# TRACE = True

    
def trace(msg):
    if TRACE: 
        print(msg)
         
def normalised(data):
    # Move the reference point of the object to (0,0,0) 
    # reference is the bottom backc voxel of a voxelised object
    
    x,y,z = np.where(data == 1)
    x_normal = min(x)
    y_normal = min(y)
    z_normal = min(z)

    if (x_normal,y_normal,z_normal) == (0,0,0):
        return data
    
    else:
        data = translate_voxel(data,(-x_normal,-y_normal,-z_normal))
        return data 
    

def read_binvox_file(filepath):
    with open(filepath, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)  # read binvox data as a 3D array
        
    return model.data  # this is a bool array

# 1. Translation function - move the voxel array by a specified shift
def translate_voxel(voxel_data, steps):
    # tem = []
    # # Use np.roll to shift the array; shift is (x, y, z) displacement
    # for each_step in steps:
    #     tem.append(-each_step)
        
    translated_data = np.roll(voxel_data, shift=steps, axis=(0, 1, 2))
    
    
    return translated_data

# 2. Rotation function - rotate the voxel array by a specified angle, rotate by x axis (1,2)  y axis  (0,2)  z axis (0,1)
def rotate_voxel(voxel_data, angle, rotate_axes):
    
    # angle is the rotation angle, axes defines the plane of rotation (default: y-z plane)
    # will move item to (0,0,0) as default in the output. 
    if angle == 0: 
        return voxel_data
    
    else:
        if rotate_axes  == 'x':
            _axes = (1,2)
        elif rotate_axes == 'y':
            _axes = (0,2)
        elif rotate_axes == "z":
            _axes = (0,1)
        else:
            print("Error in axes name, use lower case x,y,z!")

        # def rot90(m, k=1, axes=(0, 1)):
        # normalised_data = normalised(voxel_data)

        rotated_data = np.rot90(voxel_data, k = int(angle/90), axes=_axes) # rot90 won't lose metrix info
        normalised_data = normalised(rotated_data)

        return normalised_data

def add_more_space(data,space_length):
    # add space_length in three dimensions
    # it defines the size of container 
    padded_data = np.pad(data, pad_width=space_length, mode='constant', constant_values=0)
    return padded_data

def get_bounding_box(shapes):
    # calculate the max(x,y,z) of a bounding box
    # print(np.shape(shapes))
    x, y, z = np.where (shapes == 1)
    
    delta_x = max(x) - min(x) + 1
    delta_y = max(y) - min(y) + 1
    delta_z = max(z) - min(z) + 1
    
    return delta_x,delta_y,delta_z

def get_max_translation(shapes):   
    # calculate the max(x,y,z) of a shape can move
    
    x,y,z = np.where (shapes == 1)
    
    max_x = len(shapes[0]) 
    max_y = len(shapes[1]) 
    max_z = len(shapes[2]) 
    
    delta_x = max_x - max(x) 
    delta_y = max_y - max(y) 
    delta_z = max_z - max(z)
    
    return delta_x,delta_y,delta_z

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
        
def find_max_xyz(aimed_object):
    """
    Finds the maximum and minimum x, y, z indices where the value in the array is 1.
    
    Parameters:
        aimed_object (np.ndarray): A 3D binary array (0s and 1s).
        
    Returns:
        tuple: (max_x, max_y, max_z, min_x, min_y, min_z)
    """
    # Get indices where the value is 1
    indices = np.argwhere(aimed_object == 1)
    
    if indices.size == 0:
        # If there are no '1's in the object, return None or appropriate defaults
        return None  # Or (0, 0, 0, 0, 0, 0) depending on your use case
    
    # Compute max and min for each dimension
    max_x, max_y, max_z = np.max(indices, axis=0)
    min_x, min_y, min_z = np.min(indices, axis=0)
    
    return max_x, max_y, max_z, min_x, min_y, min_z

def get_intersection(object1, object2):
    # Find max and min coordinates for each box
    max_x1, max_y1, max_z1, min_x1, min_y1, min_z1 = find_max_xyz(object1)
    max_x2, max_y2, max_z2, min_x2, min_y2, min_z2 = find_max_xyz(object2)
    
    # Check if there is no overlap
    if min_x1 >= max_x2 or min_x2 >= max_x1 or min_y1 >= max_y2 or min_y2 >= max_y1 or min_z1 >= max_z2 or min_z2 >= max_z1:
        return 0  # No intersection
    
    # Calculate overlapping volume
    intersect_x = min(max_x1, max_x2) - max(min_x1, min_x2)
    intersect_y = min(max_y1, max_y2) - max(min_y1, min_y2)
    intersect_z = min(max_z1, max_z2) - max(min_z1, min_z2)

    intersect_volume = intersect_x * intersect_y * intersect_z
    return intersect_volume
    
def evaluation(topos_layout_test, current_layout_test, object_info, packing_position, box_size, _evaluation): 

    """_summary_
    
    Evaluate the performance of packing orientation
    original TOPOS: waste - overlap + distance
        
    Args:
        current_layout_test: an object_info 
        position_bin (_type_): which bin is currently looking at
        next_object (_type_): next_object has already placed at the aimed position.
        topos_layout (_type_): merged layout

        
    Returns:
        float/int : 
        The larger value of Evaluation represents a better performance.
        # The smaller, the better. 
        # The smaller, the better. 
        # The smaller, the better. 
        # The smaller, the better. 
    """
    
    # waste
    box = np.zeros(box_size)
    translated_test = insert_add(box,object_info["array"],packing_position)

    if _evaluation == "waste_overlap_distance": 
        max_x, max_y, max_z, min_x, min_y, min_z = find_max_xyz(topos_layout_test)
        
        max_x1, max_y1, max_z1, min_x1, min_y1, min_z1 = find_max_xyz(translated_test)
        
        vol_bound_rec = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        occupied = get_volume(topos_layout_test)
        waste = vol_bound_rec - occupied

        # overlap
        overlap = 0
        
        for each_object_info in current_layout_test: 
            box = np.zeros(box_size)
            # packed_polygon = Polygon(each_polygon)
            packed_test = insert_add(box, each_object_info["array"], each_object_info["translation"])
            overlap += get_intersection(translated_test,packed_test)
            
        # distance
        center_layout = ((max_x - min_x)/2 , (max_y - min_y)/2, (max_z - min_z)/2) # the center of the bounding box of the merged layout
        center_next = ((max_x1 - min_x1)/2 , (max_y1 - min_y1)/2, (max_z1 - min_z1)/2) # the center of the bounding box of the next object
        
        distance = math.sqrt((center_layout[0]-center_next[0])**2 + (center_layout[1]-center_next[1])**2 + (center_layout[2]-center_next[2])**2) 
        # The Euclidean distance of the two centers
      
        # print("EVALUATION FINISED!!")
        # print("THE WASTE IS:",waste)
        # print("THE OVERLAP IS:",overlap)
        # print("THE DISTANCE IS:",distance)
        x,y,z = np.where(topos_layout_test == 1)

        # The larger value of Evaluation represents a better performance. 
        return waste - overlap + distance + max(z)*2 
    
    elif _evaluation == "maximum connected space": 

        bin_size = translated_test.shape
        
        x,y,z = get_bounding_box(translated_test)
        
        residual_box = - max(bin_size[0]*bin_size[1]*(bin_size[2]-z), 
                                bin_size[0]*(bin_size[1]-y)*bin_size[2], 
                                (bin_size[0]-x)*bin_size[1]*bin_size[2]) 
        
        return residual_box
    
    elif _evaluation == "minimum_bb_volume": 

        bin_size = translated_test.shape
        
        x,y,z = get_bounding_box(translated_test)
        
        bb_volume = x*y*z

        x,y,z = np.where(translated_test == 1)

        return int((bb_volume + max(z)**3)/10)
        # return max(z)
    
    elif _evaluation == "minimum_bb_edges_len": 

        bin_size = topos_layout_test.shape
        
        x,y,z = get_bounding_box(topos_layout_test)
        
        bb_edges_len = x+y+z
        
        return bb_edges_len
    
    elif _evaluation == "height_only": 

        x,y,z = np.where(translated_test == 1)

        return max(z)  

# # This is for vertical only  
# def accessibility_check(next_piece, bin_position, topos_layout, container_size):
#     """_summary_
    
#     To check if the current packing position is physically reachable, by pulling the item vertically, 
#     and see if any overlap is detected. 

#     Args:
#         next_piece (3D np array): the ongoing piece after transformation.
#         bin_position (int): number of bin
#         topos_layout (3D np array): before update
#         container_size (tuple): (x,y,z) 
        
#     Returns:
#         bool: True is an acceptable position, False is an unreachable posiition.
        
#     """
#     indices = np.argwhere(next_piece == 1)
#     height = np.max(indices[2], axis=0)
    
#     max_test_distance = container_size[2] - height # to decide the max distance of vertical move of this object 
    
#     test_step = 1
    
#     stop = False
    
#     while stop == False: 
#         next_piece = translate_voxel(next_piece,(0,0,test_step))
#         check = next_piece + topos_layout[bin_position]
        
#         non_overlap = np.all(check < 1.5) # if all cells of matrix are 0 or 1, means no overlap
        
#         if non_overlap == True: 
#             height += test_step
            
#         else:
#             # if any overlap is detected during the testing process, it is not vertically reachable
#             return False
        
#         if height > max_test_distance:
#             # if it has exceeded the container boundary, stop
#             stop = True
            
#         else:
#             pass
            
        
#     return True


# def open_a_new_bin(ifv_pool, current_layout, topos_layout, orientation, each_object_info, container_size, container_shape, 
#                    position_bin, radio_list, pieces_order):
    
#     # open_a_new_bin(ifv_pool, current_layout, topos_layout, orientation, each_object_info, container_size, container_shape, 
#     #                 position_bin, radio_list,pieces_order, pieces_orientation_value_pool,pieces_packing_position_pool)
    
#     best_value_local = 9999999
#     current_layout.append([])
#     topos_layout.append(np.zeros(container_size))
#     position_bin += 1
#     radio_list.append([])
#     pieces_order.append([])

#     trace(f"A new bin {position_bin} has been opened") 
#     tem_orien_value = []
    

#     for each_degree in orientation[0]:
#         trace(f"Try next degree {each_degree}")
#         if each_degree != 0: 
#             # if we rotate it 
#             for each_axis in orientation[1]:
#                 orientation_str = each_axis + '_' + str(each_degree)
#                 trace(f"Try next axis {each_axis}")
#                 # for each orientation of the object       
#                 tem = copy.deepcopy(each_object_info)
#                 # origin_array = back_to_origin(tem)

#                 tem['array'] = rotate_voxel(tem["array"], each_degree, each_axis)
#                 tem["orientation"] = orientation_str

#                 # rotated_shape = rotate_voxel(tem["array"],each_degree,each_axis)
#                 x,y,z = np.where(tem['array'] == 1) # index of where space is occupied
            
#                 # best_value = len(rotated_shape[2])# the length of z axis
                
#                 value = max(z) # the max height of the item
#                 tem_orien_value.append((orientation_str,value))

#                 if container_shape == "cube":
#                     min_x = 0
#                     min_y = 0
                
#                 elif container_shape == "cylinder":
#                     ifv = ifv_pool.retrieve_ifv(tem, container_size, container_shape)
#                     _x, _y, _z = np.where(ifv == 1) 
                    
#                     min_x = np.min(_x) 
#                     min_y = np.min(_y[_x == min_x])
                
#                 trace(f"The value of this position is {value} current best value is {best_value_local}") 
#                 if value < best_value_local:
#                     best_value_local = value
#                     best_degree = each_degree
#                     best_axis = each_axis
#                     best_translation = (min_x, min_y, 0)
                    
#         else: 
#             # if we don't rotate it, don't need to read axis

#             # tem = copy.deepcopy(each_object_info)

#             # tem = copy.deepcopy(each_object_info)
#             # tem['array'] = back_to_origin(tem)
#             # tem["orientation"] = "x_0"

#             x,y,z = np.where(each_object_info["array"] == 1)                
#             # best_value = len(each_object_info[2])# the length of z axis       
#             value = max(z) # the max height of the item
#             tem_orien_value.append(("x_0",value))

#             if container_shape == "cube":
#                 min_x = 0
#                 min_y = 0
            
#             elif container_shape == "cylinder":
#                 ifv = ifv_pool.retrieve_ifv(each_object_info, container_size, container_shape)
#                 _x, _y, _z = np.where(ifv == 1) 
                
#                 min_x = np.min(_x) 
#                 min_y = np.min(_y[_x == min_x])
#                 # best_translation = (min_x, min_y, 0)
                
#             trace(f"The value of this position is {value} current best value is {best_value_local}") 
#             if value < best_value_local:
#                 best_value_local = value
#                 best_degree = 0
#                 best_axis = "x"
#                 best_translation = (min_x, min_y, 0)
        
            
#     first_flag = False # not the first item anymore
            
#     return best_degree, best_axis, best_translation, position_bin, first_flag

# def open_a_new_bin_repack(original_object_info_total, object_index, ifv_pool, current_layout, topos_layout, orientation, each_object_info, container_size, container_shape, 
#                    position_bin, radio_list, pieces_order):
    
#     # open_a_new_bin(ifv_pool, current_layout, topos_layout, orientation, each_object_info, container_size, container_shape, 
#     #                 position_bin, radio_list,pieces_order, pieces_orientation_value_pool,pieces_packing_position_pool)
    
#     best_value_local = 9999999
#     current_layout.append([])
#     topos_layout.append(np.zeros(container_size))
#     position_bin += 1
#     radio_list.append([])
#     pieces_order.append([])

#     trace(f"A new bin {position_bin} has been opened") 
#     tem_orien_value = []
#     tem = copy.deepcopy(original_object_info_total[object_index])

#     for each_degree in orientation[0]:
#         trace(f"Try next degree {each_degree}")
#         if each_degree != 0: 
#             # if we rotate it 
#             for each_axis in orientation[1]:
#                 orientation_str = each_axis + '_' + str(each_degree)
#                 trace(f"Try next axis {each_axis}")
#                 # for each orientation of the object       
                
#                 # origin_array = back_to_origin(tem)

#                 tem['array'] = rotate_voxel(tem['array'], each_degree, each_axis)
#                 tem["orientation"] = orientation_str

#                 # rotated_shape = rotate_voxel(each_object_info["array"],each_degree,each_axis)
#                 x,y,z = np.where(tem['array'] == 1) # index of where space is occupied
            
#                 # best_value = len(rotated_shape[2])# the length of z axis
                
#                 value = max(z) # the max height of the item
#                 tem_orien_value.append((orientation_str,value))

#                 if container_shape == "cube":
#                     min_x = 0
#                     min_y = 0
                
#                 elif container_shape == "cylinder":
#                     ifv = ifv_pool.retrieve_ifv(tem, container_size, container_shape)
#                     _x, _y, _z = np.where(ifv == 1) 
                    
#                     min_x = np.min(_x) 
#                     min_y = np.min(_y[_x == min_x])
                    
                
#                 trace(f"The value of this position is {value} current best value is {best_value_local}") 
#                 if value < best_value_local:
#                     best_value_local = value
#                     best_degree = each_degree
#                     best_axis = each_axis
#                     best_translation = (min_x, min_y, 0)
                    
#         else: 
#             # if we don't rotate it, don't need to read axis

#             # tem = copy.deepcopy(each_object_info)

#             # tem = copy.deepcopy(each_object_info)
#             # tem['array'] = back_to_origin(tem)
#             # tem["orientation"] = "x_0"

#             x,y,z = np.where(tem["array"] == 1)                
#             # best_value = len(each_object_info[2])# the length of z axis       
#             value = max(z) # the max height of the item
#             tem_orien_value.append(("x_0",value))

#             if container_shape == "cube":
#                 min_x = 0
#                 min_y = 0
            
#             elif container_shape == "cylinder":
#                 ifv = ifv_pool.retrieve_ifv(tem, container_size, container_shape)
#                 _x, _y, _z = np.where(ifv == 1) 
                
#                 min_x = np.min(_x) 
#                 min_y = np.min(_y[_x == min_x])
#                 # best_translation = (min_x, min_y, 0)
                
#             trace(f"The value of this position is {value} current best value is {best_value_local}") 
#             if value < best_value_local:
#                 best_value_local = value
#                 best_degree = 0
#                 best_axis = "x"
#                 best_translation = (min_x, min_y, 0)
        
            
#     first_flag = False # not the first item anymore
            
#     return best_degree, best_axis, best_translation, position_bin, first_flag


def get_volume(_voxel_data):

    # Essentially counting the number of voxels  
    
    x,y,z = np.where(_voxel_data == 1)
    
    return len(z)

def back_to_origin(object_info):

    axis, degree = object_info["orientation"].split("_")
    # print("Updated orientation", object_info["orientation"], "rotate ",360-int(degree), "align with ", axis)
    if int(degree) == 0:
        # it is origin itself
        return object_info["array"]
    
    # origin_array = rotate_voxel(object_info["array"], 360-int(degree), axis)
    return rotate_voxel(object_info["array"], 360-int(degree), axis)
    # this is normalised 


# @profile
def packing_3D_voxel_lookback(original_object_info_total, nfv_pool, ifv_pool, orientation, container_size, 
                              container_shape,
                              packing_alg, _evaluation,
                              SCH_nesting_strategy, density, axis, 
                              _type, _accessible_check, _encourage_dbl,
                              _select_range, random_CA, random_CA_threshold, flag_NFV_POOL, _TRACE):
    
    # packing_3D_voxel_lookback(object_info_total, nfv_pool, ifv_pool, orientations, container_size,
    #                             container_shape, rho, max_radio, 
    #                             packing_alg, orien_evaluation,
    #                             SCH_nesting_strategy, density = 5, axis = 'z', 
    #                             _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
    #                             _select_range = selection_range, random_CA= random_CA, random_CA_threshold = random_CA_threshold, _TRACE = False) 

    # the constructive algorithm

    """
    A packing alg considering checking previous bins 
    (first fit principle)

    Args:
        polygon_total (3D array): voxelised data
        orientation (2D tuple): orientation of object (degree_tuple, rotate_axis_tuple)

    Returns:
        4D array: 3d matrix of 0-1 for each container 
    
        
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

    """    
    

    global TRACE
    TRACE = _TRACE

    current_layout = [] # a 4d structure to store objects' state
    topos_layout = np.zeros(container_size)

    pieces_order = [] # to track the packing order
    # radio_list = [] # to track the radioactivity of pieces in the bin
        
    first_flag = True 
    num_piece = 0
    
    for each_object_info in original_object_info_total:
        max_height_possible = 0

        trace(f"Packing begins for object {num_piece}.")

        best_degree = 0
        best_axis = "x"

        packing_posi_info_current_bin = [] # to track the feasible packing position in the current bin

        if first_flag == True:
            trace("It is the first object in the last bin")

            # if it is the first item, pack as low as possible
            for each_degree in orientation[0]:
                trace(f"Try next degree {each_degree}")
                
                if each_degree == 0: 
                    orientations_list = ["x"]

                else:
                    orientations_list = orientation[1]

                for each_axis in orientations_list:
                    trace(f"Try next axis {each_axis}")
                    # orientation_str = each_axis + "_" + str(each_degree)
                    # for each orientation of the object

                    try_info_tem = copy.deepcopy(each_object_info)

                    if each_degree == 0:
                        pass

                    else:
                        try_info_tem["array"] = rotate_voxel(try_info_tem["array"], each_degree, each_axis)
                        try_info_tem["orientation"] = each_axis + "_" + str(each_degree)

                    # print("shape of rotated shape", np.shape(rotated_shape))

                    x,y,z = np.where(try_info_tem["array"] == 1) # index of where space is occupied

                    
                    # best_value = len(rotated_shape[2])# the length of z axis
                    
                    # To pack as low as possible
                    value = max(z) # the max height of the item

                    if container_shape == "cube":
                        min_x = 0
                        min_y = 0
                        
                    elif container_shape == "cylinder":
                        # print("treat it as cylinder")
                        # ifv needs to change 
                        max_height_possible = value
                        ifv = get_ifv_cube_strip_packing(each_object_info,container_size,max_height_possible)
                        # visualize_single_object(ifv,container_size)
                        _x, _y, _z = np.where(ifv == 1) 
                        
                        min_x = np.min(_x) 
                        min_y = np.min(_y[_x == min_x])                                       
                        
                    # trace(f"The value of this position is {value} current best value is {best_value}") 

                    packing_posi_info_current_bin.append((value,each_degree,each_axis,(min_x,min_y,0)))

                    # if value < best_value:
                    #     best_value = value
                    #     best_degree = each_degree
                    #     best_axis = each_axis
                    #     best_translation = (min_x, min_y, 0)                   
                    
            first_flag = False # not the first item anymore
        
        else: 
            # if it is not the first item in this bin
            # print("It is not the first object")
   
            for each_degree in orientation[0]:
                trace(f"Try next degree {each_degree}")
                
                if each_degree == 0: 
                    orientations_list = ["x"]

                else:
                    orientations_list = orientation[1]
                    
                for each_axis in orientation[1]:
                    trace(f"Try next axis {each_axis}")
                    # for each orientation of the object

                    # orientation_str = each_axis + "_" + str(each_degree)
                    # rotated_shape = rotate_voxel(each_object_info["array"],each_degree,each_axis)

                    try_info_tem = copy.deepcopy(each_object_info)

                    if each_degree == 0:
                        pass

                    else:
                        try_info_tem["array"] = rotate_voxel(try_info_tem["array"], each_degree, each_axis)
                        try_info_tem["orientation"] = each_axis + "_" + str(each_degree)

                    aabb_x,aabb_y,aabb_z = aabb_rotate(try_info_tem["aabb"],try_info_tem["orientation"])
                    x,y,z = np.where(topos_layout == 1)
                    # visualize_single_object(topos_layout,np.shape(topos_layout))
                    # print(np.shape(topos_layout),z,max(z))
                    # exit(-1)
                    max_height_possible = min(max(z) + aabb_z,container_size[2])

                    packing_position = SC_heuristic(nfv_pool,ifv_pool, try_info_tem, current_layout, topos_layout, max_height_possible, 
                                                    container_size, SCH_nesting_strategy, density, axis, container_shape,
                                                    _type, _accessible_check, _encourage_dbl, _select_range, flag_NFV_POOL, _TRACE)
                    # print(packing_position)
                    # def SC_heuristic(nfv_pool,ifv_pool, ongoing_object_info, current_layout, topos_layout, position_bin, 
                    #                     bin_size, nesting_strategy, density, axis, container_shape, 
                    #                     _type, _accessible_check, _encourage_dbl, _select_range, _TRACE): 

                        
                    # time_check2 = time.time() 

                    # trace(f"packing position is found, cost {time_check2-time_check1} s")

                    if isinstance(packing_position,tuple): 

                        # This is for evaluation 
                        current_layout_test = copy.deepcopy(current_layout)   
                        topos_layout_test =  copy.deepcopy(topos_layout)                                     
                        # translated_test = translate_voxel(try_info_tem["array"],packing_position)
                        insert_add(topos_layout_test,try_info_tem["array"],packing_position)

                        value = evaluation(topos_layout_test, current_layout_test, try_info_tem, packing_position, (container_size[0],container_size[1],max_height_possible),  _evaluation)

                        # trace(f"The value of this position is {value} current best value is {best_value}") 
                        
                        packing_posi_info_current_bin.append((value,each_degree,each_axis,packing_position))

                        # if value < best_value:
                            
                        #     best_value = value
                        #     best_degree = each_degree
                        #     best_axis = each_axis
                        #     best_translation = packing_position
                            
                    
                    else:
                        trace("Can't find solution in this orientation")
                        pass
                    
                    # =================================================================================================

            # this is the updates for the last bin
            trace(f"The piece {num_piece} has been packed") 

        if not packing_posi_info_current_bin:
            print("!!!!!!Can't Pack this piece!!!!!")
            pass    

        else:
            packing_posi_info_current_bin.sort(key = lambda x:x[0]) # The smaller, the better. 
            packing_posi_candidates = packing_posi_info_current_bin[:random_CA_threshold]

            position_info = []
            
            if random_CA: 
                _ = np.random.randint(0,min(random_CA_threshold,len(packing_posi_candidates)))
                position_info = packing_posi_info_current_bin[_]

            else:
                position_info = packing_posi_info_current_bin[0]

            best_degree = position_info[1]
            best_axis = position_info[2]
            best_translation = position_info[3]

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

            # update object info
            each_object_info["array"] = rotate_voxel(each_object_info["array"],best_degree,best_axis)
            # each_object_info["array"] = translate_voxel(each_object_info["array"],best_translation)
            each_object_info["translation"] = best_translation
            each_object_info["orientation"] = best_axis + "_" + str(best_degree)

            current_layout.append(each_object_info)
            # print(best_translation)
            #########################################
            insert_add(topos_layout,each_object_info["array"],best_translation)
            # topos_layout += each_object_info["array"] 
            #########################################

            pieces_order.append(num_piece) # for tracking the packing order

        # add a partial visualised here
        num_piece += 1
              
    return current_layout, topos_layout, pieces_order


def repacking_new_ILS(original_object_info_total, selected_info, nfv_pool, ifv_pool, best_orientations_list_no_bin, old_data_pool, global_best_iter, orientation, container_size, 
                        container_shape, rho, max_radio, 
                        packing_alg, _evaluation,
                        SCH_nesting_strategy, density, axis, 
                        _type, _accessible_check, _encourage_dbl,
                        _select_range,flag_NFV_POOL, _TRACE):
    

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

    global TRACE
    TRACE = _TRACE
    
    # if _neighbor_type == "orientation_only":
    # Get the new input for the packing algorithm.
    # ============================================================================================
    new_selected_index = selected_info[0]
    # new_selected_index = 0
    
    new_selected_orientation = selected_info[1]
    # new_selected_packing_position = selected_info[2]
    
    # print(old_data_pool)

    # This is for loading the current best solution as the input of the next iteration.

    filtered_data = old_data_pool[old_data_pool["iteration"] == global_best_iter]
    best_current_layout = filtered_data["bin_real_layout"].iloc[0] 
    # real_layout is a list filled with object_info

    # best_radio_layout = filtered_data["radio_layout"].iloc[0]

    # initialisation
    input_packed_layout = []
    input_packed_topos_layout = np.zeros(container_size)
    input_packed_order = []
    
    # input_packed_best_orien = []
    # input_packed_orien_value_pool = []
    # input_packed_position = []
    # input_packed_position_pool = []
    # input_packed_radio_list = [] # this is to track the radio layout
    
    
    # pieces_order_matrix = filtered_data["pieces_order"].iloc[0]
    # #print(pieces_order_matrix)
    
    first_flag = False
    # selected_piece_index = find_element_index(pieces_order_matrix, new_selected_index)
    
    # It is not the first piece, if you select the first piece in one bin, it is the last piece of the previous bin 
    # if selected_piece_index[1] == 0:   
    # # to decide if the selected item is the first item in the bin
    #     first_flag = True
        
    # initialise all input data for the repack
    for each_piece in range(new_selected_index):

        best_current_info = best_current_layout[each_piece]      
        input_packed_layout.append(best_current_info)
        insert_add(input_packed_topos_layout,best_current_info["array"],best_current_info["translation"])
        input_packed_order.append(each_piece)

    # visualize_single_object(input_packed_topos_layout,container_size)
     # ============================================================================================

    current_layout = input_packed_layout # INFO!
    topos_layout = input_packed_topos_layout
    pieces_order = input_packed_order # to track the packing order of pieces

    # radio_list = input_packed_radio_list # to track the radioactivity of pieces in the bin
    
    # if container_shape == "cube":
    #     container_volume = container_size[0]*container_size[1]*container_size[2]
        
    # elif container_shape == "cylinder": 
    #     container_volume = (math.pi * (container_size[0]/2)**2) * container_size[2]

    num_piece = new_selected_index
    is_selected_piece = True
    
    if new_selected_index == 0: 
        first_flag = True

    # object_index = new_selected_index
    for each_object_info in original_object_info_total[new_selected_index:]:
        max_height_possible = 0
        
        # this object_info has updated by CA

        trace(f"Packing begins for object {num_piece}.")
        
        best_value = 999999999
        best_degree = 0
        best_axis = "x"

        if is_selected_piece == True:
            # should only try the selected orientation for the selected piece. 
            trace("It is the selected object!")
            selected_orientation = new_selected_orientation
            is_selected_piece = False # only once

        else: 
            trace("It is NOT the selected object!")
            selected_orientation = best_orientations_list_no_bin[num_piece]

        a,b = selected_orientation.split("_")
        degrees = [int(b)] # keep the old orientation
        axiss = [a]

        # orientation = [degrees,axiss]
                
        if first_flag == True:
            trace("It is the first object in this bin")

            # try_info_tem = copy.deepcopy(each_object_info)

            # if it is the first item, pack as low as possible
            for each_degree in degrees:
                trace(f"Try next degree {each_degree}")

                if each_degree == 0: 
                    axiss = ["x"]
                    
                for each_axis in axiss:
                    trace(f"Try next axis {each_axis}")
                    # for each orientation of the object

                    # try_info_tem = copy.deepcopy(each_object_info)

                    # origin_array = back_to_origin(try_info_tem)
                    try_info_tem = copy.deepcopy(each_object_info)

                    if each_degree == 0:
                        pass
                    
                    else:
                        try_info_tem["array"] = rotate_voxel(try_info_tem["array"], each_degree, each_axis)
                        try_info_tem["orientation"] = each_axis + "_" + str(each_degree)

                    x,y,z = np.where(try_info_tem["array"] == 1) # index of where space is occupied
                    # orientation_str = each_axis + "_" + str(each_degree)
                    # best_value = len(rotated_shape[2])# the length of z axis
                    
                    # To pack as low as possible
                    value = max(z) # the max height of the item
                    
                    if container_shape == "cube":
                        min_x = 0
                        min_y = 0
                        
                    elif container_shape == "cylinder":
                        # print("treat it as cylinder")

                        max_height_possible = value
                        # ifv = ifv_pool.retrieve_ifv(try_info_tem, container_size, container_shape)

                        ifv = get_ifv_cube_strip_packing(each_object_info,container_size,max_height_possible)

                        # visualize_single_object(ifv,container_size)
                        _x, _y, _z = np.where(ifv == 1) 
                        
                        min_x = np.min(_x) 
                        min_y = np.min(_y[_x == min_x])
                        
                        
                    trace(f"The value of this position is {value} current best value is {best_value}") 

                    if value < best_value:
                        best_value = value
                        best_degree = each_degree
                        best_axis = each_axis
                        best_translation = (min_x, min_y, 0)

            trace(f"The piece {num_piece} has been packed")  
            first_flag = False
                            
        
        else: 
            # if it is not the first item in this bin
            # print("It is not the first object")
            
            # try_info_tem = copy.deepcopy(each_object_info)

            for each_degree in degrees:
                trace(f"Try next degree {each_degree}")

                if each_degree == 0: 
                    axiss = ["x"]

                for each_axis in axiss:
                    trace(f"Try next axis {each_axis}")
                    # for each orientation of the object
                    # orientation_str = each_axis + "_" + str(each_degree)
                    # rotated_shape = rotate_voxel(each_object_info["array"],each_degree,each_axis)
                    # x,y,z = np.where(rotated_shape == 1) # index of where space is occupied
                    
                    # =================================================================================================
                    # try_info_tem = copy.deepcopy(each_object_info)
                    # origin_array = back_to_origin(try_info_tem)
                    try_info_tem = copy.deepcopy(each_object_info)

                    if each_degree == 0:
                        pass
                    
                    else:
                        try_info_tem["array"] = rotate_voxel(try_info_tem["array"],each_degree,each_axis)
                        try_info_tem["orientation"] = each_axis + "_" + str(each_degree)
                    
                    aabb_x,aabb_y,aabb_z = aabb_rotate(try_info_tem["aabb"],try_info_tem["orientation"])
                    x,y,z = np.where(topos_layout == 1)

                    # time_check1 = time.time() 
                    max_height_possible = min(max(z) + aabb_z,container_size[2])

                    packing_position = SC_heuristic(nfv_pool, ifv_pool, try_info_tem, current_layout, topos_layout, max_height_possible, 
                                                    container_size, SCH_nesting_strategy, density, axis, container_shape, 
                                                    _type, _accessible_check, _encourage_dbl, _select_range,flag_NFV_POOL, _TRACE)
                    


                    # time_check2 = time.time() 
                    # trace(f"packing position is found, cost {time_check2-time_check1} s")
                    if isinstance(packing_position,tuple): 

                        # This is for evaluation 
                        current_layout_test = copy.deepcopy(current_layout)  
                        topos_layout_test =  copy.deepcopy(topos_layout) 

                        insert_add(topos_layout_test,try_info_tem["array"],packing_position)

                        value = evaluation(topos_layout_test, current_layout_test, try_info_tem, packing_position, (container_size[0],container_size[1],max_height_possible),  _evaluation)

                        trace(f"The value of this position is {value} current best value is {best_value}") 
                        
                        if value < best_value:
                            
                            best_value = value
                            best_degree = each_degree
                            best_axis = each_axis
                            best_translation = packing_position
                                
                        # else: 
                        #     trace(f"The packing position is NOT accessible vertically! Try next orientation!")
                        #     pass
                    
                    else:
                        trace("Can't find solution in this orientation")
                        pass
                    
                    # =================================================================================================
            trace(f"The piece {num_piece} has been packed")     
                
        if best_value > 99999999:
            # No feasible solution is found, check the next bin
            trace("!!!can't pack this object!!!")
            return False, False, False
            # pass

        # trace(f"The piece {num_piece} has been packed in, best oprientation is {best_axis}_{str(best_degree)}")  
        # info updating     
            
        # origin_array = back_to_origin(each_object_info)
        packed_tem = copy.deepcopy(each_object_info)

        packed_tem["array"] = rotate_voxel(packed_tem["array"], best_degree, best_axis)
        # tem = rotate_voxel(each_object_info["array"],best_degree,best_axis)
        packed_tem["translation"] = best_translation
        packed_tem["orientation"] = best_axis + "_" + str(best_degree)

        current_layout.append(packed_tem)
        insert_add(topos_layout,packed_tem["array"],best_translation)

        pieces_order.append(num_piece)

        # add a partial visualised here
        num_piece += 1

        # is_selected_piece = False
        # object_index += 1
    
    return current_layout, topos_layout, pieces_order


def kick_repacking(original_object_info_total, nfv_pool, ifv_pool, orientation, best_orientations_list_no_bin, container_size, 
                    container_shape, rho, max_radio, 
                    packing_alg, _evaluation,
                    SCH_nesting_strategy, density, axis, 
                    _type, _accessible_check, _encourage_dbl,
                    _select_range, flag_NFV_POOL, _TRACE):
    
    # the constructive algorithm

    """
    A packing alg considering checking previous bins 
    (first fit principle)

    Args:
        polygon_total (3D array): voxelised data
        orientation (2D tuple): orientation of object (degree_tuple, rotate_axis_tuple)

    Returns:
        4D array: 3d matrix of 0-1 for each container 
    
    """    
    
    global TRACE
    TRACE = _TRACE

    
    current_layout = [] # a 4d structure
    topos_layout = np.zeros(container_size)
    pieces_order = [] # to track the packing order of pieces
    # radio_list = [[]] # to track the radioactivity of pieces in the bin
    
    # if container_shape == "cube": 
    #     container_volume = container_size[0]*container_size[1]*container_size[2]
        
    # elif container_shape == "cylinder": 
    #     container_volume = (math.pi * (container_size[0]/2)**2) * container_size[2]
        
    first_flag = True
    
    # position_bin = 0
    num_piece = 0
    
    for each_object_info in original_object_info_total:
        max_height_possible = 0 
        trace(f"Packing begins for object {num_piece}.")
        
        best_value = 99999999
        best_degree = 0
        best_axis = "x"

        a,b =  best_orientations_list_no_bin[num_piece].split("_")
        degrees = [int(b)]
        axiss = [a]
        orientation = [degrees,axiss]

        if first_flag == True:
            trace("It is the first object in this bin")

            # try_info_tem = copy.deepcopy(each_object_info)
            # if it is the first item, pack as low as possible
            for each_degree in orientation[0]:
                trace(f"Try next degree {each_degree}")
            
                    
                for each_axis in orientation[1]:
                    trace(f"Try next axis {each_axis}")
                    # for each orientation of the object

                    
                    # origin_array = back_to_origin(try_info_tem)
                    try_info_tem = copy.deepcopy(each_object_info)
                    if each_degree == 0: 
                        pass 
                    
                    else:
                        try_info_tem["array"] = rotate_voxel(try_info_tem["array"],each_degree,each_axis)
                        try_info_tem["orientation"] = each_axis + "_" + str(each_degree)

                    x,y,z = np.where(try_info_tem["array"] == 1) # index of where space is occupied
                    # orientation_str = each_axis + "_" + str(each_degree)
                    # best_value = len(rotated_shape[2])# the length of z axis
                    
                    # To pack as low as possible
                    value = max(z) # the max height of the item

                    
                    if container_shape == "cube":
                        min_x = 0
                        min_y = 0
                        
                    elif container_shape == "cylinder":
                        # print("treat it as cylinder")
                        max_height_possible = value
                        ifv = get_ifv_cube_strip_packing(each_object_info,container_size,max_height_possible)
                        # visualize_single_object(ifv,container_size)
                        _x, _y, _z = np.where(ifv == 1) 
                        
                        min_x = np.min(_x) 
                        min_y = np.min(_y[_x == min_x])
                        
                        
                    trace(f"The value of this position is {value} current best value is {best_value}") 

                    if value < best_value:
                        best_value = value
                        best_degree = each_degree
                        best_axis = each_axis
                        best_translation = (min_x, min_y, 0)

            first_flag = False

        else: 
            # if it is not the first item in this bin
            # print("It is not the first object")
            
            # try_info_tem = copy.deepcopy(each_object_info)
            for each_degree in orientation[0]:
                trace(f"Try next degree {each_degree}")
                    
                for each_axis in orientation[1]:
                    trace(f"Try next axis {each_axis}")
                    # for each orientation of the object
                    # orientation_str = each_axis + "_" + str(each_degree)
                    # rotated_shape = rotate_voxel(each_object_info["array"],each_degree,each_axis)
                    # x,y,z = np.where(rotated_shape == 1) # index of where space is occupied
                    
                    # =================================================================================================
                    
                    # origin_array = back_to_origin(try_info_tem)
                    try_info_tem = copy.deepcopy(each_object_info)
                    
                    if each_degree == 0:
                        pass
                    
                    else:
                        try_info_tem["array"] = rotate_voxel(try_info_tem["array"],each_degree,each_axis)
                        try_info_tem["orientation"] = each_axis + "_" + str(each_degree)

                    # time_check1 = time.time() 
                    aabb_x,aabb_y,aabb_z = aabb_rotate(try_info_tem["aabb"],try_info_tem["orientation"])

                    x,y,z = np.where(topos_layout == 1)       
                    max_height_possible = min(max(z) + aabb_z,container_size[2])

                    packing_position = SC_heuristic(nfv_pool,ifv_pool, try_info_tem, current_layout, topos_layout, max_height_possible, 
                                                    container_size, SCH_nesting_strategy, density, axis,container_shape,
                                                    _type, _accessible_check, _encourage_dbl, _select_range,flag_NFV_POOL, _TRACE)
                        
                    # time_check2 = time.time() 
                    # trace(f"packing position is found, cost {time_check2-time_check1} s")

                    if isinstance(packing_position,tuple): 

                        # This is for evaluation 
                        current_layout_test = copy.deepcopy(current_layout)   
                        topos_layout_test =  copy.deepcopy(topos_layout)                                     
                        # translated_test = translate_voxel(try_info_tem["array"],packing_position)
                        insert_add(topos_layout_test,try_info_tem["array"],packing_position)

                        value = evaluation(topos_layout_test, current_layout_test, try_info_tem, packing_position, (container_size[0],container_size[1],max_height_possible),  _evaluation)

                        trace(f"The value of this position is {value} current best value is {best_value}") 
                        
                        if value < best_value:
                            
                            best_value = value
                            best_degree = each_degree
                            best_axis = each_axis
                            best_translation = packing_position
                                
                        # else: 
                        #     trace(f"The packing position is NOT accessible vertically! Try next orientation!")
                        #     pass
                    
                    else:
                        trace("Can't find solution in this orientation")
                        pass
                        
                        # =================================================================================================

                    # =================================================================================================
    
        if best_value > 9999999:
            # No feasible solution is found, check the next bin
            # temporarily place in a bin at the bad & infeasible position and search by LS

            trace("!!! Can't pack this object !!!")
            best_degree = 0
            best_axis = "x"
            temaabb = each_object_info["aabb"]
            best_translation = (int(container_size[0])-int(temaabb[0]),int(container_size[1])-int(temaabb[1]),int(container_size[2])-int(temaabb[2]))

            # current_layout = False
            # return False, False, False


        # tem = rotate_voxel(each_object_info["array"],best_degree,best_axis)
        # origin_array = back_to_origin(each_object_info)
        
        packed_tem = copy.deepcopy(each_object_info)

        packed_tem["array"] = rotate_voxel(packed_tem["array"], best_degree, best_axis)
        packed_tem["translation"] = best_translation
        packed_tem["orientation"] = best_axis + "_" + str(best_degree)


        current_layout.append(packed_tem)
        insert_add(topos_layout,packed_tem["array"],best_translation)
        # print(best_translation)
        # add a partial visualised here
        num_piece += 1
              
    return current_layout, topos_layout, pieces_order



def visualize_single_object(voxel_data,container_size):
    
    size = container_size
    fig = go.Figure()

    x,y,z = np.where(voxel_data == 1)
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                    mode='markers',
                                    marker=dict(
                                    size=10,              # set the size of voxel
                                    symbol='square',
                                    color=f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})",        # set the color
                                    opacity=0.7          # transparence
                                  )))
    
    max_size = max(container_size)
    aspect_ratio = dict(
        x=container_size[0] / max_size,
        y=container_size[1] / max_size,
        z=container_size[2] / max_size
    )
    fig.update_layout(
        scene=dict(aspectmode='manual',
                   aspectratio=aspect_ratio,
            xaxis=dict(nticks=10, range=[0, size[0]]),
            yaxis=dict(nticks=10, range=[0, size[1]]),
            zaxis=dict(nticks=10, range=[0, size[2]])
        ),
        title="3D Voxel Visualization",
        width=1000,   # size of picture
        height=1000,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    
    fig.show()
    

def voxel_to_mesh3d(x, y, z):
    "transfer voxel to mesh for visualisation"
    vertices = []
    faces = []
    for xi, yi, zi in zip(x, y, z):
        # 8 vertices of cube
        v = [
            [xi, yi, zi],
            [xi + 1, yi, zi],
            [xi + 1, yi + 1, zi],
            [xi, yi + 1, zi],
            [xi, yi, zi + 1],
            [xi + 1, yi, zi + 1],
            [xi + 1, yi + 1, zi + 1],
            [xi, yi + 1, zi + 1],
        ]
        idx = len(vertices)
        vertices.extend(v)
        # 12 triangle faces of cube
        cube_faces = [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [0, 3, 7], [0, 7, 4],
        ]
        for f in cube_faces:
            faces.append([idx + i for i in f])
    verts = np.array(vertices)
    faces = np.array(faces)
    return verts[:, 0], verts[:, 1], verts[:, 2], faces[:, 0], faces[:, 1], faces[:, 2]

def get_n_colors(n, colorscale='Turbo'):

    return sample_colorscale(colorscale, [i / max(1, n - 1) for i in range(n)])

# def visualize_voxel_model(layout_object_info, pieces_order, container_size, container_shape):
#     # size = np.shape(voxel_data[0])
    
#     # Use Mayavi to visualize 3D voxel data
#     # ==============================================================================================
#     # mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))  # Create a figure with a white background
    
#     # # Render voxel data using mlab.contour3d
#     # # contour3d will automatically render surfaces of non-zero values

#     # # mlab.contour3d(voxel_data, contours=[0.5], opacity=0.5, color=(0.5, 1, 0.5))  # Set color and transparency, this is just for contour
#     # for each_shape in voxel_data:
#     #     x,y,z = np.where(each_shape == (1 or True)) # read the index of voxel when data == 1 or True.
#     #     mlab.points3d(x, y, z, scale_factor=1, mode="cube", color=(np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)))

    
#     # # edges = mlab.pipeline.extract_edges(voxels)
#     # # mlab.pipeline.surface(edges, color=(0, 0, 0))
    
#     # # Show the visualization
#     # mlab.show()
#     # ===============================================================================================  
     
#     # size = container_size
#     # fig = go.Figure()

#     # x,y,z = np.where(voxel_data == 1)
    
#     # fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
#     #                                 mode='markers',
#     #                                 marker=dict(
#     #                                 size=10,              # set the size of voxel
#     #                                 symbol='square',
#     #                                 color=f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})",        # set the color
#     #                                 opacity=0.7          # transparence
#     #                                 )))

#     # fig.update_layout(
#     #     scene=dict(aspectmode='cube',
#     #         xaxis=dict(nticks=10, range=[0, size[0]]),
#     #         yaxis=dict(nticks=10, range=[0, size[1]]),
#     #         zaxis=dict(nticks=10, range=[0, size[2]])
#     #     ),
#     #     title="3D Voxel Visualization",
#     #     width=1000,   # size of picture
#     #     height=1000,
#     #     margin=dict(r=20, l=10, b=10, t=10)
#     # )
    
#     # fig.show()
    
#     # use plotly
#     # ===============================================================================================  
#     # scene = np.zeros(container_size)

#     num_containers = 1
#     n_pieces = len(layout_object_info)
#     color_list = get_n_colors(n_pieces, colorscale='Turbo')
#     legend_added_set = set()
    
#     max_cols = 3
#     rows = math.ceil(num_containers / max_cols)
#     cols = min(num_containers, max_cols)
    
#     fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'scatter3d'}] * cols for _ in range(rows)])
    
#     # print(np.shape(voxel_data))
#     which_piece = 0

#     for each_piece in layout_object_info:
#         scene = np.zeros(container_size)
#         # color = f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
#         tem = insert_add(scene,each_piece["array"],each_piece["translation"])      
#         x,y,z = np.where(tem == 1)
            
#             # Use Scatter3D
#             # fig.add_trace(
#             #     go.Scatter3d(x=x, y=y, z=z,
#             #                 mode='markers',
#             #                 marker=dict(
#             #                     size=5,              # set the size of voxel
#             #                     symbol='square',
#             #                     color=f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})",        # set the color
#             #                     opacity=0.7          # transparence
#             #             )
#             #         ),
#             #         row = row, col = col
#             #     )
            

            
#             # transfer to mesh
#         xv, yv, zv, _i, _j, _k = voxel_to_mesh3d(x, y, z)
        
#         piece_id = pieces_order[which_piece]
#         color = color_list[piece_id]
        
#         show_legend = piece_id not in legend_added_set
        
#         if show_legend:
#             legend_added_set.add(piece_id)
        
#         fig.add_trace(
#             go.Mesh3d(
#                 x=xv, y=yv, z=zv,
#                 i=_i, j=_j, k=_k,
#                 opacity=0.6,
#                 color=color,
#                 showlegend=show_legend,
#                 lighting=dict(ambient=0.2, diffuse=0.8, roughness=0.5, specular=0.6),
#                 lightposition=dict(x=100, y=200, z=0)
#             ),
#             row=1, col=1
#         )
#         which_piece += 1


#         # if x.size > 0:
#         #     cx = np.mean(x)
#         #     cy = np.mean(y)
#         #     cz = np.mean(z)
#         #     piece_id = pieces_order[i][which_piece_this_bin]  
#         #     fig.add_trace(
#         #         go.Scatter3d(
#         #             x=[cx], y=[cy], z=[cz],
#         #             mode="text",
#         #             text=[f"{piece_id+1}"], 
#         #             textposition="middle center",
#         #                 textfont=dict(
#         #                             size=18,       
#         #                             color="black",
#         #                             family="Arial Black" 
#         #                         ),
#         #             showlegend=False
#         #         ),
#         #         row=row, col=col
#         #     )
        
#         # which_piece_this_bin += 1
        
            
            
#         # if container_shape == "cylinder":
#         #     # add clinder bondary of container
            
#         #     cylinder_radius = container_size[0]/2
#         #     cylinder_height = container_size[2]
            
#         #     center_x = container_size[0] / 2
#         #     center_y = container_size[0] / 2
            
#         #     theta = np.linspace(0, 2 * np.pi, 100)  
#         #     x_side = np.concatenate([cylinder_radius * np.cos(theta) + center_x, cylinder_radius * np.cos(theta) + center_x])
#         #     y_side = np.concatenate([cylinder_radius * np.sin(theta) + center_y, cylinder_radius * np.sin(theta) + center_y])
#         #     z_side = np.concatenate([np.zeros_like(theta), np.ones_like(theta) * cylinder_height])

#         #     faces = []
#         #     num_points = len(theta)
#         #     for j in range(num_points - 1):
#         #         faces.extend([
#         #             [j, j + 1, j + num_points], 
#         #             [j + 1, j + num_points + 1, j + num_points]
#         #         ])
            
#         #     faces = np.array(faces).T  # 转置以符合 Mesh3d 的格式
            
#         #     fig.add_trace(
#         #         go.Mesh3d(
#         #             x=x_side, y=y_side, z=z_side,
#         #             i=faces[0], j=faces[1], k=faces[2],  # 三角面索引
#         #             color="rgba(0, 0, 255, 0.3)",  # 半透明蓝色
#         #             opacity=0.3
#         #         ),
#         #         row=row, col=col
#         #     )
    
#     for i in range(1, rows + 1):
#         for j in range(1, cols + 1):
#             fig.update_scenes(
#                 dict(
#                     aspectmode='cube',
#                     xaxis=dict(nticks=10, range=[0, container_size[0]]),
#                     yaxis=dict(nticks=10, range=[0, container_size[1]]),
#                     zaxis=dict(nticks=10, range=[0, container_size[2]])
#                 ), row=i, col=j
#             )
     
#     fig.update_layout(
#         scene=dict(
#             camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
#             aspectmode='cube'
#         ),
#         margin=dict(l=0, r=0, b=0, t=40),
#         title_text="3D Packing Results",
#         )   
#     fig.show()


def _box_edges(size):
    # 生成容器的12条边，用线段画出来
    X, Y, Z = size
    corners = [
        (0,0,0),(X,0,0),(0,Y,0),(X,Y,0),
        (0,0,Z),(X,0,Z),(0,Y,Z),(X,Y,Z)
    ]
    idx = {
        "000":0,"X00":1,"0Y0":2,"XY0":3,"00Z":4,"X0Z":5,"0YZ":6,"XYZ":7
    }
    c = idx
    edges = [
        (c["000"], c["X00"]), (c["000"], c["0Y0"]), (c["000"], c["00Z"]),
        (c["X00"], c["XY0"]), (c["X00"], c["X0Z"]),
        (c["0Y0"], c["XY0"]), (c["0Y0"], c["0YZ"]),
        (c["00Z"], c["X0Z"]), (c["00Z"], c["0YZ"]),
        (c["XY0"], c["XYZ"]), (c["X0Z"], c["XYZ"]), (c["0YZ"], c["XYZ"])
    ]
    lines = []
    for a,b in edges:
        xa,ya,za = corners[a]
        xb,yb,zb = corners[b]
        lines.append(( [xa,xb,None], [ya,yb,None], [za,zb,None] ))
    return lines
    
def visualize_voxel_model(
        layout_object_info, 
        pieces_order, 
        container_size, 
        container_shape=None,
        max_voxels=150_000,   # 单个 piece 允许的最大体素数
        downsample_max_factor=5,
        surface_only=True,    # 是否只渲染表面
        merge_mesh=False      # 是否把所有 piece 合并成一个 Mesh
    ):
    """
    layout_object_info: list[{"array": np.ndarray, "translation": (tx,ty,tz)}]
    pieces_order: list[int]
    container_size: (X, Y, Z)
    """

    assert len(layout_object_info) == len(pieces_order), "layout 和 pieces_order 长度不一致!"

    # -------- 自动稳健处理 n_pieces ----------
    unique_ids = sorted(set(pieces_order))
    id2idx = {pid: i for i, pid in enumerate(unique_ids)}
    color_list = get_n_colors(len(unique_ids), colorscale='Turbo')

    fig = go.Figure()
    legend_added = set()

    # =============== 合并 Mesh 数据结构（如果启用） ==================
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    all_colors = []
    vert_offset = 0

    # ================================================================
    for which_piece, each_piece in enumerate(layout_object_info):

        # 创建 container 场景
        scene = np.zeros(container_size, dtype=np.uint8)
        tem = insert_add(scene, each_piece["array"], each_piece["translation"])

        # ---------- 自动降采样 ----------
        voxel_count = np.sum(tem == 1)
        factor = 1
        while voxel_count > max_voxels and factor < downsample_max_factor:
            factor += 1
            tem = tem[::factor, ::factor, ::factor]
            voxel_count = np.sum(tem == 1)

        # ---------- 表面体素抽取 ----------
        if surface_only:
            from scipy.ndimage import binary_erosion
            tem = tem ^ binary_erosion(tem)

        x, y, z = np.where(tem == 1)
        if len(x) == 0:
            continue

        xv, yv, zv, _i, _j, _k = voxel_to_mesh3d(x, y, z)

        piece_id = pieces_order[which_piece]
        color = color_list[id2idx[piece_id]]

        # ==================== 合并 Mesh 模式 ====================
        if merge_mesh:
            all_x.append(xv)
            all_y.append(yv)
            all_z.append(zv)

            all_i.append(_i + vert_offset)
            all_j.append(_j + vert_offset)
            all_k.append(_k + vert_offset)

            all_colors.append(np.full_like(xv, id2idx[piece_id], dtype=float))

            vert_offset += len(xv)
            continue

        # ==================== 单 piece 渲染 ====================
        show_legend = piece_id not in legend_added
        if show_legend:
            legend_added.add(piece_id)

        fig.add_trace(go.Mesh3d(
            x=xv, y=yv, z=zv,
            i=_i, j=_j, k=_k,
            opacity=0.7,
            color=color,
            name=f"Piece {piece_id}",
            showlegend=show_legend,
            lighting=dict(
                ambient=0.5,
                diffuse=0.6,
                specular=0.2,
                roughness=0.8
            )
        ))

    # =============== 合并 Mesh 图层输出 ==================
    if merge_mesh and len(all_x) > 0:
        X = np.concatenate(all_x)
        Y = np.concatenate(all_y)
        Z = np.concatenate(all_z)
        I = np.concatenate(all_i)
        J = np.concatenate(all_j)
        K = np.concatenate(all_k)
        C = np.concatenate(all_colors)

        fig.add_trace(go.Mesh3d(
            x=X, y=Y, z=Z,
            i=I, j=J, k=K,
            intensity=C,
            colorscale=get_n_colors(len(unique_ids), colorscale="Turbo"),
            opacity=0.7,
            showscale=False,
            name="All Pieces"
        ))

    # ==================== 画 container 边框 ====================
    for xs, ys, zs in _box_edges(container_size):
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines", showlegend=False
        ))

    diag = np.linalg.norm(container_size)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, container_size[0]]),
            yaxis=dict(range=[0, container_size[1]]),
            zaxis=dict(range=[0, container_size[2]]),
            aspectmode="data",
            camera=dict(
                center=dict(x=0.5, y=0.5, z=0.5),
                up=dict(x=0, y=0, z=1),
                eye=dict(
                    x=diag * 2,
                    y=diag * 2,
                    z=diag * 2.5
                )
            )
        ),
        title="3D Voxel Packing (Optimized)",
        width=1000,
        height=1000,
        margin=dict(r=10, l=10, b=10, t=40)
    )

    fig.show()

# def visualize_voxel_model(layout_object_info, pieces_order, container_size, container_shape=None):
#     """
#     layout_object_info: list[{"array": np.ndarray(bool/int), "translation": (tx,ty,tz)}]
#     pieces_order: list[int]  与 layout_object_info 对应的 piece_id
#     container_size: (X, Y, Z)
#     """
#     n_pieces = len(layout_object_info)
#     assert len(pieces_order) == n_pieces, "pieces_order 长"


#     unique_ids = sorted(set(pieces_order))
#     id2idx = {pid:i for i,pid in enumerate(unique_ids)}

#     color_list = get_n_colors(len(unique_ids), colorscale='Turbo')

#     fig = go.Figure()
#     legend_added = set()

#     for which_piece, each_piece in enumerate(layout_object_info):
#         scene = np.zeros(container_size, dtype=np.uint8)
#         tem = insert_add(scene, each_piece["array"], each_piece["translation"])
#         x, y, z = np.where(tem == 1)
#         if x.size == 0:
#             continue

#         xv, yv, zv, _i, _j, _k = voxel_to_mesh3d(x, y, z)

#         piece_id = pieces_order[which_piece]
#         color = color_list[id2idx[piece_id]]
#         show_legend = piece_id not in legend_added
#         if show_legend:
#             legend_added.add(piece_id)

#         fig.add_trace(go.Mesh3d(
#             x=xv, y=yv, z=zv, i=_i, j=_j, k=_k,
#             opacity=0.75, color=color, name=f"Piece {piece_id}",
#             showlegend=show_legend,
#             lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2, roughness=0.8)
#             # lightposition=dict(x=0, y=0, z=1e5)
#         ))

#     for xs, ys, zs in _box_edges(container_size):
#         fig.add_trace(go.Scatter3d(
#             x=xs, y=ys, z=zs, mode="lines", name=None, showlegend=False
#         ))

#     diag = np.linalg.norm(container_size) 
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(range=[0, container_size[0]], autorange=False),
#             yaxis=dict(range=[0, container_size[1]], autorange=False),
#             zaxis=dict(range=[0, container_size[2]], autorange=False),

#             aspectmode="data",  

#             # dragmode="orbit",

#             camera=dict(
#                 # projection=dict(type="orthographic"),
#                 center=dict(x=0.5, y=0.5, z=0.5),
#                 up=dict(x=0, y=0, z=1),
#                 eye=dict(
#                     x=diag * 2,
#                     y=diag * 2,
#                     z=diag * 2.5
#                 )
#             )
#         ),

#     # uirevision="lock",  
#     title="3D Voxel Packing (Single Container)",
#     width=1000,
#     height=1000,
#     margin=dict(r=10, l=10, b=10, t=40)

# )

#     fig.show()
