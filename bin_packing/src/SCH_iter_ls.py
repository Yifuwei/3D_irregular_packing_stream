import math
import numpy as np
import line_profiler
from numba import njit
# from scipy.ndimage import rotate
from scipy.spatial import ConvexHull
from function_lib import get_feasible_boundary, visualize_single_object, aabb_rotate


global TRACE # a switch for output

def trace(msg): 
    
    if TRACE: 
        print(msg)
        
def normalised(data):
    # Move the reference point of the object to (0,0,0) 
    # reference is the bottom backc voxel of a voxelised object
    
    x,y,z = np.where(data == 1)
    x_normal = np.min(x)
    y_normal = np.min(y)
    z_normal = np.min(z)
    data = translate_voxel(data,(-x_normal,-y_normal,-z_normal))
    return data 

def translate_voxel(voxel_data, steps):
    # tem = []
    # # Use np.roll to shift the array; shift is (x, y, z) displacement
    # for each_step in steps:
    #     tem.append(-each_step)
        
    translated_data = np.roll(voxel_data, shift=steps, axis=(0, 1, 2))
    
    
    return translated_data

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
            
        rotated_data = np.rot90(voxel_data, k=angle/90, axes=_axes)
        rotated_data = normalised(rotated_data)
        return rotated_data
    
    
def add_more_space(data,space_length):
    # add space_length in three dimensions
    # it defines the size of container 
    padded_data = np.pad(data, pad_width=space_length, mode='constant', constant_values=0)
    return padded_data

def get_bounding_box(shapes):
    # calculate the max(x,y,z) of a bounding box
    x,y,z = np.where(shapes == 1)
    
    delta_x = np.max(x) - np.min(x) + 1
    delta_y = np.max(y) - np.min(y) + 1
    delta_z = np.max(z) - np.min(z) + 1
    
    return delta_x,delta_y,delta_z

def get_max_translation(shapes):   
    # calculate the max(x,y,z) of a shape can move
    
    x,y,z = np.where(shapes == 1)
    
    max_x = len(shapes[0]) 
    max_y = len(shapes[1]) 
    max_z = len(shapes[2]) 
    
    delta_x = max_x - np.max(x) 
    delta_y = max_y - np.max(y) 
    delta_z = max_z - np.max(z)
    
    return delta_x,delta_y,delta_z


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

def voxel_floor(num): 
    # get the closest smaller positive integer
    if num == 0:
        return 0 
    else: 
        return math.floor(num)
    
# @njit
# def translation_njit(arr, shift):
#     result = np.empty_like(arr)
#     sx, sy, sz = shift
#     x_len, y_len, z_len = arr.shape

#     for x in range(x_len):
#         src_x = (x - sx) % x_len
#         for y in range(y_len):
#             src_y = (y - sy) % y_len
#             for z in range(z_len):
#                 src_z = (z - sz) % z_len
#                 result[x, y, z] = arr[src_x, src_y, src_z]

#     return result

def nesting_evaluation(ongoing, partial_solution_in_the_bin, packing_position_list, nesting, bin_size, encourage_dbl):

    best_position = (-1,-1,-1)
    best_value = np.inf
    value_list_pairs = []
    
    for each_position in packing_position_list: 

        # This is for evaluation 
        translated_test = np.roll(ongoing, shift=each_position, axis=(0, 1, 2))
        # translated_test = translation_njit(ongoing,each_position)
        
        bin_test = partial_solution_in_the_bin + translated_test
        
        # distance to the original point, add to the value to select the point which is closer to (0,0,0)
        # smaller the value, is better. 
        distance_to_0 = 0
        
        if encourage_dbl:
            distance_to_0 =  (each_position[0])**2 + (each_position[1])**2 + (each_position[2])**2
            # trace(f"Distance to original point {distance_to_0}")
            
        # accessible = accessibility_check(translated_test, position_bin, topos_partial_solution, bin_size)
        # trace(f"The accessibility check is finished")
        # if accessible:
        # print(f"The packing position is accessible vertically.")

        x,y,z = get_bounding_box(bin_test)
        
        if nesting == 1: 
            
            value = x * y * z + distance_to_0
            
        elif nesting == 2: 
            # smaller value is better
            value = 4 * (x + y + z) + distance_to_0
            
        elif nesting == 3: 
            # as smaller value is better, so a negative mark is required
            value = - max(bin_size[0]*bin_size[1]*(bin_size[2]-z), 
                        bin_size[0]*(bin_size[1]-y)*bin_size[2], 
                        (bin_size[0]-x)*bin_size[1]*bin_size[2]) + distance_to_0
        
        value_list_pairs.append((each_position,value))

        if value < best_value:
            best_value = value
            best_position = each_position
            
            
    # sorted_positions = [pos for pos, value in sorted(value_list_pairs,key=lambda x:x[1])]
            
    return best_position

def SC_heuristic(nfv_pool, ifv_pool, ongoing_object_info, current_layout, topos_layout, position_bin, 
                 bin_size, nesting_strategy, density, axis, container_shape, 
                 _type, _accessible_check, _encourage_dbl, _select_range, flag_NFV_POOL, _TRACE): 
    
    """_summary_
        The Selection of Candidates Heuritic (SCH) 
        
    Args:
        each_object_info: a list for all info of the ongoing object
        ongoing (3D array): the next piece/ rotated shape
        current_layout (4D list): the non-merged partial solution for all bins INFO
        position_bin (int): which bin you are packing for 
        bin_size (tuple): the size of container e.g. (66,66,66)
        nesting_strategy(string):  To decide the criteria for determining what is a good packing position. 
        density (int): a parameter to decide the density of cross-section selection, when density is 2, 
                        it means the feasible region is divided into 2 
        axis (string): "x" "y" "z" to decide cut the feasible region aligned with which axis 
        _type (string): select voxel in the cross-section by 
        
    Output: 
        Best_coord(tuple): The best (x,y,z) among all candidate according to the nesting strategy. 
        packing_position_pool(list): The candidate pool of packing positions sorted from best to worst
    
    """
    global TRACE
    
    TRACE = _TRACE
    select_range = _select_range
    
    # best_value = 99999999
    
    # print("getting feasible region")
    # check1 = time.time()

    feasible_region = get_feasible_boundary(topos_layout[position_bin], current_layout[position_bin], ongoing_object_info, bin_size, container_shape, nfv_pool, ifv_pool, flag_NFV_POOL, _accessible_check)
    # visualize_single_object(feasible_region,np.shape(feasible_region))

    # check2 = time.time() 
    # print(f"got feasible region, cost {check2 - check1} s")
    
    if type(feasible_region) == bool:  
        # print("can't find a feasible position")
        
        return False
    
    if select_range == "all": 
        # original version 
        packing_position_list = selection_process(feasible_region, density, axis, _type)
        
    elif select_range == "bottom":
        # only select voxels from the bottom of fr
        packing_position_list = selection_process_bottom_only(feasible_region, _type)
    
    elif select_range == "bottom_top":
        # only select voxels from the bottom and top of fr
        packing_position_list = selection_process_bottom_top_only(feasible_region, _type)   

    elif select_range == "bottom_left_filling":
        # only select voxels from the bottom and top of fr

        return selection_process_bottom_left_filling(feasible_region)  
    # print("Packing position candidates are: ", packing_position_list)
    if nesting_strategy == "minimum volume of AABB": 
        
        nesting = 1
        
    elif nesting_strategy == "minimum length of edges of AABB": 
        # smaller value is better
        nesting = 2
        
    elif nesting_strategy == "maximum connected space": 
        # as smaller value is better, so a negative mark is required
        nesting = 3
        
    best_position = nesting_evaluation(ongoing_object_info["array"], topos_layout[position_bin], packing_position_list, nesting, bin_size, _encourage_dbl)
            
    return best_position

  

def selection_process_bottom_only(feasible_region, _type):
    # this algorithm only packing item on bottom of feasible region
    
    packing_position_list = []
    
    voxel_x, voxel_y, voxel_z = np.where(feasible_region == 1)
    # length, width, height = get_bounding_box(feasible_region)
        
    tem = min(voxel_z) # smallest coord on z axis
    bottom = feasible_region[:, :, tem] 
    x, y = np.where(bottom == 1) # if it has no point there, it will return a empty array 

    if x.size == 0:
        # not possible tho, for robustness
        coords = []
    
    else:
        if x.size <= 50: 
            z = np.full_like(x, tem)
            
            coords = list(zip(x,y,z))
            
            for each_point in coords:    # add to the overall list
                packing_position_list.append(each_point) 

        elif x.size > 50 or _type == "bounding_box":
            # inter-points with bounding box
            min_x = min(x)
            max_x = max(x)
            min_y = min(y)
            max_y = max(y)
        
            x_max_index = np.where(x == max_x)[0]
            x_min_index = np.where(x == min_x)[0]
            y_max_index = np.where(y == max_y)[0]
            y_min_index = np.where(y == min_y)[0]
            
            x = np.array((max_x,max_x,min_x,min_x,min(x[y_max_index]),max(x[y_max_index]),min(x[y_min_index]),max(x[y_min_index])))
            y = np.array((min(y[x_max_index]),max(y[x_max_index]),min(y[x_min_index]),max(y[x_min_index]),max_y,max_y,min_y,min_y))
            z = np.full_like(x,tem)
            # coords = list(zip(x,y,z))
            coords = list(dict.fromkeys(zip(x, y, z))) # get rid of replicated elements 
            
            for each_point in coords:    # add to the overall list
                packing_position_list.append(each_point)  
            
        elif _type == "convexhull":
            # inter-points with convex hull
            points = np.column_stack((x, y))
            hull = ConvexHull(points)
            coords = points[hull.vertices]
            
            for each_point in coords:    # add to the overall list
                point = list(each_point)
                point.append(bottom)
                packing_position_list.append(point)

            
    return packing_position_list 

def selection_process_bottom_left_filling(feasible_region):

    x,y,z = np.where(feasible_region == 1)

    if x.size == 0:
        return None

    bottom_layer = (z == z.min())
    x, y= x[bottom_layer], y[bottom_layer]

    min_x = x.min()
    x, y = x[x == min_x], y[x == min_x]

    min_y = y.min()
    x, y = x[y == min_y], y[y == min_y]
    # print(x[0],y[0],z.min())
    # exit(-1)
    return x[0],y[0],z.min()

def selection_process_bottom_top_only(feasible_region, _type):
    # this algorithm only packing item on bottom or on the top of feasible region
    
    packing_position_list = []
    
    voxel_x, voxel_y, voxel_z = np.where(feasible_region == 1)
    # length, width, height = get_bounding_box(feasible_region)

    bottom = min(voxel_z)
    top = max(voxel_z)
    
    top_bottom = [bottom,top]
    
    for each_crosection in top_bottom:
        
        cross_section = feasible_region[:, :, each_crosection] 
        x, y = np.where(cross_section == 1) # if it has no point there, it will return a empty array 
        
        if x.size == 0:
            coords = []
        
        else:
            
            if x.size <= 25:
                z = np.full_like(x,each_crosection)
                coords = list(zip(x,y,z))
                # print("coords are", coords)
                
                for each_point in coords:    # add to the overall list
                    packing_position_list.append(each_point)
            
            elif x.size > 25 or _type == "bounding_box":
                # inter-points with bounding box
                min_x = min(x)
                max_x = max(x)
                min_y = min(y)
                max_y = max(y)
            
                x_max_index = np.where(x == max_x)[0]
                x_min_index = np.where(x == min_x)[0]
                y_max_index = np.where(y == max_y)[0]
                y_min_index = np.where(y == min_y)[0]
                
                x = np.array((max_x,max_x,min_x,min_x,min(x[y_max_index]),max(x[y_max_index]),min(x[y_min_index]),max(x[y_min_index])))
                y = np.array((min(y[x_max_index]),max(y[x_max_index]),min(y[x_min_index]),max(y[x_min_index]),max_y,max_y,min_y,min_y))
                z = np.full_like(x,each_crosection)
                # coords = list(zip(x,y,z))
                coords = list(dict.fromkeys(zip(x, y, z))) # get rid of replicated elements 
                
                for each_point in coords:    # add to the overall list
                    packing_position_list.append(each_point)  
                
            elif _type == "convexhull":
                # inter-points with convex hull
                points = np.column_stack((x, y))
                hull = ConvexHull(points)
                coords = points[hull.vertices]
                
                for each_point in coords:    # add to the overall list
                    point = list(each_point)
                    point.append(each_crosection)
                    packing_position_list.append(point)
                        
    return packing_position_list 


def selection_process(ini_feasible_region, density, axis, _type): 
    
    packing_position_list = []
    
    voxel_x, voxel_y, voxel_z = np.where(ini_feasible_region == 1)
    length, width, height = get_bounding_box(ini_feasible_region)
        
    if axis == "z": 
        voxel_coord = []
        step = height/density
        tem = min(voxel_z)
        
        for each_step in range(density):
            tem += step
            voxel_coord.append(voxel_floor(tem))
        
        for each_crosection in voxel_coord:
            cross_section = ini_feasible_region[:, :, each_crosection] 
            x, y = np.where(cross_section == 1) # if it has no point there, it will return a empty array 
            
            if x.size == 0:
                coords = []
            
            else:
                
                if x.size <= 4: 
                    z = np.full_like(x,each_crosection)
                    coords = list(zip(x,y,z))
                    
                    for each_point in coords:    # add to the overall list
                        packing_position_list.append(each_point)  
                        
                elif _type == "bounding_box":
                    # inter-points with bounding box
                    min_x = min(x)
                    max_x = max(x)
                    min_y = min(y)
                    max_y = max(y)
                
                    x_max_index = np.where(x == max_x)[0]
                    x_min_index = np.where(x == min_x)[0]
                    y_max_index = np.where(y == max_y)[0]
                    y_min_index = np.where(y == min_y)[0]
                    
                    x = np.array((max_x,max_x,min_x,min_x,min(x[y_max_index]),max(x[y_max_index]),min(x[y_min_index]),max(x[y_min_index])))
                    y = np.array((min(y[x_max_index]),max(y[x_max_index]),min(y[x_min_index]),max(y[x_min_index]),max_y,max_y,min_y,min_y))
                    z = np.full_like(x,each_crosection)
                    
                    # coords = list(zip(x,y,z))
                    coords = list(dict.fromkeys(zip(x, y, z))) # get rid of replicated elements 
                    
                    for each_point in coords:    # add to the overall list
                        packing_position_list.append(each_point)  
                    
                elif _type == "convexhull":
                    # inter-points with convex hull
                    points = np.column_stack((x, y))
                    hull = ConvexHull(points)
                    coords = points[hull.vertices]
                    
                    for each_point in coords:    # add to the overall list
                        point = list(each_point)
                        point.append(each_crosection)
                        packing_position_list.append(point)
                    
                    
        if axis == "y": 
            
            voxel_coord = []
            step = width/density
            tem = min(voxel_y)
            
            for each_step in range(density):
                tem += step
                voxel_coord.append(voxel_floor(tem))
            
            for each_crosection in voxel_coord:
                cross_section = ini_feasible_region[:, each_crosection, :] 
                x, z = np.where(cross_section == 1) # if it has no point there, it will return a empty array 
                
                if x.size == 0:
                    coords = []

                else:
                    
                    if x.size <= 4: 
                        y = np.full_like(x,each_crosection)
                        coords = list(zip(x,y,z))
                        for each_point in coords:    # add to the overall list
                            packing_position_list.append(each_point)  
                        
                    elif _type == "bounding_box":
                        # inter-points with bounding box
                        min_x = min(x)
                        max_x = max(x)
                        min_z = min(z)
                        max_z = max(z)
                    
                        x_max_index = np.where(x == max_x)[0]
                        x_min_index = np.where(x == min_x)[0]
                        z_max_index = np.where(z == max_z)[0]
                        z_min_index = np.where(z == min_z)[0]
                        
                        x = np.array(max_x,max_x,min_x,min_x,min(x[z_max_index]),max(x[z_max_index]),min(x[z_min_index]),max(x[z_min_index]))
                        z = np.array(min(z[x_max_index]),max(z[x_max_index]),min(z[x_min_index]),max(z[x_min_index]),max_z,max_z,min_z,min_z)  
                        
                        y = np.full_like(x,each_crosection)
                        # coords = list(zip(x,y,z))
                        coords = list(dict.fromkeys(zip(x, y, z))) # get rid of replicated elements 
                        
                        for each_point in coords:    # add to the overall list
                            
                            packing_position_list.append(each_point)  
                            
                    elif _type == "convexhull":
                        # inter-points with convex hull
                        points = np.column_stack((x, z))
                        hull = ConvexHull(points)
                        coords = points[hull.vertices]
                        
                        for each_point in coords:    # add to the overall list
                            point = list(each_point)
                            point.append(each_crosection)
                            packing_position_list.append(point)
                            
                    
        if axis == "x": 
                
            voxel_coord = []
            step = length/density
            tem = min(voxel_x)
            
            for each_step in range(density):
                tem += step
                voxel_coord.append(voxel_floor(tem))
            
            for each_crosection in voxel_coord:
                
                cross_section = ini_feasible_region[each_crosection, :, :] 
                y, z = np.where(cross_section == 1) # if it has no point there, it will return a empty array 
                
                if y.size == 0:
                    coords = []
                
                else:
                    
                    if y.size <= 4: 
                        x = np.full_like(y,each_crosection)
                        coords = list(zip(x,y,z))
                        for each_point in coords:    # add to the overall list
                            packing_position_list.append(each_point)  
                        
                    elif _type == "bounding_box":
                        # inter-points with bounding box
                        min_z = min(z)
                        max_z = max(z)
                        min_y = min(y)
                        max_y = max(y)
                    
                        z_max_index = np.where(z == max_z)[0]
                        z_min_index = np.where(z == min_z)[0]
                        y_max_index = np.where(y == max_y)[0]
                        y_min_index = np.where(y == min_y)[0]
                        
                        y = np.array(max_y,max_y,min_y,min_y,min(y[z_max_index]),max(y[z_max_index]),min(y[z_min_index]),max(y[z_min_index]))
                        z = np.array(min(z[y_max_index]),max(z[y_max_index]),min(z[y_min_index]),max(z[y_min_index]),max_z,max_z,min_z,min_z)  
                        
                        x = np.full_like(y,each_crosection)
                        # coords = list(zip(x,y,z))
                        coords = list(dict.fromkeys(zip(x, y, z))) # get rid of replicated elements 
                        
                        for each_point in coords:    # add to the overall list
                            packing_position_list.append(each_point)      
          
                        
                    elif _type == "convexhull":
                        # inter-points with convex hull
                        points = np.column_stack((y, z))
                        hull = ConvexHull(points)
                        coords = points[hull.vertices]
                        
                        for each_point in coords:    # add to the overall list
                            point = list(each_point)
                            point.append(each_crosection)
                            packing_position_list.append(point) 
          
    return packing_position_list
