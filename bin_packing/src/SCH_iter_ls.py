import math
import numpy as np
import line_profiler
from numba import njit
# from scipy.ndimage import rotate
from scipy.spatial import ConvexHull
from function_lib import get_feasible_boundary, get_feasible_boundary_rigorous_acces, visualize_single_object, aabb_rotate


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

def nesting_evaluation(ongoing, partial_solution_in_the_bin, current_layout, packing_position_list, nesting, bin_size, encourage_dbl):

    best_position = None
    best_value = np.inf
    value_list_pairs = []

    _L, _W, _H = ongoing.shape
    translated_test = np.zeros((_L, _W, _H), dtype=ongoing.dtype)
    bin_test = np.empty((_L, _W, _H), dtype=partial_solution_in_the_bin.dtype)
    occupied = np.argwhere(ongoing != 0)
    if occupied.size == 0:
        return False
    lower, upper = occupied.min(axis=0), occupied.max(axis=0)
    limits = np.minimum(ongoing.shape, bin_size)

    for each_position in packing_position_list:
        # Check before slicing: truncation could otherwise make an invalid
        # placement look smaller (and therefore better) to the objective.
        shift = np.asarray(each_position)
        if (shift.shape != (3,) or not np.issubdtype(shift.dtype, np.integer)
                or np.any(lower + shift < 0) or np.any(upper + shift >= limits)):
            continue
        sx, sy, sz = each_position

        # In-place translation: no allocation per iteration, no wrap-around
        # (safe because feasible positions are always within bin bounds)
        translated_test[:] = 0
        src = (slice(max(0, -sx), _L - max(0, sx)),
               slice(max(0, -sy), _W - max(0, sy)),
               slice(max(0, -sz), _H - max(0, sz)))
        dst = (slice(max(0,  sx), _L - max(0, -sx)),
               slice(max(0,  sy), _W - max(0, -sy)),
               slice(max(0,  sz), _H - max(0, -sz)))
        translated_test[dst] = ongoing[src]
        if np.any((translated_test != 0) & (partial_solution_in_the_bin != 0)):
            continue
        np.add(partial_solution_in_the_bin, translated_test, out=bin_test)

        # distance to the original point, add to the value to select the point which is closer to (0,0,0)
        # smaller the value, is better.
        distance_to_0 = 0

        if encourage_dbl:
            distance_to_0 = sx**2 + sy**2 + sz**2
            # trace(f"Distance to original point {distance_to_0}")

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
            
        elif nesting == 4:
            current_layout_test = list(current_layout)                                
            topos_layout_test = bin_test

            max_x, max_y, max_z, min_x, min_y, min_z = find_max_xyz(topos_layout_test)
            max_x1, max_y1, max_z1, min_x1, min_y1, min_z1 = find_max_xyz(translated_test)
              # overlap
            overlap = 0
            
            for each_object_info in current_layout_test: 
                # packed_polygon = Polygon(each_polygon)
                overlap += get_intersection(translated_test,each_object_info["array"])
                
            # distance
            center_layout = ((max_x + min_x)/2 , (max_y + min_y)/2, (max_z + min_z)/2)
            center_next = ((max_x1 + min_x1)/2 , (max_y1 + min_y1)/2, (max_z1 + min_z1)/2)
            
            distance = math.sqrt((center_layout[0]-center_next[0])**2 + (center_layout[1]-center_next[1])**2 + (center_layout[2]-center_next[2])**2) 

            value = -overlap + distance

        value_list_pairs.append((each_position,value))

        if value < best_value:
            best_value = value
            best_position = each_position


    # sorted_positions = [pos for pos, value in sorted(value_list_pairs,key=lambda x:x[1])]

    return False if best_position is None else (best_value, tuple(best_position))

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
        (score, (x, y, z)), with smaller scores preferred, or False if no
        feasible candidate exists. All selection modes use this contract.
    
    """
    global TRACE
    
    TRACE = _TRACE
    select_range = _select_range
    
    # best_value = 99999999
    
    # print("getting feasible region")
    # check1 = time.time()

    # feasible_region = get_feasible_boundary(topos_layout[position_bin], current_layout[position_bin], ongoing_object_info, bin_size, container_shape, nfv_pool, ifv_pool, flag_NFV_POOL, _accessible_check)
    
    feasible_region = get_feasible_boundary_rigorous_acces(topos_layout[position_bin], current_layout[position_bin], ongoing_object_info, bin_size, container_shape, nfv_pool, ifv_pool, flag_NFV_POOL)
    
    # visualize_single_object(feasible_region,np.shape(feasible_region))

    # check2 = time.time() 
    # print(f"got feasible region, cost {check2 - check1} s")
    
    if isinstance(feasible_region, (bool, np.bool_)) or not np.any(feasible_region):
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

        position = selection_process_bottom_left_filling(feasible_region)
        packing_position_list = [] if position is None else [position]
    else:
        raise ValueError(f"Unknown selection range: {select_range}")
    
    # print("Packing position candidates are: ", packing_position_list)
    if nesting_strategy == "minimum_volume_of_AABB" or nesting_strategy == "minimum_aabb_volume": 
        
        nesting = 1
        
    elif nesting_strategy == "minimum_length_of_edges_of_AABB" or nesting_strategy == "minimum_aabb_edges_len": 
        # smaller value is better
        nesting = 2
        
    elif nesting_strategy == "maximum_connected_space" or nesting_strategy == "maximal_residual_box": 
        # as smaller value is better, so a negative mark is required
        nesting = 3

    elif nesting_strategy == "overlap_distance":
        nesting = 4
    else:
        raise ValueError(f"Unknown nesting strategy: {nesting_strategy}")

    best_position_and_value = nesting_evaluation(ongoing_object_info["array"], topos_layout[position_bin], current_layout[position_bin], packing_position_list, nesting, bin_size, _encourage_dbl)
            
    return best_position_and_value

  

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
    """Sample occupied cross-sections, returning coordinates in x/y/z order."""
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Unknown selection axis: {axis}")
    if not isinstance(density, (int, np.integer)) or density <= 0:
        raise ValueError("density must be a positive integer")
    if _type not in ("bounding_box", "convexhull"):
        raise ValueError(f"Unknown candidate selection type: {_type}")
    occupied = np.argwhere(ini_feasible_region == 1)
    if occupied.size == 0:
        return []
    axis_index = ("x", "y", "z").index(axis)
    other_axes = [i for i in range(3) if i != axis_index]
    low, high = occupied[:, axis_index].min(), occupied[:, axis_index].max()
    step = (high - low + 1) / density
    layers = dict.fromkeys(min(int(high), math.floor(low + step * i))
                           for i in range(1, density + 1))
    candidates = []
    for layer in layers:
        points = np.argwhere(np.take(ini_feasible_region, layer, axis=axis_index) == 1)
        if len(points) == 0:
            continue
        if len(points) > 4:
            if _type == "convexhull" and np.linalg.matrix_rank(points - points[0]) == 2:
                points = points[ConvexHull(points).vertices]
            else:
                # Endpoints of the occupied rows/columns at each extremum.
                # Every selected point remains a member of the feasible region.
                selected = []
                for dim in range(2):
                    for edge in (points[:, dim].min(), points[:, dim].max()):
                        boundary = points[points[:, dim] == edge]
                        selected.extend((boundary[boundary[:, 1-dim].argmin()],
                                         boundary[boundary[:, 1-dim].argmax()]))
                points = np.unique(selected, axis=0)
        for point in points:
            coordinate = [0, 0, 0]
            coordinate[axis_index] = layer
            for dim, value in zip(other_axes, point):
                coordinate[dim] = int(value)
            candidates.append(tuple(coordinate))
    return list(dict.fromkeys(candidates))
