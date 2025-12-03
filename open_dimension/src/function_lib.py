# /*
#  * @Author: Yifu Wei 
#  * @Date: 2025-05-27 13:40:23 
#  * @Last Modified by:   Yifu Wei 
#  * @Last Modified time: 2025-05-27 13:40:23 
#  */


import numpy as np 
import math
import binvox_rw
import plotly.graph_objects as go
import plotly.express as px
import kaleido # for saving image cannot delete
import line_profiler
import time 

from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from numba import njit, cuda # for GPU acceleration

from skimage.morphology import ball
from scipy.ndimage import binary_fill_holes, binary_dilation, label, binary_propagation

# from threeD_PACKING_VOXEL import get_nfv, visualize_voxel_model, visualize_single_object, get_feasible_boundary

class NFV_POOL:

    def __init__(self):

        self.nfv_pool = {}
        self.all_nfv_cal = 0
        self.val_nfv_cal = 0
        self.rep_nfv_cal = 0

    # @profile
    def pre_get_nfv_gpu(self, fix_info, ongoing_info):
        # nfv is unavailable region

        # x1,y1,z1 = get_bounding_box(fix_info["array"])
        # x2,y2,z2 = get_bounding_box(ongoing_info["array"])
        # print("fixed computation",(x1,y1,z1))
        # print("ongoing computation",(x2,y2,z2))

        x1,y1,z1 = aabb_rotate(fix_info["aabb"], fix_info["orientation"])
        x2,y2,z2 = aabb_rotate(ongoing_info["aabb"], ongoing_info["orientation"])
        # print((x1,y1,z1))
        # print((x2,y2,z2))

        # exit(-1)
        dimension = (x1+2*x2,
                    y1+2*y2,
                    z1+2*z2)  
        
        normal_fixed = pre_normalised_object(fix_info["array"],dimension)
        normal_ongoing = pre_normalised_object(ongoing_info["array"],dimension)
        
        # move nfv to the "central position of the container"
        # This is for create space to move ongoing piece
        normal_fixed = translate_voxel_quick(normal_fixed,(x2,y2,z2))

        calculations = (x1+x2, y1+y2, z1+z2)

        NFV = get_nfv_gpu(normal_fixed, normal_ongoing, dimension, calculations)
          
        return NFV
    
    # @profile
    def get_nfv_key(self, obj1_info, obj2_info):

        return (obj1_info["piece_type"], obj1_info["orientation"], obj2_info["piece_type"], obj2_info["orientation"])
    
    # @profile
    def key_check(self, key):

        # for robustness
        if self.nfv_pool is None:
            self.nfv_pool = {}

        # Memory copy?
        # 
        trial1 = self.nfv_pool.get(key, False)

        # nfv(A,B) = -nfv(B,A)
        # orientation needs to switch also with 
        mirrow_key =  (key[2],key[3],key[0],key[1])
        trial2 = self.nfv_pool.get(mirrow_key, False)

        trial2 = False

        # print(self.nfv_pool)
        # print("trail1 is", trial1)
        # print("trail2 is", trial2)

        if isinstance(trial1,bool) and isinstance(trial2,bool) :
        # can't find the exactly same key

            orien1 = key[1]
            orien2 = key[3]

            axis1, degree1 = orien1.split("_")
            axis2, degree2 = orien2.split("_")

            degree1 = int(degree1)
            degree2 = int(degree2)

            if axis1 != axis2:
                # if two objects don't rotate on the same axis, can't do anything
                return {"nfv_type": "no_nfv"}
                
            else:
                # if two object rotate on the same axis, then we consider if we can find their nfv by rotating nfv

                # check if there is same pair or mirrow pair
                target_key = (key[0], '??', key[2], '??')   
                target_key_mirror = (key[2], '??', key[0], '??')   

                matched_keys = [k for k in self.nfv_pool if k[0] == target_key[0] and k[2] == target_key[2]]
                matched_keys_mirror = [k for k in self.nfv_pool if k[0] == target_key_mirror[0] and k[2] == target_key_mirror[2]]


                if matched_keys == [] and matched_keys_mirror == []:
                    # don't consider any same pairs before
                    return {"nfv_type": "no_nfv"}
                
                elif matched_keys != []:
                    # if we calculate same pairs before
                    equal_rotate_list = []
                    tem_degree1 = degree1
                    tem_degree2 = degree2

                    # generate three equvalent rotation candidates
                    # we need to check if there is any pair have these rotation
                    for each_step in range(3):
                        tem_degree1 += 90
                        tem_degree2 += 90
                        if tem_degree1 > 300:
                            tem_degree1 = 0
                        elif tem_degree2 > 300:
                            tem_degree2 = 0
                        equal_rotate_list.append((tem_degree1,tem_degree2))

                    # if there is the same pair of item was calculated, and they are rotated on the same axis.
                    for each_common_key in matched_keys:

                        if (each_common_key[1], each_common_key[3]) in equal_rotate_list:
                            # when needed nfv has a larger rotate degree, can rotate the existing nfv directly
                            rotate_degree =  min(degree1, degree2) - min(each_common_key[1], each_common_key[3])

                            if rotate_degree < 0: 
                                # if the nfv in the pool has a larger degree of rotation
                                rotate_degree = rotate_degree + 360
 

                            return {"nfv_type": "rotate_nfv", "now_key":(key[0], each_common_key[1], key[2], each_common_key[3]), "rotate_info": axis1 + str(rotate_degree)}
                        
                        else:
                            pass

                elif matched_keys_mirror != []:
                    # if we calculate same pairs before
                    equal_rotate_list = []

                    # As it is the mirror pair, their rotations are needed to be modified
                    tem_degree1 = degree2
                    tem_degree2 = degree1

                    # generate three equvalent rotation candidates
                    # we need to check if there is any pair have these rotation
                    for each_step in range(3):
                        tem_degree1 += 90
                        tem_degree2 += 90
                        if tem_degree1 > 300:
                            tem_degree1 = 0
                        elif tem_degree2 > 300:
                            tem_degree2 = 0

                        # this list rotation is mirrowed
                        equal_rotate_list.append((tem_degree1,tem_degree2))

                    # if there is the same pair of item was calculated, and they are rotated on the same axis.
                    for each_common_key in matched_keys_mirror:

                        if (each_common_key[1], each_common_key[3]) in equal_rotate_list:
                            # when needed nfv has a larger rotate degree, can rotate the existing nfv directly
                            rotate_degree =  min(degree1, degree2) - min(each_common_key[1], each_common_key[3])

                            if rotate_degree < 0: 
                                # if the nfv in the pool has a larger degree of rotation
                                rotate_degree = rotate_degree + 360

                            return {"nfv_type": "rotate_nfv", "now_key":(key[2], each_common_key[1], key[0], each_common_key[3]), "rotate_info": axis1 + str(rotate_degree)}
                        
                        else:
                            pass
                            
                    
            return {"nfv_type": "no_nfv"}

        
        else:
            # if there is an existing nfv with the exactly same key in the pool
            # print("Same nfv Found")

            if isinstance(trial1, bool):
                # print("A mirrow key is found")
                trial2 = np.flip(trial2,axis=2)
                return {"nfv_type": "have_nfv", "nfv_array": trial2}
            
            else:
                return {"nfv_type": "have_nfv", "nfv_array": trial1}
            
    # @profile
    def retrieve_nfv(self, fixed_info, ongoing_info):
    
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

        # retrieve the nfv out
        # Key: (obj1_type, obj2_type, obj1_orien, obj2_orien)
        # obj1 and obj2 here need to be normalised
        
        # fixed_packing_position = fixed_info["translation"]

        # Create the key and retrieve the NFP from the dictionary
        key = self.get_nfv_key(fixed_info, ongoing_info)

        # check the key in the dict
        flag = self.key_check(key)

        self.all_nfv_cal += 1

        if flag["nfv_type"] == "no_nfv":
            # print(f"NFP not found for key {key}, a new NFV need to be calculated and stored.")

            # no nfv, get a new nfv and store,  get the original nfv
            nfv = self.pre_get_nfv_gpu(fixed_info, ongoing_info)
            self.val_nfv_cal += 1
            self.store_nfv(key, nfv) # no return value, direct call function

        # elif flag["nfv_type"] == "rotate_nfv":

        #     print("A rotatable nfv is found")
        #     # need key and rotation degree
        #     now_key = flag["now_key"]
        #     orientation = flag["rotate_info"]

        #     nfv_tem =  self.nfv_pool.get(now_key, None)
        #     rotate_axes, angle = orientation.split("_")
        #     angle = int(angle)

        #     nfv = rotate_voxel(nfv_tem, angle, rotate_axes)

        elif flag["nfv_type"] == "have_nfv":
            # print("A existing nfv is found")
            # read the first and the only element from the key of dict
            nfv = flag["nfv_array"]
            self.rep_nfv_cal += 1

        else:
            print("Error! No such a nfv_type")

        # =======================================================================
        # original version
        # Translate the original nfv to the right position depending on the fixed object position
        # this shape is (x1+2*x2, y1+2*y2, z1+2*z2)
        # current_shape = np.shape(nfv)

        # # visualize_single_object(nfv,np.shape(nfv))

        # # make current_shape same size of container or larger size
        # pad_x_after1 = max(container_size[0] - current_shape[0], 0)
        # pad_y_after1 = max(container_size[1] - current_shape[1], 0)
        # pad_z_after1 = max(container_size[2] - current_shape[2], 0)
        

        # x1,y1,z1 = aabb_rotate(ongoing_info["aabb"],ongoing_info["orientation"])
        # # x1,y1,z1 = get_bounding_box(ongoing_info["array"])

        # # translation coordinate for the nfv, original nfv is equvalente when fixed is at (x1,y1,z1)
        # steps = (fixed_packing_position[0]-x1, fixed_packing_position[1]-y1, fixed_packing_position[2]-z1)

        # x, y, z = nfv.shape
        
        # # make sure all dimensions are larger or equal than the bin size
        # nfv_new = np.zeros((x + pad_x_after1,
        #                     y + pad_y_after1,
        #                     z + pad_z_after1), dtype=nfv.dtype)
        
        # nfv_new[:x, :y, :z] = nfv
        
        # nfv = nfv_new

        # # make sure all dimensions are larger or equal than the bin size
        # # nfv = np.pad(
        # #     nfv,
        # #     pad_width=((0, pad_x_after1),
        # #                 (0, pad_y_after1),
        # #                 (0, pad_z_after1)),

        # #     mode='constant',
        # #     constant_values=0)

        # # # add more space for translating NFV to the right position.
        # # pad_x_before = max(-steps[0], 0)
        # # pad_x_after2 = max(steps[0], 0)
        
        # # pad_y_before = max(-steps[1], 0)
        # # pad_y_after2 = max(steps[1], 0)
        
        # # pad_z_before = max(-steps[2], 0)
        # # pad_z_after2 = max(steps[2], 0)
        # # # print("nfv shape", current_shape)

        # # # fill enough space
        # # nfv = np.pad(
        # #     nfv,
        # #     pad_width=((pad_x_before, max(pad_x_after1,pad_x_after2)),
        # #                 (pad_y_before, max(pad_y_after1,pad_y_after2)),
        # #                 (pad_z_before, max(pad_z_after1,pad_z_after2))),

        # #     mode='constant',
        # #     constant_values=0)


        # nfv = translate_voxel_quick(nfv, steps)

        # # print("Translation steps are, ", steps)


        # # # tailor the size to the shape of container
        # # start_x = max(0, min(pad_x_before, nfv.shape[0] - container_size[0]))
        # # start_y = max(0, min(pad_y_before, nfv.shape[1] - container_size[1]))
        # # start_z = max(0, min(pad_z_before, nfv.shape[2] - container_size[2]))

        # # nfv = nfv[
        # #     start_x : start_x + container_size[0],
        # #     start_y : start_y + container_size[1],
        # #     start_z : start_z + container_size[2]
        # # ]
        # # visualize_single_object(nfv,np.shape(nfv))

        # nfv = nfv[
        #     : container_size[0],
        #     : container_size[1],
        #     : container_size[2]
        # ]
        # =======================================================================

        return nfv
    
    def store_nfv(self, key, NFV):

        self.nfv_pool[key] = NFV

    def __str__(self):
        print (f"Now, NFV_pool has {len(self.nfv_pool)} pairs of NFV")
         


class IFV_POOL():

    def __init__(self):
        self.ifv_pool = {}
        self.all_ifv_cal = 0
        self.val_ifv_cal = 0
        self.rep_ifv_cal = 0
    
    def get_key(self,ongoing_info):
        
        return (ongoing_info["piece_type"], ongoing_info["orientation"])
    
    def check_key(self, key, ongoing_info, container_size, container_type):
        # this reads the existing ifv
        # and also store the new ifv 

        if self.ifv_pool is None:
            self.ifv_pool = {}

        trial1 = self.ifv_pool.get(key, False)

        if isinstance(trial1,bool):
            # no ifv in the pool, get the new one

            ifv = self.get_ifv(ongoing_info, container_size, container_type)
            self.val_ifv_cal += 1
            self.store_ifv(key,ifv)
            return ifv
        
        else:
            self.rep_ifv_cal += 1
            return trial1
        
    def store_ifv(self, key, ifv):

        self.ifv_pool[key] = ifv
        

    def retrieve_ifv(self, ongoing_info, container_size, container_type, max_height_possible):

        self.all_ifv_cal += 1
        key = self.get_key(ongoing_info)
        tem = self.check_key(key, ongoing_info, container_size, container_type)

        return tem
    

    def get_ifv(self, ongoing_info, container_size, container_type):
        
        if container_type == "cube":

            return self.get_ifv_cube(ongoing_info, container_size)

        elif container_type == "cylinder":
            obj_length, obj_width, obj_height = aabb_rotate(ongoing_info["aabb"], ongoing_info["orientation"])  

            return njit_get_ifv_cylinder(ongoing_info["array"],obj_length, obj_width, obj_height , container_size)

    def get_ifv_cube(self, ongoing_info, container_size): 
            
        h = container_size[0]
        w = container_size[1]
        d = container_size[2]

        ifv = np.zeros((h,w,d), dtype=int)      
        # print("DEBUG: ifv.shape:", ifv.shape, "container_size:", container_size)

        # obj_length, obj_width, obj_height = get_bounding_box(ongoing_info["array"])
        obj_length, obj_width, obj_height = aabb_rotate(ongoing_info["aabb"], ongoing_info["orientation"])      

        available_x  = h - obj_length
        available_y  = w - obj_width
        available_z  = d - obj_height

        ifv[:available_x, :available_y, :available_z] += 1

        return ifv  
    
  
# @njit 
# @profile
def insert_add(big, small, offset):
    """
    Insert `small` into `big` at `offset` by elementwise addition.
    Overlapping region is added; out-of-bound parts are ignored.
    """
    ox, oy, oz = offset
    Bx, By, Bz = big.shape
    sx, sy, sz = small.shape

    # valid overlapping range in big
    b0x, b0y, b0z = max(0, ox), max(0, oy), max(0, oz)
    b1x, b1y, b1z = min(Bx, ox + sx), min(By, oy + sy), min(Bz, oz + sz)

    if b0x >= b1x or b0y >= b1y or b0z >= b1z:
        return big  # no overlap

    # corresponding range in small
    s0x, s0y, s0z = b0x - ox, b0y - oy, b0z - oz
    s1x, s1y, s1z = s0x + (b1x - b0x), s0y + (b1y - b0y), s0z + (b1z - b0z)

    # add into big # just cover instead of sum 
    np.logical_or(big[b0x:b1x, b0y:b1y, b0z:b1z], small[s0x:s1x, s0y:s1y, s0z:s1z],out=big[b0x:b1x, b0y:b1y, b0z:b1z])

    return big


def get_ifv_cube_strip_packing(ongoing_info, container_size, max_height): 
        
    h = container_size[0]
    w = container_size[1]
    d = max_height

    ifv = np.zeros((h,w,d), dtype=int)      
    # print("DEBUG: ifv.shape:", ifv.shape, "container_size:", container_size)

    # obj_length, obj_width, obj_height = get_bounding_box(ongoing_info["array"])
    obj_length, obj_width, obj_height = aabb_rotate(ongoing_info["aabb"], ongoing_info["orientation"])      

    available_x  = h - obj_length
    available_y  = w - obj_width
    available_z  = d - obj_height

    # if it cannot be fitted in the bin return a zeros array
    if available_x < 0 or available_y < 0 or available_z < 0:
        return ifv
    
    ifv[:available_x, :available_y, :available_z] += 1

    return ifv  
    


@njit
def njit_get_ifv_cylinder(ongoing_array, obj_length, obj_width, obj_height, container_size): 
    
    h = int(container_size[0])
    w = int(container_size[1])
    d = int(container_size[2])
    
    ifv = np.zeros((h,w,d), dtype=np.int32)      
    # print("DEBUG: ifv.shape:", ifv.shape, "container_size:", container_size)
    
    # obj_length, obj_width, obj_height = get_bounding_box_njit_itself(ongoing_info)      
      
    available_x  = h - obj_length
    available_y  = w - obj_width
    available_z  = d - obj_height

        
    radius = h / 2
    # circle = np.zeros((h, w), dtype=np.int8)
    cx, cy = h / 2, w / 2
    
    y_indices, x_indices = np.indices((h, w))
    circle = ((x_indices - cx) ** 2 + (y_indices - cy) ** 2 > radius ** 2).astype(np.int32)
    
    # for i in range(container_size[0]):
    #     for j in range(container_size[1]):
    #         if (i - cx) ** 2 + (j - cy) ** 2 > radius ** 2:
    #             circle[i, j] = 1  

    projection = max_along_axis_2(ongoing_array).astype(np.int32)
    
    for i in range(available_x):
        for j in range(available_y):
            
            trans_projection = np.zeros_like(projection, dtype=np.int32)
            
            trans_projection[i: h, j: w] = projection[:h-i, :w-j]
            
            check = trans_projection + circle  
            
            if np.all(check <= 1):
                ifv[i, j, :available_z] = 1
                
    return ifv


@njit
def max_along_axis_2(arr):

    shape_x, shape_y, shape_z = arr.shape
    result = np.zeros((shape_x, shape_y), dtype=arr.dtype)

    for i in range(shape_x):
        for j in range(shape_y):
            for k in range(shape_z):
                
                if arr[i, j, k] == 1:
                    result[i, j] = 1
                    break
    
    return result

def accessibility_check(ongoing_info, nfv, ifv, scene_size):
    """_summary_
    This is for checking if there is feasible route to get to the packing position
    NOT necessarily vertical or straight line
    -------------------
    Args:
        ongoing_info (np.array)
        nfv (np.array) no-fit voxel 
        ifv (np.array) inner-fit voxel
        scene_size (tuple)
    ---------------------
    return:
        accessible_block - 3D binary np.array
        the small component that allows a accessible route from the top of the container

    """
    nfv_star = 1 - nfv # Get complement of nfv
    all_feasible_region = (nfv_star & ifv).astype(np.uint8) # get intersection of nfv* and ifv

    # build the stucture element for multi-connectivity check
    st = np.zeros((3,3,3), dtype=np.uint8)

    st[1,1,1] = 1
    st[0,1,1] = st[2,1,1] = 1  # ±x
    st[1,0,1] = st[1,2,1] = 1  # ±y
    st[1,1,0] = st[1,1,2] = 1  # ±z

    # aabb  = ongoing_info["aabb"]
    aabb_x,aabb_y,aabb_z = aabb_rotate(ongoing_info["aabb"],ongoing_info["orientation"])

    # =========================================================
    # Using binary_propagation, potentially quicker
    top_mask = np.zeros(scene_size, dtype=bool)
    top_mask[:, :, scene_size[2]-aabb_z-1] = True

    seeds = top_mask & all_feasible_region

    # binary flooding， when it reach 0, it will stop
    reachable = binary_propagation(input=seeds, structure=st, mask=all_feasible_region)
   
    if not reachable.any():  
        return False
    
    accessible_block = reachable.astype(np.uint8)

    return accessible_block

    # =========================================================
    # # multi connectivity check to separate all components.
    # labeled, num = label(all_feasible_region, structure=st)
    # accessible_block = np.zeros(container_size, dtype=np.uint8))

    # for i in range(1,num+1):
            
    #     mask_k = (labeled == k)

    #     # allows a accessible route from the top of the container
    #     if (mask_k[:, :, container_size[2]- aabb[2] - 1]).any():
    #         accessible_block |= mask_k.astype(np.uint8)

    #     else: 
    #         pass 


# @profile
def get_feasible_boundary(fixed, fixed_info, ongoing_info, container_size, container_type, nfv_pool, ifv_pool, max_height_possible, flag_NFV_POOL, _accessible_check):
    
    """_summary_
    
    get a hollow alternative packing area 
    NFV is an *infeasible region*
    NFV* is the supplement of NFV in container 
    (NFV* ∩ ifv) ∩ bounding_box of fixed object (this is because we only pack the reference voxel in the bounding box)
    
    Args:
        fixed (_type_): a list containing, they are INFO
        ongoing (_type_): _description_
        container_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    
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
    # call_nfv_gpu_time = 0
    # overall_nfv_time = 0

    ifv = get_ifv_cube_strip_packing(ongoing_info, container_size,max_height_possible)
    # print("ifv",ifv)
    if np.all(ifv < 0.5): # if there is no ifv is found

        return False
    
    # ============================================================
    # binary_dilation(NFV) - NFV ∩ ifv) ∩ bounding_box
    # if flag_NFV_POOL == True: 

    #     nfv = np.zeros(container_size, dtype=bool)
    #     # tem_list = []

    #     for each_packed_object_info in fixed_info:

    #         fixed_packing_position = each_packed_object_info["translation"]
        
    #         #check_nfv_0 = time.time()
    #         original_nfv = nfv_pool.retrieve_nfv(each_packed_object_info, ongoing_info)
            
    #         x1,y1,z1 = aabb_rotate(ongoing_info["aabb"],ongoing_info["orientation"])
    #         steps = (fixed_packing_position[0]-x1, fixed_packing_position[1]-y1, fixed_packing_position[2]-z1)

    #         # insert nfv on the right position
    #         nfv = insert_add(nfv, original_nfv, steps)

    # nfv = (nfv != 0)  # transfer back to bool 

        # print(each_packed_object_info)
        # L_bc, W_bc, H_bc = get_bounding_box_njit_translation(each_packed_object_info["array"])
        # max_x, max_y, max_z = get_max_translation_njit(ongoing_info["array"]

        # # to calculate range x y z for NFV
        # calculations = (min(max_x, L_bc),min(max_y, W_bc),min(max_z, H_bc))

        # tem = get_nfv_gpu(each_packed_object_info, ongoing_info, container_size, calculations)
        # nfv_pool[(each_packed_object_info["piece_type"], each_packed_object_info["orientation"], ongoing_info["piece_type"], ongoing_info["orientation"])] = \
        # nfv_pool.get((each_packed_object_info["piece_type"], each_packed_object_info["orientation"], ongoing_info["piece_type"], ongoing_info["orientation"]),0) + 1


        # check_nfv_1 = time.time()
        # visualize_single_object(tem,np.shape(tem))
        # call_nfv_gpu_time += 1
        # tem_list.append(tem)

        # nfv |= tem # np.logical_or(nfv, tem, out=nfv)

    # nfv = np.logical_or.reduce(tem_list).astype(int) # one calculation for the union of all nfvs

    # # ===============================================================
    # # L_bc, W_bc, H_bc = get_bounding_box_np(fixed)
    else:
        scene = np.zeros((container_size[0],container_size[1],max_height_possible))

        array = insert_add(scene,ongoing_info["array"],ongoing_info["translation"])
        
        L_bc, W_bc, H_bc = get_bounding_box_njit_translation(fixed)
        max_x, max_y, max_z = get_max_translation_nfv(ongoing_info,container_size) 

        # # max_x, max_y, max_z = get_max_translation_njit(ongoing_info["array"])

        range_x = min(max_x, L_bc)
        range_y = min(max_y, W_bc)
        range_z = min(max_z, H_bc)

        # print(np.shape(fixed[:, :, :max_height_possible]),array.size,(container_size[0],container_size[1],max_height_possible))
        # # check_nfv_0 = time.time()
        nfv = get_nfv_gpu(fixed[:, :, :max_height_possible], array, (container_size[0],container_size[1],max_height_possible), (range_x,range_y,range_z))    # fixed layout here is a merged partial solution
    # check_nfv_1 = time.time()

    # call_nfv_gpu_time += 1
    # # overall_nfv_time += (check_nfv_1-check_nfv_0)

    # #print("nfv cost", (check_nfv-check1))
    
    # if np.all(nfv == 0) == True: 
    #     # if can't get a nfv
    #     # return None directly
    #     return False
    
    
    # nfv and ifv must share the same size
    # get the posisble alternative of packing position
    # maybe choose the 8 vertices of feasible boundary
    
    # else: 

        #check_ifv1 = time.time()
        # ifv = get_ifv(ongoing_info, container_size, container_type)
        
        #check_ifv2 = time.time()
        
        #print("ifv cost", check_ifv2 - check_ifv1)
        
    structure = ball(1)
    # structure = np.ones((3, 3, 3), dtype=bool)
    dilated_voxel = binary_dilation(nfv, structure)
    
    tem =  dilated_voxel ^ nfv
    
    intersection = np.logical_and(tem, ifv).astype(int)
    
    if np.all(intersection < 0.5): # if there is no intersection including no ifv is find
        #  print("No intersection found between the complement of NFV and IFV.")
        #check2 = time.time()
        #print("FR costs,", check2 - check1)
        return False
    
    if _accessible_check == True: 
        # fill the hole in the nfv which means we don't consider packing items in another larger item
        # nfv = binary_fill_holes(nfv)
        accessible_region = accessibility_check(ongoing_info, nfv, ifv, (container_size[0],container_size[1],max_height_possible))

        if type(accessible_region) == bool:
            return False
        
        else: 
            _intersection = accessible_region & intersection
            
            if not _intersection.any():  
                return False

            return _intersection
        #check2 = time.time()
        #print("FR costs,", check2 - check1)
        # visualize_single_object(intersection,np.shape(intersection))

        # exit(-1)
    return intersection
    # ============================================================

def read_binvox_file(filepath, container_size):
    with open(filepath, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)  # read binvox data as a 3D array
    data = model.data   
    ready_data = pre_normalised_object(data,container_size)
    
    return ready_data# this is a bool array

def get_bounding_box(shapes):

    coords = np.argwhere(shapes == 1)
    
    if coords.shape[0] == 0:  
        return 0, 0, 0
    
    min_x, min_y, min_z = np.min(coords, axis=0)
    max_x, max_y, max_z = np.max(coords, axis=0)

    delta_x = max_x - min_x + 1
    delta_y = max_y - min_y + 1
    delta_z = max_z - min_z + 1
    
    return delta_x, delta_y, delta_z


def normalised(data):
    # Move the reference point of the object to (0,0,0) 
    # reference is the bottom backc voxel of a voxelised object
    
    x,y,z = np.where(data == 1)
    x_normal = np.min(x)
    y_normal = np.min(y)
    z_normal = np.min(z)

    data = translate_voxel_quick(data,(-x_normal,-y_normal,-z_normal))

    return data 
import numpy as np

def translate_voxel_quick(arr, shift):
    """
    Translate a 3D binary array without wrap-around.
    
    Parameters:
        arr (np.ndarray): 3D binary array (0/1).
        shift (tuple of int): (dx, dy, dz), translation offsets.
    
    Returns:
        np.ndarray: Translated array with out-of-bound parts truncated.
    """
    assert arr.ndim == 3, "Input must be a 3D array."
    x, y, z = arr.shape
    dx, dy, dz = shift

    result = np.zeros_like(arr)

    z_src_start = max(0, -dz)
    z_src_end   = min(z, z - dz)
    y_src_start = max(0, -dy)
    y_src_end   = min(y, y - dy)
    x_src_start = max(0, -dx)
    x_src_end   = min(x, x - dx)
    
    z_dst_start = max(0, dz)
    z_dst_end   = min(z, z + dz)
    y_dst_start = max(0, dy)
    y_dst_end   = min(y, y + dy)
    x_dst_start = max(0, dx)
    x_dst_end   = min(x, x + dx)

    result[x_dst_start:x_dst_end,
           y_dst_start:y_dst_end,
           z_dst_start:z_dst_end] = arr[x_src_start:x_src_end,
                                        y_src_start:y_src_end,
                                        z_src_start:z_src_end]
    return result


def translate_voxel_roll(voxel_data, steps):
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
            
        rotated_data = np.rot90(voxel_data, int(angle/90), axes=_axes)
        rotated_data = normalised(rotated_data)
        
        return rotated_data
   
def add_more_space(data,space_length):
    # add space_length in three dimensions
    # it defines the size of container 
    padded_data = np.pad(data, pad_width=space_length, mode='constant', constant_values=0)
    return padded_data


def get_max_translation(shapes):   
    # calculate the max(x,y,z) of a shape can move
    
    x,y,z = np.where (shapes == 1)
    
    max_x = len(shapes[0]) 
    max_y = len(shapes[1]) 
    max_z = len(shapes[2]) 
    
    delta_x = max_x - np.max(x) 
    delta_y = max_y - np.max(y) 
    delta_z = max_z - np.max(z)
    
    return delta_x,delta_y,delta_z

def pre_normalised_object(data, dimension):
    
    # Need to translate to the original point, not default anymore
    data = normalised(data)

    pad_x_before, pad_x_after = 0, max(dimension[0] - data.shape[0], 0)
    pad_y_before, pad_y_after = 0, max(dimension[1] - data.shape[1], 0)
    pad_z_before, pad_z_after = 0, max(dimension[2] - data.shape[2], 0)

    # extend the insufficient dimension
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
 

# def pre_get_nfv(fixed, ongoing):
#     # nfv is unavailable region
    
#     # container_size = (66,66,66)
    
#     x1,y1,z1 = get_bounding_box(fixed) 
#     x2,y2,z2 = get_bounding_box(ongoing)
    
#     dimension = (x1+2*x2,
#                  y1+2*y2,
#                  z1+2*z2)
    
#     # print(dimension)
    
    
#     normal_fixed = pre_normalised_object(fixed,dimension)
#     normal_ongoing = pre_normalised_object(ongoing,dimension)
    
#     translated_fixed = translate_voxel(normal_fixed,(x2,y2,z2))

#     NFV = np.zeros(dimension, dtype=int)
    
#     for each_x in range(x2+x1):
#         for each_y in range(y2+y1):
#             for each_z in range(z2+z1):
#                 # test any voxels in the FEASIBLE AREA!!! 
#                 # CANNOT MOVE OUT OF THE RANGE OF ARRAY
                
#                 translated_ongoing = translate_voxel(normal_ongoing,(each_x,each_y,each_z))
                
#                 test = translated_fixed + translated_ongoing
                
#                 # test if there is any voxel > 1.5, which is 2                 
#                 if np.any(test >= 1.5) == False:
#                     # if any voxels are overlapped 
                    
#                     NFV[each_x,each_y,each_z] += 1

                    
#                 # else: 
#                 #     print(f"Shapes are overlapped when it move to ({each_x},{each_y},{each_z})")
                     
#     return NFV

def get_max_translation(shapes):   
    # calculate the max(x,y,z) of a shape can move
    
    x,y,z = np.where (shapes == 1)
    
    max_x = len(shapes[0]) 
    max_y = len(shapes[1]) 
    max_z = len(shapes[2]) 
    
    delta_x = max_x - np.max(x) 
    delta_y = max_y - np.max(y) 
    delta_z = max_z - np.max(z)
    
    return delta_x,delta_y,delta_z

def get_bounding_box_np(shapes: np.ndarray):
    """
   get the bounding box, quicker way
    """
    xs, ys, zs = np.where(shapes)       
    if xs.size == 0:                    
        return 0, 0, 0
    return (xs.max() - xs.min() + 1,
            ys.max() - ys.min() + 1,
            zs.max() - zs.min() + 1)

@njit
def get_bounding_box_njit_translation(shapes):
    """
   get the bounding box 
    """
    x_vals, y_vals, z_vals = np.nonzero(shapes)  
    
    max_x = max(x_vals)
    max_y = max(y_vals)
    max_z = max(z_vals)

    return max_x + 1, max_y + 1, max_z  + 1

# @njit
# def get_bounding_box_njit_itself(shapes):
#     """
#    get the bounding box 
#     """
#     x_vals, y_vals, z_vals = np.nonzero(shapes)  

#     if len(x_vals) == 0: 
#         return 0, 0, 0
    
#     min_x, max_x = min(x_vals), max(x_vals)
#     min_y, max_y = min(y_vals), max(y_vals)
#     min_z, max_z = min(z_vals), max(z_vals)

#     return max_x - min_x  + 1, max_y - min_y + 1, max_z - min_z + 1

def get_max_translation_nfv(shapes, container_size):
    """
    max allowable translation in the container
    """
 
    x_vals, y_vals, z_vals = np.nonzero(shapes["array"]) 

    # if len(x_vals) == 0:
    #     return 0, 0, 0
    
    max_x_shape, max_y_shape, max_z_shape = x_vals[0], y_vals[0], z_vals[0]

    # for i in range(len(x_vals)):
    #     if x_vals[i] > max_x_shape:
    #         max_x_shape = x_vals[i]
    #     if y_vals[i] > max_y_shape:
    #         max_y_shape = y_vals[i]
    #     if z_vals[i] > max_z_shape:
    #         max_z_shape = z_vals[i]

    # aabb = shapes["aabb"]
    # trans = shapes["translation"]

    delta_x = container_size[0] - max_x_shape
    delta_y = container_size[1] - max_y_shape
    delta_z = container_size[2] - max_z_shape

    return delta_x, delta_y, delta_z

@njit
def get_max_translation_njit(shapes):
    """
    max allowable translation in the container
    """
    max_x, max_y, max_z = shapes.shape 
    x_vals, y_vals, z_vals = np.nonzero(shapes) 

    if len(x_vals) == 0:
        return 0, 0, 0
    
    max_x_shape, max_y_shape, max_z_shape = x_vals[0], y_vals[0], z_vals[0]

    for i in range(len(x_vals)):
        if x_vals[i] > max_x_shape:
            max_x_shape = x_vals[i]
        if y_vals[i] > max_y_shape:
            max_y_shape = y_vals[i]
        if z_vals[i] > max_z_shape:
            max_z_shape = z_vals[i]

    delta_x = max_x - max_x_shape
    delta_y = max_y - max_y_shape 
    delta_z = max_z - max_z_shape 

    return delta_x, delta_y, delta_z


def get_nfv_gpu(fixed, ongoing, dimension, calculations):
    """_summary_

    Args:
        fixed (_type_): fixed object info
        ongoing (_type_): ongoing object info

        dimension (_type_): NFV array size, nothing about the bin size
        calculations (_type_): this indicates the calculation region of NFV

    Returns:
        _type_: _description_
    """
    # L_bc, W_bc, H_bc = get_bounding_box_njit_translation(fixed)
    # max_x, max_y, max_z = get_max_translation_njit(ongoing) 
    
    # range_x = min(max_x, L_bc)
    # range_y = min(max_y, W_bc)
    # range_z = min(max_z, H_bc)

    range_x = calculations[0]
    range_y = calculations[1]
    range_z = calculations[2]

    fixed = np.ascontiguousarray(fixed) 
    fixed_gpu = cuda.to_device(fixed)
    
    
    # ongoing_gpu = cuda.to_device(ongoing)
    
    NFP = np.zeros(dimension, dtype=np.int32)
    NFP_gpu = cuda.to_device(NFP)
    
    # NFP_gpu = cuda.device_array(dimension, dtype=np.int32)
    # threads_per_block = (8, 8, 4)  # 256 THREADS
    
    # blocks_per_grid = (
    #     (range_x + threads_per_block[0] - 1) // threads_per_block[0],
    #     (range_y + threads_per_block[1] - 1) // threads_per_block[1],
    #     (range_z + threads_per_block[2] - 1) // threads_per_block[2]
    # )
    
    # threads_per_block = (8, 8, 8) # block 8x8x8 thread
    
    # blocks_per_grid = (
    #                 (range_x + 7) // 8,
    #                 (range_y + 7) // 8,
    #                 (range_z + 7) // 8
    # )
    
    threads_per_block = (8, 8, 8)
    
    blocks_per_grid = (
        max((range_x + threads_per_block[0] - 1) // threads_per_block[0], 8),
        max((range_y + threads_per_block[1] - 1) // threads_per_block[1], 8),
        max((range_z + threads_per_block[2] - 1) // threads_per_block[2], 4)
    )
    
    num_voxels_ongoing = np.count_nonzero(ongoing)

    ongoing_indices = np.column_stack(np.nonzero(ongoing)).astype(np.int32)
    # ongoing_indices = np.ascontiguousarray(ongoing_indices) 
    
    ongoing_gpu = cuda.to_device(ongoing_indices)
    
    get_nfv_cuda[blocks_per_grid, threads_per_block](fixed_gpu, ongoing_gpu, NFP_gpu,
                                                     range_x, range_y, range_z, num_voxels_ongoing)
    
    cuda.synchronize() 
    
    return NFP_gpu.copy_to_host()
    

@cuda.jit
def get_nfv_cuda(fixed, ongoing, NFP_gpu, range_x, range_y, range_z, num_voxels):
    
    """
    get nfv by cuda (gpu)
    """
    
    x_shift, y_shift, z_shift = cuda.grid(3)

    
    if x_shift >= range_x or y_shift >= range_y or z_shift >= range_z:
       # print(x_shift, y_shift, z_shift)
        return
    
    shape_x, shape_y, shape_z = fixed.shape  
    
    # calculate translation
    overlap = False
    
    for i in range(num_voxels):
        vx = ongoing[i, 0] + x_shift
        vy = ongoing[i, 1] + y_shift
        vz = ongoing[i, 2] + z_shift
        
        if vx >= shape_x or vy >= shape_y or vz >= shape_z:
            continue
        
        if fixed[vx, vy, vz] == 1:
            overlap = True
            break

    if overlap:
        NFP_gpu[x_shift, y_shift, z_shift] = 1


# def pre_get_nfv_gpu(fixed, ongoing, container_size):
#     # =============================================
#     # x1,y1,z1 = get_bounding_box(fixed) 
#     # x2,y2,z2 = get_bounding_box(ongoing)
    
#     # dimension is for ensuring the size of array


#     L_f, W_f, H_f = get_bounding_box_njit_translation(fixed)
#     L_m, W_m, H_m = get_bounding_box_njit_translation(ongoing)

#     dimension = (L_f + 2 * L_m,
#                  W_f + 2 * W_m,
#                  H_f + 2 * H_m)

#     normal_fixed = pre_normalised_object(fixed,dimension)
#     normal_ongoing = pre_normalised_object(ongoing,dimension)
    
#     translated_fixed = translate_voxel(normal_fixed,(L_m, W_m, H_m))
    
#     range_x = L_f + L_m
#     range_y = W_f + W_m
#     range_z = H_f + H_m
    
#     fixed = np.ascontiguousarray(normal_fixed) 
#     fixed_gpu = cuda.to_device(fixed)
    
#     NFV = np.zeros(dimension, dtype=int)
#     NFP_gpu = cuda.to_device(NFP)
    
#     threads_per_block = (8, 8, 4)
    
#     blocks_per_grid = (
#         max((range_x + threads_per_block[0] - 1) // threads_per_block[0], 8),
#         max((range_y + threads_per_block[1] - 1) // threads_per_block[1], 8),
#         max((range_z + threads_per_block[2] - 1) // threads_per_block[2], 4)
#     )
    
#     num_voxels_ongoing = np.count_nonzero(normal_ongoing)

#     ongoing_indices = np.column_stack(np.nonzero(normal_ongoing)).astype(np.int32)
    
#     ongoing_gpu = cuda.to_device(ongoing_indices)
    
#     get_nfv_cuda[blocks_per_grid, threads_per_block](fixed_gpu, ongoing_gpu, NFP_gpu,
#                                                      range_x, range_y, range_z, num_voxels_ongoing)
    
#     cuda.synchronize() 
    
#     NFP = NFP_gpu.copy_to_host()
    
#     return NFP
    

@cuda.jit
def get_nfv_cuda(fixed, ongoing, NFP_gpu, range_x, range_y, range_z, num_voxels):
    
    """
    get nfv by cuda (gpu)
    """
    
    x_shift, y_shift, z_shift = cuda.grid(3)

    
    if x_shift >= range_x or y_shift >= range_y or z_shift >= range_z:
       # print(x_shift, y_shift, z_shift)
        return
    
    shape_x, shape_y, shape_z = fixed.shape  
    
    # calculate translation
    overlap = False
    
    for i in range(num_voxels):
        vx = ongoing[i, 0] + x_shift
        vy = ongoing[i, 1] + y_shift
        vz = ongoing[i, 2] + z_shift
        
        if vx >= shape_x or vy >= shape_y or vz >= shape_z:
            continue
        
        if fixed[vx, vy, vz] == 1:
            overlap = True
            break

    if overlap:
        NFP_gpu[x_shift, y_shift, z_shift] = 1


# def vertical_only_accessibility_check(next_piece, bin_position, topos_layout, container_size):
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
#         next_piece = translate_voxel_quick(next_piece,(0,0,test_step))
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

def save_voxel_model(voxel_data, container_size, container_shape, result_list, pieces_order, save_path, save_type):
    
    num_containers = len(voxel_data)
    n_pieces = max(max(row) for row in pieces_order) + 1
    color_list = get_n_colors(n_pieces, colorscale='Turbo')
    legend_added_set = set()
    
    max_cols = 3
    rows = math.ceil(num_containers / max_cols)
    cols = min(num_containers, max_cols)
    
    fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'scatter3d'}] * cols for _ in range(rows)])
    
    annotations = []
    
    # print(np.shape(voxel_data))
    which_piece = 0
    
    for i, each_container in enumerate(voxel_data):
        
        # color = f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
        
        row = i // max_cols + 1
        col = i % max_cols + 1
        
        which_piece_this_bin = 0
        for each_shape in each_container:
            
            x,y,z = np.where(each_shape == 1)
            
            
            # Use Scatter3D
            # fig.add_trace(
            #     go.Scatter3D(x=x, y=y, z=z,
            #                 mode='markers',
            #                 marker=dict(
            #                     size=5,              # set the size of voxel
            #                     symbol='square',
            #                     color=f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})",        # set the color
            #                     opacity=0.7          # transparence
            #             )
            #         ),
            #         row = row, col = col
            #     )
            
            # transfer to mesh
            xv, yv, zv, _i, _j, _k = voxel_to_mesh3d(x, y, z)
            
            piece_id = pieces_order[i][which_piece_this_bin]
            color = color_list[piece_id]
            
            show_legend = piece_id not in legend_added_set
            
            if show_legend:
                legend_added_set.add(piece_id)
                
            fig.add_trace(
                go.Mesh3d(
                    x=xv, y=yv, z=zv,
                    i=_i, j=_j, k=_k,
                    opacity=0.6,
                    color=color,
                    name = f"piece {which_piece}",
                    showlegend=show_legend,
                    lighting=dict(ambient=0.2, diffuse=0.8, roughness=0.5, specular=0.6),
                    lightposition=dict(x=100, y=200, z=0)
                ),
                row=row, col=col
            )
            
            which_piece_this_bin += 1
            which_piece += 1
            
        if container_shape == "cylinder":
            # add clinder bondary of container
            
            cylinder_radius = container_size[0]/2
            cylinder_height = container_size[2]
            
            center_x = container_size[0] / 2
            center_y = container_size[0] / 2
            
            theta = np.linspace(0, 2 * np.pi, 100)  
            x_side = np.concatenate([cylinder_radius * np.cos(theta) + center_x, cylinder_radius * np.cos(theta) + center_x])
            y_side = np.concatenate([cylinder_radius * np.sin(theta) + center_y, cylinder_radius * np.sin(theta) + center_y])
            z_side = np.concatenate([np.zeros_like(theta), np.ones_like(theta) * cylinder_height])

            faces = []
            num_points = len(theta)
            for j in range(num_points - 1):
                faces.extend([
                    [j, j + 1, j + num_points], 
                    [j + 1, j + num_points + 1, j + num_points]
                ])
            
            faces = np.array(faces).T  
            
            fig.add_trace(
                go.Mesh3d(
                    x=x_side, y=y_side, z=z_side,
                    i=faces[0], j=faces[1], k=faces[2],
                    color="rgba(0, 0, 255, 0.3)", 
                    opacity=0.3
                ),
                row=row, col=col
            )
            
        # For adding numeric data
        labels = ["BinsUsed", "U", "U_star", "Time"]
        info_text = '<br>'.join([f"{label}: {value:.5f}" if isinstance(value, float) else f"{label}: {value}"
                                for label, value in zip(labels, result_list)])
        
        if i == len(voxel_data) - 1:
            
            annotations.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=(col - 1 + 0.95) / cols,  
                    y=(rows - row + 0.05) / rows,
                    xanchor='right',
                    yanchor='bottom',
                    text=info_text,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            )
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_scenes(
                dict(
                    aspectmode='cube',
                    xaxis=dict(nticks=10, range=[0, container_size[0]]),
                    yaxis=dict(nticks=10, range=[0, container_size[1]]),
                    zaxis=dict(nticks=10, range=[0, container_size[2]])
                ), row=i, col=j
            )
            
    # for lighting and camera       
    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title_text="3D Packing Results",
        annotations=annotations
        )
    
    
    try:
        if save_type == "html":
            fig.write_html(save_path + ".html")
            
        elif save_type in ["png", "jpg", "pdf", "svg"]:
            subplot_width = 800
            subplot_height = 700
            width = subplot_width * cols
            height = subplot_height * rows
            fig.write_image(save_path + f".{save_type}", width=width, height=height)

        else:
            raise ValueError("Unsupported save_type. Use 'html', 'png', etc.")
        
    except Exception as e:
        print(f"Failed to save {save_path}: {e}")
        



# def quick_nfv_njit(position_bin, topos_layout, ongoing, container_size, container_shape, _accessible_check):
#     # _accessible_check = True
    
#     fr = get_feasible_boundary(topos_layout[position_bin], ongoing, container_size, container_shape, _accessible_check)
    
#     if type(fr) == bool: 
#         # if can't get a nfv
#         # return None directly
#         return False
    
#     else:        
#         return find_best_voxel(fr)

# @njit
# def find_best_voxel(fr):

#     for z in range(fr.shape[2]):
#         for y in range(fr.shape[1]):
#             for x in range(fr.shape[0]):
#                 if fr[x, y, z] == 1:
#                     return (x,y,z)

# def pre_get_nfv(fixed, ongoing):
#     # nfv is unavailable region
    
#     # container_size = (66,66,66)
    
#     x1,y1,z1 = get_bounding_box(fixed) 
#     x2,y2,z2 = get_bounding_box(ongoing)
    
#     dimension = (x1+2*x2,
#                  y1+2*y2,
#                  z1+2*z2)
    
#     normal_fixed = pre_normalised_object(fixed,dimension)
#     normal_ongoing = pre_normalised_object(ongoing,dimension)
    
#     translated_fixed = translate_voxel(normal_fixed,(x2,y2,z2))

#     NFV = np.zeros(dimension, dtype=int)
    
#     for each_x in range(x2+x1):
#         for each_y in range(y2+y1):
#             for each_z in range(z2+z1):
#                 # test any voxels in the FEASIBLE AREA!!! 
#                 # CANNOT MOVE OUT OF THE RANGE OF ARRAY
                
#                 translated_ongoing = translate_voxel(normal_ongoing,(each_x,each_y,each_z))
                
#                 test = translated_fixed + translated_ongoing
                
#                 # test if there is any voxel > 1.5, which is 2                 
#                 if np.any(test >= 1.5) == False:
#                     # if any voxels are overlapped 
                    
#                     NFV[each_x,each_y,each_z] += 1

                    
#                 # else: 
#                 #     print(f"Shapes are overlapped when it move to ({each_x},{each_y},{each_z})")
                     
#     return NFV


def visualize_single_object(voxel_data, container_size):
    
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


def aabb_rotate(original_aabb, new_orien):
    """
    Generate new aabb after rotation
    """
    new_axis, new_degree = new_orien.split("_") 

    if int(new_degree) == 180 or int(new_degree) == 0: 

        return original_aabb
    
    else: 

        if new_axis == "x": 
            # print("reading",(original_aabb[0],original_aabb[2],original_aabb[1]))
            # exit(-1)
            # print((x2,y2,z2))
            return (original_aabb[0],original_aabb[2],original_aabb[1])
        
        elif new_axis == "y": 
            # print((original_aabb[2],original_aabb[1],original_aabb[0]))
            # exit(-1)
            return (original_aabb[2],original_aabb[1],original_aabb[0])
        
        elif new_axis == "z": 
            # print((original_aabb[1],original_aabb[0],original_aabb[2]))
            # exit(-1)
            return (original_aabb[1],original_aabb[0],original_aabb[2])
        
        else:
            raise ValueError