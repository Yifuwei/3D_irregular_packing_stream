"""Experimental exact cylinder IFV builder; not wired into production search."""
import numpy as np
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(function):
        return function


def convex_hull(points):
    """Integer monotone chain; handles empty, singleton and collinear sets."""
    points = sorted(set(map(tuple, points)))
    if len(points) <= 1:
        return np.asarray(points, dtype=np.int64).reshape(-1, 2)
    def cross(o, a, b):
        return (int(a[0])-int(o[0]))*(int(b[1])-int(o[1]))-(int(a[1])-int(o[1]))*(int(b[0])-int(o[0]))
    lower=[]
    for point in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],point)<=0:
            lower.pop()
        lower.append(point)
    upper=[]
    for point in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],point)<=0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1]+upper[:-1],dtype=np.int64)


@njit
def xy_mask(vertices, h, w, obj_length, obj_width):
    mask=np.zeros((h,w),dtype=np.uint8)
    for i in range(max(0,h-obj_length+1)):
        for j in range(max(0,w-obj_width+1)):
            valid=True
            for k in range(len(vertices)):
                # Preserve legacy np.indices coordinate convention exactly,
                # including non-square grids. Use doubled integer coordinates.
                dx=2*(i+vertices[k,0])-w
                dy=2*(j+vertices[k,1])-h
                if dx*dx+dy*dy > h*h:
                    valid=False
                    break
            if valid:
                mask[i,j]=1
    return mask


def prepare_vertices(ongoing_array, obj_length, obj_width):
    # The legacy projection marks ==1 only, not arbitrary nonzero voxels.
    points=np.argwhere(np.any(ongoing_array==1,axis=2))
    # Legacy truncates at the grid edge for malformed bounding dimensions.
    # Reject such inputs rather than silently claiming equivalence.
    if len(points) and (points[:,0].max()>=obj_length or points[:,1].max()>=obj_width):
        raise ValueError('Projection must lie within origin-normalized bounding dimensions')
    return convex_hull(points)


def expand_mask(mask, obj_height, container_size):
    h,w,d=map(int,container_size)
    available_z=max(0,d-obj_height+1)
    result=np.zeros((h,w,d),dtype=np.int32)
    result[:,:,:available_z]=mask[:,:,None]
    return result


def build_ifv(ongoing_array,obj_length,obj_width,obj_height,container_size):
    """Same int32 3-D output as legacy; includes projection/hull/expansion cost."""
    h,w,_=map(int,container_size)
    if ongoing_array.shape[:2]!=(h,w):
        raise ValueError('XY array dimensions must match container')
    vertices=prepare_vertices(ongoing_array,obj_length,obj_width)
    return expand_mask(xy_mask(vertices,h,w,obj_length,obj_width),obj_height,container_size)
