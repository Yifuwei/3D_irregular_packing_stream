"""Compare the exact original cylinder IFV with the experimental hull method.

CPU-only. Compilation excluded from timing; full output equality is mandatory.
"""
import argparse
import ast
import csv
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import cylinder_ifv_candidate as candidate


def original_function(project, compiled=True):
    tree=ast.parse((project/'src/function_lib.py').read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('max_along_axis_2','njit_get_ifv_cylinder')]
    if len(tree.body)!=2:
        raise ValueError('Original functions not found')
    if compiled:
        from numba import njit
    else:
        njit=lambda function:function
    scope={'np':np,'njit':njit}
    exec(compile(tree,'original_function_lib.py','exec'),scope)
    return scope['njit_get_ifv_cylinder']


def verify(project):
    old=original_function(project,compiled=False)
    rng=np.random.default_rng(17)
    count=0
    for h,w in ((7,7),(8,8),(10,10),(9,11)):
        for kind in ('empty','point','line','rectangle','random'):
            a=np.zeros((h,w,5),dtype=np.int64)
            if kind=='point':a[0,0,0]=1
            elif kind=='line':a[:3,0,0]=1
            elif kind=='rectangle':a[:3,:4,:2]=1
            elif kind=='random':a[:3,:4,:2]=rng.integers(0,2,(3,4,2))
            for d in (5,8,1):
                args=(a,3,4,2,(h,w,d))
                left,right=old(*args),candidate.build_ifv(*args)
                assert left.dtype==right.dtype and np.array_equal(left,right),(h,w,kind,d)
                count+=1
    print(f'PASS: {count} complete 3-D masks match, including empty/point/line/random/expanded-height cases.')


def cases(project,dataset,seed,max_pieces,orientations):
    path=project/'instances'/f'{dataset}_seq{seed}_cylinder_999999_1.txt'
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    size=tuple(ast.literal_eval(lines[0].split('\t')[1]))
    references=sorted(set(line.split('\t')[0] for line in lines[1:] if line.strip()))
    sys.path.insert(0,str(project/'src'))
    import binvox_rw
    from scipy.ndimage import binary_fill_holes
    pieces=[]
    for reference in references:
        cleaned=reference.strip().replace('\\','/')
        if cleaned.startswith('../'):cleaned=cleaned[3:]
        with (project/'data'/cleaned).open('rb') as stream:
            data=binary_fill_holes(binvox_rw.read_as_3d_array(stream).data)
        coordinates=np.argwhere(data)
        lo,hi=coordinates.min(0),coordinates.max(0)+1
        local=data[tuple(slice(x,y) for x,y in zip(lo,hi))]
        pieces.append((int(local.sum()),reference,local))
    pieces.sort(key=lambda x:x[0])
    if len(pieces)>max_pieces:
        indices=sorted(set(np.linspace(0,len(pieces)-1,max_pieces,dtype=int)))
        pieces=[pieces[i] for i in indices]
    for _,reference,local in pieces:
        for orientation in orientations:
            axis,degree=orientation.split('_')
            rotated=np.rot90(local,int(degree)//90,axes={'x':(1,2),'y':(0,2),'z':(0,1)}[axis])
            dims=rotated.shape
            if any(a>b for a,b in zip(dims,size)):
                continue
            a=np.zeros(size,dtype=np.int64)
            a[tuple(slice(0,n) for n in dims)]=rotated
            yield reference,orientation,a,dims,size


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dataset',default='st04_Example5_normal')
    parser.add_argument('--seed',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--max-pieces',type=int,default=3)
    parser.add_argument('--orientations',nargs='+',default=['x_0','z_90'])
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('cylinder_ifv_benchmark'))
    args=parser.parse_args()
    if min(args.repeats,args.max_pieces)<1:parser.error('repeats/max-pieces must be positive')
    project=args.project.resolve()
    verify(project)
    if args.verify_only:return
    if not candidate.HAS_NUMBA:
        parser.error('Timing requires Numba for both methods; verify-only works without it')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'timings.csv').exists():parser.error('Use a fresh output directory')
    old=original_function(project,compiled=True)
    rows=[]
    all_samples=[]
    for reference,orientation,array,dims,size in cases(project,args.dataset,args.seed,args.max_pieces,args.orientations):
        for height_kind,height in [('normal',size[2]),('accessibility_extended',size[2]+dims[2]-1)]:
            container=(size[0],size[1],height)
            call=(array,*dims,container)
            # Compile and verify outside timed repeats; no output is retained.
            before,after=old(*call),candidate.build_ifv(*call)
            if not np.array_equal(before,after):
                raise AssertionError(f'Output mismatch: {reference}/{orientation}/{height_kind}')
            valid_positions=int(np.count_nonzero(before[:,:,0]))
            del before,after
            vertices=candidate.prepare_vertices(array,dims[0],dims[1])
            mask=candidate.xy_mask(vertices,size[0],size[1],dims[0],dims[1])
            samples={'original':[],'hull_uncached':[],'reuse_xy_mask':[]}
            methods={'original':lambda:old(*call),
                     'hull_uncached':lambda:candidate.build_ifv(*call),
                     # Exact repeated-geometry scenario; include full 3-D output creation.
                     'reuse_xy_mask':lambda:candidate.expand_mask(mask,dims[2],container)}
            for repeat in range(args.repeats):
                order=list(methods)
                if repeat%2:order.reverse()
                for method in order:
                    gc.collect()
                    start=time.perf_counter()
                    result=methods[method]()
                    duration=time.perf_counter()-start
                    samples[method].append(duration)
                    del result
            medians={name:statistics.median(values) for name,values in samples.items()}
            row=dict(piece=reference,orientation=orientation,height_kind=height_kind,
                     container=str(container),object_dims=str(dims),hull_vertices=len(vertices),
                     valid_xy_positions=valid_positions,repeats=args.repeats,
                     **{name+'_median_s':value for name,value in medians.items()},
                     uncached_speedup=medians['original']/medians['hull_uncached'],
                     cached_speedup=medians['original']/medians['reuse_xy_mask'])
            rows.append(row)
            all_samples.append(dict(case=row,samples=samples))
            print(json.dumps(row),flush=True)
            with (output/'timings.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row))
                writer.writeheader();writer.writerows(rows)
            (output/'samples.json').write_text(json.dumps(dict(cases=all_samples,
                note='Cache-hit time assumes previously computed XY mask; initial projection/hull/mask cost reported as hull_uncached.'),indent=2))
    if not rows:raise ValueError('No valid cases selected')
    print(f'Saved {len(rows)} timing cases to {output}')


if __name__=='__main__':main()
