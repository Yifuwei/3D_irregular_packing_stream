"""Real chess/CUDA smoke test using its stored radiation proxy values."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--iterations',type=int,default=3)
p.add_argument('--max-radio',type=float)
p.add_argument('--rho',type=float,default=0.8)
p.add_argument('--cache-repeat-test',action='store_true')
p.add_argument('--verify-ca-observer',action='store_true')
a=p.parse_args();project=a.project.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(project/'src'),str(project/'test')];os.chdir(project/'test')
import code_launcher as launcher
import packing_iter_ls as packing
import new_ILS_southampton as ils
from function_lib import NFV_POOL,IFV_POOL
from ca_parameters import best_ca_parameters
from improved_ILS_weight_radio import Solution,RankedRepacker,improved_ILS_weight_radio,geometry_search_config
from numba import cuda
assert cuda.is_available()
shape,size,_,_,objects=launcher.load_instance(launcher.make_instance_id('chess',13,'cube',999999,1))
volume=int(np.prod(size));weight_factor=1.0;max_weight=a.rho*volume
total_radio=sum(info['radio'] for info in objects)
# Proxy observations are unchanged. These capacity limits are explicit test choices.
max_radio=a.max_radio if a.max_radio is not None else max(max(info['radio'] for info in objects),total_radio/3)
orientations=([0,90,180,270],['x','y','z'])
values=['x_0']+[f'{ax}_{d}' for d in [90,180,270] for ax in ['x','y','z']]
strategy,selection_range=best_ca_parameters(shape)
nfv,ifv=NFV_POOL(),IFV_POOL()
np.random.seed(13)
started=time.perf_counter()
log=(out/'events.jsonl').open('w',encoding='utf-8',buffering=1)
phase='ca';repack_id=0
def event(**e):
    global repack_id
    if e['event']=='REPACK_START': repack_id+=1
    log.write(json.dumps(dict(phase=phase,repack_id=repack_id,**e))+'\n')
    if e['event']=='EVALUATION': print(json.dumps(e),flush=True)
def geometry(info,layout,occupancy,b):
    if not layout[b]: return 0,(0,0,0)
    return packing.SC_heuristic(nfv,ifv,info,layout,occupancy,b,size,strategy,5,'z',shape,
        'bounding_box',False,True,selection_range,False,False)
engine=RankedRepacker(objects,size,volume,max_radio,max_weight,geometry,
    packing.rotate_voxel,packing.translate_voxel,weight_factor=weight_factor,rho=a.rho,
    search_config=geometry_search_config(size,strategy,select_range=selection_range,shape=shape),event=event)
result=ils.improved_ILS(copy.deepcopy(objects),nfv,ifv,max_radio,a.rho,orientations,orientations,
    values,'SCH','bounding_box',selection_range,False,strategy,None,'fixed_CA',size,shape,
    visualisation=False,_TRACE=False,ca_search_observer=engine.record_ca_search)
initial=Solution(result[6],result[8],[[p['radio'] for p in group] for group in result[6]],result[9])
if a.verify_ca_observer:
    # The observer must not affect CA membership, orientation or voxel positions.
    baseline=ils.improved_ILS(copy.deepcopy(objects),NFV_POOL(),IFV_POOL(),max_radio,a.rho,
        orientations,orientations,values,'SCH','bounding_box',selection_range,False,
        strategy,None,'fixed_CA',size,shape,visualisation=False,_TRACE=False)
    assert baseline[9]==initial.order
    assert all(np.array_equal(a,b) for a,b in zip(baseline[8],initial.occupancy))
    assert all(a['orientation']==b['orientation'] and tuple(a['translation'])==tuple(b['translation'])
        for new,old in zip(initial.layout,baseline[6]) for a,b in zip(new,old))
phase='ils'
def objective(state):
    n,u,us=ils.get_final_performance(objects,size,shape,state.layout)
    return n-(u if us is None else us)
best,history=improved_ILS_weight_radio(initial,engine,values,objective,
    iterations=a.iterations,seed=13,kick_trigger=1)
engine.validate(best)
if a.cache_repeat_test:
    # Deliberately repeat an identical query to exercise matching-state cache
    # reuse. This is separate from stochastic ILS evaluations and timing claims.
    base_values=[None]*len(objects)
    for ids,group in zip(initial.order,initial.layout):
        for i,info in zip(ids,group): base_values[i]=info['orientation']
    base_values[0]=next(o for o in values if o!=base_values[0])
    phase='repeat_first';first=engine.repack(initial,0,base_values)
    phase='repeat_second';second=engine.repack(initial,0,base_values)
    assert first.order==second.order
    assert all(np.array_equal(x,y) for x,y in zip(first.occupancy,second.occupancy))
cuda.synchronize();log.close()
report=dict(dataset='chess',pieces=len(objects),seed=13,iterations=len(history),
    radiation_source='unchanged radio fields from chess_seq13_cube_999999_1.txt',
    radiation_values=[float(o['radio']) for o in objects],max_radio=max_radio,rho=a.rho,
    weight_factor=weight_factor,max_weight=max_weight,weight_units='volume proxy, not calibrated kg',
    initial_bins=len(initial.layout),best_bins=len(best.layout),
    initial_counts=[len(g) for g in initial.order],best_counts=[len(g) for g in best.order],
    best_radio=[sum(p['radio'] for p in g) for g in best.layout],
    best_weight=[sum(p['volume'] for p in g)*weight_factor for g in best.layout],
    seconds=time.perf_counter()-started,passed=True,history=history,
    ca_observer_equivalence_verified=a.verify_ca_observer,
    note='Framework smoke test, not a timing comparison. Capacities are test choices; CA uses the same hard limits.')
(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PASS',json.dumps(report),flush=True)
