"""Exercise actual controller bodies and shared packing without CUDA imports."""
import ast
import copy
import math
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from radio_bin_priority import with_radio_bin_priority
from improved_GRASP_weight_radio import GlobalRankedPacking


def load_controllers():
    source=Path(__file__).resolve().parents[1]/'src/new_ILS_southampton.py'
    names={'trace','get_final_performance','get_aabb_length','update_data','prune_data_pool',
           'improve_pieces_selection_ls','improved_ILS','improved_GRASP','improved_GRASP_ILS'}
    nodes=[n for n in ast.parse(source.read_text(encoding='utf-8-sig')).body
           if isinstance(n,ast.FunctionDef) and n.name in names]
    scope=dict(np=np,copy=copy,math=math,time=time,TRACE=False)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    return types.SimpleNamespace(**scope)


def translate(array,position):
    result=np.zeros_like(array)
    points=np.argwhere(array!=0)+np.asarray(position)
    result[tuple(points.T)]=1
    return result


def geometry(info,state,b,mode,i):
    free=np.argwhere(state['occupancy'][b]==0)
    return False if not len(free) else (1.,tuple(free[0]))


class SharedControllerTests(unittest.TestCase):
    def test_empty_bin_real_pes_and_rcl_for_cube_and_cylinder(self):
        from test_sch_bounds import load_functions
        from improved_GRASP_weight_radio import _geometry_search
        namespace=load_functions('SCH_iter_ls.py',{'nesting_evaluation','get_bounding_box',
            'find_max_xyz','get_intersection','select_pes_rcl','score_position_pes',
            'selection_process_bottom_top_extrema'})
        sch=types.SimpleNamespace(**namespace)
        packing=types.SimpleNamespace(SC_heuristic=lambda *a,**kw:None)
        a=np.zeros((6,6,6),dtype=np.uint8);a[:2,:2,:2]=1
        region=np.zeros_like(a);region[1:4,1:4,:5]=1
        pool=types.SimpleNamespace(retrieve_ifv=lambda *args:region)
        for shape in ('cube','cylinder'):
            for strategy in ('minimum_aabb_volume','minimum_aabb_edges_len','maximal_residual_box','overlap_distance'):
                with self.subTest(shape=shape,strategy=strategy):
                    events=[]
                    with patch.dict(sys.modules,packing_iter_ls=packing,SCH_iter_ls=sch):
                        search=_geometry_search(None,pool,(6,6,6),shape,strategy,'bounding_box',
                            'all',True,False,.15,np.random.default_rng(1),events.append)
                        value,position=search(dict(array=a,orientation='x_0'),
                            dict(layout=[[]],occupancy=[a*0]),0,'CA',0)
                    self.assertTrue(np.isfinite(value))
                    self.assertTrue(1<=len(events[0]['candidates'])<=16)
                    self.assertLessEqual(value,events[0]['best_pes']+.15*abs(events[0]['best_pes'])+1e-10)
                    self.assertTrue(all(0<=v<=4 for v in position))
                    if shape=='cylinder':self.assertEqual(region[position],1)

    def test_empty_random_ca_also_evaluates_extreme_positions(self):
        from improved_GRASP_weight_radio import _geometry_search
        calls=[]
        def score(*args,**kwargs):
            calls.append(kwargs)
            return 1.,(0,0,0)
        packing=types.SimpleNamespace(SC_heuristic=lambda *a,**kw:None)
        sch=types.SimpleNamespace(score_position_pes=score)
        a=np.zeros((4,4,4),dtype=np.uint8);a[0,0,0]=1
        with patch.dict(sys.modules,packing_iter_ls=packing,SCH_iter_ls=sch):
            search=_geometry_search(None,None,(4,4,4),'cube','minimum_aabb_volume',
                                    'bounding_box','all',True,False,.15,np.random.default_rng(1),None)
            found=search(dict(array=a,orientation='x_0'),dict(layout=[[]],occupancy=[a*0]),0,'CA',0)
        self.assertEqual(found,(1.,(0,0,0)))
        self.assertEqual(len(calls[0]['positions']),8)
        self.assertIn((3,3,3),calls[0]['positions'])
        self.assertEqual(calls[0]['rcl_tolerance'],.15)

    def test_real_wrapped_controllers_geometry_only_and_constrained(self):
        alg=load_controllers()
        packing=types.SimpleNamespace(rotate_voxel=lambda a,d,ax:a.copy(),translate_voxel=translate)
        a=np.zeros((4,1,1),dtype=np.uint8);a[0,0,0]=1
        objects=[dict(array=a.copy(),volume=1,radio=1,piece_type=i,
                      orientation='x_0',translation=(0,0,0),bin_position=None) for i in range(3)]
        common=dict(object_info_total=objects,nfv_pool=None,ifv_pool=None,max_radio=10,rho=1,
            orientations=([0,90],['x']),ils_orientations=([0,90],['x']),
            orientations_list=['x_0','x_90'],packing_alg='SCH',selection_type='bounding_box',
            selection_range='bottom_top',accessible_check=False,SCH_nesting_strategy='minimum_aabb_volume',
            orien_evaluation=None,container_size=(4,1,1),container_shape='cube',
            flag_NFV_POOL=False,visualisation=False,_TRACE=False,time_limit=1000)
        options={
            'improved_ILS':dict(ALG='ILS',iteration_limit=4,kick_trigger_time=1,kick_level='small'),
            'improved_GRASP':dict(iter_limit_per_iter=3,alpha=1,restart_limit=1),
            'improved_GRASP_ILS':dict(iter_limit_per_iter=2,alpha=1,restart_limit=1,
                                     kick_trigger_time=1,kick_level='small',kick_seed=303)}
        for name,extra in options.items():
            for radio in (False,True):
                with self.subTest(algorithm=name,radio_priority=radio):
                    controller=with_radio_bin_priority(getattr(alg,name))
                    events=[];calls=[]
                    with patch.dict(sys.modules,packing_iter_ls=packing), \
                         patch('improved_GRASP_weight_radio._geometry_search',return_value=geometry):
                        result=controller(**common,**extra,max_weight=2,radio_priority=radio,
                            cache_observer=lambda **e:events.append(e),packing_observer=lambda k,v:calls.append(k))
                    self.assertEqual(result[0],2)
                    self.assertEqual(sorted(i for ids in result[9] for i in ids),[0,1,2])
                    self.assertEqual(calls.count('construction'),1)
                    self.assertGreater(calls.count('ls'),0)
                    if name!='improved_ILS':self.assertTrue(any(e['event']=='ORIENTATION_RCL' for e in events))
                    if name!='improved_GRASP':self.assertGreater(calls.count('kick'),0)

    def test_fixed_work_exactly_50_with_actual_controller(self):
        alg=load_controllers()
        packing=types.SimpleNamespace(rotate_voxel=lambda a,d,ax:a.copy(),translate_voxel=translate)
        a=np.zeros((4,1,1),dtype=np.uint8);a[0,0,0]=1
        objects=[dict(array=a.copy(),volume=1,radio=1,piece_type=i) for i in range(3)]
        calls=[]
        with patch.dict(sys.modules,packing_iter_ls=packing), \
             patch('improved_GRASP_weight_radio._geometry_search',return_value=geometry):
            result=with_radio_bin_priority(alg.improved_GRASP)(objects,None,None,10,1,
                ([0,90],['x']),([0,90],['x']),['x_0','x_90'],'SCH','bounding_box','bottom_top',False,
                'minimum_aabb_volume',None,(4,1,1),'cube',50,float('inf'),1,False,False,False,
                no_improvement_limit=50,ls_evaluation_limit=50,restart_limit=1,radio_priority=False,
                max_weight=2,packing_observer=lambda k,v:calls.append(k))
        self.assertEqual(calls.count('construction'),1)
        self.assertEqual(calls.count('ls'),50)
        self.assertEqual(calls.count('kick'),0)
        self.assertEqual(result[0],2)

    def test_stochastic_cache_keys_change_but_deterministic_ls_reuses(self):
        counter=[0];calls=[]
        a=np.ones((1,1,1),dtype=np.uint8)
        engine=GlobalRankedPacking([dict(array=a,radio=1,volume=1)],(1,1,1),1,10,1,1,1,
            lambda *a:(calls.append(1) or (1.,(0,0,0))),lambda a,*args:a.copy(),lambda a,p:a.copy(),
            dict(CA=('ca',),LS=('ls',)),random_state_key=lambda:counter[0])
        state=engine.empty_state(1)
        engine.query(state,0,'x_0',0,'CA');counter[0]+=1
        engine.query(state,0,'x_0',0,'CA')
        self.assertEqual(len(calls),2)
        engine.query(state,0,'x_0',0,'LS');counter[0]+=1
        engine.query(state,0,'x_0',0,'LS')
        self.assertEqual(len(calls),3)


if __name__=='__main__':unittest.main()
