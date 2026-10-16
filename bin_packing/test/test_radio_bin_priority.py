"""Post-placement metric and production-controller integration tests on CPU."""
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from radio_bin_priority import post_placement_score,with_radio_bin_priority
from improved_GRASP_weight_radio import GlobalRankedPacking


def geometry(info,state,b,mode,i):
    free=np.argwhere(state['occupancy'][b]==0)
    return False if len(free)==0 else (0.,tuple(free[0]))


class MetricTests(unittest.TestCase):
    def test_post_placement_can_reverse_old_order(self):
        # Old: A=.6*.3=.18 > B=.2*.8=.16; new: A=.24 < B=.32.
        self.assertAlmostEqual(post_placement_score(60,20,100,60,10,100),.24)
        self.assertAlmostEqual(post_placement_score(20,20,100,10,10,100),.32)

    def test_empty_bin_has_positive_post_fill_score(self):
        self.assertAlmostEqual(post_placement_score(0,20,100,0,10,100),.18)

    def test_both_ranked_engines_use_same_formula(self):
        from improved_ILS_weight_radio import RankedRepacker,Solution
        objects=[]
        for i,(v,r) in enumerate([(60,60),(20,10),(20,10)]):
            a=np.zeros((100,1,1),dtype=np.uint8);a[:v,0,0]=1
            objects.append(dict(array=a,volume=v,radio=r,piece_type=i,orientation='x_0'))
        events=[]
        e=GlobalRankedPacking(objects,(100,1,1),100,100,1,100,1,geometry,
            lambda a,d,ax:a,lambda a,p:np.roll(a,p[0],axis=0),{'CA':'ca','LS':'ls'},
            lambda **event:events.append(event))
        state=e.empty_state(2)
        for i in [0,1]:e.place(state,i,i,dict(objects[i],translation=(0,0,0)))
        self.assertEqual(e.ranked_bins(state,2),[1,0])
        self.assertAlmostEqual(events[-1]['ranked'][0][1],.32)
        ref=Solution([state['layout'][0],state['layout'][1],
                      [dict(objects[2],translation=(0,0,0),bin_position=2)]],
                     [state['occupancy'][0],state['occupancy'][1],objects[2]['array'].copy()],
                     [[60],[10],[10]],[[0],[1],[2]])
        events=[]
        r=RankedRepacker(objects,(100,1,1),100,100,100,
            lambda info,layout,occ,b:(0,(sum(p['volume'] for p in layout[b]),0,0)),
            lambda a,d,ax:a.copy(),lambda a,p:np.roll(a,p[0],axis=0),
            event=lambda **event:events.append(event))
        r.repack(ref,2,['x_0','x_0','x_90'])
        rank=next(x for x in events if x['event']=='BIN_RANK')
        self.assertEqual(rank['ranked'][0][0],1)
        self.assertAlmostEqual(rank['ranked'][0][1],.32)
        self.assertAlmostEqual(rank['details'][0]['filling_rate_after'],.4)


class ControllerIntegrationTests(unittest.TestCase):
    def test_all_three_controllers_use_shared_ranked_packing(self):
        from test_paper_controllers import load_controllers
        alg=load_controllers()
        a=np.zeros((4,1,1),dtype=np.uint8);a[0,0,0]=1
        objects=[dict(array=a.copy(),volume=1,radio=1,piece_type=i,
                      orientation='x_0',translation=(0,0,0),bin_position=None) for i in range(3)]
        common=dict(object_info_total=objects,nfv_pool=None,ifv_pool=None,max_radio=10,rho=1,
            orientations=([0,90],['x']),ils_orientations=([0,90],['x']),
            orientations_list=['x_0','x_90'],packing_alg='SCH',selection_type='bounding_box',
            selection_range='bottom_top',accessible_check=False,SCH_nesting_strategy='minimum_aabb_volume',
            orien_evaluation=None,container_size=(4,1,1),container_shape='cube',
            flag_NFV_POOL=False,visualisation=False,_TRACE=False)
        options={
            'improved_ILS':dict(ALG='ILS',iteration_limit=4,time_limit=1000,kick_trigger_time=1,kick_level='small'),
            'improved_GRASP':dict(iter_limit_per_iter=3,time_limit=1000,alpha=1,restart_limit=1),
            'improved_GRASP_ILS':dict(iter_limit_per_iter=2,time_limit=1000,alpha=1,restart_limit=1,
                                     kick_trigger_time=1,kick_level='small',kick_seed=303)}
        for name,extra in options.items():
            with self.subTest(name=name):
                controller=with_radio_bin_priority(inspect.unwrap(getattr(alg,name)))
                events=[];selector_flags=[]
                selector=alg.improve_pieces_selection_ls
                def tracked(*a,**kw):
                    selector_flags.append(kw['exclude_last_bin']);return selector(*a,**kw)
                from test_paper_controllers import translate
                import types
                packing=types.SimpleNamespace(rotate_voxel=lambda a,d,ax:a.copy(),translate_voxel=translate)
                with patch('improved_GRASP_weight_radio._geometry_search',return_value=geometry), \
                     patch.dict(sys.modules,packing_iter_ls=packing), \
                     patch.dict(controller.__wrapped__.__globals__,improve_pieces_selection_ls=tracked):
                    result=controller(**common,**extra,max_weight=2,
                                      cache_observer=lambda **event:events.append(event))
                self.assertEqual(result[0],2)
                self.assertTrue(any(x['event']=='BIN_RANK' for x in events))
                self.assertTrue(any(x['event']=='SEARCH' and x['mode']=='LS' for x in events))
                self.assertTrue(selector_flags and not any(selector_flags))
                self.assertEqual(events[-1]['algorithm'],name)
                self.assertEqual(events[-1]['metric'],'post_placement_fill_times_radio_left')

    def test_optional_legacy_path_does_not_create_a_ranked_engine(self):
        called=[]
        def controller(x):called.append(x);return x+1
        # The bypass must work without loading or configuring packing dependencies.
        f=with_radio_bin_priority(controller)
        self.assertEqual(f(2,radio_priority=False,cache_first_fit=False),3)
        self.assertEqual(called,[2])


if __name__=='__main__':unittest.main()
