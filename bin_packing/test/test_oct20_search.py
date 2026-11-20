"""Size-scaled complete-piece stalls and independent repacking budgets."""
import ast
import copy
import math
from pathlib import Path
import types
import unittest
import numpy as np

from test_paper_controllers import load_controllers,translate,geometry
from radio_bin_priority import with_radio_bin_priority
from unittest.mock import patch
import sys
from improved_GRASP_weight_radio import GlobalRankedPacking
from orientation_rcl import choose_orientation


class SearchCountingTests(unittest.TestCase):
    def search(self,algorithm,n=5,improve_at=None,restarts=1,limit=29,labels=None):
        alg=load_controllers()
        scope=getattr(alg,algorithm).__globals__
        labels=labels or ['x_0']+[f'{axis}_{d}' for d in (90,180,270) for axis in 'xyz']
        clock=[0.];events=[];ls=[0];kick_changes=[]
        def packed(score,orientations):
            layout=[[dict(orientation=o,score=score,bin_position=int(i>=2),array=np.ones((1,1,1)))
                     for i,o in enumerate(orientations) if (i<2)==(b==0)] for b in (0,1)]
            return layout,[np.ones((1,1,1))]*2,[[],[]],[list(range(2)),list(range(2,n))]
        def construct(*a,**kw):
            events.append(('construct',ls[0]));clock[0]+=.01
            return packed(.5,['x_0']*n)
        def repack(*a,**kw):
            ls[0]+=1;clock[0]+=.01;events.append(('ls',ls[0]))
            baseline=a[5][a[6]]
            score=baseline['bin_real_layout'][0][0]['score']
            values=list(a[4]);values[a[1][0]]=a[1][1]
            return packed(.4 if ls[0]==improve_at else score+.01,values)
        def kick(*a,**kw):
            clock[0]+=.01;events.append(('kick',ls[0]))
            kick_changes.append(sum(o!='x_0' for o in a[4]))
            return packed(.8,a[4])
        scope.update(packing_3D_voxel_lookback=construct,improved_repack=repack,kick_repacking=kick,
                     time=types.SimpleNamespace(perf_counter=lambda:clock[0]),
                     improve_pieces_selection_ls=lambda *a,**kw:0,
                     get_final_performance=lambda o,c,s,layout:(2,0,2-layout[0][0]['score']))
        common=dict(object_info_total=[{}]*n,nfv_pool=None,ifv_pool=None,max_radio=1,rho=1,
            orientations=None,ils_orientations=None,orientations_list=labels,packing_alg='SCH',
            selection_type='bounding_box',selection_range='all',accessible_check=False,
            SCH_nesting_strategy='minimum_aabb_volume',orien_evaluation=None,
            container_size=(1,1,1),container_shape='cube',kick_trigger_time='auto',
            kick_level='small',time_limit=100,flag_NFV_POOL=False,visualisation=False,_TRACE=False)
        if algorithm=='improved_ILS':common.update(ALG='ILS',iteration_limit=limit)
        else:common.update(iter_limit_per_iter=None,restart_limit=restarts,alpha=1,kick_seed=1)
        result=getattr(alg,algorithm)(**common)
        return events,kick_changes,result

    def test_ils_kick_after_three_complete_nine_orientation_failures(self):
        events,changes,_=self.search('improved_ILS')
        self.assertEqual([v for k,v in events if k=='kick'],[27])
        self.assertEqual(sum(k=='ls' for k,v in events),28)
        self.assertEqual(changes,[2]) # ceil(.25*5)

    def test_improvement_resets_piece_failure_counter(self):
        events,_,_=self.search('improved_ILS',improve_at=4,limit=33)
        self.assertEqual([v for k,v in events if k=='kick'],[31])

    def test_hybrid_9n_budget_not_reset_by_improvement_or_kick(self):
        events,_,_=self.search('improved_GRASP_ILS',improve_at=4,restarts=2)
        self.assertEqual([v for k,v in events if k=='construct'],[0,45])
        self.assertEqual(sum(k=='ls' for k,v in events),90)
        self.assertEqual([v for k,v in events if k=='kick'],[31,72])

    def test_odd_43_piece_trigger_rounds_up(self):
        events,_,_=self.search('improved_ILS',n=43,labels=['x_0','x_90'],limit=24)
        self.assertEqual([v for k,v in events if k=='kick'],[22])


class PooledRclTests(unittest.TestCase):
    def test_public_top_k_works_with_tolerance_none(self):
        alg=load_controllers()
        a=np.zeros((4,1,1),dtype=np.uint8);a[0,0,0]=1
        objects=[dict(array=a.copy(),volume=1,radio=1,piece_type=i,
                      orientation='x_0',translation=(0,0,0),bin_position=None) for i in range(3)]
        packing=types.SimpleNamespace(rotate_voxel=lambda a,d,ax:a.copy(),translate_voxel=translate)
        events=[]
        with patch.dict(sys.modules,packing_iter_ls=packing), \
             patch('improved_GRASP_weight_radio._geometry_search',return_value=geometry):
            result=with_radio_bin_priority(alg.improved_GRASP)(objects,None,None,10,1,
                ([0,90],['x']),None,['x_0','x_90'],'SCH','bounding_box','all',False,
                'minimum_aabb_volume',None,(4,1,1),'cube',time_limit=1000,
                restart_limit=1,ls_evaluation_limit=0,max_weight=2,rcl_tolerance=None,
                rcl_top_k=30,rcl_observer=events.append)
        self.assertEqual(result[0],2)
        self.assertTrue(events)
        self.assertTrue(all(event['top_k']==30 for event in events))

    def test_pool_combines_all_positions_before_one_draw_and_reuses_coordinates(self):
        class Last:
            def integers(self,n):return n-1
        events=[];searches=[]
        a=np.zeros((4,1,1),dtype=np.uint8);a[0,0,0]=1
        objects=[dict(array=a,volume=1,radio=0,piece_type=0,orientation='x_0')]
        def search(info,*args):
            searches.append(info['orientation'])
            return [(1.,(0,0,0)),(1.05,(1,0,0))] if info['orientation']=='x_0' else [
                   (1.08,(2,0,0)),(1.3,(3,0,0))]
        choose=lambda values:choose_orientation(values,.1,Last(),events.append)
        e=GlobalRankedPacking(objects,(4,1,1),4,1,1,4,1,search,lambda a,d,ax:a,
                             translate,{'CA':('ca',),'LS':('ls',)},pooled_selector=choose)
        first=e.construct(['x_0','x_90']);second=e.construct(['x_0','x_90'])
        self.assertEqual(searches,['x_0','x_90'])
        self.assertEqual(first['layout'][0][0]['translation'],(2,0,0))
        self.assertEqual(first['layout'][0][0]['orientation'],'x_90')
        self.assertEqual(events[0]['candidate_count'],4)
        self.assertEqual(events[0]['active_rcl_size'],3)
        self.assertEqual(len(events),2) # cache hits still draw from the pool
        self.assertFalse(any(isinstance(value,np.ndarray) for values in e.cache.values() for value in values))

    def test_top_k_is_global_and_clamped_to_actual_pool(self):
        class Last:
            def integers(self,n):return n-1
        values=[(float(i),('o',i)) for i in range(160)]
        for k in (10,30):self.assertEqual(choose_orientation(values,None,Last(),top_k=k)[0],k-1)
        self.assertEqual(choose_orientation(values[:3],None,Last(),top_k=30)[0],2)


if __name__=='__main__':unittest.main()
