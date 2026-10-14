import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from improved_ILS_weight_radio import RankedRepacker, Solution, improved_ILS_weight_radio


class FrameworkTests(unittest.TestCase):
    def setup_case(self):
        objects=[]
        for i,v in enumerate([2,4,1,1]):
            array=np.zeros((10,1,1));array[:v,0,0]=1
            objects.append(dict(array=array,volume=v,radio=1,piece_type=i,orientation='x_0'))
        order=[[0],[1],[2,3]]
        layout=[]; occupancy=[]
        for b,ids in enumerate(order):
            group=[];cube=np.zeros((10,1,1));offset=0
            for i in ids:
                info=dict(objects[i],array=np.roll(objects[i]['array'],offset,axis=0),translation=(offset,0,0),bin_position=b)
                group.append(info);cube+=info['array'];offset+=objects[i]['volume']
            layout.append(group);occupancy.append(cube)
        return objects,Solution(layout,occupancy,[[1],[1],[1,1]],order)

    def engine(self,objects,events,search=None,max_weight=10):
        def geometry(info,layout,occupancy,b):
            if info['piece_type']==2 and b<2: return False
            offset=sum(p['volume'] for p in layout[b])
            return (0,(offset,0,0)) if offset+info['volume']<=10 else False
        return RankedRepacker(objects,(10,1,1),10,10,max_weight,search or geometry,
            lambda a,d,ax:a.copy(),lambda a,pos:np.roll(a,pos[0],axis=0),event=lambda **e:events.append(e))

    def test_rank_and_search_untried_unchanged_earlier_bin(self):
        objects,state=self.setup_case();events=[];r=self.engine(objects,events)
        result=r.repack(state,2,['x_0','x_0','x_90','x_0'])
        rank=next(e for e in events if e['event']=='BIN_RANK' and e['piece']==3)
        self.assertEqual(rank['ranked'][0][0],1)
        self.assertTrue(any(e['event']=='SEARCH' and e['piece']==3 and e['bin']==2 and not e['changed'] for e in events))
        self.assertIn(3,result.order[1])

    def test_cached_failure_reused_then_invalidated_by_orientation(self):
        objects,state=self.setup_case();events=[];r=self.engine(objects,events)
        values=['x_0','x_0','x_90','x_0']
        r.repack(state,2,values);events.clear();r.repack(state,2,values)
        self.assertTrue(any(e['event']=='SKIP_CACHED_FAILURE' and e['piece']==2 for e in events))
        events.clear();r.repack(state,2,['x_0','x_0','y_90','x_0'])
        self.assertTrue(any(e['event']=='SEARCH' and e['piece']==2 and e['bin']==2 for e in events))

    def test_ca_failure_seeds_first_repack_and_changed_prefix_invalidates(self):
        objects,state=self.setup_case();events=[];r=self.engine(objects,events)
        info=dict(objects[2],orientation='x_90')
        r.record_ca_search(2,info,state.layout,state.order,1,False,r.config)
        self.assertEqual(len(r.cache),1)
        r.repack(state,2,['x_0','x_0','x_90','x_0'])
        self.assertTrue(any(e['event']=='SKIP_CACHED_FAILURE' and e.get('source')=='CA'
                            and e['piece']==2 and e['bin']==2 for e in events))
        state.layout[1][0]['orientation']='y_90'
        events.clear();r.repack(state,2,['x_0','y_90','x_90','x_0'])
        self.assertTrue(any(e['event']=='SEARCH' and e['piece']==2 and e['bin']==2 for e in events))

    def test_ca_ignores_success_empty_bin_and_mismatched_config(self):
        objects,state=self.setup_case();events=[];r=self.engine(objects,events)
        info=dict(objects[2],orientation='x_90')
        r.record_ca_search(2,info,state.layout,state.order,1,(0,(0,0,0)),r.config)
        r.record_ca_search(2,info,state.layout,state.order,1,False,('different',))
        r.record_ca_search(2,info,[[]],[[]],0,False,r.config)
        self.assertEqual(r.cache,{})

    def test_weight_hard_constraint(self):
        objects,state=self.setup_case();events=[];r=self.engine(objects,events,max_weight=4)
        result=r.repack(state,2,['x_0','x_0','x_90','x_0'])
        self.assertTrue(any(e['event']=='HARD_CONSTRAINT_REJECT' and e['piece']==3 and e['bin']==2 for e in events))
        r.validate(result)

    def test_prefix_change_invalidates_failed_cache(self):
        objects,state=self.setup_case();events=[];r=self.engine(objects,events)
        r.repack(state,2,['x_0','x_0','x_90','x_0'])
        state.layout[1][0]['orientation']='y_90'
        events.clear();r.repack(state,2,['x_0','y_90','x_90','x_0'])
        self.assertTrue(any(e['event']=='SEARCH' and e['piece']==2 and e['bin']==2 for e in events))

    def test_radiation_hard_constraint_and_new_bin(self):
        objects,state=self.setup_case();events=[]
        objects[1]['radio']=2;state.layout[1][0]['radio']=2
        objects[3]['radio']=9;state.layout[2][1]['radio']=9
        state.radio=[[1],[2],[1,9]]
        r=self.engine(objects,events)
        result=r.repack(state,2,['x_0','x_0','x_90','x_0'])
        self.assertTrue(any(e['event']=='HARD_CONSTRAINT_REJECT' and e['piece']==3 and e['bin']==2 for e in events))
        r.validate(result)

    def test_ils_budget_and_kick_preserve_global_best(self):
        objects,state=self.setup_case();events=[];r=self.engine(objects,events)
        best,history=improved_ILS_weight_radio(state,r,['x_0','x_90'],lambda s:0,
            iterations=5,seed=13,kick_trigger=1,selector=lambda s,rng,stage:2)
        self.assertEqual(len(history),5)
        self.assertTrue(any(h['kick'] for h in history))
        self.assertTrue(all(h['best']==0 for h in history))
        r.validate(best)


if __name__=='__main__': unittest.main(verbosity=2)
