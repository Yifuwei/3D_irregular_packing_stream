"""Complete sequence scans, first improvement, and honest neighborhood proofs."""
import ast
import copy
from pathlib import Path
import types
import unittest
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'src/new_ILS_southampton.py'


class SystematicGraspTests(unittest.TestCase):
    def search(self,move=None,budget=1000,cap=None,orientations=('x_0','x_90','x_180'),restarts=1):
        clock=[0];calls=[];ends=[]
        def packed(score,values):
            pieces=[dict(score=score,orientation=o,bin_position=0 if i<2 else 1,
                         array=np.ones((1,1,1))) for i,o in enumerate(values)]
            return [pieces[:2],pieces[2:]],[],[],[[0,1],[2]]
        def construct(*a,**kw):
            clock[0]+=1
            return packed(5,['x_0']*3)
        def repack(*a,**kw):
            state=a[5][0];selected,o=a[1];values=list(a[4]);baseline=state['score']
            calls.append((baseline,selected,o,tuple(values)))
            score=move(baseline,selected,o,values) if move else baseline
            values[selected]=o;clock[0]+=1
            # Mutation must not corrupt a rejected trial's accepted state.
            state['bin_real_layout'][0][0]['score']=-999
            return packed(score,values)
        nodes=[n for n in ast.parse(SOURCE.read_text(encoding='utf-8-sig')).body
               if isinstance(n,ast.FunctionDef) and n.name=='improved_GRASP']
        def forbidden(*a,**kw):raise AssertionError('Systematic LS must not randomly choose a piece')
        scope=dict(np=np,copy=copy,time=types.SimpleNamespace(perf_counter=lambda:clock[0]),trace=lambda m:None,
            packing_3D_voxel_lookback=construct,improved_repack=repack,improve_pieces_selection_ls=forbidden,
            get_final_performance=lambda objects,size,shape,layout:(layout[0][0]['score'],0,None))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)
        result=scope['improved_GRASP']([{}, {}, {}],None,None,0,1,None,None,list(orientations),'SCH',
            'bounding_box','bottom',True,'minimum_aabb_volume',None,(1,1,1),'cube',1,budget,1,False,False,False,
            no_improvement_limit=1,restart_limit=restarts,ls_evaluation_limit=cap,ls_observer=ends.append)
        return result,calls,ends

    def test_complete_sequence_and_all_orientations_including_last_bin(self):
        _,calls,ends=self.search()
        self.assertEqual([(p,o) for _,p,o,_ in calls],[(0,'x_90'),(0,'x_180'),(1,'x_90'),
                                                        (1,'x_180'),(2,'x_90'),(2,'x_180')])
        self.assertEqual(ends[0]['stop_reason'],'local_optimum')
        self.assertTrue(ends[0]['local_optimum'])
        self.assertEqual(ends[0]['ls_evaluations'],6) # old threshold=1 did not truncate

    def test_first_improvement_restarts_from_first_on_new_solution(self):
        def move(score,p,o,values):
            if score==5 and p==1 and o=='x_90':return 4
            if score==4 and p==0 and o=='x_90':return 3
            return score
        result,calls,ends=self.search(move)
        self.assertEqual([(s,p,o) for s,p,o,_ in calls[:4]],
                         [(5,0,'x_90'),(5,0,'x_180'),(5,1,'x_90'),(4,0,'x_90')])
        self.assertEqual(calls[3][3],('x_0','x_90','x_0'))
        self.assertEqual([(p,o) for s,p,o,_ in calls if s==3],
                         [(0,'x_0'),(0,'x_180'),(1,'x_0'),(1,'x_180'),(2,'x_90'),(2,'x_180')])
        self.assertEqual(result[0],3)
        self.assertEqual(result[3],5)
        self.assertEqual(ends[0]['stop_reason'],'local_optimum')

    def test_deadline_does_not_claim_local_optimum(self):
        _,calls,ends=self.search(budget=3)
        self.assertEqual(len(calls),2)
        self.assertEqual(ends[0]['stop_reason'],'time_limit')
        self.assertFalse(ends[0]['local_optimum'])

    def test_evaluation_cap_does_not_claim_local_optimum(self):
        _,calls,ends=self.search(cap=2)
        self.assertEqual(len(calls),2)
        self.assertEqual(ends[0]['stop_reason'],'evaluation_limit')
        self.assertFalse(ends[0]['local_optimum'])

    def test_cap_exactly_completes_neighborhood_is_a_valid_proof(self):
        _,_,ends=self.search(cap=6)
        self.assertTrue(ends[0]['local_optimum'])

    def test_empty_neighborhood_and_independent_restart_scans(self):
        _,calls,ends=self.search(orientations=('x_0',),restarts=2)
        self.assertEqual(calls,[])
        self.assertEqual(len(ends),2)
        self.assertTrue(all(e['local_optimum'] for e in ends))


if __name__=='__main__':unittest.main()
