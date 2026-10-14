"""CPU regression tests for ranked GRASP and exact-state cache correctness."""
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import improved_GRASP_weight_radio as module


def objects(n=3):
    a = np.zeros((4,1,1), dtype=np.uint8); a[0,0,0] = 1
    return [dict(array=a.copy(), volume=1, radio=1, piece_type=i) for i in range(n)]


def translate(a, p):
    out = np.zeros_like(a)
    points = np.argwhere(a) + np.asarray(p)
    out[tuple(points.T)] = 1
    return out


def first_free(info, state, b, mode, i):
    free = np.argwhere(state['occupancy'][b] == 0)
    return False if not len(free) else (float(free[0,0]), tuple(free[0]))


class CacheAndRankingTests(unittest.TestCase):
    def engine(self, search=first_free, n=3, radio=10, weight=4, factor=1):
        return module.GlobalRankedPacking(objects(n), (4,1,1), 4, radio, 1,
                    weight, factor, search, lambda a,d,ax:a.copy(), translate,
                    {'CA':('random',.1), 'LS':('greedy',)})

    def test_success_is_restored_without_search_or_array_aliasing(self):
        calls=[]
        def search(*a):calls.append(a);return first_free(*a)
        e=self.engine(search);s=e.empty_state(1)
        a=e.query(s,0,'x_0',0,'CA');a[1]['array'][:]=0
        b=e.query(s,0,'x_0',0,'CA')
        self.assertEqual(len(calls),1)
        self.assertEqual(np.count_nonzero(b[1]['array']),1)
        self.assertEqual(e.stats['RESTORE_CACHED_SUCCESS'],1)

    def test_failure_is_skipped_but_prefix_orientation_policy_are_distinct(self):
        calls=[]
        def failed(*a):calls.append(a);return False
        e=self.engine(failed);s=e.empty_state(1)
        e.query(s,1,'x_0',0,'CA');e.query(s,1,'x_0',0,'CA')
        self.assertEqual(len(calls),1)
        e.query(s,1,'x_90',0,'CA');e.query(s,1,'x_0',0,'LS')
        info=dict(e.objects[0],orientation='x_0',translation=(0,0,0))
        e.place(s,0,0,info);e.query(s,1,'x_0',0,'CA')
        self.assertEqual(len(calls),4)
        self.assertEqual(e.stats['SKIP_CACHED_FAILURE'],1)

    def test_cache_persists_across_construction_restarts(self):
        e=self.engine(n=2)
        a=e.construct(['x_0']);before=e.stats['SEARCH']
        b=e.construct(['x_0'])
        self.assertEqual(e.stats['SEARCH'],before)
        self.assertEqual(a['order'],b['order'])
        self.assertGreater(e.stats['RESTORE_CACHED_SUCCESS'],0)

    def test_radio_weight_capacity_and_empty_bin_rejection(self):
        for kw in [dict(radio=.5),dict(weight=.5),dict(weight=1,factor=2)]:
            with self.assertRaisesRegex(ValueError,'empty bin capacity'):self.engine(**kw)
        for kw in [dict(radio=1),dict(weight=1)]:
            e=self.engine(n=3,**kw);s=e.construct(['x_0'])
            self.assertEqual(len(s['layout']),3)
            e.validate(s)

    def test_capacity_checked_before_a_cached_success(self):
        e=self.engine(radio=1);s=e.empty_state(1)
        e.query(s,0,'x_0',0,'CA')
        s['radio'][0]=[1]
        self.assertIsNone(e.query(s,0,'x_0',0,'CA'))
        self.assertEqual(e.stats['RESTORE_CACHED_SUCCESS'],0)

    def test_ranking_uses_fill_before_and_radio_after_with_numeric_ties(self):
        e=self.engine();s=e.empty_state(3)
        e.place(s,0,1,dict(e.objects[0],orientation='x_0',translation=(0,0,0)))
        self.assertEqual(e.ranked_bins(s,2),[1,0,2])
        s['radio'][1]=[10]
        self.assertEqual(e.ranked_bins(s,2),[0,2])

    def test_changed_state_cannot_use_an_old_success(self):
        e=self.engine();s=e.empty_state(1)
        old=e.query(s,1,'x_0',0,'LS')
        e.place(s,0,0,dict(e.objects[0],orientation='x_0',translation=(0,0,0)))
        new=e.query(s,1,'x_0',0,'LS')
        self.assertEqual(old[1]['translation'],(0,0,0))
        self.assertEqual(new[1]['translation'],(1,0,0))

    def test_repack_reuses_same_queries_across_evaluations(self):
        e=self.engine(n=3,weight=2);ref=e.construct(['x_0'])
        a=e.repack(ref,0,['x_90','x_0','x_0']);before=e.stats['SEARCH']
        b=e.repack(ref,0,['x_90','x_0','x_0'])
        self.assertEqual(a['order'],b['order'])
        self.assertEqual(e.stats['SEARCH'],before)
        self.assertGreater(e.stats['RESTORE_CACHED_SUCCESS'],0)

    def test_reference_restore_requires_exact_prefix(self):
        e=self.engine(n=3,weight=2);ref=e.construct(['x_0'])
        e.repack(ref,2,['x_0']*3)
        self.assertGreater(e.stats['RESTORE_REFERENCE_SUCCESS'],0)
        ref2=copy.deepcopy(ref)
        ref2['layout'][0][0]['translation']=(2,0,0)
        # A changed prefix must be a separate cache key, even with same IDs.
        s=e.empty_state(1)
        e.query(s,2,'x_0',0,'LS')
        e.place(s,0,0,dict(e.objects[0],orientation='x_0',translation=(2,0,0),
                         array=translate(e.objects[0]['array'],(2,0,0))))
        before=e.stats['SEARCH'];e.query(s,2,'x_0',0,'LS')
        self.assertEqual(e.stats['SEARCH'],before+1)

    def test_invalid_geometry_does_not_enter_cache(self):
        e=self.engine(lambda *a:(0,(4,0,0)));s=e.empty_state(1)
        with self.assertRaisesRegex(ValueError,'out-of-bounds'):e.query(s,0,'x_0',0,'CA')
        self.assertEqual(len(e.cache),0)


class GraspControllerTests(unittest.TestCase):
    def test_real_controller_returns_valid_tuple_and_keeps_all_pieces_eligible(self):
        events=[];exclusion=[]
        def selector(info,*a,**kw):exclusion.append(kw['exclude_last_bin']);return 0
        def metrics(objs,size,shape,layout):
            n=len(layout);u=len(objs)/(4*n)
            us=u if n<2 else (len(objs)-min(len(g) for g in layout))/(4*(n-1))
            return n,u,us
        packing=types.SimpleNamespace(rotate_voxel=lambda a,d,ax:a.copy(),
                translate_voxel=translate,visualize_voxel_model=lambda *a:None)
        alg=types.SimpleNamespace(get_final_performance=metrics,improve_pieces_selection_ls=selector)
        original=objects(3)
        with patch.dict(sys.modules,packing_iter_ls=packing,new_ILS_southampton=alg), \
             patch.object(module,'_geometry_search',return_value=first_free):
            result=module.improved_GRASP_weight_radio(original,None,None,10,1,
                ([0,90],['x']),([0,90],['x']),['x_0','x_90'],'SCH','bounding_box',
                'bottom_top',False,'minimum_aabb_volume',None,(4,1,1),'cube',3,100,
                1,False,False,False,restart_limit=2,max_weight=2,cache_observer=lambda **e:events.append(e))
        self.assertEqual(len(result),11)
        self.assertEqual(result[0],2)
        self.assertEqual(sorted(i for ids in result[9] for i in ids),[0,1,2])
        self.assertTrue(exclusion and not any(exclusion))
        self.assertTrue(any(e['event']=='RESTORE_CACHED_SUCCESS' for e in events))
        self.assertEqual(events[-1]['restarts'],2)
        self.assertTrue(all('translation' not in p for p in original))


if __name__ == '__main__':
    unittest.main()
