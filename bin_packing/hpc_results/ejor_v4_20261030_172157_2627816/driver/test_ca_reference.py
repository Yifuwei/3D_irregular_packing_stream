"""Historical placement recovery needs no GPU or third-party packages."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ca_reference import recover,index_results


class HistoricalCalibrationTests(unittest.TestCase):
    def example(self,root,one_bin=False):
        instance=root/'instance.txt'
        instance.write_text('cube\t(10,10,10)\t999999\t1\n'
            '../a.binvox\t100\t1000\t1\n../a.binvox\t100\t0\t1\n../b.binvox\t50\t0\t2\n')
        result=dict(container_shape='cube',container_size=[10,10,10],
            placements=[dict(bin=0,piece_type=1),dict(bin=0,piece_type=1),dict(bin=0 if one_bin else 1,piece_type=2)],
            metrics=dict(best_N=1 if one_bin else 2,best_U=.25 if one_bin else .125,
                         best_U_star=.25 if one_bin else .2))
        return instance,result

    def test_recovers_emptiest_by_volume_and_piece_counts(self):
        with tempfile.TemporaryDirectory() as name:
            instance,result=self.example(Path(name))
            record=recover(result,instance)
            self.assertEqual(record['bin_piece_counts'],[2,1])
            self.assertEqual(record['bin_volumes'],[200,50])
            self.assertEqual(record['emptiest_bin'],1)
            self.assertEqual(record['mean_piece_count_without_emptiest'],2)
            self.assertEqual(record['max_piece_radio'],1000)

    def test_single_bin_uses_all_pieces(self):
        with tempfile.TemporaryDirectory() as name:
            instance,result=self.example(Path(name),True)
            record=recover(result,instance)
            self.assertEqual(record['mean_piece_count_without_emptiest'],3)
            self.assertEqual(record['best_U_star'],.25)

    def test_rejects_wrong_metrics_and_piece_counts(self):
        with tempfile.TemporaryDirectory() as name:
            instance,result=self.example(Path(name))
            result['metrics']['best_U_star']=.19
            with self.assertRaisesRegex(ValueError,'metric disagrees'):recover(result,instance)
            result['placements'].pop()
            with self.assertRaisesRegex(ValueError,'multiplicities differ'):recover(result,instance)

    def test_rejects_ambiguous_piece_type_volume(self):
        with tempfile.TemporaryDirectory() as name:
            instance,result=self.example(Path(name))
            instance.write_text(instance.read_text().replace('100\t1000\t1','99\t1000\t1'))
            with self.assertRaisesRegex(ValueError,'uniquely identify volume'):recover(result,instance)

    def test_rejects_duplicate_results(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            result=dict(config=dict(instance_id='x_seq3_cube',alg='fixed_CA',constraint_mode='geometry_only',
                pes='minimum_aabb_volume',oes='minimum_aabb_volume',range='bottom'),
                status='success',validation=dict(ok=True))
            for n in ('a.json','b.json'):(root/n).write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'Duplicate'):index_results(root)


if __name__=='__main__':unittest.main()
