import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from protocol import *
from worker import settings_for,valid_saved,verify_runtime,read_selections,restore_radio_precision
from suite import freeze,collect,submit,save,memory_for
from orientation_rcl import choose_orientation
from smoke import verify_smoke


class ProtocolTests(unittest.TestCase):
    def test_runtime_radiation_matches_full_precision_calibration(self):
        with tempfile.TemporaryDirectory() as name:
            instance=Path(name)/'instance.txt'
            instance.write_text('cube\t(4,4,4)\t999999\t1\n../a.binvox\t1\t1096.7582033835115\t7\n')
            objects=[dict(piece_type=7,volume=1,radio=1096.758203)]
            restore_radio_precision(objects,instance)
            self.assertEqual(objects[0]['radio'],1096.7582033835115)
            objects[0]['piece_type']=8
            with self.assertRaisesRegex(ValueError,'order/identity'):restore_radio_precision(objects,instance)

    def test_complete_scale_and_grasp_four_arms(self):
        p=make_plan([str(i) for i in range(15)],['0','1','2'])
        self.assertEqual({s:len(v) for s,v in p['plans'].items()},
                         dict(calibration=150,rcl=270,tuning=180,full=4050))
        self.assertEqual(sum(map(len,p['plans'].values())),4650)
        self.assertEqual(len([c for c in tuning_configs() if c['algorithm']=='improved_GRASP']),4)

    def test_worker_settings_preserve_limits(self):
        p=make_plan(['x'],['x'])
        selections={'rcl':{'x|cube':{'tolerance':.15}}}
        for c in tuning_configs():
            case=dict(c,stage='tuning',dataset='x',shape='cube')
            alg,kw=settings_for(case,p,selections)
            self.assertEqual(kw['time_limit'],3600)
            if alg=='improved_ILS':self.assertIsNone(kw['iteration_limit'])
            elif alg=='improved_GRASP':self.assertEqual(kw['ls_evaluation_limit'],c['total_cap'])
            else:self.assertEqual(kw['iter_limit_per_iter'],300)
        _,kw=settings_for(p['plans']['rcl'][0],p,{})
        self.assertEqual((kw['restart_limit'],kw['ls_evaluation_limit']),(1,50))
        self.assertTrue(math.isinf(kw['time_limit']))
        _,kw=settings_for(p['plans']['calibration'][0],p,{})
        self.assertEqual((kw['ALG'],kw['iteration_limit']),('fixed_CA',0))

    def test_calibration_excludes_emptiest_and_individual_floor(self):
        records=[dict(best_U_star=.8,mean_piece_count_without_emptiest=10,bin_capacity=100,
                      max_piece_volume=50,max_piece_radio=1500)]*5
        result=calibrate(records)
        self.assertEqual(result['weight_capacity'],dict(strong=50,medium=60.00000000000001,weak=80))
        self.assertEqual(result['radio_capacity'],dict(strong=2500,medium=3750,weak=5000))
        self.assertEqual(result['reference_count'],5)

    def test_tolerance_selects_mean_objective_and_records_ties(self):
        records=[dict(dataset='x',shape='cube',tolerance=t,objective=v)
                 for t,v in ((.1,3),(.15,2),(.2,2))]
        r=select_tolerances(records)['x|cube']
        self.assertEqual(r['tolerance'],.15)
        self.assertEqual(r['tied_tolerances'],[.15,.2])

    def test_tuning_uses_rank_not_large_dataset_scale(self):
        records=[]
        for ds in ('small','large'):
            for shape in SHAPES:
                for i,c in enumerate(tuning_configs()):
                    records.append(dict(c,dataset=ds,shape=shape,objective=(1000 if ds=='large' else 1)+i))
        out=select_tuning(records)
        self.assertEqual(out['improved_GRASP']['config_id'],'G_NI100')
        self.assertEqual(out['improved_ILS']['config_id'],'I_K100')
        self.assertEqual(out['improved_GRASP_ILS']['config_id'],'GI_B300_K100')

    def test_orientation_second_level_and_negative_pes(self):
        class Last:
            def integers(self,n):return n-1
        events=[]
        self.assertEqual(choose_orientation([(100,'a'),(110,'b'),(130,'c')],.1,Last(),events.append),(110,'b'))
        self.assertEqual(events[0]['active_rcl_size'],2)
        self.assertEqual(choose_orientation([(-100,'a'),(-90,'b'),(-80,'c')],.1,Last()),(-90,'b'))
        self.assertEqual(choose_orientation([(0,'a'),(1,'b')],.2,Last()),(0,'a'))
        self.assertEqual(choose_orientation([(3,'a'),(2,'b')],None),(2,'b'))


class FrozenWorkflowTests(unittest.TestCase):
    def test_memory_probe_has_three_hour_long_tasks_and_fixed_parameters(self):
        from unittest.mock import patch
        import memory_probe
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);bundle,m=self.fixture(root)
            with patch.object(memory_probe,'freeze',return_value=m),contextlib.redirect_stdout(io.StringIO()):
                m=memory_probe.prepare(root/'project',root/'ca.csv',bundle)
            self.assertEqual(set(m['profile']['plans']),{'memory'})
            self.assertEqual(len(m['profile']['plans']['memory']),3)
            # The real freeze receives 3600, fake fixture time is 30 here.
            m['profile']['time_limit']=3600
            for case in m['profile']['plans']['memory']:
                algorithm,kw=settings_for(case,m['profile'],{})
                self.assertEqual(kw['time_limit'],3600)
                if algorithm=='improved_ILS':self.assertEqual(kw['kick_trigger_time'],100)
                else:
                    self.assertEqual(kw['rcl_tolerance'],.15)
                    if algorithm=='improved_GRASP':
                        self.assertEqual(kw['no_improvement_limit'],100)
                        self.assertIsNone(kw['ls_evaluation_limit'])
                    else:self.assertEqual((kw['iter_limit_per_iter'],kw['kick_trigger_time']),(300,100))

    def test_historical_calibration_import_skips_gpu_and_passes_smoke(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            original,m=self.fixture(root)
            project=root/'project'
            history=root/'history';history.mkdir()
            for shape in SHAPES:
                for seed in (3,7,13,19,53):
                    source=project/f'instances/x_seq3_{shape}_999999_1.txt'
                    target=project/f'instances/x_seq{seed}_{shape}_999999_1.txt'
                    if seed!=3:target.write_text(source.read_text())
                    capacity=64 if shape=='cube' else math.pi*4*4
                    save(history/f'{shape}_{seed}.json',dict(status='success',validation=dict(ok=True),
                        config=dict(instance_id=f'x_seq{seed}_{shape}',alg='fixed_CA',constraint_mode='geometry_only',
                                    pes='0',oes='0',range='all'),
                        container_shape=shape,container_size=[4,4,4],algorithm_time_s=1,
                        placements=[dict(bin=0,piece_type=0)],
                        metrics=dict(best_N=1,best_U=1/capacity,best_U_star=1/capacity)))
            bundle=root/'imported'
            with contextlib.redirect_stdout(io.StringIO()):
                m=freeze(project,root/'ca.csv',bundle,['x'],['x'],1800,True,('cylinder',),history)
                self.assertEqual(collect(bundle,'calibration'),0)
                jobs=submit(bundle,dry_run=True)
            self.assertEqual(len(m['profile']['plans']['calibration']),5)
            self.assertTrue(jobs[0]['command'][-1].endswith('run_collect.slurm'))
            self.assertFalse(any('--gres=' in a for a in jobs[0]['command']))
            for stage in ('rcl','tuning','full'):
                for i,c in enumerate(m['profile']['plans'][stage]):
                    self.success(bundle,m,c,objective=2.4+i/100,best_U_star=.6,
                                 best_U=.5,algorithm_wall_s=1800,total_wall_s=1801)
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,stage),0)
            self.assertEqual(verify_smoke(bundle)['status'],'PASS')
            self.assertEqual(len(m['profile']['plans']['rcl']),3)
            self.assertEqual(len(m['profile']['plans']['tuning']),10)
            self.assertEqual(len(m['profile']['plans']['full']),27)

    def success(self,bundle,manifest,case,**fields):
        path=bundle/'results'/case['stage']/f"task_{case['task_id']}.json"
        selections=read_selections(bundle,case)
        algorithm,settings=settings_for(case,manifest['profile'],selections)
        if settings.get('time_limit') is not None and not math.isfinite(settings['time_limit']):
            settings['time_limit']=None
        fields.setdefault('algorithm',algorithm)
        fields.setdefault('search_settings',settings)
        fields.setdefault('method',dict(radio_priority=case['stage']=='full',warmup='none'))
        fields.setdefault('counts',dict(construction=1,ls=case['ls_budget'] if case['stage']=='rcl' else 0,kick=0))
        fields.setdefault('best_N',3)
        if case['stage']=='full':
            limits=selections['calibration'][f"{case['dataset']}|{case['shape']}"]
            fields.setdefault('weight_capacity',limits['weight_capacity'][case['weight_level']])
            fields.setdefault('radio_capacity',limits['radio_capacity'][case['radio_level']])
        save(path,dict(case=case,bundle_id=manifest['bundle_id'],status='success',validation_ok=True,
                       **fields))

    def test_all_four_stages_write_and_consume_selected_parameters(self):
        """Synthetic results exercise real collection and worker configuration code."""
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            plans=m['profile']['plans']
            for c in plans['calibration']:
                self.success(bundle,m,c,best_U_star=.8,mean_piece_count_without_emptiest=10,
                    bin_capacity=100,max_piece_volume=30,max_piece_radio=1000)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'calibration'),0)
            for c in plans['rcl']:
                winner=.15 if c['shape']=='cube' else .20
                self.success(bundle,m,c,objective=2.2 if c['tolerance']==winner else 3.2)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'rcl'),0)
            selected=json.loads((bundle/'results/rcl/selection.json').read_text())
            self.assertEqual(selected['x|cube']['tolerance'],.15)
            self.assertEqual(selected['x|cylinder']['tolerance'],.20)
            winners={'improved_GRASP':'G_B125','improved_ILS':'I_K150',
                     'improved_GRASP_ILS':'GI_B300_K125'}
            for c in plans['tuning']:
                selection=read_selections(bundle,c)
                algorithm,kw=settings_for(c,m['profile'],selection)
                if algorithm!='improved_ILS':
                    self.assertEqual(kw['rcl_tolerance'],.15 if c['shape']=='cube' else .20)
                self.success(bundle,m,c,objective=2.1 if c['config_id']==winners[algorithm] else 3.1)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'tuning'),0)
            for c in plans['full']:
                selection=read_selections(bundle,c)
                algorithm,kw=settings_for(c,m['profile'],selection)
                self.assertEqual(selection['tuning'][algorithm]['config_id'],winners[algorithm])
                self.assertEqual(selection['calibration'][f"x|{c['shape']}"]['weight_capacity']['strong'],40)
                if algorithm=='improved_GRASP':self.assertEqual(kw['ls_evaluation_limit'],125)
                elif algorithm=='improved_ILS':self.assertEqual(kw['kick_trigger_time'],150)
                else:
                    self.assertEqual(kw['iter_limit_per_iter'],300)
                    self.assertEqual(kw['kick_trigger_time'],125)
                if algorithm!='improved_ILS':
                    self.assertEqual(kw['rcl_tolerance'],.15 if c['shape']=='cube' else .20)
                self.success(bundle,m,c,best_N=3,best_U=.5,best_U_star=.6,objective=2.4,
                             algorithm_wall_s=30,total_wall_s=31)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'full'),0)
            with (bundle/'results/full/summary.csv').open(encoding='utf-8') as stream:
                summary=list(csv.DictReader(stream))
            self.assertEqual(len(summary),54)
            self.assertTrue(all(float(row['objective_mean'])==2.4 for row in summary))
            self.assertEqual(json.loads((bundle/'smoke_verification.json').read_text())['status'],'PASS')
            # A valid-looking result with the wrong runtime tolerance must fail.
            c=next(c for c in plans['full'] if c['algorithm']=='improved_GRASP')
            path=bundle/'results/full'/f"task_{c['task_id']}.json"
            report=json.loads(path.read_text())
            report['search_settings']['rcl_tolerance']=.99
            save(path,report)
            verified=verify_smoke(bundle)
            self.assertEqual(verified['status'],'FAIL')
            self.assertTrue(any('Runtime parameters' in p for p in verified['problems']))

    def test_incomplete_rcl_does_not_produce_tolerance(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            for c in m['profile']['plans']['rcl'][:-1]:
                self.success(bundle,m,c,objective=2.5)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'rcl'),1)
            self.assertFalse((bundle/'results/rcl/selection.json').exists())
            with self.assertRaises(FileNotFoundError):
                read_selections(bundle,m['profile']['plans']['tuning'][0])
            verification=verify_smoke(bundle)
            self.assertEqual(verification['status'],'FAIL')
            self.assertTrue(any('Missing/failed result: rcl/' in p for p in verification['problems']))

    def fixture(self,root):
        project=root/'project'
        for name in ('src','test','hpc','instances','data/Shapes'):(project/name).mkdir(parents=True)
        (project/'src/module.py').write_text('x=1\n')
        (project/'test/code_launcher.py').write_text('# loader\n')
        (project/'hpc/metaheuristic_iterations.py').write_text('# validation\n')
        (project/'data/Shapes/a.binvox').write_bytes(b'geometry')
        for shape in SHAPES:
            (project/f'instances/x_seq3_{shape}_999999_1.txt').write_text(
                f'{shape}\t(4,4,4)\t999999\t1\n../Shapes/a.binvox\t1\t0\t0\n')
        summary=root/'ca.csv'
        fields=['dataset','bin_shape','nesting_strategy','selection_range','constraint_mode','complete',
                'best_N_mean','best_U_star_mean','algorithm_time_s_mean']
        with summary.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
            for shape in SHAPES:
                for n in range(4):
                    for rg in ('all','bottom','bottom_top'):
                        w.writerow(dict(zip(fields,['x',shape,str(n),rg,'geometry_only','TRUE',2,.5,n+1])))
        bundle=root/'bundle'
        with contextlib.redirect_stdout(io.StringIO()):
            manifest=freeze(project,summary,bundle,['x'],['x'],30,True)
        return bundle,manifest

    def test_freeze_hashes_and_portable_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            c=m['profile']['plans']['rcl'][0]
            self.assertEqual(c['ls_budget'],3)
            verify_runtime(bundle,m,c)
            (bundle/'runtime/bin_packing/src/module.py').write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'Frozen runtime'):verify_runtime(bundle,m,c)

    def test_incomplete_stage_cannot_select_and_resume_skips_success(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            c=m['profile']['plans']['calibration'][0]
            path=bundle/'results/calibration/task_0.json'
            save(path,dict(case=c,bundle_id=m['bundle_id'],status='success',validation_ok=True))
            self.assertTrue(valid_saved(path,c,m['bundle_id']))
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'calibration'),1)
            self.assertFalse((bundle/'results/calibration/selection.json').exists())
            with contextlib.redirect_stdout(io.StringIO()):jobs=submit(bundle,dry_run=True,pending=True)
            arrays=[j['command'] for j in jobs if any('--array=' in a for a in j['command'])]
            self.assertEqual(len(arrays),4)
            self.assertIn('--array=0-0%3',arrays[0])
            # Stage 2 only starts after successful calibration finalizer.
            self.assertIn('--dependency=afterok:900001',arrays[1])
            self.assertTrue(any('--gres=gpu:1' in a for a in arrays[0]))
            controls=[j for j in jobs if 'run_collect.slurm' in j['command'][-1]]
            self.assertEqual(len(controls),4)
            self.assertIn('--dependency=afterany:900000',controls[0]['command'])

    def test_collect_complete_stage_produces_calibration(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            for c in m['profile']['plans']['calibration']:
                save(bundle/'results/calibration'/f"task_{c['task_id']}.json",dict(
                    case=c,bundle_id=m['bundle_id'],status='success',validation_ok=True,best_U_star=.8,
                    mean_piece_count_without_emptiest=10,bin_capacity=100,max_piece_volume=30,max_piece_radio=1000))
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'calibration'),0)
            selected=json.loads((bundle/'results/calibration/selection.json').read_text())
            self.assertEqual(selected['x|cube']['weight_capacity']['strong'],40)
            self.assertEqual(selected['x|cylinder']['radio_capacity']['strong'],2000)

    def test_memory_groups_and_chunked_array_dependencies(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            m['profile']['plans']['calibration']=[dict(m['profile']['plans']['calibration'][0],task_id=i,
                dataset='shapesnew' if i==0 else 'large') for i in range(1805)]
            save(bundle/'manifest.json',m)
            with contextlib.redirect_stdout(io.StringIO()):jobs=submit(bundle,dry_run=True,through='calibration')
            self.assertEqual(len(jobs),5) # 1 small + 3 large chunks + CPU gate
            self.assertIn('--mem=16G',jobs[0]['command'])
            self.assertIn('--mem=32G',jobs[1]['command'])
            self.assertIn('--array=0-899%3',jobs[1]['command'])
            self.assertIn('--dependency=afterany:900000',jobs[1]['command'])
            self.assertIn('--dependency=afterany:900000:900001:900002:900003',jobs[-1]['command'])


if __name__=='__main__':unittest.main()
