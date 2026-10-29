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
            dict(calibration=150,sensitivity=810,rcl=3240,tuning=486,full=2430))
        self.assertEqual(sum(map(len,p['plans'].values())),7116)
        self.assertEqual({c['algorithm'] for c in tuning_configs()},{'improved_ILS'})

    def test_all_formal_stages_only_use_three_sequence_seeds(self):
        p=make_plan(['x'],['x'])
        for stage,cases in p['plans'].items():
            self.assertEqual({c['seed'] for c in cases},set(REFERENCE_SEEDS if stage=='calibration' else SEQUENCE_SEEDS),stage)
            self.assertEqual([c['task_id'] for c in cases],list(range(len(cases))))
        for field in ('seeds','reference_seeds','full_seeds'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                make_plan(['x'],['x'],**{field:(3,17)})

    def test_worker_settings_preserve_limits(self):
        p=make_plan(['x'],['x'])
        for c in tuning_configs():
            alg,kw=settings_for(dict(c,stage='tuning'),p,{})
            self.assertEqual(kw['time_limit'],1800)
            self.assertEqual(kw['kick_trigger_time'],'auto')
            self.assertIsNone(kw['iteration_limit'])
        for case in p['plans']['rcl']:
            _,kw=settings_for(case,p,{})
            self.assertEqual((kw['restart_limit'],kw['ls_evaluation_limit']),(1,0))
            self.assertTrue(math.isinf(kw['time_limit']))
            self.assertEqual(kw['rcl_top_k'],case['top_k'])
        _,kw=settings_for(p['plans']['calibration'][0],p,{})
        self.assertEqual((kw['ALG'],kw['iteration_limit']),('fixed_CA',0))

    def test_calibration_excludes_emptiest_and_individual_floor(self):
        records=[dict(best_U_star=.8,mean_piece_count_without_emptiest=10,bin_capacity=100,
                      max_piece_volume=50,max_piece_radio=1500)]*5
        result=calibrate(records)
        self.assertEqual(result['weight_capacity'],dict(strong=50,medium=60.00000000000001,weak=80))
        self.assertEqual(result['radio_capacity'],dict(strong=1500,medium=2000,weak=2500))
        self.assertEqual(result['reference_count'],5)

    def test_tolerance_selects_mean_objective_and_records_ties(self):
        records=[dict(c,dataset='x',shape='cube',objective=2 if c['config_id'] in ('TOP10','TOP30') else 3)
                 for c in RCL_CONFIGS]
        r=select_tolerances(records)['global_rcl']
        self.assertEqual(r['top_k'],10)
        self.assertEqual(r['tied_configs'],['TOP10','TOP30'])
        with self.assertRaises(ValueError):select_tolerances(records[:-1])

    def test_tuning_uses_raw_mean_objective_globally(self):
        records=[]
        for ds,values in (('small',(1,2,3)),('large',(100,80,99))):
            for c,value in zip(tuning_configs(),values):
                records.append(dict(c,dataset=ds,shape='cube',objective=value))
        out=select_tuning(records)
        self.assertEqual(out['improved_ILS']['config_id'],'I_S50')
        self.assertEqual(out['improved_GRASP']['ls_policy'],'systematic')
        self.assertEqual(out['improved_GRASP_ILS']['kick_level'],'medium')
        self.assertEqual(out['improved_GRASP_ILS']['ls_budget'],'n')
        self.assertEqual(out['improved_ILS']['metric'],'mean_N_minus_U_star')

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
    def test_cli_defaults_to_sensitivity_and_submits_no_later_stages(self):
        from unittest.mock import patch
        import suite
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            with patch.object(sys,'argv',['suite.py','submit-all','--bundle',str(bundle),'--dry-run']), \
                 patch.object(suite,'submit',return_value=[]) as submitted:
                self.assertEqual(suite.main(),0)
                self.assertEqual(submitted.call_args.args[7],'sensitivity')
            with contextlib.redirect_stdout(io.StringIO()):
                jobs=submit(bundle,dry_run=True,through='sensitivity')
            self.assertEqual(len(jobs),4)
            names=[arg for job in jobs for arg in job['command'] if arg.startswith('--job-name=')]
            self.assertEqual(names,['--job-name=ejor_calibration','--job-name=ejor_collect_calibration',
                                    '--job-name=ejor_sensitivity','--job-name=ejor_collect_sensitivity'])

    def test_memory_probe_has_three_hour_long_tasks_and_fixed_parameters(self):
        from unittest.mock import patch
        import memory_probe
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);bundle,m=self.fixture(root)
            with patch.object(memory_probe,'freeze',return_value=m),contextlib.redirect_stdout(io.StringIO()):
                m=memory_probe.prepare(root/'project',root/'ca.csv',bundle)
            self.assertEqual(set(m['profile']['plans']),{'memory'})
            self.assertEqual(len(m['profile']['plans']['memory']),3)
            parameter_plan=json.loads((bundle/'parameter_plan.json').read_text())
            self.assertEqual(set(parameter_plan['profile']['plans']),{'memory'})
            # The real freeze receives 3600, fake fixture time is 30 here.
            m['profile']['time_limit']=3600
            for case in m['profile']['plans']['memory']:
                algorithm,kw=settings_for(case,m['profile'],{})
                self.assertEqual(kw['time_limit'],3600)
                if algorithm=='improved_ILS':self.assertEqual(kw['kick_trigger_time'],'auto')
                else:
                    self.assertEqual(kw['rcl_tolerance'],.10)
                    if algorithm=='improved_GRASP':
                        self.assertEqual(kw['ls_policy'],'systematic')
                        self.assertEqual(kw['iter_limit_per_iter'],1)
                        self.assertIsNone(kw['ls_evaluation_limit'])
                    else:self.assertEqual((kw['iter_limit_per_iter'],kw['kick_trigger_time']),(None,'auto'))

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
            for stage in ('sensitivity','rcl','tuning','full'):
                for i,c in enumerate(m['profile']['plans'][stage]):
                    self.success(bundle,m,c,objective=2.4+i/100,best_U_star=.6,
                                 best_U=.5,algorithm_wall_s=1800,total_wall_s=1801)
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,stage),0)
            self.assertEqual(verify_smoke(bundle)['status'],'PASS')
            self.assertEqual(len(m['profile']['plans']['rcl']),36)
            self.assertEqual(len(m['profile']['plans']['tuning']),27)
            self.assertEqual(len(m['profile']['plans']['full']),27)
            # Reuse also preserves and hashes each original historical CA JSON.
            continued=root/'continued_historical'
            with contextlib.redirect_stdout(io.StringIO()):
                reused=freeze(project,root/'ca.csv',continued,['x'],['x'],1800,True,
                              ('cylinder',),reuse_bundle=bundle)
            self.assertEqual(reused['completed_stage_source']['counts'],dict(calibration=5,sensitivity=9))
            for case in reused['profile']['plans']['calibration']:
                report=json.loads((continued/'results/calibration'/f"task_{case['task_id']}.json").read_text())
                reference=report['source_reference']
                self.assertTrue((continued/reference).exists())
                self.assertIn(reference,reused['files_sha256'])

    def test_formal_freeze_preserves_three_seeds_in_frozen_task_maps(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            _,_=self.fixture(root)
            project=root/'project'
            for shape in SHAPES:
                source=project/f'instances/x_seq3_{shape}_999999_1.txt'
                for seed in (7,13,19,53):
                    (project/f'instances/x_seq{seed}_{shape}_999999_1.txt').write_bytes(source.read_bytes())
            bundle=root/'formal'
            with contextlib.redirect_stdout(io.StringIO()):
                m=freeze(project,root/'ca.csv',bundle,['x'],['x'])
                jobs=submit(bundle,dry_run=True,through='full')
            self.assertTrue(jobs)
            parameter=json.loads((bundle/'parameter_plan.json').read_text())
            self.assertEqual(parameter['profile'],m['profile'])
            for stage,cases in m['profile']['plans'].items():
                self.assertEqual({c['seed'] for c in cases},set(REFERENCE_SEEDS if stage=='calibration' else SEQUENCE_SEEDS),stage)
            names={p.name for p in (bundle/'runtime/bin_packing/instances').glob('*.txt')}
            self.assertEqual(len(names),10)
            experiments=[c for stage,cases in m['profile']['plans'].items() if stage!='calibration' for c in cases]
            self.assertTrue(all(c['seed'] in (3,7,13) for c in experiments))
            self.assertIn('repacking',parameter['conventions']['kick_trigger'])
            self.assertTrue(parameter['conventions']['hybrid_budget'].startswith('n LS'))

    def test_continue_after_sensitivity_reuses_results_and_schedules_only_later_gpu_stages(self):
        from reuse_completed import import_completed
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            self.fixture(root)
            project=root/'project'
            for shape in SHAPES:
                source=project/f'instances/x_seq3_{shape}_999999_1.txt'
                for seed in (7,13,19,53):
                    (project/f'instances/x_seq{seed}_{shape}_999999_1.txt').write_bytes(source.read_bytes())
            with contextlib.redirect_stdout(io.StringIO()):
                source=root/'completed'
                old=freeze(project,root/'ca.csv',source,['x'],['x'])
                for c in old['profile']['plans']['calibration']:
                    self.success(source,old,c,best_U_star=.8,mean_piece_count_without_emptiest=10,
                                 bin_capacity=100,max_piece_volume=30,max_piece_radio=1000)
                self.assertEqual(collect(source,'calibration'),0)
                for c in old['profile']['plans']['sensitivity']:
                    self.success(source,old,c,objective=2.4,best_U=.5,best_U_star=.6,
                                 algorithm_wall_s=1,total_wall_s=2)
                self.assertEqual(collect(source,'sensitivity'),0)
                instance=source/'runtime/bin_packing/instances/x_seq3_cube_999999_1.txt'
                import hashlib
                normalized=instance.read_bytes().replace(b'\r\n',b'\n')
                old['files_sha256']['runtime/bin_packing/instances/x_seq3_cube_999999_1.txt']=hashlib.sha256(normalized).hexdigest()
                save(source/'manifest.json',old)
                instance.write_bytes(normalized.replace(b'\n',b'\r\n'))
                bundle=root/'continued'
                m=freeze(project,root/'ca.csv',bundle,['x'],['x'],reuse_bundle=source)
                jobs=submit(bundle,dry_run=True,through='full')
            self.assertEqual(m['completed_stage_source']['counts'],dict(calibration=10,sensitivity=54))
            self.assertEqual(json.loads((bundle/'results/calibration/selection.json').read_text()),
                             json.loads((source/'results/calibration/selection.json').read_text()))
            arrays=[j['command'] for j in jobs if any('--array=' in a for a in j['command'])]
            names=[a for command in arrays for a in command if a.startswith('--job-name=')]
            self.assertEqual(names,['--job-name=ejor_rcl','--job-name=ejor_tuning','--job-name=ejor_full'])
            for stage in ('calibration','sensitivity'):
                for c in m['profile']['plans'][stage]:
                    report=bundle/'results'/stage/f"task_{c['task_id']}.json"
                    self.assertTrue(valid_saved(report,c,m['bundle_id']))
                    self.assertIn('reused_from',json.loads(report.read_text()))
            # Reject incompatible capacities rather than silently relabeling results.
            first=source/'results/sensitivity/task_0.json'
            report=json.loads(first.read_text());report['weight_capacity']*=2;save(first,report)
            with self.assertRaisesRegex(ValueError,'capacity mismatch'):
                import_completed(bundle,source)

    def success(self,bundle,manifest,case,**fields):
        path=bundle/'results'/case['stage']/f"task_{case['task_id']}.json"
        selections=read_selections(bundle,case)
        algorithm,settings=settings_for(case,manifest['profile'],selections)
        if settings.get('time_limit') is not None and not math.isfinite(settings['time_limit']):
            settings['time_limit']=None
        fields.setdefault('algorithm',algorithm)
        fields.setdefault('search_settings',settings)
        fields.setdefault('method',dict(radio_priority=case['stage'] in ('sensitivity','rcl','tuning','full'),warmup='none'))
        fields.setdefault('counts',dict(construction=1,ls=case['ls_budget'] if case['stage']=='rcl' else 0,kick=0))
        fields.setdefault('best_N',3)
        if case['stage'] in ('sensitivity','rcl','tuning','full'):
            limits=selections['calibration'][f"{case['dataset']}|{case['shape']}"]
            fields.setdefault('weight_capacity',limits['weight_capacity'][case['weight_level']])
            fields.setdefault('radio_capacity',limits['radio_capacity'][case['radio_level']])
        save(path,dict(case=case,bundle_id=manifest['bundle_id'],status='success',validation_ok=True,
                       **fields))

    def test_all_five_stages_write_and_consume_selected_parameters(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            plans=m['profile']['plans']
            for c in plans['calibration']:
                self.success(bundle,m,c,best_U_star=.8,mean_piece_count_without_emptiest=10,
                             bin_capacity=100,max_piece_volume=30,max_piece_radio=1000)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'calibration'),0)
            for stage in ('sensitivity','rcl','tuning','full'):
                for c in plans[stage]:
                    algorithm,kw=settings_for(c,m['profile'],read_selections(bundle,c))
                    if stage=='full':
                        self.assertEqual(kw['time_limit'],30)
                        if algorithm=='improved_GRASP':
                            self.assertIsNone(kw['ls_evaluation_limit'])
                            self.assertEqual(kw['ls_policy'],'systematic')
                        if algorithm=='improved_GRASP_ILS':
                            self.assertIsNone(kw['iter_limit_per_iter'])
                            self.assertEqual(kw['kick_trigger_time'],'auto')
                            self.assertEqual(kw['kick_level'],'large')
                        if algorithm!='improved_ILS':self.assertEqual(kw['rcl_top_k'],30)
                    winner=c['config_id'] in ('TOP30','I_S75')
                    self.success(bundle,m,c,objective=2. if winner else 3.,best_N=3,
                                 best_U=.5,best_U_star=.6,algorithm_wall_s=30,total_wall_s=31)
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,stage),0)
            self.assertEqual(verify_smoke(bundle)['status'],'PASS')
            c=next(c for c in plans['full'] if c['algorithm']=='improved_GRASP')
            path=bundle/'results/full'/f"task_{c['task_id']}.json"
            report=json.loads(path.read_text());report['search_settings']['rcl_top_k']=10
            save(path,report)
            self.assertEqual(verify_smoke(bundle)['status'],'FAIL')

    def test_incomplete_rcl_does_not_produce_tolerance(self):
        with tempfile.TemporaryDirectory() as name:
            bundle,m=self.fixture(Path(name))
            # Build prerequisite capacities, then omit one RCL task.
            for c in m['profile']['plans']['calibration']:
                self.success(bundle,m,c,best_U_star=.8,mean_piece_count_without_emptiest=10,
                             bin_capacity=100,max_piece_volume=30,max_piece_radio=1000)
            with contextlib.redirect_stdout(io.StringIO()):collect(bundle,'calibration')
            for c in m['profile']['plans']['rcl'][:-1]:self.success(bundle,m,c,objective=2.5)
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(collect(bundle,'rcl'),1)
            self.assertFalse((bundle/'results/rcl/selection.json').exists())
            with self.assertRaises(FileNotFoundError):read_selections(bundle,m['profile']['plans']['full'][0])
            self.assertEqual(verify_smoke(bundle)['status'],'FAIL')

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
            self.assertEqual(c['ls_budget'],0)
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
            self.assertEqual(len(arrays),5)
            self.assertIn('--array=0-0%6',arrays[0])
            # Stage 2 only starts after successful calibration finalizer.
            self.assertIn('--dependency=afterok:900001',arrays[1])
            self.assertTrue(any('--gres=gpu:1' in a for a in arrays[0]))
            controls=[j for j in jobs if 'run_collect.slurm' in j['command'][-1]]
            self.assertEqual(len(controls),5)
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
            self.assertEqual(selected['x|cylinder']['radio_capacity']['strong'],1000)

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
            self.assertIn('--array=0-899%6',jobs[1]['command'])
            self.assertIn('--dependency=afterany:900000',jobs[1]['command'])
            self.assertIn('--dependency=afterany:900000:900001:900002:900003',jobs[-1]['command'])


if __name__=='__main__':unittest.main()
