"""October 30 Word plan: two levels and fixed overall best PES per shape."""
from collections import defaultdict
from protocol import (REFERENCE_SEEDS,SEQUENCE_SEEDS,SHAPES,RCL_CONFIGS,STAGES,
                      kick_tuning_datasets,tuning_configs,_global_selection)

VERSION='ejor-two-level-fixed-pes-v4'
LEVELS=('strong','weak')
PES=('minimum_aabb_volume','minimum_aabb_edges_len','maximal_residual_box','overlap_distance')
RANGES=('all','bottom','bottom_top')
BEST_PES={'cube':dict(pes='maximal_residual_box',selection_range='bottom'),
          'cylinder':dict(pes='minimum_aabb_edges_len',selection_range='bottom_top')}
PARENTS={'calibration':(),'sensitivity':('calibration',),'rcl':('calibration',),
         'tuning':('rcl',),'full':('tuning',)}



def condition(case):
    return '|'.join(case[k] for k in ('shape','weight_level','radio_level'))


def make_plan(datasets,tuning_datasets=None,seeds=SEQUENCE_SEEDS,reference_seeds=REFERENCE_SEEDS,
              full_seeds=SEQUENCE_SEEDS,time_limit=3600,smoke=False,shapes=SHAPES):
    # Reuse input validation, not the legacy task grid or fixed heuristic settings.
    from protocol import make_plan as validate
    tuning_datasets=tuning_datasets or kick_tuning_datasets(datasets)
    validate(datasets,tuning_datasets,seeds,reference_seeds,full_seeds,time_limit,smoke,shapes)
    plans={s:[] for s in STAGES}
    def add(stage,**kw):
        plans[stage].append(dict(task_id=len(plans[stage]),stage=stage,protocol=VERSION,**kw))
    for dataset in datasets:
        for shape in shapes:
            for seed in reference_seeds:
                add('calibration',dataset=dataset,shape=shape,seed=seed,
                    algorithm='improved_ILS',config_id='CA_REFERENCE')
            for weight in LEVELS:
                for radio in LEVELS:
                    base=dict(dataset=dataset,shape=shape,weight_level=weight,radio_level=radio)
                    for seed in full_seeds:
                        for config in (dict(config_id='CA_FIXED',**BEST_PES[shape]),):
                            add('sensitivity',**base,seed=seed,algorithm='improved_ILS',**config)
                        for alg in ('improved_ILS','improved_GRASP','improved_GRASP_ILS'):
                            add('full',**base,seed=seed,algorithm=alg,config_id=f'{alg}_W{weight}_R{radio}',
                                packing_selection='fixed_global_by_shape')
                    for seed in seeds:
                        for config in RCL_CONFIGS:
                            add('rcl',**base,seed=seed,algorithm='improved_GRASP',ls_budget=0,**config,
                                packing_selection='fixed_global_by_shape')
                        if dataset in tuning_datasets:
                            for config in tuning_configs():
                                add('tuning',**base,seed=seed,**config,
                                    packing_selection='fixed_global_by_shape')
    return dict(protocol=VERSION,datasets=list(datasets),tuning_datasets=list(tuning_datasets),
        seeds=list(seeds),reference_seeds=list(reference_seeds),full_seeds=list(full_seeds),
        shapes=list(shapes),levels=list(LEVELS),plans=plans,time_limit=time_limit,
        tuning_time_limit=min(time_limit,30.) if smoke else 1800.,smoke=smoke,
        selection_scope='global_by_shape',selection_metric='mean_N_minus_U_star',
        sensitivity_selection_datasets=list(datasets),rcl_selection_datasets=list(datasets),
        kick_selection_datasets=list(tuning_datasets),
        capacity_policy='preserve original five-seed calibration; only strong/weak levels used',
        iteration_policy='latest user-confirmed repacking counts; no multiplication by nine',
        submission_policy='independent sensitivity and RCL -> tuning -> full; each flow max 6')


def select_sensitivity(records,profile):
    """Summarize fixed-PES sensitivity; this stage does not select parameters."""
    wanted={(d,shape,seed,w,r) for d in profile['datasets'] for shape in profile['shapes']
            for seed in profile['full_seeds'] for w in LEVELS for r in LEVELS}
    cells=[tuple(v[k] for k in ('dataset','shape','seed','weight_level','radio_level')) for v in records]
    if len(cells)!=len(wanted) or set(cells)!=wanted:
        raise ValueError('Sensitivity requires every dataset, shape, seed and capacity condition')
    groups=defaultdict(list)
    for record in records:groups[condition(record)].append(record)
    import statistics
    return dict(protocol=VERSION,scope='fixed_global_by_shape',parameter_selection=False,
                fixed_pes=BEST_PES,summary=[dict(condition=k,**BEST_PES[k.split('|')[0]],
                repetitions=len(v),mean_N_minus_U_star=statistics.mean(r['objective'] for r in v))
                for k,v in sorted(groups.items())])


def packing_settings(case,manifest,selections=None):
    if case['stage']=='calibration':
        return dict(manifest['ca_parameters'][f"{case['dataset']}|{case['shape']}"])
    return dict(**BEST_PES[case['shape']],parameter_scope='global_by_shape')
