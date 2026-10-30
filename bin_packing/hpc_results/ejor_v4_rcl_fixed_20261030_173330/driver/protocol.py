"""October 29 protocol: three sequence seeds and repacking-based budgets."""
import math
import statistics
from collections import defaultdict

VERSION='ejor-three-seed-repacking-v3'
SEQUENCE_SEEDS=(3,7,13)
REFERENCE_SEEDS=(3,7,13,19,53)
KICK_TUNING_DATASETS=('shapesnew','Merged4_normal','st04_Example5_normal')

def kick_tuning_datasets(datasets):
    selected=[d for d in KICK_TUNING_DATASETS if d in datasets]
    return selected or list(datasets)

LEVELS=('strong','medium','weak')
SHAPES=('cube','cylinder')
SHAPE_PARAMETERS={
    'cube':dict(pes='maximal_residual_box',selection_range='bottom'),
    'cylinder':dict(pes='minimum_aabb_edges_len',selection_range='bottom_top'),
}
STAGES=('calibration','sensitivity','rcl','tuning','full')
TOLERANCES=(.10,.20)
RCL_CONFIGS=(dict(config_id='RCL10',tolerance=.10,top_k=None),
             dict(config_id='RCL20',tolerance=.20,top_k=None),
             dict(config_id='TOP10',tolerance=None,top_k=10),
             dict(config_id='TOP30',tolerance=None,top_k=30))


def tuning_configs():
    return [dict(config_id=f'I_S{round(f*100)}',algorithm='improved_ILS',
                 kick='auto',kick_level=level,kick_fraction=f)
            for level,f in (('small',.25),('medium',.5),('large',.75))]


def make_plan(datasets,tuning_datasets,seeds=SEQUENCE_SEEDS,reference_seeds=REFERENCE_SEEDS,
              full_seeds=SEQUENCE_SEEDS,time_limit=3600,smoke=False,shapes=SHAPES):
    if not shapes or len(set(shapes))!=len(shapes) or not set(shapes)<=set(SHAPES):
        raise ValueError('Shapes must be distinct cube/cylinder values')
    if len(set(datasets))!=len(datasets) or not set(tuning_datasets)<=set(datasets):
        raise ValueError('Duplicate datasets or unknown tuning dataset')
    if not datasets or not tuning_datasets or time_limit<=0 or not math.isfinite(time_limit):
        raise ValueError('Nonempty datasets and positive finite time budget required')
    for values,allowed in ((seeds,SEQUENCE_SEEDS),(reference_seeds,REFERENCE_SEEDS),(full_seeds,SEQUENCE_SEEDS)):
        if not values or len(values)!=len(set(values)):
            raise ValueError('Seeds must be nonempty and distinct')
        if not set(values)<=set(allowed):
            raise ValueError('Experiment seeds must be 3, 7, 13; reference seeds may additionally use 19, 53')
    if not smoke and set(tuning_datasets)!=set(kick_tuning_datasets(datasets)):
        raise ValueError('Kick tuning must use the confirmed three representative datasets together')
    plans={stage:[] for stage in STAGES}
    def add(stage,**kw):
        if stage in ('sensitivity','rcl','tuning','full'):
            kw.update(SHAPE_PARAMETERS[kw['shape']])
            kw['parameter_scope']='global_by_shape'
        if stage in ('rcl','tuning','full'):
            kw['selection_dataset_count']=len(tuning_datasets) if stage=='tuning' else len(datasets)
        if stage=='full':
            kw['kick_selection_dataset_count']=len(tuning_datasets)
        plans[stage].append(dict(task_id=len(plans[stage]),stage=stage,protocol=VERSION,**kw))
    for dataset in datasets:
        for shape in shapes:
            for seed in reference_seeds:
                add('calibration',dataset=dataset,shape=shape,seed=seed,
                    algorithm='improved_ILS',config_id='CA_REFERENCE')
            for weight in LEVELS:
                for radiation in LEVELS:
                    constraints=dict(weight_level=weight,radio_level=radiation)
                    for seed in full_seeds:
                        add('sensitivity',dataset=dataset,shape=shape,seed=seed,
                            algorithm='improved_ILS',config_id='CA_SENSITIVITY',**constraints)
                        for algorithm in ('improved_ILS','improved_GRASP','improved_GRASP_ILS'):
                            add('full',dataset=dataset,shape=shape,seed=seed,algorithm=algorithm,
                                config_id=f'{algorithm}_W{weight}_R{radiation}',**constraints)
                    for seed in seeds:
                        for config in RCL_CONFIGS:
                            add('rcl',dataset=dataset,shape=shape,seed=seed,
                                algorithm='improved_GRASP',ls_budget=0,**config,**constraints)
                        if dataset in tuning_datasets:
                            for config in tuning_configs():
                                add('tuning',dataset=dataset,shape=shape,seed=seed,**config,**constraints)
    return dict(protocol=VERSION,datasets=list(datasets),tuning_datasets=list(tuning_datasets),
                metaheuristic_parameters={shape:dict(SHAPE_PARAMETERS[shape]) for shape in shapes},
                seeds=list(seeds),reference_seeds=list(reference_seeds),full_seeds=list(full_seeds),
                time_limit=time_limit,tuning_time_limit=min(time_limit,30.) if smoke else 1800.,
                smoke=smoke,shapes=list(shapes),plans=plans,
                submission_policy='sensitivity independent; RCL -> tuning -> full',
                per_branch_gpu_limit_default=6,
                selection_metric='mean_N_minus_U_star',selection_scope='global',
                selection_datasets=list(datasets),selection_dataset_count=len(datasets),
                kick_selection_datasets=list(tuning_datasets),kick_selection_dataset_count=len(tuning_datasets),
                duplicate_capacity_policy='keep nominal level combinations; annotate equal capacities')


def choose_ca_parameters(rows,datasets):
    """Apply the user-confirmed global-by-shape CA configuration to all datasets.

    Per-dataset rows supply the matching baseline, never a per-dataset winner.
    Historical capacity calibration metadata is retained separately on reuse.
    """
    selected={}
    for dataset in datasets:
        for shape in SHAPES:
            group=[r for r in rows if r['dataset']==dataset and r['bin_shape']==shape
                   and r.get('complete','').lower()=='true' and r.get('constraint_mode')=='geometry_only']
            if len(group)!=12 or len({(r['nesting_strategy'],r['selection_range']) for r in group})!=12:
                raise ValueError(f'Expected 12 complete geometry-only CA groups: {dataset}/{shape}')
            chosen=SHAPE_PARAMETERS[shape]
            row=next((r for r in group if r['nesting_strategy']==chosen['pes'] and
                      r['selection_range']==chosen['selection_range']),None)
            if row is None:raise ValueError('Missing global-by-shape CA baseline: '+dataset+'/'+shape)
            selected[f'{dataset}|{shape}']=dict(chosen,parameter_scope='bin_shape_global',
                reference_mean_objective=float(row['best_N_mean'])-float(row['best_U_star_mean']))
    return selected


def calibrate(records,weight_factor=1.):
    if not records or weight_factor<=0:
        raise ValueError('Calibration needs records and a positive weight factor')
    ustar=statistics.mean(r['best_U_star'] for r in records)
    nstar=statistics.mean(r['mean_piece_count_without_emptiest'] for r in records)
    volumes={r['bin_capacity'] for r in records}
    if len(volumes)!=1:raise ValueError('Reference bin sizes differ across order seeds')
    capacity=volumes.pop()
    largest=max(r['max_piece_volume'] for r in records)
    rmax=max(r['max_piece_radio'] for r in records)
    rub=rmax+1000*max(0.,.2*nstar-1)
    radio={level:max(value,1e-12) for level,value in zip(LEVELS,(rmax,(rmax+rub)/2,rub))}
    rho={level:max(mult*ustar,largest/capacity) for level,mult in zip(LEVELS,(.5,.75,1.))}
    return dict(U_star_reference=ustar,n_star_reference=nstar,bin_capacity=capacity,
                max_piece_volume=largest,max_piece_radio=rmax,weight_factor=weight_factor,
                weight_capacity={level:max(largest,rho[level]*capacity)*weight_factor for level in LEVELS},
                weight_capacity_ratio=rho,radiation_reference=rub,radio_capacity=radio,
                coincident_radio_levels=len(set(radio.values()))<3,
                reference_count=len(records),single_bin_convention='U*=U and n*=piece_count')


def _global_selection(records,configs):
    """Arithmetic mean across matched dataset/shape/seed/constraint cells."""
    groups=defaultdict(dict)
    expected={c['config_id'] for c in configs}
    for r in records:
        key=tuple(r.get(k) for k in ('dataset','shape','seed','weight_level','radio_level'))
        name=r['config_id']
        if name not in expected or name in groups[key]:raise ValueError('Unknown/duplicate tuning record')
        if not math.isfinite(r['objective']):raise ValueError('Nonfinite objective')
        groups[key][name]=r['objective']
    if not groups or any(set(v)!=expected for v in groups.values()):
        raise ValueError('Incomplete matched parameter candidates')
    means={c['config_id']:statistics.mean(v[c['config_id']] for v in groups.values()) for c in configs}
    best=min(means.values())
    ties=[c['config_id'] for c in configs if abs(means[c['config_id']]-best)<=1e-9]
    name=ties[0]
    return dict(next(c for c in configs if c['config_id']==name),scope='global',
                metric='mean_N_minus_U_star',mean_objective=means[name],
                mean_objectives=means,tied_configs=ties,matched_cells=len(groups),
                datasets=sorted({r['dataset'] for r in records if r.get('dataset') is not None}),
                dataset_count=len({r['dataset'] for r in records if r.get('dataset') is not None}))


def select_tolerances(records):
    return dict(global_rcl=_global_selection(records,RCL_CONFIGS))


def select_tuning(records):
    selected=_global_selection(records,tuning_configs())
    return {'improved_ILS':selected,
            'improved_GRASP':dict(algorithm='improved_GRASP',ls_policy='systematic'),
            'improved_GRASP_ILS':dict(algorithm='improved_GRASP_ILS',
                ls_budget='n',kick='auto',kick_level=selected['kick_level'],
                kick_fraction=selected['kick_fraction'],inherited_from='improved_ILS')}
