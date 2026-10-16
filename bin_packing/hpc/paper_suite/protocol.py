"""Standard-library protocol definitions, calibration and parameter selection."""
import math
import statistics
from collections import defaultdict

VERSION='ejor-two-level-rcl-v1'
LEVELS=('strong','medium','weak')
SHAPES=('cube','cylinder')
TOLERANCES=(.10,.15,.20)


def tuning_configs():
    return ([dict(config_id='G_NI100',algorithm='improved_GRASP',stall=100,total_cap=None)]
            +[dict(config_id=f'G_B{n}',algorithm='improved_GRASP',stall=n,total_cap=n) for n in (100,125,150)]
            +[dict(config_id=f'I_K{n}',algorithm='improved_ILS',kick=n) for n in (100,125,150)]
            +[dict(config_id=f'GI_B300_K{n}',algorithm='improved_GRASP_ILS',ls_budget=300,kick=n) for n in (100,125,150)])


def make_plan(datasets,tuning_datasets,seeds=(3,7,13),reference_seeds=(3,7,13,19,53),
              full_seeds=(3,7,13,19,53),time_limit=3600,smoke=False,shapes=SHAPES):
    if not shapes or len(set(shapes))!=len(shapes) or not set(shapes)<=set(SHAPES):
        raise ValueError('Shapes must be distinct cube/cylinder values')
    if len(set(datasets))!=len(datasets) or not set(tuning_datasets)<=set(datasets):
        raise ValueError('Duplicate datasets or unknown tuning dataset')
    if not datasets or not tuning_datasets or time_limit<=0 or not math.isfinite(time_limit):
        raise ValueError('Nonempty datasets and positive finite time budget required')
    for values in (seeds,reference_seeds,full_seeds):
        if not values or len(values)!=len(set(values)):
            raise ValueError('Seeds must be nonempty and distinct')
    plans={stage:[] for stage in ('calibration','rcl','tuning','full')}
    def add(stage,**kw):
        plans[stage].append(dict(task_id=len(plans[stage]),stage=stage,protocol=VERSION,**kw))
    for dataset in datasets:
        for shape in shapes:
            for seed in reference_seeds:
                add('calibration',dataset=dataset,shape=shape,seed=seed,algorithm='improved_ILS',config_id='CA_REFERENCE')
            for seed in seeds:
                for tolerance in TOLERANCES:
                    add('rcl',dataset=dataset,shape=shape,seed=seed,algorithm='improved_GRASP',
                        config_id=f'RCL{round(tolerance*100)}',tolerance=tolerance,ls_budget=50)
            if dataset in tuning_datasets:
                for seed in seeds:
                    for config in tuning_configs():
                        add('tuning',dataset=dataset,shape=shape,seed=seed,**config)
            for seed in full_seeds:
                for algorithm in ('improved_ILS','improved_GRASP','improved_GRASP_ILS'):
                    for weight in LEVELS:
                        for radiation in LEVELS:
                            add('full',dataset=dataset,shape=shape,seed=seed,algorithm=algorithm,
                                config_id=f'{algorithm}_W{weight}_R{radiation}',weight_level=weight,
                                radio_level=radiation)
    return dict(protocol=VERSION,datasets=list(datasets),tuning_datasets=list(tuning_datasets),
                seeds=list(seeds),reference_seeds=list(reference_seeds),full_seeds=list(full_seeds),
                time_limit=time_limit,smoke=smoke,shapes=list(shapes),plans=plans)


def choose_ca_parameters(rows,datasets):
    """Historical CA data only; minimize dataset/shape mean N-U*, break ties by time."""
    selected={}
    for dataset in datasets:
        for shape in SHAPES:
            group=[r for r in rows if r['dataset']==dataset and r['bin_shape']==shape
                   and r.get('complete','').lower()=='true'
                   and r.get('constraint_mode')=='geometry_only']
            if len(group)!=12 or len({(r['nesting_strategy'],r['selection_range']) for r in group})!=12:
                raise ValueError(f'Expected 12 complete geometry-only CA groups: {dataset}/{shape}')
            row=min(group,key=lambda r:(float(r['best_N_mean'])-float(r['best_U_star_mean']),
                                       float(r['algorithm_time_s_mean']),r['nesting_strategy'],r['selection_range']))
            selected[f'{dataset}|{shape}']=dict(pes=row['nesting_strategy'],selection_range=row['selection_range'],
                reference_mean_objective=float(row['best_N_mean'])-float(row['best_U_star_mean']))
    return selected


def calibrate(records,weight_factor=1.):
    if not records or weight_factor<=0:
        raise ValueError('Calibration needs records and a positive weight factor')
    ustar=statistics.mean(r['best_U_star'] for r in records)
    nstar=statistics.mean(r['mean_piece_count_without_emptiest'] for r in records)
    volumes={r['bin_capacity'] for r in records}
    if len(volumes)!=1:
        raise ValueError('Reference bin sizes differ across order seeds')
    capacity=volumes.pop()
    largest=max(r['max_piece_volume'] for r in records)
    rmax=max(r['max_piece_radio'] for r in records)
    reference_radio=rmax+1000*max(0.,.2*nstar-1)
    # All-zero-radiation instances have an unconstraining positive implementation cap.
    reference_radio=max(reference_radio,1e-12)
    rho={level:max(mult*ustar,largest/capacity) for level,mult in zip(LEVELS,(.5,.75,1.))}
    return dict(U_star_reference=ustar,n_star_reference=nstar,bin_capacity=capacity,
                max_piece_volume=largest,max_piece_radio=rmax,weight_factor=weight_factor,
                weight_capacity={level:max(largest,rho[level]*capacity)*weight_factor for level in LEVELS},
                weight_capacity_ratio=rho,
                radiation_reference=reference_radio,
                radio_capacity={level:reference_radio*mult for level,mult in zip(LEVELS,(1.,1.5,2.))},
                reference_count=len(records),single_bin_convention='U*=U and n*=piece_count')


def select_tolerances(records):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in records:
        grouped[f"{r['dataset']}|{r['shape']}"][r['tolerance']].append(r['objective'])
    selected={}
    for key,values in grouped.items():
        if set(values)!=set(TOLERANCES):
            raise ValueError('Incomplete tolerance candidates')
        means={tau:statistics.mean(scores) for tau,scores in values.items()}
        best=min(means.values())
        ties=[tau for tau in sorted(means) if abs(means[tau]-best)<=1e-9]
        selected[key]=dict(tolerance=min(ties),tied_tolerances=ties,
                           mean_objectives={str(k):v for k,v in means.items()})
    return selected


def select_tuning(records):
    groups=defaultdict(lambda:defaultdict(list))
    for r in records:
        groups[(r['algorithm'],r['dataset'],r['shape'])][r['config_id']].append(r['objective'])
    rank_values=defaultdict(list)
    for (algorithm,dataset,shape),values in groups.items():
        expected={c['config_id'] for c in tuning_configs() if c['algorithm']==algorithm}
        if set(values)!=expected:
            raise ValueError('Incomplete tuning configurations')
        means={name:statistics.mean(scores) for name,scores in values.items()}
        for name,score in means.items():
            lower=sum(other<score-1e-9 for other in means.values())
            tied=sum(abs(other-score)<=1e-9 for other in means.values())
            rank_values[(algorithm,name)].append(1+lower+(tied-1)/2)
    output={}
    for algorithm in ('improved_GRASP','improved_ILS','improved_GRASP_ILS'):
        candidates={name:statistics.mean(v) for (alg,name),v in rank_values.items() if alg==algorithm}
        if not candidates:
            raise ValueError('No tuning data for '+algorithm)
        name=min(candidates,key=lambda n:(candidates[n],n))
        output[algorithm]=dict(next(c for c in tuning_configs() if c['config_id']==name),
                               mean_rank=candidates[name],ranks=candidates)
    return output
