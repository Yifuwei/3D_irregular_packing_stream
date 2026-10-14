"""Profile multiple RCL tolerances on identical 10%-construction trajectories."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

RATES = (.05, .10, .15, .20)


def worker(args):
    import numpy as np
    import code_launcher as launcher
    from numba import cuda
    if not cuda.is_available():
        raise RuntimeError('A real CUDA environment is required')
    events, placements = [], {}
    actual = launcher.improved_GRASP
    def captured(*a, **kw):
        result = actual(*a, **kw)
        # This profiling run always ends after one complete initial construction.
        for objects, ids in zip(result[7], result[9]):
            for info, index in zip(objects, ids):
                placements[index] = (int(info['bin_position']), info['orientation'],
                                     tuple(int(v) for v in info['translation']))
        if any(np.any(array > 1) for array in result[8]):
            raise RuntimeError('Overlapping output voxels')
        return result
    launcher.improved_GRASP = captured
    rows = launcher.testing([args.dataset], [args.seed], ['improved_GRASP'], ['cube'],
                            experiment_iterations=1, experiment_time=1e-9,
                            selection_ranges=['bottom_top'], nesting_strategies=[args.pes],
                            rcl_tolerance=.10, piece_seed=101, position_seed=202,
                            rcl_observer=events.append)
    for i, event in enumerate(events, 1):
        event.update(call_id=i, dataset=args.dataset, order_seed=args.seed,
                     piece_seed=101, position_seed=202, pes_strategy=args.pes)
        event['candidate_count'] = len(event['candidates'])
        key = (event['bin_index'], event['orientation'], tuple(event['selected_position'] or []))
        event['used_for_placement'] = key == placements.get(event['piece_index'])
        for rate in RATES:
            label = f'rcl_{int(rate*100)}pct'
            if event['best_pes'] is None:
                event[label] = 0
            else:
                threshold = event['best_pes'] + rate * abs(event['best_pes'])
                eps = 8 * np.finfo(float).eps * max(1.0, abs(threshold))
                event[label] = int(sum(c['pes'] <= threshold + eps for c in event['candidates']))
        assert event['active_rcl_size'] == event['rcl_10pct']
        sizes = [event[f'rcl_{int(r*100)}pct'] for r in RATES]
        assert sizes == sorted(sizes)
    Path(args.output).write_text(json.dumps(dict(events=events, launcher_row=rows[0]), indent=2,
                                          default=lambda value: value.item()), encoding='utf-8')


def parent(args):
    import numpy as np
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    events = []
    for seed in args.seeds:
        label = f'{args.dataset}_seed{seed}'
        output = out / (label+'.json')
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--dataset', args.dataset,
               '--seed', str(seed), '--pes', args.pes, '--output', str(output)]
        print('Profiling '+label, flush=True)
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            process = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if process.returncode:
            print((out/(label+'.log')).read_text(encoding='utf-8')[-4000:],flush=True)
            raise RuntimeError(label+' failed')
        data=json.loads(output.read_text(encoding='utf-8'))
        events.extend(data['events'])
        nonempty=[e for e in data['events'] if e['candidate_count']]
        print(f"{label}: {len(nonempty)} candidate-bearing RCL calls; mean 10% size = {np.mean([e['rcl_10pct'] for e in nonempty]):.3f}",flush=True)
    fields=['dataset','order_seed','call_id','restart','piece_index','bin_index','orientation',
            'pes_strategy','candidate_count','best_pes','active_rcl_size','used_for_placement',
            'rcl_5pct','rcl_10pct','rcl_15pct','rcl_20pct']
    with (out/'per_call.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(events)
    summary=[]
    groups=[('all_calls',None,[e for e in events if e['candidate_count']]),
            ('used_for_placement',None,[e for e in events if e['candidate_count'] and e['used_for_placement']])]
    groups += [('per_order_seed',seed,[e for e in events if e['candidate_count'] and e['order_seed']==seed]) for seed in args.seeds]
    for scope,seed,group in groups:
        for rate in RATES:
            if not group:
                continue
            sizes=np.array([e[f'rcl_{int(rate*100)}pct'] for e in group])
            summary.append(dict(scope=scope,order_seed=seed,tolerance_percent=int(rate*100),
                calls=len(group),mean=float(sizes.mean()),median=float(np.median(sizes)),
                q25=float(np.quantile(sizes,.25)),q75=float(np.quantile(sizes,.75)),
                minimum=int(sizes.min()),maximum=int(sizes.max()),
                percent_size_1=float(100*np.mean(sizes==1)),
                percent_size_3_to_4=float(100*np.mean((sizes>=3)&(sizes<=4))),
                percent_size_at_most_4=float(100*np.mean(sizes<=4)),
                percent_all_candidates=float(100*np.mean(sizes==np.array([e['candidate_count'] for e in group])))))
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,list(summary[0]));writer.writeheader();writer.writerows(summary)
    distribution=[]
    for rate in RATES:
        sizes=[e[f'rcl_{int(rate*100)}pct'] for e in events if e['candidate_count']]
        for size in sorted(set(sizes)):
            count=sizes.count(size)
            distribution.append(dict(tolerance_percent=int(rate*100),rcl_size=size,count=count,percent=100*count/len(sizes)))
    with (out/'distribution.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,list(distribution[0]));writer.writeheader();writer.writerows(distribution)
    report=['# Position RCL size profile','',
        f'Dataset: {args.dataset}; order seeds: {args.seeds}; PES: `{args.pes}`; cube bins; piece seed 101; position seed 202.',
        'Each seed runs exactly one complete construction through code_launcher.testing with the actual position tolerance fixed at 10%. The negligible LS time budget stops after construction; no LS or restart is profiled.',
        'All four tolerance sizes are computed from the same unique scored candidates, using best + tolerance * abs(best) and the production roundoff allowance. No extra random numbers are drawn for observation.',
        'A call is one nonempty position RCL for one piece/orientation/bin query. All evaluated orientations are counted, including those not ultimately placed. Empty-bin starting placements and AFR queries that fail before RCL creation are not RCL calls.',
        'Means weight each nonempty RCL call equally. used_for_placement reports only RCL calls matching final constructed placements; empty-bin placements remain excluded.',
        '', '| Scope | Tolerance | Calls | Mean size | Median | Q25–Q75 | Size 1 | Size 3–4 | All candidates |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in summary:
        if row['scope']=='per_order_seed':continue
        report.append(f"| {row['scope']} | {row['tolerance_percent']}% | {row['calls']} | {row['mean']:.3f} | {row['median']:.1f} | {row['q25']:.1f}–{row['q75']:.1f} | {row['percent_size_1']:.1f}% | {row['percent_size_3_to_4']:.1f}% | {row['percent_all_candidates']:.1f}% |")
    report += ['', 'A mean near 3–4 is a selectivity target, not proof of better final packing. All retained candidates meet the local PES threshold, but PES is not the final N-U* objective. The raw JSON files retain every scored candidate; per_call.csv lists all call sizes and summary.csv includes per-order-seed means.']
    report += ['', '## RCL size frequencies (all evaluated nonempty RCL calls)', '',
               '| Tolerance | Size 1 | Size 2 | Size 3 | Size 4 | Size >=5 |',
               '| --- | --- | --- | --- | --- | --- |']
    for rate in (5, 10, 15, 20):
        frequency = {d['rcl_size']: d['count'] for d in distribution if d['tolerance_percent'] == rate}
        total = sum(frequency.values())
        counts = [frequency.get(i, 0) for i in (1, 2, 3, 4)]
        counts.append(sum(c for size, c in frequency.items() if size >= 5))
        report.append('| '+str(rate)+'% | '+' | '.join(f'{c} ({100*c/total:.1f}%)' for c in counts)+' |')
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Report:',out/'report.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dataset',default='chess')
    parser.add_argument('--seed',type=int,default=13)
    parser.add_argument('--seeds',nargs='+',type=int,default=[3,7,13,19,53])
    parser.add_argument('--pes',default='maximal_residual_box')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    worker(args) if args.worker else parent(args)
