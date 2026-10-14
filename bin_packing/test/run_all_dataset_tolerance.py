"""Extend the frozen tolerance pilot to every locally available dataset."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
from types import SimpleNamespace

import run_tolerance_experiment as base

RATES = (.10, .15, .20)
SEEDS = (3, 7, 13)
SHAPES = ('cube', 'cylinder')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(c):
    return f"{c['dataset']}_{c['shape']}_seed{c['order_seed']}_t{round(c['tolerance']*100)}"


def discover():
    datasets = set()
    for p in (base.ROOT / 'bin_packing/instances').glob('*.txt'):
        m = re.fullmatch(r'(.+)_seq(\d+)_(cube|cylinder)_999999_1', p.stem)
        if m:
            datasets.add(m[1])
    if not datasets:
        raise RuntimeError('No datasets found')
    first = [x for x in ('chess', 'engine', 'Merged2_normal') if x in datasets]
    return first + sorted(datasets - set(first))


def input_hashes(datasets):
    paths = set()
    for name in datasets:
        for seed in SEEDS:
            for shape in SHAPES:
                p = base.ROOT / f'bin_packing/instances/{name}_seq{seed}_{shape}_999999_1.txt'
                paths.add(p)
                for line in p.read_text(encoding='utf-8').splitlines()[1:]:
                    if not line.strip():
                        continue
                    raw = re.split(r'\t+', line.strip())[0].replace('\\', '/')
                    if raw.startswith('../'):
                        raw = raw[3:]
                    asset = (base.ROOT / 'bin_packing/data' / raw).resolve()
                    if not asset.is_relative_to(base.ROOT.resolve()):
                        raise ValueError(f'Unexpected external asset: {asset}')
                    paths.add(asset)
    return {str(p.relative_to(base.ROOT)).replace('\\', '/'): digest(p) for p in sorted(paths)}


def verify_case(j, c):
    for field in ('dataset', 'shape', 'order_seed', 'tolerance'):
        assert j[field] == c[field], (label(c), field)
    assert j['status'] == 'success' and j['validation_errors'] == []
    assert j['piece_seed'] == 101 and j['position_seed'] == 202
    assert j['time_limit'] == 180 and j['ls_limit'] == 20
    assert j['selection_range'] == 'bottom_top' and j['pes'] == base.PES[c['shape']]
    assert abs(j['F'] - (j['best_N'] - j['best_U_star'])) < 1e-10
    h = {int(k): v for k, v in j['rcl_histogram'].items()}
    assert sum(h.values()) == j['rcl_calls']
    if j['rcl_calls']:
        assert abs(sum(k*v for k, v in h.items()) / j['rcl_calls'] - j['rcl_mean']) < 1e-10


def prepare(out, reuse):
    datasets = discover()
    inputs = input_hashes(datasets)
    hashes = {p: digest(base.ROOT / p) for p in base.SOURCE}
    runner_hash = digest(__file__)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert m['hashes'] == hashes and m['input_hashes'] == inputs
        assert m['runner_hash'] == runner_hash
        return m
    old = json.loads((reuse / 'manifest.json').read_text(encoding='utf-8'))
    assert old['hashes'] == hashes, 'Pilot sources do not match current sources'
    plan = []
    index = 0
    for dataset in datasets:
        for seed in SEEDS:
            for shape in SHAPES:
                rates = list(RATES)
                offset = index % 3
                for rate in rates[offset:] + rates[:offset]:
                    plan.append(dict(dataset=dataset, shape=shape, order_seed=seed,
                                     tolerance=rate, time_limit=180, iterations=20))
                index += 1
    planned = {label(c): c for c in plan}
    reused = []
    for c in old['plan']:
        name = label(c)
        assert name in planned
        j = json.loads((reuse / (name + '.json')).read_text(encoding='utf-8'))
        verify_case(j, planned[name])
        with (reuse / (name + '.rcl.csv')).open(newline='', encoding='utf-8-sig') as f:
            h = Counter(int(x['rcl_size']) for x in csv.DictReader(f))
        assert dict(h) == {int(k): v for k, v in j['rcl_histogram'].items()}
        for suffix in ('.json', '.rcl.csv', '.log'):
            shutil.copy2(reuse / (name + suffix), out / (name + suffix))
        reused.append(name)
    for rel in base.SOURCE:
        p = out / 'source_snapshot' / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.ROOT / rel, p)
    shutil.copy2(__file__, out / 'source_snapshot' / Path(__file__).name)
    m = dict(hashes=hashes, plan=plan, datasets=datasets, input_hashes=inputs,
             runner_hash=runner_hash, reused_from=str(reuse), reused_cases=reused,
             configuration=dict(seeds=list(SEEDS), shapes=list(SHAPES),
                                rates=list(RATES), piece_seed=101, position_seed=202))
    base.save_json(manifest_path, m)
    return m


def outcome(a, b):
    d = a-b
    return 'win' if d < -1e-10 else 'loss' if d > 1e-10 else 'tie'


def selection_rows(plan, results):
    valid = {(r['dataset'], r['shape'], r['order_seed'], r['tolerance']): r
             for r in results if r['status'] == 'success'}
    table = []
    for ds in dict.fromkeys(c['dataset'] for c in plan):
        for sh in SHAPES:
            row = dict(dataset=ds, shape=sh)
            means = {}
            for rate in RATES:
                group = [valid[(ds, sh, s, rate)] for s in SEEDS if (ds, sh, s, rate) in valid]
                tag = str(round(rate*100))
                row['valid_'+tag] = len(group)
                row['mean_F_'+tag] = statistics.mean(x['F'] for x in group) if group else None
                row['std_F_'+tag] = statistics.stdev(x['F'] for x in group) if len(group)>1 else None
                row['zero_ls_'+tag] = sum(x['ls_calls']==0 for x in group)
                calls = sum(x['rcl_calls'] for x in group)
                hist = Counter()
                for x in group:
                    hist.update({int(k): v for k, v in x['rcl_histogram'].items()})
                row['mean_RCL_'+tag] = sum(k*v for k,v in hist.items())/calls if calls else None
                row['RCL_3_4_percent_'+tag] = 100*(hist[3]+hist[4])/calls if calls else None
                means[rate] = row['mean_F_'+tag]
            complete = all(row['valid_'+str(round(t*100))] == len(SEEDS) for t in RATES)
            row['status'] = 'complete' if complete else 'partial'
            row['best_tolerance_percent'] = ''
            if complete:
                best = min(means.values())
                row['best_tolerance_percent'] = '/'.join(str(round(t*100)) for t in RATES if abs(means[t]-best)<=1e-10)
            for a,b in ((.15,.10),(.20,.10),(.15,.20)):
                tag = f'{round(a*100)}_vs_{round(b*100)}'
                outcomes = Counter(outcome(valid[(ds,sh,s,a)]['F'],valid[(ds,sh,s,b)]['F'])
                                   for s in SEEDS if (ds,sh,s,a) in valid and (ds,sh,s,b) in valid)
                row[tag+'_wins'] = outcomes['win']
                row[tag+'_ties'] = outcomes['tie']
                row[tag+'_losses'] = outcomes['loss']
            table.append(row)
    return table


original_summarize = base.summarize


def summarize(out, plan, results, running=None):
    original_summarize(out, plan, results, running)
    table = selection_rows(plan, results)
    base.write_csv(out/'best_tolerance_by_dataset.csv', table)
    complete = len(results) == len(plan)
    lines = ['# Dataset-specific RCL tolerance results', '',
             f'Status: {"finished" if complete else "partial"}; completed {len(results)}/{len(plan)}.',
             f'Current task: {running or "none"}.', '',
             'Each dataset and container shape is evaluated separately. Lower N-U* is better. '
             'Three order seeds (3/7/13), piece seed 101, position seed 202, 180 seconds, '
             '20 LS evaluations per restart; shape-specific PES and bottom/top candidates.',
             'The best rate is selected only when all nine cases in a dataset/shape group are valid. '
             'Missing or failed runs are never filled with zeros. Ties retain all tied rates.', '',
             '| Dataset | Shape | Valid runs 10/15/20 | Mean F 10% | Mean F 15% | Mean F 20% | Best rate | 15 vs 20 W/T/L |',
             '|---|---|---|---:|---:|---:|---|---|']
    for r in table:
        means = ['' if r['mean_F_'+t] is None else f"{r['mean_F_'+t]:.6f}" for t in ('10','15','20')]
        counts = '/'.join(str(r['valid_'+t]) for t in ('10','15','20'))
        wtl = '/'.join(str(r['15_vs_20_'+x]) for x in ('wins','ties','losses'))
        lines.append('| '+ ' | '.join([r['dataset'],r['shape'],counts,*means,
                                      str(r['best_tolerance_percent']) or 'pending',wtl])+' |')
    lines += ['', '## Interpretation', '',
              'These are parameter-selection results on the same three seeds, not independent '
              'confirmation of generalization. Validate chosen rates on held-out seeds before '
              'claiming a dataset-specific improvement. RCL means describe all evaluated '
              'orientation-level RCL calls, including orientations not ultimately packed.',
              'Some runs may exhaust the budget during construction and execute no LS. '
              'The CSV includes zero-LS counts per rate. Non-preemptive packing calls can exceed '
              'the nominal budget. Geometry validation checks IDs, boundaries, overlap and '
              'occupancy; full accessibility trajectories are not certified.', '',
              'Saved tables: best_tolerance_by_dataset.csv, summary_by_dataset.csv, '
              'paired_comparison.csv, results.csv, rcl_frequency.csv. Per-run JSON, RCL CSV, '
              'logs, manifest and source snapshots retain provenance.']
    (out/'best_tolerance_by_dataset.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if complete:
        errors = []
        for c in plan:
            try:
                j = json.loads((out/(label(c)+'.json')).read_text(encoding='utf-8'))
                verify_case(j,c)
                with (out/(label(c)+'.rcl.csv')).open(newline='',encoding='utf-8-sig') as f:
                    hist = Counter(int(x['rcl_size']) for x in csv.DictReader(f))
                assert dict(hist)=={int(k):v for k,v in j['rcl_histogram'].items()}
            except Exception as e:
                errors.append(dict(case=label(c), error=str(e) or repr(e)))
        base.save_json(out/'final_audit.json',dict(total=len(plan), valid=len(plan)-len(errors), errors=errors))
        (out/'final_report.md').write_text('\n'.join(lines)+f'\n\nFinal audit: {len(plan)-len(errors)}/{len(plan)} valid cases.\n',encoding='utf-8')


def self_test():
    plan=[dict(dataset='x', shape=s) for s in SHAPES]
    results=[]
    for s in SHAPES:
        for seed in SEEDS:
            for t,F in zip(RATES,(2.3,2.1,2.2)):
                results.append(dict(dataset='x',shape=s,order_seed=seed,tolerance=t,
                                    status='success',F=F,ls_calls=1,rcl_calls=2,rcl_histogram={'3':2}))
    rows=selection_rows(plan,results)
    assert all(r['best_tolerance_percent']=='15' and r['15_vs_20_wins']==3 for r in rows)
    results[0]['status']='failed'
    rows=selection_rows(plan,results)
    assert rows[0]['best_tolerance_percent']=='' and rows[0]['valid_10']==2
    for r in results:r['F']=2.1; r['status']='success'
    assert selection_rows(plan,results)[0]['best_tolerance_percent']=='10/15/20'
    print('PASS: paired comparison, failure exclusion, partial groups and ties')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--reuse')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.output or not args.reuse:parser.error('--output and --reuse are required')
        out=Path(args.output).resolve(); reuse=Path(args.reuse).resolve()
        os.chdir(base.TEST)
        m=prepare(out,reuse)
        base.summarize=summarize
        existing=[json.loads((out/(label(c)+'.json')).read_text(encoding='utf-8'))
                  for c in m['plan'] if (out/(label(c)+'.json')).exists()]
        summarize(out,m['plan'],existing)
        print(f"Plan: {len(m['plan'])}; reused/existing: {len(existing)}; remaining: {len(m['plan'])-len(existing)}",flush=True)
        if not args.prepare_only:
            assert input_hashes(m['datasets'])==m['input_hashes']
            base.parent(SimpleNamespace(output=str(out)))
