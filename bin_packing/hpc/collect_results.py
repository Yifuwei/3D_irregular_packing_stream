"""Collect completed Slurm tasks, preserving failures and missing datasets."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def collect(root, expected=None):
    if expected is None and (root / 'submission_manifest.json').exists():
        expected = json.loads((root / 'submission_manifest.json').read_text())['datasets']
    files = sorted(root.glob('task_*.csv'))
    if (root / 'comparison.csv').exists():
        files.append(root / 'comparison.csv')
    if not files and not expected:
        raise ValueError('No task CSV files found; jobs may still be running')
    rows, seen = [], set()
    for file in files:
        with file.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                key = (row['dataset'], row['repeat'], row['seed'], row.get('ls_seed', row['seed']))
                if key in seen:
                    raise ValueError(f'Duplicate result: {key}')
                seen.add(key)
                if row['status'] == 'passed':
                    if row['original_evaluations'] != row['improved_evaluations']:
                        raise ValueError(f'Unequal budgets: {key}')
                    if min(float(row['original_seconds']), float(row['improved_seconds'])) <= 0:
                        raise ValueError(f'Invalid timing: {key}')
                rows.append(row)
    if rows:
        with (root / 'collected_runs.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    names = set(row['dataset'] for row in rows)
    if expected:
        names.update(expected)
    summary = []
    for name in sorted(names):
        selected = [row for row in rows if row['dataset'] == name]
        valid = [row for row in selected if row['status'] == 'passed']
        item = dict(dataset=name, valid_pairs=len(valid), failed_pairs=len(selected)-len(valid),
                    status='missing' if not selected else 'partial/failed' if len(valid) < len(selected) else 'passed')
        if valid:
            old = [float(row['original_seconds']) for row in valid]
            new = [float(row['improved_seconds']) for row in valid]
            item.update(original_median_s=statistics.median(old), improved_median_s=statistics.median(new),
                original_mean_s=statistics.mean(old), improved_mean_s=statistics.mean(new),
                original_std_s=statistics.stdev(old) if len(old)>1 else None, improved_std_s=statistics.stdev(new) if len(new)>1 else None,
                original_min_s=min(old), original_max_s=max(old), improved_min_s=min(new), improved_max_s=max(new),
                paired_reduction_median_pct=statistics.median(100*(1-n/o) for o, n in zip(old, new)))
        if valid and all(row.get('original_N') and row.get('improved_N') for row in valid):
            item.update(original_N_mean=statistics.mean(float(row['original_N']) for row in valid),
                        improved_N_mean=statistics.mean(float(row['improved_N']) for row in valid),
                        improved_N_better=sum(float(row['improved_N']) < float(row['original_N']) for row in valid),
                        improved_N_equal=sum(float(row['improved_N']) == float(row['original_N']) for row in valid),
                        improved_N_worse=sum(float(row['improved_N']) > float(row['original_N']) for row in valid))
        if (root / 'submission_manifest.json').exists():
            manifest = json.loads((root / 'submission_manifest.json').read_text())
            if 'tasks' in manifest:
                expected_pairs = sum(task['dataset'] == name for task in manifest['tasks']) * manifest['repeats']
                if len(selected) < expected_pairs:
                    item['status'] = 'partial/missing'
        summary.append(item)
    fields = ['dataset', 'status', 'valid_pairs', 'failed_pairs', 'original_median_s', 'improved_median_s',
              'paired_reduction_median_pct', 'original_mean_s', 'improved_mean_s', 'original_std_s', 'improved_std_s',
              'original_N_mean', 'improved_N_mean', 'improved_N_better', 'improved_N_equal', 'improved_N_worse', 'original_min_s', 'original_max_s', 'improved_min_s', 'improved_max_s']
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = ['| Dataset | Status | Valid pairs | ILS median (s) | Improved median (s) | Paired reduction |',
             '|---|---|---:|---:|---:|---:|']
    for item in summary:
        def fmt(key):
            return f"{item[key]:.2f}" if key in item else '—'
        reduction = fmt('paired_reduction_median_pct')
        lines.append(f"| {item['dataset']} | {item['status']} | {item['valid_pairs']} | "
                     f"{fmt('original_median_s')} | {fmt('improved_median_s')} | {reduction} |")
    lines.extend(['', 'Reduction is a percentage; positive means improved is faster. Failures excluded from medians.',
                  'With one pair, the median is only a single measurement. See collected_runs.csv for budgets, kicks and solution quality.',
                  'One-bin continuation is a benchmark adaptation; these are complete-algorithm timings.'])
    (root / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--project', type=Path, help='bin_packing directory; include missing dataset names')
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    expected = None
    if args.project:
        suffix = f'_seq{args.seed}_cube_999999_1.txt'
        expected = [p.name.removesuffix(suffix) for p in (args.project / 'instances').glob('*' + suffix)]
    collect(args.results.resolve(), expected)
    print('Written collected_runs.csv, summary.csv and report.md')
