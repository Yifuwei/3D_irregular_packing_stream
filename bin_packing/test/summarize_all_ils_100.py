"""Keep a complete dataset table and local copies while the benchmark runs."""
import csv
import json
from pathlib import Path
import shutil
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'logs' / 'compare_all_ils_100'
LOCAL = Path(r'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing\test')
DEST = LOCAL / 'logs' / 'compare_all_ils_100'
DATASETS = sorted(p.name.removesuffix('_seq13_cube_999999_1.txt') for p in
    (LOCAL.parent / 'instances').glob('*_seq13_cube_999999_1.txt'))
DEST.mkdir(parents=True, exist_ok=True)
while True:
    rows = []
    for dataset in DATASETS:
        reports = {}
        failed = False
        source_dir = SOURCE / dataset
        target_dir = DEST / dataset
        target_dir.mkdir(exist_ok=True)
        if source_dir.exists():
            for path in source_dir.iterdir():
                if path.is_file():
                    shutil.copy2(path, target_dir / path.name)
            for name in ['original', 'improved']:
                path = source_dir / f'{name}.json'
                if path.exists():
                    reports[name] = json.loads(path.read_text())
                log = source_dir / f'{name}.log'
                if log.exists() and 'Traceback (most recent call last)' in log.read_text(encoding='utf-8'):
                    failed = True
        row = dict(dataset=dataset, status='failed' if failed else 'running/pending')
        for name, report in reports.items():
            row[name + '_seconds'] = report['wall_seconds']
            row[name + '_evaluations'] = report['evaluations']
            row[name + '_kicks'] = report['kicks']
        if len(reports) == 2:
            old, new = reports['original'], reports['improved']
            assert old['evaluations'] == new['evaluations'] == 100
            assert old['origin_N'] == new['origin_N']
            assert abs(old['origin_U_star'] - new['origin_U_star']) < 1e-12
            row['status'] = 'passed'
            row['reduction_percent'] = 100*(1-new['wall_seconds']/old['wall_seconds'])
        rows.append(row)
    fields = ['dataset', 'status', 'original_seconds', 'improved_seconds', 'reduction_percent',
              'original_evaluations', 'improved_evaluations', 'original_kicks', 'improved_kicks']
    temporary = DEST / 'comparison.tmp'
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(DEST / 'comparison.csv')
    lines = ['Each algorithm: 100 candidate evaluations including kicks; seed 13; cube; kick trigger 100.',
             'Independent processes; sequential execution; warm-up excluded; original fixes confined to test adapter.',
             '', '| Dataset | Status | ILS (s) | Improved ILS (s) | Time reduction |',
             '|---|---|---:|---:|---:|']
    def display(row, key, suffix=''):
        return f"{row[key]:.2f}{suffix}" if key in row else '—'
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['status']} | {display(row, 'original_seconds')} | "
                     f"{display(row, 'improved_seconds')} | {display(row, 'reduction_percent', '%')} |")
    lines.extend(['', 'Positive reduction means improved ILS is faster. These are complete-algorithm comparisons, not isolated skip-bin timings.'])
    (DEST / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    if all(row['status'] in ['passed', 'failed'] for row in rows):
        break
    time.sleep(60)
