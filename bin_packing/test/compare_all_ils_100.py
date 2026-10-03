"""Sequential, resumable benchmark of every seed-13 cube dataset."""
import csv
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
INSTANCES = Path(r'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing\instances')
ROOT = HERE / 'logs' / 'compare_all_ils_100'
ROOT.mkdir(parents=True, exist_ok=True)
rows = []
for instance in sorted(INSTANCES.glob('*_seq13_cube_999999_1.txt')):
    dataset = instance.name.removesuffix('_seq13_cube_999999_1.txt')
    output = ROOT / dataset
    output.mkdir(exist_ok=True)
    for algorithm in ['original', 'improved']:
        report_path = output / f'{algorithm}.json'
        if not report_path.exists():
            print('START', dataset, algorithm, flush=True)
            with (output / f'{algorithm}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, '-u', str(HERE / 'compare_ils_500.py'),
                    '--evaluations', '100', '--worker', algorithm, '--dataset', dataset, '--output', str(output)],
                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print('FAILED', dataset, algorithm, flush=True)
                break
    paths = [output / f'{a}.json' for a in ['original', 'improved']]
    if not all(p.exists() for p in paths):
        rows.append(dict(dataset=dataset, status='failed; see log'))
    else:
        old, new = [json.loads(p.read_text()) for p in paths]
        assert old['evaluations'] == new['evaluations'] == 100
        assert old['origin_N'] == new['origin_N']
        assert abs(old['origin_U_star'] - new['origin_U_star']) < 1e-12
        rows.append(dict(dataset=dataset, status='passed', original_seconds=old['wall_seconds'],
            improved_seconds=new['wall_seconds'], reduction_percent=100*(1-new['wall_seconds']/old['wall_seconds']),
            original_kicks=old['kicks'], improved_kicks=new['kicks'],
            original_best_N=old['best_N'], improved_best_N=new['best_N'],
            original_U_star=old['best_U_star'], improved_U_star=new['best_U_star']))
        print('DONE', dataset, flush=True)
    with (ROOT / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'status', 'original_seconds',
            'improved_seconds', 'reduction_percent', 'original_kicks', 'improved_kicks',
            'original_best_N', 'improved_best_N', 'original_U_star', 'improved_U_star'])
        writer.writeheader()
        writer.writerows(rows)
print('RESULTS', ROOT, flush=True)
