"""Wait for the current round, then repeat Merged2 timing and trace bin skips once."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LOCAL = Path(r'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing\test')
STATUS = LOCAL / 'logs' / 'compare_all_ils_100' / 'comparison.csv'
OUTPUT = HERE / 'logs' / 'merged2_repeat_100'
OUTPUT.mkdir(parents=True, exist_ok=True)
MARKER = OUTPUT / 'started.json'
if MARKER.exists():
    raise SystemExit('Repeat already started; refusing a duplicate run')
print('Waiting for all 15 datasets to finish the current round.', flush=True)
while True:
    with STATUS.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) == 15 and all(row['status'] in ['passed', 'failed'] for row in rows):
        break
    time.sleep(60)
with MARKER.open('x', encoding='utf-8') as stream:
    json.dump({'dataset': 'Merged2_normal', 'evaluations': 100, 'seed': 13}, stream)
commands = [
    [sys.executable, '-u', str(HERE / 'compare_ils_500.py'), '--dataset', 'Merged2_normal',
     '--evaluations', '100', '--seed', '13', '--output', str(OUTPUT)],
    [sys.executable, '-u', str(HERE / 'run_logged_ils_case.py'), '--dataset', 'Merged2_normal',
     '--iterations', '100', '--time-limit', 'inf', '--kick-trigger', '100',
     '--tag', 'merged2_skip_validation_100'],
]
try:
    for command in commands:
        print('RUN', command, flush=True)
        subprocess.run(command, check=True)
    (OUTPUT / 'completed.json').write_text(json.dumps({'status': 'passed'}))
except Exception as error:
    (OUTPUT / 'failed.json').write_text(json.dumps({'error': str(error)}))
    raise
finally:
    target = LOCAL / 'logs' / OUTPUT.name
    shutil.copytree(OUTPUT, target, dirs_exist_ok=True)
    for name in ['merged2_skip_validation_100.log', 'merged2_skip_validation_100_summary.json']:
        path = HERE / 'logs' / name
        if path.exists():
            shutil.copy2(path, LOCAL / 'logs' / name)
