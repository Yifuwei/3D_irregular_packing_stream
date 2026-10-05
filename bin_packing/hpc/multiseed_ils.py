"""Prepare/run paired sequence and LS seeds, one Slurm task per dataset/seed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def hashes(project):
    paths = list((project / 'src').glob('*.py')) + [Path(__file__).with_name('benchmark_ils.py')]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(project, seeds, evaluations=100, repeats=1, kick_trigger=100):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and distinct')
    if min(evaluations, repeats, kick_trigger) < 1:
        raise ValueError('Budgets and kick threshold must be positive')
    names = None
    tasks = []
    for seed in seeds:
        suffix = f'_seq{seed}_cube_999999_1.txt'
        files = sorted((project / 'instances').glob('*' + suffix))
        present = {p.name.removesuffix(suffix) for p in files}
        if not present or (names is not None and present != names):
            raise ValueError(f'Dataset coverage differs/missing at sequence seed {seed}')
        names = present
        for instance in files:
            tasks.append(dict(dataset=instance.name.removesuffix(suffix), sequence_seed=seed,
                ls_seed=seed, instance=instance.name, instance_sha256=hashlib.sha256(instance.read_bytes()).hexdigest()))
    return dict(datasets=sorted(names), seeds=seeds, tasks=tasks, source_hashes=hashes(project),
                evaluations=evaluations, repeats=repeats, kick_trigger=kick_trigger,
                design='paired seeds: sequence and LS seed vary together; not a factorial design')


def run(project, manifest, index, output):
    if manifest['source_hashes'] != hashes(project):
        raise ValueError('Source changed after submission')
    if not 0 <= index < len(manifest['tasks']):
        raise ValueError('Task index outside manifest')
    task = manifest['tasks'][index]
    instance = project / 'instances' / task['instance']
    if hashlib.sha256(instance.read_bytes()).hexdigest() != task['instance_sha256']:
        raise ValueError('Instance changed after submission')
    target = output / f'task_{index}_runs'
    command = [sys.executable, '-u', str(Path(__file__).with_name('benchmark_ils.py')),
        '--project', str(project), '--dataset', task['dataset'], '--seed', str(task['sequence_seed']),
        '--ls-seed', str(task['ls_seed']), '--evaluations', str(manifest['evaluations']),
        '--repeats', str(manifest['repeats']), '--kick-trigger', str(manifest['kick_trigger']),
        '--output', str(target)]
    # Balance execution order across tasks by reusing the repeat-offset option.
    command += ['--order-offset', str(index % 2)]
    result = subprocess.run(command)
    paired = target / 'comparison.csv'
    if paired.exists():
        shutil.copy2(paired, output / f'task_{index}.csv')
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--seeds', nargs='+', type=int, default=[3, 7, 19, 53])
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evaluations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--kick-trigger', type=int, default=100)
    args = parser.parse_args()
    project = args.project.resolve()
    if args.prepare:
        manifest = prepare(project, args.seeds, args.evaluations, args.repeats, args.kick_trigger)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(len(manifest['tasks']))
    else:
        if args.task_index is None or args.output is None:
            parser.error('run requires --task-index and --output')
        run(project, json.loads(args.manifest.read_text()), args.task_index, args.output.resolve())
