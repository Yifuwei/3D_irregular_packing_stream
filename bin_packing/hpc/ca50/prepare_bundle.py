"""Create an uploadable, source-frozen CA+50 LS Slurm experiment bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def prepare(experiment,output):
    source=json.loads((experiment/'manifest.json').read_text(encoding='utf-8'))
    if source['protocol']!='one_CA_plus_50_LS':raise ValueError('Expected CA+50 experiment')
    runtime=experiment/'frozen_runtime'
    for rel,h in dict(source['hashes'],**source['input_hashes']).items():
        if hashlib.sha256((runtime/rel).read_bytes()).hexdigest()!=h:
            raise ValueError('Frozen source/input changed: '+rel)
    worker=runtime/'bin_packing/test/run_ca50_tolerance_experiment.py'
    if hashlib.sha256(worker.read_bytes()).hexdigest()!=source['runner_hash']:
        raise ValueError('CA+50 runner changed')
    if len(source['plan'])!=270 or len({(x['dataset'],x['shape'],x['order_seed'],x['tolerance']) for x in source['plan']})!=270:
        raise ValueError('Expected exactly 270 unique experiment cases')
    if output.exists():raise FileExistsError('Use a new output directory: '+str(output))
    output.mkdir(parents=True)
    shutil.copytree(runtime,output/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    template=Path(__file__).resolve().parent
    for name in ['task.py','collect.py','run.slurm','submit.sh','README.md']:
        # Normalize shell scripts to LF for Linux execution after Windows packaging.
        (output/name).write_text((template/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    (output/'original_experiment_manifest.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    for rel in source['input_hashes']:
        p=output/'runtime'/rel
        # Check Linux-sensitive component casing against the actual bundle inventory.
        parent=output
        for part in p.relative_to(output).parts:
            if part not in {x.name for x in parent.iterdir()}:
                raise ValueError('Path casing is not portable to Linux: '+str(p))
            parent=parent/part
    files={str(p.relative_to(output)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(output.rglob('*')) if p.is_file()}
    manifest=dict(protocol=source['protocol'],plan=source['plan'],configuration=source['configuration'],
                  files_sha256=files,source_experiment=str(experiment),
                  resources=dict(default_memory_gb=32,small_memory_gb=16,
                                 small_datasets=['chess','shapesnew'],group_order=[16,32],
                                 concurrency='Dependent arrays; global maximum MAX_PARALLEL'),
                  source_runner_hash=source['runner_hash'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=output.with_suffix('.zip')
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,arcname=str(Path(output.name)/p.relative_to(output)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP integrity failure')
    print(f'Prepared {len(source["plan"])} tasks; bundle: {output}; upload archive: {archive}')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();prepare(a.experiment.resolve(),a.output.resolve())
