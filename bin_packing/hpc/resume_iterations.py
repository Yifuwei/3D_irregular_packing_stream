"""Plan a safe restart without changing algorithm parameters or successful results."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def check_sources(project, manifest):
    hashes = {}
    for name, expected in manifest['source_hashes'].items():
        data = (project/name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        # A Windows git checkout may change LF to CRLF; accept only this difference.
        normalized = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
        if expected not in (actual, normalized):
            raise ValueError(f'Search source changed: {name}. Do not mix changed algorithms into this experiment.')
        hashes[name] = actual
    return hashes


def plan(result_dir):
    manifest = json.loads((result_dir/'manifest.json').read_text(encoding='utf-8'))
    pending, complete, archive = [], [], []
    for task in manifest['tasks']:
        task_id = task['task_id']
        report_path = result_dir/f'task_{task_id}.json'
        success = False
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding='utf-8'))
                success = (all(report.get(k)==v for k,v in task.items())
                           and report.get('status')=='success' and report.get('validation_ok') is True
                           and report.get('settings',{}).get('time_limit')==manifest['time_limit'])
            except (ValueError, OSError):
                pass
        if success:
            complete.append(task_id)
        else:
            pending.append(task_id)
            for path in (report_path, result_dir/f'task_{task_id}_events.jsonl'):
                if path.exists():
                    archive.append(path)
    return manifest, complete, pending, archive


def array_spec(indices):
    if not indices:
        return ''
    indices = sorted(indices)
    segments = []
    start = end = indices[0]
    for index in indices[1:]:
        if index==end+1:
            end=index
        else:
            segments.append(str(start) if start==end else f'{start}-{end}')
            start=end=index
    segments.append(str(start) if start==end else f'{start}-{end}')
    return ','.join(segments)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write-manifest',type=Path)
    parser.add_argument('--archive-incomplete',action='store_true')
    args=parser.parse_args()
    result_dir=args.results.resolve()
    manifest,complete,pending,archive=plan(result_dir)
    hashes=check_sources(args.project.resolve(),manifest)
    if args.archive_incomplete and args.write_manifest is None:
        parser.error('archive-incomplete requires write-manifest')
    if args.write_manifest:
        if args.write_manifest.exists():
            raise FileExistsError('Resume manifest already exists')
        resumed=dict(manifest,source_hashes=hashes,
                     resume=dict(completed_task_ids=complete,pending_task_ids=pending,
                                 original_manifest=str(result_dir/'manifest.json'),
                                 resource_change='16GB host memory; search settings unchanged'))
        args.write_manifest.write_text(json.dumps(resumed,indent=2),encoding='utf-8')
    if args.archive_incomplete and archive:
        destination=result_dir/'resume_archive'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        destination.mkdir(parents=True)
        for path in archive:
            shutil.move(str(path),str(destination/path.name))
    print(json.dumps(dict(completed=len(complete),pending=len(pending),
                          array=array_spec(pending),incomplete_files_to_archive=len(archive))))


if __name__=='__main__':
    main()
