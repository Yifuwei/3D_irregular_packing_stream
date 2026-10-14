"""Select largest dense input footprint and prepare three resource-only tasks."""
import argparse
import ast
import json
import math
from pathlib import Path
from metaheuristic_iterations import prepare


def largest_instance(project, seed):
    candidates=[]
    for shape in ('cube','cylinder'):
        suffix=f'_seq{seed}_{shape}_999999_1.txt'
        for path in (project/'instances').glob('*'+suffix):
            lines=path.read_text(encoding='utf-8-sig').splitlines()
            size=ast.literal_eval(lines[0].split('\t')[1])
            count=sum(bool(line.strip()) for line in lines[1:])
            footprint=count*math.prod(size)
            candidates.append((footprint,path.name.removesuffix(suffix),shape,count,size))
    if not candidates:
        raise ValueError('No instances found')
    return max(candidates)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--seed',type=int,default=3)
    parser.add_argument('--time-limit',type=float,default=3600)
    parser.add_argument('--kick-trigger',type=int,default=100)
    args=parser.parse_args()
    if args.manifest.exists():
        parser.error('Use a fresh manifest path')
    if args.kick_trigger<1:
        parser.error('kick-trigger must be positive')
    project=args.project.resolve()
    footprint,dataset,shape,count,size=largest_instance(project,args.seed)
    manifest=prepare(project,[dataset],[shape],[args.seed],args.time_limit)
    tasks=[task for task in manifest['tasks'] if task['config_id'] in ('G150','I100','GI300_K100')]
    for index,task in enumerate(tasks):
        task['task_id']=index
        if task['kick_trigger'] is not None:
            task['kick_trigger']=args.kick_trigger
        task['config_id']='MEMORY_'+task['algorithm']
    manifest['tasks']=tasks
    manifest['resource_probe']=dict(selection='largest piece_count * dense grid voxels',
        dataset=dataset,shape=shape,piece_count=count,grid_size=size,
        dense_input_voxels=footprint,kick_trigger=args.kick_trigger,
        warning='Resource-only results; do not merge into iteration-performance experiment. Largest input does not guarantee largest search peak.')
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['resource_probe']))


if __name__=='__main__':
    main()
