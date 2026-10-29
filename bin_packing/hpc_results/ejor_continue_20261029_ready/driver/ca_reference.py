"""Recover calibration statistics from historical CA placements, without CUDA."""
import ast
from collections import Counter,defaultdict
import hashlib
import json
import math
import re

from protocol import calibrate


def index_results(root):
    results={}
    for path in sorted(root.rglob('*.json')):
        result=json.loads(path.read_text(encoding='utf-8-sig'))
        config=result.get('config',{})
        match=re.fullmatch(r'(.+)_seq(\d+)_(cube|cylinder)(?:_999999_1)?',config.get('instance_id',''))
        if not match or config.get('alg')!='fixed_CA' or config.get('constraint_mode')!='geometry_only':
            continue
        key=(*match.groups(),config['pes'],config['range'])
        if key in results:raise ValueError('Duplicate historical CA result: '+str(key))
        if result.get('status')!='success' or result.get('validation',{}).get('ok') is not True:
            raise ValueError('Unsuccessful historical CA result: '+str(path))
        if config.get('oes')!=config['pes']:
            raise ValueError('Historical OES differs from PES: '+str(path))
        results[key]=(path,result)
    return results


def recover(result,instance):
    lines=instance.read_text(encoding='utf-8-sig').splitlines()
    header=re.split(r'\t+',lines[0].strip())
    shape=header[0];size=tuple(ast.literal_eval(header[1]))
    if shape!=result['container_shape'] or tuple(result['container_size'])!=size:
        raise ValueError('Historical container differs from instance: '+str(instance))
    objects=[];volumes_by_type={}
    for line in lines[1:]:
        if not line.strip():continue
        fields=re.split(r'\t+',line.strip())
        volume,radio,kind=float(fields[1]),float(fields[2]),int(fields[3])
        if not math.isfinite(volume) or volume<=0 or not math.isfinite(radio) or radio<0:
            raise ValueError('Invalid input volume/radio')
        if kind in volumes_by_type and volumes_by_type[kind]!=volume:
            raise ValueError('Piece type does not uniquely identify volume: '+str(kind))
        volumes_by_type[kind]=volume
        objects.append((kind,volume,radio))
    placements=result['placements']
    if Counter(p['piece_type'] for p in placements)!=Counter(p[0] for p in objects):
        raise ValueError('Historical piece multiplicities differ from instance: '+str(instance))
    bin_volumes=defaultdict(float);bin_counts=Counter()
    for item in placements:
        b=item['bin']
        if not isinstance(b,int) or b<0:raise ValueError('Invalid historical bin index')
        bin_volumes[b]+=volumes_by_type[item['piece_type']]
        bin_counts[b]+=1
    n=len(bin_counts)
    if set(bin_counts)!=set(range(n)):raise ValueError('Historical bin indices are not contiguous')
    capacity=math.prod(size) if shape=='cube' else math.pi*(size[0]/2)**2*size[2]
    emptiest=min(bin_volumes,key=lambda b:(bin_volumes[b],b))
    u=sum(bin_volumes.values())/(n*capacity)
    us=(sum(bin_volumes.values())-bin_volumes[emptiest])/((n-1)*capacity) if n>1 else u
    metrics=result['metrics']
    for key,value in (('best_N',n),('best_U',u),('best_U_star',us)):
        if not math.isclose(metrics[key],value,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('Historical metric disagrees with placements: '+key+' / '+str(instance))
    return dict(best_N=n,best_U=u,best_U_star=us,objective=n-us,
        mean_piece_count_without_emptiest=(len(objects)-bin_counts[emptiest])/(n-1) if n>1 else len(objects),
        bin_capacity=capacity,max_piece_volume=max(p[1] for p in objects),
        max_piece_radio=max(p[2] for p in objects),piece_count=len(objects),
        bin_volumes=[bin_volumes[b] for b in range(n)],bin_piece_counts=[bin_counts[b] for b in range(n)],
        emptiest_bin=emptiest)


def recover_selected(manifest,project,root):
    indexed=index_results(root)
    records=[]
    for case in manifest['profile']['plans']['calibration']:
        config=manifest['ca_parameters'][f"{case['dataset']}|{case['shape']}"]
        key=(case['dataset'],str(case['seed']),case['shape'],config['pes'],config['selection_range'])
        if key not in indexed:raise ValueError('Missing reference CA result: '+str(key))
        path,result=indexed[key]
        instance=project/'instances'/case['instance']
        record=dict(recover(result,instance),case=case,source_file=str(path),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    historical_git_commit=result.get('git_commit'),
                    historical_config=result['config'],algorithm_wall_s=result.get('algorithm_time_s'))
        records.append(record)
    groups=defaultdict(list)
    for record in records:
        case=record['case'];groups[f"{case['dataset']}|{case['shape']}"].append(record)
    return records,{key:calibrate(values) for key,values in groups.items()},len(indexed)
