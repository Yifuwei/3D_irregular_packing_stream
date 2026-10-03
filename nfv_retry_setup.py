import sys, json, shutil, csv, statistics, math
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

ROOT = Path(r'C:\work\NFV_POOL_TESTING')
RUN = ROOT/'result/cpu_gpu_compare/full_20260930_134714_828297'
sys.path.insert(0, str(ROOT/'src'))

def main():
    import cpu_gpu_extreme as runner
    import cpu_gpu_full as full
    cases = json.loads((RUN/'results.json').read_text())
    meta = json.loads((RUN/'metadata.json').read_text())
    targets = [c for c in cases if c['cpu']['status']=='setup_timeout']
    assert len(targets)==1, 'Expected exactly one setup failure'
    c = targets[0]
    archive = RUN/('retry_setup_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    archive.mkdir()
    for p in RUN.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.md','.png','.svg'):
            shutil.copy2(p, archive/p.name)
    print('Retry CPU only:', {k:c[k] for k in ('case_id','instance_id','grid','piece','density','seed')},flush=True)
    result = runner.run_backend(c,'cpu',meta['arguments']['repeats'],meta['arguments']['cpu_timeout'])
    full.write_json(archive/'retry_result.json',result)
    hashes={m['candidate_sha256'] for e in (result,c['gpu']) for m in e['measurements']}
    assert len(hashes)<=1
    c['cpu']=result
    meta.setdefault('retry_history',[]).append(dict(date=datetime.now().isoformat(),case_id=c['case_id'],backend='cpu',original_status='setup_timeout',replacement_status=result['status'],archive=str(archive),reason='User requested one retry of setup failure; original results preserved'))
    full.write_json(RUN/'results.json',cases)
    full.write_json(RUN/'metadata.json',meta)
    full.save_report(RUN,cases,meta,figures=True)
    rows=list(csv.DictReader((RUN/'summary.csv').open()))
    extra=['Speedup comparison type','Conservative timeout comparison lower bound','Speedup display','Instance speedup comparisons']
    for row in rows:
        group=sorted([x for x in cases if (x['grid'],x['piece'],x['density'])==(int(row['Grid side length']),int(row['Moving-object bounding-box side length']),float(row['Fixed-grid occupancy probability']))],key=lambda x:x['instance_id'])
        bounds=[]; details=[]
        for x in group:
            cpu,gpu=x['cpu'],x['gpu']
            gm=statistics.mean(m['seconds'] for m in gpu['measurements'])
            if cpu['status']=='completed':
                value=statistics.mean(m['seconds'] for m in cpu['measurements'])/gm
                details.append(f"I{x['instance_id']}: {value:.2f}x measured")
            elif cpu['status']=='timeout' and cpu.get('timeout_phase')=='function':
                value=cpu['timeout_s']/gm; bounds.append(value)
                details.append(f"I{x['instance_id']}: >{math.floor(value*100)/100:.2f}x timeout comparison")
            else: details.append(f"I{x['instance_id']}: N/A ({cpu['status']})")
        row['Instance speedup comparisons']='; '.join(details)
        row['Conservative timeout comparison lower bound']=''
        if row['Measured speedup']:
            row['Speedup comparison type']='Measured ratio of complete hierarchical means'
            row['Speedup display']=f"{float(row['Measured speedup']):.2f}x"
        elif len(bounds)==3:
            row['Speedup comparison type']='Minimum of three timeout-call / GPU-instance-mean lower bounds'
            row['Conservative timeout comparison lower bound']=min(bounds)
            row['Speedup display']=f">{math.floor(min(bounds)*100)/100:.2f}x"
        else:
            row['Speedup comparison type']='Mixed outcomes; see instance comparisons'
            row['Speedup display']='Mixed'
    with (RUN/'summary_with_speedup.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    note='''\n\n## Updated retry and speedup interpretation\n\nThe single CPU setup failure was retried once at the user's request, using the same input, seed, three planned repetitions and 100 s per-call computation budget. Original files and the retry record are preserved in the retry_setup archive named in metadata.json. No other instance or GPU timing was rerun. The retry replaces the setup-failure record in current aggregates.\n\nMeasured speedup is the CPU/GPU ratio of hierarchical means only when all nine timings per backend are complete. For a computation-timeout instance, the reported comparison is >100 / GPU instance mean time. For three timeout instances, the group display uses the minimum of those three bounds. This bounds each observed timeout-call comparison against its GPU instance mean; it is NOT a bound on the unobserved nine-call mean speedup or a statistical confidence bound. Mixed outcomes are given individually, with no combined speedup. Bounds are rounded down for display. Setup failures never imply computation speedup bounds.\n\nA prior Windows worker-cleanup fix is documented in metadata.json source_revisions; NFV functions and timing limits were unchanged. These are implementation comparisons, not isolated hardware speedups.\n'''
    note+=f"\nRetry outcome: {result['status']}; successful retry timings: {len(result['measurements'])}.\n"
    for name in ('report.md','paper_text.md'):
        with (RUN/name).open('a',encoding='utf-8') as f:f.write(note)
    headers=['Grid side length','Moving-object bounding-box side length','Fixed-grid occupancy probability','Number of candidate translations','CPU complete instances','CPU timed-out instances','GPU mean seconds','Speedup display']
    lines=['# NFV CPU/GPU speedup table',note,'',' | '.join(headers),' | '.join(['---']*len(headers))]
    lines+=[' | '.join(str(row[h]) for h in headers) for row in rows]
    lines+=['','## Mixed outcomes']+[row['Instance speedup comparisons'] for row in rows if row['Speedup display']=='Mixed']
    (RUN/'speedup_table.md').write_text('\n'.join(lines),encoding='utf-8')
    assert len(rows)==64
    from collections import Counter
    print('CPU statuses:',dict(Counter(x['cpu']['status'] for x in cases)))
    print('Comparison types:',dict(Counter(row['Speedup comparison type'] for row in rows)))
    print('Finished:',RUN/'summary_with_speedup.csv',flush=True)

if __name__=='__main__':
    mp.freeze_support()
    main()
