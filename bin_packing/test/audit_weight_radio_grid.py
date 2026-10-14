"""Independently recompute logged scores, visit order and cache provenance."""
import argparse
import csv
import json
import math
from pathlib import Path


def audit(folder):
    record=json.loads((folder/'result.json').read_text())
    assert record['iterations']==100 and record['validated']
    assert all(f<=record['rho']+1e-12 for f in record['filling_rates'])
    assert all(r<=record['max_radio']+1e-9 for r in record['radiation'])
    prior={};rank=None;attempts=[];checks=0;skips=0;non_numeric=0
    def finish():
        if rank is None:return
        expected=[int(b)+1 for b,s in rank['ranked']]+[rank['existing_bin_count']+1]
        assert attempts==expected[:len(attempts)]
    for line in (folder/'events.jsonl').open(encoding='utf-8'):
        e=json.loads(line)
        if e['event']=='BIN_RANK':
            finish();rank=e;attempts=[];checks+=1
            assert e['ranked']==sorted(e['ranked'],key=lambda x:(-x[1],x[0]))
            ids=[d['bin'] for d in e['details']]
            non_numeric+=ids!=sorted(ids)
            for d in e['details']:
                score=d['filling_rate']*(d['max_radio']-d['current_radio']-d['piece_radio'])/d['max_radio']
                assert math.isclose(d['score'],score,rel_tol=1e-12,abs_tol=1e-12)
        elif e['event'] in ('SEARCH','RESTORE_ORIGINAL','REUSE_SUCCESS','SKIP_CACHED_FAILURE'):
            assert rank is not None and e['piece']==rank['piece']
            attempts.append(e['bin'])
        if e['event'] in ('SEARCH_FAILED','CA_SEARCH_FAILED'):prior[e['state_id']]=e
        if e['event']=='SKIP_CACHED_FAILURE':
            failed=prior[e['state_id']]
            assert all(failed[k]==e[k] for k in ['piece','bin','orientation'])
            skips+=1
    finish()
    assert checks>0
    return dict(max_radio=record['max_radio'],rho=record['rho'],rank_queries_checked=checks,
        non_numeric_priorities=non_numeric,cache_failure_skips_verified=skips,passed=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    audits=[audit(f) for f in sorted(a.root.glob('*_improved_ILS_weight_radio')) if (f/'result.json').exists()]
    assert len(audits)==9,len(audits)
    rows=list(csv.DictReader((a.root/'comparison.csv').open(encoding='utf-8-sig')))
    assert len(rows)==9
    summary=dict(runs=18,paired_settings=9,audits=audits,
        total_rank_queries_checked=sum(v['rank_queries_checked'] for v in audits),
        total_cache_skips_verified=sum(v['cache_failure_skips_verified'] for v in audits),
        same_candidate_tape_pairs=sum(r['same_candidate_tape']=='True' for r in rows),
        fewer_bins=sum(int(r['weight_radio_bins'])<int(r['improved_ILS_bins']) for r in rows),
        same_bins=sum(int(r['weight_radio_bins'])==int(r['improved_ILS_bins']) for r in rows),
        more_bins=sum(int(r['weight_radio_bins'])>int(r['improved_ILS_bins']) for r in rows),
        baseline_total_seconds=sum(float(r['improved_ILS_seconds']) for r in rows),
        ranked_total_seconds=sum(float(r['weight_radio_seconds']) for r in rows))
    summary['overall_time_reduction_percent']=100*(1-summary['ranked_total_seconds']/summary['baseline_total_seconds'])
    (a.root/'audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (a.root/'report.md').open('a',encoding='utf-8') as f:
        f.write(f"\nIndependent trace audit passed: {summary['total_rank_queries_checked']} metric/visit-order queries and {summary['total_cache_skips_verified']} failure-cache skips. Identical actual candidate tapes in {summary['same_candidate_tape_pairs']}/9 pairs.\n")
        f.write(f"\nWeight/radio produced fewer/same/more bins in {summary['fewer_bins']}/{summary['same_bins']}/{summary['more_bins']} settings. Summed formal wall time: {summary['baseline_total_seconds']:.2f}s versus {summary['ranked_total_seconds']:.2f}s; reduction {summary['overall_time_reduction_percent']:.2f}%. This aggregate is descriptive, not a statistical significance result.\n")
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
