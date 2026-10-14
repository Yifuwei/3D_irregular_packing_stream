"""Audit real rank/attempt traces and failure-cache reuse; produce a visual."""
import argparse
from collections import Counter
import json
from pathlib import Path
import math


def inspect_run(folder):
    result=json.loads((folder/'result.json').read_text())
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    prior_failures={};skips=[];examples=[];rank_checks=0;attempt_checks=0
    rank=None;attempts=[]
    def finish():
        nonlocal attempt_checks
        if rank is None:return
        expected=[int(b)+1 for b,_ in rank['ranked']]+[rank['existing_bin_count']+1]
        assert attempts==expected[:len(attempts)],(rank,attempts,expected)
        attempt_checks+=1
        if len(rank['details'])>=2 and expected[:len(rank['details'])]!=sorted(expected[:len(rank['details'])]):
            examples.append(dict(rank=rank,attempts=list(attempts)))
    for e in events:
        if e['event']=='BIN_RANK':
            finish();rank=e;attempts=[]
            pairs=e['ranked'];assert pairs==sorted(pairs,key=lambda x:(-x[1],x[0]))
            for d in e['details']:
                expected=d['filling_rate']*((d['max_radio']-d['current_radio']-d['piece_radio'])/d['max_radio'])
                assert math.isclose(d['score'],expected,rel_tol=1e-12,abs_tol=1e-12)
            rank_checks+=1
        if e['event'] in ('SEARCH','RESTORE_ORIGINAL','SKIP_CACHED_FAILURE','REUSE_SUCCESS'):
            assert rank is not None and e['piece']==rank['piece']
            attempts.append(e['bin'])
        if e['event'] in ('SEARCH_FAILED','CA_SEARCH_FAILED'):prior_failures[e['state_id']]=e
        if e['event']=='SKIP_CACHED_FAILURE':
            assert e['state_id'] in prior_failures
            old=prior_failures[e['state_id']]
            assert (old['piece'],old['bin'],old['orientation'])==(e['piece'],e['bin'],e['orientation'])
            skips.append(dict(failed=old,skipped=e))
    finish()
    assert all(v<=result['max_radio']+1e-9 for v in result['best_radio'])
    assert all(v<=result['max_weight']+1e-9 for v in result['best_weight'])
    repeat=[e for e in events if e['phase']=='repeat_second']
    assert repeat and not any(e['event']=='SEARCH' for e in repeat)
    assert any(e['event']=='SKIP_CACHED_FAILURE' for e in repeat)
    audit=dict(max_radio=result['max_radio'],initial_bins=result['initial_bins'],best_bins=result['best_bins'],
        initial_counts=result['initial_counts'],best_counts=result['best_counts'],
        ranks_checked=rank_checks,attempt_orders_checked=attempt_checks,
        events=dict(Counter(e['event'] for e in events)),
        ils_events=dict(Counter(e['event'] for e in events if e['phase']=='ils')),
        repeat_second_events=dict(Counter(e['event'] for e in repeat)),
        ranking_example=examples[0] if examples else None,
        cache_example=next((s for s in skips if not s['skipped']['changed'] and s['skipped']['phase']=='ils'),skips[0] if skips else None),
        passed=True)
    (folder/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args()
    audits=[inspect_run(a.root/f'radio_{cap}') for cap in [2000,3000,4000]]
    example=next(r for r in audits if r['ranking_example'] and r['cache_example'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cache_run=next(r for r in audits if r['cache_example'] and not r['cache_example']['skipped']['changed'])
    rank=example['ranking_example']['rank'];ds=rank['details'];cache=cache_run['cache_example']
    fig,axes=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[1,1.45]})
    ax=axes[0];labels=[f"Bin {d['bin']}" for d in ds]
    ax.bar(labels,[d['score'] for d in ds],color='#2d7a93')
    ax.set_ylabel('Metric score');ax.set_title(f"Actual ranked query: Rmax={example['max_radio']:g}, piece {rank['piece']} (0-based)")
    for j,d in enumerate(ds):
        ax.text(j,d['score'],f"F={d['filling_rate']:.3f}\nremaining={d['remaining_ratio']:.3f}",ha='center',va='bottom',fontsize=9)
    ax.set_ylim(0,max(d['score'] for d in ds)*1.5+0.01)
    ax=axes[1];ax.axis('off')
    order=' -> '.join(f"Bin {d['bin']}" for d in ds)
    tried=' -> '.join(f'Bin {b}' for b in example['ranking_example']['attempts'])
    old,new=cache['failed'],cache['skipped']
    message=(f"Metric priority: {order}\nActual visits: {tried}\n\n"
        f"Failure-cache proof (Rmax={cache_run['max_radio']:g}, real chess):\n"
        f"Piece {new['piece']} / bin {new['bin']} / orientation {new['orientation']}\n"
        f"First: {old['phase']}, repack {old['repack_id']} -> SEARCH_FAILED\n"
        f"Later: {new['phase']}, repack {new['repack_id']} -> SKIP_CACHED_FAILURE\n"
        f"Identical state ID: {new['state_id']}\n"
        f"changed flag at skip: {new['changed']}\n\n"
        "This skip occurred naturally within the ILS run.\n"
        "Every recorded metric and attempted-bin order was audited.\n"
        "Separate repeat tests also required zero second-repack searches.")
    ax.text(0,0.95,message,va='top',fontsize=11,linespacing=1.7)
    fig.suptitle('Ranked-bin search and safe skipping: log-backed example',fontsize=15)
    fig.tight_layout();fig.savefig(a.root/'logic_evidence.png',dpi=180);plt.close(fig)
    (a.root/'summary.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    lines=['# Chess weight/radiation logic verification','',
        'Real chess radiation proxies; seed 13; 10 ILS evaluations per limit. Weight factor=1 and per-bin weight cap=80% container volume.',
        'Each run also repeats an identical initial-state repack twice to explicitly verify failure-cache reuse; these repetitions are not ILS evaluations.',
        '', '| Radio limit | Initial bins | Best bins | ILS cached-failure skips | Repeat-test skips | Metric/order queries audited |',
        '|---|---:|---:|---:|---:|---:|']
    for r in audits:
        lines.append(f"| {r['max_radio']:g} | {r['initial_bins']} | {r['best_bins']} | {r['ils_events'].get('SKIP_CACHED_FAILURE',0)} | {r['repeat_second_events'].get('SKIP_CACHED_FAILURE',0)} | {r['ranks_checked']} |")
    lines+=['','All hard constraints, metric recomputations, attempt order and matching failure history checks passed.',
        'Piece indices are zero-based; bin labels are one-based. This is logic validation, not evidence of global optimality or performance gains.']
    (a.root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__': main()
