"""Local controlled ILS diagnosis; modifies benchmark copies, never production code.

Runs improved, original with the same selection policy, then original replaying
the improved candidate tape. Equal policy/seed does not ensure equal choices:
accepted layouts and neighborhood stopping differ. Replay ensures equal
piece AND orientation per evaluation; accepted layouts can still diverge.
"""
import argparse
import ast
from collections import defaultdict
import csv
import inspect
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


class Probe:
    def __init__(self, ils, output, mode, tape_path):
        self.ils, self.output, self.mode = ils, output, mode
        self.stats = defaultdict(lambda: dict(calls=0, seconds=0.0))
        self.events = defaultdict(int)
        self.candidates, self.selections = [], []
        self.tape = json.loads(tape_path.read_text()) if mode == 'replay' else []
        self.exclude = True
        self.last_local, self.age = 'origin', 0
        self.reference = None
        self.active = False
        self.changed_reasons = {}
        self.bin_summaries = []
        self.trace = (output / 'bin_events.jsonl').open('w', encoding='utf-8', buffering=1)

    def emit(self, event, **values):
        self.trace.write(json.dumps(dict(evaluation=len(self.candidates), event=event, **values))+'\n')

    def begin(self, args):
        data = args[5][args[6]]
        self.changed_reasons = {}
        self.emit('REPACK_START', selected_piece=int(args[1][0]), orientation=args[1][1],
            reference_counts=[len(b) for b in data['pieces_order']],
            reference_piece_ids=data['pieces_order'])

    def finish(self, result):
        counts = [len(b) for b in result[3]]
        self.bin_summaries.append(dict(evaluation=len(self.candidates), counts=counts))
        self.emit('REPACK_END', counts=counts, piece_ids=result[3])

    def mark(self, flags, index, cause, piece):
        was_changed = bool(flags[index])
        flags[index] = True
        self.changed_reasons.setdefault(int(index), []).append(cause)
        self.emit('MARK_CHANGED', bin=int(index)+1, piece=int(piece), cause=cause, previously_changed=was_changed)

    def visit(self, piece, selected, old_bin, target, flags, layout):
        if piece == selected:
            reason = 'SELECTED_PIECE_MUST_TRY'
        elif flags[target]:
            reason = 'BIN_CHANGED_CANNOT_SKIP'
        elif target < old_bin:
            reason = 'SKIP_UNCHANGED_EARLIER'
        elif target == old_bin:
            reason = 'RESTORE_UNCHANGED_ORIGINAL'
        else:
            reason = 'UNCHANGED_LATER_NOT_PREVIOUSLY_TESTED'
        self.events['VISIT_'+reason] += 1
        self.emit('BIN_VISIT', piece=int(piece), original_bin=int(old_bin)+1,
            target_bin=int(target)+1, reason=reason, flags=[bool(v) for v in flags],
            partial_counts=[len(b) for b in layout],
            changed_causes=self.changed_reasons.get(int(target), []))

    def timed(self, name, function):
        def call(*a, **kw):
            if not self.active and name not in ('repack', 'CA'):
                return function(*a, **kw)
            start = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                self.stats[name]['calls'] += 1
                self.stats[name]['seconds'] += time.perf_counter()-start
        return call

    def event(self, name):
        self.events[name] += 1

    def signature(self, data):
        return tuple((i, b, info['orientation'], tuple(info['translation']))
            for b, ids in enumerate(data['pieces_order'])
            for i, info in zip(ids, data['bin_real_layout'][b]))

    def select(self, state):
        if self.mode == 'replay':
            index = int(self.tape[len(self.candidates)][0])
        else:
            data = state['data_pool'][state['local_best_iteration']]
            signature = self.signature(data)
            if self.reference is None:
                self.reference = signature
            if signature != self.reference:
                self.exclude = False
            if state['local_best_iteration'] != self.last_local:
                self.age = 0
            self.last_local = state['local_best_iteration']
            info = [None]*len(state['object_info_total'])
            for b, ids in enumerate(data['pieces_order']):
                for j, i in enumerate(ids):
                    info[i] = data['bin_real_layout'][b][j]
            trigger = state['kick_trigger_time']
            stage = (trigger-state['no_change_time'])/trigger
            index = self.ils.improve_pieces_selection_ls(info, self.age, stage,
                max(2,len(data['bin_real_layout'])), seed=None, deterministic=False,
                exclude_last_bin=self.exclude)
            self.age += 1
        self.selections.append(int(index))
        return index

    def replay_orientations(self):
        return [self.tape[len(self.candidates)][1]]

    def install(self):
        import packing_iter_ls as packing
        # Nested function timings overlap: do not sum them as exclusive totals.
        for name in ['SC_heuristic', 'rotate_voxel', 'translate_voxel', 'get_volume']:
            setattr(packing, name, self.timed(name, getattr(packing,name)))
        original = self.ils.repacking_new_ILS
        self.ils.packing_3D_voxel_lookback = self.timed('CA', self.ils.packing_3D_voxel_lookback)
        def old(*a, **kw):
            if self.mode == 'replay':
                expected = self.tape[len(self.candidates)]
                assert list(a[1]) == expected, f'Candidate mismatch at evaluation {len(self.candidates)+1}'
            self.candidates.append(list(a[1]))
            self.begin(a)
            self.active = True
            try:
                result = self.timed('repack', original)(*a, **kw)
                self.finish(result)
                return result
            finally:
                self.active = False
        self.ils.repacking_new_ILS = old
        source = inspect.getsource(self.ils.improved_repack)
        detailed = {
            '                bin_state_flag_list.append(True)': '                bin_state_flag_list.append(False)\n                probe.mark(bin_state_flag_list, each_bin, "NEW_BIN", num_piece)',
            'bin_state_flag_list[old_piece_info[new_selected_index][0]] = True': 'probe.mark(bin_state_flag_list, old_piece_info[new_selected_index][0], "SELECTED_ORIGINAL_BIN", new_selected_index)',
            'bin_state_flag_list[original_bin_index] = True': 'probe.mark(bin_state_flag_list, original_bin_index, "ORIENTATION_CHANGE_OR_DEPARTURE", num_piece)',
            'bin_state_flag_list[each_bin] = True': 'probe.mark(bin_state_flag_list, each_bin, "ARRIVAL_OR_POSITION_CHANGE", num_piece)',
            '            if num_piece != new_selected_index and not bin_state_flag_list[each_bin]:': '            probe.visit(num_piece, new_selected_index, original_bin_index, each_bin, bin_state_flag_list, current_layout)\n            if num_piece != new_selected_index and not bin_state_flag_list[each_bin]:',
            '            result = False': '            diagnostic_searched = False\n            result = False',
            '                result = search(': '                diagnostic_searched = True\n                result = search(',
            '            if not isinstance(result, tuple):': '            if not isinstance(result, tuple):\n                probe.emit("PLACEMENT_FAILED", piece=int(num_piece), bin=int(each_bin)+1, reason="GEOMETRY_SEARCH_FAILED" if diagnostic_searched else "CAPACITY_OR_RADIO_REJECTED", flags=[bool(v) for v in bin_state_flag_list])',
        }
        for old_text, new_text in detailed.items():
            assert old_text in source, old_text
            source = source.replace(old_text,new_text)
        # Count actual branches without timing each line or tracing all voxel loops.
        replacements = {
            '                    continue\n': '                    probe.event("SKIP_UNCHANGED_EARLIER")\n                    continue\n',
            '                    place(num_piece, each_bin, best_current_info)': '                    probe.event("RESTORE_UNCHANGED_ORIGINAL")\n                    place(num_piece, each_bin, best_current_info)',
            '            result = False': '            if not bin_state_flag_list[each_bin] and each_bin > original_bin_index:\n                probe.event("VISIT_UNCHANGED_LATER")\n            result = False',
            '                result = search(': '                probe.event("SEARCH")\n                result = search(',
            '            if not isinstance(result, tuple):': '            if not isinstance(result, tuple):\n                probe.event("FAILED_ATTEMPT")',
            'not np.array_equal(': 'not probe.timed("array_equal", np.array_equal)(',
            'np.count_nonzero(topos_layout[each_bin])': 'probe.timed("capacity_scan", np.count_nonzero)(topos_layout[each_bin])',
        }
        for old_text, new_text in replacements.items():
            assert old_text in source, old_text
            source = source.replace(old_text,new_text)
        scope = dict(packing.__dict__,probe=self)
        exec(compile(source,'diagnostic_repack.py','exec'),scope)
        improved = scope['improved_repack']
        def new(*a, **kw):
            self.candidates.append(list(a[1]))
            self.begin(a)
            self.active = True
            try:
                result = self.timed('repack',improved)(*a,**kw)
                self.finish(result)
                return result
            finally:
                self.active = False
        self.ils.improved_repack = new
        self.output.joinpath('diagnostic_repack.py').write_text(source,encoding='utf-8')

    def save(self):
        self.output.joinpath('candidate_tape.json').write_text(json.dumps(self.candidates),encoding='utf-8')
        self.output.joinpath('diagnostics.json').write_text(json.dumps(dict(mode=self.mode,
            stats=self.stats,events=self.events,selections=self.selections,candidates=self.candidates,bin_summaries=self.bin_summaries,
            note='Host wall durations; nested timings overlap. SC_heuristic timing excludes warmup and CA. Branch probes add overhead.'),indent=2),encoding='utf-8')


def make_worker(template, helper, mode, tape):
    code = template.read_text(encoding='utf-8')
    marker = "    budget = dict(count=0, regular=0, kicks=0)"
    injection = ("    sys.path.insert(0, " + repr(str(helper.parent)) + ")\n"
        "    from diagnose_st05_runtime import Probe\n"
        f"    probe = Probe(ils, args.output, {mode!r}, Path({str(tape)!r}))\n"
        "    probe.install()\n")
    assert marker in code
    code = code.replace(marker,injection+marker,1)
    marker = "        ast.parse(adapted)"
    patch = '''        adapted = adapted.replace('selected_index = pieces_selection_ls(object_info_total, alpha = alpha, seed=None)', 'selected_index = probe.select(locals())', 1)
        if probe.mode == 'replay':
            adapted = adapted.replace('other_orientations = list(orientations_list)', 'other_orientations = probe.replay_orientations()', 1)
            adapted = adapted.replace('        other_orientations.remove(local_best_orientations_list_no_bin[selected_index])', '', 1)
'''
    assert marker in code
    code = code.replace(marker,patch+marker,1)
    code = code.replace("        scope = dict(ils.__dict__)","        scope = dict(ils.__dict__, probe=probe)",1)
    marker = "    elapsed = time.perf_counter() - start"
    code = code.replace(marker,marker+"\n    probe.save()",1)
    ast.parse(code)
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--dataset',default='St_05_Example3_normal')
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--evaluations',type=int,default=100)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Reuse completed runs; incomplete runs restart from evaluation 1')
    a=p.parse_args()
    assert a.evaluations>0
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    template=a.project/'hpc'/'benchmark_ils.py'
    tape=out/'improved'/'candidate_tape.json'
    reports=[]
    for mode,worker in [('improved','improved'),('shared_policy','original'),('replay','original')]:
        target=out/mode;target.mkdir(exist_ok=True)
        if (target/f'{worker}.json').exists():
            if not a.resume:
                raise ValueError('Choose a fresh output directory or --resume')
            report=json.loads((target/f'{worker}.json').read_text())
            assert report['evaluations']==a.evaluations and report['seed']==a.seed and report['dataset']==a.dataset
            assert all(hashlib.sha256((a.project/'src'/n).read_bytes()).hexdigest()==h for n,h in report['source_hashes'].items()), 'Source changed since completed run'
            diag=json.loads((target/'diagnostics.json').read_text())
            reports.append((mode,report,diag))
            print('REUSE COMPLETED',mode,flush=True)
            continue
        script=target/'worker.py'
        script.write_text(make_worker(template,Path(__file__).resolve(),mode,tape),encoding='utf-8')
        command=[sys.executable,'-u',str(script),'--project',str(a.project.resolve()),
            '--worker',worker,'--dataset',a.dataset,'--seed',str(a.seed),
            '--evaluations',str(a.evaluations),'--kick-trigger',str(a.evaluations+1),
            '--output',str(target),'--progress']
        print('START',mode,flush=True)
        with (target/'run.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{mode} failed: {target / "run.log"}')
        report=json.loads((target/f'{worker}.json').read_text())
        diag=json.loads((target/'diagnostics.json').read_text())
        reports.append((mode,report,diag))
    baseline=reports[0][2]['candidates']
    assert all(len(d['candidates']) == r['evaluations'] == a.evaluations for _, r, d in reports)
    assert reports[2][2]['candidates']==baseline,'Replay candidate mismatch'
    consistency = {mode: dict(piece_sequence_equal=[c[0] for c in d['candidates']]==[c[0] for c in baseline],
        piece_and_orientation_equal=d['candidates']==baseline, evaluations=len(d['candidates']))
        for mode, _, d in reports}
    (out/'selection_consistency.json').write_text(json.dumps(consistency,indent=2),encoding='utf-8')
    assert all(r['origin_N']==reports[0][1]['origin_N'] and abs(r['origin_U_star']-reports[0][1]['origin_U_star'])<1e-12 for _,r,_ in reports)
    rows=[]
    for mode,r,d in reports:
        rows.append(dict(mode=mode,seconds=r['wall_seconds'],best_N=r['best_N'],best_U_star=r['best_U_star'],
            evaluations=r['evaluations'],same_candidates=d['candidates']==baseline,
            ca_seconds=d['stats']['CA']['seconds'],repack_seconds=d['stats']['repack']['seconds'],search_calls=d['stats'].get('SC_heuristic',{}).get('calls',0),
            search_seconds=d['stats'].get('SC_heuristic',{}).get('seconds',0)))
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[f'# {a.dataset} runtime diagnosis','',
        'Same seed and selection policy do not guarantee identical candidates. Replay enforces identical piece/orientation per evaluation; accepted layouts may differ.',
        'All runs use current CA defaults, fresh processes/pools, excluded warmup, exact evaluation budgets, and no kicks. Diagnostic timings include probe overhead.',
        '', '| Mode | Wall seconds | Repack seconds | Search calls | Search seconds | Best bins | Best U* | Identical candidates |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['mode']} | {r['seconds']:.2f} | {r['repack_seconds']:.2f} | {r['search_calls']} | {r['search_seconds']:.2f} | {r['best_N']} | {r['best_U_star']:.6f} | {r['same_candidates']} |")
    lines+=['','Nested operation timings must not be summed. See diagnostics.json for rotation, array comparison, capacity scans and skip/restore counts. This is a current-code experiment, not proof of the historical HPC timing cause.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
