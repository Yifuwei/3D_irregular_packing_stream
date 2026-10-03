"""Run local-source branch checks and a real chess/CUDA ILS case with a log.

Run with cutting_stable Python. Trace hooks observe the real functions without
changing their code. Branch checks substitute geometry search; the chess case
uses the real constructive algorithm, SC heuristic, rotation and CUDA backend.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import re
from pathlib import Path
import sys
import time
import unittest

import numpy as np

LOCAL = Path(r'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='chess')
parser.add_argument('--time-limit', type=float, default=60)
parser.add_argument('--kick-trigger', type=int, default=10)
parser.add_argument('--iterations', type=int, default=None)
parser.add_argument('--tag', default='ils_chess_seed13')
args = parser.parse_args()
OUTPUT = Path(__file__).resolve().parent / 'logs'
OUTPUT.mkdir(exist_ok=True)
LOG = OUTPUT / f'{args.tag}.log'
SUMMARY = OUTPUT / f'{args.tag}_summary.json'
sys.path.insert(0, str(LOCAL / 'src'))
sys.path.insert(0, str(LOCAL / 'test'))

source = (LOCAL / 'src' / 'packing_iter_ls.py').read_text(encoding='utf-8')
source_lines = source.splitlines()
counts = {}


def record(event, **values):
    counts[event] = counts.get(event, 0) + 1
    print('EVENT', json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def observer(frame, event, arg):
    if frame.f_code.co_name != 'improved_repack':
        return None
    loc = frame.f_locals
    if event == 'call':
        record('REPACK_START', piece=int(loc['selected_info'][0]), orientation=str(loc['selected_info'][1]))
    if event != 'line':
        return observer
    line = source_lines[frame.f_lineno - 1].strip()
    if line == 'continue' and 'each_bin' in loc:
        b, original = loc['each_bin'], loc.get('original_bin_index')
        preceding = source_lines[frame.f_lineno - 2].strip()
        if 'first-fit rejection' in preceding:
            assert loc['num_piece'] != loc['new_selected_index']
            assert b < original and not loc['bin_state_flag_list'][b]
            record('SKIP_UNCHANGED_EARLIER', piece=int(loc['num_piece']), bin=b, original_bin=original)
        elif 'stay False' in preceding or preceding == "# p's original bin stay False after trying p.":
            record('SEARCH_FAILED_FLAG_PRESERVED', piece=int(loc['num_piece']), bin=b,
                   changed=bool(loc['bin_state_flag_list'][b]))
    if line == 'place(num_piece, each_bin, best_current_info)':
        assert loc['each_bin'] == loc['original_bin_index']
        assert not loc['bin_state_flag_list'][loc['each_bin']]
        record('RESTORE_UNCHANGED_ORIGINAL', piece=int(loc['num_piece']), bin=loc['each_bin'])
    if line.startswith('result = search('):
        b, original = loc['each_bin'], loc['original_bin_index']
        is_selected = loc['num_piece'] == loc['new_selected_index']
        flag = bool(loc['bin_state_flag_list'][b])
        assert is_selected or flag or b > original
        event_name = 'SEARCH_UNCHANGED_LATER' if not is_selected and not flag else 'SEARCH'
        record(event_name, piece=int(loc['num_piece']), bin=b, original_bin=original,
               selected_piece=is_selected, changed=flag)
    if line == 'place(num_piece, each_bin, packed_tem)':
        record('PLACE_SEARCH_RESULT', piece=int(loc['num_piece']), bin=loc['each_bin'])
    if line == 'bin_state_flag_list[each_bin] = True':
        record('MARK_CHANGED', piece=int(loc['num_piece']), bin=loc['each_bin'])
    return observer


with LOG.open('w', encoding='utf-8', buffering=1) as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
    print('SOURCE', LOCAL, flush=True)
    print('PYTHON', sys.executable, flush=True)
    print('BIN NUMBERS IN EVENTS ARE ZERO-BASED', flush=True)
    print('REPACK SHA256', hashlib.sha256(source.encode()).hexdigest(), flush=True)
    from numba import cuda
    print('CUDA AVAILABLE', cuda.is_available(), flush=True)
    assert cuda.is_available(), 'The real experiment requires CUDA'

    print('\nPHASE 1: deterministic branch checks; geometry search is substituted.', flush=True)
    spec = importlib.util.spec_from_file_location('branch_checks', Path(__file__).with_name('test_changed_repack.py'))
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tests.SRC = LOCAL / 'src'
    sys.settrace(observer)
    try:
        checks = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    finally:
        sys.settrace(None)
    assert checks.wasSuccessful()
    assert counts.get('SKIP_UNCHANGED_EARLIER', 0) > 0
    assert counts.get('RESTORE_UNCHANGED_ORIGINAL', 0) > 0
    assert counts.get('SEARCH_UNCHANGED_LATER', 0) > 0
    branch_counts = dict(counts)
    counts.clear()

    print(f'\nPHASE 2: real {args.dataset}, seed=13, iteration limit={args.iterations}, time limit={args.time_limit}s, kick threshold={args.kick_trigger}.', flush=True)
    import code_launcher as launcher
    import new_ILS_southampton as ils
    from function_lib import NFV_POOL, IFV_POOL
    assert Path(inspect.getfile(ils)).resolve() == (LOCAL / 'src' / 'new_ILS_southampton.py').resolve()
    os.chdir(LOCAL / 'test')
    np.random.seed(13)
    instance = launcher.make_instance_id(args.dataset, 13, 'cube', 999999, 1)
    shape, size, max_radio, rho, objects = launcher.load_instance(instance)
    orientations = ([0, 90, 180, 270], ['x', 'y', 'z'])
    values = ['x_0'] + [f'{axis}_{degree}' for degree in [90, 180, 270] for axis in ['x', 'y', 'z']]
    sys.settrace(observer)
    try:
        result = ils.improved_ILS(
            objects, NFV_POOL(), IFV_POOL(), max_radio, rho, orientations,
            orientations, values, 'SCH', 'bounding_box', 'bottom', False,
            'minimum_aabb_volume', None, 'ILS', size, shape,
            iteration_limit=args.iterations, time_limit=args.time_limit, kick_trigger_time=args.kick_trigger,
            kick_level='medium', flag_NFV_POOL=False, visualisation=False, _TRACE=True)
    finally:
        sys.settrace(None)
    best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star = result[:6]
    assert best_N - best_U_star <= origin_N - origin_U_star + 1e-12
    assert sorted(i for ids in result[9] for i in ids) == list(range(len(objects)))
    assert all(np.max(array) <= 1 for array in result[8])
    for b, layout in enumerate(result[6]):
        for j, info in enumerate(layout):
            assert info['orientation'] == result[11][result[9][b][j]]
    assert counts.get('REPACK_START', 0) > 0
    assert counts.get('SKIP_UNCHANGED_EARLIER', 0) > 0
    assert counts.get('RESTORE_UNCHANGED_ORIGINAL', 0) > 0
    stream.flush()
    evaluations = re.findall(r'Evaluation (\d+):.*accepted=(True|False), kick=(True|False)', LOG.read_text(encoding='utf-8'))
    summary = dict(dataset=args.dataset, time_limit=args.time_limit, kick_trigger=args.kick_trigger,
                   iteration_limit=args.iterations, evaluations=len(evaluations),
                   kick_count=sum(kick == 'True' for _, accepted, kick in evaluations),
                   accepted_neighbors=sum(accepted == 'True' and kick == 'False' for _, accepted, kick in evaluations),
                   branch_tests=checks.testsRun, branch_events=branch_counts,
                   real_events=dict(counts), piece_count=len(objects), seed=13,
                   best_N=best_N, origin_N=origin_N, best_U_star=best_U_star,
                   origin_U_star=origin_U_star, elapsed_seconds=result[-1],
                   source=str(LOCAL), python=sys.executable,
                   note='Trace hooks add runtime overhead; this is functional validation, not a timing benchmark.')
    print('\nPASS: branch behavior, actual ILS execution, complete packing, no overlap, best-state consistency.', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding='utf-8')

print('PASS; log:', LOG)
print('Summary:', SUMMARY)
