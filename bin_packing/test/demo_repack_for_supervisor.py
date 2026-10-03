"""Minimal deterministic demonstration of the local repacking control flow.

Geometry search is scripted to isolate bin reuse; the real local function body
is executed unchanged except for substituting its nested search dependency.
Run: python demo_repack_for_supervisor.py
"""
import ast
import contextlib
import io
from pathlib import Path
import sys
import numpy as np

SOURCE = Path(r'C:\work\paper_repo\3D_irregular_packing_stream\bin_packing\src\packing_iter_ls.py')
names = ['a', 'p', 'q', 'r']
shape = (4, 1, 1)
objects = []
for i in range(4):
    array = np.zeros(shape)
    array[0, 0, 0] = 1
    objects.append(dict(array=array, piece_type=i, volume=1, radio=1,
                        orientation='x_0', translation=(0, 0, 0)))
old_order = [[0], [1, 2], [3]]
old_layout = []
for b, ids in enumerate(old_order):
    old_layout.append([])
    for j, i in enumerate(ids):
        info = dict(objects[i], bin_position=b, translation=(j, 0, 0))
        info['array'] = np.roll(info['array'], j, axis=0)
        old_layout[b].append(info)
calls = []


def search(info, layouts, occupancy, b):
    i = info['piece_type']
    calls.append((i, b))
    # p cannot enter bin 1, q cannot remain in bin 2. Other attempts succeed.
    success = (i, b) not in {(1, 0), (2, 1)}
    print(f"SEARCH {names[i]} in bin {b + 1}: {'SUCCESS' if success else 'FAIL'}")
    return (0, (len(layouts[b]), 0, 0)) if success else False


source = SOURCE.read_text(encoding='utf-8')
source_lines = source.splitlines()
tree = ast.parse(source)
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'improved_repack']
tree.body[0].body = [n for n in tree.body[0].body if not isinstance(n, ast.FunctionDef) or n.name != 'search']
scope = dict(np=np, search=search, rotate_voxel=lambda a, d, ax: a.copy(),
             translate_voxel=lambda a, pos: np.roll(a, pos[0], axis=0))
exec(compile(tree, str(SOURCE), 'exec'), scope)


def trace(frame, event, arg):
    if frame.f_code.co_name != 'improved_repack':
        return None
    if event == 'line':
        line = source_lines[frame.f_lineno - 1].strip()
        loc = frame.f_locals
        if line == 'continue' and 'first-fit rejection' in source_lines[frame.f_lineno - 2]:
            print(f"SKIP {names[loc['num_piece']]} in unchanged earlier bin {loc['each_bin'] + 1}")
        if line == 'result = search(try_info_tem, current_layout, topos_layout, each_bin)':
            if not loc['bin_state_flag_list'][loc['each_bin']] and loc['num_piece'] != loc['new_selected_index']:
                print(f"TRY unchanged later bin {loc['each_bin'] + 1}")
        if line == 'bin_state_flag_list[each_bin] = True':
            print(f"MARK bin {loc['each_bin'] + 1} changed")
    return trace


buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    print('CONTROL-FLOW DEMO: scripted geometry; real local improved_repack body.')
    print('Fixed sequence: a -> p -> q -> r')
    print('Old bins: [a] | [p, q] | [r]')
    print('Change p orientation; preserve a; repack p, q, r.\n')
    sys.settrace(trace)
    try:
        result = scope['improved_repack'](
            objects, (1, 'x_90'), None, None, ['x_0'] * 4,
            {'old': dict(bin_real_layout=old_layout, pieces_order=old_order)},
            'old', None, shape, 'cube', 1, 100, None, None, None,
            5, 'z', None, False, True, None, False, False)
    finally:
        sys.settrace(None)
    assert calls == [(1, 0), (1, 1), (2, 1), (2, 2), (3, 1)]
    assert result[3] == [[0], [1, 3], [2]]
    assert all(np.max(a) <= 1 for a in result[1])
    print('\nNew bins:', ' | '.join(str([names[i] for i in ids]) for ids in result[3]))
    print('PASS: q skips unchanged bin 1, searches unchanged bin 3 after bin 2 fails,')
    print('and marks bin 3 changed after entering it. r then finds its first fit in bin 2.')

output = buffer.getvalue()
print(output, end='')
Path(__file__).with_suffix('.log').write_text(output, encoding='utf-8')
