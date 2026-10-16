"""CPU equivalence tests and warmed benchmarks against frozen original source.

Run with a Python environment containing numpy and numba:
  python bin_packing/test/benchmark_cylinder_ifv.py
No CUDA or optional plotting imports are needed. Times exclude JIT compilation.
"""
import ast
import json
import platform
import statistics
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import numba
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src' / 'function_lib.py'
BASELINE = Path(__file__).parent / 'fixtures' / 'cylinder_ifv_baseline_source.txt'
NAMES = {'IFV_POOL', 'aabb_rotate', 'njit_get_ifv_cylinder',
         'max_along_axis_2', 'njit_cylinder_xy', 'njit_expand_cylinder_xy'}


def load(source, names=None):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and (names is None or n.name in names)]
    scope = {'np': np, 'njit': njit}
    exec(compile(tree, str(SOURCE), 'exec'), scope)
    return scope


OLD = load(BASELINE.read_text(encoding='utf-8'))
NEW = load(SOURCE.read_text(encoding='utf-8'), NAMES)
# Stage 1 keeps the original output dtype to isolate convex hull acceleration.
HULL32 = load(SOURCE.read_text(encoding='utf-8').replace(
    'ifv = np.zeros((xy.shape[0], xy.shape[1], height), dtype=np.uint8)',
    'ifv = np.zeros((xy.shape[0], xy.shape[1], height), dtype=np.int32)'), NAMES)


def padded(piece, size):
    result = np.zeros(size, dtype=np.int64)
    result[tuple(slice(0, n) for n in piece.shape)] = piece
    return result


class EquivalenceTests(unittest.TestCase):
    def check(self, piece, size, dimensions=None):
        arr = padded(piece, size)
        dims = piece.shape if dimensions is None else dimensions
        expected = OLD['njit_get_ifv_cylinder'](arr, *dims, size)
        for scope in (NEW, HULL32):
            actual = scope['njit_get_ifv_cylinder'](arr, *dims, size)
            np.testing.assert_array_equal(actual, expected)

    def test_randomized_projections(self):
        rng = np.random.default_rng(42)
        for _ in range(120):
            # Include non-square containers to retain the legacy centre order.
            size = tuple(int(x) for x in rng.integers(4, 19, 3))
            dims = tuple(int(rng.integers(1, n + 1)) for n in size)
            piece = (rng.random(dims) < rng.uniform(0, 1)).astype(np.int64)
            self.check(piece, size)

    def test_degenerate_and_boundary_projections(self):
        size = (10, 10, 10)
        for piece in (np.zeros((1, 1, 1), dtype=int),
                      np.ones((1, 1, 1), dtype=int),
                      np.ones((1, 7, 2), dtype=int),
                      np.eye(6, dtype=int)[:, :, None],
                      np.ones(size, dtype=int)):
            self.check(piece, size)
        point = np.ones((1, 1, 1), dtype=int)
        result = NEW['njit_get_ifv_cylinder'](padded(point, size), 1, 1, 1, size)
        self.assertEqual(result[0, 5, 0], 1)  # exactly on the circle
        self.assertEqual(result[0, 4, 0], 0)
        for dims in ((11, 1, 1), (1, 11, 1), (1, 1, 11)):
            self.check(point, size, dims)

    def test_cache_height_reuse_eviction_and_keys(self):
        pool = NEW['IFV_POOL'](max_size=2)
        piece = np.ones((2, 3, 4), dtype=int)
        info = {'piece_type': 1, 'orientation': 'x_0', 'aabb': piece.shape,
                'array': padded(piece, (12, 12, 12))}
        with patch.dict(NEW, njit_cylinder_xy=unittest.mock.Mock(wraps=NEW['njit_cylinder_xy'])):
            for size in ((12, 12, 12), (12, 12, 15), (12, 12, 5)):
                actual = pool.retrieve_ifv(info, size, 'cylinder')
                # Original projection only uses XY; input Z need not match height.
                expected = OLD['njit_get_ifv_cylinder'](info['array'], *piece.shape, size)
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(NEW['njit_cylinder_xy'].call_count, 1)
        self.assertEqual(len(pool.cylinder_xy_pool), 1)
        self.assertEqual(len(pool.ifv_pool), 2)
        first = pool.retrieve_ifv(info, (12, 12, 5), 'cylinder')
        self.assertIs(first, pool.retrieve_ifv(info, (12, 12, 5), 'cylinder'))
        for side in (13, 14):
            info['array'] = padded(piece, (side, side, 12))
            pool.retrieve_ifv(info, (side, side, 12), 'cylinder')
        self.assertEqual(len(pool.cylinder_xy_pool), 2)
        self.assertEqual(len(pool.ifv_pool), 2)

    def test_orientation_and_piece_keys_are_separate(self):
        pool = NEW['IFV_POOL']()
        piece = np.ones((2, 3, 4), dtype=int)
        size = (12, 12, 12)
        for piece_type, orientation, arr in (
                (1, 'x_0', piece), (1, 'x_90', np.rot90(piece, axes=(1, 2))),
                (2, 'x_0', piece)):
            info = {'piece_type': piece_type, 'orientation': orientation,
                    'aabb': piece.shape, 'array': padded(arr, size)}
            expected = OLD['njit_get_ifv_cylinder'](info['array'], *arr.shape, size)
            np.testing.assert_array_equal(pool.retrieve_ifv(info, size, 'cylinder'), expected)
        self.assertEqual(len(pool.cylinder_xy_pool), 3)

    def test_strict_accessibility_reuses_both_heights(self):
        from scipy.ndimage import binary_dilation
        scope = load(SOURCE.read_text(encoding='utf-8'),
                     NAMES | {'get_feasible_boundary_rigorous_acces'})
        size = (10, 10, 10)
        fixed = np.zeros(size, dtype=np.uint8)
        fixed[4:6, 4:6, :3] = 1
        piece = np.ones((2, 3, 4), dtype=int)
        info = {'piece_type': 1, 'orientation': 'x_0', 'aabb': piece.shape,
                'array': padded(piece, size)}
        scope.update(get_bounding_box_njit_translation=lambda _: (6, 6, 3),
                     get_max_translation_nfv=lambda *_: size,
                     get_nfv_gpu=lambda *args: fixed.copy(),
                     binary_dilation=binary_dilation,
                     accessibility_check=lambda info, nfv, ifv, size: ifv)
        pool = scope['IFV_POOL']()
        call = lambda: scope['get_feasible_boundary_rigorous_acces'](
            fixed, [], info, size, 'cylinder', None, pool, False)
        actual = call()
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(pool.val_ifv_cal, 2)
        self.assertEqual(len(pool.cylinder_xy_pool), 1)
        np.testing.assert_array_equal(call(), actual)
        self.assertEqual(pool.val_ifv_cal, 2)
        self.assertEqual(pool.rep_ifv_cal, 2)
        class OriginalPool:
            def retrieve_ifv(self, info, size, shape):
                return OLD['njit_get_ifv_cylinder'](info['array'], *piece.shape, size)
        expected = scope['get_feasible_boundary_rigorous_acces'](
            fixed, [], info, size, 'cylinder', None, OriginalPool(), False)
        np.testing.assert_array_equal(actual, expected)


def median_ms(fn, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def real_cases():
    sys.path.insert(0, str(ROOT / 'src'))
    import binvox_rw
    # Different concavities and sizes, taken directly from checked-in instances.
    for dataset in ('chess', 'engine', 'liu', 'Merged1_normal',
                    'Merged4_normal', 'shapesnew', 'St_05_Example3_normal'):
        path = ROOT / 'instances' / (dataset + '_seq3_cylinder.txt')
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        size = tuple(ast.literal_eval(lines[0].split('\t')[1]))
        rel = lines[1].split()[0].replace('\\', '/').removeprefix('../')
        with (ROOT / 'data' / rel).open('rb') as f:
            raw = binvox_rw.read_as_3d_array(f).data.astype(np.int64)
        coords = np.where(raw == 1)
        piece = raw[tuple(slice(c.min(), c.max() + 1) for c in coords)]
        for label, axes in (('z0', None), ('x90', (1, 2)), ('y90', (0, 2))):
            rotated = piece if axes is None else np.rot90(piece, axes=axes)
            if all(a <= b for a, b in zip(rotated.shape, size)):
                yield dataset + '/' + label, np.ascontiguousarray(rotated), size


def benchmark():
    # Compile every timed path before recording. This is separate from tests.
    warm = padded(np.ones((2, 3, 4), dtype=int), (8, 8, 8))
    start = time.perf_counter()
    for scope in (OLD, HULL32, NEW):
        scope['njit_get_ifv_cylinder'](warm, 2, 3, 4, (8, 8, 8))
    info = {'piece_type': 1, 'orientation': 'x_0', 'aabb': (2, 3, 4), 'array': warm}
    NEW['IFV_POOL']().retrieve_ifv(info, (8, 8, 8), 'cylinder')
    warmup_seconds = time.perf_counter() - start
    rows = []
    for name, piece, size in real_cases():
        arr = padded(piece, size)
        args = (arr, *piece.shape, size)
        old = OLD['njit_get_ifv_cylinder'](*args)
        new = NEW['njit_get_ifv_cylinder'](*args)
        np.testing.assert_array_equal(new, old)
        old_ms = median_ms(lambda: OLD['njit_get_ifv_cylinder'](*args))
        hull_ms = median_ms(lambda: HULL32['njit_get_ifv_cylinder'](*args))
        new_ms = median_ms(lambda: NEW['njit_get_ifv_cylinder'](*args))
        info = {'piece_type': name, 'orientation': 'x_0', 'aabb': piece.shape, 'array': arr}
        larger = (size[0], size[1], size[2] + piece.shape[2] - 1)
        pool = NEW['IFV_POOL']()
        pool.retrieve_ifv(info, size, 'cylinder')
        large_expected = OLD['njit_get_ifv_cylinder'](arr, *piece.shape, larger)
        np.testing.assert_array_equal(pool.retrieve_ifv(info, larger, 'cylinder'), large_expected)
        old_repeat_ms = median_ms(lambda: OLD['njit_get_ifv_cylinder'](arr, *piece.shape, larger))
        # XY cached, but force a fresh 3D result to measure height reuse alone.
        xy_ms = median_ms(lambda: pool.get_ifv(info, larger, 'cylinder'))
        cached_ms = median_ms(lambda: pool.retrieve_ifv(info, larger, 'cylinder'), 21)
        row = dict(case=name, size=size, aabb=piece.shape, equivalent=True,
                   original_ms=old_ms, hull_int32_ms=hull_ms, hull_uint8_ms=new_ms,
                   fresh_speedup=old_ms/new_ms, original_larger_ms=old_repeat_ms,
                   cached_xy_fresh_3d_ms=xy_ms, cached_3d_ms=cached_ms,
                   repeated_speedup=old_repeat_ms/cached_ms,
                   original_bytes=old.nbytes, optimized_bytes=new.nbytes)
        rows.append(row)
        print(f'{name}: {old_ms:.3f} -> {hull_ms:.3f} -> {new_ms:.3f} ms '
              f'({old_ms/new_ms:.1f}x); cached {cached_ms:.4f} ms', flush=True)
    result = dict(python=sys.version, numpy=np.__version__, numba=numba.__version__,
                  platform=platform.platform(), processor=platform.processor(),
                  warmup_seconds=warmup_seconds, repeats=5, rows=rows)
    out = ROOT / 'test' / 'cylinder_ifv_benchmark.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved', out)


if __name__ == '__main__':
    if '--benchmark-only' not in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(EquivalenceTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            sys.exit(1)
    if '--test-only' not in sys.argv:
        benchmark()
