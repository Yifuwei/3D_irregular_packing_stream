import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
import prepare
import task
import collect


class WorkflowTests(unittest.TestCase):
    def test_legacy_and_constrained_instances_are_geometry_equivalent(self):
        legacy = run.read_instance("chess_seq13_cube")
        constrained = run.read_instance("chess_seq13_cube_999999_1")
        self.assertEqual(legacy, constrained)
        self.assertEqual(legacy[2:4], (0, 1))
        self.assertTrue(all(record[2] == 0 for record in legacy[4]))

    def test_fractional_weight_header_is_ignored(self):
        instance = run.read_instance("Merged4_normal_seq13_cube_2000_0.2")
        self.assertEqual(instance[2:4], (0, 1))
        self.assertTrue(all(record[2] == 0 for record in instance[4]))

    def test_geometry_dedup_preserves_all_source_names(self):
        names = ["chess_seq13_cube", "chess_seq13_cube_999999_1", "chess_seq19_cube"]
        groups = prepare.geometry_groups(names)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[names[0]], names[:2])
        self.assertEqual(sorted(n for aliases in groups.values() for n in aliases), sorted(names))

    def test_default_matrix(self):
        rows = prepare.build_matrix(["chess_seq13_cube_999999_1"], run.STRATEGIES,
                                    ["bottom", "bottom_top", "all"], "hpc/results/test", 42)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["output"] for r in rows}), 12)
        self.assertEqual({r["bin_size"] for r in rows}, {(56, 56, 56)})
        self.assertTrue(all(run.parser().parse_args(row["args"]).alg == "fixed_CA" for row in rows))

    def test_instance_size_is_preserved(self):
        base = run.read_instance("chess_seq13_cube")
        args = run.parser().parse_args(["--instance-id", "chess_seq13_cube", "--output", "unused"])
        self.assertEqual(run.resize_instance(base, args)[1], base[1])
        self.assertNotIn("--bin-scale", run.parser().format_help())
        self.assertNotIn("--bin-size", run.parser().format_help())

    def test_parameter_row_zero_bom_crlf_and_quoting(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "params.txt"
            path.write_bytes(b'\xef\xbb\xbf--output "a b.json"\r\n--output c.json\r\n')
            self.assertEqual(task.read_task(path, 0), ["--output", "a b.json"])
            self.assertEqual(task.read_task(path, 1), ["--output", "c.json"])
            with self.assertRaises(ValueError):
                task.read_task(path, -1)
            with self.assertRaises(ValueError):
                task.read_task(path, 2)

    def test_algorithm_keyword_signatures_match_repository(self):
        # Inspect actual signatures without importing unavailable CUDA dependencies.
        source = ast.parse((run.PROJECT / "src/new_ILS_southampton.py").read_text(encoding="utf-8"))
        functions = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef)}
        runner = ast.parse(inspect.getsource(run.run_experiment))
        common = next(node.value for node in ast.walk(runner)
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "common" for t in node.targets))
        common_keys = {kw.arg for kw in common.keywords}
        for call in ast.walk(runner):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in functions:
                function = functions[call.func.id]
                accepted = {arg.arg for arg in function.args.args}
                supplied = common_keys | {kw.arg for kw in call.keywords if kw.arg}
                self.assertFalse(supplied - accepted, (call.func.id, supplied - accepted))
                required = {arg.arg for arg in function.args.args[:len(function.args.args)-len(function.args.defaults)]}
                self.assertFalse(required - supplied, (call.func.id, required - supplied))

    def test_manifest_collection_preserves_missing_and_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            rows = []
            for index, status in enumerate(("success", "invalid", "missing")):
                path = directory / f"{index}.json"
                rows.append(dict(task_id=index, instance_id="example", nesting_strategy="overlap_distance",
                                 selection_range="bottom", bin_size=[4, 4, 4], output=str(path)))
                if status != "missing":
                    run.write_result(path, {"status": status, "metrics": {"best_N": 2},
                                            "validation": {"ok": status == "success", "errors": []}})
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual([r["status"] for r in collect.collect(manifest)], ["success", "invalid", "missing"])

    def test_validator_detects_wrapped_object(self):
        import numpy as np
        original = np.zeros((4, 4, 4), dtype=int)
        original[:2, :2, :2] = 1
        info = {"array": original, "piece_type": 1, "orientation": "x_0", "translation": (0, 0, 0)}
        valid = dict(info, array=np.roll(original, (2, 0, 0), axis=(0, 1, 2)), translation=(2, 0, 0))
        self.assertTrue(run.validate_layout([[valid]], [info], (4, 4, 4), "cube")[0]["ok"])
        wrapped = dict(info, array=np.roll(original, (3, 0, 0), axis=(0, 1, 2)), translation=(3, 0, 0))
        report, _ = run.validate_layout([[wrapped]], [info], (4, 4, 4), "cube")
        self.assertFalse(report["ok"])
        self.assertIn("out of rectangular bounds", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
