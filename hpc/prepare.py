"""Generate a nesting strategy x selection range experiment matrix."""
import argparse
import itertools
import json
from pathlib import Path
import shlex

from run import ROOT, PROJECT, STRATEGIES, RANGES, parser as run_parser, read_instance, resize_instance


def geometry_groups(instances):
    """Keep shape, size, ordered geometry and type IDs; discard radio/weight."""
    groups = {}
    for name in instances:
        shape, size, _, _, records = read_instance(name)
        key = (shape, size, tuple((str(path), volume, kind) for path, volume, _, kind in records))
        groups.setdefault(key, []).append(name)
    return {names[0]: names for names in groups.values()}


def build_matrix(instances, strategies, ranges, output_dir, rng_seed):
    rows = []
    loaded = {name: read_instance(name) for name in instances}
    for instance, strategy, selection in itertools.product(instances, strategies, ranges):
        name = f"{instance}__{strategy}__{selection}__instance_size__rng{rng_seed}"
        args = ["--instance-id", instance, "--alg", "fixed_CA", "--pes", strategy,
                "--oes", strategy, "--range", selection,
                "--rng-seed", str(rng_seed), "--output", f"{output_dir}/{name}.json", "--resume"]
        parsed = run_parser().parse_args(args)
        resized = resize_instance(loaded[instance], parsed)
        rows.append({"task_id": len(rows), "instance_id": instance, "nesting_strategy": strategy,
                     "selection_range": selection, "bin_size": resized[1], "args": args,
                     "output": parsed.output})
    outputs = [row["output"] for row in rows]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Duplicate experiment combinations/output paths")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", nargs="+", help="Default: every .txt file in bin_packing/instances")
    p.add_argument("--unique-geometry", action="store_true", default=True, help="Default: merge radio/weight variants, retaining every source filename")
    p.add_argument("--keep-duplicates", action="store_false", dest="unique_geometry", help="Run every source file separately")
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    p.add_argument("--ranges", nargs="+", choices=RANGES, default=["bottom", "bottom_top", "all"])
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--output-dir", default="hpc/results/all_instances_geometry")
    p.add_argument("--params", default="hpc/all_instances.txt")
    args = p.parse_args(argv)
    if args.instances is None:
        args.instances = sorted(path.stem for path in (PROJECT / "instances").glob("*.txt"))
    aliases = geometry_groups(args.instances) if args.unique_geometry else {name: [name] for name in args.instances}
    args.instances = list(aliases)
    rows = build_matrix(args.instances, args.strategies, args.ranges,
                        args.output_dir, args.rng_seed)
    for row in rows:
        row["source_instances"] = aliases[row["instance_id"]]
        row["constraint_mode"] = "geometry_only"
        row["bin_size_source"] = "instance"
    path = Path(args.params)
    if not path.is_absolute():
        path = ROOT / path
    if path.exists() or path.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Use a new --params path; refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(shlex.join(row["args"]) + "\n" for row in rows), encoding="utf-8", newline="\n")
    path.with_suffix(".manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} tasks to {path}; array indices 0-{len(rows)-1}")


if __name__ == "__main__":
    main()
