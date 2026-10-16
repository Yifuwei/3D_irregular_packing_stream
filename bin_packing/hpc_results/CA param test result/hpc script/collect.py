"""Collect a manifest into CSV, retaining failed and missing tasks."""
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from run import ROOT


def collect(manifest):
    rows = []
    for task in json.loads(Path(manifest).read_text(encoding="utf-8")):
        row = {k: task[k] for k in ("task_id", "instance_id", "nesting_strategy", "selection_range")}
        row["bin_size"] = "x".join(map(str, task["bin_size"]))
        row["source_instances"] = ";".join(task.get("source_instances", [task["instance_id"]]))
        row["constraint_mode"] = task.get("constraint_mode", "geometry_only")
        match = re.fullmatch(r"(.+)_seq(\d+)_(cube|cylinder)(?:_.*)?", row["instance_id"])
        row.update(dataset=match[1] if match else row["instance_id"],
                   sequence_seed=int(match[2]) if match else None,
                   bin_shape=match[3] if match else "unknown")
        path = Path(task["output"])
        if not path.is_absolute():
            path = ROOT / path
        row.update(status="missing", output=str(path))
        if path.exists():
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                row.update(status=result["status"], elapsed_s=result.get("elapsed_s"),
                           algorithm_time_s=result.get("algorithm_time_s"),
                           validation_ok=result.get("validation", {}).get("ok"),
                           error=result.get("error") or "; ".join(result.get("validation", {}).get("errors", [])))
                row.update(result.get("metrics", {}))
            except (ValueError, KeyError):
                row["status"] = "unreadable"
        rows.append(row)
    return rows


METRICS = ("best_N", "best_U", "best_U_star", "origin_N", "origin_U", "origin_U_star",
           "algorithm_time_s", "elapsed_s")
EXPECTED_SEEDS = {3, 7, 13, 19, 53}


def aggregate(rows):
    fields = ("dataset", "bin_shape", "bin_size", "nesting_strategy", "selection_range", "constraint_mode")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    summary = []
    for key, members in sorted(groups.items()):
        seeds = [row["sequence_seed"] for row in members]
        if None in seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"Unknown or duplicate sequence seed in group {key}")
        good = [row for row in members if row["status"] == "success" and row.get("validation_ok") is True]
        successful_seeds = {row["sequence_seed"] for row in good}
        result = dict(zip(fields, key))
        result.update(expected_seed_count=5, task_count=len(members), success_count=len(good),
                      complete=successful_seeds == EXPECTED_SEEDS and set(seeds) == EXPECTED_SEEDS,
                      successful_seeds=";".join(map(str, sorted(successful_seeds))),
                      missing_or_unsuccessful_seeds=";".join(map(str, sorted(EXPECTED_SEEDS-successful_seeds))))
        for metric in METRICS:
            values = [row[metric] for row in good if isinstance(row.get(metric), (int, float))
                      and not isinstance(row[metric], bool) and math.isfinite(row[metric])]
            result[f"{metric}_n"] = len(values)
            result[f"{metric}_mean"] = statistics.mean(values) if values else None
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        summary.append(result)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest")
    p.add_argument("--output", default="hpc/summary.csv")
    p.add_argument("--stats-output", help="Default: <output stem>_by_dataset.csv")
    args = p.parse_args()
    rows = collect(args.manifest)
    path = Path(args.output)
    stats = aggregate(rows)
    stats_path = Path(args.stats_output) if args.stats_output else path.with_name(path.stem + "_by_dataset.csv")
    if stats_path.resolve() == path.resolve():
        p.error("Detailed output and statistics output must be different files")
    write_csv(path, rows)
    write_csv(stats_path, stats)
    print(f"Wrote {len(rows)} tasks to {path}")
    print(f"Wrote {len(stats)} groups to {stats_path}; {sum(row['complete'] for row in stats)} complete")


if __name__ == "__main__":
    main()
