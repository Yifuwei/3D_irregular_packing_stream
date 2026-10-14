"""Run exactly one bin_packing experiment; --dry-run needs only Python."""
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "bin_packing"
ALGORITHMS = ("fixed_CA", "random_CA", "BLF", "ILS", "GRASP", "GRASP_ILS")
STRATEGIES = ("minimum_aabb_volume", "minimum_aabb_edges_len", "maximal_residual_box", "overlap_distance")
RANGES = ("bottom", "bottom_top", "all", "bottom_left_filling")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance-id", "--instance_id", required=True)
    p.add_argument("--alg", choices=ALGORITHMS, default="fixed_CA")
    p.add_argument("--range", choices=RANGES, default="bottom")
    p.add_argument("--pes", choices=STRATEGIES, default=STRATEGIES[0])
    p.add_argument("--oes", choices=STRATEGIES + ("waste_overlap_distance", "height_only"), default=STRATEGIES[0])
    p.add_argument("--rng-seed", type=int, default=42, help="Algorithm RNG; instance already contains piece order")
    p.add_argument("--time-limit", type=int, default=3600)
    p.add_argument("--iteration-limit", type=int, default=1000)
    p.add_argument("--grasp-iterations", type=int, default=100)
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--kick-trigger", type=int, default=120)
    p.add_argument("--kick-level", choices=("low", "medium", "high"), default="medium")
    p.add_argument("--output", required=True, help="Absolute path or path relative to repository root")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip only matching successful results")
    p.add_argument("--trace", action="store_true")
    return p


def read_instance(instance_id):
    if Path(instance_id).name != instance_id or "/" in instance_id or "\\" in instance_id:
        raise ValueError("instance-id must be a filename stem")
    path = PROJECT / "instances" / f"{instance_id}.txt"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    fields = lines[0].split("\t")
    shape = fields[0]
    size = tuple(ast.literal_eval(fields[1]))
    if shape not in ("cube", "cylinder") or len(size) != 3 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError(f"Invalid container header: {path}")
    records = []
    for line in lines[1:]:
        if not line.strip():
            continue
        row = re.split(r"\t+", line.strip())
        relative = row[0].replace("\\", "/")
        if relative.startswith("../"):
            relative = relative[3:]
        voxel_path = (PROJECT / "data" / relative).resolve()
        if not voxel_path.is_relative_to((PROJECT / "data").resolve()):
            raise ValueError(f"Voxel path outside data directory: {row[0]}")
        if not voxel_path.is_file():
            raise FileNotFoundError(voxel_path)
        # Geometry-only experiments deliberately ignore per-piece radio values.
        records.append((voxel_path, int(row[1]), 0.0, int(row[3])))
    if not records:
        raise ValueError(f"Empty instance: {path}")
    # Both legacy two-field headers and radio/weight variants are accepted.
    # rho=1 removes the weight-derived fill limit; all radio values are zero.
    return shape, size, 0, 1, records


def resize_instance(instance, args):
    shape, original_size, radio, rho, records = instance
    size = original_size
    # Current rotation implementation uses full arrays and requires equal axes.
    if len(set(size)) != 1:
        raise ValueError("Current packing rotations require X=Y=Z; non-cubic array sizes are not supported")
    return shape, size, radio, rho, records


def load_objects(records, size):
    import numpy as np
    import binvox_rw
    objects = []
    for path, volume, radio, piece_type in records:
        with path.open("rb") as stream:
            data = binvox_rw.read_as_3d_array(stream).data
        points = np.argwhere(data)
        if not len(points):
            raise ValueError(f"Empty voxel object: {path}")
        low, high = points.min(0), points.max(0) + 1
        data = data[tuple(slice(a, b) for a, b in zip(low, high))]
        if any(a > b for a, b in zip(data.shape, size)):
            raise ValueError(f"Object {path} with shape {data.shape} exceeds {size}; refusing to crop")
        padded = np.zeros(size, dtype=int)
        padded[tuple(slice(0, n) for n in data.shape)] = data
        objects.append(dict(array=padded, translation=(0, 0, 0), orientation="x_0",
                            bin_position=None, volume=volume, radio=radio,
                            piece_type=piece_type, aabb=data.shape))
    return objects


def validate_layout(layout, originals, size, shape):
    """Reconstruct unwrapped coordinates to detect np.roll boundary crossings."""
    import numpy as np
    by_type = {p["piece_type"]: p for p in originals}
    errors, placements = [], []
    seen = Counter()
    for bin_id, pieces in enumerate(layout):
        occupied = np.zeros(size, dtype=bool)
        for item in pieces:
            kind = item["piece_type"]
            seen[kind] += 1
            placements.append({"bin": bin_id, "piece_type": int(kind),
                               "orientation": item["orientation"],
                               "translation": [int(v) for v in item["translation"]]})
            axis, angle = item["orientation"].split("_")
            source = np.rot90(by_type[kind]["array"], int(angle)//90,
                              axes={"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis])
            points = np.argwhere(source != 0)
            points = points - points.min(0) + np.asarray(item["translation"])
            if np.any(points < 0) or np.any(points >= size):
                errors.append(f"bin {bin_id} type {kind}: out of rectangular bounds")
                continue
            if shape == "cylinder":
                radial = (points[:, 0]-size[0]/2)**2 + (points[:, 1]-size[1]/2)**2
                if np.any(radial > (size[0]/2)**2):
                    errors.append(f"bin {bin_id} type {kind}: outside cylinder voxel mask")
            index = tuple(points.T)
            if occupied[index].any():
                errors.append(f"bin {bin_id} type {kind}: overlap")
            occupied[index] = True
            actual = item["array"] != 0
            if actual.shape != tuple(size) or np.count_nonzero(actual) != len(points) or not actual[index].all():
                errors.append(f"bin {bin_id} type {kind}: array disagrees with orientation/translation")
    if seen != Counter(p["piece_type"] for p in originals):
        errors.append("Packed piece counts do not match input")
    return {"ok": not errors, "errors": errors}, placements


def run_experiment(args, instance):
    import numpy as np
    from numba import cuda
    from function_lib import NFV_POOL, IFV_POOL
    from new_ILS_southampton import ILS_from_the_first_piece, GRASP, GRASP_ILS
    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run inside a GPU allocation with the packing environment")
    shape, size, radio, rho, records = instance
    objects = load_objects(records, size)
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)
    orientations = ([0, 90, 180, 270], ["x", "y", "z"])
    orientation_list = ["x_0"] + [f"{a}_{d}" for d in (90, 180, 270) for a in ("x", "y", "z")]
    common = dict(object_info_total=objects, nfv_pool=NFV_POOL(), ifv_pool=IFV_POOL(),
                  max_radio=radio, rho=rho, orientations=orientations,
                  ils_orientations=orientations, orientations_list=orientation_list,
                  packing_alg="SCH", selection_type="bounding_box", selection_range=args.range,
                  accessible_check=True, SCH_nesting_strategy=args.pes, orien_evaluation=args.oes,
                  container_size=size, container_shape=shape, time_limit=args.time_limit,
                  alpha=args.alpha, flag_NFV_POOL=False, visualisation=False, _TRACE=args.trace)
    if args.alg in ("fixed_CA", "random_CA", "BLF", "ILS"):
        values = ILS_from_the_first_piece(**common, ALG=args.alg, iteration_limit=args.iteration_limit,
                                          kick_trigger_time=args.kick_trigger, kick_level=args.kick_level)
    elif args.alg == "GRASP":
        values = GRASP(**common, iter_limit_per_iter=args.grasp_iterations)
    else:
        values = GRASP_ILS(**common, iter_limit_per_iter=args.grasp_iterations,
                           kick_trigger_time=args.kick_trigger, kick_level=args.kick_level)
    checks, placements = validate_layout(values[6], objects, size, shape)
    metrics = dict(zip(("best_N", "best_U", "best_U_star", "origin_N", "origin_U", "origin_U_star"), values[:6]))
    return {"status": "success" if checks["ok"] else "invalid", "metrics": metrics,
            "validation": checks, "placements": placements,
            "algorithm_time_s": float(values[-1]), "container_shape": shape, "container_size": size}


def write_result(path, result):
    def scalar(value):
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, default=scalar, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.alg == "BLF":
        args.range = "bottom_left_filling"
    if min(args.time_limit, args.iteration_limit, args.grasp_iterations, args.kick_trigger) <= 0:
        raise ValueError("Time and iteration limits must be positive")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    config = {k: v for k, v in vars(args).items() if k not in ("output", "resume", "dry_run")}
    config["constraint_mode"] = "geometry_only"
    config["bin_size_source"] = "instance"
    instance = resize_instance(read_instance(args.instance_id), args)
    if args.dry_run:
        print(json.dumps({"config": config, "container": instance[:4], "pieces": len(instance[4]),
                          "output": str(output)}, indent=2))
        return 0
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if args.resume and saved.get("status") == "success" and saved.get("config") == config:
            print(f"[SKIP] {output}")
            return 0
        if not args.resume or saved.get("config") != config:
            raise FileExistsError(f"Output exists; use a new output path or --resume for matching runs: {output}")
    sys.path.insert(0, str(PROJECT / "src"))
    started = time.time()
    result = {"config": config, "slurm": {k: os.environ.get(k) for k in
              ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "CUDA_VISIBLE_DEVICES")}}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    try:
        result.update(run_experiment(args, instance))
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        print(result["error"], file=sys.stderr)
    result["elapsed_s"] = time.time() - started
    write_result(output, result)
    print(f"[{result['status'].upper()}] {output}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
