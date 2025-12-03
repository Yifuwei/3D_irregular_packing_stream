# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

# # === 0) Paths ===
# JSON_ROOT = Path(r"../HPCoutput_radio")
# # CSV_META  = Path(r"./all_results_metaheuristic_radio.csv")

# list_datasets_name = ["Merged4_normal"]

# # === 1) Regex: extract dataset, seq_seed, container_shape, MAR, rho ===
# # Example: Merged4_normal_seq3_cube_2000_0.3
# IID_PAT = re.compile(
#     r'^(?P<dataset>[A-Za-z0-9_]+)_seq(?P<seq_seed>\d+)_(?P<container_shape>cube|cylinder)_(?P<MAR>\d+)_?(?P<rho>[0-9.]+)?$'
# )

# def parse_instance(instance_id: str):
#     """
#     Extract dataset, seq_seed, container_shape, MAR, rho from instance_id.
#     MAR and rho are optional, so they may be None if not present.
#     """
#     m = IID_PAT.match(instance_id)
#     if not m:
#         # Fallback: try to find container_shape manually if regex fails
#         container_shape = "cube" if "cube" in instance_id else "cylinder" if "cylinder" in instance_id else None
#         return None, None, container_shape, None, None

#     dataset = m.group("dataset")
#     seq_seed = int(m.group("seq_seed"))
#     container_shape = m.group("container_shape")
#     MAR = int(m.group("MAR")) if m.group("MAR") else None
#     rho = float(m.group("rho")) if m.group("rho") else None
#     return dataset, seq_seed, container_shape, MAR, rho

# # === 2) Collect rows from all JSON files ===
# rows = []

# for dataset in list_datasets_name:
#     # Find all JSON files for this dataset
#     json_files = list(JSON_ROOT.rglob(f"{dataset}_*.json"))
#     print(f"📁 Dataset '{dataset}': Found {len(json_files)} JSON files")

#     for p in json_files:
#         try:
#             data = json.loads(p.read_text(encoding="utf-8"))
#         except Exception as e:
#             print(f"[WARN] Failed to read {p}: {e}")
#             continue

#         instance_id = data.get("instance_id", p.stem)
#         ds, seq_seed, container_shape, MAR, rho = parse_instance(instance_id)
#         ALG = data.get("ALG")

#         best_N = data.get("best_N")
#         best_U_star = data.get("best_U_star")
#         seed = data.get("seed")
#         U_improve_pct = data.get("U_improve_pct")
#         U_star_improve_pct = data.get("U_star_improve_pct")

#         delta = best_N - best_U_star if isinstance(best_N, (int, float)) and isinstance(best_U_star, (int, float)) else None

#         rows.append({
#             "dataset": ds or dataset,
#             "seq_seed": seq_seed,
#             "container_shape": container_shape,
#             "MAR": MAR,
#             "rho": rho,
#             "ALG": ALG,
#             "best_N": best_N,
#             "best_U_star": best_U_star,
#             "delta": delta,
#             "seed": seed,
#             "U_improve_pct": U_improve_pct,
#             "U_star_improve_pct": U_star_improve_pct,
#             "instance_id": instance_id,
#             "file": p.name,
#         })

# # === 3) Build DataFrame with stable column order ===
# cols = [
#     "dataset","seq_seed","container_shape","MAR","rho","ALG",
#     "best_N","best_U_star","delta","seed",
#     "U_improve_pct","U_star_improve_pct",
#     "instance_id","file"
# ]

# df = pd.DataFrame(rows)

# for c in cols:
#     if c not in df.columns:
#         df[c] = None
# df = df[cols]

# # === 4) Sort and save ===
# df = df.sort_values(by=["dataset","seq_seed","container_shape","MAR","rho","instance_id","ALG"], kind="stable") \
#        .reset_index(drop=True)

# print(f"[INFO] Total rows: {len(df)}")
# # print(df.head(10))

# # Output CSV (no extra merge)
# df.to_csv("all_results_radio.csv", index=False, encoding="utf-8-sig")
# print("[INFO] Saved")


# # =========================
# # Load original results
# df = pd.read_csv("all_results_radio.csv")

# num_cols = ["best_N", "best_U_star", "U_improve_pct", "U_star_improve_pct", "rho", "MAR", "seed"]
# for c in num_cols:
#     if c in df.columns:
#         df[c] = pd.to_numeric(df[c], errors="coerce")

# #  Convert numeric columns to float
# df["best_N"] = pd.to_numeric(df["best_N"], errors="coerce")
# df["best_U_star"] = pd.to_numeric(df["best_U_star"], errors="coerce")
# df["U_improve_pct"] = pd.to_numeric(df["U_improve_pct"], errors="coerce")
# df["U_star_improve_pct"] = pd.to_numeric(df["U_star_improve_pct"], errors="coerce")

# # ✅ Now compute N_U_star safely
# df["N_U_star"] = df["best_N"] - df["best_U_star"]

# # 3️⃣ Group by dataset + container_shape + ALG, and compute means
# summary = df.groupby(["rho", "MAR", "container_shape", "ALG"]).agg(
#     N_U_star_mean=("N_U_star", "mean"),
#     U_improve_mean=("U_improve_pct", "mean"),
#     U_star_improve_mean=("U_star_improve_pct", "mean"),
# ).reset_index()

# # 4️⃣ Pivot the table to make algorithms into columns (now includes BLF)
# pivot_df = summary.pivot(
#     index=["dataset", "container_shape"],
#     columns="ALG",
#     values=["N_U_star_mean", "U_improve_mean", "U_star_improve_mean"]
# )

# # 5️⃣ Flatten multi-index column names
# pivot_df.columns = [
#     f"{alg}_{metric.replace('_mean', '')}"
#     for metric, alg in pivot_df.columns
# ]

# # 6️⃣ Reorder columns to include BLF too
# final_cols = [
#     "BLF_N_U_star", "BLF_U_improve", "BLF_U_star_improve",
#     "ILS_N_U_star", "ILS_U_improve", "ILS_U_star_improve",
#     "GRASP_N_U_star", "GRASP_U_improve", "GRASP_U_star_improve",
#     "GRASP_ILS_N_U_star", "GRASP_ILS_U_improve", "GRASP_ILS_U_star_improve",
# ]

# # If BLF doesn't exist in some cases, pandas will fill NaN — that's OK
# pivot_df = pivot_df.reindex(columns=final_cols)

# # 7️⃣ Reset index
# pivot_df = pivot_df.reset_index()

# # 8️⃣ Save final summary
# pivot_df.to_csv("metaheuristic_summary_radio.csv", index=False)
# print("✅ Saved metaheuristic_summary_radio.csv")


# =========================Summary===================================
# 1) Load original results
df = pd.read_csv("all_results_radio.csv")

# 2) Ensure numeric dtypes
num_cols = ["best_N", "best_U_star", "U_improve_pct", "U_star_improve_pct", "rho", "MAR", "seed"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 3) Derived metric
#    N_U_star = best_N - best_U_star
df["N_U_star"] = df["best_N"] - df["best_U_star"]

# 4) Group by (rho, MAR, container_shape) since we only have GRASP and one dataset
group_keys = ["rho", "MAR", "container_shape"]

summary = (
    df.groupby(group_keys, dropna=False)
      .agg(
          N_U_star_mean=("N_U_star", "mean"),
          N_U_star_std=("N_U_star", "std"),
          U_improve_mean=("U_improve_pct", "mean"),
          U_improve_std=("U_improve_pct", "std"),
          U_star_improve_mean=("U_star_improve_pct", "mean"),
          U_star_improve_std=("U_star_improve_pct", "std"),
          seeds=("seed", "nunique")  # number of distinct seeds in each group
      )
      .reset_index()
      .sort_values(group_keys)
      .reset_index(drop=True)
)

# 5) (Optional) Keep only groups with exactly 5 seeds
# summary = summary[summary["seeds"] == 5].reset_index(drop=True)

# 6) Save final summary (no pivot, GRASP-only)
summary.to_csv("metaheuristic_summary_radio.csv", index=False)
print("✅ Saved metaheuristic_summary_radio.csv ")
