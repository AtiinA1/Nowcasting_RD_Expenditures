from __future__ import annotations
import os
import csv
import json
from typing import List, Dict, Any

from wandb.apis.public import Api

PROJECT = "epfl_stip/nowcasting-rd-mlp"
GROUPS = ["fixed_cfg", "consensus_config"]  # run groups we export

# Variants we recognize from tags or names
KNOWN_VARIANTS = {"AGT","AGTwRD","AllVar","LagRD","Macros","MGT","MGTwRD"}

# Metrics we will export (extend if needed)
SUMMARY_KEYS = [
    # Validation (ensemble)
    "final_ensemble_val_rmse",
    "final_ensemble_val_mae",
    "final_ensemble_val_mape",
    # Aggregates
    "final_avg_train_rmse",
    "final_avg_val_rmse",
    # Test (ensemble)
    "test_rmse",
    "test_mae",
    "test_mape",
    "test_r2",
]

# Hyperparameters to include in the config blob for quick inspection
HPARAM_KEYS = [
    "learning_rate","batch_size","hidden1_dim","hidden2_dim","hidden3_dim","embedding_dim",
    "size_ensemble","num_epochs","patience","lr_milestone","lr_gamma","optimizer","weight_decay","dropout_rate",
]


def to_float(val: Any) -> float | None:
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def extract_variant(run) -> str:
    # Prefer a known tag
    tags = set(run.tags or [])
    tagged = list(KNOWN_VARIANTS & tags)
    if tagged:
        return tagged[0]
    # Fallback: try prefix of name like "AGT_fixed"
    name = (run.name or "").split("_")[0]
    if name in KNOWN_VARIANTS:
        return name
    return "unknown"


def extract_config_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in HPARAM_KEYS:
        if k in cfg:
            v = cfg[k]
            if isinstance(v, dict) and "value" in v:
                v = v["value"]
            out[k] = v
    return out


def export_group(group: str, out_dir: str, api: Api):
    runs = api.runs(PROJECT, {"group": group})
    rows: List[Dict[str, Any]] = []
    for r in runs:
        variant = extract_variant(r)
        row: Dict[str, Any] = {
            "group": group,
            "variant": variant,
            "run_name": r.name,
            "run_id": r.id,
            "state": r.state,
            "url": r.url,
        }
        # metrics
        for k in SUMMARY_KEYS:
            row[k] = to_float(r.summary.get(k))
        # config (subset)
        cfg_subset = extract_config_subset(r.config or {})
        row["config"] = json.dumps(cfg_subset, sort_keys=True)
        rows.append(row)

    # Save one CSV for the whole group
    os.makedirs(out_dir, exist_ok=True)
    group_csv = os.path.join(out_dir, f"{group}_runs.csv")
    fieldnames = ["group","variant","run_name","run_id","state","url"] + SUMMARY_KEYS + ["config"]
    with open(group_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Also save per-variant CSVs
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    for v, vrows in by_variant.items():
        v_csv = os.path.join(out_dir, f"{group}_{v}_runs.csv")
        with open(v_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(vrows)

    print(f"{group}: saved {len(rows)} runs → {group_csv}")


def main():
    api = Api()
    out_dir = "fixed_results"
    for g in GROUPS:
        export_group(g, out_dir, api)
    print("All done. Output dir:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()




