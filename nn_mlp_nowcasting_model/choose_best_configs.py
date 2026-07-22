from __future__ import annotations
import os
import json
import csv
from collections import Counter, defaultdict
from statistics import median
from typing import Dict, Any, List, Tuple

from wandb.apis.public import Api


PROJECT = "epfl_stip/nowcasting-rd-mlp"

# Variant → sweep id (update if you start new sweeps)
SWEEPS: Dict[str, str] = {
    "AGT":    "km019fag",
    "AGTwRD": "5ia24lux",
    "AllVar": "ibuldymc",
    "LagRD":  "02ufx6ki",
    "Macros": "xhlqftqp",
    "MGT":    "6glj87rl",
    "MGTwRD": "mqgc38my",
}

# Primary ranking metric
RANK_METRIC = "test_mape"

# Additional metrics to report
SUMMARY_KEYS = [
    "final_ensemble_val_rmse",
    "final_ensemble_val_mae",
    "final_ensemble_val_mape",
    "final_avg_train_rmse",
    "final_avg_val_rmse",
    "test_rmse",
    "test_mae",
    "test_mape",
    "test_r2",
]

# Hparams we consider when comparing configs across variants
HPARAM_KEYS = [
    "learning_rate",
    "batch_size",
    "hidden1_dim",
    "hidden2_dim",
    "hidden3_dim",
    "embedding_dim",
    "size_ensemble",
    "num_epochs",
    "patience",
    "lr_milestone",
    "lr_gamma",
    "optimizer",
    "weight_decay",
    "dropout_rate",
]


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def get_summary_value(run, key: str) -> float | None:
    return to_float(run.summary.get(key))


def extract_value(possibly_wrapped):
    # Some wandb configs might appear as {"value": X}
    if isinstance(possibly_wrapped, dict) and "value" in possibly_wrapped:
        return possibly_wrapped["value"]
    return possibly_wrapped


def extract_hparams(config: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in HPARAM_KEYS:
        if k in config:
            out[k] = extract_value(config[k])
    return out


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Cast ints where appropriate, keep floats as floats
    norm: Dict[str, Any] = {}
    int_like = {"batch_size", "hidden1_dim", "hidden2_dim", "hidden3_dim", "embedding_dim",
                "size_ensemble", "num_epochs", "patience", "lr_milestone"}
    for k, v in cfg.items():
        if v is None:
            continue
        if k in int_like:
            try:
                v = int(v)
            except Exception:
                pass
        elif isinstance(v, (int, float)):
            # keep numeric types as-is
            v = float(v) if isinstance(v, float) else v
        norm[k] = v
    return norm


def cfg_key(cfg: Dict[str, Any]) -> str:
    # Deterministic string key for dict comparison
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


# -------- Relaxed common-config matching helpers --------
# Only consider a small subset of hparams for matching across variants
RELAX_INCLUDE_KEYS = {"hidden1_dim", "hidden2_dim", "hidden3_dim", "embedding_dim", "optimizer"}
RELAX_FLOAT_ROUND_PLACES = 4  # kept for completeness

def relaxed_signature(cfg: Dict[str, Any]) -> Dict[str, Any]:
    relaxed: Dict[str, Any] = {}
    for k in RELAX_INCLUDE_KEYS:
        if k not in cfg:
            continue
        v = cfg[k]
        if isinstance(v, float):
            relaxed[k] = round(v, RELAX_FLOAT_ROUND_PLACES)
        else:
            relaxed[k] = v
    return relaxed


def main():
    out_dir = "best_configs"
    os.makedirs(out_dir, exist_ok=True)

    api = Api()

    best_per_variant: List[Dict[str, Any]] = []
    # Keyed by normalized config string → list of (variant, test_mape, run_url)
    common_cfg_map: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
    normalized_cfg_store: Dict[str, Dict[str, Any]] = {}

    # 1) Pick the best run per variant by test_mape
    for variant, sweep_id in SWEEPS.items():
        sw = api.sweep(f"{PROJECT}/{sweep_id}")
        runs = list(sw.runs)
        # Only finished, with valid test_mape
        finished = [r for r in runs if r.state == "finished" and get_summary_value(r, RANK_METRIC) is not None]
        finished.sort(key=lambda r: get_summary_value(r, RANK_METRIC))
        if not finished:
            print(f"[WARN] No finished runs with {RANK_METRIC} for {variant} ({sweep_id})")
            continue

        top = finished[0]
        # Extract metrics
        row = {
            "variant": variant,
            "sweep_id": sweep_id,
            "run_name": top.name,
            "run_id": top.id,
            "url": top.url,
            "state": top.state,
        }
        for k in SUMMARY_KEYS:
            row[k] = get_summary_value(top, k)

        # Extract and normalize hparams
        hparams = normalize_config(extract_hparams(top.config or {}))
        row["config"] = json.dumps(hparams, sort_keys=True)
        best_per_variant.append(row)

        # Track for common-config analysis
        relaxed = relaxed_signature(hparams)
        key = cfg_key(relaxed)
        if row[RANK_METRIC] is not None:
            common_cfg_map[key].append((variant, float(row[RANK_METRIC]), row["url"]))
            normalized_cfg_store[key] = relaxed

    # Save best per variant
    best_csv = os.path.join(out_dir, "best_per_variant.csv")
    fieldnames = ["variant", "sweep_id", "run_name", "run_id", "state", "url"] + SUMMARY_KEYS + ["config"]
    with open(best_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(best_per_variant)

    print(f"Saved best-per-variant table: {os.path.abspath(best_csv)}")

    # 2) Find configs that appear in multiple variants and rank by average test_mape
    common_rows = []
    for key, lst in common_cfg_map.items():
        if len(lst) < 2:
            continue  # only consider configs seen in at least 2 variants
        variants = [v for (v, _m, _u) in lst]
        metrics = [m for (_v, m, _u) in lst]
        urls = [u for (_v, _m, u) in lst]
        avg_mape = sum(metrics) / len(metrics)
        max_mape = max(metrics)
        common_rows.append({
            "num_variants": len(set(variants)),
            "variants": ",".join(sorted(set(variants))),
            "avg_test_mape": avg_mape,
            "max_test_mape": max_mape,
            "config": key,
            "example_urls": ";".join(urls[:3]),
        })

    # Sort by: more variants first, then lower avg mape
    common_rows.sort(key=lambda r: (-r["num_variants"], r["avg_test_mape"]))

    common_csv = os.path.join(out_dir, "global_common_configs.csv")
    with open(common_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["num_variants","variants","avg_test_mape","max_test_mape","config","example_urls"])
        w.writeheader()
        w.writerows(common_rows)

    print(f"Saved common-configs table: {os.path.abspath(common_csv)}")

    # 3) Propose a consensus config if no strong common config emerges
    # Use the best per-variant configs and aggregate: median for numeric, mode for categorical
    if best_per_variant:
        numeric_keys = {"learning_rate","weight_decay","dropout_rate","lr_gamma"}
        integer_keys = {"batch_size","hidden1_dim","hidden2_dim","hidden3_dim","embedding_dim",
                        "size_ensemble","num_epochs","patience","lr_milestone"}
        categorical_keys = {"optimizer"}

        collected: Dict[str, List[Any]] = defaultdict(list)
        for row in best_per_variant:
            cfg = json.loads(row["config"])
            for k, v in cfg.items():
                collected[k].append(v)

        consensus: Dict[str, Any] = {}
        for k in HPARAM_KEYS:
            if k not in collected or not collected[k]:
                continue
            vals = collected[k]
            if k in numeric_keys:
                try:
                    consensus[k] = float(median([float(x) for x in vals]))
                except Exception:
                    consensus[k] = vals[0]
            elif k in integer_keys:
                try:
                    # nearest integer median
                    consensus[k] = int(round(median([float(x) for x in vals])))
                except Exception:
                    # fall back to majority vote
                    consensus[k] = Counter(vals).most_common(1)[0][0]
            elif k in categorical_keys:
                consensus[k] = Counter(vals).most_common(1)[0][0]
            else:
                consensus[k] = Counter(vals).most_common(1)[0][0]

        with open(os.path.join(out_dir, "consensus_config.json"), "w") as f:
            json.dump(consensus, f, indent=2, sort_keys=True)
        print("Wrote consensus suggestion:", os.path.abspath(os.path.join(out_dir, "consensus_config.json")))


if __name__ == "__main__":
    main()


