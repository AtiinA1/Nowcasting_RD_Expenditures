from wandb.apis.public import Api
import csv, os, json

project = "epfl_stip/nowcasting-rd-mlp"
sweeps = {
    "AGT":    "km019fag",
    "AGTwRD": "5ia24lux",
    "AllVar": "ibuldymc",
    "LagRD":  "02ufx6ki",
    "Macros": "xhlqftqp",
    "MGT":    "6glj87rl",
    "MGTwRD": "mqgc38my",
}

# Rank by out-of-sample performance (lower is better)
rank_metric = "test_mape"
top_k = 3

# Metrics to extract (saved in CSV)
summary_keys = [
    # Validation (ensemble prediction)
    "final_ensemble_val_rmse", "final_ensemble_val_mae", "final_ensemble_val_mape",
    # Aggregates across ensemble members
    "final_avg_train_rmse", "final_avg_val_rmse",
    # Test (ensemble prediction)
    "test_rmse", "test_mae", "test_mape", "test_r2",
]

out_dir = "top_configs"
os.makedirs(out_dir, exist_ok=True)

api = Api()

def get_float(summary, key):
    v = summary.get(key, None)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def rank_value(run, default=float("inf")):
    v = get_float(run.summary, rank_metric)
    return v if v is not None else default

for variant, sweep_id in sweeps.items():
    sweep = api.sweep(f"{project}/{sweep_id}")
    runs = [r for r in sweep.runs if rank_value(r) != float("inf")]
    # Prefer finished, then sort ascending by test MAPE
    runs.sort(key=lambda r: (r.state != "finished", rank_value(r)))
    top = runs[:top_k]

    print(f"\n=== {variant} ({sweep_id}) Top {len(top)} by {rank_metric} ===")

    rows = []
    for r in top:
        row = {
            "variant": variant,
            "sweep_id": sweep_id,
            "run_name": r.name,
            "run_id": r.id,
            "state": r.state,
            "url": r.url,
            "config": json.dumps(r.config, sort_keys=True),
        }
        for k in summary_keys:
            row[k] = get_float(r.summary, k)

        print(f"- {r.name} | {rank_metric}: {row.get(rank_metric)} | state: {r.state}")
        print(f"  url: {r.url}")
        print(f"  config: {row['config']}\n")
        rows.append(row)

    fieldnames = ["variant","sweep_id","run_name","run_id","state","url","config"] + summary_keys
    csv_path = os.path.join(out_dir, f"top_{variant.lower()}_{sweep_id}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)

print("\nSaved CSVs in:", os.path.abspath(out_dir))
