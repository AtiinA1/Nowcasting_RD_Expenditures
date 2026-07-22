"""AGT lag-order sensitivity for cumulative lag orders 2 and 4.

The production feature matrix contains annual topic averages at lags 1-3. Lag
4 is constructed by shifting each lag-3 annual average by one additional year
within country. All model and training settings are inherited unchanged from
the current production pure-NN module.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ROBUST = ROOT / "additional_analysis" / "robustness_overfit"
PRODUCTION = ROBUST / "10_all_configs_updated_pure_nn.py"
OUT = ROBUST / "out" / "agt_lag_order_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)


def load_production_module():
    spec = importlib.util.spec_from_file_location("production_pure_nn", PRODUCTION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PRODUCTION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_lag4(prep):
    frame = prep.frame.copy()
    lag3 = sorted(c for c in prep.configs["AGT"] if c.endswith("_yearly_avg_lag3"))
    annual = frame[["Country", "Year", *lag3]].drop_duplicates(["Country", "Year"])
    annual = annual.sort_values(["Country", "Year"])
    lag4_names = []
    for col in lag3:
        new_col = col.replace("_lag3", "_lag4")
        annual[new_col] = annual.groupby("Country")[col].shift(1)
        lag4_names.append(new_col)
    frame = frame.merge(annual[["Country", "Year", *lag4_names]], on=["Country", "Year"], how="left")
    return replace(prep, frame=frame), lag4_names


def main() -> None:
    production = load_production_module()
    prep = production.prepare_data()
    prep, lag4_cols = add_lag4(prep)

    base = prep.configs["AGT"]
    lag1 = sorted(c for c in base if c.endswith("_yearly_avg_lag1"))
    lag2 = sorted(c for c in base if c.endswith("_yearly_avg_lag2"))
    lag3 = sorted(c for c in base if c.endswith("_yearly_avg_lag3"))
    if not (len(lag1) == len(lag2) == len(lag3) == len(lag4_cols) == 57):
        raise RuntimeError("Expected 57 Google Trends topic features at every lag")

    configs = dict(prep.configs)
    configs["AGT_L2"] = lag1 + lag2
    configs["AGT_L4"] = lag1 + lag2 + lag3 + lag4_cols
    prep = replace(prep, configs=configs)

    result_parts = []
    for config in ("AGT_L2", "AGT_L4"):
        results, history, predictions, member_predictions = production.run_config(prep, config)
        results.to_csv(OUT / f"{config.lower()}_results.csv", index=False)
        history.to_csv(OUT / f"{config.lower()}_history.csv", index=False)
        predictions.to_csv(OUT / f"{config.lower()}_predictions.csv", index=False)
        member_predictions.to_csv(OUT / f"{config.lower()}_member_predictions.csv", index=False)
        result_parts.append(results)

    new_results = pd.concat(result_parts, ignore_index=True)
    lag1_results = pd.read_csv(
        ROBUST / "out" / "agt_lag1_sensitivity" / "agt_lag1_results.csv"
    )
    production_results = pd.read_csv(
        ROBUST / "out" / "all_configs_updated_pure_nn" / "all_configs_updated_pure_nn_results.csv"
    )
    rows = []
    sources = [
        (1, "AGT_L1", 57, lag1_results),
        (2, "AGT_L2", 114, new_results),
        (3, "AGT", 171, production_results),
        (4, "AGT_L4", 228, new_results),
    ]
    for order, config, n_features, source in sources:
        row = source[(source.Config == config) & (source.ensemble_size == 15)].iloc[0]
        rows.append({
            "Lag_order": order,
            "n_GT_features": n_features,
            "Validation_MAPE": row.val_MAPE,
            "Test_MAPE": row.test_MAPE,
            "Test_RMSE": row.test_RMSE,
            "Test_R2": row.test_R2,
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "agt_lag_orders_1_to_4_metrics.csv", index=False)
    print("\nAGT lag-order sensitivity, 15-member ensemble")
    print(comparison.to_string(index=False))
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
