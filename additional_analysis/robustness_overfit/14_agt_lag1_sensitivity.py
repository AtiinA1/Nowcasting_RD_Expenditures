"""AGT lag sensitivity under the current pure-NN temporal setup.

This experiment changes only the AGT input lag order: the production model uses
annual Google Trends topic averages at lags 1, 2, and 3, while this sensitivity
run retains lag 1 only. Architecture, temporal split, preprocessing, training,
early stopping, seeds, and ensemble size are inherited from the production
pure-NN module without modification.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ROBUST = ROOT / "additional_analysis" / "robustness_overfit"
PRODUCTION = ROBUST / "10_all_configs_updated_pure_nn.py"
OUT = ROBUST / "out" / "agt_lag1_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)


def load_production_module():
    spec = importlib.util.spec_from_file_location("production_pure_nn", PRODUCTION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PRODUCTION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    production = load_production_module()
    prep = production.prepare_data()
    lag1_cols = [c for c in prep.configs["AGT"] if c.endswith("_yearly_avg_lag1")]
    if len(prep.configs["AGT"]) != 3 * len(lag1_cols):
        raise RuntimeError("The production AGT feature set is not balanced across lags 1-3")

    configs = dict(prep.configs)
    configs["AGT_L1"] = lag1_cols
    lag1_prep = replace(prep, configs=configs)
    results, history, predictions, member_predictions = production.run_config(lag1_prep, "AGT_L1")

    results.to_csv(OUT / "agt_lag1_results.csv", index=False)
    history.to_csv(OUT / "agt_lag1_history.csv", index=False)
    predictions.to_csv(OUT / "agt_lag1_predictions.csv", index=False)
    member_predictions.to_csv(OUT / "agt_lag1_member_predictions.csv", index=False)

    lag1 = results.loc[results.ensemble_size == 15].iloc[0]
    production_results = pd.read_csv(
        ROBUST / "out" / "all_configs_updated_pure_nn" / "all_configs_updated_pure_nn_results.csv"
    )
    lag3 = production_results[
        (production_results.Config == "AGT") & (production_results.ensemble_size == 15)
    ].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "Specification": "AGT lag 1",
                "n_GT_features": len(lag1_cols),
                "Validation_MAPE": lag1.val_MAPE,
                "Test_MAPE": lag1.test_MAPE,
                "Test_RMSE": lag1.test_RMSE,
                "Test_R2": lag1.test_R2,
            },
            {
                "Specification": "AGT lags 1-3",
                "n_GT_features": len(prep.configs["AGT"]),
                "Validation_MAPE": lag3.val_MAPE,
                "Test_MAPE": lag3.test_MAPE,
                "Test_RMSE": lag3.test_RMSE,
                "Test_R2": lag3.test_R2,
            },
        ]
    )
    comparison.to_csv(OUT / "agt_lag1_vs_lag3_metrics.csv", index=False)
    print("\nAGT lag sensitivity, 15-member ensemble")
    print(comparison.to_string(index=False))
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()
