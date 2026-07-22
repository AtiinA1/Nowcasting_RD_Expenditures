"""Validation-tuned Ridge and Elastic Net benchmarks.

The existing paper table already includes OLS-like same-input benchmarks and a
high-dimensional sg-LASSO-MIDAS. This script adds conventional regularized
linear models on the exact same engineered feature spaces as the NN.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

sys.path.append(os.path.dirname(__file__))
from common import CONFIG_ORDER, OUT, TAB, annual_metrics, design_matrix, metrics, prepare_data  # noqa: E402


warnings.filterwarnings("ignore", category=ConvergenceWarning)


RIDGE_ALPHAS = np.logspace(-3, 3, 13)
ENET_ALPHAS = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
L1_RATIOS = [0.05, 0.5]


def predict_levels(prep, z: np.ndarray) -> np.ndarray:
    return np.exp(z * prep.country_std_vec + prep.country_mean_vec)


def evaluate_validation(prep, z: np.ndarray) -> float:
    tmp = prep.frame[prep.masks["val"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    tmp["pred"] = predict_levels(prep, z)[prep.masks["val"]]
    return annual_metrics(tmp, "pred")["MAPE"]


def fit_ols(X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X[fit_mask], y[fit_mask])
    return model


def fit_ridge(prep, X: np.ndarray) -> tuple[Ridge, float, float]:
    best = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(X[prep.masks["train"]], prep.ystd[prep.masks["train"]])
        val_mape = evaluate_validation(prep, model.predict(X))
        if best is None or val_mape < best[0]:
            best = (val_mape, alpha)
    _, alpha = best
    fit_mask = prep.masks["train"] | prep.masks["val"]
    model = Ridge(alpha=alpha)
    model.fit(X[fit_mask], prep.ystd[fit_mask])
    return model, float(alpha), float(best[0])


def fit_elastic_net(prep, X: np.ndarray) -> tuple[ElasticNet, float, float, float]:
    best = None
    for l1_ratio in L1_RATIOS:
        for alpha in ENET_ALPHAS:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, tol=1e-4)
            model.fit(X[prep.masks["train"]], prep.ystd[prep.masks["train"]])
            val_mape = evaluate_validation(prep, model.predict(X))
            if best is None or val_mape < best[0]:
                best = (val_mape, alpha, l1_ratio)
    _, alpha, l1_ratio = best
    fit_mask = prep.masks["train"] | prep.masks["val"]
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, tol=1e-4)
    model.fit(X[fit_mask], prep.ystd[fit_mask])
    return model, float(alpha), float(l1_ratio), float(best[0])


def test_metrics_for_model(prep, model, X: np.ndarray) -> dict[str, float]:
    test = prep.frame[prep.masks["test"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    test["pred"] = predict_levels(prep, model.predict(X))[prep.masks["test"]]
    return annual_metrics(test, "pred")


def nn_metrics_from_existing() -> pd.DataFrame:
    path = os.path.join(OUT, "..", "..", "out", "temporal_annual_all.csv")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        path = "/Users/atin/Nowcasting/Nowcasting_github/additional_analysis/out/temporal_annual_all.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    ann = pd.read_csv(path)
    rows = []
    for cfg in CONFIG_ORDER:
        col = f"NN_{cfg}"
        if col in ann.columns:
            m = metrics(ann.GERD.values, ann[col].values)
            rows.append({"Config": cfg, "Model": "NN", **m})
    if "SGL" in ann.columns:
        m = metrics(ann.GERD.values, ann.SGL.values)
        rows.append({"Config": "AGT", "Model": "sg-LASSO-MIDAS", **m})
    else:
        sgl = "/Users/atin/Nowcasting/Nowcasting_github/additional_analysis/out/sg_lasso_midas_pred.csv"
        if os.path.exists(sgl):
            s = pd.read_csv(sgl)
            merged = ann.merge(s, on=["Country", "Year"], how="left")
            m = metrics(merged.GERD.values, merged.SGL.values)
            rows.append({"Config": "AGT", "Model": "sg-LASSO-MIDAS", **m})
    return pd.DataFrame(rows)


def latex_table(results: pd.DataFrame) -> None:
    subset = results[results.Config.isin(["AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"])].copy()
    order_model = {"OLS": 0, "Ridge": 1, "Elastic Net": 2, "NN": 3, "sg-LASSO-MIDAS": 4}
    subset["mo"] = subset.Model.map(order_model).fillna(9)
    subset.sort_values(["Config", "mo"], inplace=True)
    rows = [
        "Configuration & Model & MAPE (\\%) & RMSE & $R^2$ & Tuning \\\\",
        "\\midrule",
    ]
    for _, r in subset.iterrows():
        tuning = r.get("tuning", "")
        rows.append(
            f"\\textit{{{r.Config}}} & {r.Model} & {r.MAPE:.2f} & {r.RMSE:.2f} & {r.R2:.2f} & {tuning} \\\\"
        )
    text = (
        "% Source: additional_analysis/robustness_overfit/02_regularized_linear_benchmarks.py\n"
        "\\begin{table}[!htb]\n\\centering\n"
        "\\caption{Validation-tuned regularized linear benchmarks under the temporal split. "
        "Ridge and Elastic Net use the same engineered feature spaces, month indicators, country indicators, "
        "and within-country log-standardized target as the neural network. Hyperparameters are selected on the validation fold and performance is reported on annual test country-years.}\n"
        "\\label{tab:regularized_linear_benchmarks}\n"
        "\\begin{tabular}{l l c c c l}\n"
        "\\toprule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(TAB, "regularized_linear_benchmarks_table.tex"), "w") as handle:
        handle.write(text)


def main() -> None:
    prep = prepare_data()
    rows = []
    for cfg in CONFIG_ORDER:
        cols = prep.configs[cfg]
        X, _, _ = design_matrix(prep, cols, prep.masks["train"])
        fit_mask = prep.masks["train"] | prep.masks["val"]

        ols = fit_ols(X, prep.ystd, fit_mask)
        rows.append({"Config": cfg, "Model": "OLS", "tuning": "", **test_metrics_for_model(prep, ols, X)})

        ridge, alpha, val_mape = fit_ridge(prep, X)
        rows.append({
            "Config": cfg,
            "Model": "Ridge",
            "tuning": f"$\\alpha={alpha:.3g}$",
            "val_MAPE": val_mape,
            **test_metrics_for_model(prep, ridge, X),
        })

        enet, alpha, l1_ratio, val_mape = fit_elastic_net(prep, X)
        rows.append({
            "Config": cfg,
            "Model": "Elastic Net",
            "tuning": f"$\\alpha={alpha:.3g}$, $l_1={l1_ratio:.2g}$",
            "val_MAPE": val_mape,
            "nonzero": int(np.sum(np.abs(enet.coef_) > 1e-10)),
            **test_metrics_for_model(prep, enet, X),
        })
        print(f"finished {cfg}", flush=True)

    results = pd.DataFrame(rows)
    nn = nn_metrics_from_existing()
    if len(nn):
        nn["tuning"] = ""
        results = pd.concat([results, nn], ignore_index=True, sort=False)
    results.to_csv(os.path.join(OUT, "regularized_linear_benchmarks.csv"), index=False)
    latex_table(results)
    print(results.sort_values(["Config", "Model"])[["Config", "Model", "MAPE", "RMSE", "R2", "tuning"]].to_string(index=False))
    print("saved regularized linear benchmarks")


if __name__ == "__main__":
    main()
