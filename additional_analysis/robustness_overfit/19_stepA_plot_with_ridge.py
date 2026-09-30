"""Add a validation-tuned Ridge baseline to the matched-refit Step A plot."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "out" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "14_regenerate_fair_fig2_candidates.py"
SPEC = importlib.util.spec_from_file_location("stepa_matched_refit", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT = HERE / "out" / "paper_stepA_refresh"
CODE = Path(__file__).resolve().parents[2]
ASSETS = Path(
    os.environ.get(
        "NOWCASTING_DECK_ASSETS",
        CODE / "paper" / "figures" / "UpdatedResults",
    )
)
FIGURE = ASSETS / "c1_stepA_dist_matched_refit_with_ridge.png"
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

NN_COLOR = "#35658A"
OLS_COLOR = "#B7BC8B"
RIDGE_COLOR = "#C88B32"
EDGE_COLOR = "#333333"
FILL_ALPHA = 0.80


def design(prep, cfg: str, fit_mask: np.ndarray) -> np.ndarray:
    raw = prep.frame[prep.configs[cfg]].fillna(0).astype(float).to_numpy()
    scaler = StandardScaler().fit(raw[fit_mask])
    return np.column_stack(
        [scaler.transform(raw), prep.months.to_numpy(), prep.countries.to_numpy()]
    )


def annual_metrics(frame: pd.DataFrame, pred_col: str = "pred") -> dict[str, float]:
    annual = base.annual_errors(frame, pred_col)
    return {
        "n": len(annual),
        "MAPE": annual.APE.mean(),
        "median_APE": annual.APE.median(),
        "R2": r2_score(annual.actual, annual.pred),
        "RMSE": np.sqrt(np.mean((annual.actual - annual.pred) ** 2)),
    }


def validation_tuned_ridge(
    prep,
    cfg: str,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    refit_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    train_target = base.target_transform(prep.frame, train_mask)
    train_x = design(prep, cfg, train_mask)
    val_base = prep.frame.loc[val_mask, ["Country", "Year", "Month", "rd_expenditure"]].copy()
    tuning_rows = []

    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=float(alpha)).fit(
            train_x[train_mask], train_target.ystd[train_mask]
        )
        z = model.predict(train_x[val_mask])
        prediction = np.exp(
            z * train_target.country_std_vec[val_mask]
            + train_target.country_mean_vec[val_mask]
        )
        frame = val_base.copy()
        frame["pred"] = prediction
        tuning_rows.append(
            {
                "Config": cfg,
                "alpha": float(alpha),
                "validation_MAPE": base.annual_errors(frame, "pred").APE.mean(),
            }
        )

    tuning = pd.DataFrame(tuning_rows)
    selected_alpha = float(tuning.loc[tuning.validation_MAPE.idxmin(), "alpha"])

    refit_target = base.target_transform(prep.frame, refit_mask)
    refit_x = design(prep, cfg, refit_mask)
    model = Ridge(alpha=selected_alpha).fit(
        refit_x[refit_mask], refit_target.ystd[refit_mask]
    )
    z = model.predict(refit_x[test_mask])
    prediction = np.exp(
        z * refit_target.country_std_vec[test_mask]
        + refit_target.country_mean_vec[test_mask]
    )
    return prediction, selected_alpha, tuning


def style_boxplot(boxplot, color: str) -> None:
    for box in boxplot["boxes"]:
        box.set(
            facecolor=color,
            edgecolor=EDGE_COLOR,
            alpha=FILL_ALPHA,
            linewidth=0.8,
        )
    for element in ("whiskers", "caps"):
        for artist in boxplot[element]:
            artist.set(color=EDGE_COLOR, linewidth=0.8)


def add_boxplot(ax, values, positions, color):
    boxplot = ax.boxplot(
        values,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": EDGE_COLOR,
            "markersize": 4,
        },
        medianprops={"color": EDGE_COLOR, "linewidth": 1.5},
    )
    style_boxplot(boxplot, color)
    return boxplot


def render_plot(annual: dict[str, dict[str, pd.DataFrame]]) -> None:
    positions = np.arange(len(base.CONFIG_ORDER))
    nn_values = [annual[cfg]["Neural network"].APE.to_numpy() for cfg in base.CONFIG_ORDER]
    ols_values = [annual[cfg]["Centered min.-norm OLS"].APE.to_numpy() for cfg in base.CONFIG_ORDER]
    ridge_values = [annual[cfg]["Ridge"].APE.to_numpy() for cfg in base.CONFIG_ORDER]

    fig, ax = plt.subplots(figsize=(8.1, 4.5))
    nn_plot = add_boxplot(ax, nn_values, positions - 0.25, NN_COLOR)
    ols_plot = add_boxplot(ax, ols_values, positions, OLS_COLOR)
    ridge_plot = add_boxplot(ax, ridge_values, positions + 0.25, RIDGE_COLOR)

    ax.set_xticks(positions)
    ax.set_xticklabels(base.CONFIG_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Absolute percentage error (%)")
    ax.set_title("(a) Test-error dispersion")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#666666")
    ax.tick_params(axis="both", colors="#222222")
    ax.legend(
        [nn_plot["boxes"][0], ols_plot["boxes"][0], ridge_plot["boxes"][0]],
        ["Neural network", "Centered min.-norm OLS", "Ridge"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.47, 0.995),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
        fontsize=9,
    )
    ax.annotate(
        "diamond = mean",
        xy=(0.99, 0.90),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    prep = base.prepare_data()
    train_mask = prep.masks["train"]
    val_mask = prep.masks["val"]
    refit_mask = train_mask | val_mask
    test_mask = prep.masks["test"]
    test_base = prep.frame.loc[
        test_mask, ["Country", "Year", "Month", "rd_expenditure"]
    ].copy()
    refit_target = base.target_transform(prep.frame, refit_mask)

    saved_nn = pd.read_csv(OUT / "c1_stepA_matched_refit_nn_predictions.csv")
    annual: dict[str, dict[str, pd.DataFrame]] = {}
    tuning_parts = []
    audit_rows = []

    for cfg in base.CONFIG_ORDER:
        nn_frame = saved_nn[saved_nn.Config.eq(cfg)].copy()
        nn_annual = base.annual_errors(nn_frame, "pred")

        ols_frame = test_base.copy()
        ols_frame["pred"] = base.ols_predictions(
            prep, cfg, refit_mask, refit_target
        )[test_mask]
        ols_annual = base.annual_errors(ols_frame, "pred")

        ridge_prediction, selected_alpha, tuning = validation_tuned_ridge(
            prep, cfg, train_mask, val_mask, refit_mask, test_mask
        )
        tuning_parts.append(tuning)
        ridge_frame = test_base.copy()
        ridge_frame["pred"] = ridge_prediction
        ridge_annual = base.annual_errors(ridge_frame, "pred")

        annual[cfg] = {
            "Neural network": nn_annual,
            "Centered min.-norm OLS": ols_annual,
            "Ridge": ridge_annual,
        }
        for model_name, frame in annual[cfg].items():
            metrics = {
                "Config": cfg,
                "Model": model_name,
                **annual_metrics(
                    frame.rename(columns={"actual": "rd_expenditure", "pred": "pred"}),
                    "pred",
                ),
                "selected_alpha": selected_alpha if model_name == "Ridge" else np.nan,
            }
            audit_rows.append(metrics)
        print(f"{cfg}: Ridge alpha={selected_alpha:g}", flush=True)

    audit = pd.DataFrame(audit_rows)
    tuning = pd.concat(tuning_parts, ignore_index=True)
    audit.to_csv(OUT / "c1_stepA_matched_refit_with_ridge_audit.csv", index=False)
    tuning.to_csv(OUT / "c1_stepA_matched_refit_ridge_tuning.csv", index=False)
    render_plot(annual)

    print(FIGURE)
    print("\n=== Matched-refit annual test metrics ===")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
