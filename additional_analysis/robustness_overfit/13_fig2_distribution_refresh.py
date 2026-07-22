"""Build an updated Figure 2 variant with error dispersion and RMSE.

The figure consumes saved updated pure-NN temporal predictions and reconstructs
the same chronological OLS comparison at each input space. It writes only a
paper-ready figure and an audit CSV; no experiments are retrained.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from common import CONFIG_ORDER, prepare_data  # noqa: E402


CODE = Path(os.environ.get("NOWCASTING_ROOT", Path(__file__).resolve().parents[2]))
PAPER = Path(os.environ.get("NOWCASTING_PAPER_DIR", CODE / "paper"))
ROBUST = CODE / "additional_analysis" / "robustness_overfit"
UPDATED = ROBUST / "out" / "all_configs_updated_pure_nn"
OUT = ROBUST / "out" / "paper_stepA_refresh"
FIG_TEMP = PAPER / "figures" / "Nowcast_Model_Temporal"

# Restrained journal palette: focal model in blue, benchmark in light olive.
NN_COLOR = "#35658A"
OLS_COLOR = "#B7BC8B"
EDGE_COLOR = "#333333"
FILL_ALPHA = 0.80

PALETTE_PREVIEWS = {
    "option3_teal_gray": {
        "nn": "#2A6F6F",
        "ols": "#B7B1A8",
        "edge": "#333333",
        "alpha": 0.82,
    },
    "option5_blue_olive": {
        "nn": "#35658A",
        "ols": "#8A8F5A",
        "edge": "#333333",
        "alpha": 0.80,
    },
    "option5_blue_lightolive": {
        "nn": "#35658A",
        "ols": "#B7BC8B",
        "edge": "#333333",
        "alpha": 0.80,
    },
}


def annual_ape(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    ann = (
        df.groupby(["Country", "Year"], as_index=False)
        .agg(actual=("rd_expenditure", "mean"), pred=(pred_col, "mean"))
    )
    ann["APE"] = np.abs(ann["actual"] - ann["pred"]) / ann["actual"] * 100
    return ann


def ols_predictions(prep, cfg: str) -> np.ndarray:
    """Centered minimum-norm OLS on the same target/input space."""
    fit_mask = prep.masks["train"] | prep.masks["val"]
    Xraw = prep.frame[prep.configs[cfg]].fillna(0).astype(float).values
    mu = Xraw[prep.masks["train"]].mean(axis=0)
    sd = Xraw[prep.masks["train"]].std(axis=0)
    sd[sd == 0] = 1.0
    X = np.column_stack([
        (Xraw - mu) / sd,
        prep.months.values,
        prep.countries.values,
    ])
    model = LinearRegression(fit_intercept=True)
    model.fit(X[fit_mask], prep.ystd[fit_mask])
    z = model.predict(X)
    return np.exp(z * prep.country_std_vec + prep.country_mean_vec)


def render_figure(
    audit: pd.DataFrame,
    nn_apes: list[np.ndarray],
    ols_apes: list[np.ndarray],
    nn_color: str,
    ols_color: str,
    edge_color: str,
    fill_alpha: float,
    outfile: Path,
) -> None:
    rmse = audit.pivot(index="Config", columns="Model", values="RMSE").loc[CONFIG_ORDER]
    pos = np.arange(len(CONFIG_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.45, 1.0]})

    b1 = axes[0].boxplot(
        nn_apes,
        positions=pos - 0.18,
        widths=0.32,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": edge_color, "markersize": 4},
        medianprops={"color": edge_color, "linewidth": 1.5},
    )
    b2 = axes[0].boxplot(
        ols_apes,
        positions=pos + 0.18,
        widths=0.32,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": edge_color, "markersize": 4},
        medianprops={"color": edge_color, "linewidth": 1.5},
    )
    for boxes, color in [(b1["boxes"], nn_color), (b2["boxes"], ols_color)]:
        for box in boxes:
            box.set(facecolor=color, edgecolor=edge_color, alpha=fill_alpha, linewidth=0.8)
    for element in ["whiskers", "caps"]:
        for artist in b1[element] + b2[element]:
            artist.set(color=edge_color, linewidth=0.8)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(CONFIG_ORDER, rotation=35, ha="right")
    axes[0].set_ylabel("Absolute percentage error (%)")
    axes[0].set_title("(a) Test-error dispersion")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].spines[["left", "bottom"]].set_color("#666666")
    axes[0].tick_params(axis="both", colors="#222222")
    axes[0].legend(
        [b1["boxes"][0], b2["boxes"][0]],
        ["Neural network", "Centered min.-norm OLS"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.44, 0.99),
        ncol=2,
    )
    axes[0].annotate("diamond = mean", xy=(0.98, 0.93), xycoords="axes fraction", ha="right", fontsize=8)

    width = 0.34
    axes[1].bar(
        pos - width / 2,
        rmse["Neural network"],
        width,
        color=nn_color,
        edgecolor=edge_color,
        linewidth=0.6,
        alpha=fill_alpha,
    )
    axes[1].bar(
        pos + width / 2,
        rmse["OLS"],
        width,
        color=ols_color,
        edgecolor=edge_color,
        linewidth=0.6,
        alpha=fill_alpha,
    )
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(CONFIG_ORDER, rotation=35, ha="right")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("(b) Root mean squared error")
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["left", "bottom"]].set_color("#666666")
    axes[1].tick_params(axis="both", colors="#222222")

    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prep = prepare_data()
    test = prep.frame[prep.masks["test"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    nn_pred = pd.read_csv(UPDATED / "all_configs_updated_pure_nn_predictions.csv")
    nn_pred = nn_pred.groupby(["Config", "Country", "Year", "Month"], as_index=False).agg(
        rd_expenditure=("rd_expenditure", "first"),
        pred=("pred", "mean"),
    )
    nn_apes: list[np.ndarray] = []
    ols_apes: list[np.ndarray] = []
    audit_rows = []
    for cfg in CONFIG_ORDER:
        nn_cfg = nn_pred[nn_pred["Config"] == cfg].copy()
        nn_ann = annual_ape(nn_cfg, "pred")
        ols_frame = test.copy()
        ols_frame["pred"] = ols_predictions(prep, cfg)[prep.masks["test"]]
        ols_ann = annual_ape(ols_frame, "pred")
        nn_apes.append(nn_ann["APE"].to_numpy())
        ols_apes.append(ols_ann["APE"].to_numpy())
        for model, ann in [("Neural network", nn_ann), ("OLS", ols_ann)]:
            audit_rows.append(
                {
                    "Config": cfg,
                    "Model": model,
                    "n": len(ann),
                    "mean_APE": ann["APE"].mean(),
                    "median_APE": ann["APE"].median(),
                    "iqr_APE": ann["APE"].quantile(0.75) - ann["APE"].quantile(0.25),
                    "p90_APE": ann["APE"].quantile(0.90),
                    "RMSE": np.sqrt(np.mean((ann["actual"] - ann["pred"]) ** 2)),
                }
            )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "updated_fig2_distribution_audit.csv", index=False)

    render_figure(
        audit,
        nn_apes,
        ols_apes,
        NN_COLOR,
        OLS_COLOR,
        EDGE_COLOR,
        FILL_ALPHA,
        FIG_TEMP / "NNvsOLS_Temporal_distribution.png",
    )
    for name, palette in PALETTE_PREVIEWS.items():
        render_figure(
            audit,
            nn_apes,
            ols_apes,
            palette["nn"],
            palette["ols"],
            palette["edge"],
            palette["alpha"],
            FIG_TEMP / f"_preview_NNvsOLS_Temporal_distribution_{name}.png",
        )
    print(FIG_TEMP / "NNvsOLS_Temporal_distribution.png")
    for name in PALETTE_PREVIEWS:
        print(FIG_TEMP / f"_preview_NNvsOLS_Temporal_distribution_{name}.png")
    print(OUT / "updated_fig2_distribution_audit.csv")


if __name__ == "__main__":
    main()
