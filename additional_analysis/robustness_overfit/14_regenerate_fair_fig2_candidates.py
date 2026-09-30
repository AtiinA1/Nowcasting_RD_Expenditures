"""Regenerate two fair-comparison candidates for the Step A result figure.

Candidate 1 keeps the published visual treatment and saved neural-network
predictions, but fits centered minimum-norm OLS on the training split only.

Candidate 2 applies a matched final-refit protocol. The validation-selected
epoch for each neural-network ensemble member is taken from the saved training
history, both models are refit on train + validation with preprocessing
recomputed on that combined sample, and evaluation remains on the untouched
test split. It also overlays the individual country-year errors; triangles at
the top denote observations above the 100 percent display range.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "out", "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.dirname(__file__))
from common import CONFIG_ORDER, prepare_data  # noqa: E402


CODE = Path(os.environ.get("NOWCASTING_ROOT", Path(__file__).resolve().parents[2]))
ROBUST = CODE / "additional_analysis" / "robustness_overfit"
UPDATED = ROBUST / "out" / "all_configs_updated_pure_nn"
OUT = ROBUST / "out" / "paper_stepA_refresh"
DECK_ASSETS = Path(
    os.environ.get(
        "NOWCASTING_DECK_ASSETS",
        CODE / "paper" / "figures" / "UpdatedResults",
    )
)

TRAIN_ONLY_FIG = DECK_ASSETS / "c1_stepA_dist_train_only_ols.png"
MATCHED_REFIT_FIG = DECK_ASSETS / "c1_stepA_dist_matched_refit_points.png"
MATCHED_REFIT_CLEAN_FIG = DECK_ASSETS / "c1_stepA_dist_matched_refit.png"

NN_COLOR = "#35658A"
OLS_COLOR = "#B7BC8B"
EDGE_COLOR = "#333333"
FILL_ALPHA = 0.80

HIDDEN = (64, 16)
DROPOUT = 0.12
WEIGHT_DECAY = 5e-4
LR = 3e-3
ENSEMBLE_SIZE = 15


class WideDeepNN(nn.Module):
    def __init__(self, d: int, n_countries: int):
        super().__init__()
        self.emb = nn.Embedding(n_countries, 4)
        dims = [d + 4] + list(HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(HIDDEN)))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for h in HIDDEN)
        self.out = nn.Linear(dims[-1], 1)
        self.skip = nn.Linear(d, 1)
        self.drop = nn.Dropout(DROPOUT)
        self.act = nn.SiLU()

    def forward(self, x, c):
        z = torch.cat([x, self.emb(c)], dim=1)
        for layer, norm in zip(self.layers, self.norms):
            z = self.drop(self.act(norm(layer(z))))
        return self.out(z) + self.skip(x)


@dataclass(frozen=True)
class TargetTransform:
    ystd: np.ndarray
    country_mean_vec: np.ndarray
    country_std_vec: np.ndarray


def annual_errors(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    ann = (
        df.groupby(["Country", "Year"], as_index=False)
        .agg(actual=("rd_expenditure", "mean"), pred=(pred_col, "mean"))
    )
    ann["APE"] = np.abs(ann["actual"] - ann["pred"]) / ann["actual"] * 100
    return ann


def target_transform(frame: pd.DataFrame, fit_mask: np.ndarray) -> TargetTransform:
    logy = np.log(frame.rd_expenditure.to_numpy(dtype=float))
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for country in frame.Country.unique():
        country_mask = frame.Country.eq(country).to_numpy()
        values = logy[country_mask & fit_mask]
        means[country] = float(values.mean())
        stds[country] = float(max(values.std(), 0.05))
    mean_vec = frame.Country.map(means).to_numpy(dtype=float)
    std_vec = frame.Country.map(stds).to_numpy(dtype=float)
    return TargetTransform(
        ystd=(logy - mean_vec) / std_vec,
        country_mean_vec=mean_vec,
        country_std_vec=std_vec,
    )


def ols_predictions(prep, cfg: str, fit_mask: np.ndarray, target: TargetTransform) -> np.ndarray:
    """Centered minimum-norm OLS with preprocessing estimated on fit_mask."""
    raw = prep.frame[prep.configs[cfg]].fillna(0).astype(float).to_numpy()
    scaler = StandardScaler().fit(raw[fit_mask])
    x = np.column_stack([scaler.transform(raw), prep.months.to_numpy(), prep.countries.to_numpy()])
    model = LinearRegression(fit_intercept=True)
    model.fit(x[fit_mask], target.ystd[fit_mask])
    z = model.predict(x)
    return np.exp(z * target.country_std_vec + target.country_mean_vec)


def saved_nn_annual_errors() -> dict[str, pd.DataFrame]:
    pred = pd.read_csv(UPDATED / "all_configs_updated_pure_nn_predictions.csv")
    pred = pred.groupby(["Config", "Country", "Year", "Month"], as_index=False).agg(
        rd_expenditure=("rd_expenditure", "first"),
        pred=("pred", "mean"),
    )
    return {cfg: annual_errors(pred[pred.Config == cfg].copy(), "pred") for cfg in CONFIG_ORDER}


def selected_epochs() -> dict[tuple[str, int], int]:
    history = pd.read_csv(UPDATED / "all_configs_updated_pure_nn_history.csv")
    best = history.loc[history.groupby(["Config", "seed"]).val_MAPE.idxmin()]
    return {(row.Config, int(row.seed)): int(row.epoch) for row in best.itertuples()}


def refit_nn_predictions(prep, cfg: str, fit_mask: np.ndarray, target: TargetTransform, epochs) -> np.ndarray:
    frame = prep.frame
    encoder = LabelEncoder()
    country_codes = encoder.fit_transform(frame.Country)
    raw = frame[prep.configs[cfg]].fillna(0).astype(float).to_numpy()
    scaler = StandardScaler().fit(raw[fit_mask])
    x = np.column_stack([scaler.transform(raw), prep.months.to_numpy()])

    x_fit = torch.FloatTensor(x[fit_mask])
    c_fit = torch.LongTensor(country_codes[fit_mask])
    y_fit = torch.FloatTensor(target.ystd[fit_mask].reshape(-1, 1))
    x_test = torch.FloatTensor(x[prep.masks["test"]])
    c_test = torch.LongTensor(country_codes[prep.masks["test"]])
    criterion = nn.SmoothL1Loss(beta=0.5)

    members = []
    for seed in range(ENSEMBLE_SIZE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        network = WideDeepNN(x.shape[1], len(encoder.classes_))
        optimizer = optim.AdamW(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(epochs[(cfg, seed)]):
            network.train()
            permutation = torch.randperm(len(x_fit))
            for start in range(0, len(x_fit), 64):
                idx = permutation[start : start + 64]
                optimizer.zero_grad()
                loss = criterion(network(x_fit[idx], c_fit[idx]), y_fit[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                optimizer.step()
        network.eval()
        with torch.no_grad():
            z_test = network(x_test, c_test).numpy().ravel()
        test_mask = prep.masks["test"]
        members.append(
            np.exp(
                z_test * target.country_std_vec[test_mask]
                + target.country_mean_vec[test_mask]
            )
        )
        print(
            f"matched refit {cfg}: member {seed + 1}/{ENSEMBLE_SIZE}, "
            f"epochs={epochs[(cfg, seed)]}",
            flush=True,
        )
    return np.column_stack(members).mean(axis=1)


def build_audit(
    nn_annual: dict[str, pd.DataFrame],
    ols_annual: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for cfg in CONFIG_ORDER:
        for model, ann in [("Neural network", nn_annual[cfg]), ("OLS", ols_annual[cfg])]:
            rows.append(
                {
                    "Config": cfg,
                    "Model": model,
                    "n": len(ann),
                    "mean_APE": ann.APE.mean(),
                    "median_APE": ann.APE.median(),
                    "iqr_APE": ann.APE.quantile(0.75) - ann.APE.quantile(0.25),
                    "p90_APE": ann.APE.quantile(0.90),
                    "max_APE": ann.APE.max(),
                    "RMSE": np.sqrt(np.mean((ann.actual - ann.pred) ** 2)),
                }
            )
    return pd.DataFrame(rows)


def style_boxplot(boxplot, color: str) -> None:
    for box in boxplot["boxes"]:
        box.set(facecolor=color, edgecolor=EDGE_COLOR, alpha=FILL_ALPHA, linewidth=0.8)
    for element in ["whiskers", "caps"]:
        for artist in boxplot[element]:
            artist.set(color=EDGE_COLOR, linewidth=0.8)


def overlay_points(ax, values: list[np.ndarray], centers: np.ndarray, color: str, rng) -> int:
    clipped_count = 0
    for center, group in zip(centers, values):
        group = np.asarray(group, dtype=float)
        jitter = rng.uniform(-0.075, 0.075, size=len(group))
        inside = group <= 100
        ax.scatter(
            center + jitter[inside],
            group[inside],
            s=11,
            color=color,
            edgecolors="white",
            linewidths=0.25,
            alpha=0.48,
            zorder=3,
        )
        above = ~inside
        clipped_count += int(above.sum())
        if above.any():
            ax.scatter(
                center + jitter[above],
                np.full(above.sum(), 98.0),
                marker="^",
                s=24,
                color=color,
                edgecolors=EDGE_COLOR,
                linewidths=0.45,
                alpha=0.92,
                zorder=4,
            )
    return clipped_count


def render_figure(
    audit: pd.DataFrame,
    nn_apes: list[np.ndarray],
    ols_apes: list[np.ndarray],
    outfile: Path,
    show_points: bool,
    matched_refit: bool,
) -> None:
    rmse = audit.pivot(index="Config", columns="Model", values="RMSE").loc[CONFIG_ORDER]
    pos = np.arange(len(CONFIG_ORDER))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.5),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    b1 = axes[0].boxplot(
        nn_apes,
        positions=pos - 0.18,
        widths=0.32,
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
    b2 = axes[0].boxplot(
        ols_apes,
        positions=pos + 0.18,
        widths=0.32,
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
    style_boxplot(b1, NN_COLOR)
    style_boxplot(b2, OLS_COLOR)

    clipped = 0
    if show_points:
        rng = np.random.default_rng(20260902)
        clipped += overlay_points(axes[0], nn_apes, pos - 0.18, NN_COLOR, rng)
        clipped += overlay_points(axes[0], ols_apes, pos + 0.18, OLS_COLOR, rng)

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
    if show_points:
        axes[0].annotate(
            "diamond = mean",
            xy=(0.98, 0.885),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
        )
        if clipped:
            axes[0].annotate(
                f"triangle at 98% = error >100% ({clipped})",
                xy=(0.98, 0.84),
                xycoords="axes fraction",
                ha="right",
                fontsize=8,
            )
    else:
        axes[0].annotate(
            "diamond = mean",
            xy=(0.98, 0.93),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
        )

    width = 0.34
    axes[1].bar(
        pos - width / 2,
        rmse["Neural network"],
        width,
        color=NN_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        alpha=FILL_ALPHA,
    )
    axes[1].bar(
        pos + width / 2,
        rmse["OLS"],
        width,
        color=OLS_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        alpha=FILL_ALPHA,
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
    OUT.mkdir(parents=True, exist_ok=True)
    DECK_ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    prep = prepare_data()
    train_mask = prep.masks["train"]
    refit_mask = prep.masks["train"] | prep.masks["val"]
    test_mask = prep.masks["test"]
    test_base = prep.frame[test_mask][["Country", "Year", "Month", "rd_expenditure"]].copy()

    # Candidate 1: saved NN, OLS trained on exactly the same training split.
    saved_nn = saved_nn_annual_errors()
    train_target = target_transform(prep.frame, train_mask)
    train_only_ols: dict[str, pd.DataFrame] = {}
    for cfg in CONFIG_ORDER:
        frame = test_base.copy()
        frame["pred"] = ols_predictions(prep, cfg, train_mask, train_target)[test_mask]
        train_only_ols[cfg] = annual_errors(frame, "pred")
    train_audit = build_audit(saved_nn, train_only_ols)
    train_audit.to_csv(OUT / "c1_stepA_dist_train_only_ols_audit.csv", index=False)
    render_figure(
        train_audit,
        [saved_nn[cfg].APE.to_numpy() for cfg in CONFIG_ORDER],
        [train_only_ols[cfg].APE.to_numpy() for cfg in CONFIG_ORDER],
        TRAIN_ONLY_FIG,
        show_points=False,
        matched_refit=False,
    )
    print(TRAIN_ONLY_FIG, flush=True)

    # Candidate 2: validation-selected duration followed by matched final refits.
    refit_target = target_transform(prep.frame, refit_mask)
    epochs = selected_epochs()
    refit_nn: dict[str, pd.DataFrame] = {}
    refit_ols: dict[str, pd.DataFrame] = {}
    refit_prediction_parts = []
    for cfg in CONFIG_ORDER:
        nn_pred = refit_nn_predictions(prep, cfg, refit_mask, refit_target, epochs)
        nn_frame = test_base.copy()
        nn_frame["pred"] = nn_pred
        nn_frame["Config"] = cfg
        refit_prediction_parts.append(nn_frame)
        refit_nn[cfg] = annual_errors(nn_frame, "pred")

        ols_frame = test_base.copy()
        ols_frame["pred"] = ols_predictions(prep, cfg, refit_mask, refit_target)[test_mask]
        refit_ols[cfg] = annual_errors(ols_frame, "pred")

    pd.concat(refit_prediction_parts, ignore_index=True).to_csv(
        OUT / "c1_stepA_matched_refit_nn_predictions.csv",
        index=False,
    )
    refit_audit = build_audit(refit_nn, refit_ols)
    refit_audit.to_csv(OUT / "c1_stepA_dist_matched_refit_points_audit.csv", index=False)
    render_figure(
        refit_audit,
        [refit_nn[cfg].APE.to_numpy() for cfg in CONFIG_ORDER],
        [refit_ols[cfg].APE.to_numpy() for cfg in CONFIG_ORDER],
        MATCHED_REFIT_FIG,
        show_points=True,
        matched_refit=True,
    )
    print(MATCHED_REFIT_FIG, flush=True)
    print(OUT / "c1_stepA_dist_train_only_ols_audit.csv", flush=True)
    print(OUT / "c1_stepA_dist_matched_refit_points_audit.csv", flush=True)


def rerender_matched_candidate() -> None:
    """Redraw candidate 2 from its saved NN predictions without retraining."""
    prep = prepare_data()
    refit_mask = prep.masks["train"] | prep.masks["val"]
    test_mask = prep.masks["test"]
    test_base = prep.frame[test_mask][["Country", "Year", "Month", "rd_expenditure"]].copy()
    refit_target = target_transform(prep.frame, refit_mask)

    saved = pd.read_csv(OUT / "c1_stepA_matched_refit_nn_predictions.csv")
    refit_nn = {
        cfg: annual_errors(saved[saved.Config == cfg].copy(), "pred")
        for cfg in CONFIG_ORDER
    }
    refit_ols: dict[str, pd.DataFrame] = {}
    for cfg in CONFIG_ORDER:
        ols_frame = test_base.copy()
        ols_frame["pred"] = ols_predictions(prep, cfg, refit_mask, refit_target)[test_mask]
        refit_ols[cfg] = annual_errors(ols_frame, "pred")
    audit = build_audit(refit_nn, refit_ols)
    audit.to_csv(OUT / "c1_stepA_dist_matched_refit_points_audit.csv", index=False)
    render_figure(
        audit,
        [refit_nn[cfg].APE.to_numpy() for cfg in CONFIG_ORDER],
        [refit_ols[cfg].APE.to_numpy() for cfg in CONFIG_ORDER],
        MATCHED_REFIT_FIG,
        show_points=True,
        matched_refit=True,
    )
    print(MATCHED_REFIT_FIG, flush=True)


def render_matched_candidate_clean() -> None:
    """Draw the matched-refit results in the original figure style."""
    prep = prepare_data()
    refit_mask = prep.masks["train"] | prep.masks["val"]
    test_mask = prep.masks["test"]
    test_base = prep.frame[test_mask][["Country", "Year", "Month", "rd_expenditure"]].copy()
    refit_target = target_transform(prep.frame, refit_mask)

    saved = pd.read_csv(OUT / "c1_stepA_matched_refit_nn_predictions.csv")
    refit_nn = {
        cfg: annual_errors(saved[saved.Config == cfg].copy(), "pred")
        for cfg in CONFIG_ORDER
    }
    refit_ols: dict[str, pd.DataFrame] = {}
    for cfg in CONFIG_ORDER:
        ols_frame = test_base.copy()
        ols_frame["pred"] = ols_predictions(prep, cfg, refit_mask, refit_target)[test_mask]
        refit_ols[cfg] = annual_errors(ols_frame, "pred")
    audit = build_audit(refit_nn, refit_ols)
    render_figure(
        audit,
        [refit_nn[cfg].APE.to_numpy() for cfg in CONFIG_ORDER],
        [refit_ols[cfg].APE.to_numpy() for cfg in CONFIG_ORDER],
        MATCHED_REFIT_CLEAN_FIG,
        show_points=False,
        matched_refit=True,
    )
    print(MATCHED_REFIT_CLEAN_FIG, flush=True)


if __name__ == "__main__":
    if "--render-matched-clean" in sys.argv:
        render_matched_candidate_clean()
    elif "--rerender-matched" in sys.argv:
        rerender_matched_candidate()
    else:
        main()
