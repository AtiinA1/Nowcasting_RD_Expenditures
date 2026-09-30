"""Gaussian-perturbation robustness check for the matched-refit Step B model."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "out" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "16_regenerate_stepB_matched_refit.py"
SPEC = importlib.util.spec_from_file_location("stepb_matched_refit", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT = HERE / "out" / "stepB_matched_refit" / "gaussian_relevance"
CODE = Path(__file__).resolve().parents[2]
ASSETS = Path(
    os.environ.get(
        "NOWCASTING_DECK_ASSETS",
        CODE / "paper" / "figures" / "UpdatedResults",
    )
)
SIGMAS = (0.05, 0.10, 0.25)
N_PAIRS = 50
RNG_SEED = 20260903
NORMAL_ABS_MEAN = np.sqrt(2.0 / np.pi)


def predict_levels(
    model,
    x: np.ndarray,
    country_codes: np.ndarray,
    country_means: np.ndarray,
    country_stds: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        z = model(
            torch.FloatTensor(x), torch.LongTensor(country_codes)
        ).numpy().ravel()
    return np.exp(z * country_stds + country_means)


def gaussian_relevance(
    models,
    x: np.ndarray,
    country_codes: np.ndarray,
    target,
    n_gt: int,
    sigmas: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Expected absolute relative response per standardized-input unit."""
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.standard_normal(N_PAIRS)
    antithetic = np.concatenate([draws, -draws])
    results = {sigma: np.zeros((len(x), n_gt), dtype=float) for sigma in sigmas}

    for member_index, model in enumerate(models):
        model.eval()
        base_level = predict_levels(
            model,
            x,
            country_codes,
            target.country_mean_vec,
            target.country_std_vec,
        )
        denominator = np.where(np.abs(base_level) > 1e-12, base_level, np.nan)

        for sigma in sigmas:
            epsilon = sigma * antithetic
            country_batch = np.tile(country_codes, len(epsilon))
            mean_batch = np.tile(target.country_mean_vec, len(epsilon))
            std_batch = np.tile(target.country_std_vec, len(epsilon))
            for j in range(n_gt):
                perturbed_x = np.tile(x, (len(epsilon), 1))
                perturbation = np.repeat(epsilon, len(x))
                perturbed_x[:, j] += perturbation
                perturbed_level = predict_levels(
                    model,
                    perturbed_x,
                    country_batch,
                    mean_batch,
                    std_batch,
                ).reshape(len(epsilon), len(x))
                relative_change = np.abs(
                    (perturbed_level - base_level[None, :]) / denominator[None, :]
                )
                results[sigma][:, j] += (
                    relative_change.mean(axis=0) / (sigma * NORMAL_ABS_MEAN)
                )
        print(
            f"Gaussian relevance: member {member_index + 1}/{len(models)}",
            flush=True,
        )

    return {sigma: values / len(models) for sigma, values in results.items()}


def correlations(monthly: pd.DataFrame, sigma: float) -> list[dict[str, float]]:
    quarterly = monthly.copy()
    quarterly["Quarter"] = ((quarterly.Month - 1) // 3) + 1
    quarterly = quarterly.groupby(["Year", "Quarter"], as_index=False)[
        ["NN", "Mosley", "Sax"]
    ].sum()
    rows = []
    for benchmark in ("Mosley", "Sax"):
        level_r, level_p = stats.pearsonr(monthly.NN, monthly[benchmark])
        monthly_r, monthly_p = stats.pearsonr(
            monthly.NN.pct_change().iloc[1:], monthly[benchmark].pct_change().iloc[1:]
        )
        quarterly_r, quarterly_p = stats.pearsonr(
            quarterly.NN.pct_change().iloc[1:],
            quarterly[benchmark].pct_change().iloc[1:],
        )
        rows.append(
            {
                "sigma": sigma,
                "benchmark": benchmark,
                "level_correlation": level_r,
                "level_p_value": level_p,
                "monthly_growth_correlation": monthly_r,
                "monthly_growth_p_value": monthly_p,
                "quarterly_growth_correlation": quarterly_r,
                "quarterly_growth_p_value": quarterly_p,
                "estimate_min": monthly.NN.min(),
                "estimate_max": monthly.NN.max(),
                "monthly_growth_sd": monthly.NN.pct_change().std(),
            }
        )
    return rows


def render_panels(monthly: pd.DataFrame, sigma: float) -> None:
    frame = monthly.copy()
    frame["date"] = pd.to_datetime(dict(year=frame.Year, month=frame.Month, day=1))
    lower = frame[["NN", "Sax", "Mosley"]].min(axis=1)
    upper = frame[["NN", "Sax", "Mosley"]].max(axis=1)
    sigma_label = f"s{int(round(sigma * 100)):03d}"
    panels = [
        (
            "NN",
            "#2c6fbb",
            "NN-driven model-implied sensitivity estimate",
            "NN relevance",
            f"c1_disagg_nn_matched_refit_gaussian_{sigma_label}.png",
        ),
        (
            "Mosley",
            "green",
            "Sparse temporal disaggregation (Mosley)",
            "Sparse (Mosley)",
            f"c1_disagg_mosley_matched_refit_gaussian_{sigma_label}.png",
        ),
        (
            "Sax",
            "purple",
            "Chow-Lin temporal disaggregation (Sax)",
            "Chow-Lin (Sax)",
            f"c1_disagg_sax_matched_refit_gaussian_{sigma_label}.png",
        ),
    ]
    for column, color, title, label, filename in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(
            frame.date,
            lower,
            upper,
            color="gray",
            alpha=0.30,
            label="Range across methods",
        )
        ax.plot(frame.date, frame[column], color=color, marker="o", ms=3, lw=1, label=label)
        ax.set_ylabel("Monthly R&D (USD bn)")
        ax.set_title(title)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        output = ASSETS / filename
        fig.savefig(output, dpi=220)
        plt.close(fig)
        print(output, flush=True)


def render_sigma_stability(series: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {0.05: "#7aa6d8", 0.10: "#2c6fbb", 0.25: "#173f75"}
    for sigma in SIGMAS:
        frame = series[series.sigma.eq(sigma)].copy()
        frame["date"] = pd.to_datetime(dict(year=frame.Year, month=frame.Month, day=1))
        ax.plot(
            frame.date,
            frame.NN,
            color=colors[sigma],
            lw=1.2,
            label=rf"$\sigma={sigma:.2f}$ SD",
        )
    ax.set_ylabel("Monthly R&D (USD bn)")
    ax.set_title("Gaussian-perturbation stability")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = ASSETS / "c1_disagg_nn_matched_refit_gaussian_stability.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    prep = base.prepare_data()
    fit_mask = prep.masks["train"] | prep.masks["val"]
    target = base.target_transform(prep.frame, fit_mask)
    models, raw, scaler, x, country_codes, columns, _ = base.train_matched_agt(
        prep, fit_mask, target
    )
    del raw, scaler

    us_fit = fit_mask & prep.frame.Country.eq("US").to_numpy()
    us_target = base.TargetTransform(
        ystd=target.ystd[us_fit],
        country_mean_vec=target.country_mean_vec[us_fit],
        country_std_vec=target.country_std_vec[us_fit],
    )
    relevance = gaussian_relevance(
        models,
        x[us_fit],
        country_codes[us_fit],
        us_target,
        len(columns),
        SIGMAS,
    )

    topics = base.topics_from_columns(columns)
    us_frame = prep.frame.loc[us_fit].reset_index(drop=True)
    all_weights = []
    all_series = []
    all_metrics = []
    gt, references, annual_us = base.load_stepb_inputs(prep)

    for sigma in SIGMAS:
        weights = base.topic_weights(
            us_frame,
            columns,
            topics,
            relevance[sigma],
            np.ones(us_fit.sum(), dtype=bool),
        )
        all_weights.extend(
            {"sigma": sigma, "topic": topic, "relevance": value}
            for topic, value in weights.items()
        )
        monthly = base.disaggregate_us(weights, topics, gt, annual_us).rename(
            columns={"estimate": "NN"}
        )
        monthly = (
            monthly.merge(references, on=["Year", "Month"], how="inner")
            .dropna()
            .sort_values(["Year", "Month"])
            .reset_index(drop=True)
        )
        all_metrics.extend(correlations(monthly, sigma))
        monthly.insert(0, "sigma", sigma)
        all_series.append(monthly)
        if np.isclose(sigma, 0.10):
            render_panels(monthly, sigma)

    weights_frame = pd.DataFrame(all_weights)
    series_frame = pd.concat(all_series, ignore_index=True)
    metrics_frame = pd.DataFrame(all_metrics)
    weights_frame.to_csv(OUT / "gaussian_topic_relevance.csv", index=False)
    series_frame.to_csv(OUT / "gaussian_monthly_estimates.csv", index=False)
    metrics_frame.to_csv(OUT / "gaussian_agreement.csv", index=False)
    render_sigma_stability(series_frame)

    pivot = series_frame.pivot_table(
        index=["Year", "Month"], columns="sigma", values="NN"
    )
    stability_rows = []
    for left, right in ((0.05, 0.10), (0.10, 0.25), (0.05, 0.25)):
        stability_rows.append(
            {
                "sigma_left": left,
                "sigma_right": right,
                "level_correlation": stats.pearsonr(pivot[left], pivot[right])[0],
                "mean_absolute_difference": np.mean(np.abs(pivot[left] - pivot[right])),
                "max_absolute_difference": np.max(np.abs(pivot[left] - pivot[right])),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUT / "gaussian_sigma_stability.csv", index=False)

    print("\n=== Gaussian agreement ===")
    print(metrics_frame.to_string(index=False))
    print("\n=== Sigma stability ===")
    print(stability.to_string(index=False))
    print(f"\nOutputs saved under {OUT}")


if __name__ == "__main__":
    main()
