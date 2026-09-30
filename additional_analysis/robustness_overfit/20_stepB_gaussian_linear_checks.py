"""Apply Gaussian Step B relevance to matched-refit OLS and Ridge models."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "16_regenerate_stepB_matched_refit.py"
SPEC = importlib.util.spec_from_file_location("stepb_matched_refit", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT = HERE / "out" / "stepB_matched_refit" / "gaussian_relevance"
NN_SERIES = OUT / "gaussian_monthly_estimates.csv"
SIGMA = 0.10
N_PAIRS = 50
RNG_SEED = 20260903
NORMAL_ABS_MEAN = np.sqrt(2.0 / np.pi)


def level_predictions(model, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.exp(model.predict(x) * std + mean)


def gaussian_relevance(
    model,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_gt: int,
) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.standard_normal(N_PAIRS)
    epsilon = SIGMA * np.concatenate([draws, -draws])
    base_level = level_predictions(model, x, mean, std)
    denominator = np.where(np.abs(base_level) > 1e-12, base_level, np.nan)
    relevance = np.zeros((len(x), n_gt), dtype=float)

    mean_batch = np.tile(mean, len(epsilon))
    std_batch = np.tile(std, len(epsilon))
    for j in range(n_gt):
        perturbed_x = np.tile(x, (len(epsilon), 1))
        perturbed_x[:, j] += np.repeat(epsilon, len(x))
        perturbed_level = level_predictions(
            model, perturbed_x, mean_batch, std_batch
        ).reshape(len(epsilon), len(x))
        relative_change = np.abs(
            (perturbed_level - base_level[None, :]) / denominator[None, :]
        )
        relevance[:, j] = relative_change.mean(axis=0) / (
            SIGMA * NORMAL_ABS_MEAN
        )
    return relevance


def correlations(frame: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    quarterly = frame.copy()
    quarterly["Quarter"] = ((quarterly.Month - 1) // 3) + 1
    quarterly = quarterly.groupby(["Year", "Quarter"], as_index=False)[
        [left, right]
    ].sum()
    level_r, level_p = stats.pearsonr(frame[left], frame[right])
    growth_r, growth_p = stats.pearsonr(
        frame[left].pct_change().iloc[1:], frame[right].pct_change().iloc[1:]
    )
    quarter_r, quarter_p = stats.pearsonr(
        quarterly[left].pct_change().iloc[1:],
        quarterly[right].pct_change().iloc[1:],
    )
    return {
        "level_correlation": level_r,
        "level_p_value": level_p,
        "monthly_growth_correlation": growth_r,
        "monthly_growth_p_value": growth_p,
        "quarterly_growth_correlation": quarter_r,
        "quarterly_growth_p_value": quarter_p,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prep = base.prepare_data()
    fit_mask = prep.masks["train"] | prep.masks["val"]
    target = base.target_transform(prep.frame, fit_mask)
    columns = prep.configs["AGT"]
    topics = base.topics_from_columns(columns)
    raw = prep.frame[columns].fillna(0).astype(float).to_numpy()

    ridge, x, _, ridge_alpha, ridge_tuning = base.fit_matched_ridge(
        prep, raw, fit_mask, target
    )
    ridge_tuning.to_csv(OUT / "gaussian_linear_check_ridge_tuning.csv", index=False)
    ols = LinearRegression(fit_intercept=True).fit(
        x[fit_mask], target.ystd[fit_mask]
    )

    us_fit = fit_mask & prep.frame.Country.eq("US").to_numpy()
    us_frame = prep.frame.loc[us_fit].reset_index(drop=True)
    model_weights = {}
    weight_rows = []
    for name, model in (("OLS", ols), ("Ridge", ridge)):
        relevance = gaussian_relevance(
            model,
            x[us_fit],
            target.country_mean_vec[us_fit],
            target.country_std_vec[us_fit],
            len(columns),
        )
        weights = base.topic_weights(
            us_frame,
            columns,
            topics,
            relevance,
            np.ones(us_fit.sum(), dtype=bool),
        )
        model_weights[name] = weights
        weight_rows.extend(
            {
                "Model": name,
                "topic": topic,
                "gaussian_relevance": value,
                "ridge_alpha": ridge_alpha if name == "Ridge" else np.nan,
            }
            for topic, value in weights.items()
        )

    gt, _, annual_us = base.load_stepb_inputs(prep)
    nn = pd.read_csv(NN_SERIES)
    nn = nn[np.isclose(nn.sigma, SIGMA)][["Year", "Month", "NN"]].copy()
    monthly = nn
    for name, weights in model_weights.items():
        candidate = base.disaggregate_us(weights, topics, gt, annual_us).rename(
            columns={"estimate": name}
        )
        monthly = monthly.merge(candidate, on=["Year", "Month"], how="inner")
    monthly = monthly.sort_values(["Year", "Month"]).reset_index(drop=True)

    comparisons = []
    for left, right in (("NN", "OLS"), ("NN", "Ridge"), ("OLS", "Ridge")):
        comparisons.append(
            {
                "comparison": f"{left} vs {right}",
                "sigma": SIGMA,
                "ridge_alpha": ridge_alpha,
                **correlations(monthly, left, right),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)

    pd.DataFrame(weight_rows).to_csv(
        OUT / "gaussian_linear_model_topic_relevance.csv", index=False
    )
    monthly.to_csv(OUT / "gaussian_linear_model_monthly_estimates.csv", index=False)
    comparison_frame.to_csv(OUT / "gaussian_linear_model_checks.csv", index=False)

    print("\n=== Gaussian linear-model checks ===")
    print(comparison_frame.to_string(index=False))
    print("\n=== Annual adding-up maximum absolute errors ===")
    sums = monthly.groupby("Year")[["NN", "OLS", "Ridge"]].sum()
    expected = annual_us.reindex(sums.index)
    print(sums.sub(expected, axis=0).abs().max().to_string())


if __name__ == "__main__":
    main()
