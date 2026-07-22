"""Test whether Step B shares add employment-related timing beyond uniform allocation.

The test removes annual levels by converting every series into monthly shares of
its own annual total. Uniform allocation is then exactly 1/12. Inference uses
years as blocks, preserving all twelve observations within a sampled year.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CODE = Path(__file__).resolve().parents[2]
STEPB = CODE / "additional_analysis" / "robustness_overfit" / "out" / "updated_stepB_agt"
OUT = CODE / "additional_analysis" / "pre_raw_stepb_method_audit" / "out"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["NN", "Mosley", "Sax"]
UNIFORM = 1.0 / 12.0
BOOTSTRAPS = 20_000
SEED = 20260714


def load_shares() -> pd.DataFrame:
    estimates = pd.read_csv(STEPB / "combined_estimates_temporal_level_updated_agt.csv")
    estimates["date"] = pd.to_datetime(estimates.date)
    employment = pd.read_csv(CODE / "data" / "datausa.io" / "Monthly Employment.csv")
    employment["date"] = pd.to_datetime(employment.Date)
    employment = employment[["date", "NSA Employees"]].rename(columns={"NSA Employees": "Employment"})
    frame = estimates.merge(employment, on="date", how="inner").sort_values("date").reset_index(drop=True)
    frame["Year"] = frame.date.dt.year
    frame["Month"] = frame.date.dt.month
    full_years = frame.groupby("Year").Month.nunique()
    frame = frame[frame.Year.isin(full_years[full_years.eq(12)].index)].copy()
    for column in METHODS + ["Employment"]:
        frame[f"{column}_share"] = frame[column] / frame.groupby("Year")[column].transform("sum")
        frame[f"{column}_dev"] = frame[f"{column}_share"] - UNIFORM
        month_mean = frame.groupby("Month")[f"{column}_dev"].transform("mean")
        frame[f"{column}_season_resid"] = frame[f"{column}_dev"] - month_mean
    frame.to_csv(OUT / "incremental_timing_shares.csv", index=False)
    return frame


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-14 and np.std(y) > 1e-14 else np.nan


def bootstrap_by_year(frame: pd.DataFrame, method: str) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    years = np.array(sorted(frame.Year.unique()))
    blocks = {year: frame[frame.Year.eq(year)] for year in years}
    raw_corr = np.empty(BOOTSTRAPS)
    residual_corr = np.empty(BOOTSTRAPS)
    delta_rmse = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sampled = rng.choice(years, size=len(years), replace=True)
        sample = pd.concat([blocks[year] for year in sampled], ignore_index=True)
        raw_corr[draw] = safe_corr(sample[f"{method}_dev"], sample.Employment_dev)
        residual_corr[draw] = safe_corr(
            sample[f"{method}_season_resid"], sample.Employment_season_resid
        )
        method_rmse = np.sqrt(np.mean((sample[f"{method}_share"] - sample.Employment_share) ** 2))
        uniform_rmse = np.sqrt(np.mean((UNIFORM - sample.Employment_share) ** 2))
        delta_rmse[draw] = method_rmse - uniform_rmse
    return {
        "raw_corr_ci_low": np.nanpercentile(raw_corr, 2.5),
        "raw_corr_ci_high": np.nanpercentile(raw_corr, 97.5),
        "season_resid_corr_ci_low": np.nanpercentile(residual_corr, 2.5),
        "season_resid_corr_ci_high": np.nanpercentile(residual_corr, 97.5),
        "delta_rmse_ci_low": np.nanpercentile(delta_rmse, 2.5),
        "delta_rmse_ci_high": np.nanpercentile(delta_rmse, 97.5),
    }


def leave_one_year_out(frame: pd.DataFrame, method: str, positive_slope: bool) -> tuple[float, list[float]]:
    predictions = []
    actuals = []
    slopes = []
    for year in sorted(frame.Year.unique()):
        train = frame[frame.Year.ne(year)]
        test = frame[frame.Year.eq(year)]
        x = train[f"{method}_dev"].to_numpy()
        y = train.Employment_dev.to_numpy()
        denominator = float(x @ x)
        slope = float((x @ y) / denominator) if denominator > 1e-14 else 0.0
        if positive_slope:
            slope = max(0.0, slope)
        slopes.append(slope)
        predictions.extend(UNIFORM + slope * test[f"{method}_dev"].to_numpy())
        actuals.extend(test.Employment_share.to_numpy())
    rmse = float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(actuals)) ** 2)))
    return rmse, slopes


def main():
    frame = load_shares()
    uniform_month_error = UNIFORM - frame.Employment_share
    uniform_rmse = float(np.sqrt(np.mean(uniform_month_error**2)))
    uniform_mae = float(np.mean(np.abs(uniform_month_error)))
    uniform_year_mse = frame.assign(error2=uniform_month_error**2).groupby("Year").error2.mean()
    rows = []
    for method in METHODS:
        raw_corr, raw_p = stats.pearsonr(frame[f"{method}_dev"], frame.Employment_dev)
        rank_corr, rank_p = stats.spearmanr(frame[f"{method}_dev"], frame.Employment_dev)
        residual_corr, residual_p = stats.pearsonr(
            frame[f"{method}_season_resid"], frame.Employment_season_resid
        )
        error = frame[f"{method}_share"] - frame.Employment_share
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        method_year_mse = frame.assign(error2=error**2).groupby("Year").error2.mean()
        difference = method_year_mse - uniform_year_mse
        paired_t = stats.ttest_1samp(difference, popmean=0.0)
        wilcoxon = stats.wilcoxon(difference)
        loyo_positive_rmse, positive_slopes = leave_one_year_out(frame, method, positive_slope=True)
        loyo_free_rmse, free_slopes = leave_one_year_out(frame, method, positive_slope=False)
        row = {
            "Method": method,
            "n_years": frame.Year.nunique(),
            "n_months": len(frame),
            "share_corr": raw_corr,
            "share_corr_p_naive": raw_p,
            "share_spearman": rank_corr,
            "share_spearman_p_naive": rank_p,
            "season_residual_corr": residual_corr,
            "season_residual_corr_p_naive": residual_p,
            "share_RMSE": rmse,
            "uniform_RMSE": uniform_rmse,
            "delta_RMSE_vs_uniform": rmse - uniform_rmse,
            "share_MAE": mae,
            "uniform_MAE": uniform_mae,
            "paired_year_MSE_t_p": paired_t.pvalue,
            "paired_year_MSE_wilcoxon_p": wilcoxon.pvalue,
            "years_method_beats_uniform_MSE": int((difference < 0).sum()),
            "LOYO_positive_slope_RMSE": loyo_positive_rmse,
            "LOYO_positive_slope_median": float(np.median(positive_slopes)),
            "LOYO_unrestricted_RMSE": loyo_free_rmse,
            "LOYO_unrestricted_slope_median": float(np.median(free_slopes)),
            **bootstrap_by_year(frame, method),
        }
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "incremental_timing_test_results.csv", index=False)

    compact = results[
        [
            "Method", "share_corr", "raw_corr_ci_low", "raw_corr_ci_high",
            "season_residual_corr", "season_resid_corr_ci_low", "season_resid_corr_ci_high",
            "share_RMSE", "uniform_RMSE", "delta_RMSE_vs_uniform",
            "delta_rmse_ci_low", "delta_rmse_ci_high", "years_method_beats_uniform_MSE",
            "LOYO_positive_slope_RMSE", "LOYO_positive_slope_median",
        ]
    ]
    print(f"Sample: {frame.Year.min()}-{frame.Year.max()}, {frame.Year.nunique()} complete years, N={len(frame)}")
    print("\nIncremental within-year timing test")
    print(compact.round(6).to_string(index=False))
    print("\nPositive delta RMSE means uniform allocation is more accurate for employment shares.")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
