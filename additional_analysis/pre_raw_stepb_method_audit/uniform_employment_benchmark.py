"""Evaluate uniform monthly allocation against the current Step B estimators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CODE = Path(__file__).resolve().parents[2]
STEPB = CODE / "additional_analysis" / "robustness_overfit" / "out" / "updated_stepB_agt"
OUT = CODE / "additional_analysis" / "pre_raw_stepb_method_audit" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def lag_correlations(series: np.ndarray, employment: np.ndarray, lags=range(-12, 13)):
    series_growth = pd.Series(series).pct_change().to_numpy()[1:]
    employment_growth = pd.Series(employment).pct_change().to_numpy()[1:]
    rows = []
    for lag in lags:
        if lag < 0:
            first, second = series_growth[-lag:], employment_growth[:lag]
        elif lag > 0:
            first, second = series_growth[:-lag], employment_growth[lag:]
        else:
            first, second = series_growth, employment_growth
        mask = np.isfinite(first) & np.isfinite(second)
        correlation, p_value = stats.pearsonr(first[mask], second[mask])
        rows.append({"Lag": lag, "Correlation": correlation, "P_Value": p_value, "N": int(mask.sum())})
    return pd.DataFrame(rows)


def annual_total_from_monthly(frame: pd.DataFrame) -> pd.Series:
    # Every current Step B series is benchmarked to the same observed annual GERD.
    return frame.groupby("Year").NN.sum()


def main():
    estimates = pd.read_csv(STEPB / "combined_estimates_temporal_level_updated_agt.csv")
    estimates["date"] = pd.to_datetime(estimates.date)
    annual = annual_total_from_monthly(estimates)
    estimates["Uniform"] = estimates.Year.map(annual) / 12.0

    employment = pd.read_csv(CODE / "data" / "datausa.io" / "Monthly Employment.csv")
    employment["date"] = pd.to_datetime(employment.Date)
    employment = employment[["date", "NSA Employees"]].rename(columns={"NSA Employees": "Employment"})
    merged = estimates.merge(employment, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged.to_csv(OUT / "uniform_employment_merged_monthly.csv", index=False)

    all_lags = []
    for method in ["NN", "Mosley", "Sax", "Uniform"]:
        result = lag_correlations(merged[method].to_numpy(), merged.Employment.to_numpy())
        result.insert(0, "Method", method)
        all_lags.append(result)
    all_lags = pd.concat(all_lags, ignore_index=True)
    all_lags.to_csv(OUT / "uniform_employment_monthly_lags.csv", index=False)

    summary_rows = []
    for method, group in all_lags.groupby("Method"):
        peak = group.loc[group.Correlation.abs().idxmax()]
        lag_zero = group[group.Lag.eq(0)].iloc[0]
        summary_rows.append(
            {
                "Method": method,
                "Lag0_corr": lag_zero.Correlation,
                "Lag0_p": lag_zero.P_Value,
                "Peak_lag": int(peak.Lag),
                "Peak_corr": peak.Correlation,
                "Peak_p": peak.P_Value,
                "p_lt_0.01_lags": int((group.P_Value < 0.01).sum()),
                "p_lt_0.05_lags": int((group.P_Value < 0.05).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "uniform_employment_monthly_summary.csv", index=False)

    quarterly = merged.copy()
    quarterly["Quarter"] = quarterly.date.dt.quarter
    quarterly = quarterly.groupby([quarterly.date.dt.year.rename("Year"), "Quarter"], as_index=False).agg(
        NN=("NN", "sum"), Mosley=("Mosley", "sum"), Sax=("Sax", "sum"),
        Uniform=("Uniform", "sum"), Employment=("Employment", "mean")
    )
    quarterly_rows = []
    employment_growth = quarterly.Employment.pct_change()
    for method in ["NN", "Mosley", "Sax", "Uniform"]:
        method_growth = quarterly[method].pct_change()
        valid = method_growth.notna() & employment_growth.notna()
        correlation, p_value = stats.pearsonr(method_growth[valid], employment_growth[valid])
        quarterly_rows.append(
            {"Method": method, "Quarterly_growth_corr": correlation, "P_Value": p_value, "N": int(valid.sum())}
        )
    quarterly_summary = pd.DataFrame(quarterly_rows)
    quarterly_summary.to_csv(OUT / "uniform_employment_quarterly_summary.csv", index=False)

    # Directly isolate within-year timing information. Each method's R&D shares
    # are used to distribute the observed annual employment total. Uniform has
    # no share variation, so its share correlation is undefined, but its
    # recovery error remains a meaningful baseline.
    full_years = merged.groupby(merged.date.dt.year).date.count()
    full_years = full_years[full_years.eq(12)].index
    within = merged[merged.date.dt.year.isin(full_years)].copy()
    within["Year"] = within.date.dt.year
    within["Employment_total"] = within.groupby("Year").Employment.transform("sum")
    within["Employment_share"] = within.Employment / within.Employment_total
    within_rows = []
    for method in ["NN", "Mosley", "Sax", "Uniform"]:
        within[f"{method}_share"] = within[method] / within.groupby("Year")[method].transform("sum")
        recovered = within.Employment_total * within[f"{method}_share"]
        nrmse = np.sqrt(np.mean((recovered - within.Employment) ** 2)) / within.Employment.mean()
        share = within[f"{method}_share"].to_numpy()
        employment_share = within.Employment_share.to_numpy()
        share_corr = (
            np.corrcoef(share - 1 / 12, employment_share - 1 / 12)[0, 1]
            if np.std(share) > 1e-12 else np.nan
        )
        within_rows.append(
            {
                "Method": method,
                "Within_year_share_corr": share_corr,
                "Employment_recovery_nRMSE": nrmse,
                "n_months": len(within),
            }
        )
    within_summary = pd.DataFrame(within_rows)
    within_summary.to_csv(OUT / "uniform_employment_within_year_summary.csv", index=False)

    print(f"Monthly overlap: {merged.date.min():%Y-%m} to {merged.date.max():%Y-%m}, N={len(merged)}")
    print("\nMonthly employment-growth cross-correlation summary, lags -12 to +12")
    print(summary.round(4).to_string(index=False))
    print("\nUniform monthly lags with p < 0.05")
    print(all_lags[(all_lags.Method == "Uniform") & (all_lags.P_Value < 0.05)].round(4).to_string(index=False))
    print("\nContemporaneous quarterly-growth comparison")
    print(quarterly_summary.round(4).to_string(index=False))
    print("\nWithin-year timing and employment-recovery comparison")
    print(within_summary.round(4).to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
