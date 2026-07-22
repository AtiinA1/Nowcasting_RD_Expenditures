"""Quarterly diagnostics for the corrected raw-scale Step B elasticities."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from scipy import stats


CODE = Path(__file__).resolve().parents[2]
SOURCE = CODE / "additional_analysis" / "robustness_overfit" / "05_quarterly_stepb_values.py"
OUT = CODE / "additional_analysis" / "stepb_raw_scale_elasticity_refresh" / "out"


def load_module():
    spec = importlib.util.spec_from_file_location("quarterly_stepb", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def employment_with_p_values(quarterly: pd.DataFrame) -> pd.DataFrame:
    emp = pd.read_csv(CODE / "data" / "datausa.io" / "Monthly Employment.csv")
    emp["date"] = pd.to_datetime(emp.Date)
    emp["Year"] = emp.date.dt.year
    emp["Quarter"] = emp.date.dt.quarter
    emp_q = emp.groupby(["Year", "Quarter"], as_index=False).agg(emp=("NSA Employees", "mean"))
    merged = quarterly.merge(emp_q, on=["Year", "Quarter"], how="inner").sort_values(["Year", "Quarter"])
    rows = []
    for col in ["NN", "Ridge_signed", "Ridge_positive", "Mosley", "Sax"]:
        rd_growth = merged[col].pct_change().dropna().values
        emp_growth = merged.emp.pct_change().dropna().values
        corr, p_value = stats.pearsonr(rd_growth, emp_growth)
        rows.append(
            {
                "Series": col,
                "quarterly_employment_growth_corr": float(corr),
                "p_value": float(p_value),
                "N_growth_observations": int(len(rd_growth)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    module = load_module()
    monthly = pd.read_csv(OUT / "corrected_temporal_monthly_estimates.csv")
    linear = pd.read_csv(OUT / "corrected_linear_stepb_monthly_estimates.csv")
    ridge = (
        linear[(linear.Model == "Ridge") & linear.Variant.isin(["signed", "positive_part"])]
        .pivot_table(index=["Year", "Month"], columns="Variant", values="estimate", aggfunc="mean")
        .reset_index()
        .rename(columns={"signed": "Ridge_signed", "positive_part": "Ridge_positive"})
    )
    monthly = monthly.merge(ridge, on=["Year", "Month"], how="left")
    monthly["date"] = pd.to_datetime(dict(year=monthly.Year, month=monthly.Month, day=1))
    monthly["Quarter"] = monthly.date.dt.quarter
    monthly["quarter_date"] = pd.PeriodIndex(
        year=monthly.Year, quarter=monthly.Quarter, freq="Q"
    ).to_timestamp()

    quarterly = module.quarterly_from_monthly(monthly)
    checks = module.annual_sum_check(monthly, quarterly)
    smoothness = module.summary_stats(monthly, quarterly)
    agreement = module.agreement_table(quarterly)
    employment = employment_with_p_values(quarterly)

    monthly.to_csv(OUT / "corrected_monthly_series_used_for_quarterly.csv", index=False)
    quarterly.to_csv(OUT / "corrected_quarterly_all_methods.csv", index=False)
    checks.to_csv(OUT / "corrected_quarterly_annual_sum_checks.csv", index=False)
    smoothness.to_csv(OUT / "corrected_quarterly_smoothness_summary.csv", index=False)
    agreement.to_csv(OUT / "corrected_quarterly_agreement_all_methods.csv", index=False)
    employment.to_csv(OUT / "corrected_quarterly_employment_correlations.csv", index=False)

    print("=== corrected quarterly agreement ===")
    print(agreement.to_string(index=False))
    print("\n=== corrected monthly vs quarterly variation ===")
    print(
        smoothness[
            ["Series", "monthly_growth_sd", "quarterly_growth_sd", "monthly_abs_growth_mean", "quarterly_abs_growth_mean"]
        ].to_string(index=False)
    )
    print("\n=== corrected quarterly employment correlations ===")
    print(employment.to_string(index=False))
    print("\n=== aggregation checks ===")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
