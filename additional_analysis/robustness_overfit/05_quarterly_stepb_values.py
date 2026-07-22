"""Quarterly aggregation of Step B monthly R&D estimates.

The monthly Step B series is the finest-resolution output of the framework, but
R&D expenditure is a smooth investment flow. This script aggregates the monthly
estimates to quarters to check whether the monitoring signal is clearer at a
quarterly horizon before adding anything to the paper.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "out", "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(__file__))
from common import FIG, OUT, ROOT, SOURCE_OUT, TAB  # noqa: E402


QUARTER_OUT = os.path.join(OUT, "quarterly_stepb")
os.makedirs(QUARTER_OUT, exist_ok=True)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(stats.pearsonr(a[mask], b[mask])[0])


def pct_change(x: pd.Series) -> np.ndarray:
    v = x.astype(float).values
    return np.diff(v) / v[:-1]


def load_monthly_series() -> pd.DataFrame:
    main = pd.read_csv(os.path.join(SOURCE_OUT, "combined_estimates_temporal_level.csv"))
    main = main[["Year", "Month", "NN", "Sax", "Mosley"]].copy()

    alt_path = os.path.join(OUT, "stepb_model_agnostic", "stepb_alt_monthly_estimates.csv")
    if os.path.exists(alt_path):
        alt = pd.read_csv(alt_path)
        ridge = (
            alt[(alt.Model == "Ridge") & (alt.Variant.isin(["signed", "positive_part"]))]
            .pivot_table(index=["Year", "Month"], columns="Variant", values="estimate", aggfunc="mean")
            .reset_index()
            .rename(columns={"signed": "Ridge_signed", "positive_part": "Ridge_positive"})
        )
        main = main.merge(ridge, on=["Year", "Month"], how="left")
    main["date"] = pd.to_datetime(dict(year=main.Year, month=main.Month, day=1))
    main["Quarter"] = main["date"].dt.quarter
    main["quarter_date"] = pd.PeriodIndex(year=main.Year, quarter=main.Quarter, freq="Q").to_timestamp()
    return main


def quarterly_from_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in ["NN", "Ridge_signed", "Ridge_positive", "Mosley", "Sax"] if c in monthly.columns]
    q = (
        monthly.groupby(["Year", "Quarter", "quarter_date"], as_index=False)[value_cols]
        .sum(min_count=3)
        .sort_values(["Year", "Quarter"])
        .reset_index(drop=True)
    )
    q["quarter_label"] = q.Year.astype(str) + "Q" + q.Quarter.astype(str)
    return q


def annual_sum_check(monthly: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in ["NN", "Ridge_signed", "Ridge_positive", "Mosley", "Sax"] if c in monthly.columns]
    m = monthly.groupby("Year", as_index=False)[value_cols].sum()
    q = quarterly.groupby("Year", as_index=False)[value_cols].sum()
    rows = []
    for col in value_cols:
        diff = (m[col] - q[col]).abs()
        rows.append({"Series": col, "max_abs_monthly_vs_quarterly_annual_sum_diff": float(diff.max())})
    return pd.DataFrame(rows)


def summary_stats(monthly: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in ["NN", "Ridge_signed", "Ridge_positive", "Mosley", "Sax"] if c in quarterly.columns]
    rows = []
    for col in value_cols:
        mg = pct_change(monthly[col])
        qg = pct_change(quarterly[col])
        rows.append(
            {
                "Series": col,
                "monthly_level_min": float(monthly[col].min()),
                "monthly_level_max": float(monthly[col].max()),
                "quarterly_level_min": float(quarterly[col].min()),
                "quarterly_level_max": float(quarterly[col].max()),
                "monthly_growth_sd": float(np.nanstd(mg, ddof=1)),
                "quarterly_growth_sd": float(np.nanstd(qg, ddof=1)),
                "monthly_abs_growth_mean": float(np.nanmean(np.abs(mg))),
                "quarterly_abs_growth_mean": float(np.nanmean(np.abs(qg))),
            }
        )
    return pd.DataFrame(rows)


def agreement_table(quarterly: pd.DataFrame) -> pd.DataFrame:
    series = [c for c in ["Ridge_signed", "Ridge_positive", "Mosley", "Sax"] if c in quarterly.columns]
    rows = []
    for col in series:
        rows.append(
            {
                "Series": col,
                "NN_level_corr": pearson(quarterly["NN"].values, quarterly[col].values),
                "NN_growth_corr": pearson(pct_change(quarterly["NN"]), pct_change(quarterly[col])),
                "Mosley_level_corr": pearson(quarterly["Mosley"].values, quarterly[col].values)
                if col != "Mosley"
                else 1.0,
                "Mosley_growth_corr": pearson(pct_change(quarterly["Mosley"]), pct_change(quarterly[col]))
                if col != "Mosley"
                else 1.0,
            }
        )
    return pd.DataFrame(rows)


def employment_quarterly_corr(quarterly: pd.DataFrame) -> pd.DataFrame:
    emp = pd.read_csv(os.path.join(ROOT, "data", "datausa.io", "Monthly Employment.csv"))
    emp["date"] = pd.to_datetime(emp["Date"])
    emp["Year"] = emp.date.dt.year
    emp["Quarter"] = emp.date.dt.quarter
    emp_q = emp.groupby(["Year", "Quarter"], as_index=False).agg(emp=("NSA Employees", "mean"))
    q = quarterly.merge(emp_q, on=["Year", "Quarter"], how="inner").sort_values(["Year", "Quarter"])
    rows = []
    for col in [c for c in ["NN", "Ridge_signed", "Ridge_positive", "Mosley", "Sax"] if c in q.columns]:
        rows.append(
            {
                "Series": col,
                "quarterly_employment_growth_corr": pearson(pct_change(q[col]), pct_change(q["emp"])),
                "N_quarters": int(len(q)),
            }
        )
    return pd.DataFrame(rows)


def plot_quarterly(quarterly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    styles = {
        "NN": ("#2c6fbb", 2.1),
        "Ridge_signed": ("#20845c", 1.7),
        "Ridge_positive": ("#65a765", 1.3),
        "Mosley": ("#777777", 1.7),
        "Sax": ("#b15a1c", 1.3),
    }
    for col, (color, lw) in styles.items():
        if col in quarterly.columns:
            ax.plot(quarterly["quarter_date"], quarterly[col], label=col.replace("_", " "), color=color, lw=lw)
    ax.set_title("Quarterly aggregation of monthly Step B R&D estimates")
    ax.set_ylabel("Quarterly R&D (USD bn)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "quarterly_stepb_estimates.png"), dpi=200)
    plt.close(fig)


def write_latex(summary: pd.DataFrame, agreement: pd.DataFrame) -> None:
    keep = agreement.copy()
    keep = keep[keep.Series.isin(["Ridge_signed", "Ridge_positive", "Mosley", "Sax"])]
    rows = ["Series & NN level & NN growth & Mosley level & Mosley growth \\\\", "\\midrule"]
    for _, r in keep.iterrows():
        rows.append(
            f"{r.Series.replace('_', ' ')} & {r.NN_level_corr:.2f} & {r.NN_growth_corr:.2f} & "
            f"{r.Mosley_level_corr:.2f} & {r.Mosley_growth_corr:.2f} \\\\"
        )
    text = (
        "% Source: additional_analysis/robustness_overfit/05_quarterly_stepb_values.py\n"
        "\\begin{table}[!htb]\n\\centering\n"
        "\\caption{Quarterly robustness of Step B estimates. Quarterly values preserve annual totals.}\n"
        "\\label{tab:quarterly_stepb_robustness}\n"
        "\\begin{tabular}{l c c c c}\n"
        "\\toprule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(TAB, "quarterly_stepb_robustness_table.tex"), "w") as handle:
        handle.write(text)


def main() -> None:
    monthly = load_monthly_series()
    quarterly = quarterly_from_monthly(monthly)
    checks = annual_sum_check(monthly, quarterly)
    summary = summary_stats(monthly, quarterly)
    agreement = agreement_table(quarterly)
    emp_corr = employment_quarterly_corr(quarterly)

    monthly.to_csv(os.path.join(QUARTER_OUT, "monthly_series_used_for_quarterly.csv"), index=False)
    quarterly.to_csv(os.path.join(QUARTER_OUT, "quarterly_stepb_estimates.csv"), index=False)
    checks.to_csv(os.path.join(QUARTER_OUT, "quarterly_annual_sum_checks.csv"), index=False)
    summary.to_csv(os.path.join(QUARTER_OUT, "quarterly_smoothness_summary.csv"), index=False)
    agreement.to_csv(os.path.join(QUARTER_OUT, "quarterly_agreement_summary.csv"), index=False)
    emp_corr.to_csv(os.path.join(QUARTER_OUT, "quarterly_employment_correlations.csv"), index=False)
    plot_quarterly(quarterly)
    write_latex(summary, agreement)

    print("\nQuarterly agreement with NN and Mosley")
    print(agreement.to_string(index=False))
    print("\nMonthly vs quarterly growth variation")
    print(summary[["Series", "monthly_growth_sd", "quarterly_growth_sd", "monthly_abs_growth_mean", "quarterly_abs_growth_mean"]].to_string(index=False))
    print("\nQuarterly employment growth correlations")
    print(emp_corr.to_string(index=False))
    print("\nAnnual sum check")
    print(checks.to_string(index=False))
    print(f"\nsaved quarterly Step B outputs to {QUARTER_OUT}")


if __name__ == "__main__":
    main()
