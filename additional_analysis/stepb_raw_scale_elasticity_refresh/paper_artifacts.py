"""Build paper artifacts for the raw-scale Step B elasticity implementation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parents[2]
PAPER = CODE / "paper"
OUT = CODE / "additional_analysis" / "stepb_raw_scale_elasticity_refresh" / "out"
FIG_DISAGG = PAPER / "figures" / "Disaggregation"
FIG_REV = PAPER / "figures" / "Revision"
TABLES = PAPER / "tables"

NN_COLOR = "#3D5A80"
MOSLEY_COLOR = "#7A7A72"
CHOW_COLOR = "#A66A3F"
EMP_COLOR = "#6B8E6B"


def growth(values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.diff(values) / values[:-1]


def lag_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    xg, yg = growth(x), growth(y)
    if lag < 0:
        a, b = xg[-lag:], yg[:lag]
    elif lag > 0:
        a, b = xg[:-lag], yg[lag:]
    else:
        a, b = xg, yg
    return float(np.corrcoef(a, b)[0, 1])


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(OUT / "corrected_temporal_monthly_estimates.csv")
    monthly["date"] = pd.to_datetime(dict(year=monthly.Year, month=monthly.Month, day=1))
    employment = pd.read_csv(CODE / "data" / "datausa.io" / "Monthly Employment.csv")
    employment["date"] = pd.to_datetime(employment.Date)
    employment = employment[["date", "NSA Employees"]].rename(columns={"NSA Employees": "emp"})
    return monthly, employment


def plot_monthly(monthly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    low = monthly[["NN", "Sax", "Mosley"]].min(axis=1)
    high = monthly[["NN", "Sax", "Mosley"]].max(axis=1)
    ax.fill_between(monthly.date, low, high, color="#B8B8B0", alpha=0.28, label="Range across methods")
    ax.plot(monthly.date, monthly.NN, color=NN_COLOR, lw=1.4, label="NN elasticity")
    ax.set_ylabel("Monthly R&D (billions of 2015 USD PPP)")
    ax.set_title("NN elasticity-based monthly allocation")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DISAGG / "NNelasticity_temporal_level.png", dpi=300)
    plt.close(fig)


def plot_employment(monthly: pd.DataFrame, employment: pd.DataFrame) -> None:
    merged = monthly.merge(employment, on="date", how="inner").sort_values("date")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    line1 = ax1.plot(merged.date, merged.NN, color=NN_COLOR, lw=1.4, label="NN elasticity R&D")
    ax1.set_ylabel("Monthly R&D (billions of 2015 USD PPP)", color=NN_COLOR)
    ax1.tick_params(axis="y", labelcolor=NN_COLOR)
    ax2 = ax1.twinx()
    line2 = ax2.plot(merged.date, merged.emp, color=EMP_COLOR, lw=1.2, alpha=0.9, label="R&D-services employment")
    ax2.set_ylabel("Employees", color=EMP_COLOR)
    ax2.tick_params(axis="y", labelcolor=EMP_COLOR)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(True, alpha=0.22)
    ax1.legend(line1 + line2, [line.get_label() for line in line1 + line2], fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DISAGG / "employee_temporal_level.png", dpi=300)
    plt.close(fig)


def plot_ccf(monthly: pd.DataFrame, employment: pd.DataFrame) -> None:
    merged = monthly.merge(employment, on="date", how="inner").sort_values("date").reset_index(drop=True)
    lags = list(range(-12, 13))
    band = 1.96 / np.sqrt(len(merged) - 1)
    methods = [
        ("NN elasticity", "NN", NN_COLOR),
        ("Sparse temporal disaggregation", "Mosley", MOSLEY_COLOR),
        ("Chow-Lin", "Sax", CHOW_COLOR),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, column, color in methods:
        values = [lag_corr(merged[column].values, merged.emp.values, lag) for lag in lags]
        ax.plot(lags, values, marker="o", ms=3, lw=1.3, label=label, color=color)
    ax.axhspan(-band, band, color="#B8B8B0", alpha=0.22, label="95% band")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Lag (months), negative values indicate R&D leading employment")
    ax.set_ylabel("Growth-rate cross-correlation")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIG_REV / "disagg_ccf.png", dpi=300)
    plt.close(fig)


def write_model_table() -> None:
    summary = pd.read_csv(OUT / "corrected_linear_stepb_summary.csv")
    keep = summary[
        summary.Variant.isin(["signed", "positive_part"])
        & summary.Model.isin(["Elastic Net", "OLS", "Ridge"])
    ].copy()
    order = {
        ("Elastic Net", "positive_part"): 0,
        ("Elastic Net", "signed"): 1,
        ("OLS", "positive_part"): 2,
        ("OLS", "signed"): 3,
        ("Ridge", "positive_part"): 4,
        ("Ridge", "signed"): 5,
    }
    keep["order"] = [order[(m, v)] for m, v in zip(keep.Model, keep.Variant)]
    keep.sort_values("order", inplace=True)
    rows = []
    for row in keep.itertuples(index=False):
        variant = "positive part" if row.Variant == "positive_part" else "signed"
        rows.append(
            f"{row.Model} & {variant} & {row.NN_level_corr:.2f} & {row.NN_growth_corr:.2f} & "
            f"{row.Mosley_level_corr:.2f} & {row.Mosley_growth_corr:.2f} & "
            f"{row.negative_topic_elasticity_share:.2f} & {int(row.employment_sig_lags_p01)} \\\\"
        )
    table = """% Source: additional_analysis/stepb_raw_scale_elasticity_refresh/raw_scale_linear_robustness.py
\\begin{table}[!htb]
\\centering
\\caption{Raw-scale elasticity robustness across Step A models.}
\\label{tab:stepb_model_agnostic_elasticities}
\\small
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{l l c c c c c c}
\\toprule
Model & Variant & NN level & NN growth & Mosley level & Mosley growth & Neg. topics & Emp. sig. lags \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (TABLES / "stepb_model_agnostic_elasticities_table.tex").write_text(table)


def write_quarterly_table() -> None:
    agreement = pd.read_csv(OUT / "corrected_quarterly_agreement_all_methods.csv")
    labels = {
        "Ridge_signed": "Ridge signed",
        "Ridge_positive": "Ridge positive",
        "Mosley": "Mosley",
        "Sax": "Chow-Lin",
    }
    rows = []
    for row in agreement.itertuples(index=False):
        rows.append(
            f"{labels[row.Series]} & {row.NN_level_corr:.2f} & {row.NN_growth_corr:.2f} & "
            f"{row.Mosley_level_corr:.2f} & {row.Mosley_growth_corr:.2f} \\\\"
        )
    table = """% Source: additional_analysis/stepb_raw_scale_elasticity_refresh/raw_scale_quarterly_robustness.py
\\begin{table}[!htb]
\\centering
\\caption{Quarterly robustness of raw-scale elasticity allocations.}
\\label{tab:quarterly_stepb_robustness}
\\begin{tabular}{l c c c c}
\\toprule
Series & NN level & NN growth & Mosley level & Mosley growth \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (TABLES / "quarterly_stepb_robustness_table.tex").write_text(table)


def main() -> None:
    monthly, employment = load_inputs()
    plot_monthly(monthly)
    plot_employment(monthly, employment)
    plot_ccf(monthly, employment)
    write_model_table()
    write_quarterly_table()
    print("Wrote corrected raw-scale Step B paper artifacts.")


if __name__ == "__main__":
    main()
