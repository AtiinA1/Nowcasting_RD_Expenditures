"""Dependence- and lag-search-robust inference for employment cross-correlations.

The year-block bootstrap preserves month-of-year means, within-year dependence,
and dependence across the alternative R&D allocations. Employment year blocks
are sampled independently under the null. Simultaneous p-values use the maximum
absolute correlation over all 25 lags, either within each method or globally.
"""

import os
from pathlib import Path

LOCAL_CACHE = Path(__file__).resolve().parent / "out" / "cache"
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "out" / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
ESTIMATES = ROOT / "additional_analysis/out/combined_estimates_temporal_level.csv"
EMPLOYMENT = ROOT / "data/datausa.io/Monthly Employment.csv"
OUT = Path(__file__).resolve().parent / "out"
LAGS = np.arange(-12, 13)
METHODS = ["NN", "Mosley", "Sax", "Uniform"]
N_BOOT = 10_000
SEED = 20260715


def lagged_pair(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive lag compares earlier R&D growth with later employment growth."""
    if lag > 0:
        return x[:-lag], y[lag:]
    if lag < 0:
        return x[-lag:], y[:lag]
    return x, y


def correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = []
    for lag in LAGS:
        x_lag, y_lag = lagged_pair(x, y, int(lag))
        values.append(np.corrcoef(x_lag, y_lag)[0, 1])
    return np.asarray(values)


def load_balanced_growth_panel() -> pd.DataFrame:
    estimates = pd.read_csv(ESTIMATES, parse_dates=["date"]).sort_values("date")
    annual = estimates.groupby("Year")["NN"].transform("sum")
    estimates["Uniform"] = annual / 12.0
    for method in METHODS:
        estimates[f"g_{method}"] = estimates[method].pct_change()

    employment = pd.read_csv(EMPLOYMENT, parse_dates=["Date"])
    employment = employment.sort_values("Date")
    employment["g_Employment"] = employment["NSA Employees"].pct_change()

    panel = estimates.merge(
        employment[["Date", "g_Employment"]], left_on="date", right_on="Date", how="inner"
    )
    panel = panel.loc[panel["Year"].between(2009, 2021)].copy()
    counts = panel.groupby("Year")["Month"].nunique()
    complete_years = counts[counts == 12].index
    panel = panel[panel["Year"].isin(complete_years)].sort_values(["Year", "Month"])
    expected = len(complete_years) * 12
    if len(panel) != expected or panel[[f"g_{m}" for m in METHODS] + ["g_Employment"]].isna().any().any():
        raise ValueError("The balanced growth panel is incomplete.")
    return panel


def year_month_matrix(panel: pd.DataFrame, column: str) -> np.ndarray:
    return panel.pivot(index="Year", columns="Month", values=column).sort_index().to_numpy()


def hac_results(method: str, x: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for lag in LAGS:
        x_lag, y_lag = lagged_pair(x, y, int(lag))
        x_std = (x_lag - x_lag.mean()) / x_lag.std(ddof=0)
        y_std = (y_lag - y_lag.mean()) / y_lag.std(ddof=0)
        fit = sm.OLS(y_std, sm.add_constant(x_std)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 12, "use_correction": True}
        )
        rows.append(
            {
                "method": method,
                "lag": int(lag),
                "n": len(x_lag),
                "correlation": float(np.corrcoef(x_lag, y_lag)[0, 1]),
                "hac_se": float(fit.bse[1]),
                "hac_t": float(fit.tvalues[1]),
                "hac_p": float(fit.pvalues[1]),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_balanced_growth_panel()
    matrices = {m: year_month_matrix(panel, f"g_{m}") for m in METHODS}
    employment = year_month_matrix(panel, "g_Employment")
    observed = {m: correlations(matrices[m].ravel(), employment.ravel()) for m in METHODS}

    rng = np.random.default_rng(SEED)
    n_years = employment.shape[0]
    rd_month_means = {m: matrices[m].mean(axis=0) for m in METHODS}
    rd_residuals = {m: matrices[m] - rd_month_means[m] for m in METHODS}
    emp_month_mean = employment.mean(axis=0)
    emp_residuals = employment - emp_month_mean

    boot_corr = {m: np.empty((N_BOOT, len(LAGS))) for m in METHODS}
    for b in range(N_BOOT):
        rd_idx = rng.integers(0, n_years, size=n_years)
        emp_idx = rng.integers(0, n_years, size=n_years)
        emp_b = (emp_month_mean + emp_residuals[emp_idx]).ravel()
        for method in METHODS:
            rd_b = (rd_month_means[method] + rd_residuals[method][rd_idx]).ravel()
            boot_corr[method][b] = correlations(rd_b, emp_b)

    method_max = {m: np.max(np.abs(boot_corr[m]), axis=1) for m in METHODS}
    global_max = np.max(np.column_stack([method_max[m] for m in METHODS]), axis=1)

    hac_rows = []
    for method in METHODS:
        hac_rows.extend(hac_results(method, matrices[method].ravel(), employment.ravel()))
    results = pd.DataFrame(hac_rows)

    for method in METHODS:
        mask = results["method"].eq(method)
        results.loc[mask, "hac_p_holm_method"] = multipletests(
            results.loc[mask, "hac_p"], method="holm"
        )[1]
        obs_abs = np.abs(observed[method])
        results.loc[mask, "block_p_max_method"] = [
            (1 + np.count_nonzero(method_max[method] >= value)) / (N_BOOT + 1)
            for value in obs_abs
        ]
        results.loc[mask, "block_p_max_global"] = [
            (1 + np.count_nonzero(global_max >= value)) / (N_BOOT + 1)
            for value in obs_abs
        ]
    results["hac_p_holm_global"] = multipletests(results["hac_p"], method="holm")[1]
    results = results.sort_values(["method", "lag"])
    results.to_csv(OUT / "lag_inference.csv", index=False)

    critical_rows = []
    for method in METHODS:
        critical_rows.append(
            {
                "scope": method,
                "q90": np.quantile(method_max[method], 0.90),
                "q95": np.quantile(method_max[method], 0.95),
                "q99": np.quantile(method_max[method], 0.99),
            }
        )
    critical_rows.append(
        {
            "scope": "Global (4 methods x 25 lags)",
            "q90": np.quantile(global_max, 0.90),
            "q95": np.quantile(global_max, 0.95),
            "q99": np.quantile(global_max, 0.99),
        }
    )
    pd.DataFrame(critical_rows).to_csv(OUT / "simultaneous_critical_values.csv", index=False)

    strongest = (
        results.assign(abs_r=results["correlation"].abs())
        .sort_values(["method", "abs_r"], ascending=[True, False])
        .groupby("method", as_index=False)
        .first()
    )
    strongest.to_csv(OUT / "strongest_lag_by_method.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True, sharey=True)
    colors = {"NN": "#2F5D7E", "Mosley": "#748C56", "Sax": "#A6654E", "Uniform": "#666666"}
    for ax, method in zip(axes.ravel(), METHODS):
        row = results[results["method"].eq(method)]
        band = np.quantile(method_max[method], 0.95)
        ax.axhspan(-band, band, color="#E6E6E6", zorder=0)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.plot(row["lag"], row["correlation"], marker="o", markersize=3, color=colors[method])
        ax.set_title(method)
        ax.set_ylim(-0.7, 0.7)
        ax.grid(axis="y", alpha=0.25)
    axes[1, 0].set_xlabel("Lag (months), positive means R&D leads employment")
    axes[1, 1].set_xlabel("Lag (months), positive means R&D leads employment")
    axes[0, 0].set_ylabel("Growth correlation")
    axes[1, 0].set_ylabel("Growth correlation")
    fig.suptitle("Employment cross-correlations with 95% year-block max-|r| bands")
    fig.tight_layout()
    fig.savefig(OUT / "employment_ccf_simultaneous_bands.png", dpi=220)
    plt.close(fig)

    print(f"Balanced period: {panel['Year'].min()}-{panel['Year'].max()}, n={len(panel)}, years={n_years}")
    print("\nSimultaneous critical values:")
    print(pd.DataFrame(critical_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nStrongest lag by method:")
    cols = ["method", "lag", "correlation", "hac_p", "hac_p_holm_method", "block_p_max_method", "block_p_max_global"]
    print(strongest[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
