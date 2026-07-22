"""Refresh paper Step A artifacts after the updated pure-NN temporal run.

This script intentionally consumes the saved outputs from
10_all_configs_updated_pure_nn.py rather than retraining anything. It overwrites
only paper-ready Step A figures/tables in Nowcasting_Oxford_submission and also
saves audit CSVs beside the robustness results.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CODE = Path(os.environ.get("NOWCASTING_ROOT", Path(__file__).resolve().parents[2]))
PAPER = Path(os.environ.get("NOWCASTING_PAPER_DIR", CODE / "paper"))
ROBUST = CODE / "additional_analysis" / "robustness_overfit"
UPDATED = ROBUST / "out" / "all_configs_updated_pure_nn"
OUT = ROBUST / "out" / "paper_stepA_refresh"
FIG_TEMP = PAPER / "figures" / "Nowcast_Model_Temporal"
FIG_APP = FIG_TEMP / "appendix"
FIG_REV = PAPER / "figures" / "Revision"
TABLES = PAPER / "tables"

CONFIG_ORDER = ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]
COLORS = {
    "LagRD": "#4C78A8",
    "Macros": "#F58518",
    "AGT": "#54A24B",
    "MGT": "#B279A2",
    "AGTwRD": "#E45756",
    "MGTwRD": "#72B7B2",
    "AllVar": "#FF9DA6",
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return {
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1 - np.sum(err**2) / np.sum((y_true - np.mean(y_true)) ** 2)),
    }


def dm_vs_rw(y_true: np.ndarray, y_model: np.ndarray, y_ref: np.ndarray) -> tuple[float, float]:
    d = (y_true - y_model) ** 2 - (y_true - y_ref) ** 2
    n = len(d)
    if n < 3 or np.isclose(np.var(d, ddof=1), 0):
        return math.nan, math.nan
    dm = float(np.mean(d) / math.sqrt(np.var(d, ddof=1) / n))
    dm *= math.sqrt((n - 1) / n)
    p = float(2 * stats.t.sf(abs(dm), df=n - 1))
    return dm, p


def latex_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def load_updated_annual() -> pd.DataFrame:
    preds = pd.read_csv(UPDATED / "all_configs_updated_pure_nn_predictions.csv")
    annual = (
        preds.groupby(["Config", "Country", "Year"], as_index=False)
        .agg(GERD=("rd_expenditure", "first"), pred=("pred", "mean"))
    )
    wide = annual.pivot(index=["Country", "Year", "GERD"], columns="Config", values="pred").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={cfg: f"NN_{cfg}" for cfg in CONFIG_ORDER})
    return wide


def build_annual_panel() -> pd.DataFrame:
    base = pd.read_csv(CODE / "additional_analysis" / "out" / "temporal_annual_all.csv")
    keep = ["Country", "Year", "GERD", "RW1", "RW2", "RW3", "AR1", "AR2", "AR3", "HistMean", "MIDAS", "UMIDAS"]
    base = base[keep]
    updated = load_updated_annual()
    merged = base.merge(updated.drop(columns=["GERD"]), on=["Country", "Year"], how="left")
    sgl = pd.read_csv(CODE / "additional_analysis" / "out" / "sg_lasso_midas_pred.csv")
    merged = merged.merge(sgl, on=["Country", "Year"], how="left")
    merged.to_csv(OUT / "updated_temporal_annual_all.csv", index=False)
    return merged


def build_benchmark_table(panel: pd.DataFrame) -> pd.DataFrame:
    model_cols = [(f"NN_{cfg}", f"\\textit{{{cfg}}}") for cfg in CONFIG_ORDER]
    model_cols += [
        ("RW1", "\\textit{RW (L=1)}"),
        ("RW2", "\\textit{RW (L=2)}"),
        ("RW3", "\\textit{RW (L=3, feasible)}"),
        ("AR3", "\\textit{AR(1) (L=3)}"),
        ("MIDAS", "\\textit{MIDAS (GT)}"),
        ("UMIDAS", "\\textit{U-MIDAS (GT)}"),
        ("SGL", "\\textit{sg-LASSO-MIDAS (GT, full)}"),
        ("HistMean", "\\textit{Hist. mean}"),
    ]
    y = panel["GERD"].to_numpy()
    rw = panel["RW3"].to_numpy()
    rows = []
    for col, label in model_cols:
        m = metrics(y, panel[col].to_numpy())
        dm, p = (math.nan, math.nan) if col == "RW3" else dm_vs_rw(y, panel[col].to_numpy(), rw)
        rows.append({"Model": label, "col": col, **m, "DM": dm, "p": p})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "updated_temporal_benchmarks_metrics.csv", index=False)

    lines = [
        "% Source: additional_analysis/robustness_overfit/11_refresh_stepA_paper_artifacts.py",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\caption{Temporal-split out-of-sample accuracy (annual test points, $n=23$). DM tests compare each model with the feasible RW($L{=}3$) benchmark.}",
        "\\label{tab:temporal_benchmarks}",
        "\\begin{tabular}{l c c c c c}",
        "\\toprule",
        "Model & MAPE (\\%) & RMSE & $R^2$ & DM vs RW(3) & $p$ \\\\",
        "\\midrule",
    ]
    for i, row in out.iterrows():
        if i == 7 or i == 10:
            lines.append("\\midrule")
        lines.append(
            f"{row['Model']} & {latex_num(row.MAPE)} & {latex_num(row.RMSE)} & "
            f"{latex_num(row.R2)} & {latex_num(row.DM)} & {latex_num(row.p)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / "temporal_benchmarks_table.tex").write_text("\n".join(lines) + "\n")
    return out


def build_coverage_table() -> pd.DataFrame:
    members = pd.read_csv(UPDATED / "all_configs_updated_pure_nn_member_predictions.csv")
    ann = (
        members.groupby(["Config", "seed", "Country", "Year"], as_index=False)
        .agg(GERD=("rd_expenditure", "first"), pred=("pred", "mean"))
    )
    rows = []
    for cfg in CONFIG_ORDER:
        sub = ann[ann["Config"] == cfg]
        mat = sub.pivot(index=["Country", "Year", "GERD"], columns="seed", values="pred")
        y = mat.index.get_level_values("GERD").to_numpy(dtype=float)
        vals = mat.to_numpy(dtype=float)
        mean = vals.mean(axis=1)
        sd = vals.std(axis=1, ddof=1)
        qlo, qhi = np.quantile(vals, [0.025, 0.975], axis=1)
        glo, ghi = mean - 1.96 * sd, mean + 1.96 * sd
        rows.append(
            {
                "Configuration": cfg,
                "Quantile coverage": float(np.mean((y >= qlo) & (y <= qhi)) * 100),
                "Gaussian coverage": float(np.mean((y >= glo) & (y <= ghi)) * 100),
                "Avg rel width": float(np.mean((ghi - glo) / y) * 100),
                "Avg member sd": float(np.mean(sd)),
                "MAE": float(np.mean(np.abs(y - mean))),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "updated_temporal_coverage.csv", index=False)
    lines = [
        "% Source: additional_analysis/robustness_overfit/11_refresh_stepA_paper_artifacts.py",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\caption{Temporal-split coverage of 95\\% ensemble intervals. Bands reflect across-member dispersion only.}",
        "\\label{tab:temporal_coverage}",
        "\\begin{tabular}{l c c c}",
        "\\toprule",
        "Configuration & Quantile cov. (\\%) & Gaussian cov. (\\%) & Avg.\\ rel.\\ width (\\%) \\\\",
        "\\midrule",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"\\textit{{{row.Configuration}}} & {row['Quantile coverage']:.0f} & "
            f"{row['Gaussian coverage']:.0f} & {row['Avg rel width']:.0f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / "temporal_coverage_table.tex").write_text("\n".join(lines) + "\n")
    return out


def plot_nn_vs_ols(bench: pd.DataFrame) -> None:
    reg = pd.read_csv(ROBUST / "out" / "regularized_linear_benchmarks.csv")
    ols = reg[reg["Model"] == "OLS"].set_index("Config")
    nn = pd.read_csv(UPDATED / "all_configs_updated_pure_nn_results.csv")
    nn = nn[nn["ensemble_size"] == 15].set_index("Config")
    x = np.arange(len(CONFIG_ORDER))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    for ax, metric, ylabel in zip(axes, ["test_MAPE", "test_RMSE"], ["MAPE (%)", "RMSE"]):
        nn_vals = [nn.loc[cfg, metric] for cfg in CONFIG_ORDER]
        ols_metric = "MAPE" if metric == "test_MAPE" else "RMSE"
        ols_vals = [ols.loc[cfg, ols_metric] for cfg in CONFIG_ORDER]
        colors = [COLORS[cfg] for cfg in CONFIG_ORDER]
        ax.bar(x - width / 2, nn_vals, width, color=colors, edgecolor="black", linewidth=0.4, label="Neural network")
        ax.bar(
            x + width / 2,
            ols_vals,
            width,
            color=colors,
            edgecolor="black",
            linewidth=0.4,
            hatch="///",
            alpha=0.65,
            label="OLS",
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(CONFIG_ORDER, rotation=35, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_TEMP / "NNvsOLS_Temporal.png", dpi=300)
    plt.close(fig)


def plot_allvar_scatter(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    y = panel["GERD"].to_numpy()
    p = panel["NN_AllVar"].to_numpy()
    ax.scatter(y, p, c="#2C6FBB", edgecolor="white", linewidth=0.6, s=52)
    lo = min(y.min(), p.min()) * 0.95
    hi = max(y.max(), p.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1, linestyle="--")
    ax.set_xlabel("Observed annual GERD")
    ax.set_ylabel("Predicted annual GERD")
    ax.set_title("AllVar temporal test predictions")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_TEMP / "AllVar_TrueVsPred_Temporal.png", dpi=300)
    plt.close(fig)


def observed_series() -> pd.DataFrame:
    df = pd.read_csv(CODE / "additional_analysis" / "out" / "merged_features.csv", usecols=["Country", "Year", "rd_expenditure"])
    return df.groupby(["Country", "Year"], as_index=False).agg(GERD=("rd_expenditure", "first"))


def plot_country_trajectory(ax, hist: pd.DataFrame, pred: pd.DataFrame, country: str, cfg: str) -> None:
    h = hist[hist["Country"] == country].sort_values("Year")
    p = pred[pred["Country"] == country].sort_values("Year")
    ax.plot(h["Year"], h["GERD"], color="#333333", marker="o", markersize=3, linewidth=1.1, label="Observed")
    y_max = h["GERD"].max()
    if len(p):
        ax.plot(p["Year"], p[f"NN_{cfg}"], color=COLORS[cfg], marker="s", markersize=3, linewidth=1.3, label="Predicted")
        ax.axvspan(p["Year"].min() - 0.35, p["Year"].max() + 0.35, color="#E8EEF7", alpha=0.65, zorder=0)
        y_max = max(y_max, p[f"NN_{cfg}"].max())
    ax.set_ylim(0, y_max * 1.9)
    ax.set_title(country, fontsize=9)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)


def plot_appendix(panel: pd.DataFrame) -> None:
    hist = observed_series()
    countries = list(hist["Country"].drop_duplicates())
    for cfg in CONFIG_ORDER:
        fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharex=False)
        for ax, country in zip(axes.ravel(), countries):
            plot_country_trajectory(ax, hist, panel, country, cfg)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(f"{cfg} temporal test predictions", y=0.98, fontsize=12)
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        fig.savefig(FIG_APP / f"{cfg}_combined.png", dpi=300)
        plt.close(fig)

    for cfg in ["AGT", "AllVar"]:
        for country in countries:
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            plot_country_trajectory(ax, hist, panel, country, cfg)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG_APP / f"{cfg}_{country}.png", dpi=300)
            plt.close(fig)


def plot_fanchart() -> None:
    hist = observed_series()
    members = pd.read_csv(UPDATED / "all_configs_updated_pure_nn_member_predictions.csv")
    agt = members[members["Config"] == "AGT"]
    ann = (
        agt.groupby(["seed", "Country", "Year"], as_index=False)
        .agg(GERD=("rd_expenditure", "first"), pred=("pred", "mean"))
    )
    countries = ["CA", "DE", "JP"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=False)
    for ax, country in zip(axes, countries):
        h = hist[hist["Country"] == country].sort_values("Year")
        mat = ann[ann["Country"] == country].pivot(index="Year", columns="seed", values="pred").sort_index()
        years = mat.index.to_numpy()
        vals = mat.to_numpy()
        mean = vals.mean(axis=1)
        lo, hi = np.quantile(vals, [0.025, 0.975], axis=1)
        ax.plot(h["Year"], h["GERD"], color="#333333", marker="o", markersize=3, linewidth=1.1, label="Observed")
        ax.axvspan(years.min() - 0.35, years.max() + 0.35, color="#E8EEF7", alpha=0.65, zorder=0)
        ax.fill_between(years, lo, hi, color="#6BAED6", alpha=0.28, label="95% ensemble interval")
        ax.plot(years, mean, color="#1F5A94", marker="s", markersize=3, linewidth=1.4, label="AGT ensemble mean")
        y_max = max(h["GERD"].max(), np.nanmax(hi), np.nanmax(mean))
        ax.set_ylim(0, y_max * 1.75)
        ax.set_title(country, fontsize=10)
        ax.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(FIG_TEMP / "fanchart_panel_AGT.png", dpi=300)
    plt.close(fig)


def plot_calibration() -> None:
    members = pd.read_csv(UPDATED / "all_configs_updated_pure_nn_member_predictions.csv")
    agt = members[members["Config"] == "AGT"]
    ann = (
        agt.groupby(["seed", "Country", "Year"], as_index=False)
        .agg(GERD=("rd_expenditure", "first"), pred=("pred", "mean"))
    )
    mat = ann.pivot(index=["Country", "Year", "GERD"], columns="seed", values="pred")
    y = mat.index.get_level_values("GERD").to_numpy(dtype=float)
    vals = mat.to_numpy(dtype=float)
    mean = vals.mean(axis=1)
    sd = vals.std(axis=1, ddof=1)
    resid = y - mean
    loo_var = np.array([np.mean(np.delete(resid, i) ** 2) for i in range(len(resid))])
    sd_recal = np.sqrt(sd**2 + loo_var)

    nominal = np.linspace(0.05, 0.95, 19)
    rows = []
    for nom in nominal:
        z = stats.norm.ppf((1 + nom) / 2)
        ens_cov = np.mean((y >= mean - z * sd) & (y <= mean + z * sd))
        recal_cov = np.mean((y >= mean - z * sd_recal) & (y <= mean + z * sd_recal))
        rows.append({"nominal": nom, "ensemble_only": ens_cov, "recalibrated": recal_cov})
    calib = pd.DataFrame(rows)
    calib.to_csv(OUT / "updated_agt_calibration.csv", index=False)

    pit_ens = stats.norm.cdf((y - mean) / np.maximum(sd, 1e-9))
    pit_recal = stats.norm.cdf((y - mean) / np.maximum(sd_recal, 1e-9))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
    axes[0].plot(calib["nominal"], calib["ensemble_only"], marker="o", label="Ensemble only")
    axes[0].plot(calib["nominal"], calib["recalibrated"], marker="s", label="Recalibrated")
    axes[0].set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Reliability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    bins = np.linspace(0, 1, 9)
    axes[1].hist(pit_ens, bins=bins, alpha=0.65, label="Ensemble only", color="#4C78A8")
    axes[1].hist(pit_recal, bins=bins, alpha=0.55, label="Recalibrated", color="#F58518")
    axes[1].set_xlabel("PIT")
    axes[1].set_ylabel("Count")
    axes[1].set_title("PIT histograms")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_REV / "calibration.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_APP.mkdir(parents=True, exist_ok=True)
    FIG_REV.mkdir(parents=True, exist_ok=True)
    panel = build_annual_panel()
    bench = build_benchmark_table(panel)
    coverage = build_coverage_table()
    plot_nn_vs_ols(bench)
    plot_allvar_scatter(panel)
    plot_appendix(panel)
    plot_fanchart()
    plot_calibration()
    print("Updated benchmark rows:")
    print(bench[["Model", "MAPE", "RMSE", "R2", "DM", "p"]].to_string(index=False))
    print("\nUpdated coverage:")
    print(coverage.to_string(index=False))
    print(f"\nSaved audit outputs to {OUT}")


if __name__ == "__main__":
    main()
