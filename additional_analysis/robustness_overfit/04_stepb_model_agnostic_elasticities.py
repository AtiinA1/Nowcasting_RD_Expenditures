"""Model-agnostic Step B elasticity robustness.

This additive analysis asks whether the monthly disaggregation evidence is
specific to the neural network, or whether other strong Step A models produce
similar Step B allocations once converted to model-implied elasticities.

All outputs are written inside additional_analysis/robustness_overfit.
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "out", "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from common import OUT, FIG, TAB, ROOT, SOURCE_OUT, load_features  # noqa: E402


warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.makedirs(os.path.join(OUT, "stepb_model_agnostic"), exist_ok=True)

STEPB_OUT = os.path.join(OUT, "stepb_model_agnostic")
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
ENET_ALPHAS = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
L1_RATIOS = [0.05, 0.5]


def chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    split = {}
    for ctry, g in df.groupby("Country"):
        years = np.array(sorted(g.Year.unique()))
        n = len(years)
        n_train = int(round(0.64 * n))
        n_val = int(round(0.16 * n))
        for i, year in enumerate(years):
            split[(ctry, int(year))] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    out = df.copy()
    out["split"] = [split[(c, int(y))] for c, y in zip(out.Country, out.Year)]
    return out


def agt_topics(cols: list[str]) -> list[str]:
    topics = set()
    for col in cols:
        for lag in (1, 2, 3):
            suffix = f"_yearly_avg_lag{lag}"
            if col.endswith(suffix):
                topics.add(col.replace(suffix, ""))
    return sorted(topics)


def annual_mape(df: pd.DataFrame, pred_col: str) -> float:
    ann = df.groupby(["Country", "Year"], as_index=False).agg(
        actual=("rd_expenditure", "mean"),
        pred=(pred_col, "mean"),
    )
    return float(np.mean(np.abs((ann.actual - ann.pred) / ann.actual)) * 100)


def predict_level(model, X: np.ndarray, y_mean: float, y_sd: float) -> np.ndarray:
    return model.predict(X) * y_sd + y_mean


def tune_ridge(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, y_mean: float, y_sd: float) -> tuple[Ridge, str, float]:
    train = (df.split == "train").values
    val = (df.split == "val").values
    best = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(X[train], y[train])
        tmp = df[val][["Country", "Year", "Month", "rd_expenditure"]].copy()
        tmp["pred"] = predict_level(model, X[val], y_mean, y_sd)
        score = annual_mape(tmp, "pred")
        if best is None or score < best[0]:
            best = (score, alpha)
    _, alpha = best
    fit = train | val
    model = Ridge(alpha=alpha)
    model.fit(X[fit], y[fit])
    return model, f"alpha={alpha:.3g}", float(best[0])


def tune_elastic_net(
    X: np.ndarray, y: np.ndarray, df: pd.DataFrame, y_mean: float, y_sd: float
) -> tuple[ElasticNet, str, float]:
    train = (df.split == "train").values
    val = (df.split == "val").values
    best = None
    for l1_ratio in L1_RATIOS:
        for alpha in ENET_ALPHAS:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=30000, tol=1e-4)
            model.fit(X[train], y[train])
            tmp = df[val][["Country", "Year", "Month", "rd_expenditure"]].copy()
            tmp["pred"] = predict_level(model, X[val], y_mean, y_sd)
            score = annual_mape(tmp, "pred")
            if best is None or score < best[0]:
                best = (score, alpha, l1_ratio)
    _, alpha, l1_ratio = best
    fit = train | val
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=30000, tol=1e-4)
    model.fit(X[fit], y[fit])
    return model, f"alpha={alpha:.3g}, l1={l1_ratio:.2g}", float(best[0])


def load_monthly_references() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(os.path.join(ROOT, "data", "GT", "trends_data_by_topic_filtered.csv"))
    gt["date"] = pd.to_datetime(gt["date"])
    gt["Year"] = gt.date.dt.year
    gt["Month"] = gt.date.dt.month

    ref = pd.read_csv(os.path.join(ROOT, "temporal_disaggregation", "results", "combined_estimates.csv"))
    ref = ref[ref.Country == "US"][
        ["Year", "Month", "Monthly_RD_Expenditure_Tempdisagg_Sax", "Monthly_RD_Expenditure_Tempdisagg_Mosley"]
    ].copy()
    ref.rename(
        columns={
            "Monthly_RD_Expenditure_Tempdisagg_Sax": "Sax",
            "Monthly_RD_Expenditure_Tempdisagg_Mosley": "Mosley",
        },
        inplace=True,
    )
    nn = pd.read_csv(os.path.join(SOURCE_OUT, "combined_estimates_temporal_level.csv"))[
        ["Year", "Month", "NN"]
    ].copy()
    emp = pd.read_csv(os.path.join(ROOT, "data", "datausa.io", "Monthly Employment.csv"))
    emp["date"] = pd.to_datetime(emp["Date"])
    emp = emp[["date", "NSA Employees"]].rename(columns={"NSA Employees": "emp"})
    return gt, ref.set_index(["Year", "Month"])["Sax"], nn.merge(ref, on=["Year", "Month"]), emp


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(stats.pearsonr(a[mask], b[mask])[0])


def growth(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.diff(x) / x[:-1]


def employment_lag_summary(series: pd.DataFrame, emp: pd.DataFrame) -> tuple[int, float, int]:
    merged = series.merge(emp, on="date", how="inner").sort_values("date")
    rdg = growth(merged["estimate"].values)
    eg = growth(merged["emp"].values)
    rows = []
    for lag in range(-5, 6):
        if lag < 0:
            a, b = rdg[-lag:], eg[:lag]
        elif lag > 0:
            a, b = rdg[:-lag], eg[lag:]
        else:
            a, b = rdg, eg
        if len(a) > 3 and np.std(a) > 0 and np.std(b) > 0:
            r, p = stats.pearsonr(a, b)
            rows.append((lag, float(r), float(p)))
    sig = [r for r in rows if r[2] < 0.01]
    if not rows:
        return 0, np.nan, 0
    best = max(rows, key=lambda row: abs(row[1]))
    return len(sig), float(best[1]), int(best[0])


def disaggregate_us(
    model_name: str,
    eta_us: dict[str, float],
    topics: list[str],
    gt: pd.DataFrame,
    rd_us: pd.Series,
    references: pd.DataFrame,
    emp: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    tp_ok = [topic for topic in topics if f"US_{topic}" in gt.columns and np.isfinite(eta_us.get(topic, np.nan))]
    g = gt[["Year", "Month"] + [f"US_{topic}" for topic in tp_ok]].copy()
    g = g[g.Year.isin([year for year in g.Year.unique() if year in rd_us.index])].copy()
    raw_adj = np.zeros(len(g), dtype=float)
    pos_adj = np.zeros(len(g), dtype=float)
    abs_adj = np.zeros(len(g), dtype=float)
    for topic in tp_ok:
        col = f"US_{topic}"
        den = g.groupby("Year")[col].transform("sum").values
        share = np.where(den > 0, g[col].values / np.where(den > 0, den, 1.0), 0.0)
        eta = float(eta_us[topic])
        raw_adj += eta * share
        pos_adj += max(eta, 0.0) * share
        abs_adj += abs(eta) * share
    g["signed_weight"] = raw_adj
    g["positive_weight"] = pos_adj
    g["absolute_weight"] = abs_adj

    records = []
    series = []
    for variant, weight_col in [
        ("signed", "signed_weight"),
        ("positive_part", "positive_weight"),
        ("absolute", "absolute_weight"),
    ]:
        out = g[["Year", "Month", weight_col]].copy()
        out.rename(columns={weight_col: "weight"}, inplace=True)
        out["annual_weight"] = out.groupby("Year")["weight"].transform("sum")
        out["estimate"] = np.where(
            np.abs(out["annual_weight"]) > 1e-12,
            out.Year.map(rd_us) * out["weight"] / out["annual_weight"],
            np.nan,
        )
        out["date"] = pd.to_datetime(dict(year=out.Year, month=out.Month, day=1))
        out["Model"] = model_name
        out["Variant"] = variant
        ref = out.merge(references, on=["Year", "Month"], how="inner").dropna(subset=["estimate"])
        if len(ref):
            emp_count, emp_best, emp_lag = employment_lag_summary(ref[["date", "estimate"]], emp)
            records.append(
                {
                    "Model": model_name,
                    "Variant": variant,
                    "N": int(len(ref)),
                    "min_estimate": float(ref.estimate.min()),
                    "negative_months": int((ref.estimate < 0).sum()),
                    "NN_level_corr": pearson(ref.estimate.values, ref.NN.values),
                    "NN_growth_corr": pearson(growth(ref.estimate.values), growth(ref.NN.values)),
                    "Mosley_level_corr": pearson(ref.estimate.values, ref.Mosley.values),
                    "Mosley_growth_corr": pearson(growth(ref.estimate.values), growth(ref.Mosley.values)),
                    "Sax_level_corr": pearson(ref.estimate.values, ref.Sax.values),
                    "Sax_growth_corr": pearson(growth(ref.estimate.values), growth(ref.Sax.values)),
                    "employment_sig_lags_p01": emp_count,
                    "employment_best_abs_corr": emp_best,
                    "employment_best_lag": emp_lag,
                }
            )
        series.append(out)
    return records, pd.concat(series, ignore_index=True)


def main() -> None:
    df = chronological_split(load_features())
    agt_cols = [col for col in df.columns if "_yearly_avg_lag" in col]
    topics = agt_topics(agt_cols)
    months = pd.get_dummies(df.Month.astype(int), prefix="M").astype(float)
    countries = pd.get_dummies(df.Country, prefix="c").astype(float)
    train = (df.split == "train").values
    fit = (df.split.isin(["train", "val"])).values

    scaler = StandardScaler().fit(df[agt_cols].fillna(0).astype(float).values[train])
    X_gt = scaler.transform(df[agt_cols].fillna(0).astype(float).values)
    X = np.column_stack([X_gt, months.values, countries.values])
    y_level = df.rd_expenditure.values.astype(float)
    y_mean = float(y_level[train].mean())
    y_sd = float(y_level[train].std())
    y = (y_level - y_mean) / y_sd

    ols = LinearRegression()
    ols.fit(X[fit], y[fit])
    ridge, ridge_tuning, ridge_val = tune_ridge(X, y, df, y_mean, y_sd)
    enet, enet_tuning, enet_val = tune_elastic_net(X, y, df, y_mean, y_sd)
    models = [
        ("OLS", ols, "", np.nan),
        ("Ridge", ridge, ridge_tuning, ridge_val),
        ("Elastic Net", enet, enet_tuning, enet_val),
    ]

    gt, _, references, emp = load_monthly_references()
    rd_us = df[df.Country == "US"].groupby("Year").rd_expenditure.mean()
    all_records = []
    all_series = []
    all_topics = []
    all_elasticities = []

    for model_name, model, tuning, val_mape in models:
        base = predict_level(model, X, y_mean, y_sd)
        elasticities = np.zeros((len(df), len(agt_cols)), dtype=float)
        for j in range(len(agt_cols)):
            Xp = X.copy()
            Xp[:, j] *= 1.01
            perturbed = predict_level(model, Xp, y_mean, y_sd)
            denom = np.where(np.abs(base) > 1e-12, base, np.nan)
            elasticities[:, j] = ((perturbed - base) / denom) / 0.01
        eldf = pd.DataFrame(elasticities, columns=agt_cols)
        eldf["Country"] = df.Country.values
        eldf["split"] = df.split.values
        eta = eldf[eldf.split == "train"].groupby("Country")[agt_cols].mean()
        eta_us = {}
        for topic in topics:
            lag_cols = [f"{topic}_yearly_avg_lag{lag}" for lag in (1, 2, 3) if f"{topic}_yearly_avg_lag{lag}" in eta.columns]
            eta_us[topic] = float(np.nanmean(eta.loc["US", lag_cols].values)) if lag_cols else np.nan

        top = sorted(eta_us.items(), key=lambda item: -abs(item[1]) if np.isfinite(item[1]) else -np.inf)
        for rank, (topic, value) in enumerate(top[:20], 1):
            all_topics.append(
                {
                    "Model": model_name,
                    "rank": rank,
                    "topic": topic,
                    "elasticity": value,
                    "abs_elasticity": abs(value),
                }
            )
        for topic, value in eta_us.items():
            all_elasticities.append({"Model": model_name, "topic": topic, "elasticity": value})

        records, series = disaggregate_us(model_name, eta_us, topics, gt, rd_us, references, emp)
        for record in records:
            record["tuning"] = tuning
            record["val_MAPE"] = val_mape
            record["negative_topic_elasticity_share"] = float(np.mean([v < 0 for v in eta_us.values() if np.isfinite(v)]))
        all_records.extend(records)
        all_series.append(series)
        print(f"finished {model_name}: {tuning or 'unregularized'}", flush=True)

    summary = pd.DataFrame(all_records)
    monthly = pd.concat(all_series, ignore_index=True)
    topics_out = pd.DataFrame(all_topics)
    elasticities_out = pd.DataFrame(all_elasticities)
    summary.to_csv(os.path.join(STEPB_OUT, "stepb_alt_model_summary.csv"), index=False)
    monthly.to_csv(os.path.join(STEPB_OUT, "stepb_alt_monthly_estimates.csv"), index=False)
    topics_out.to_csv(os.path.join(STEPB_OUT, "stepb_alt_top_topics.csv"), index=False)
    elasticities_out.to_csv(os.path.join(STEPB_OUT, "stepb_alt_topic_elasticities.csv"), index=False)

    plot_df = monthly[monthly.Variant == "positive_part"].pivot_table(
        index="date", columns="Model", values="estimate", aggfunc="mean"
    )
    nn = references.copy()
    nn["date"] = pd.to_datetime(dict(year=nn.Year, month=nn.Month, day=1))
    plot_df = plot_df.join(nn.set_index("date")[["NN", "Mosley"]], how="left")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for col, color in [
        ("NN", "#2c6fbb"),
        ("Ridge", "#20845c"),
        ("Elastic Net", "#b15a1c"),
        ("OLS", "#7a4bb3"),
        ("Mosley", "#777777"),
    ]:
        if col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col], label=col, lw=1.5 if col != "NN" else 2.0, color=color)
    ax.set_title("Step B monthly estimates from alternative Step A elasticities")
    ax.set_ylabel("Monthly R&D (USD bn)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "stepb_alt_model_elasticities.png"), dpi=200)
    plt.close(fig)

    rows = [
        "Model & Variant & NN level & NN growth & Mosley level & Mosley growth & Neg. topics & Emp. sig. lags \\\\",
        "\\midrule",
    ]
    table_df = summary[summary.Variant.isin(["signed", "positive_part"])].copy()
    table_df.sort_values(["Model", "Variant"], inplace=True)
    for _, row in table_df.iterrows():
        variant = str(row.Variant).replace("_", " ")
        rows.append(
            f"{row.Model} & {variant} & {row.NN_level_corr:.2f} & {row.NN_growth_corr:.2f} & "
            f"{row.Mosley_level_corr:.2f} & {row.Mosley_growth_corr:.2f} & "
            f"{row.negative_topic_elasticity_share:.2f} & {int(row.employment_sig_lags_p01)} \\\\"
        )
    latex = (
        "% Source: additional_analysis/robustness_overfit/04_stepb_model_agnostic_elasticities.py\n"
        "\\begin{table}[!htb]\n\\centering\n"
        "\\caption{Model-agnostic Step B robustness using alternative Step A elasticities.}\n"
        "\\label{tab:stepb_model_agnostic_elasticities}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{l l c c c c c c}\n"
        "\\toprule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(TAB, "stepb_model_agnostic_elasticities_table.tex"), "w") as handle:
        handle.write(latex)

    cols = [
        "Model",
        "Variant",
        "NN_level_corr",
        "NN_growth_corr",
        "Mosley_level_corr",
        "Mosley_growth_corr",
        "Sax_level_corr",
        "Sax_growth_corr",
        "negative_topic_elasticity_share",
        "negative_months",
        "employment_sig_lags_p01",
        "employment_best_abs_corr",
        "employment_best_lag",
    ]
    print(summary[cols].sort_values(["Model", "Variant"]).to_string(index=False))
    print(f"saved Step B model-agnostic robustness outputs to {STEPB_OUT}")


if __name__ == "__main__":
    main()
