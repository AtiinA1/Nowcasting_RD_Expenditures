"""Refresh Step B using the updated pure-AGT NN architecture.

The Step A forecast table uses a log-standardized target, but Step B in the
paper is a level-based temporal allocation. This script therefore keeps the
existing Step B level-elasticity estimand and replaces the old AGT MLP with the
updated pure-NN architecture/training recipe used in the Step A refresh.

Paper-facing outputs are written to additional_analysis/out and
Nowcasting_Oxford_submission/figures. Audit copies are written under
robustness_overfit/out/updated_stepB_agt.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("NOWCASTING_ROOT", Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "additional_analysis" / "robustness_overfit" / "out" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler


CODE = REPO_ROOT
PAPER = Path(os.environ.get("NOWCASTING_PAPER_DIR", CODE / "paper"))
SOURCE_OUT = CODE / "additional_analysis" / "out"
ROBUST_OUT = CODE / "additional_analysis" / "robustness_overfit" / "out" / "updated_stepB_agt"
FIG_DISAGG = PAPER / "figures" / "Disaggregation"
FIG_REV = PAPER / "figures" / "Revision"

for path in [SOURCE_OUT, ROBUST_OUT, FIG_DISAGG, FIG_REV]:
    path.mkdir(parents=True, exist_ok=True)

HIDDEN = (64, 16)
DROPOUT = 0.12
WEIGHT_DECAY = 5e-4
LR = 3e-3
ENSEMBLE_SIZE = 15
SEED = 0
CHOWLIN6 = ["Capitalization", "Investment_management", "Patent_office", "Tax_credit", "Cost", "Technology"]


class WideDeepLevelNN(nn.Module):
    def __init__(self, d: int, n_countries: int):
        super().__init__()
        self.emb = nn.Embedding(n_countries, 4)
        dims = [d + 4] + list(HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(HIDDEN)))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for h in HIDDEN)
        self.out = nn.Linear(dims[-1], 1)
        self.skip = nn.Linear(d, 1)
        self.drop = nn.Dropout(DROPOUT)
        self.act = nn.SiLU()

    def forward(self, x, c):
        z = torch.cat([x, self.emb(c)], dim=1)
        for layer, norm in zip(self.layers, self.norms):
            z = self.drop(self.act(norm(layer(z))))
        return self.out(z) + self.skip(x)


def tensor(x: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(x)


def load_inputs() -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    merged = pd.read_csv(SOURCE_OUT / "merged_features.csv")
    merged = merged[merged.Year >= 2004].copy()
    merged.sort_values(["Country", "Year", "Month"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    agt_cols = [c for c in merged.columns if "_yearly_avg_lag" in c]
    topics = sorted(
        {
            c.replace("_yearly_avg_lag1", "").replace("_yearly_avg_lag2", "").replace("_yearly_avg_lag3", "")
            for c in agt_cols
        }
    )

    gt = pd.read_csv(CODE / "data" / "GT" / "trends_data_by_topic_filtered.csv")
    gt["date"] = pd.to_datetime(gt["date"])
    gt["Year"] = gt.date.dt.year
    gt["Month"] = gt.date.dt.month

    rd_us = merged[merged.Country == "US"].groupby("Year").rd_expenditure.mean()
    refs = pd.read_csv(CODE / "temporal_disaggregation" / "results" / "combined_estimates.csv")
    refs = refs[refs.Country == "US"][
        ["Year", "Month", "Monthly_RD_Expenditure_Tempdisagg_Sax", "Monthly_RD_Expenditure_Tempdisagg_Mosley"]
    ].copy()
    refs.rename(
        columns={
            "Monthly_RD_Expenditure_Tempdisagg_Sax": "Sax",
            "Monthly_RD_Expenditure_Tempdisagg_Mosley": "Mosley",
        },
        inplace=True,
    )

    emp = pd.read_csv(CODE / "data" / "datausa.io" / "Monthly Employment.csv")
    emp["date"] = pd.to_datetime(emp["Date"])
    emp = emp[["date", "NSA Employees"]].rename(columns={"NSA Employees": "emp"})
    return merged, agt_cols, topics, gt, rd_us, refs, emp


def make_split(df: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    tr = np.zeros(len(df), dtype=bool)
    va = np.zeros(len(df), dtype=bool)
    countries = df.Country.values
    if kind == "random":
        for country in np.unique(countries):
            idx = np.where(countries == country)[0]
            rng.shuffle(idx)
            n = len(idx)
            tr[idx[: int(0.64 * n)]] = True
            va[idx[int(0.64 * n) : int(0.80 * n)]] = True
    elif kind == "temporal":
        for country in np.unique(countries):
            years = np.array(sorted(df[df.Country == country].Year.unique()))
            n = len(years)
            train_years = set(years[: int(round(0.64 * n))])
            val_years = set(years[int(round(0.64 * n)) : int(round(0.80 * n))])
            mask = countries == country
            tr |= mask & df.Year.isin(train_years).values
            va |= mask & df.Year.isin(val_years).values
    elif kind == "alldata":
        va = rng.random(len(df)) < 0.15
        tr = ~va
    else:
        raise ValueError(f"unknown split kind: {kind}")
    return tr, va


def annual_mape(frame: pd.DataFrame, mask: np.ndarray, pred: np.ndarray) -> float:
    tmp = frame.loc[mask, ["Country", "Year", "rd_expenditure"]].copy()
    tmp["pred"] = pred
    ann = tmp.groupby(["Country", "Year"], as_index=False).agg(actual=("rd_expenditure", "mean"), pred=("pred", "mean"))
    return float(np.mean(np.abs((ann.actual - ann.pred) / ann.actual)) * 100)


def train_split_models(df: pd.DataFrame, agt_cols: list[str], kind: str) -> tuple[list[nn.Module], np.ndarray, np.ndarray, LabelEncoder, float, float, np.ndarray, int, np.ndarray]:
    train, val = make_split(df, kind)
    y = df.rd_expenditure.values.astype(float)
    y_mean = float(y[train].mean())
    y_sd = float(y[train].std())
    y_std = (y - y_mean) / y_sd

    months = pd.get_dummies(df.Month.astype(int), prefix="M").astype(float)
    scaler = StandardScaler().fit(df[agt_cols].fillna(0).astype(float).values[train])
    X_gt = scaler.transform(df[agt_cols].fillna(0).astype(float).values)
    X = np.column_stack([X_gt, months.values])
    n_gt = len(agt_cols)

    le = LabelEncoder()
    country_codes = le.fit_transform(df.Country)
    Xtr, Xva = tensor(X[train]), tensor(X[val])
    ctr, cva = torch.LongTensor(country_codes[train]), torch.LongTensor(country_codes[val])
    ytr = tensor(y_std[train].reshape(-1, 1))
    crit = nn.SmoothL1Loss(beta=0.5)
    models: list[nn.Module] = []
    histories = []

    def to_level(z: np.ndarray) -> np.ndarray:
        return z * y_sd + y_mean

    for seed in range(ENSEMBLE_SIZE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = WideDeepLevelNN(X.shape[1], len(le.classes_))
        opt = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best = np.inf
        best_state = None
        bad = 0
        for epoch in range(500):
            net.train()
            perm = torch.randperm(len(Xtr))
            for start in range(0, len(Xtr), 64):
                idx = perm[start : start + 64]
                opt.zero_grad()
                loss = crit(net(Xtr[idx], ctr[idx]), ytr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                val_pred = to_level(net(Xva, cva).numpy().ravel())
            score = annual_mape(df, val, val_pred)
            histories.append({"split": kind, "seed": seed, "epoch": epoch + 1, "val_MAPE": score})
            if score < best - 1e-5:
                best = score
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 65:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        models.append(net)
        print(f"{kind}: seed {seed + 1}/{ENSEMBLE_SIZE}, best val MAPE={best:.2f}", flush=True)
    pd.DataFrame(histories).to_csv(ROBUST_OUT / f"updated_stepB_{kind}_history.csv", index=False)
    return models, X, country_codes, le, y_mean, y_sd, train, n_gt, X_gt


def compute_elasticities(
    models: list[nn.Module],
    X: np.ndarray,
    country_codes: np.ndarray,
    y_mean: float,
    y_sd: float,
    n_gt: int,
) -> np.ndarray:
    call = torch.LongTensor(country_codes)
    Xall = tensor(X)
    out = np.zeros((X.shape[0], n_gt), dtype=float)
    for model in models:
        model.eval()
        with torch.no_grad():
            base = model(Xall, call).numpy().ravel() * y_sd + y_mean
        denom = np.where(np.abs(base) > 1e-12, base, np.nan)
        for j in range(n_gt):
            Xp = X.copy()
            Xp[:, j] *= 1.01
            with torch.no_grad():
                pert = model(tensor(Xp), call).numpy().ravel() * y_sd + y_mean
            out[:, j] += ((pert - base) / denom) / 0.01
    return out / len(models)


def topic_elasticities(df: pd.DataFrame, agt_cols: list[str], topics: list[str], elasticities: np.ndarray, train: np.ndarray) -> dict[str, float]:
    eldf = pd.DataFrame(elasticities, columns=agt_cols)
    eldf["Country"] = df.Country.values
    eldf["train"] = train
    eta = eldf[eldf.train].groupby("Country")[agt_cols].mean()
    eta_us = {}
    for topic in topics:
        lag_cols = [f"{topic}_yearly_avg_lag{lag}" for lag in (1, 2, 3) if f"{topic}_yearly_avg_lag{lag}" in eta.columns]
        eta_us[topic] = float(np.nanmean(eta.loc["US", lag_cols].values)) if lag_cols else np.nan
    return eta_us


def disaggregate_us(eta_us: dict[str, float], topics: list[str], gt: pd.DataFrame, rd_us: pd.Series, refs: pd.DataFrame) -> pd.DataFrame:
    tp_ok = [topic for topic in topics if f"US_{topic}" in gt.columns and np.isfinite(eta_us.get(topic, np.nan))]
    g = gt[["Year", "Month"] + [f"US_{topic}" for topic in tp_ok]].copy()
    g = g[g.Year.isin([year for year in g.Year.unique() if year in rd_us.index])].copy()
    adj = np.zeros(len(g), dtype=float)
    for topic in tp_ok:
        col = f"US_{topic}"
        den = g.groupby("Year")[col].transform("sum").values
        share = np.where(den > 0, g[col].values / np.where(den > 0, den, 1.0), 0.0)
        adj += float(eta_us[topic]) * share
    g["adj"] = adj
    g["annual_adj"] = g.groupby("Year")["adj"].transform("sum")
    g["NN"] = g.Year.map(rd_us) * g["adj"] / g["annual_adj"]
    out = g[["Year", "Month", "NN"]].merge(refs, on=["Year", "Month"], how="inner").dropna().reset_index(drop=True)
    out["date"] = pd.to_datetime(dict(year=out.Year, month=out.Month, day=1))
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(a[mask], b[mask])
    return float(r), float(p)


def growth(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.diff(x) / x[:-1]


def method_agreement(df: pd.DataFrame) -> dict[str, float]:
    nn = df.NN.values
    mos = df.Mosley.values
    sax = df.Sax.values
    return {
        "N": int(len(df)),
        "Mosley_lvl": pearson(nn, mos)[0],
        "Mosley_gr": pearson(growth(nn), growth(mos))[0],
        "Mosley_gr_p": pearson(growth(nn), growth(mos))[1],
        "Sax_lvl": pearson(nn, sax)[0],
        "Sax_gr": pearson(growth(nn), growth(sax))[0],
        "Sax_gr_p": pearson(growth(nn), growth(sax))[1],
    }


def lag_correlations(series: np.ndarray, emp: np.ndarray, lags: range) -> list[dict[str, float]]:
    sg = growth(series)
    eg = growth(emp)
    rows = []
    for lag in lags:
        if lag < 0:
            a, b = sg[-lag:], eg[:lag]
        elif lag > 0:
            a, b = sg[:-lag], eg[lag:]
        else:
            a, b = sg, eg
        r, p = pearson(a, b)
        rows.append({"Lag": int(lag), "Correlation": r, "P_Value": p, "N": int(len(a))})
    return rows


def ccf_outputs(prod_df: pd.DataFrame, emp: pd.DataFrame) -> None:
    merged = prod_df.merge(emp, on="date", how="inner").sort_values("date").reset_index(drop=True)
    methods = {"NN-elasticity": "NN", "Chow-Lin (Sax)": "Sax", "Sparse (Mosley)": "Mosley"}
    for label, col in methods.items():
        merged[label + "_g"] = merged[col].pct_change()
    merged["emp_g"] = merged.emp.pct_change()
    cdf = merged.dropna().reset_index(drop=True)
    lags = range(-12, 13)
    band = 1.96 / np.sqrt(len(cdf))
    colors = {"NN-elasticity": "#2c7fb8", "Chow-Lin (Sax)": "#d95f0e", "Sparse (Mosley)": "#31a354"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary = []
    long_rows = []
    for label in methods:
        vals = {}
        for lag in lags:
            x = cdf[label + "_g"].values
            y = cdf.emp_g.values
            if lag < 0:
                a, b = x[-lag:], y[:lag]
            elif lag > 0:
                a, b = x[:-lag], y[lag:]
            else:
                a, b = x, y
            r, p = pearson(a, b)
            vals[lag] = r
            long_rows.append({"Method": label, "Lag": lag, "Correlation": r, "P_Value": p})
        ax.plot(list(vals.keys()), list(vals.values()), marker="o", ms=3, label=label, color=colors[label])
        peak = max(vals, key=lambda key: abs(vals[key]) if np.isfinite(vals[key]) else -np.inf)
        summary.append(
            {
                "Method": label,
                "Peak |corr|": abs(vals[peak]),
                "Peak corr": vals[peak],
                "Peak lag (months)": peak,
                "corr at lag0": vals[0],
                "# sig lags (of 25)": int(sum(abs(v) > band for v in vals.values() if np.isfinite(v))),
                "N": int(len(cdf)),
                "band": band,
            }
        )
    ax.axhspan(-band, band, color="grey", alpha=0.2, label="95% band")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Lag (months); negative = R&D expenditure leads employment")
    ax.set_ylabel("Cross-correlation of growth rates")
    ax.set_title("R&D expenditure estimates vs R&D-services employment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SOURCE_OUT / "disagg_ccf.png", dpi=180)
    fig.savefig(FIG_REV / "disagg_ccf.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(summary).to_csv(SOURCE_OUT / "disagg_ccf_summary.csv", index=False)
    pd.DataFrame(long_rows).to_csv(SOURCE_OUT / "disagg_ccf_full.csv", index=False)
    level_corr = prod_df[["NN", "Sax", "Mosley"]].corr()
    growth_corr = prod_df[["NN", "Sax", "Mosley"]].pct_change().dropna().corr()
    level_corr.to_csv(SOURCE_OUT / "method_agreement_levels.csv")
    growth_corr.to_csv(SOURCE_OUT / "method_agreement_growth.csv")


def employment_outputs(prod_df: pd.DataFrame, emp: pd.DataFrame) -> pd.DataFrame:
    merged = prod_df.merge(emp, on="date", how="inner").sort_values("date").reset_index(drop=True)
    rows = []
    for label, col in [
        ("Neural Network-driven elasticity-based estimates", "NN"),
        ("\\citet{mosley2022sparse}-based estimates", "Mosley"),
        ("\\citet{chowlin1971best}-based estimates", "Sax"),
    ]:
        for row in lag_correlations(merged[col].values, merged.emp.values, range(-5, 6)):
            row["Method"] = label
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_OUT / "stepB_employment_lags_all_methods.csv", index=False)
    out[out.Method.str.startswith("Neural")].drop(columns=["Method"]).to_csv(
        SOURCE_OUT / "stepB_employment_lags.csv", index=False
    )
    return out


def write_employment_table(all_lags: pd.DataFrame) -> None:
    sig = all_lags[(all_lags.P_Value < 0.01)].copy()
    sig["abs_corr"] = sig.Correlation.abs()
    order = [
        "Neural Network-driven elasticity-based estimates",
        "\\citet{mosley2022sparse}-based estimates",
        "\\citet{chowlin1971best}-based estimates",
    ]
    rows = [
        "% Table source code: additional_analysis/robustness_overfit/12_refresh_stepB_updated_agt.py",
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Correlation between Monthly R\\&D Expenditure and R\\&D Services Employees Growths}",
        "\\label{tab:correlation_employees}",
        "\\begin{tabular}{l r r r}",
        "\\toprule",
        "\\textbf{Monthly R\\&D Expenditure Estimates Growth} & \\textbf{Lag} & \\textbf{Correlation} & \\textbf{P-Value} \\\\",
        "\\midrule",
    ]
    for mi, method in enumerate(order):
        sub = sig[sig.Method == method].sort_values("abs_corr", ascending=False)
        if len(sub) == 0:
            rows.append(f"\\multirow{{1}}{{*}}{{{method}}} & -- & -- & -- \\\\")
        else:
            for i, (_, r) in enumerate(sub.iterrows()):
                label = f"\\multirow{{{len(sub)}}}{{*}}{{{method}}}" if i == 0 else ""
                ptxt = "<0.0001" if r.P_Value < 1e-4 else f"{r.P_Value:.4f}"
                rows.append(f"{label} & {int(r.Lag)} & {r.Correlation:.2f} & {ptxt} \\\\ ")
        if mi < len(order) - 1:
            rows.append("\\midrule")
    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    marker = "\\begin{table}[h!]\n\\centering\n\\caption{Correlation between Monthly R\\&D Expenditure and R\\&D Services Employees Growths}"
    tex = (PAPER / "template-arxiv.tex").read_text()
    start = tex.find("% Table source code: temporal_disaggregation/temp_disagg_analysis_employee.py")
    if start == -1:
        start = tex.find(marker)
    end = tex.find("\\end{table}", start)
    if start == -1 or end == -1:
        raise RuntimeError("could not locate employment correlation table in template-arxiv.tex")
    end += len("\\end{table}")
    (PAPER / "template-arxiv.tex").write_text(tex[:start] + "\n".join(rows) + tex[end:])


def plot_stepb(prod_df: pd.DataFrame, emp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    minv = prod_df[["NN", "Sax", "Mosley"]].min(axis=1)
    maxv = prod_df[["NN", "Sax", "Mosley"]].max(axis=1)
    ax.fill_between(prod_df.date, minv, maxv, color="gray", alpha=0.28, label="Range across methods")
    ax.plot(prod_df.date, prod_df.NN, color="#2c6fbb", marker="o", ms=3, lw=1.2, label="NN-elasticity")
    ax.set_ylabel("Monthly R&D (USD bn)")
    ax.set_title("NN-driven elasticity-based estimate (AGT)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DISAGG / "NNelasticity_temporal_level.png", dpi=220)
    plt.close(fig)

    merged = prod_df.merge(emp, on="date", how="inner")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(merged.date, merged.NN, color="blue", label="NN-elasticity R&D")
    ax1.set_ylabel("Monthly R&D (USD bn)", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(merged.date, merged.emp, color="red", alpha=0.7, label="Employment")
    ax2.set_ylabel("Employees", color="red")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("US monthly R&D (AGT NN) vs R&D-services employment")
    fig.tight_layout()
    fig.savefig(FIG_DISAGG / "employee_temporal_level.png", dpi=220)
    plt.close(fig)


def main() -> None:
    df, agt_cols, topics, gt, rd_us, refs, emp = load_inputs()
    all_corr = []
    all_top = []
    prod_df = None
    for split in ["temporal", "random", "alldata"]:
        print(f"=== updated Step B AGT split: {split} ===", flush=True)
        models, X, cc, _le, y_mean, y_sd, train, n_gt, _ = train_split_models(df, agt_cols, split)
        elasticities = compute_elasticities(models, X, cc, y_mean, y_sd, n_gt)
        eta_us = topic_elasticities(df, agt_cols, topics, elasticities, train)
        stepb = disaggregate_us(eta_us, topics, gt, rd_us, refs)
        corr = method_agreement(stepb)
        corr["split"] = split
        all_corr.append(corr)
        top = sorted(eta_us.items(), key=lambda item: -abs(item[1]) if np.isfinite(item[1]) else -np.inf)
        for rank, (topic, value) in enumerate(top[:15], 1):
            all_top.append({"split": split, "rank": rank, "topic": topic, "abs_elasticity": abs(value), "elasticity": value})
        pd.DataFrame({"topic": list(eta_us.keys()), "elasticity": list(eta_us.values())}).to_csv(
            ROBUST_OUT / f"updated_stepB_{split}_topic_elasticities.csv", index=False
        )
        stepb.to_csv(ROBUST_OUT / f"updated_stepB_{split}_monthly_estimates.csv", index=False)
        print(
            f"{split}: NN-Mosley level={corr['Mosley_lvl']:.3f}, growth={corr['Mosley_gr']:.3f}; "
            f"NN-Sax level={corr['Sax_lvl']:.3f}, growth={corr['Sax_gr']:.3f}",
            flush=True,
        )
        print("top topics:", [topic for topic, _ in top[:8]], flush=True)
        if split == "temporal":
            prod_df = stepb.copy()

    assert prod_df is not None
    prod_df.to_csv(SOURCE_OUT / "combined_estimates_temporal_level.csv", index=False)
    prod_df.to_csv(ROBUST_OUT / "combined_estimates_temporal_level_updated_agt.csv", index=False)
    corr_df = pd.DataFrame(all_corr)[["split", "N", "Mosley_lvl", "Mosley_gr", "Mosley_gr_p", "Sax_lvl", "Sax_gr", "Sax_gr_p"]]
    corr_df.to_csv(SOURCE_OUT / "stepB_correlations.csv", index=False)
    corr_df.to_csv(ROBUST_OUT / "stepB_correlations_updated_agt.csv", index=False)
    top_df = pd.DataFrame(all_top)
    top_df.to_csv(SOURCE_OUT / "stepB_top_topics.csv", index=False)
    top_df.to_csv(ROBUST_OUT / "stepB_top_topics_updated_agt.csv", index=False)
    all_lags = employment_outputs(prod_df, emp)
    all_lags.to_csv(ROBUST_OUT / "stepB_employment_lags_all_methods_updated_agt.csv", index=False)
    ccf_outputs(prod_df, emp)
    plot_stepb(prod_df, emp)
    write_employment_table(all_lags)

    print("\n=== Updated Step B correlations ===")
    print(corr_df.to_string(index=False))
    print("\n=== Updated temporal top-6 topics ===")
    print(top_df[(top_df.split == "temporal") & (top_df["rank"] <= 6)].to_string(index=False))
    print("\n=== Updated significant employment lags (p<0.01) ===")
    sig = all_lags[all_lags.P_Value < 0.01].copy()
    sig["abs_corr"] = sig.Correlation.abs()
    print(sig.sort_values(["Method", "abs_corr"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved audit outputs to {ROBUST_OUT}")


if __name__ == "__main__":
    main()
