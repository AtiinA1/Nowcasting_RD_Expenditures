"""
Additional analysis for the R&D nowcasting paper (Step A).

Goal: strengthen the predictive-accuracy evidence to the standard expected by
JBES / Oxford Bulletin referees, by adding:
  (1) Naive / econometric benchmarks (random walk, AR(1), historical mean, drift RW)
  (2) Forecast-skill scores: RMSE, MAE, MAPE, MASE, and out-of-sample R^2
      (relative to the random-walk benchmark)
  (3) Diebold-Mariano (1995) tests with the Harvey-Leybourne-Newbold (1997)
      small-sample correction, on the SHARED test set used by all 7 NN configs.

All seven NN configurations were trained/evaluated on an identical test set
(verified: 314 identical (country, year, month) test rows), which permits paired
forecast-comparison tests across configurations and against benchmarks.

Inputs : nn_mlp_nowcasting_model/results/<CONFIG>/df_combined_pred_vs_true.csv
Outputs: additional_analysis/out/*.csv, *.tex, *.pdf
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
RES = os.path.join(ROOT, "nn_mlp_nowcasting_model", "results")
OUT = os.path.join(ROOT, "additional_analysis", "out")
os.makedirs(OUT, exist_ok=True)

CONFIGS = ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]

# ---------------------------------------------------------------------------
# Load predictions (identical test set across configs)
# ---------------------------------------------------------------------------
def load_config(cfg):
    df = pd.read_csv(os.path.join(RES, cfg, "df_combined_pred_vs_true.csv"))
    df = df.rename(columns={"Predicted_Values": cfg})
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    return df[["Year", "Month", "Country", "True_Values", "Type", cfg]]

base = load_config(CONFIGS[0])
merged = base.copy()
for cfg in CONFIGS[1:]:
    d = load_config(cfg)[["Year", "Month", "Country", cfg]]
    merged = merged.merge(d, on=["Year", "Month", "Country"], how="inner")

# Annual GERD panel (target is annual, constant within year)
panel = (merged.groupby(["Country", "Year"])["True_Values"].mean()
         .reset_index().rename(columns={"True_Values": "GERD"}))

# Non-test (train+val) years per country -> used to fit benchmarks (no leakage
# of test-year level into benchmark estimation beyond the realised t-1 value).
nontest = (merged[merged.Type != "Test"]
           .groupby(["Country", "Year"])["True_Values"].mean().reset_index())
train_years = {c: set(g.Year) for c, g in nontest.groupby("Country")}

# ---------------------------------------------------------------------------
# Benchmarks at the (country, year) level
# ---------------------------------------------------------------------------
def prev_obs(country, year):
    """Most recent observed annual GERD strictly before `year`."""
    g = panel[(panel.Country == country) & (panel.Year < year)].sort_values("Year")
    return g.GERD.iloc[-1] if len(g) else np.nan

def hist_mean(country, year):
    yrs = [y for y in train_years.get(country, set()) if y < year]
    vals = panel[(panel.Country == country) & (panel.Year.isin(yrs))].GERD
    return vals.mean() if len(vals) else np.nan

def drift_rw(country, year):
    """Random walk with drift: last obs grown by avg historical growth."""
    g = panel[(panel.Country == country) & (panel.Year < year)].sort_values("Year")
    if len(g) < 2:
        return prev_obs(country, year)
    growth = g.GERD.pct_change().mean()
    return g.GERD.iloc[-1] * (1 + growth)

def ar1(country, year):
    """One-step AR(1) on log GERD, params from pre-test years, applied to actual t-1."""
    yrs = sorted([y for y in train_years.get(country, set())])
    s = panel[(panel.Country == country) & (panel.Year.isin(yrs))].sort_values("Year")
    s = s.dropna()
    if len(s) < 4:
        return prev_obs(country, year)
    y = np.log(s.GERD.values)
    y0, y1 = y[:-1], y[1:]
    X = np.column_stack([np.ones_like(y0), y0])
    try:
        beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
    except Exception:
        return prev_obs(country, year)
    prev = prev_obs(country, year)
    if not np.isfinite(prev) or prev <= 0:
        return np.nan
    return float(np.exp(beta[0] + beta[1] * np.log(prev)))

BENCH = {"RW": prev_obs, "RW_drift": drift_rw, "AR1": ar1, "HistMean": hist_mean}

# Build per-row predictions on the test set
test = merged[merged.Type == "Test"].copy()
for name, fn in BENCH.items():
    test[name] = [fn(c, y) for c, y in zip(test.Country, test.Year)]

# ---------------------------------------------------------------------------
# Metrics (pooled over the 314 test rows)
# ---------------------------------------------------------------------------
def metrics(true, pred):
    true = np.asarray(true, float); pred = np.asarray(pred, float)
    m = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[m], pred[m]
    err = true - pred
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    mape = np.mean(np.abs(err / true)) * 100
    return rmse, mae, mape, m.sum()

# MASE scaling: in-sample naive (RW) MAE on the annual series per country
def mase_scale():
    diffs = []
    for c, g in panel.groupby("Country"):
        g = g.sort_values("Year").dropna()
        diffs.append(np.abs(np.diff(g.GERD.values)))
    return np.mean(np.concatenate(diffs))

scale = mase_scale()

rows = []
all_preds = {**{c: test[c] for c in CONFIGS}, **{b: test[b] for b in BENCH}}
for name, pred in all_preds.items():
    rmse, mae, mape, n = metrics(test.True_Values, pred)
    rows.append({"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape,
                 "MASE": mae / scale, "N": n})
res = pd.DataFrame(rows).set_index("Model")

# Out-of-sample R^2 relative to RW benchmark (a la Campbell-Thompson)
mse_rw = metrics(test.True_Values, test["RW"])[0] ** 2
res["OOS_R2_vs_RW"] = 1 - (res["RMSE"] ** 2) / mse_rw
res = res.reindex(CONFIGS + list(BENCH))
res.to_csv(os.path.join(OUT, "benchmark_skill_scores.csv"))
print("=== Forecast skill scores (pooled test set, n=314) ===")
print(res.round(3).to_string())

# ---------------------------------------------------------------------------
# Diebold-Mariano tests on annual-collapsed test points
# ---------------------------------------------------------------------------
# Collapse NN preds to annual mean per (country, year); benchmarks already annual.
ann = test.groupby(["Country", "Year"]).agg(
    True_Values=("True_Values", "mean"),
    **{c: (c, "mean") for c in CONFIGS},
    **{b: (b, "mean") for b in BENCH},
).reset_index()

def dm_test(true, p1, p2, h=1, power=2):
    """Diebold-Mariano with HLN small-sample correction. Positive stat => p1 worse."""
    true = np.asarray(true, float); p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    m = np.isfinite(true) & np.isfinite(p1) & np.isfinite(p2)
    true, p1, p2 = true[m], p1[m], p2[m]
    e1, e2 = true - p1, true - p2
    if power == 2:
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)
    n = len(d)
    if n < 5:
        return np.nan, np.nan, n
    dbar = d.mean()
    # autocovariances up to h-1
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, h):
        cov = np.cov(d[k:], d[:-k])[0, 1]
        var += 2 * cov
    var_dbar = var / n
    if var_dbar <= 0:
        return np.nan, np.nan, n
    dm = dbar / np.sqrt(var_dbar)
    # Harvey-Leybourne-Newbold correction
    corr = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_hln = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return dm_hln, p, n

# (a) Each model vs RW
print("\n=== DM test vs Random Walk (squared loss; negative stat => model better) ===")
dm_rows = []
for cfg in CONFIGS + ["AR1", "RW_drift", "HistMean"]:
    stat, p, n = dm_test(ann.True_Values, ann[cfg], ann["RW"], power=2)
    dm_rows.append({"Model": cfg, "DM_vs_RW": stat, "p_value": p, "n": n})
dm_rw = pd.DataFrame(dm_rows).set_index("Model")
print(dm_rw.round(3).to_string())
dm_rw.to_csv(os.path.join(OUT, "dm_vs_rw.csv"))

# (b) Pairwise DM matrix between the 7 NN configs (squared loss)
print("\n=== Pairwise DM matrix (row vs col; negative => row better), p-values below ===")
mat = pd.DataFrame(index=CONFIGS, columns=CONFIGS, dtype=float)
pmat = pd.DataFrame(index=CONFIGS, columns=CONFIGS, dtype=float)
for a in CONFIGS:
    for b in CONFIGS:
        if a == b:
            mat.loc[a, b] = np.nan; pmat.loc[a, b] = np.nan; continue
        stat, p, n = dm_test(ann.True_Values, ann[a], ann[b], power=2)
        mat.loc[a, b] = stat; pmat.loc[a, b] = p
print("DM stats:\n", mat.round(2).to_string())
print("\np-values:\n", pmat.round(3).to_string())
mat.to_csv(os.path.join(OUT, "dm_matrix_stat.csv"))
pmat.to_csv(os.path.join(OUT, "dm_matrix_pval.csv"))

# ---------------------------------------------------------------------------
# Figure: skill-score bar chart + DM-vs-RW
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
order = CONFIGS + list(BENCH)
colors = ["#2c7fb8"]*len(CONFIGS) + ["#d95f0e"]*len(BENCH)
axes[0].bar(order, res.loc[order, "MAPE"], color=colors)
axes[0].set_ylabel("MAPE (%)"); axes[0].set_title("Pooled out-of-sample MAPE")
axes[0].tick_params(axis="x", rotation=45)
axes[1].bar(res.index[:len(CONFIGS)], res["OOS_R2_vs_RW"][:len(CONFIGS)], color="#2c7fb8")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("Out-of-sample $R^2$ vs RW"); axes[1].set_title("Skill relative to random walk")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "skill_scores.png"), dpi=150)
print("\nSaved figure -> out/skill_scores.png")

# LaTeX table of skill scores
latex = res.round(2).to_latex(
    columns=["MAPE", "RMSE", "MASE", "OOS_R2_vs_RW"],
    caption="Out-of-sample forecast accuracy and skill scores on the shared test set "
            "(pooled, $n=314$). MASE and out-of-sample $R^2$ are computed relative to a "
            "no-change random walk; positive $R^2$ indicates the model beats the random walk.",
    label="tab:skill_scores", float_format="%.2f")
with open(os.path.join(OUT, "skill_scores.tex"), "w") as f:
    f.write(latex)
print("Saved LaTeX -> out/skill_scores.tex")
