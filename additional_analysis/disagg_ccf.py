"""
Strengthen the Step-B (temporal disaggregation) validation.

The paper currently validates the monthly R&D series against monthly employment in
Scientific R&D Services by reporting a handful of individually significant lags
(Table 3). Referees will want (i) the full cross-correlation function (CCF) with
significance bands, and (ii) a like-for-like comparison across the three methods.

This script:
  - merges the three monthly estimates (NN-elasticity, Chow-Lin/Sax, sparse/Mosley)
    with US R&D-services employment,
  - computes month-on-month growth rates,
  - estimates the full CCF (lags -12..12) with +-1.96/sqrt(N) bands,
  - reports the peak |correlation|, its lag, and the share of significant lags,
  - saves a CCF figure and a tidy results table.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out")
os.makedirs(OUT, exist_ok=True)

est = pd.read_csv(os.path.join(ROOT, "temporal_disaggregation", "results", "combined_estimates.csv"))
est = est[est.Country == "US"].copy()
est["date"] = pd.to_datetime(dict(year=est.Year, month=est.Month, day=1))

emp = pd.read_csv(os.path.join(ROOT, "data", "datausa.io", "Monthly Employment.csv"))
emp["date"] = pd.to_datetime(emp["Date"])
emp = emp[["date", "NSA Employees"]].rename(columns={"NSA Employees": "employees"})

df = est.merge(emp, on="date", how="inner").sort_values("date").reset_index(drop=True)

methods = {
    "NN-elasticity": "Monthly_RD_Expenditure_Tempdisagg_GT_NNelasticity",
    "Chow-Lin (Sax)": "Monthly_RD_Expenditure_Tempdisagg_Sax",
    "Sparse (Mosley)": "Monthly_RD_Expenditure_Tempdisagg_Mosley",
}

# month-on-month growth rates
for m, col in methods.items():
    df[m + "_g"] = df[col].pct_change()
df["emp_g"] = df["employees"].pct_change()
df = df.dropna().reset_index(drop=True)
N = len(df)
band = 1.96 / np.sqrt(N)
print(f"Merged monthly sample: {df.date.min():%Y-%m} to {df.date.max():%Y-%m}, N={N}")
print(f"95% significance band: +-{band:.3f}")

lags = range(-12, 13)
def ccf(x, y, lags):
    out = {}
    for k in lags:
        if k < 0:
            a, b = x[-k:], y[:k]            # x leads y by |k|
        elif k > 0:
            a, b = x[:-k], y[k:]
        else:
            a, b = x, y
        if len(a) > 3:
            out[k] = np.corrcoef(a, b)[0, 1]
    return out

fig, ax = plt.subplots(figsize=(9, 4.5))
summary = []
colors = {"NN-elasticity": "#2c7fb8", "Chow-Lin (Sax)": "#d95f0e", "Sparse (Mosley)": "#31a354"}
for m in methods:
    c = ccf(df[m + "_g"].values, df["emp_g"].values, lags)
    ks = list(c.keys()); vs = list(c.values())
    ax.plot(ks, vs, marker="o", ms=3, label=m, color=colors[m])
    peak_lag = max(c, key=lambda k: abs(c[k]))
    nsig = sum(abs(v) > band for v in vs)
    summary.append({"Method": m, "Peak |corr|": abs(c[peak_lag]),
                    "Peak corr": c[peak_lag], "Peak lag (months)": peak_lag,
                    "corr at lag0": c[0], "# sig lags (of 25)": nsig})

ax.axhspan(-band, band, color="grey", alpha=0.2, label="95% band")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Lag (months); negative = R&D expenditure leads employment")
ax.set_ylabel("Cross-correlation of growth rates")
ax.set_title("R&D expenditure (monthly estimate) vs R&D-services employment")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "disagg_ccf.png"), dpi=150)

sm = pd.DataFrame(summary).set_index("Method")
print("\n=== CCF summary vs R&D employment growth ===")
print(sm.round(3).to_string())
sm.round(3).to_csv(os.path.join(OUT, "disagg_ccf_summary.csv"))

# Also: agreement between methods (levels & growth), generalising the paper's r-values
lvl = pd.DataFrame({m: df[methods[m]] for m in methods})
grw = pd.DataFrame({m: df[m + "_g"] for m in methods})
print("\nPairwise correlation in LEVELS:\n", lvl.corr().round(2).to_string())
print("\nPairwise correlation in GROWTH:\n", grw.corr().round(2).to_string())
lvl.corr().round(3).to_csv(os.path.join(OUT, "method_agreement_levels.csv"))
grw.corr().round(3).to_csv(os.path.join(OUT, "method_agreement_growth.csv"))
print("\nSaved -> out/disagg_ccf.png, disagg_ccf_summary.csv, method_agreement_*.csv")
