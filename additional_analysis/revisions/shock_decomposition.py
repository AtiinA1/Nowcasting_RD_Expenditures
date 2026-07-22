"""
Fix 1 (turning-point / shock analysis): the random walk's dominance is an
average over a smooth sample and is NOT uniform. Decompose test-set error by
the local curvature (acceleration) of the GERD series, and isolate 2020 (COVID).

Inputs (read-only):
  robustness_overfit/out/paper_stepA_refresh/updated_temporal_annual_all.csv
      -- refreshed test-set true GERD + every benchmark
  out/merged_features.csv              -- full per-country annual GERD series (bn)
Outputs:
  revisions/out/shock_decomposition.csv
  revisions/figs/shock_decomposition.png
Prints the numbers needed for the draft.
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "out"); os.makedirs(OUTDIR, exist_ok=True)
FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
PAPER_FIGDIR = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Revision"
os.makedirs(PAPER_FIGDIR, exist_ok=True)
SRC = os.path.normpath(os.path.join(HERE, "..", "out"))
UPDATED = os.path.normpath(os.path.join(
    HERE, "..", "robustness_overfit", "out", "paper_stepA_refresh"
))

pred = pd.read_csv(os.path.join(UPDATED, "updated_temporal_annual_all.csv"))

# Full per-country annual GERD series (bn) from merged features (monthly rows -> unique year)
mf = pd.read_csv(os.path.join(SRC, "merged_features.csv"))
series = (mf[["Country", "Year", "rd_expenditure"]]
          .drop_duplicates(subset=["Country", "Year"])
          .sort_values(["Country", "Year"]))
series = series.rename(columns={"rd_expenditure": "GERD_full"})

def gerd(country, year):
    row = series[(series.Country == country) & (series.Year == year)]
    return float(row.GERD_full.iloc[0]) if len(row) else np.nan

rows = []
for _, r in pred.iterrows():
    c, t, y = r.Country, int(r.Year), r.GERD
    y1, y2, y3 = gerd(c, t - 1), gerd(c, t - 2), gerd(c, t - 3)
    g_t  = np.log(y / y1)   if (y1 and y1 > 0) else np.nan          # current log growth
    g_t1 = np.log(y1 / y2)  if (y2 and y2 > 0) else np.nan          # prior log growth
    acc  = abs(g_t - g_t1)                                          # |acceleration|
    def ape(pred_val):
        return abs(pred_val - y) / y * 100.0
    rows.append(dict(
        Country=c, Year=t, GERD=y,
        accel=acc, growth=g_t,
        ape_AGT=ape(r.NN_AGT), ape_AllVar=ape(r.NN_AllVar),
        ape_RW1=ape(y1) if y1 else np.nan,        # RW(L=1): previous-year actual
        ape_RW3=ape(y3) if y3 else np.nan,        # RW(L=3): publication-lag-consistent
        ape_AR1=ape(r.AR1),
        is2020=(t == 2020),
    ))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTDIR, "shock_decomposition.csv"), index=False)

valid = df.dropna(subset=["accel"]).copy()
# Split smooth vs shock by acceleration tercile
hi = valid.accel.quantile(2/3)
valid["regime"] = np.where(valid.accel >= hi, "shock", "smooth")

print("=== N test country-years with acceleration defined:", len(valid))
print("\n=== Mean absolute % error by regime (acceleration terciles) ===")
print(valid.groupby("regime")[["ape_AGT", "ape_AllVar", "ape_RW1", "ape_RW3", "ape_AR1"]].mean().round(2))

print("\n=== 2020 (COVID) test country-years vs the rest ===")
print(df.groupby("is2020")[["ape_AGT", "ape_AllVar", "ape_RW1", "ape_RW3", "ape_AR1"]].mean().round(2))
print("2020 countries:", sorted(df[df.is2020].Country.tolist()))

# Gap = nowcast error minus RW(L=1) error; positive => RW better. Correlate with acceleration.
valid["gap_AGT_RW1"] = valid.ape_AGT - valid.ape_RW1
valid["gap_AGT_RW3"] = valid.ape_AGT - valid.ape_RW3
from scipy.stats import pearsonr
for col, lab in [("gap_AGT_RW1", "AGT - RW(L=1)"), ("gap_AGT_RW3", "AGT - RW(L=3)")]:
    sub = valid.dropna(subset=[col])
    rr, pp = pearsonr(sub.accel, sub[col])
    print(f"\ncorr(acceleration, {lab} error gap): r={rr:.2f}, p={pp:.3f}  (negative => nowcast gains as curvature rises)")

# ---- Figure: error vs acceleration, RW(L=1) and RW(L=3) vs NN nowcast ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
v = valid.sort_values("accel")
ax[0].scatter(v.accel, v.ape_RW1, s=42, marker="s", color="#BA7517", label="RW (L=1, oracle)", zorder=3)
ax[0].scatter(v.accel, v.ape_AGT, s=42, marker="o", color="#185FA5", label="NN (AGT)", zorder=3)
ax[0].set_xlabel("|acceleration| of log GERD  (turning-point intensity)")
ax[0].set_ylabel("absolute % error")
ax[0].set_title("(a) Absolute error vs. turning-point intensity")
ax[0].set_ylim(0, 50)
ax[0].legend(frameon=False, fontsize=9)
ax[0].grid(alpha=0.25)

# grouped bars: smooth vs shock, three predictors
g = valid.groupby("regime")[["ape_RW1", "ape_RW3", "ape_AGT"]].mean().reindex(["smooth", "shock"])
x = np.arange(2); w = 0.26
ax[1].bar(x - w, g.ape_RW1, w, color="#BA7517", label="RW (L=1, oracle)")
ax[1].bar(x,     g.ape_RW3, w, color="#EF9F27", label="RW (L=3, feasible)")
ax[1].bar(x + w, g.ape_AGT, w, color="#185FA5", label="NN (AGT)")
ax[1].set_xticks(x); ax[1].set_xticklabels(["smooth years", "turning-point years"])
ax[1].set_ylabel("mean absolute % error")
ax[1].set_title("(b) The oracle RW advantage shrinks at turning points")
ax[1].set_ylim(0, 20)
ax[1].legend(frameon=False, fontsize=9)
ax[1].grid(alpha=0.25, axis="y")
plt.tight_layout()
local_fig = os.path.join(FIGDIR, "shock_decomposition.png")
paper_fig = os.path.join(PAPER_FIGDIR, "shock_decomposition.png")
plt.savefig(local_fig, dpi=200)
shutil.copyfile(local_fig, paper_fig)
print("\nsaved figs/shock_decomposition.png")
print("\nGrouped means (smooth vs shock):")
print(g.round(2))
