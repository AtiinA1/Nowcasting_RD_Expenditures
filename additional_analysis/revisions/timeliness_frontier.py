"""
Insight figure: timeliness-accuracy frontier. The search-based nowcast is
available *now* (publication lag 0), whereas the autoregressive benchmarks
require the most recent published annual figure, which under R&D's 2-3 year
publication lag is itself stale. Plot pooled out-of-sample MAPE of RW(L) and
AR1-trend(L) as the assumed publication lag L grows, against the flat,
lag-0 nowcast.

Inputs (read-only):
  out/temporal_annual_predictions.csv   -- test true GERD + NN_AGT + AR1
  out/merged_features.csv               -- full per-country annual GERD series
Output:
  revisions/figs/timeliness_frontier.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
SRC = os.path.normpath(os.path.join(HERE, "..", "out"))

pred = pd.read_csv(os.path.join(SRC, "temporal_annual_predictions.csv"))
mf = pd.read_csv(os.path.join(SRC, "merged_features.csv"))
series = (mf[["Country", "Year", "rd_expenditure"]]
          .drop_duplicates(["Country", "Year"]).sort_values(["Country", "Year"]))

def gerd(c, y):
    r = series[(series.Country == c) & (series.Year == y)]
    return float(r.rd_expenditure.iloc[0]) if len(r) else np.nan

def pooled_mape(pred_col):
    e = []
    for _, r in pred.iterrows():
        p = pred_col(r)
        if p is not None and np.isfinite(p) and r.GERD > 0:
            e.append(abs(p - r.GERD) / r.GERD * 100)
    return float(np.mean(e))

# RW(L): carry forward the actual figure from L years ago. AR1-trend(L): project the
# log-linear drift estimated from the two most recent *available* (>= L years old) points.
lags = [1, 2, 3]
rw = [pooled_mape(lambda r, L=L: gerd(r.Country, int(r.Year) - L)) for L in lags]

def ar1_trend(r, L):
    yL, yL1 = gerd(r.Country, int(r.Year) - L), gerd(r.Country, int(r.Year) - L - 1)
    if not (yL and yL1 and yL > 0 and yL1 > 0):
        return None
    g = np.log(yL / yL1)                  # most recent observable drift
    return yL * np.exp(g * L)             # project forward L years
ar = [pooled_mape(lambda r, L=L: ar1_trend(r, L)) for L in lags]

nowcast = pooled_mape(lambda r: r.NN_AGT)   # available at lag 0
print("RW(L) MAPE:", dict(zip(lags, np.round(rw, 1))))
print("AR1-trend(L) MAPE:", dict(zip(lags, np.round(ar, 1))))
print("Search nowcast (AGT, lag 0) MAPE:", round(nowcast, 1))

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.axhline(nowcast, color="#185FA5", lw=2.2, label="Search nowcast (available now)")
ax.scatter([0], [nowcast], color="#185FA5", s=55, zorder=4)
ax.plot(lags, rw, "-s", color="#BA7517", label="Random walk RW(L)")
ax.plot(lags, ar, "-^", color="#993C1D", label="Trend extrapolation AR1(L)")
ax.set_xlabel("publication lag L of the most recent available annual figure (years)")
ax.set_ylabel("pooled out-of-sample MAPE (%)")
ax.set_xticks([0, 1, 2, 3])
ax.set_title("Timeliness--accuracy frontier")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25)
ax.annotate("nowcast uses\ninformation available now", (0, nowcast),
            xytext=(0.35, nowcast + 3), fontsize=8, color="#185FA5")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "timeliness_frontier.png"), dpi=200)
print("saved figs/timeliness_frontier.png")
