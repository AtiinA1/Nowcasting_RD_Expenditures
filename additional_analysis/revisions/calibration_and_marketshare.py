"""
(1) Calibration / PIT gap: the ensemble bands capture seed dispersion only and
    undercover badly (Table: AGT 13% vs 95% nominal). Construct a calibrated
    predictive interval by adding the missing residual-variance component and
    show empirical coverage approaches nominal. Residual variance is estimated
    leave-one-out (LOO) to avoid using each point's own residual.

(2) Country-heterogeneity gap: turn the China/Korea/Japan caveat into a result
    by relating per-country nowcast accuracy to Google's search market share.

Inputs (read-only):
  out/temporal_split_predictions.csv   -- ensemble members m0..m9 (monthly rows)
  out/temporal_annual_predictions.csv  -- annual true + NN_AGT etc.
Outputs:
  revisions/out/calibration.csv, country_marketshare.csv
  revisions/figs/calibration.png, figs/accuracy_vs_marketshare.png
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "out"); os.makedirs(OUTDIR, exist_ok=True)
FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
SRC = os.path.normpath(os.path.join(HERE, "..", "out"))
UPDATED = os.path.normpath(os.path.join(HERE, "..", "robustness_overfit", "out", "all_configs_updated_pure_nn"))
PAPER_FIGDIR = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Revision"
os.makedirs(PAPER_FIGDIR, exist_ok=True)

# ---------- (1) Calibration ----------
sp = pd.read_csv(os.path.join(UPDATED, "all_configs_updated_pure_nn_member_predictions.csv"))
agt = sp[sp.Config == "AGT"].copy()
# annual ensemble: average each member across the 12 months of a country-year
ann_long = (agt.groupby(["Country", "Year", "seed"], as_index=False)
              .agg(pred=("pred", "mean"), true=("rd_expenditure", "first")))
ann = ann_long.pivot_table(index=["Country", "Year", "true"], columns="seed", values="pred").reset_index()
member_cols = [c for c in ann.columns if isinstance(c, (int, np.integer))]
M = ann[member_cols].values                  # (n_cy, n_members)
true = ann["true"].values
ens_mean = M.mean(axis=1)
ens_std = M.std(axis=1, ddof=1)              # across-member (seed) dispersion only
resid = true - ens_mean
n = len(true)

# LOO residual standard deviation (the missing component)
loo_resid_std = np.array([resid[np.arange(n) != i].std(ddof=1) for i in range(n)])
tot_std = np.sqrt(ens_std**2 + loo_resid_std**2)

def coverage(level, sd):
    z = norm.ppf(0.5 + level/2)
    lo, hi = ens_mean - z*sd, ens_mean + z*sd
    return np.mean((true >= lo) & (true <= hi)) * 100

levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
cov_ens = [coverage(l, ens_std) for l in levels]
cov_cal = [coverage(l, tot_std) for l in levels]
pd.DataFrame({"nominal": (levels*100).round(0),
              "cov_ensemble_only": np.round(cov_ens, 1),
              "cov_calibrated": np.round(cov_cal, 1)}).to_csv(
              os.path.join(OUTDIR, "calibration.csv"), index=False)
print("=== Calibration (AGT, temporal test set, n=%d) ===" % n)
print("nominal:        ", (levels*100).astype(int))
print("ensemble-only:  ", np.round(cov_ens, 0))
print("calibrated:     ", np.round(cov_cal, 0))
print("ensemble-only 95%% coverage = %.0f%%; calibrated = %.0f%%" % (cov_ens[-1], cov_cal[-1]))
print("mean ensemble std = %.2f bn; mean LOO residual std = %.2f bn" % (ens_std.mean(), loo_resid_std.mean()))

# PIT values
pit_ens = norm.cdf((true - ens_mean) / ens_std)
pit_cal = norm.cdf((true - ens_mean) / tot_std)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot([0, 100], [0, 100], "--", color="#888780", lw=1, label="perfect calibration")
ax[0].plot(levels*100, cov_ens, "-s", color="#BA7517", label="ensemble (seed) only")
ax[0].plot(levels*100, cov_cal, "-o", color="#185FA5", label="+ residual variance")
ax[0].set_xlabel("nominal coverage (%)"); ax[0].set_ylabel("empirical coverage (%)")
ax[0].set_title("(a) Reliability curve"); ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=0.25)
ax[1].hist(pit_ens, bins=np.linspace(0, 1, 9), color="#EF9F27", alpha=0.85, label="ensemble only")
ax[1].hist(pit_cal, bins=np.linspace(0, 1, 9), color="#378ADD", alpha=0.6, label="calibrated")
ax[1].axhline(n/8, ls="--", color="#888780", lw=1, label="uniform")
ax[1].set_xlabel("PIT value"); ax[1].set_ylabel("count")
ax[1].set_title("(b) PIT histogram"); ax[1].legend(frameon=False, fontsize=9)
local_fig = os.path.join(FIGDIR, "calibration.png")
paper_fig = os.path.join(PAPER_FIGDIR, "calibration.png")
plt.tight_layout(); plt.savefig(local_fig, dpi=200)
shutil.copyfile(local_fig, paper_fig)
print("saved figs/calibration.png")

# ---------- (2) Accuracy vs Google market share ----------
pred = pd.read_csv(os.path.join(SRC, "temporal_annual_predictions.csv"))
percountry = (pred.assign(ape=lambda d: (d.NN_AGT - d.GERD).abs()/d.GERD*100)
                  .groupby("Country").ape.mean())
# Representative Google search market share (%), desktop+mobile, recent (StatCounter).
# NOTE TO AUTHOR: replace with exact StatCounter values before submission.
google_share = {"US": 88, "GB": 93, "DE": 91, "CH": 92, "CA": 91,
                "JP": 77, "KR": 55, "CN": 3}
ms = pd.DataFrame({"Country": list(google_share),
                   "google_share": list(google_share.values())})
ms["MAPE"] = ms.Country.map(percountry)
ms = ms.dropna()
ms.to_csv(os.path.join(OUTDIR, "country_marketshare.csv"), index=False)
r, p = pearsonr(ms.google_share, ms.MAPE)
rs, ps = pearsonr(ms.google_share.rank(), ms.MAPE.rank())
print("\n=== Accuracy (AGT MAPE) vs Google market share ===")
print(ms.sort_values("google_share").to_string(index=False))
print("Pearson r=%.2f (p=%.3f); Spearman rho=%.2f (p=%.3f)" % (r, p, rs, ps))

fig, ax = plt.subplots(figsize=(6, 4.4))
ax.scatter(ms.google_share, ms.MAPE, s=55, color="#185FA5", zorder=3)
for _, rr in ms.iterrows():
    ax.annotate(rr.Country, (rr.google_share, rr.MAPE), fontsize=9,
                xytext=(4, 3), textcoords="offset points")
b, a = np.polyfit(ms.google_share, ms.MAPE, 1)
xs = np.linspace(ms.google_share.min(), ms.google_share.max(), 50)
ax.plot(xs, a + b*xs, color="#BA7517", lw=1.5, label="OLS fit (r=%.2f)" % r)
ax.set_xlabel("Google search market share (%)")
ax.set_ylabel("per-country nowcast MAPE (%), AGT")
ax.set_title("Per-country accuracy vs. Google search market share")
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "accuracy_vs_marketshare.png"), dpi=200)
print("saved figs/accuracy_vs_marketshare.png")
