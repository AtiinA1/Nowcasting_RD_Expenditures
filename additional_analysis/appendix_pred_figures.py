"""
Regenerate the appendix out-of-sample prediction figures under the leakage-free
country-year split, using the trained ensemble predictions from
country_year_split.py (cy_split_predictions.csv).

  App B (per configuration) : combined true-vs-predicted scatter -> <CONFIG>_combined.png
  App C (per country)       : true-vs-predicted scatter for AllVar and AGT -> <CONFIG>_<CC>.png

Output dir: Nowcasting_Oxford_submission/figures/Nowcast_Model_CYsplit/appendix/
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out")
FIGDIR = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_CYsplit/appendix"
os.makedirs(FIGDIR, exist_ok=True)
CONFIGS = ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]

pred = pd.read_csv(os.path.join(OUT, "cy_split_predictions.csv"))

def combined_plot(cfg):
    d = pred[pred.Config == cfg]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for ctry, g in d.groupby("Country"):
        ax.scatter(g.True_Values, g.pred_mean, s=16, alpha=0.7, label=ctry)
    lim = [d.True_Values.min()*0.8, d.True_Values.max()*1.2]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True GERD (USD bn, log)"); ax.set_ylabel("Predicted GERD (USD bn, log)")
    ax.legend(ncol=2, fontsize=7, title="Country")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, f"{cfg}_combined.png"), dpi=180); plt.close()

def country_plot(cfg, cc):
    d = pred[(pred.Config == cfg) & (pred.Country == cc)].sort_values(["Year", "Month"])
    if not len(d): return
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    # collapse to annual (one point per test year) for clarity
    ann = d.groupby("Year").agg(t=("True_Values", "mean"), p=("pred_mean", "mean")).reset_index()
    ax.scatter(ann.t, ann.p, s=40, color="#2c7fb8", zorder=3)
    for _, r in ann.iterrows():
        ax.annotate(int(r.Year), (r.t, r.p), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [min(ann.t.min(), ann.p.min())*0.95, max(ann.t.max(), ann.p.max())*1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True GERD (USD bn)"); ax.set_ylabel("Predicted GERD (USD bn)")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, f"{cfg}_{cc}.png"), dpi=180); plt.close()

for cfg in CONFIGS:
    combined_plot(cfg)
for cfg in ["AllVar", "AGT"]:
    for cc in ["US", "KR", "GB", "DE", "CA", "JP", "CN", "CH"]:
        country_plot(cfg, cc)
print("Saved appendix prediction figures to", FIGDIR)
print("files:", len(os.listdir(FIGDIR)))
