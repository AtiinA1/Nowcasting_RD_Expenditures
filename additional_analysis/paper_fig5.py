"""
Figure-5 replacement: AllVar out-of-sample true vs predicted GERD (test set) under
the leakage-free country-year split. Uses the trained ensemble predictions from
country_year_split.py. Output written to the Oxford submission figures folder.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out")
FIGDIR = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_CYsplit"
os.makedirs(FIGDIR, exist_ok=True)

pred = pd.read_csv(os.path.join(OUT, "cy_split_predictions.csv"))
d = pred[pred.Config == "AllVar"]
fig, ax = plt.subplots(figsize=(6, 6))
for ctry, g in d.groupby("Country"):
    ax.scatter(g.True_Values, g.pred_mean, s=18, alpha=0.7, label=ctry)
lim = [min(d.True_Values.min(), d.pred_mean.min())*0.8,
       max(d.True_Values.max(), d.pred_mean.max())*1.2]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("True GERD (USD bn, log scale)"); ax.set_ylabel("Predicted GERD (USD bn, log scale)")
ax.legend(ncol=2, fontsize=8, title="Country")
plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "AllVar_TrueVsPred_CYsplit.png"), dpi=200)
print("saved AllVar_TrueVsPred_CYsplit.png")
