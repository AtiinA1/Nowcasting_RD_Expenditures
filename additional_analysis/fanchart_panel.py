"""Main-body fan-chart figure: 3-country panel (JP, DE, CA).
Now shows the FULL observed GERD trajectory (training + test) with the held-out
test period shaded, and overlays the AGT ensemble nowcast + 95% interval on the
test years only. Source: out/temporal_split_predictions.csv (test preds) and
out/merged_features.csv (full observed history). Output:
figures/Nowcast_Model_Temporal/fanchart_panel_AGT.png
"""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
FIGDIR="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_Temporal"
df=pd.read_csv(os.path.join(OUT,"temporal_split_predictions.csv"))
feat=pd.read_csv(os.path.join(OUT,"merged_features.csv")); feat=feat[feat.Year>=2004]
panel=feat.groupby(["Country","Year"]).rd_expenditure.mean().reset_index().rename(columns={"rd_expenditure":"GERD"})
mcols=[f"m{i}" for i in range(10)]
agt=df[df.Config=="AGT"]
panels=[("JP","Japan"),("DE","Germany"),("CA","Canada")]
band_c="#b9d0ea"; mean_c="#2c6fbb"; obs_c="#555555"; test_shade="#fde0c5"

fig,axes=plt.subplots(1,3,figsize=(13,3.9))
for ax,(code,name) in zip(axes,panels):
    obs=panel[panel.Country==code].sort_values("Year")
    d=agt[agt.Country==code]
    tyrs=np.array(sorted(d.Year.unique()))                       # test years
    mu=np.array([d[d.Year==y]["pred_mean"].mean() for y in tyrs])
    lo=np.array([np.percentile(d[d.Year==y][mcols].values,2.5) for y in tyrs])
    hi=np.array([np.percentile(d[d.Year==y][mcols].values,97.5) for y in tyrs])
    t0=tyrs.min()
    # shade the held-out test span
    ax.axvspan(t0-0.5, obs.Year.max()+0.5, color=test_shade, alpha=.8, lw=0, zorder=0,
               label="Test period (held out)")
    ax.axvline(t0-0.5, color="#d98a3d", lw=1, ls=":", zorder=1)
    # full observed history (training + test)
    ax.plot(obs.Year, obs.GERD, "-o", color=obs_c, lw=1.3, ms=4, mfc="white", mec=obs_c,
            zorder=3, label="Observed GERD")
    # ensemble nowcast + 95% interval on test years
    ax.fill_between(tyrs, lo, hi, color=band_c, alpha=.7, lw=0, zorder=2,
                    label="95% ensemble interval")
    ax.plot(tyrs, mu, "-o", color=mean_c, lw=2, ms=5, zorder=4, label="AGT ensemble nowcast")
    ax.set_title(name, fontsize=11); ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(alpha=.2, lw=.5)
    ax.spines[["top","right"]].set_visible(False)
    # zero-anchored axis with generous headroom (natural scale for an expenditure level):
    # the ~2-6% nowcast error is then correctly shown as a small fraction of the level
    ax.set_ylim(0, max(obs.GERD.max(), hi.max())*2.3)
axes[0].set_ylabel("GERD (USD bn)")
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(0.5,1.08),fontsize=9.5)
plt.tight_layout()
out=os.path.join(FIGDIR,"fanchart_panel_AGT.png"); plt.savefig(out,dpi=200,bbox_inches="tight"); plt.close()
print("saved",out)
