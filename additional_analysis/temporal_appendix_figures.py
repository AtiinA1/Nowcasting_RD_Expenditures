"""Appendix prediction figures under the TEMPORAL split (per-config combined +
per-country true-vs-predicted), from temporal_split_predictions.csv.
Output: figures/Nowcast_Model_Temporal/appendix/"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
FIG="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_Temporal/appendix"; os.makedirs(FIG,exist_ok=True)
CONFIGS=["LagRD","Macros","AGT","MGT","AGTwRD","MGTwRD","AllVar"]
pred=pd.read_csv(os.path.join(OUT,"temporal_split_predictions.csv"))
def combined(cfg):
    d=pred[pred.Config==cfg]; fig,ax=plt.subplots(figsize=(5.2,5.2))
    for c,g in d.groupby("Country"): ax.scatter(g.True_Values,g.pred_mean,s=16,alpha=.7,label=c)
    lim=[d.True_Values.min()*.8,d.True_Values.max()*1.2]; ax.plot(lim,lim,"k--",lw=1)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(lim);ax.set_ylim(lim)
    ax.set_xlabel("True GERD (USD bn, log)");ax.set_ylabel("Predicted GERD (USD bn, log)");ax.legend(ncol=2,fontsize=7,title="Country")
    plt.tight_layout();plt.savefig(os.path.join(FIG,f"{cfg}_combined.png"),dpi=180);plt.close()
def country(cfg,cc):
    d=pred[(pred.Config==cfg)&(pred.Country==cc)]
    if not len(d): return
    a=d.groupby("Year").agg(t=("True_Values","mean"),p=("pred_mean","mean")).reset_index()
    fig,ax=plt.subplots(figsize=(4.6,4.0)); ax.scatter(a.t,a.p,s=40,color="#2c7fb8",zorder=3)
    for _,r in a.iterrows(): ax.annotate(int(r.Year),(r.t,r.p),fontsize=7,xytext=(3,3),textcoords="offset points")
    lim=[min(a.t.min(),a.p.min())*.95,max(a.t.max(),a.p.max())*1.05]; ax.plot(lim,lim,"k--",lw=1);ax.set_xlim(lim);ax.set_ylim(lim)
    ax.set_xlabel("True GERD (USD bn)");ax.set_ylabel("Predicted GERD (USD bn)")
    plt.tight_layout();plt.savefig(os.path.join(FIG,f"{cfg}_{cc}.png"),dpi=180);plt.close()
for cfg in CONFIGS: combined(cfg)
for cfg in ["AllVar","AGT"]:
    for cc in ["US","KR","GB","DE","CA","JP","CN","CH"]: country(cfg,cc)
print("saved temporal appendix figures:",len(os.listdir(FIG)))
