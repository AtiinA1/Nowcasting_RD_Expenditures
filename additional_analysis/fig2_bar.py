"""Polished aggregate bar chart for main Fig 2 (temporal): NN vs OLS MAPE & RMSE
by configuration. Overwrites figures/Nowcast_Model_Temporal/NNvsOLS_Temporal.png."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
FT="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_Temporal"
CONFIGS=["LagRD","Macros","AGT","MGT","AGTwRD","MGTwRD","AllVar"]; NN_C="#2c6fbb"; OLS_C="#e07b39"
# seaborn Set3 palette, matching ML_comparison_figure_lag3 (per-configuration colors)
CONFIG_COLORS={"MGT":"#8dd3c7","AGT":"#ffffb3","MGTwRD":"#bebada","AGTwRD":"#fb8072",
               "AllVar":"#80b1d3","Macros":"#fdb462","LagRD":"#b3de69"}
from matplotlib.patches import Patch
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,"figure.dpi":150,"savefig.bbox":"tight"})
feat=pd.read_csv(os.path.join(OUT,"merged_features.csv")); feat=feat[feat.Year>=2004].copy(); feat.sort_values(["Country","Year","Month"],inplace=True)
AR=[f"rd_expenditure_lag{l}" for l in (1,2,3)]; MAC=[f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc=[c for c in feat.columns if "_yearly_avg_lag" in c]; YTD=[c for c in feat.columns if c.endswith("_mean_YTD")]
FEATS={"LagRD":AR,"Macros":AR+MAC,"AGT":AGTc,"MGT":AGTc+YTD,"AGTwRD":AR+AGTc,"MGTwRD":AR+AGTc+YTD,"AllVar":AR+MAC+AGTc+YTD}
split={}
for ctry,g in feat.groupby("Country"):
    yrs=np.array(sorted(g.Year.unique())); n=len(yrs); ntr=int(round(n*.64)); nva=int(round(n*.16))
    for i,y in enumerate(yrs): split[(ctry,int(y))]="train" if i<ntr else ("val" if i<ntr+nva else "test")
feat["split"]=[split[(c,int(y))] for c,y in zip(feat.Country,feat.Year)]
masks={s:(feat.split==s).values for s in ("train","val","test")}; fitm=masks["train"]|masks["val"]
logy=np.log(feat.rd_expenditure.values.astype(float)); cmean,cstd={},{}
for ctry,g in feat.groupby("Country"):
    v=np.log(g.rd_expenditure.values[(g.split=="train").values]); cmean[ctry]=v.mean(); cstd[ctry]=max(v.std(),0.05)
cm=feat.Country.map(cmean).values; cs=feat.Country.map(cstd).values; ystd=(logy-cm)/cs
months=pd.get_dummies(feat.Month,prefix="M").astype(float); cdum=pd.get_dummies(feat.Country,prefix="c").astype(float)
def ols_level(cols):
    Xc=feat[cols].fillna(0).astype(float).values; mu,sd=Xc[masks["train"]].mean(0),Xc[masks["train"]].std(0); sd[sd==0]=1
    X=np.column_stack([np.ones(len(feat)),(Xc-mu)/sd,months.values,cdum.values])
    b=np.linalg.solve(X[fitm].T@X[fitm]+1e-2*np.eye(X.shape[1]),X[fitm].T@ystd[fitm]); return np.exp((X@b)*cs+cm)
pred=pd.read_csv(os.path.join(OUT,"temporal_split_predictions.csv"))
def agg(cfg):
    d=pred[pred.Config==cfg]; nn=d.groupby(["Country","Year"]).agg(t=("True_Values","mean"),p=("pred_mean","mean")).reset_index()
    nn_mape=np.mean(np.abs(nn.t-nn.p)/nn.t)*100; nn_rmse=np.sqrt(np.mean((nn.t-nn.p)**2))
    ol=ols_level(FEATS[cfg]); ft=feat[masks["test"]].copy(); ft["ol"]=ol[masks["test"]]
    oa=ft.groupby(["Country","Year"]).agg(t=("rd_expenditure","mean"),p=("ol","mean")).reset_index()
    ol_mape=np.mean(np.abs(oa.t-oa.p)/oa.t)*100; ol_rmse=np.sqrt(np.mean((oa.t-oa.p)**2))
    return nn_mape,ol_mape,nn_rmse,ol_rmse
M=np.array([agg(c) for c in CONFIGS]); ccols=[CONFIG_COLORS[c] for c in CONFIGS]
fig,ax=plt.subplots(1,2,figsize=(12,4.3)); x=np.arange(len(CONFIGS)); w=0.38
for k,(ni,oi,lab) in enumerate([(0,1,"MAPE (%)"),(2,3,"RMSE (USD bn)")]):
    ax[k].bar(x-w/2,M[:,ni],w,color=ccols,edgecolor="#333",linewidth=0.7)
    ax[k].bar(x+w/2,M[:,oi],w,color=ccols,edgecolor="#333",linewidth=0.7,alpha=0.5,hatch="////")
    ax[k].set_xticks(x); ax[k].set_xticklabels(CONFIGS,rotation=30,ha="right"); ax[k].set_ylabel(lab)
leg=[Patch(facecolor="#bbbbbb",edgecolor="#333",label="Neural network (solid)"),
     Patch(facecolor="#bbbbbb",edgecolor="#333",alpha=0.5,hatch="////",label="OLS, same inputs (hatched)")]
ax[0].legend(handles=leg,frameon=False,loc="upper center"); ax[1].legend(handles=leg,frameon=False,loc="upper center")
ax[0].set_title("Out-of-sample accuracy by configuration: neural network vs OLS (temporal split)",loc="left",fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FT,"NNvsOLS_Temporal.png")); print("saved styled NNvsOLS_Temporal.png")
