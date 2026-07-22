"""
Restyled / improved paper figures.
(1) Fig 2 (temporal): NN vs OLS by configuration as polished BOX PLOTS over
    per-country-year absolute percentage errors (honest distribution view).
(2) Appendix CY-split box plots: restyled.
Saves PNGs into the Oxford submission figures folders.
"""
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
FT="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_Temporal"
FC="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_CYsplit"
CONFIGS=["LagRD","Macros","AGT","MGT","AGTwRD","MGTwRD","AllVar"]
NN_C="#35658A"; OLS_C="#B7BC8B"
# seaborn Set3 palette, matching ML_comparison_figure_lag3 (per-configuration colors)
CONFIG_COLORS={"MGT":"#8dd3c7","AGT":"#ffffb3","MGTwRD":"#bebada","AGTwRD":"#fb8072",
               "AllVar":"#80b1d3","Macros":"#fdb462","LagRD":"#b3de69"}
from matplotlib.patches import Patch

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"grid.linestyle":"-","axes.axisbelow":True,
    "figure.dpi":150,"savefig.bbox":"tight"})

def style_box(bp,color):
    for b in bp["boxes"]: b.set(facecolor=color,alpha=0.75,edgecolor="#333",linewidth=1.0)
    for w in bp["whiskers"]: w.set(color="#333",linewidth=1.0)
    for c in bp["caps"]: c.set(color="#333",linewidth=1.0)
    for m in bp["medians"]: m.set(color="#111",linewidth=1.6)

# ---------- per-config OLS (temporal split, within-country log-std target) ----------
feat=pd.read_csv(os.path.join(OUT,"merged_features.csv")); feat=feat[feat.Year>=2004].copy()
feat.sort_values(["Country","Year","Month"],inplace=True)
AR=[f"rd_expenditure_lag{l}" for l in (1,2,3)]
MAC=[f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
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

def ape_by_cy(cfg):
    d=pred[pred.Config==cfg]
    nn=d.groupby(["Country","Year"]).agg(t=("True_Values","mean"),p=("pred_mean","mean")).reset_index()
    nn_ape=(np.abs(nn.t-nn.p)/nn.t*100).values
    ol=ols_level(FEATS[cfg]); ft=feat[masks["test"]].copy(); ft["ol"]=ol[masks["test"]]
    olann=ft.groupby(["Country","Year"]).agg(t=("rd_expenditure","mean"),p=("ol","mean")).reset_index()
    ol_ape=(np.abs(olann.t-olann.p)/olann.t*100).values
    return nn_ape,ol_ape

nn_data=[]; ol_data=[]
for cfg in CONFIGS:
    a,b=ape_by_cy(cfg); nn_data.append(a); ol_data.append(b)
print("median APE by config (NN | OLS):")
for cfg,a,b in zip(CONFIGS,nn_data,ol_data): print(f"  {cfg:8s}: {np.median(a):5.1f} | {np.median(b):5.1f}  (mean {np.mean(a):5.1f} | {np.mean(b):5.1f})")

fig,ax=plt.subplots(figsize=(8.5,4.4)); pos=np.arange(len(CONFIGS)); w=0.34
b1=ax.boxplot(nn_data,positions=pos-0.19,widths=w,patch_artist=True,showfliers=True,
   flierprops=dict(marker="o",ms=2.5,mfc=NN_C,mec="none",alpha=.5),showmeans=True,
   meanprops=dict(marker="D",mfc="white",mec="#111",ms=4)); style_box(b1,NN_C)
b2=ax.boxplot(ol_data,positions=pos+0.19,widths=w,patch_artist=True,showfliers=True,
   flierprops=dict(marker="o",ms=2.5,mfc=OLS_C,mec="none",alpha=.5),showmeans=True,
   meanprops=dict(marker="D",mfc="white",mec="#111",ms=4)); style_box(b2,OLS_C)
ax.set_xticks(pos); ax.set_xticklabels(CONFIGS); ax.set_ylabel("Absolute percentage error (%)")
ax.set_title("Out-of-sample APE by configuration: neural network vs OLS (temporal split)")
ax.legend([b1["boxes"][0],b2["boxes"][0]],["Neural network","OLS (same inputs)"],loc="upper left",frameon=False)
ax.set_ylim(0,min(120,max(np.percentile(np.concatenate(ol_data),98),np.percentile(np.concatenate(nn_data),98))*1.15))
plt.savefig(os.path.join(FT,"NNvsOLS_Temporal_box.png")); plt.close()
print("saved NNvsOLS_Temporal_box.png")

# ---------- appendix CY box plots restyle (per-country-year APE) ----------
cyp=os.path.join(OUT,"cy_split_predictions.csv")
if os.path.exists(cyp):
    cyt={(c,int(y)):s for (c,y),s in {}.items()}  # placeholder
    cy=pd.read_csv(cyp)
    # recompute CY split masks for OLS
    rng=np.random.default_rng(42); cysplit={}
    f2=pd.read_csv(os.path.join(OUT,"merged_features.csv")); f2=f2[f2.Year>=2004].copy(); f2.sort_values(["Country","Year","Month"],inplace=True)
    for ctry,g in f2.groupby("Country"):
        ys=np.array(sorted(g.Year.unique())); rng.shuffle(ys); n=len(ys); ntr=int(round(n*.64)); nva=int(round(n*.16))
        for i,y in enumerate(ys): cysplit[(ctry,int(y))]="train" if i<ntr else ("val" if i<ntr+nva else "test")
    f2["split"]=[cysplit[(c,int(y))] for c,y in zip(f2.Country,f2.Year)]
    m2={s:(f2.split==s).values for s in ("train","val","test")}; fit2=m2["train"]
    mo2=pd.get_dummies(f2.Month,prefix="M").astype(float); cd2=pd.get_dummies(f2.Country,prefix="c").astype(float)
    y2=f2.rd_expenditure.values.astype(float)
    def ols_cy(cols):
        Xc=f2[cols].fillna(0).astype(float).values; mu,sd=Xc[m2["train"]].mean(0),Xc[m2["train"]].std(0); sd[sd==0]=1
        X=np.column_stack([np.ones(len(f2)),(Xc-mu)/sd,mo2.values,cd2.values])
        b=np.linalg.solve(X[fit2].T@X[fit2]+1e-2*np.eye(X.shape[1]),X[fit2].T@y2[fit2]); return X@b
    for kind,fn,ylab,cap in [("MAPE","MAPE_BoxPlot_CYsplit","Absolute percentage error (%)",110),
                              ("RMSE","RMSE_BoxPlot_CYsplit","Per-country RMSE (USD bn)",None)]:
        nnd=[];old=[]
        for cfg in CONFIGS:
            d=cy[cy.Config==cfg]
            if kind=="MAPE":
                ann=d.groupby(["Country","Year"]).agg(t=("True_Values","mean"),p=("pred_mean","mean")).reset_index()
                nnd.append((np.abs(ann.t-ann.p)/ann.t*100).values)
                ol=ols_cy(FEATS[cfg]); ft=f2[m2["test"]].copy(); ft["ol"]=ol[m2["test"]]
                oa=ft.groupby(["Country","Year"]).agg(t=("rd_expenditure","mean"),p=("ol","mean")).reset_index()
                old.append((np.abs(oa.t-oa.p)/oa.t*100).values)
            else:
                vals=[]
                for c,g in d.groupby("Country"):
                    vals.append(np.sqrt(np.mean((g.True_Values-g.pred_mean)**2)))
                nnd.append(np.array(vals))
                ol=ols_cy(FEATS[cfg]); ft=f2[m2["test"]].copy(); ft["ol"]=ol[m2["test"]]
                ov=[np.sqrt(np.mean((gg.rd_expenditure-gg.ol)**2)) for _,gg in ft.groupby("Country")]
                old.append(np.array(ov))
        fig,ax=plt.subplots(figsize=(8.5,4.4)); pos=np.arange(len(CONFIGS))
        ccols=[CONFIG_COLORS[c] for c in CONFIGS]
        b1=ax.boxplot(nnd,positions=pos-0.19,widths=0.34,patch_artist=True,showfliers=True,
           flierprops=dict(marker="o",ms=2.5,mfc="#666",mec="none",alpha=.5))
        b2=ax.boxplot(old,positions=pos+0.19,widths=0.34,patch_artist=True,showfliers=True,
           flierprops=dict(marker="o",ms=2.5,mfc="#666",mec="none",alpha=.5))
        for i,bx in enumerate(b1["boxes"]): bx.set(facecolor=ccols[i],alpha=0.85,edgecolor="#333",linewidth=1.0)
        for i,bx in enumerate(b2["boxes"]): bx.set(facecolor=ccols[i],alpha=0.45,edgecolor="#333",linewidth=1.0,hatch="////")
        for bp in (b1,b2):
            for w in bp["whiskers"]: w.set(color="#333",linewidth=1.0)
            for c in bp["caps"]: c.set(color="#333",linewidth=1.0)
            for m in bp["medians"]: m.set(color="#111",linewidth=1.6)
        ax.set_xticks(pos); ax.set_xticklabels(CONFIGS); ax.set_ylabel(ylab)
        ax.set_title(f"Country-year split: {kind} by configuration (NN vs OLS)")
        leg=[Patch(facecolor="#bbbbbb",edgecolor="#333",label="Neural network (solid)"),
             Patch(facecolor="#bbbbbb",edgecolor="#333",alpha=0.5,hatch="////",label="OLS, same inputs (hatched)")]
        ax.legend(handles=leg,loc="upper left",frameon=False)
        all_vals = np.concatenate([np.asarray(v, dtype=float) for v in nnd + old if len(v)])
        if len(all_vals):
            upper = np.nanmax(all_vals) * 1.20
            if kind == "MAPE":
                upper = max(125, upper)
            else:
                upper = max(30, upper)
            ax.set_ylim(0, upper)
        plt.savefig(os.path.join(FC,fn+".png")); plt.close(); print("saved",fn)
print("DONE")
