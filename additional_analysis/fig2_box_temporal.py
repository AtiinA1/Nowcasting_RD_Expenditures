"""Temporal-split Fig 2 as a box plot emphasizing DISPERSION (OLS more variable
than NN), with mean markers so the average gap is also visible.
Prints spread stats (std, IQR) to confirm the variance message, then saves figure."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
FT="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_Temporal"
CONFIGS=["LagRD","Macros","AGT","MGT","AGTwRD","MGTwRD","AllVar"]; NN_C="#2c6fbb"; OLS_C="#e07b39"
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
def ape(cfg):
    d=pred[pred.Config==cfg]; nn=d.groupby(["Country","Year"]).agg(t=("True_Values","mean"),p=("pred_mean","mean")).reset_index()
    nn_ape=(np.abs(nn.t-nn.p)/nn.t*100).values
    ol=ols_level(FEATS[cfg]); ft=feat[masks["test"]].copy(); ft["ol"]=ol[masks["test"]]
    oa=ft.groupby(["Country","Year"]).agg(t=("rd_expenditure","mean"),p=("ol","mean")).reset_index()
    return nn_ape,(np.abs(oa.t-oa.p)/oa.t*100).values
nn_data=[];ol_data=[]
print("config | NN: median mean std IQR | OLS: median mean std IQR")
for cfg in CONFIGS:
    a,b=ape(cfg); nn_data.append(a); ol_data.append(b)
    iqr=lambda x:np.percentile(x,75)-np.percentile(x,25)
    print(f"{cfg:8s} | NN {np.median(a):5.1f} {np.mean(a):5.1f} {np.std(a):5.1f} {iqr(a):5.1f} | OLS {np.median(b):5.1f} {np.mean(b):5.1f} {np.std(b):5.1f} {iqr(b):5.1f}")
fig,ax=plt.subplots(figsize=(9,4.6)); pos=np.arange(len(CONFIGS)); w=0.34
mp=dict(marker="D",mfc="white",mec="#111",ms=5,zorder=5)
b1=ax.boxplot(nn_data,positions=pos-0.19,widths=w,patch_artist=True,showmeans=True,meanprops=mp,
   flierprops=dict(marker="o",ms=3,mfc=NN_C,mec="none",alpha=.55),whis=1.5)
b2=ax.boxplot(ol_data,positions=pos+0.19,widths=w,patch_artist=True,showmeans=True,meanprops=mp,
   flierprops=dict(marker="o",ms=3,mfc=OLS_C,mec="none",alpha=.55),whis=1.5)
for bp,c in [(b1,NN_C),(b2,OLS_C)]:
    for b in bp["boxes"]: b.set(facecolor=c,alpha=.75,edgecolor="#333")
    for el in bp["whiskers"]+bp["caps"]: el.set(color="#333")
    for med in bp["medians"]: med.set(color="#111",linewidth=1.6)
ax.set_xticks(pos); ax.set_xticklabels(CONFIGS); ax.set_ylabel("Absolute percentage error (%)")
ax.set_ylim(0,60)
ax.set_title("Out-of-sample error dispersion by configuration: NN vs OLS (temporal split)")
ax.legend([b1["boxes"][0],b2["boxes"][0]],["Neural network","OLS (same inputs)"],loc="upper left",frameon=False)
ax.annotate("$\\diamond$ = mean",xy=(0.99,0.96),xycoords="axes fraction",ha="right",fontsize=9,color="#333")
plt.savefig(os.path.join(FT,"NNvsOLS_Temporal_box.png")); print("saved NNvsOLS_Temporal_box.png")
