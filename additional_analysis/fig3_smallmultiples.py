"""
Redesigned Figure 3: per-country small multiples showing the chronological
train / validation / test phases, the observed (ground-truth) annual GERD as
faint markers, and the AllVar ensemble prediction (with band) as a colored line.
Retrains the AllVar temporal model and predicts ALL country-years.
Output: figures/Nowcast_Model_Temporal/AllVar_TrueVsPred_Temporal.png
"""
import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
FT="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_Temporal"
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED)
m=pd.read_csv(os.path.join(OUT,"merged_features.csv")); m=m[m.Year>=2004].copy(); m.sort_values(["Country","Year","Month"],inplace=True)
AR=[f"rd_expenditure_lag{l}" for l in (1,2,3)]; MAC=[f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc=[c for c in m.columns if "_yearly_avg_lag" in c]; YTD=[c for c in m.columns if c.endswith("_mean_YTD")]
cols=AR+MAC+AGTc+YTD                      # AllVar
split={}
for ctry,g in m.groupby("Country"):
    yrs=np.array(sorted(g.Year.unique())); n=len(yrs); ntr=int(round(n*.64)); nva=int(round(n*.16))
    for i,y in enumerate(yrs): split[(ctry,int(y))]="train" if i<ntr else ("val" if i<ntr+nva else "test")
m["split"]=[split[(c,int(y))] for c,y in zip(m.Country,m.Year)]
months=pd.get_dummies(m.Month,prefix="M").astype(float); le=LabelEncoder(); cc=le.fit_transform(m.Country)
masks={s:(m.split==s).values for s in ("train","val","test")}
logy=np.log(m.rd_expenditure.values.astype(float)); cmean,cstd={},{}
for ctry,g in m.groupby("Country"):
    v=np.log(g.rd_expenditure.values[(g.split=="train").values]); cmean[ctry]=v.mean(); cstd[ctry]=max(v.std(),0.05)
cm=m.Country.map(cmean).values; cs=m.Country.map(cstd).values; ystd=(logy-cm)/cs
sc=StandardScaler().fit(m[cols].fillna(0).astype(float).values[masks["train"]])
X=np.hstack([sc.transform(m[cols].fillna(0).astype(float).values),months.values])
def T(a): return torch.FloatTensor(a)
Xtr=T(X[masks["train"]]);Xva=T(X[masks["val"]]);ctr=torch.LongTensor(cc[masks["train"]]);cva=torch.LongTensor(cc[masks["val"]])
ytr=T(ystd[masks["train"]].reshape(-1,1));yva=T(ystd[masks["val"]].reshape(-1,1)); call=torch.LongTensor(cc); Xall=T(X)
class MLP(nn.Module):
    def __init__(s,d,h,nc,emb=4,p=0.1):
        super().__init__(); s.emb=nn.Embedding(nc,emb); dims=[d+emb]+h
        s.lin=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(len(h)))
        s.bn=nn.ModuleList(nn.BatchNorm1d(x) for x in h); s.out=nn.Linear(dims[-1],1); s.drop=nn.Dropout(p); s.relu=nn.ReLU()
    def forward(s,x,c):
        x=torch.cat([x,s.emb(c)],1)
        for lin,bn in zip(s.lin,s.bn): x=s.drop(s.relu(bn(lin(x))))
        return s.out(x)
print("training AllVar (10)...",flush=True); preds=[]
for mm in range(10):
    torch.manual_seed(mm); net=MLP(X.shape[1],[200,20,20],len(le.classes_))
    opt=optim.AdamW(net.parameters(),lr=0.01,weight_decay=1e-4); sch=MultiStepLR(opt,[300],0.1); crit=nn.MSELoss()
    best=np.inf;bad=0;bs=64;bstate=None
    for ep in range(600):
        net.train(); perm=torch.randperm(len(Xtr))
        for i in range(0,len(Xtr),bs):
            idx=perm[i:i+bs]; opt.zero_grad(); loss=crit(net(Xtr[idx],ctr[idx]),ytr[idx]); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad(): vl=crit(net(Xva,cva),yva).item()
        if vl<best-1e-7: best=vl;bad=0;bstate={k:v.clone() for k,v in net.state_dict().items()}
        else:
            bad+=1
            if bad>=70: break
    if bstate: net.load_state_dict(bstate); net.eval()
    with torch.no_grad(): preds.append(np.exp(net(Xall,call).numpy().ravel()*cs+cm))
P=np.column_stack(preds); m["pred_mean"]=P.mean(1); m["pred_lo"]=np.quantile(P,.1,1); m["pred_hi"]=np.quantile(P,.9,1)
print("done",flush=True)
# annual collapse
ann=m.groupby(["Country","Year"]).agg(true=("rd_expenditure","mean"),pred=("pred_mean","mean"),
     lo=("pred_lo","mean"),hi=("pred_hi","mean"),split=("split","first")).reset_index()
order=["US","CN","JP","DE","KR","GB","CA","CH"]
cols_c=plt.cm.tab10(np.linspace(0,1,10))
fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=False); axes=axes.ravel()
phase_col={"train":"#f0f0f0","val":"#fff3d6","test":"#dbeeff"}
for k,ctry in enumerate(order):
    ax=axes[k]; d=ann[ann.Country==ctry].sort_values("Year"); col=cols_c[k]
    # phase shading by year
    for ph in ["train","val","test"]:
        ys=d[d.split==ph].Year.values
        if len(ys): ax.axvspan(ys.min()-0.5,ys.max()+0.5,color=phase_col[ph],zorder=0)
    ax.fill_between(d.Year,d.lo,d.hi,color=col,alpha=0.18,lw=0,zorder=1)
    ax.plot(d.Year,d.pred,"-",color=col,lw=1.8,zorder=3,label="Prediction")
    ax.scatter(d.Year,d.true,s=26,facecolor="none",edgecolor="#555",linewidth=1.1,zorder=4,label="Observed (true)")
    ax.set_title(ctry,fontsize=11,fontweight="bold"); ax.set_yscale("log")
    # widen the (log) y-range so the prediction-vs-observed gap is not visually exaggerated
    vv=np.concatenate([d.true.values,d.lo.values,d.hi.values,d.pred.values]); vv=vv[vv>0]
    lmin,lmax=np.log10(vv.min()),np.log10(vv.max()); lr=max(lmax-lmin,1e-3); pad=2.0*lr
    ax.set_ylim(10**(lmin-pad),10**(lmax+pad))
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
    ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    if k%4==0: ax.set_ylabel("GERD (USD bn, log)")
# shared legend (phases + series)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles=[Patch(facecolor=phase_col["train"],label="Train"),Patch(facecolor=phase_col["val"],label="Validation"),
         Patch(facecolor=phase_col["test"],label="Test"),
         Line2D([0],[0],color="#444",lw=1.8,label="Ensemble prediction"),
         Line2D([0],[0],marker="o",mfc="none",mec="#555",lw=0,label="Observed (true) GERD")]
fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Annual GERD: observed vs AllVar ensemble nowcast, by country and split phase (temporal split)",y=1.0,fontsize=12)
plt.tight_layout(rect=[0,0.03,1,0.99]); plt.savefig(os.path.join(FT,"AllVar_TrueVsPred_Temporal.png"),dpi=160,bbox_inches="tight")
print("saved AllVar_TrueVsPred_Temporal.png")
