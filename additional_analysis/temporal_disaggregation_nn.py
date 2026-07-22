"""
Recompute the NN-elasticity temporal disaggregation (Step B) using the
TEMPORAL-SPLIT AGT model (within-country log-standardized target), for
consistency with the main-body Step A. Faithfully reproduces the original
pipeline:
  - elasticities via compute_elasticity_torch: perturb each standardized AGT
    feature by +1%, measure the relative change in the LEVEL prediction, divide
    by 0.01; average over a country's training samples and the 10 ensemble members
    (MLP_AGT.py, lines 1381-1463);
  - average elasticities over the 3 annual lags -> one elasticity per topic
    (MLP_AGT_4TempDisagg.py, lines 1380-1388);
  - monthly share p_{topic,m} = GT_{topic,m} / sum_m GT_{topic}; adjusted share
    = elasticity_topic * p; monthly R&D = yearly R&D * adjusted / sum_m adjusted.
The classical estimators (Chow-Lin/Sax, sparse/Mosley) are split-independent
(computed in R from the monthly GT topics) and are reused from combined_estimates.csv.

Outputs: out/combined_estimates_temporal.csv, correlations printed,
figures in figures/Disaggregation/ (temporal versions).
"""
import os, numpy as np, pandas as pd
from scipy import stats
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED)
merged=pd.read_csv(os.path.join(OUT,"merged_features.csv")); merged=merged[merged.Year>=2004].copy()
merged.sort_values(["Country","Year","Month"],inplace=True)
AGTc=[c for c in merged.columns if "_yearly_avg_lag" in c]            # 171 features (57 topics x 3 lags)

# temporal split + logstd target (identical to main model)
split={}
for ctry,g in merged.groupby("Country"):
    yrs=np.array(sorted(g.Year.unique())); n=len(yrs); n_tr=int(round(n*0.64)); n_va=int(round(n*0.16))
    for i,y in enumerate(yrs): split[(ctry,int(y))]="train" if i<n_tr else ("val" if i<n_tr+n_va else "test")
merged["split"]=[split[(c,int(y))] for c,y in zip(merged.Country,merged.Year)]
months=pd.get_dummies(merged.Month,prefix="M").astype(float)
le=LabelEncoder(); merged["cc"]=le.fit_transform(merged.Country)
masks={s:(merged.split==s).values for s in ("train","val","test")}
logy=np.log(merged.rd_expenditure.values.astype(float))
cmean,cstd={},{}
for ctry,g in merged.groupby("Country"):
    v=np.log(g.rd_expenditure.values[(g.split=="train").values]); cmean[ctry]=v.mean(); cstd[ctry]=max(v.std(),0.05)
cm=merged.Country.map(cmean).values; cs=merged.Country.map(cstd).values; ystd=(logy-cm)/cs

sc=StandardScaler().fit(merged[AGTc].fillna(0).astype(float).values[masks["train"]])
Xc=sc.transform(merged[AGTc].fillna(0).astype(float).values)
X=np.hstack([Xc,months.values]); n_gt=len(AGTc)
cc=merged.cc.values
def T(a): return torch.FloatTensor(a)
Xtr=T(X[masks["train"]]);Xva=T(X[masks["val"]]);ctr=torch.LongTensor(cc[masks["train"]]);cva=torch.LongTensor(cc[masks["val"]])
ytr=T(ystd[masks["train"]].reshape(-1,1)); yva=T(ystd[masks["val"]].reshape(-1,1))

class MLP(nn.Module):
    def __init__(s,d,h,nc,emb=4,p=0.1):
        super().__init__(); s.emb=nn.Embedding(nc,emb); dims=[d+emb]+h
        s.lin=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(len(h)))
        s.bn=nn.ModuleList(nn.BatchNorm1d(x) for x in h); s.out=nn.Linear(dims[-1],1); s.drop=nn.Dropout(p); s.relu=nn.ReLU()
    def forward(s,x,c):
        x=torch.cat([x,s.emb(c)],1)
        for lin,bn in zip(s.lin,s.bn): x=s.drop(s.relu(bn(lin(x))))
        return s.out(x)

print("training AGT temporal ensemble (10 members)...",flush=True)
models=[]
for m in range(10):
    torch.manual_seed(m); net=MLP(X.shape[1],[200,20,20],len(le.classes_))
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
    if bstate: net.load_state_dict(bstate); net.eval(); models.append(net)
print("done training",flush=True)

# ---- elasticities: perturb each standardized GT feature by +1%, relative change in LEVEL ----
Xall=T(X); call=torch.LongTensor(cc)
mu_c=cm; sd_c=cs
def level_pred(net,Xt):
    with torch.no_grad(): z=net(Xt,call).numpy().ravel()
    return np.exp(z*sd_c+mu_c)
# accumulate per-sample elasticity for each feature, averaged across models
el_sum=np.zeros((len(merged),n_gt))
for net in models:
    base=level_pred(net,Xall)
    for j in range(n_gt):
        Xp=X.copy(); Xp[:,j]*=1.01
        pert=level_pred(net,T(Xp))
        el_sum[:,j]+=((pert-base)/base)/0.01
el=el_sum/len(models)                                   # (n_samples, n_gt) per-sample elasticity
# average over TRAIN samples within each country -> eta_{country, feature}
eldf=pd.DataFrame(el,columns=AGTc); eldf["Country"]=merged.Country.values; eldf["istrain"]=masks["train"]
eta=eldf[eldf.istrain].groupby("Country")[AGTc].mean()  # country x feature
# average over the 3 lags -> per (country, topic)
topics=sorted(set(c.replace("_yearly_avg_lag1","").replace("_yearly_avg_lag2","").replace("_yearly_avg_lag3","") for c in AGTc))
eta_topic={}
for ctry in eta.index:
    for tp in topics:
        cols=[f"{tp}_yearly_avg_lag{l}" for l in (1,2,3) if f"{tp}_yearly_avg_lag{l}" in eta.columns]
        eta_topic[(ctry,tp)]=float(np.nanmean(eta.loc[ctry,cols].values))
print(f"computed elasticities for {len(eta.index)} countries x {len(topics)} topics",flush=True)

# ---- monthly GT proportions (US) + yearly RD -> disaggregation ----
gt=pd.read_csv(os.path.join(ROOT,"data","GT","trends_data_by_topic_filtered.csv"))
gt["date"]=pd.to_datetime(gt["date"]); gt["Year"]=gt.date.dt.year; gt["Month"]=gt.date.dt.month
COUNTRY="US"
gcols={tp:f"{COUNTRY}_{tp}" for tp in topics if f"{COUNTRY}_{tp}" in gt.columns}
g=gt[["Year","Month"]+list(gcols.values())].copy()
# yearly RD (US), observed
rd=merged[merged.Country==COUNTRY].groupby("Year").rd_expenditure.mean()
years=sorted([y for y in g.Year.unique() if y in rd.index])
g=g[g.Year.isin(years)].copy()
# proportions per topic within each year (NaN-safe: zero-search topics contribute 0 share)
for tp,gc in gcols.items():
    denom=g.groupby("Year")[gc].transform("sum").values
    g[tp+"_p"]=np.where(denom>0, g[gc].values/np.where(denom>0,denom,1.0), 0.0)
# adjusted proportions with US elasticities
adj=np.zeros(len(g))
for tp in gcols:
    e=eta_topic[(COUNTRY,tp)]
    if np.isfinite(e): adj=adj+e*g[tp+"_p"].values
g["adj"]=adj
g["adj_year"]=g.groupby("Year")["adj"].transform("sum")
g["rd_year"]=g.Year.map(rd)
g["NN_temporal"]=g["rd_year"]*g["adj"]/g["adj_year"]

# ---- merge with classical estimators (split-independent) ----
ce=pd.read_csv(os.path.join(ROOT,"temporal_disaggregation","results","combined_estimates.csv"))
ce=ce[ce.Country==COUNTRY][["Year","Month","Monthly_RD_Expenditure_Tempdisagg_Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley"]]
df=g[["Year","Month","NN_temporal"]].merge(ce,on=["Year","Month"],how="inner").dropna().reset_index(drop=True)
df=df.rename(columns={"Monthly_RD_Expenditure_Tempdisagg_Sax":"Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley":"Mosley"})
df.to_csv(os.path.join(OUT,"combined_estimates_temporal.csv"),index=False)
N=len(df); print(f"\nUS monthly disaggregation: {df.Year.min()}-{df.Year.max()}, N={N}")

def corr(a,b):
    r,p=stats.pearsonr(a,b); return r,p
def g_(x): return np.diff(x)/x[:-1]
print("\n=== Correlations (temporal NN-elasticity) ===")
for nm,col in [("Mosley","Mosley"),("Sax","Sax")]:
    rl,pl=corr(df.NN_temporal.values,df[col].values)
    rg,pg=corr(g_(df.NN_temporal.values),g_(df[col].values))
    print(f"NN vs {nm:7s}: level r({N-2})={rl:.2f} (p={pl:.1e}) | growth r={rg:.2f} (p={pg:.2f})")
rl,pl=corr(df.Sax.values,df.Mosley.values); rg,pg=corr(g_(df.Sax.values),g_(df.Mosley.values))
print(f"Sax vs Mosley: level r={rl:.2f} (p={pl:.1e}) | growth r={rg:.2f} (p={pg:.2f})")

# ---- employment validation (Table 5) ----
emp=pd.read_csv(os.path.join(ROOT,"data","datausa.io","Monthly Employment.csv"))
emp["date"]=pd.to_datetime(emp["Date"]); emp=emp[["date","NSA Employees"]].rename(columns={"NSA Employees":"emp"})
df["date"]=pd.to_datetime(dict(year=df.Year,month=df.Month,day=1))
m=df.merge(emp,on="date",how="inner").sort_values("date").reset_index(drop=True)
me=np.diff(m.emp.values)/m.emp.values[:-1]
def lagcorr(series):
    sg=np.diff(series)/series[:-1]; out=[]
    for L in range(-5,6):
        if L<0: a,b=sg[-L:],me[:L]
        elif L>0: a,b=sg[:-L],me[L:]
        else: a,b=sg,me
        if len(a)>5:
            r,p=stats.pearsonr(a,b); out.append((L,r,p))
    return out
print("\n=== Employment growth lag-correlations (|p|<0.01), temporal NN ===")
for L,r,p in sorted(lagcorr(m.NN_temporal.values),key=lambda x:-abs(x[1])):
    if p<0.01: print(f"  lag {L:+d}: r={r:.2f} p={p:.4f}")

# ---- figure (temporal NN series for US) ----
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(df.date,df.NN_temporal,label="NN-elasticity (temporal)",color="#2c7fb8")
ax.plot(df.date,df.Mosley,label="Sparse (Mosley)",color="#31a354",alpha=.8)
ax.plot(df.date,df.Sax,label="Chow-Lin (Sax)",color="#d95f0e",alpha=.8)
ax.set_ylabel("Monthly R&D (USD bn)"); ax.set_title("US monthly R&D: NN-elasticity (temporal split) vs classical"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"temporal_disagg_US.png"),dpi=150)
print("\nsaved out/combined_estimates_temporal.csv + out/temporal_disagg_US.png")
