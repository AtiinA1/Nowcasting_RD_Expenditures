"""
Step-B (NN-elasticity) disaggregation using ALL-DATA elasticities (AGT model
trained on all available years). Rationale: temporal disaggregation is a
retrospective, within-sample allocation of KNOWN annual totals -- like Chow-Lin
and the sparse (Mosley) estimators, which are themselves fit on the full sample
-- so the elasticities (the analog of the indicator loadings) should be
estimated on all data, not the temporal training subset. A random 15% holdout
is used only for early stopping; elasticities are averaged over ALL samples.

Compares correlations with the classical (split-independent) estimators and the
employment series against the random-split and temporal-split scenarios.
"""
import os, numpy as np, pandas as pd
from scipy import stats
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED)
merged=pd.read_csv(os.path.join(OUT,"merged_features.csv")); merged=merged[merged.Year>=2004].copy()
merged.sort_values(["Country","Year","Month"],inplace=True)
AGTc=[c for c in merged.columns if "_yearly_avg_lag" in c]

# ALL-DATA: random 15% holdout only for early stopping
rng=np.random.default_rng(SEED)
val_mask=rng.random(len(merged))<0.15
months=pd.get_dummies(merged.Month,prefix="M").astype(float)
le=LabelEncoder(); merged["cc"]=le.fit_transform(merged.Country)
logy=np.log(merged.rd_expenditure.values.astype(float))
# per-country mean/std over ALL years
cmean,cstd={},{}
for ctry,g in merged.groupby("Country"):
    v=np.log(g.rd_expenditure.values); cmean[ctry]=v.mean(); cstd[ctry]=max(v.std(),0.05)
cm=merged.Country.map(cmean).values; cs=merged.Country.map(cstd).values; ystd=(logy-cm)/cs
sc=StandardScaler().fit(merged[AGTc].fillna(0).astype(float).values[~val_mask])
X=np.hstack([sc.transform(merged[AGTc].fillna(0).astype(float).values),months.values]); n_gt=len(AGTc)
cc=merged.cc.values
def T(a): return torch.FloatTensor(a)
Xtr=T(X[~val_mask]);Xva=T(X[val_mask]);ctr=torch.LongTensor(cc[~val_mask]);cva=torch.LongTensor(cc[val_mask])
ytr=T(ystd[~val_mask].reshape(-1,1));yva=T(ystd[val_mask].reshape(-1,1))
class MLP(nn.Module):
    def __init__(s,d,h,nc,emb=4,p=0.1):
        super().__init__(); s.emb=nn.Embedding(nc,emb); dims=[d+emb]+h
        s.lin=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(len(h)))
        s.bn=nn.ModuleList(nn.BatchNorm1d(x) for x in h); s.out=nn.Linear(dims[-1],1); s.drop=nn.Dropout(p); s.relu=nn.ReLU()
    def forward(s,x,c):
        x=torch.cat([x,s.emb(c)],1)
        for lin,bn in zip(s.lin,s.bn): x=s.drop(s.relu(bn(lin(x))))
        return s.out(x)
print("training all-data AGT ensemble...",flush=True)
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
print("done",flush=True)

Xall=T(X); call=torch.LongTensor(cc)
def level_pred(net,Xt):
    with torch.no_grad(): z=net(Xt,call).numpy().ravel()
    return np.exp(z*cs+cm)
el_sum=np.zeros((len(merged),n_gt))
for net in models:
    base=level_pred(net,Xall)
    for j in range(n_gt):
        Xp=X.copy(); Xp[:,j]*=1.01; el_sum[:,j]+=((level_pred(net,T(Xp))-base)/base)/0.01
el=el_sum/len(models)
eldf=pd.DataFrame(el,columns=AGTc); eldf["Country"]=merged.Country.values
eta=eldf.groupby("Country")[AGTc].mean()                    # over ALL samples
topics=sorted(set(c.replace("_yearly_avg_lag1","").replace("_yearly_avg_lag2","").replace("_yearly_avg_lag3","") for c in AGTc))
eta_topic={}
for ctry in eta.index:
    for tp in topics:
        cols=[f"{tp}_yearly_avg_lag{l}" for l in (1,2,3) if f"{tp}_yearly_avg_lag{l}" in eta.columns]
        eta_topic[(ctry,tp)]=float(np.nanmean(eta.loc[ctry,cols].values))

gt=pd.read_csv(os.path.join(ROOT,"data","GT","trends_data_by_topic_filtered.csv"))
gt["date"]=pd.to_datetime(gt["date"]); gt["Year"]=gt.date.dt.year; gt["Month"]=gt.date.dt.month
COUNTRY="US"; gcols={tp:f"{COUNTRY}_{tp}" for tp in topics if f"{COUNTRY}_{tp}" in gt.columns}
g=gt[["Year","Month"]+list(gcols.values())].copy()
rd=merged[merged.Country==COUNTRY].groupby("Year").rd_expenditure.mean()
g=g[g.Year.isin([y for y in g.Year.unique() if y in rd.index])].copy()
for tp,gc in gcols.items():
    denom=g.groupby("Year")[gc].transform("sum").values
    g[tp+"_p"]=np.where(denom>0,g[gc].values/np.where(denom>0,denom,1.0),0.0)
adj=np.zeros(len(g))
for tp in gcols:
    e=eta_topic[(COUNTRY,tp)]
    if np.isfinite(e): adj=adj+e*g[tp+"_p"].values
g["adj"]=adj; g["adj_year"]=g.groupby("Year")["adj"].transform("sum"); g["rd_year"]=g.Year.map(rd)
g["NN_alldata"]=g["rd_year"]*g["adj"]/g["adj_year"]
ce=pd.read_csv(os.path.join(ROOT,"temporal_disaggregation","results","combined_estimates.csv"))
ce=ce[ce.Country==COUNTRY][["Year","Month","Monthly_RD_Expenditure_Tempdisagg_Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley"]]
df=g[["Year","Month","NN_alldata"]].merge(ce,on=["Year","Month"],how="inner").dropna().reset_index(drop=True)
df=df.rename(columns={"Monthly_RD_Expenditure_Tempdisagg_Sax":"Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley":"Mosley"})
df.to_csv(os.path.join(OUT,"combined_estimates_alldata.csv"),index=False)
N=len(df); print(f"\nUS disaggregation {df.Year.min()}-{df.Year.max()} N={N}")
def cr(a,b): r,p=stats.pearsonr(a,b); return r,p
def gr(x): return np.diff(x)/x[:-1]
for nm in ["Mosley","Sax"]:
    rl,pl=cr(df.NN_alldata.values,df[nm].values); rg,pg=cr(gr(df.NN_alldata.values),gr(df[nm].values))
    print(f"NN(all-data) vs {nm:7s}: level r({N-2})={rl:.2f} (p={pl:.1e}) | growth r={rg:.2f} (p={pg:.2f})")
# employment
emp=pd.read_csv(os.path.join(ROOT,"data","datausa.io","Monthly Employment.csv"))
emp["date"]=pd.to_datetime(emp["Date"]); emp=emp[["date","NSA Employees"]].rename(columns={"NSA Employees":"emp"})
df["date"]=pd.to_datetime(dict(year=df.Year,month=df.Month,day=1))
m=df.merge(emp,on="date",how="inner").sort_values("date").reset_index(drop=True)
me=np.diff(m.emp.values)/m.emp.values[:-1]; sg=np.diff(m.NN_alldata.values)/m.NN_alldata.values[:-1]
print("Employment lag-correlations (p<0.01):")
for L in range(-5,6):
    if L<0: a,b=sg[-L:],me[:L]
    elif L>0: a,b=sg[:-L],me[L:]
    else: a,b=sg,me
    r,p=stats.pearsonr(a,b)
    if p<0.01: print(f"  lag {L:+d}: r={r:.2f} p={p:.4f}")
print("saved out/combined_estimates_alldata.csv")
