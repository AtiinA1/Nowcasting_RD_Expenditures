"""
Full temporal-split evaluation with within-country log-standardized targets
(target mode 'logstd_country', selected by temporal_experiment.py).

For each of the 7 configurations:
  - train a 10-member ensemble NN  (per-row monthly, annual target),
  - fit an OLS at the SAME input space (ridge) on the same target transform,
and compare to econometric benchmarks (RW, AR(1), historical mean, and GT-only
U-MIDAS / MIDAS-Beta) on the identical chronological test set.

Reports level MAPE/RMSE/out-of-sample R^2 and Diebold-Mariano tests, and isolates
the GT-only comparison (NN-AGT vs OLS-AGT vs MIDAS-GT).
Outputs: out/temporal_skill.csv, temporal_nn_vs_ols.csv, temporal_dm.csv
"""
import os, numpy as np, pandas as pd
from scipy import stats, optimize
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"; OUT = os.path.join(ROOT, "additional_analysis", "out")
SEED = 0; np.random.seed(SEED); torch.manual_seed(SEED)
merged = pd.read_csv(os.path.join(OUT, "merged_features.csv"))
merged = merged[merged.Year >= 2004].copy(); merged.sort_values(["Country","Year","Month"], inplace=True)

AR  = [f"rd_expenditure_lag{l}" for l in (1,2,3)]
MAC = [f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc=[c for c in merged.columns if "_yearly_avg_lag" in c]; YTD=[c for c in merged.columns if c.endswith("_mean_YTD")]
CONFIGS={"LagRD":AR,"Macros":AR+MAC,"AGT":AGTc,"MGT":AGTc+YTD,"AGTwRD":AR+AGTc,"MGTwRD":AR+AGTc+YTD,"AllVar":AR+MAC+AGTc+YTD}

split={}
for ctry,g in merged.groupby("Country"):
    yrs=np.array(sorted(g.Year.unique())); n=len(yrs); n_tr=int(round(n*0.64)); n_va=int(round(n*0.16))
    for i,y in enumerate(yrs): split[(ctry,int(y))]="train" if i<n_tr else ("val" if i<n_tr+n_va else "test")
merged["split"]=[split[(c,int(y))] for c,y in zip(merged.Country,merged.Year)]
months=pd.get_dummies(merged.Month,prefix="M").astype(float)
le=LabelEncoder(); merged["cc"]=le.fit_transform(merged.Country)
masks={s:(merged.split==s).values for s in ("train","val","test")}
y_level=merged.rd_expenditure.values.astype(float); logy=np.log(y_level)
cmean,cstd={},{}
for ctry,g in merged.groupby("Country"):
    v=np.log(g.rd_expenditure.values[(g.split=="train").values]); cmean[ctry]=v.mean(); cstd[ctry]=max(v.std(),0.05)
cm=merged.Country.map(cmean).values; cs=merged.Country.map(cstd).values
ystd=(logy-cm)/cs                                   # within-country standardized log target
fitmask = masks["train"] | masks["val"]
test_meta=merged[masks["test"]][["Country","Year"]].reset_index(drop=True)
y_test=y_level[masks["test"]]

class MLP(nn.Module):
    def __init__(s,d,h,nc,emb=4,p=0.1):
        super().__init__(); s.emb=nn.Embedding(nc,emb); dims=[d+emb]+h
        s.lin=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(len(h)))
        s.bn=nn.ModuleList(nn.BatchNorm1d(x) for x in h); s.out=nn.Linear(dims[-1],1); s.drop=nn.Dropout(p); s.relu=nn.ReLU()
    def forward(s,x,c):
        x=torch.cat([x,s.emb(c)],1)
        for lin,bn in zip(s.lin,s.bn): x=s.drop(s.relu(bn(lin(x))))
        return s.out(x)

def nn_predict(cols,size=10,epochs=600,patience=70):
    feat=merged[cols].fillna(0).astype(float).values
    sc=StandardScaler().fit(feat[masks["train"]]); X=np.hstack([sc.transform(feat),months.values])
    cc=merged.cc.values
    def T(a): return torch.FloatTensor(a)
    Xtr,Xva,Xte=T(X[masks["train"]]),T(X[masks["val"]]),T(X[masks["test"]])
    ctr=torch.LongTensor(cc[masks["train"]]);cva=torch.LongTensor(cc[masks["val"]]);cte=torch.LongTensor(cc[masks["test"]])
    ytr=T(ystd[masks["train"]].reshape(-1,1)); yva=T(ystd[masks["val"]].reshape(-1,1))
    preds=[]
    for m in range(size):
        torch.manual_seed(m); net=MLP(X.shape[1],[200,20,20],len(le.classes_))
        opt=optim.AdamW(net.parameters(),lr=0.01,weight_decay=1e-4); sch=MultiStepLR(opt,[300],0.1)
        crit=nn.MSELoss(); best=np.inf; bad=0; bs=64; bstate=None
        for ep in range(epochs):
            net.train(); perm=torch.randperm(len(Xtr))
            for i in range(0,len(Xtr),bs):
                idx=perm[i:i+bs]; opt.zero_grad(); loss=crit(net(Xtr[idx],ctr[idx]),ytr[idx]); loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad(): vl=crit(net(Xva,cva),yva).item()
            if vl<best-1e-7: best=vl;bad=0;bstate={k:v.clone() for k,v in net.state_dict().items()}
            else:
                bad+=1
                if bad>=patience: break
        if bstate: net.load_state_dict(bstate); net.eval()
        with torch.no_grad(): preds.append(net(Xte,cte).numpy().ravel())
    z=np.mean(preds,0); return np.exp(z*cs[masks["test"]]+cm[masks["test"]])

def ols_predict(cols):
    feat=merged[cols].fillna(0).astype(float).values
    mu,sd=feat[masks["train"]].mean(0),feat[masks["train"]].std(0); sd[sd==0]=1
    cdum=pd.get_dummies(merged.Country,prefix="c").astype(float).values
    X=np.column_stack([np.ones(len(merged)),(feat-mu)/sd,months.values,cdum])
    beta=np.linalg.solve(X[fitmask].T@X[fitmask]+1e-2*np.eye(X.shape[1]), X[fitmask].T@ystd[fitmask])
    z=(X@beta)[masks["test"]]; return np.exp(z*cs[masks["test"]]+cm[masks["test"]])

# ---- econometric benchmarks (annual) ----
panel=merged.groupby(["Country","Year"]).rd_expenditure.mean().reset_index().rename(columns={"rd_expenditure":"GERD"})
panel["istest"]=[split[(c,int(y))]=="test" for c,y in zip(panel.Country,panel.Year)]
def prev(c,y):
    g=panel[(panel.Country==c)&(panel.Year<y)].sort_values("Year"); return g.GERD.iloc[-1] if len(g) else np.nan
def ar1(c,y):
    s=panel[(panel.Country==c)&(~panel.istest)&(panel.Year<y)].sort_values("Year").GERD.values
    if len(s)<4: return prev(c,y)
    ly=np.log(s); b=np.linalg.lstsq(np.column_stack([np.ones(len(ly)-1),ly[:-1]]),ly[1:],rcond=None)[0]
    p=prev(c,y); return np.exp(b[0]+b[1]*np.log(p)) if p and p>0 else np.nan
def hmean(c,y):
    g=panel[(panel.Country==c)&(~panel.istest)&(panel.Year<y)]; return g.GERD.mean() if len(g) else np.nan

# GT-only MIDAS / U-MIDAS on monthly composite, target = within-country std log
gt=pd.read_csv(os.path.join(ROOT,"data","GT","trends_data_by_topic_filtered.csv"))
gt["date"]=pd.to_datetime(gt["date"]); gt["Year"]=gt.date.dt.year; gt["Month"]=gt.date.dt.month
lo=gt.melt(id_vars=["date","Year","Month"],var_name="ck",value_name="v"); lo[["Country","topic"]]=lo.ck.str.split("_",n=1,expand=True)
comp=lo.groupby(["Country","Year","Month"]).v.mean().reset_index()
comp["z"]=comp.groupby("Country").v.transform(lambda s:(s-s.mean())/s.std())
wide=comp.pivot_table(index=["Country","Year"],columns="Month",values="z").reset_index()
wide.columns=["Country","Year"]+[f"gt{m}" for m in range(1,13)]
reg=panel.merge(wide,on=["Country","Year"],how="left").dropna(subset=[f"gt{m}" for m in range(1,13)]).reset_index(drop=True)
reg["ystd"]=[(np.log(G)-cmean[c])/cstd[c] for c,G in zip(reg.Country,reg.GERD)]
reg["istest"]=[split[(c,int(y))]=="test" for c,y in zip(reg.Country,reg.Year)]
cdr=pd.get_dummies(reg.Country,prefix="c").astype(float).values; GT=reg[[f"gt{m}" for m in range(1,13)]].values
trm=~reg.istest.values
def lvl_from_z(z): return np.array([np.exp(zz*cstd[c]+cmean[c]) for zz,c in zip(z,reg.Country)])
# U-MIDAS
Xu=np.column_stack([np.ones(len(reg)),cdr,GT]); bu=np.linalg.solve(Xu[trm].T@Xu[trm]+1e-3*np.eye(Xu.shape[1]),Xu[trm].T@reg.ystd.values[trm])
reg["UMIDAS"]=lvl_from_z(Xu@bu)
# MIDAS Beta
def bw(p):
    t1,t2=np.exp(p); x=np.linspace(1e-3,1,12); w=x**(t1-1)*(1-x)**(t2-1); return w/w.sum()
def mres(p):
    idx=GT@bw(p); X=np.column_stack([np.ones(len(reg)),cdr,idx]); b=np.linalg.solve(X[trm].T@X[trm]+1e-3*np.eye(X.shape[1]),X[trm].T@reg.ystd.values[trm]); return reg.ystd.values[trm]-X[trm]@b
r=optimize.least_squares(mres,np.log([2.,2.]),method="lm",max_nfev=2000); idx=GT@bw(r.x)
Xm=np.column_stack([np.ones(len(reg)),cdr,idx]); bm=np.linalg.solve(Xm[trm].T@Xm[trm]+1e-3*np.eye(Xm.shape[1]),Xm[trm].T@reg.ystd.values[trm])
reg["MIDAS"]=lvl_from_z(Xm@bm)
midas_map={(c,int(y)):(u,m) for c,y,u,m in zip(reg.Country,reg.Year,reg.UMIDAS,reg.MIDAS)}

# ---- collect annual predictions ----
ann=test_meta.copy(); ann["GERD"]=y_test
ann=ann.groupby(["Country","Year"]).GERD.mean().reset_index()
def to_annual(levelpreds):
    d=test_meta.copy(); d["p"]=levelpreds; return d.groupby(["Country","Year"]).p.mean().reset_index()
print(f"[temporal] test country-years={len(ann)}",flush=True)
nn_ann, ols_ann={}, {}
for cfg in CONFIGS:
    nn_ann[cfg]=to_annual(nn_predict(CONFIGS[cfg])).set_index(["Country","Year"]).p
    ols_ann[cfg]=to_annual(ols_predict(CONFIGS[cfg])).set_index(["Country","Year"]).p
    print(f"  trained+OLS {cfg}",flush=True)
A=ann.set_index(["Country","Year"]).copy()
for cfg in CONFIGS: A[f"NN_{cfg}"]=nn_ann[cfg]; A[f"OLS_{cfg}"]=ols_ann[cfg]
A["RW"]=[prev(c,y) for c,y in A.index]; A["AR1"]=[ar1(c,y) for c,y in A.index]; A["HistMean"]=[hmean(c,y) for c,y in A.index]
A["UMIDAS"]=[midas_map.get((c,y),(np.nan,np.nan))[0] for c,y in A.index]
A["MIDAS"]=[midas_map.get((c,y),(np.nan,np.nan))[1] for c,y in A.index]
A=A.reset_index()

def sk(t,p):
    t=np.asarray(t,float);p=np.asarray(p,float);m=np.isfinite(t)&np.isfinite(p);t,p=t[m],p[m];e=t-p
    return np.sqrt(np.mean(e**2)),np.mean(np.abs(e/t))*100,1-np.sum(e**2)/np.sum((t-t.mean())**2)
mse_rw=sk(A.GERD,A.RW)[0]**2
rows=[]
for name in [f"NN_{c}" for c in CONFIGS]+["RW","AR1","HistMean","UMIDAS","MIDAS"]:
    rmse,mape,r2=sk(A.GERD,A[name]); rows.append({"Model":name,"MAPE":mape,"RMSE":rmse,"R2":r2,"OOSR2_RW":1-rmse**2/mse_rw})
skill=pd.DataFrame(rows).set_index("Model"); print("\n=== Temporal skill (annual, logstd target) ==="); print(skill.round(2).to_string())
skill.round(3).to_csv(os.path.join(OUT,"temporal_skill.csv"))

nvo=pd.DataFrame({"Config":list(CONFIGS),
    "NN_MAPE":[sk(A.GERD,A[f"NN_{c}"])[1] for c in CONFIGS],
    "OLS_MAPE":[sk(A.GERD,A[f"OLS_{c}"])[1] for c in CONFIGS],
    "NN_R2":[sk(A.GERD,A[f"NN_{c}"])[2] for c in CONFIGS],
    "OLS_R2":[sk(A.GERD,A[f"OLS_{c}"])[2] for c in CONFIGS]}).set_index("Config")
print("\n=== NN vs OLS (same input space), temporal split ==="); print(nvo.round(2).to_string())
nvo.round(3).to_csv(os.path.join(OUT,"temporal_nn_vs_ols.csv"))

def dm(t,p1,p2):
    t=np.asarray(t,float);p1=np.asarray(p1,float);p2=np.asarray(p2,float);m=np.isfinite(t)&np.isfinite(p1)&np.isfinite(p2)
    t,p1,p2=t[m],p1[m],p2[m];d=(t-p1)**2-(t-p2)**2;n=len(d)
    if n<5 or np.var(d)==0: return np.nan,np.nan
    s=d.mean()/np.sqrt(np.var(d,ddof=1)/n)*np.sqrt((n+1)/n); return s,2*(1-stats.t.cdf(abs(s),df=n-1))
print("\n=== DM tests (neg => first better) ===")
dmrows=[]
for c in CONFIGS:
    s,p=dm(A.GERD,A[f"NN_{c}"],A[f"OLS_{c}"]); dmrows.append({"pair":f"NN_{c} vs OLS_{c}","DM":s,"p":p})
for b in ["UMIDAS","MIDAS","RW","AR1"]:
    s,p=dm(A.GERD,A["NN_AGT"],A[b]); dmrows.append({"pair":f"NN_AGT vs {b}","DM":s,"p":p})
dmt=pd.DataFrame(dmrows); print(dmt.round(3).to_string(index=False)); dmt.round(3).to_csv(os.path.join(OUT,"temporal_dm.csv"),index=False)
A.to_csv(os.path.join(OUT,"temporal_annual_predictions.csv"),index=False)
print("\nDONE")
