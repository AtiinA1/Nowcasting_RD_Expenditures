"""
Train the temporal-split ensembles (within-country log-standardized target) and
save per-member, per-row test predictions in levels -- same schema as
cy_split_predictions.csv -- so the paper artifacts (box plots, fan chart,
coverage, NN-vs-OLS, SHAP) can be regenerated for the temporal split.

Output: out/temporal_split_predictions.csv
"""
import os, numpy as np, pandas as pd
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
logy=np.log(merged.rd_expenditure.values.astype(float))
cmean,cstd={},{}
for ctry,g in merged.groupby("Country"):
    v=np.log(g.rd_expenditure.values[(g.split=="train").values]); cmean[ctry]=v.mean(); cstd[ctry]=max(v.std(),0.05)
cm=merged.Country.map(cmean).values; cs=merged.Country.map(cstd).values
ystd=(logy-cm)/cs

class MLP(nn.Module):
    def __init__(s,d,h,nc,emb=4,p=0.1):
        super().__init__(); s.emb=nn.Embedding(nc,emb); dims=[d+emb]+h
        s.lin=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(len(h)))
        s.bn=nn.ModuleList(nn.BatchNorm1d(x) for x in h); s.out=nn.Linear(dims[-1],1); s.drop=nn.Dropout(p); s.relu=nn.ReLU()
    def forward(s,x,c):
        x=torch.cat([x,s.emb(c)],1)
        for lin,bn in zip(s.lin,s.bn): x=s.drop(s.relu(bn(lin(x))))
        return s.out(x)

def train(cols,size=10,epochs=600,patience=70):
    feat=merged[cols].fillna(0).astype(float).values
    sc=StandardScaler().fit(feat[masks["train"]]); X=np.hstack([sc.transform(feat),months.values])
    cc=merged.cc.values
    def T(a): return torch.FloatTensor(a)
    Xtr,Xva,Xte=T(X[masks["train"]]),T(X[masks["val"]]),T(X[masks["test"]])
    ctr=torch.LongTensor(cc[masks["train"]]);cva=torch.LongTensor(cc[masks["val"]]);cte=torch.LongTensor(cc[masks["test"]])
    ytr=T(ystd[masks["train"]].reshape(-1,1)); yva=T(ystd[masks["val"]].reshape(-1,1))
    cs_te=cs[masks["test"]]; cm_te=cm[masks["test"]]; zmembers=[]
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
        with torch.no_grad(): zmembers.append(net(Xte,cte).numpy().ravel())
        print(f"      member {m+1}/{size}",flush=True)
    Z=np.column_stack(zmembers)                       # (n_test, size) standardized-log preds
    member_levels=np.exp(Z*cs_te[:,None]+cm_te[:,None])
    mean_level=np.exp(Z.mean(1)*cs_te+cm_te)
    return member_levels, mean_level

meta=merged[masks["test"]][["Country","Year","Month","rd_expenditure"]].reset_index(drop=True)
rows=[]
for cfg in CONFIGS:
    print(f"=== {cfg} ===",flush=True)
    ml,mean=train(CONFIGS[cfg])
    for i in range(len(meta)):
        d=dict(Config=cfg,Country=meta.Country[i],Year=int(meta.Year[i]),Month=int(meta.Month[i]),
               True_Values=meta.rd_expenditure[i],pred_mean=mean[i])
        for k in range(ml.shape[1]): d[f"m{k}"]=ml[i,k]
        rows.append(d)
pd.DataFrame(rows).to_csv(os.path.join(OUT,"temporal_split_predictions.csv"),index=False)
print("saved temporal_split_predictions.csv")
