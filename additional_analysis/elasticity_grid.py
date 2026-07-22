"""
Isolate what drives the agreement between our NN-elasticity disaggregation and the
classical estimators (Chow-Lin/Sax, sparse/Mosley): the train-test SPLIT or the
TARGET representation used to fit the AGT elasticity model.

Grid: split in {random(row-level, leaky), temporal(year-grouped), alldata} x
      target in {raw level, within-country log-standardized}.
For each cell: train AGT ensemble (5), compute LEVEL elasticities
(perturb each standardized feature +1%, relative change in recovered LEVEL /0.01,
avg over the cell's training rows and members), average over 3 lags -> per-topic,
disaggregate US, and correlate with Sax/Mosley.
"""
import os, numpy as np, pandas as pd
from scipy import stats
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
SEED=0
merged=pd.read_csv(os.path.join(OUT,"merged_features.csv")); merged=merged[merged.Year>=2004].copy()
merged.sort_values(["Country","Year","Month"],inplace=True)
AGTc=[c for c in merged.columns if "_yearly_avg_lag" in c]
months=pd.get_dummies(merged.Month,prefix="M").astype(float)
le=LabelEncoder(); cc=le.fit_transform(merged.Country)
y_level=merged.rd_expenditure.values.astype(float); logy=np.log(y_level)
ctry_arr=merged.Country.values

def make_split(kind):
    rng=np.random.default_rng(SEED); tr=np.zeros(len(merged),bool); va=np.zeros(len(merged),bool)
    if kind=="random":            # row-level random per country (leaky)
        for c in np.unique(ctry_arr):
            idx=np.where(ctry_arr==c)[0]; rng.shuffle(idx)
            n=len(idx); tr[idx[:int(.64*n)]]=True; va[idx[int(.64*n):int(.80*n)]]=True
    elif kind=="temporal":        # year-grouped chronological
        for c in np.unique(ctry_arr):
            g=merged[merged.Country==c]; yrs=np.array(sorted(g.Year.unique())); n=len(yrs)
            tryrs=set(yrs[:int(round(.64*n))]); vay=set(yrs[int(round(.64*n)):int(round(.80*n))])
            m=(ctry_arr==c)
            tr|=m&merged.Year.isin(tryrs).values; va|=m&merged.Year.isin(vay).values
    else:                          # alldata: all train, random 15% val for early stop
        va=rng.random(len(merged))<0.15; tr=~va
    return tr,va

class MLP(nn.Module):
    def __init__(s,d,h,nc,emb=4,p=0.1):
        super().__init__(); s.emb=nn.Embedding(nc,emb); dims=[d+emb]+h
        s.lin=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(len(h)))
        s.bn=nn.ModuleList(nn.BatchNorm1d(x) for x in h); s.out=nn.Linear(dims[-1],1); s.drop=nn.Dropout(p); s.relu=nn.ReLU()
    def forward(s,x,c):
        x=torch.cat([x,s.emb(c)],1)
        for lin,bn in zip(s.lin,s.bn): x=s.drop(s.relu(bn(lin(x))))
        return s.out(x)

def run(kind_split, target, size=5):
    tr,va=make_split(kind_split)
    # target + per-row inverse-to-level
    if target=="level":
        ymu,ysd=y_level[tr].mean(),y_level[tr].std(); yt=(y_level-ymu)/ysd
        to_level=lambda z: z*ysd+ymu
    else:  # logstd_country
        cm=np.zeros(len(merged)); cs=np.ones(len(merged))
        for c in np.unique(ctry_arr):
            m=ctry_arr==c; v=logy[m&tr]; cm[m]=v.mean(); cs[m]=max(v.std(),0.05)
        yt=(logy-cm)/cs; to_level=lambda z: np.exp(z*cs+cm)
    sc=StandardScaler().fit(merged[AGTc].fillna(0).astype(float).values[tr])
    X=np.hstack([sc.transform(merged[AGTc].fillna(0).astype(float).values),months.values]); n_gt=len(AGTc)
    def T(a): return torch.FloatTensor(a)
    Xtr=T(X[tr]);Xva=T(X[va]);ctr=torch.LongTensor(cc[tr]);cva=torch.LongTensor(cc[va])
    ytr=T(yt[tr].reshape(-1,1));yva=T(yt[va].reshape(-1,1)); call=torch.LongTensor(cc); Xall=T(X)
    models=[]
    for m in range(size):
        torch.manual_seed(m); net=MLP(X.shape[1],[200,20,20],len(le.classes_))
        opt=optim.AdamW(net.parameters(),lr=0.01,weight_decay=1e-4); sch=MultiStepLR(opt,[300],0.1); crit=nn.MSELoss()
        best=np.inf;bad=0;bstate=None
        for ep in range(500):
            net.train(); perm=torch.randperm(len(Xtr))
            for i in range(0,len(Xtr),64):
                idx=perm[i:i+64]; opt.zero_grad(); loss=crit(net(Xtr[idx],ctr[idx]),ytr[idx]); loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad(): vl=crit(net(Xva,cva),yva).item()
            if vl<best-1e-7: best=vl;bad=0;bstate={k:v.clone() for k,v in net.state_dict().items()}
            else:
                bad+=1
                if bad>=60: break
        if bstate: net.load_state_dict(bstate); net.eval(); models.append(net)
    # level elasticities, averaged over members + the cell's TRAIN rows
    el_sum=np.zeros((len(merged),n_gt))
    for net in models:
        with torch.no_grad(): base=to_level(net(Xall,call).numpy().ravel())
        for j in range(n_gt):
            Xp=X.copy(); Xp[:,j]*=1.01
            with torch.no_grad(): pert=to_level(net(T(Xp),call).numpy().ravel())
            el_sum[:,j]+=((pert-base)/base)/0.01
    el=el_sum/len(models)
    eldf=pd.DataFrame(el,columns=AGTc); eldf["Country"]=ctry_arr; eldf["tr"]=tr
    eta=eldf[eldf.tr].groupby("Country")[AGTc].mean()
    topics=sorted(set(c.replace("_yearly_avg_lag1","").replace("_yearly_avg_lag2","").replace("_yearly_avg_lag3","") for c in AGTc))
    etaU={t:float(np.nanmean(eta.loc["US",[f"{t}_yearly_avg_lag{l}" for l in (1,2,3) if f"{t}_yearly_avg_lag{l}" in eta.columns]].values)) for t in topics}
    # disaggregate US
    gt=pd.read_csv(os.path.join(ROOT,"data","GT","trends_data_by_topic_filtered.csv"))
    gt["date"]=pd.to_datetime(gt["date"]); gt["Year"]=gt.date.dt.year; gt["Month"]=gt.date.dt.month
    rd=merged[merged.Country=="US"].groupby("Year").rd_expenditure.mean()
    tp_ok=[t for t in topics if f"US_{t}" in gt.columns]
    g=gt[["Year","Month"]+[f"US_{t}" for t in tp_ok]].copy(); g=g[g.Year.isin([y for y in g.Year.unique() if y in rd.index])].copy()
    adj=np.zeros(len(g))
    for t in tp_ok:
        gc=f"US_{t}"; den=g.groupby("Year")[gc].transform("sum").values
        p=np.where(den>0,g[gc].values/np.where(den>0,den,1.0),0.0)
        if np.isfinite(etaU[t]): adj+=etaU[t]*p
    g["adj"]=adj; g["ay"]=g.groupby("Year")["adj"].transform("sum"); g["NN"]=g.Year.map(rd)*g["adj"]/g["ay"]
    ce=pd.read_csv(os.path.join(ROOT,"temporal_disaggregation","results","combined_estimates.csv"))
    ce=ce[ce.Country=="US"][["Year","Month","Monthly_RD_Expenditure_Tempdisagg_Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley"]]
    df=g[["Year","Month","NN"]].merge(ce,on=["Year","Month"],how="inner").dropna()
    df=df.rename(columns={"Monthly_RD_Expenditure_Tempdisagg_Sax":"Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley":"Mosley"})
    def cc_(a,b): return stats.pearsonr(a,b)[0]
    gr=lambda x:np.diff(x)/x[:-1]
    return dict(split=kind_split,target=target,N=len(df),
        Mosley_lvl=cc_(df.NN,df.Mosley),Mosley_gr=cc_(gr(df.NN.values),gr(df.Mosley.values)),
        Sax_lvl=cc_(df.NN,df.Sax),Sax_gr=cc_(gr(df.NN.values),gr(df.Sax.values)))

rows=[]
for sp in ["random","temporal","alldata"]:
    for tg in ["level","logstd_country"]:
        r=run(sp,tg); rows.append(r)
        print(f"split={sp:8s} target={tg:14s} N={r['N']} | NN-Mosley lvl={r['Mosley_lvl']:.2f} gr={r['Mosley_gr']:.2f} | NN-Sax lvl={r['Sax_lvl']:.2f} gr={r['Sax_gr']:.2f}",flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT,"elasticity_grid.csv"),index=False)
print("saved out/elasticity_grid.csv")
