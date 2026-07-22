"""
Step B production run, consistent with the temporal Step A: level-target AGT
elasticities under the TEMPORAL split (ensemble 10). Also computes the top
contributing topics under each split (random / temporal / all-data) for the
cross-split robustness/contributor analysis (the top features feed the Chow-Lin
estimator, per the original text).

Outputs (additional_analysis/out/):
  combined_estimates_temporal_level.csv   US monthly NN(temporal,level) + Sax + Mosley
  stepB_correlations.csv                  per-split NN-Mosley / NN-Sax (level+growth)
  stepB_top_topics.csv                    per-split top-15 topics by |elasticity|
  stepB_employment_lags.csv               temporal-level NN employment lag-correlations
Figures (Disaggregation/): NNelasticity_temporal_level.png, employee_temporal_level.png
"""
import os, numpy as np, pandas as pd
from scipy import stats
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
FIG="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Disaggregation"
SEED=0
merged=pd.read_csv(os.path.join(OUT,"merged_features.csv")); merged=merged[merged.Year>=2004].copy()
merged.sort_values(["Country","Year","Month"],inplace=True)
AGTc=[c for c in merged.columns if "_yearly_avg_lag" in c]
months=pd.get_dummies(merged.Month,prefix="M").astype(float)
le=LabelEncoder(); cc=le.fit_transform(merged.Country)
y_level=merged.rd_expenditure.values.astype(float); ctry_arr=merged.Country.values
topics=sorted(set(c.replace("_yearly_avg_lag1","").replace("_yearly_avg_lag2","").replace("_yearly_avg_lag3","") for c in AGTc))
CHOWLIN6=["Capitalization","Investment_management","Patent_office","Tax_credit","Cost","Technology"]

def make_split(kind):
    rng=np.random.default_rng(SEED); tr=np.zeros(len(merged),bool); va=np.zeros(len(merged),bool)
    if kind=="random":
        for c in np.unique(ctry_arr):
            idx=np.where(ctry_arr==c)[0]; rng.shuffle(idx); n=len(idx)
            tr[idx[:int(.64*n)]]=True; va[idx[int(.64*n):int(.80*n)]]=True
    elif kind=="temporal":
        for c in np.unique(ctry_arr):
            yrs=np.array(sorted(merged[merged.Country==c].Year.unique())); n=len(yrs)
            tryrs=set(yrs[:int(round(.64*n))]); vay=set(yrs[int(round(.64*n)):int(round(.80*n))]); m=ctry_arr==c
            tr|=m&merged.Year.isin(tryrs).values; va|=m&merged.Year.isin(vay).values
    else:
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

gt=pd.read_csv(os.path.join(ROOT,"data","GT","trends_data_by_topic_filtered.csv"))
gt["date"]=pd.to_datetime(gt["date"]); gt["Year"]=gt.date.dt.year; gt["Month"]=gt.date.dt.month
rd=merged[merged.Country=="US"].groupby("Year").rd_expenditure.mean()
ce=pd.read_csv(os.path.join(ROOT,"temporal_disaggregation","results","combined_estimates.csv"))
ce=ce[ce.Country=="US"][["Year","Month","Monthly_RD_Expenditure_Tempdisagg_Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley"]]

def run(kind, size=10):
    tr,va=make_split(kind)
    ymu,ysd=y_level[tr].mean(),y_level[tr].std(); yt=(y_level-ymu)/ysd; to_level=lambda z:z*ysd+ymu
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
    el_sum=np.zeros((len(merged),n_gt))
    for net in models:
        with torch.no_grad(): base=to_level(net(Xall,call).numpy().ravel())
        for j in range(n_gt):
            Xp=X.copy(); Xp[:,j]*=1.01
            with torch.no_grad(): el_sum[:,j]+=((to_level(net(T(Xp),call).numpy().ravel())-base)/base)/0.01
    el=el_sum/len(models); eldf=pd.DataFrame(el,columns=AGTc); eldf["Country"]=ctry_arr; eldf["tr"]=tr
    eta=eldf[eldf.tr].groupby("Country")[AGTc].mean()
    etaU={t:float(np.nanmean(eta.loc["US",[f"{t}_yearly_avg_lag{l}" for l in (1,2,3) if f"{t}_yearly_avg_lag{l}" in eta.columns]].values)) for t in topics}
    # disaggregate US
    tp_ok=[t for t in topics if f"US_{t}" in gt.columns]
    g=gt[["Year","Month"]+[f"US_{t}" for t in tp_ok]].copy(); g=g[g.Year.isin([y for y in g.Year.unique() if y in rd.index])].copy()
    adj=np.zeros(len(g))
    for t in tp_ok:
        gc=f"US_{t}"; den=g.groupby("Year")[gc].transform("sum").values
        p=np.where(den>0,g[gc].values/np.where(den>0,den,1.0),0.0)
        if np.isfinite(etaU[t]): adj+=etaU[t]*p
    g["adj"]=adj; g["ay"]=g.groupby("Year")["adj"].transform("sum"); g["NN"]=g.Year.map(rd)*g["adj"]/g["ay"]
    df=g[["Year","Month","NN"]].merge(ce,on=["Year","Month"],how="inner").dropna().reset_index(drop=True)
    df=df.rename(columns={"Monthly_RD_Expenditure_Tempdisagg_Sax":"Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley":"Mosley"})
    cc_=lambda a,b: stats.pearsonr(a,b)[0]; gr=lambda x:np.diff(x)/x[:-1]
    corr=dict(split=kind,N=len(df),Mosley_lvl=cc_(df.NN,df.Mosley),Mosley_gr=cc_(gr(df.NN.values),gr(df.Mosley.values)),
              Sax_lvl=cc_(df.NN,df.Sax),Sax_gr=cc_(gr(df.NN.values),gr(df.Sax.values)))
    top=sorted(etaU.items(),key=lambda kv:-abs(kv[1]))
    return corr, top, etaU, df

allcorr=[]; alltop=[]
prod_df=None
for kind in ["temporal","random","alldata"]:
    print(f"=== {kind} ===",flush=True)
    corr,top,etaU,df=run(kind)
    allcorr.append(corr)
    for r,(tp,v) in enumerate(top[:15],1): alltop.append(dict(split=kind,rank=r,topic=tp,abs_elasticity=abs(v),elasticity=v))
    print(f"  NN-Mosley lvl={corr['Mosley_lvl']:.2f} gr={corr['Mosley_gr']:.2f} | NN-Sax lvl={corr['Sax_lvl']:.2f} gr={corr['Sax_gr']:.2f}",flush=True)
    print("  top10 topics:", [t for t,_ in top[:10]],flush=True)
    if kind=="temporal": prod_df=df.copy()
pd.DataFrame(allcorr).to_csv(os.path.join(OUT,"stepB_correlations.csv"),index=False)
topdf=pd.DataFrame(alltop); topdf.to_csv(os.path.join(OUT,"stepB_top_topics.csv"),index=False)

# overlap of top-6 across splits + vs Chow-Lin6
print("\n=== top-6 topics by |elasticity| per split ===")
for kind in ["temporal","random","alldata"]:
    t6=topdf[(topdf.split==kind)&(topdf['rank']<=6)].topic.tolist(); print(f"  {kind}: {t6}")
print("  Chow-Lin (original, SHAP top-6):", CHOWLIN6)

# production: temporal-level series + correlations + employment + figure
prod_df["date"]=pd.to_datetime(dict(year=prod_df.Year,month=prod_df.Month,day=1))
prod_df.to_csv(os.path.join(OUT,"combined_estimates_temporal_level.csv"),index=False)
emp=pd.read_csv(os.path.join(ROOT,"data","datausa.io","Monthly Employment.csv")); emp["date"]=pd.to_datetime(emp["Date"])
emp=emp[["date","NSA Employees"]].rename(columns={"NSA Employees":"emp"})
m=prod_df.merge(emp,on="date",how="inner").sort_values("date").reset_index(drop=True)
me=np.diff(m.emp.values)/m.emp.values[:-1]
def lc(series):
    sg=np.diff(series)/series[:-1]; out=[]
    for L in range(-5,6):
        a,b=(sg[-L:],me[:L]) if L<0 else ((sg[:-L],me[L:]) if L>0 else (sg,me))
        r,p=stats.pearsonr(a,b); out.append((L,r,p))
    return out
emprows=[{"Lag":L,"Correlation":round(r,2),"P_Value":p} for L,r,p in lc(m.NN.values)]
pd.DataFrame(emprows).to_csv(os.path.join(OUT,"stepB_employment_lags.csv"),index=False)
print("\n=== temporal-level NN employment lag-correlations (p<0.01) ===")
for d in sorted(emprows,key=lambda x:-abs(x["Correlation"])):
    if d["P_Value"]<0.01: print(f"  lag {d['Lag']:+d}: r={d['Correlation']} p={d['P_Value']:.4f}")

import matplotlib.dates as mdates
fig,ax=plt.subplots(figsize=(7,4))
_mn=prod_df[["NN","Sax","Mosley"]].min(axis=1); _mx=prod_df[["NN","Sax","Mosley"]].max(axis=1)
ax.fill_between(prod_df.date,_mn,_mx,color="gray",alpha=0.3,label="Range (Min-Max)")
ax.plot(prod_df.date,prod_df.NN,color="#2c6fbb",marker="o",ms=3,lw=1,label="NN-elasticity")
ax.set_ylabel("Monthly R&D (USD bn)"); ax.set_title("NN-driven elasticity-based estimate (temporal split, level target)")
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True,alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"NNelasticity_temporal_level.png"),dpi=200); plt.close()
fig,ax1=plt.subplots(figsize=(7,4))
ax1.plot(m.date,m.NN,color="blue",label="NN-elasticity R&D"); ax1.set_ylabel("Monthly R&D (USD bn)",color="blue")
ax2=ax1.twinx(); ax2.plot(m.date,m.emp,color="red",alpha=.7); ax2.set_ylabel("Employees",color="red")
plt.title("US monthly R&D (temporal-level NN) vs R&D-services employment"); fig.tight_layout()
plt.savefig(os.path.join(FIG,"employee_temporal_level.png"),dpi=200); plt.close()
print("\nsaved combined_estimates_temporal_level.csv, stepB_*.csv, figures")
