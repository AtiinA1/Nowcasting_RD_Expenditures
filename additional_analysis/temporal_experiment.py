"""
Temporal-split experiment: can a target transformation rescue the GT-only
configurations under a strict chronological (real-time) evaluation, and can the
neural network beat linear benchmarks that use the SAME Google-trends information?

Temporal split: per country, years sorted ascending; first 64% of years -> train,
next 16% -> validation, last 20% -> test (whole country-years, so it is both
chronological and leakage-free).

Target modes (chosen to address the non-stationarity that crushes GT-only models
in raw levels):
  level          : predict GERD level                       (current paper)
  log            : predict log GERD, evaluate on levels
  logstd_country : predict within-country-standardized log GERD using TRAIN-period
                   mean/std (a country-specific location/scale constant, analogous
                   to the country embedding; no lagged target enters as a feature,
                   so the GT-only interpretation is preserved)

Reports level MAPE / RMSE / out-of-sample R^2 (vs sample mean) at the annual level.
Usage: python temporal_experiment.py [screen|full]
"""
import os, sys, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out"); os.makedirs(OUT, exist_ok=True)
SEED = 0; np.random.seed(SEED); torch.manual_seed(SEED)

merged = pd.read_csv(os.path.join(OUT, "merged_features.csv"))
merged = merged[merged.Year >= 2004].copy()
merged.sort_values(["Country", "Year", "Month"], inplace=True)

AR  = [f"rd_expenditure_lag{l}" for l in (1, 2, 3)]
MAC = [f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc = [c for c in merged.columns if "_yearly_avg_lag" in c]; YTD = [c for c in merged.columns if c.endswith("_mean_YTD")]
CONFIGS = {"LagRD":AR,"Macros":AR+MAC,"AGT":AGTc,"MGT":AGTc+YTD,"AGTwRD":AR+AGTc,"MGTwRD":AR+AGTc+YTD,"AllVar":AR+MAC+AGTc+YTD}

# ---- temporal split by whole country-years ----
split = {}
for ctry, g in merged.groupby("Country"):
    yrs = np.array(sorted(g.Year.unique())); n = len(yrs)
    n_tr = int(round(n*0.64)); n_va = int(round(n*0.16))
    for i, y in enumerate(yrs):
        split[(ctry, int(y))] = "train" if i < n_tr else ("val" if i < n_tr+n_va else "test")
merged["split"] = [split[(c, int(y))] for c, y in zip(merged.Country, merged.Year)]
months = pd.get_dummies(merged.Month, prefix="M").astype(float)
le = LabelEncoder(); merged["cc"] = le.fit_transform(merged.Country)
masks = {s: (merged.split == s).values for s in ("train", "val", "test")}
y_level = merged.rd_expenditure.values.astype(float)

# within-country train stats on log target
logy = np.log(y_level)
cmean, cstd = {}, {}
for ctry, g in merged.groupby("Country"):
    tr = (g.split == "train").values
    v = np.log(g.rd_expenditure.values[tr])
    cmean[ctry] = v.mean(); cstd[ctry] = max(v.std(), 0.05)
cm = merged.Country.map(cmean).values; cs = merged.Country.map(cstd).values

def make_target(mode):
    if mode == "level": return y_level.copy(), None
    if mode == "log":   return logy.copy(), None
    if mode == "logstd_country": return (logy - cm) / cs, None
    raise ValueError(mode)

def invert(mode, pred, cm_s=None, cs_s=None):
    if mode == "level": return pred
    if mode == "log":   return np.exp(pred)
    if mode == "logstd_country": return np.exp(pred * cs_s + cm_s)

class MLP(nn.Module):
    def __init__(self, d, hidden, nc, emb=4, p=0.1):
        super().__init__(); self.emb=nn.Embedding(nc, emb); dims=[d+emb]+hidden
        self.lin=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(len(hidden)))
        self.bn=nn.ModuleList(nn.BatchNorm1d(h) for h in hidden)
        self.out=nn.Linear(dims[-1],1); self.drop=nn.Dropout(p); self.relu=nn.ReLU()
    def forward(self,x,c):
        x=torch.cat([x,self.emb(c)],1)
        for lin,bn in zip(self.lin,self.bn): x=self.drop(self.relu(bn(lin(x))))
        return self.out(x)

def train_predict(cols, mode, size=5, epochs=600, patience=70, hidden=(200,20,20)):
    feat = merged[cols].fillna(0).astype(float).values
    sc = StandardScaler().fit(feat[masks["train"]]); X = np.hstack([sc.transform(feat), months.values])
    yt, _ = make_target(mode)
    # standardize target for optimization stability (always), invert later via target stats
    tmu, tsd = yt[masks["train"]].mean(), yt[masks["train"]].std(); tsd = max(tsd, 1e-6)
    ys = (yt - tmu) / tsd
    cc = merged.cc.values
    def T(a): return torch.FloatTensor(a)
    Xtr,Xva,Xte = T(X[masks["train"]]),T(X[masks["val"]]),T(X[masks["test"]])
    ctr=torch.LongTensor(cc[masks["train"]]);cva=torch.LongTensor(cc[masks["val"]]);cte=torch.LongTensor(cc[masks["test"]])
    ytr=T(ys[masks["train"]].reshape(-1,1)); yva=T(ys[masks["val"]].reshape(-1,1))
    preds=[]
    for m in range(size):
        torch.manual_seed(m); net=MLP(X.shape[1],list(hidden),len(le.classes_))
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
        if bstate: net.load_state_dict(bstate)
        net.eval()
        with torch.no_grad(): preds.append((net(Xte,cte).numpy().ravel()*tsd+tmu))
    pred_t = np.mean(preds,0)
    return invert(mode, pred_t, cm[masks["test"]], cs[masks["test"]])

# ---- annual collapse + metrics ----
test_meta = merged[masks["test"]][["Country","Year"]].reset_index(drop=True)
y_test_level = y_level[masks["test"]]
def annual(levelpreds):
    df = test_meta.copy(); df["t"]=y_test_level; df["p"]=levelpreds
    a = df.groupby(["Country","Year"]).agg(t=("t","mean"),p=("p","mean")).reset_index()
    return a
def metrics(a):
    e=a.t.values-a.p.values
    rmse=np.sqrt(np.mean(e**2)); mape=np.mean(np.abs(e/a.t.values))*100
    r2=1-np.sum(e**2)/np.sum((a.t.values-a.t.values.mean())**2)
    return rmse,mape,r2

mode_list = ["level","log","logstd_country"]
if len(sys.argv)>1 and sys.argv[1]=="full":
    cfg_list = list(CONFIGS)
else:
    cfg_list = ["Macros","AGT","MGT","AGTwRD","AllVar"]

print(f"Temporal split: train={masks['train'].sum()} val={masks['val'].sum()} test={masks['test'].sum()} rows; "
      f"test country-years={annual(np.zeros(masks['test'].sum())).shape[0]}", flush=True)
rows=[]
for cfg in cfg_list:
    for mode in mode_list:
        lp = train_predict(CONFIGS[cfg], mode)
        rmse,mape,r2 = metrics(annual(lp))
        rows.append({"Config":cfg,"target":mode,"MAPE":mape,"RMSE":rmse,"R2":r2})
        print(f"  {cfg:8s} | {mode:15s} | MAPE={mape:7.2f}  RMSE={rmse:8.2f}  R2={r2:6.2f}", flush=True)
res=pd.DataFrame(rows)
res.to_csv(os.path.join(OUT,"temporal_target_screen.csv"),index=False)
print("\nsaved temporal_target_screen.csv")
