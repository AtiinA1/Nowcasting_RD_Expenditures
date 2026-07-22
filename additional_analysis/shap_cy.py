"""
Recompute SHAP global-interpretability summaries under the leakage-free
country-year split, for the main configuration (AllVar, Fig. 6) and all seven
configurations (Appendix B). Retrains the same ensemble MLP as
country_year_split.py (identical split, seed, architecture), then applies
model-agnostic Kernel SHAP per country (background = k-means of the country's
training rows) and pools the attributions for the summary plot.

Output dir: Nowcasting_Oxford_submission/figures/Nowcast_Model_CYsplit/shap/
"""
import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out")
FIGDIR = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_CYsplit/shap"
os.makedirs(FIGDIR, exist_ok=True)
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

merged = pd.read_csv(os.path.join(OUT, "merged_features.csv"))
merged = merged[merged.Year >= 2004].copy()
merged.sort_values(["Country", "Year", "Month"], inplace=True)

AR  = [f"rd_expenditure_lag{l}" for l in (1, 2, 3)]
MAC = [f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc = [c for c in merged.columns if "_yearly_avg_lag" in c]; YTD = [c for c in merged.columns if c.endswith("_mean_YTD")]
CONFIGS = {"LagRD":AR,"Macros":AR+MAC,"AGT":AGTc,"MGT":AGTc+YTD,"AGTwRD":AR+AGTc,"MGTwRD":AR+AGTc+YTD,"AllVar":AR+MAC+AGTc+YTD}

# identical country-year split
rng = np.random.default_rng(SEED)
merged["cy"] = merged.Country + "_" + merged.Year.astype(int).astype(str)
split_map = {}
for ctry, g in merged.groupby("Country"):
    yrs = np.array(sorted(g.Year.unique())); rng.shuffle(yrs)
    n=len(yrs); n_tr=int(round(n*0.64)); n_va=int(round(n*0.16))
    for i, yv in enumerate(yrs):
        split_map[f"{ctry}_{int(yv)}"] = "train" if i<n_tr else ("val" if i<n_tr+n_va else "test")
merged["split"] = merged.cy.map(split_map)
months = pd.get_dummies(merged.Month, prefix="M").astype(float)
le = LabelEncoder(); merged["cc"] = le.fit_transform(merged.Country)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_c, emb=4, p=0.1):
        super().__init__(); self.emb=nn.Embedding(num_c, emb)
        dims=[in_dim+emb]+hidden
        self.lin=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(len(hidden)))
        self.bn=nn.ModuleList(nn.BatchNorm1d(h) for h in hidden)
        self.out=nn.Linear(dims[-1],1); self.drop=nn.Dropout(p); self.relu=nn.ReLU()
    def forward(self,x,c):
        x=torch.cat([x,self.emb(c)],1)
        for lin,bn in zip(self.lin,self.bn): x=self.drop(self.relu(bn(lin(x))))
        return self.out(x)

def run_config(name, cols, size=10, epochs=700, patience=80):
    feat = merged[cols].fillna(0).astype(float)
    masks = {s:(merged.split==s).values for s in ("train","val","test")}
    sc = StandardScaler(); Xc = sc.fit(feat[masks["train"]]).transform(feat)
    feat_names = list(cols) + list(months.columns)
    X = np.hstack([Xc, months.values])
    cc = merged.cc.values; y = merged.rd_expenditure.values.astype(float).reshape(-1,1)
    def t(a): return torch.FloatTensor(a)
    Xtr=t(X[masks["train"]]); Xva=t(X[masks["val"]])
    ctr=torch.LongTensor(cc[masks["train"]]); cva=torch.LongTensor(cc[masks["val"]])
    ytr=t(y[masks["train"]]); yva=t(y[masks["val"]])
    hidden=[200,20,20]; models=[]
    for m in range(size):
        torch.manual_seed(m); net=MLP(X.shape[1],hidden,len(le.classes_))
        opt=optim.AdamW(net.parameters(),lr=0.01,weight_decay=1e-4); sch=MultiStepLR(opt,[300],0.1)
        crit=nn.MSELoss(); best=np.inf; bad=0; bs=64; bstate=None
        for ep in range(epochs):
            net.train(); perm=torch.randperm(len(Xtr))
            for i in range(0,len(Xtr),bs):
                idx=perm[i:i+bs]; opt.zero_grad()
                loss=crit(net(Xtr[idx],ctr[idx]),ytr[idx]); loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad(): vl=crit(net(Xva,cva),yva).item()
            if vl<best-1e-6: best=vl; bad=0; bstate={k:v.clone() for k,v in net.state_dict().items()}
            else:
                bad+=1
                if bad>=patience: break
        if bstate is not None: net.load_state_dict(bstate)
        net.eval(); models.append(net)
    print(f"[{name}] trained {size} members", flush=True)

    # SHAP per country, pooled
    all_sv=[]; all_fx=[]
    for ctry in le.classes_:
        cidx=int(le.transform([ctry])[0])
        tr_rows=(merged.split=="train").values & (merged.Country==ctry).values
        te_rows=(merged.split=="test").values & (merged.Country==ctry).values
        if te_rows.sum()==0 or tr_rows.sum()<3: continue
        bg=shap.kmeans(X[tr_rows], min(5, tr_rows.sum()))
        expl_idx=np.where(te_rows)[0]
        if len(expl_idx)>5: expl_idx=expl_idx[:5]
        Z=X[expl_idx]
        def f(zz):
            tz=torch.FloatTensor(zz); ci=torch.LongTensor([cidx]*len(zz))
            with torch.no_grad(): preds=torch.stack([mm(tz,ci) for mm in models]).mean(0)
            return preds.numpy().ravel()
        ex=shap.KernelExplainer(f, bg)
        sv=ex.shap_values(Z, nsamples=200, silent=True)
        all_sv.append(np.array(sv)); all_fx.append(Z)
    SV=np.vstack(all_sv); FX=np.vstack(all_fx)
    plt.figure()
    shap.summary_plot(SV, FX, feature_names=feat_names, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, f"shap_summary_{name}.png"), dpi=160, bbox_inches="tight"); plt.close()
    print(f"[{name}] saved shap_summary_{name}.png  (pooled {FX.shape[0]} points)", flush=True)

import sys
todo = sys.argv[1:] if len(sys.argv) > 1 else ["AllVar","AGT","MGT","AGTwRD","MGTwRD","Macros","LagRD"]
for name in todo:
    run_config(name, CONFIGS[name])
print("DONE")
