"""
Robustness: country-year GROUP split (no country-year appears in both train and
test) -> removes the within-year target leakage of the randomized split while
keeping the cross-sectional (pooled, non-temporal) evaluation philosophy the
authors prefer for the main results.

Trains all 7 input-space configurations with the paper's ensemble MLP, saves
per-ensemble-member test predictions (for uncertainty bands), and writes a long
CSV consumed by `evaluate_country_year.py`.
"""
import os, sys, importlib.util
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out")
os.makedirs(OUT, exist_ok=True)
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ---- load preprocessed feature matrix (built by _probe_cols.py) ----
merged = pd.read_csv(os.path.join(OUT, "merged_features.csv"))
merged = merged[merged.Year >= 2004].copy()
merged.sort_values(["Country", "Year", "Month"], inplace=True)

# ---- feature-group definitions (paper Table 1) ----
AR   = [f"rd_expenditure_lag{l}" for l in (1, 2, 3)]
MAC  = [f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc = [c for c in merged.columns if "_yearly_avg_lag" in c]
YTD  = [c for c in merged.columns if c.endswith("_mean_YTD")]

CONFIGS = {
    "LagRD":  AR,
    "Macros": AR + MAC,
    "AGT":    AGTc,
    "MGT":    AGTc + YTD,
    "AGTwRD": AR + AGTc,
    "MGTwRD": AR + AGTc + YTD,
    "AllVar": AR + MAC + AGTc + YTD,
}

# ---- country-year GROUP split (per country: 64/16/20 over its country-years) ----
rng = np.random.default_rng(SEED)
merged["cy"] = merged.Country + "_" + merged.Year.astype(int).astype(str)
split_map = {}
for ctry, g in merged.groupby("Country"):
    yrs = np.array(sorted(g.Year.unique()))
    rng.shuffle(yrs)
    n = len(yrs); n_tr = int(round(n*0.64)); n_va = int(round(n*0.16))
    for i, y in enumerate(yrs):
        split_map[f"{ctry}_{int(y)}"] = ("train" if i < n_tr else
                                         "val" if i < n_tr + n_va else "test")
merged["split"] = merged.cy.map(split_map)
# sanity: no country-year in >1 split (guaranteed by construction)
print("rows per split:", merged.split.value_counts().to_dict())
print("test country-years:", merged[merged.split=="test"].cy.nunique(),
      "| overlap train&test:",
      len(set(merged[merged.split=="train"].cy) & set(merged[merged.split=="test"].cy)))

# ---- model (paper architecture) ----
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_countries, emb=4, p=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_countries, emb)
        dims = [in_dim + emb] + hidden
        self.lin = nn.ModuleList(nn.Linear(dims[i], dims[i+1]) for i in range(len(hidden)))
        self.bn  = nn.ModuleList(nn.BatchNorm1d(h) for h in hidden)
        self.out = nn.Linear(dims[-1], 1)
        self.drop = nn.Dropout(p); self.relu = nn.ReLU()
    def forward(self, x, c):
        x = torch.cat([x, self.emb(c)], 1)
        for lin, bn in zip(self.lin, self.bn):
            x = self.drop(self.relu(bn(lin(x))))
        return self.out(x)

def build_xy(cols):
    """One-hot months, encode country, standardize features fit on TRAIN only."""
    months = pd.get_dummies(merged.Month, prefix="M")
    feat = merged[cols].fillna(0.0).astype(float)
    le = LabelEncoder(); cc = le.fit_transform(merged.Country)
    sc = StandardScaler()
    masks = {s: (merged.split == s).values for s in ("train", "val", "test")}
    Xtr_c = sc.fit(feat[masks["train"]]).transform(feat)
    X = np.hstack([Xtr_c, months.values.astype(float)])
    return X, cc, le

def train_ensemble(cols, size=10, epochs=700, patience=80, lr=0.01, bs=64):
    X, cc, le = build_xy(cols)
    hidden = [200, 20, 20]
    masks = {s: (merged.split == s).values for s in ("train", "val", "test")}
    y = merged.rd_expenditure.values.astype(float).reshape(-1, 1)
    def t(a): return torch.FloatTensor(a)
    Xtr, Xva, Xte = t(X[masks["train"]]), t(X[masks["val"]]), t(X[masks["test"]])
    ctr = torch.LongTensor(cc[masks["train"]]); cva = torch.LongTensor(cc[masks["val"]]); cte = torch.LongTensor(cc[masks["test"]])
    ytr, yva = t(y[masks["train"]]), t(y[masks["val"]])
    member_preds = []
    for m in range(size):
        torch.manual_seed(m)
        net = MLP(X.shape[1], hidden, len(le.classes_))
        opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sch = MultiStepLR(opt, milestones=[300], gamma=0.1)
        crit = nn.MSELoss(); best = np.inf; bad = 0; best_state = None
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), bs):
                idx = perm[i:i+bs]
                opt.zero_grad()
                loss = crit(net(Xtr[idx], ctr[idx]), ytr[idx])
                loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad():
                vl = crit(net(Xva, cva), yva).item()
            if vl < best - 1e-6:
                best = vl; bad = 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            member_preds.append(net(Xte, cte).numpy().ravel())
        print(f"      member {m+1}/{size} done (stopped epoch {ep}, best val {best:.2f})", flush=True)
    return np.column_stack(member_preds), masks["test"]

rows = []
for name, cols in CONFIGS.items():
    print(f"\n=== training {name} ({len(cols)} base features) ===", flush=True)
    preds, test_mask = train_ensemble(cols)
    meta = merged.loc[test_mask, ["Country", "Year", "Month", "rd_expenditure"]].reset_index(drop=True)
    ens = preds.mean(1)
    mape = np.mean(np.abs((meta.rd_expenditure.values - ens) / meta.rd_expenditure.values)) * 100
    rmse = np.sqrt(np.mean((meta.rd_expenditure.values - ens) ** 2))
    print(f"   {name}: test MAPE={mape:.2f}%  RMSE={rmse:.2f}  (n={len(meta)})", flush=True)
    for i in range(len(meta)):
        base = dict(Config=name, Country=meta.Country[i], Year=int(meta.Year[i]),
                    Month=int(meta.Month[i]), True_Values=meta.rd_expenditure[i],
                    pred_mean=ens[i])
        for mm in range(preds.shape[1]):
            base[f"m{mm}"] = preds[i, mm]
        rows.append(base)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "cy_split_predictions.csv"), index=False)
print("\nSaved -> out/cy_split_predictions.csv  shape", out.shape)
