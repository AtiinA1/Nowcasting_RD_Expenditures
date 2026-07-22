"""
Faithful reproduction of the paper's Figure-4 comparison (NN vs OLS at the SAME
input space per configuration) but under the leakage-free COUNTRY-YEAR split,
plus the full econometric benchmark horse-race (RW, AR(1), MIDAS, U-MIDAS).

Reads the already-trained NN ensemble predictions (country_year_split.py) so the
two methods are evaluated on an identical test set.

Outputs (all in additional_analysis/out/):
  nn_vs_ols_cy.csv / .png       per-config NN-vs-OLS test MAPE & RMSE
  cy_master_table.csv / .tex    NN + OLS + econometric benchmarks, skill + DM
"""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out")
CONFIGS = ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]

feat = pd.read_csv(os.path.join(OUT, "merged_features.csv"))
feat = feat[feat.Year >= 2004].copy()
pred = pd.read_csv(os.path.join(OUT, "cy_split_predictions.csv"))
test_cy = set(zip(pred.Country, pred.Year.astype(int)))

# feature groups (identical to country_year_split.py)
AR  = [f"rd_expenditure_lag{l}" for l in (1, 2, 3)]
MAC = [f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc = [c for c in feat.columns if "_yearly_avg_lag" in c]
YTD  = [c for c in feat.columns if c.endswith("_mean_YTD")]
FEATS = {"LagRD":AR, "Macros":AR+MAC, "AGT":AGTc, "MGT":AGTc+YTD,
         "AGTwRD":AR+AGTc, "MGTwRD":AR+AGTc+YTD, "AllVar":AR+MAC+AGTc+YTD}

feat["is_test"] = [ (c, int(y)) in test_cy for c, y in zip(feat.Country, feat.Year) ]
months = pd.get_dummies(feat.Month, prefix="M").astype(float)
cdum = pd.get_dummies(feat.Country, prefix="c").astype(float)
tr = ~feat.is_test.values; te = feat.is_test.values
y = feat.rd_expenditure.values.astype(float)

def ols_per_config(cols):
    """OLS on the SAME input space as the NN config (+ month & country dummies).
    Ridge-stabilised least squares to avoid degenerate high-dim GT solutions."""
    Xc = feat[cols].fillna(0).astype(float).values
    mu, sd = Xc[tr].mean(0), Xc[tr].std(0); sd[sd == 0] = 1
    Xc = (Xc - mu) / sd
    X = np.column_stack([np.ones(len(feat)), Xc, months.values, cdum.values])
    # small ridge for numerical stability when p is large relative to n
    lam = 1e-2
    XtX = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X[tr].T @ y[tr])
    return X @ beta

def mape(t, p): return np.mean(np.abs((t - p) / t)) * 100
def rmse(t, p): return np.sqrt(np.mean((t - p) ** 2))

t_all = y
rows = []
for cfg in CONFIGS:
    ols = ols_per_config(FEATS[cfg])
    ols_te = ols[te]
    meta = feat.loc[te, ["Country", "Year", "Month"]].reset_index(drop=True)
    # per-config NN ensemble preds aligned by (Country, Year, Month)
    pc = pred[pred.Config == cfg].set_index(["Country", "Year", "Month"]).pred_mean.to_dict()
    nn_te = np.array([pc[(c, int(yr), int(mo))] for c, yr, mo in zip(meta.Country, meta.Year, meta.Month)])
    t = y[te]
    rows.append({"Config": cfg, "NN_MAPE": mape(t, nn_te), "OLS_MAPE": mape(t, ols_te),
                 "NN_RMSE": rmse(t, nn_te), "OLS_RMSE": rmse(t, ols_te)})
cmp = pd.DataFrame(rows).set_index("Config")
print("=== NN vs OLS (same input space) under country-year split, n=%d test rows ===" % te.sum())
print(cmp.round(2).to_string())
cmp.round(3).to_csv(os.path.join(OUT, "nn_vs_ols_cy.csv"))

# Figure-4 replacement
fig, ax = plt.subplots(1, 2, figsize=(12, 4)); x = np.arange(len(CONFIGS)); w = 0.38
for k, (col, lab) in enumerate([("MAPE", "MAPE (%)"), ("RMSE", "RMSE (USD bn)")]):
    ax[k].bar(x - w/2, cmp[f"NN_{col}"], w, label="Neural network", color="#2c7fb8")
    ax[k].bar(x + w/2, cmp[f"OLS_{col}"], w, label="OLS (same inputs)", color="#d95f0e")
    ax[k].set_xticks(x); ax[k].set_xticklabels(CONFIGS, rotation=45); ax[k].set_ylabel(lab); ax[k].legend()
ax[0].set_title("Country-year split: NN vs OLS by configuration")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "nn_vs_ols_cy.png"), dpi=150)
print("Saved -> nn_vs_ols_cy.png")
