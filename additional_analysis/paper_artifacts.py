"""
Generate paper-ready artifacts for the COUNTRY-YEAR (leakage-free) split:
  Figure 4 replacement  : RMSE & MAPE box plots, NN ensemble vs OLS, by config
  Table (benchmarks)    : NN configs + RW/AR(1)/MIDAS/U-MIDAS skill + DM vs RW
  Table (coverage)      : ensemble 95% interval empirical coverage
  Figure (fan chart)    : US AGT nowcast with ensemble band

Figures are written directly into the Oxford submission figures folder.
Reads: additional_analysis/out/{cy_split_predictions.csv, merged_features.csv,
       cy_skill_scores.csv, cy_dm_vs_rw.csv, cy_ensemble_coverage.csv}
"""
import os, shutil, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out")
FIGDIR = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_CYsplit"
os.makedirs(FIGDIR, exist_ok=True)
CONFIGS = ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]
NN_COLOR = "#35658A"
OLS_COLOR = "#B7BC8B"
EDGE_COLOR = "#333333"

pred = pd.read_csv(os.path.join(OUT, "cy_split_predictions.csv"))
member_cols = [c for c in pred.columns if c.startswith("m") and c[1:].isdigit()]
feat = pd.read_csv(os.path.join(OUT, "merged_features.csv")); feat = feat[feat.Year >= 2004].copy()
test_cy = set(zip(pred.Country, pred.Year.astype(int)))

# ---- per-config OLS (same input space) for the figure ----
AR  = [f"rd_expenditure_lag{l}" for l in (1, 2, 3)]
MAC = [f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc = [c for c in feat.columns if "_yearly_avg_lag" in c]; YTD = [c for c in feat.columns if c.endswith("_mean_YTD")]
FEATS = {"LagRD":AR,"Macros":AR+MAC,"AGT":AGTc,"MGT":AGTc+YTD,"AGTwRD":AR+AGTc,"MGTwRD":AR+AGTc+YTD,"AllVar":AR+MAC+AGTc+YTD}
feat["is_test"] = [(c, int(y)) in test_cy for c, y in zip(feat.Country, feat.Year)]
months = pd.get_dummies(feat.Month, prefix="M").astype(float); cdum = pd.get_dummies(feat.Country, prefix="c").astype(float)
tr = ~feat.is_test.values; te = feat.is_test.values; y = feat.rd_expenditure.values.astype(float)
def ols_pred(cols):
    Xc = feat[cols].fillna(0).astype(float).values; mu, sd = Xc[tr].mean(0), Xc[tr].std(0); sd[sd==0]=1
    X = np.column_stack([np.ones(len(feat)), (Xc-mu)/sd, months.values, cdum.values])
    beta = np.linalg.solve(X[tr].T@X[tr] + 1e-2*np.eye(X.shape[1]), X[tr].T@y[tr]); return X@beta

# ---- assemble per-country distributions ----
def per_country_metric(cfg, kind):
    d = pred[pred.Config == cfg]
    ols = ols_pred(FEATS[cfg]); feat_te = feat[te].reset_index(drop=True); ols_te = ols[te]
    ols_map = {(c, int(yr), int(mo)): p for c, yr, mo, p in
               zip(feat_te.Country, feat_te.Year, feat_te.Month, ols_te)}
    nn_vals, ols_vals = [], []
    for ctry, g in d.groupby("Country"):
        tv = g.True_Values.values
        # NN: one metric per ensemble member (distribution)
        for mc in member_cols:
            pv = g[mc].values
            nn_vals.append(np.sqrt(np.mean((tv-pv)**2)) if kind=="RMSE" else np.mean(np.abs((tv-pv)/tv))*100)
        ov = np.array([ols_map[(c, int(yr), int(mo))] for c, yr, mo in zip(g.Country, g.Year, g.Month)])
        ols_vals.append(np.sqrt(np.mean((tv-ov)**2)) if kind=="RMSE" else np.mean(np.abs((tv-ov)/tv))*100)
    return nn_vals, ols_vals

for kind, ylab, fname in [("RMSE","RMSE (USD bn)","RMSE_BoxPlot_CYsplit"),
                          ("MAPE","MAPE (%)","MAPE_BoxPlot_CYsplit")]:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pos = np.arange(len(CONFIGS))
    nn_data = []; ols_data = []
    for cfg in CONFIGS:
        nv, ov = per_country_metric(cfg, kind); nn_data.append(nv); ols_data.append(ov)
    b1 = ax.boxplot(nn_data, positions=pos-0.2, widths=0.35, patch_artist=True,
                    boxprops=dict(facecolor=NN_COLOR, edgecolor=EDGE_COLOR, alpha=0.80),
                    medianprops=dict(color=EDGE_COLOR, linewidth=1.4), showfliers=False)
    b2 = ax.boxplot(ols_data, positions=pos+0.2, widths=0.35, patch_artist=True,
                    boxprops=dict(facecolor=OLS_COLOR, edgecolor=EDGE_COLOR, alpha=0.80),
                    medianprops=dict(color=EDGE_COLOR, linewidth=1.4), showfliers=False)
    for bp in (b1, b2):
        for element in ["whiskers", "caps"]:
            for artist in bp[element]:
                artist.set(color=EDGE_COLOR, linewidth=0.8)
    ax.set_xticks(pos); ax.set_xticklabels(CONFIGS, rotation=30); ax.set_ylabel(ylab)
    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in nn_data + ols_data if len(v)])
    if len(all_vals):
        upper = np.nanmax(all_vals) * 1.20
        if kind == "MAPE":
            upper = max(80, upper)
        else:
            upper = max(25, upper)
        ax.set_ylim(0, upper)
    ax.legend([b1["boxes"][0], b2["boxes"][0]], ["Neural network", "OLS (same inputs)"], loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, fname + ".png"), dpi=200)
    print("saved", fname)

# ---- fan chart (US, AGT) ----
d = pred[(pred.Config == "AGT") & (pred.Country == "US")]
if len(d):
    g = d.groupby("Year")
    yrs = np.array(sorted(d.Year.unique()))
    mu = np.array([d[d.Year==yy][member_cols].values.mean() for yy in yrs])
    lo = np.array([np.quantile(d[d.Year==yy][member_cols].values, .025) for yy in yrs])
    hi = np.array([np.quantile(d[d.Year==yy][member_cols].values, .975) for yy in yrs])
    tv = np.array([d[d.Year==yy].True_Values.mean() for yy in yrs])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(yrs, lo, hi, alpha=.3, color="#2c7fb8", label="95% ensemble interval")
    ax.plot(yrs, mu, "o-", color="#2c7fb8", label="AGT ensemble mean")
    ax.plot(yrs, tv, "ks", label="True GERD")
    ax.set_xlabel("Year"); ax.set_ylabel("GERD (USD bn)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "fanchart_US_AGT.png"), dpi=200)
    print("saved fanchart_US_AGT")

# ---- LaTeX: benchmark/skill table ----
skill = pd.read_csv(os.path.join(OUT, "cy_skill_scores.csv"), index_col=0)
dm = pd.read_csv(os.path.join(OUT, "cy_dm_vs_rw.csv"), index_col=0)
order = ["LagRD","Macros","AGT","MGT","AGTwRD","MGTwRD","AllVar","RW","AR1","HistMean","UMIDAS","MIDAS"]
tab = skill.reindex(order)[["MAPE","RMSE","MASE","OOS_R2_vs_RW"]].copy()
tab["DM_vs_RW"] = dm["DM_vs_RW"]; tab["DM_p"] = dm["p"]
def fmt(r):
    name = r.name
    dmv = "" if pd.isna(r.DM_vs_RW) else f"{r.DM_vs_RW:.2f}"
    dmp = "" if pd.isna(r.DM_p) else f"{r.DM_p:.2f}"
    return f"\\textit{{{name}}} & {r.MAPE:.2f} & {r.RMSE:.2f} & {r.MASE:.2f} & {r.OOS_R2_vs_RW:.2f} & {dmv} & {dmp} \\\\"
lines = [fmt(tab.loc[i]) for i in order]
latex = ("% Source: additional_analysis/evaluate_country_year.py (skill+DM), nn_vs_ols_benchmarks.py (OLS)\n"
         "\\begin{table}[!htb]\n\\centering\n"
         "\\caption{Out-of-sample accuracy under the country-year split. "
         "DM tests compare each model with the random walk.}\n"
         "\\label{tab:cy_benchmarks}\n\\begin{tabular}{l c c c c c c}\n\\toprule\n"
         "Model & MAPE (\\%) & RMSE & MASE & OOS $R^2$ & DM vs RW & $p$ \\\\\n\\midrule\n"
         + "\n".join(lines[:7]) + "\n\\midrule\n" + "\n".join(lines[7:]) +
         "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(OUT, "cy_benchmarks_table.tex"), "w").write(latex)
print("saved cy_benchmarks_table.tex")

# ---- LaTeX: coverage table ----
cov = pd.read_csv(os.path.join(OUT, "cy_ensemble_coverage.csv"), index_col=0)
clines = [f"\\textit{{{i}}} & {cov.loc[i,'cov_quantile_%']:.0f} & {cov.loc[i,'cov_gaussian_%']:.0f} & {cov.loc[i,'avg_rel_width_%']:.0f} \\\\"
          for i in CONFIGS]
clatex = ("% Source: additional_analysis/evaluate_country_year.py\n"
          "\\begin{table}[!htb]\n\\centering\n"
          "\\caption{Country-year split coverage of 95\\% ensemble intervals.}\n"
          "\\label{tab:cy_coverage}\n\\begin{tabular}{l c c c}\n\\toprule\n"
          "Configuration & Quantile cov. (\\%) & Gaussian cov. (\\%) & Avg.\\ rel.\\ width (\\%) \\\\\n\\midrule\n"
          + "\n".join(clines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(OUT, "cy_coverage_table.tex"), "w").write(clatex)
print("saved cy_coverage_table.tex")
print("\nFigures written to:", FIGDIR)
