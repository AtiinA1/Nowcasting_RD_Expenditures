"""
Evaluate the country-year group-split experiment:
  - econometric baselines: Random Walk, AR(1), Historical mean, OLS,
    U-MIDAS and MIDAS (Beta weights) on the SAME test country-years
  - skill scores (MAPE/RMSE/MASE/OOS-R2 vs RW)
  - Diebold-Mariano tests (HLN small-sample correction)
  - ensemble-based prediction intervals + empirical coverage
"""
import os, numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"
OUT = os.path.join(ROOT, "additional_analysis", "out")
CONFIGS = ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]

pred = pd.read_csv(os.path.join(OUT, "cy_split_predictions.csv"))
member_cols = [c for c in pred.columns if c.startswith("m") and c[1:].isdigit()]

# annual GERD panel (levels) from the feature matrix
feat = pd.read_csv(os.path.join(OUT, "merged_features.csv"))
panel = feat.groupby(["Country", "Year"]).rd_expenditure.mean().reset_index().rename(columns={"rd_expenditure": "GERD"})

# test country-years (from NN experiment); everything else is the fitting sample
test_cy = set(zip(pred.Country, pred.Year))
panel["is_test"] = [ (c, y) in test_cy for c, y in zip(panel.Country, panel.Year) ]
fit = panel[~panel.is_test]

# ---------- monthly composite Google-Trends index (z-scored within country) ----------
gt = pd.read_csv(os.path.join(ROOT, "data", "GT", "trends_data_by_topic_filtered.csv"))
gt["date"] = pd.to_datetime(gt["date"]); gt["Year"] = gt.date.dt.year; gt["Month"] = gt.date.dt.month
long = gt.melt(id_vars=["date", "Year", "Month"], var_name="ck", value_name="v")
long[["Country", "topic"]] = long.ck.str.split("_", n=1, expand=True)
comp = long.groupby(["Country", "Year", "Month"]).v.mean().reset_index()      # mean over 57 topics
comp["z"] = comp.groupby("Country").v.transform(lambda s: (s - s.mean()) / s.std())
wide = comp.pivot_table(index=["Country", "Year"], columns="Month", values="z").reset_index()
wide.columns = ["Country", "Year"] + [f"gt{m}" for m in range(1, 13)]

# build regression frame: annual GERD + 12 monthly GT + AR(1)
panel = panel.sort_values(["Country", "Year"])
panel["GERD_lag1"] = panel.groupby("Country").GERD.shift(1)
reg = panel.merge(wide, on=["Country", "Year"], how="left").dropna(subset=["GERD_lag1"] + [f"gt{m}" for m in range(1, 13)])
reg["is_test"] = [ (c, y) in test_cy for c, y in zip(reg.Country, reg.Year) ]
cdum = pd.get_dummies(reg.Country, prefix="c")
GTcols = [f"gt{m}" for m in range(1, 13)]

tr = ~reg.is_test.values; te = reg.is_test.values
def design(extra):
    return np.column_stack([np.ones(len(reg)), reg.GERD_lag1.values, cdum.values] + extra)

def ols_fit_pred(X):
    beta, *_ = np.linalg.lstsq(X[tr], reg.GERD.values[tr], rcond=None)
    return X @ beta

# MIDAS / U-MIDAS are Google-Trends-ONLY here (no lagged GERD level), using a
# within-country log-standardized target and ridge -- the IDENTICAL specification
# to the temporal split (additional_analysis/temporal_artifacts.py). This makes them
# the SAME model evaluated under the country-year split, and an apples-to-apples
# comparison with the GT-only neural network (AGT), which also receives no AR term.
cmean = reg[~reg.is_test].groupby("Country").GERD.apply(lambda s: np.log(s).mean()).to_dict()
cstd  = reg[~reg.is_test].groupby("Country").GERD.apply(lambda s: np.log(s).std()).to_dict()
reg["ystd"] = [(np.log(G) - cmean[c]) / cstd[c] for c, G in zip(reg.Country, reg.GERD)]
ystd = reg.ystd.values
def lvlz(z):
    return np.array([np.exp(zz * cstd[c] + cmean[c]) for zz, c in zip(z, reg.Country)])

# U-MIDAS: 12 unrestricted monthly coefficients (+ country intercepts), GT only
Xu = np.column_stack([np.ones(len(reg)), cdum.values, reg[GTcols].values])
bu = np.linalg.solve(Xu[tr].T @ Xu[tr] + 1e-3 * np.eye(Xu.shape[1]), Xu[tr].T @ ystd[tr])
reg["UMIDAS"] = lvlz(Xu @ bu)

# MIDAS with normalized Beta weights w_j(t1,t2): collapse 12 months -> 1 index, NLS
GTmat = reg[GTcols].values
def beta_w(p):
    t1, t2 = np.exp(p)  # keep positive
    x = np.linspace(1e-3, 1, 12)
    w = x**(t1 - 1) * (1 - x)**(t2 - 1)
    return w / w.sum()
def midas_resid(p):
    idx = GTmat @ beta_w(p)
    X = np.column_stack([np.ones(len(reg)), cdum.values, idx])
    b = np.linalg.solve(X[tr].T @ X[tr] + 1e-3 * np.eye(X.shape[1]), X[tr].T @ ystd[tr])
    return ystd[tr] - X[tr] @ b
res = optimize.least_squares(midas_resid, x0=np.log([2.0, 2.0]), method="lm", max_nfev=2000)
idx = GTmat @ beta_w(res.x)
Xm = np.column_stack([np.ones(len(reg)), cdum.values, idx])
bm = np.linalg.solve(Xm[tr].T @ Xm[tr] + 1e-3 * np.eye(Xm.shape[1]), Xm[tr].T @ ystd[tr])
reg["MIDAS"] = lvlz(Xm @ bm)

# OLS on AR + macro-free GT yearly index (simple linear benchmark; retains lagged level)
reg["OLS"] = ols_fit_pred(design([reg[GTcols].mean(1).values.reshape(-1, 1)]))

# simple benchmarks
def prev_obs(c, y):
    g = panel[(panel.Country == c) & (panel.Year < y)].sort_values("Year")
    return g.GERD.iloc[-1] if len(g) else np.nan
def hist_mean(c, y):
    g = fit[(fit.Country == c) & (fit.Year < y)]
    return g.GERD.mean() if len(g) else np.nan
def ar1(c, y):
    s = fit[fit.Country == c].sort_values("Year").GERD.values
    if len(s) < 4: return prev_obs(c, y)
    ly = np.log(s); b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ly)-1), ly[:-1]]), ly[1:], rcond=None)
    p = prev_obs(c, y)
    return np.exp(b[0] + b[1]*np.log(p)) if p and p > 0 else np.nan

reg_test = reg[te].copy()
reg_test["RW"] = [prev_obs(c, y) for c, y in zip(reg_test.Country, reg_test.Year)]
reg_test["AR1"] = [ar1(c, y) for c, y in zip(reg_test.Country, reg_test.Year)]
reg_test["HistMean"] = [hist_mean(c, y) for c, y in zip(reg_test.Country, reg_test.Year)]
bench = reg_test[["Country", "Year", "RW", "AR1", "HistMean", "UMIDAS", "MIDAS", "OLS", "GERD"]]

# ---------- annual NN predictions (collapse months) ----------
nn_ann = pred.groupby(["Config", "Country", "Year"]).agg(
    True_Values=("True_Values", "mean"), pred=("pred_mean", "mean")).reset_index()
nn_wide = nn_ann.pivot_table(index=["Country", "Year"], columns="Config", values="pred").reset_index()
ann = bench.merge(nn_wide, on=["Country", "Year"], how="inner")
print(f"Annual test points: {len(ann)} country-years")

# ---------- skill scores (annual) ----------
def sk(true, p):
    true = np.asarray(true, float); p = np.asarray(p, float)
    m = np.isfinite(true) & np.isfinite(p); true, p = true[m], p[m]
    e = true - p
    return np.sqrt(np.mean(e**2)), np.mean(np.abs(e)), np.mean(np.abs(e/true))*100
scale = np.mean(np.abs(np.concatenate([np.diff(g.sort_values("Year").GERD.values)
                                       for _, g in panel.groupby("Country") if len(g) > 1])))
mse_rw = sk(ann.GERD, ann.RW)[0]**2
rows = []
for name in CONFIGS + ["UMIDAS", "MIDAS", "OLS", "RW", "AR1", "HistMean"]:
    rmse, mae, mape = sk(ann.GERD, ann[name])
    rows.append({"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape,
                 "MASE": mae/scale, "OOS_R2_vs_RW": 1 - rmse**2/mse_rw})
skill = pd.DataFrame(rows).set_index("Model")
print("\n=== Country-year split: annual skill scores (n=%d) ===" % len(ann))
print(skill.round(3).to_string())
skill.round(3).to_csv(os.path.join(OUT, "cy_skill_scores.csv"))

# ---------- Diebold-Mariano ----------
def dm(true, p1, p2, power=2):
    true=np.asarray(true,float); p1=np.asarray(p1,float); p2=np.asarray(p2,float)
    m=np.isfinite(true)&np.isfinite(p1)&np.isfinite(p2); true,p1,p2=true[m],p1[m],p2[m]
    e1,e2=true-p1,true-p2; d=(e1**2-e2**2) if power==2 else (np.abs(e1)-np.abs(e2))
    n=len(d)
    if n<5 or np.var(d)==0: return np.nan,np.nan,n
    stat=d.mean()/np.sqrt(np.var(d,ddof=1)/n); corr=np.sqrt((n+1)/n)
    s=stat*corr; return s,2*(1-stats.t.cdf(abs(s),df=n-1)),n
print("\n=== DM vs Random Walk (neg => model better than RW) ===")
dmrows=[]
for name in CONFIGS+["UMIDAS","MIDAS","AR1"]:
    s,p,n=dm(ann.GERD, ann[name], ann.RW); dmrows.append({"Model":name,"DM_vs_RW":s,"p":p,"n":n})
dmrw=pd.DataFrame(dmrows).set_index("Model"); print(dmrw.round(3).to_string())
dmrw.round(3).to_csv(os.path.join(OUT,"cy_dm_vs_rw.csv"))
print("\n=== DM: best GT config (AGT) vs benchmarks (neg => AGT better) ===")
for b in ["UMIDAS","MIDAS","OLS","Macros","LagRD","AGTwRD"]:
    s,p,n=dm(ann.GERD, ann.AGT, ann[b]); print(f"  AGT vs {b:8s}: DM={s:6.2f}  p={p:.3f}")

# ---------- ensemble prediction intervals + coverage ----------
print("\n=== Ensemble-based 95% prediction intervals: empirical coverage ===")
cov_rows=[]
for cfg in CONFIGS:
    d=pred[pred.Config==cfg]
    M=d[member_cols].values
    lo=np.quantile(M,0.025,axis=1); hi=np.quantile(M,0.975,axis=1)
    mu=M.mean(1); sd=M.std(1)
    cov_q=np.mean((d.True_Values.values>=lo)&(d.True_Values.values<=hi))*100
    cov_g=np.mean((d.True_Values.values>=mu-1.96*sd)&(d.True_Values.values<=mu+1.96*sd))*100
    width=np.mean((hi-lo)/np.abs(mu))*100
    cov_rows.append({"Config":cfg,"cov_quantile_%":cov_q,"cov_gaussian_%":cov_g,"avg_rel_width_%":width})
cov=pd.DataFrame(cov_rows).set_index("Config"); print(cov.round(1).to_string())
cov.round(2).to_csv(os.path.join(OUT,"cy_ensemble_coverage.csv"))

# ---------- figures ----------
fig,ax=plt.subplots(1,2,figsize=(12,4))
order=CONFIGS+["UMIDAS","MIDAS","OLS","RW","AR1"]
colors=["#2c7fb8"]*7+["#d95f0e"]*5
ax[0].bar(order, skill.loc[order,"MAPE"], color=colors); ax[0].set_ylabel("MAPE (%)")
ax[0].set_title("Country-year split: out-of-sample MAPE"); ax[0].tick_params(axis="x",rotation=45)
ax[1].bar(CONFIGS, skill.loc[CONFIGS,"OOS_R2_vs_RW"], color="#2c7fb8"); ax[1].axhline(0,color="k",lw=.8)
ax[1].set_title("Skill vs random walk (OOS $R^2$)"); ax[1].tick_params(axis="x",rotation=45)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"cy_skill_scores.png"),dpi=150)

# fan chart for US (AGT) over test years
d=pred[(pred.Config=="AGT")&(pred.Country=="US")].groupby("Year")
if len(d):
    yrs=[]; mu=[]; lo=[]; hi=[]; tv=[]
    for y,g in d:
        M=g[member_cols].values
        yrs.append(y); mu.append(M.mean()); lo.append(np.quantile(M,.025)); hi.append(np.quantile(M,.975)); tv.append(g.True_Values.mean())
    o=np.argsort(yrs); yrs=np.array(yrs)[o]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.fill_between(yrs, np.array(lo)[o], np.array(hi)[o], alpha=.3, label="95% ensemble band")
    ax.plot(yrs, np.array(mu)[o], "o-", label="AGT ensemble mean")
    ax.plot(yrs, np.array(tv)[o], "ks", label="True GERD")
    ax.set_title("US: AGT nowcast with ensemble interval (country-year split)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"cy_fanchart_US.png"),dpi=150)
print("\nSaved figures -> cy_skill_scores.png, cy_fanchart_US.png")
