"""
sg-LASSO-MIDAS benchmark (Babii, Ghysels & Striaukas, 2022) on the TEMPORAL split.

Unlike the single-composite MIDAS/U-MIDAS in temporal_artifacts.py, this benchmark
uses the FULL set of Google-Trends topics (the same information as the NN), projects
each topic's 12 within-year monthly values onto a low-order Legendre basis (MIDAS
weighting), and selects/shrinks topics with a sparse-group LASSO (group = topic).
This is the fair high-dimensional *linear* mixed-frequency comparator for the network.

The Legendre degree, the LASSO penalty lambda, and the sparse/group mix alpha are ALL
selected on the validation fold (no hand-tuning). Target, split and within-country
log-standardization are identical to temporal_artifacts.py.

Outputs:
  - prints test-set MAPE / RMSE / R^2 (annual) and Diebold-Mariano vs RW(L=3) and NN-AGT
  - writes out/sg_lasso_midas_pred.csv  (per test country-year prediction, for the table)
"""
import os, numpy as np, pandas as pd
from numpy.polynomial import legendre as L
from scipy import stats

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"; OUT = os.path.join(ROOT, "additional_analysis", "out")

# ---------- panel, split, logstd target (mirror temporal_artifacts.py) ----------
feat = pd.read_csv(os.path.join(OUT, "merged_features.csv")); feat = feat[feat.Year >= 2004].copy()
panel = feat.groupby(["Country", "Year"]).rd_expenditure.mean().reset_index().rename(columns={"rd_expenditure": "GERD"})
split = {}
for ctry, g in panel.groupby("Country"):
    yrs = np.array(sorted(g.Year.unique())); n = len(yrs); n_tr = int(round(n*0.64)); n_va = int(round(n*0.16))
    for i, yy in enumerate(yrs):
        split[(ctry, int(yy))] = "train" if i < n_tr else ("val" if i < n_tr+n_va else "test")
cmean, cstd = {}, {}
for ctry, g in panel.groupby("Country"):
    v = np.log(g.GERD.values[[split[(ctry, int(y))] == "train" for y in g.Year]]); cmean[ctry] = v.mean(); cstd[ctry] = max(v.std(), 0.05)

# ---------- per-topic monthly Google-Trends, within-country z-scored ----------
gt = pd.read_csv(os.path.join(ROOT, "data", "GT", "trends_data_by_topic_filtered.csv"))
gt["date"] = pd.to_datetime(gt["date"]); gt["Year"] = gt.date.dt.year; gt["Month"] = gt.date.dt.month
long = gt.melt(id_vars=["date", "Year", "Month"], var_name="ck", value_name="v")
long[["Country", "topic"]] = long.ck.str.split("_", n=1, expand=True)
topics = sorted(set.intersection(*[set(t) for t in long.groupby("Country").topic.unique()]))
long = long[long.topic.isin(topics)].copy()
long["z"] = long.groupby(["Country", "topic"]).v.transform(lambda s: (s - s.mean())/(s.std() if s.std() > 0 else 1))
monthly = {}
for (c, y, tp), g in long.groupby(["Country", "Year", "topic"]):
    if g.Month.nunique() == 12:
        monthly[(c, int(y), tp)] = g.sort_values("Month").z.values[:12]
print(f"common topics across all countries: {len(topics)}")

# country-years with all topics available (DEG-independent)
panel["ok"] = [all((c, int(y), tp) in monthly for tp in topics) for c, y in zip(panel.Country, panel.Year)]
P = panel[panel.ok].reset_index(drop=True)
P["sp"] = [split[(c, int(y))] for c, y in zip(P.Country, P.Year)]
ystd = np.array([(np.log(G) - cmean[c])/cstd[c] for c, G in zip(P.Country, P.GERD)])
cdum = pd.get_dummies(P.Country, prefix="c").astype(float).values
Xu = np.column_stack([np.ones(len(P)), cdum])            # unpenalized: intercept + country dummies
tr = (P.sp == "train").values; va = (P.sp == "val").values; te = (P.sp == "test").values

def midas_features(DEG):
    xm = (np.arange(1, 13) - 0.5)/12 * 2 - 1
    B = np.column_stack([L.legval(xm, [0]*l + [1]) for l in range(DEG+1)])
    B = B / np.sqrt((B**2).sum(0))
    X = np.zeros((len(P), len(topics)*(DEG+1)))
    for r, (c, y) in enumerate(zip(P.Country, P.Year)):
        for k, tp in enumerate(topics):
            X[r, k*(DEG+1):(k+1)*(DEG+1)] = B.T @ monthly[(c, int(y), tp)]
    groups = [np.arange(k*(DEG+1), (k+1)*(DEG+1)) for k in range(len(topics))]
    return X, groups

def sg_lasso(Xp, y, mask, groups, lam, alpha, pg, iters=3000):
    """proximal-gradient sparse-group LASSO; Xu unpenalized, Xp penalized (grouped)."""
    Xuf, Xpf, yf = Xu[mask], Xp[mask], y[mask]; N = mask.sum()
    Lip = np.linalg.norm(np.column_stack([Xuf, Xpf]), 2)**2 / N; t = 1.0/Lip
    bu = np.zeros(Xu.shape[1]); bp = np.zeros(Xp.shape[1])
    for _ in range(iters):
        bu = bu + t*(Xuf.T@(yf - Xuf@bu - Xpf@bp))/N
        z = bp + t*(Xpf.T@(yf - Xuf@bu - Xpf@bp))/N
        z = np.sign(z)*np.maximum(np.abs(z) - t*lam*alpha, 0.0)
        for g in groups:
            nz = np.linalg.norm(z[g]); bp[g] = z[g]*max(0.0, 1 - t*lam*(1-alpha)*pg/nz) if nz > 0 else 0.0
    return bu, bp

def to_level(carr, predz):
    return np.array([np.exp(p*cstd[c] + cmean[c]) for p, c in zip(predz, carr)])
def scores(carr, ylvl, predz):
    lvl = to_level(carr, predz); e = ylvl - lvl
    return np.sqrt(np.mean(e**2)), np.mean(np.abs(e/ylvl))*100, 1 - np.sum(e**2)/np.sum((ylvl-ylvl.mean())**2), lvl

# ---------- joint selection of (DEG, lambda, alpha) on the validation fold ----------
best = None
for DEG in (2, 3, 4):
    Xt, groups = midas_features(DEG); pg = np.sqrt(DEG+1)
    mu = Xt[tr].mean(0); sd = Xt[tr].std(0); sd[sd == 0] = 1; Xp = (Xt - mu)/sd
    for alpha in (0.05, 0.2, 0.5):
        for lam in np.geomspace(0.001, 1.0, 16):
            bu, bp = sg_lasso(Xp, ystd, tr, groups, lam, alpha, pg, iters=1200)
            _, mapev, _, _ = scores(P.Country.values[va], P.GERD.values[va], Xu[va]@bu + Xp[va]@bp)
            if best is None or mapev < best[0]:
                best = (mapev, DEG, lam, alpha)
_, DEG, lam, alpha = best
print(f"selected on validation: DEG={DEG}, lambda={lam:.4g}, alpha={alpha} (val MAPE={best[0]:.2f}%)")

# ---------- refit on train+val, evaluate on test ----------
Xt, groups = midas_features(DEG); pg = np.sqrt(DEG+1)
mu = Xt[tr].mean(0); sd = Xt[tr].std(0); sd[sd == 0] = 1; Xp = (Xt - mu)/sd
trva = tr | va
bu, bp = sg_lasso(Xp, ystd, trva, groups, lam, alpha, pg, iters=6000)
rmse, mape, r2, lvl = scores(P.Country.values[te], P.GERD.values[te], Xu[te]@bu + Xp[te]@bp)
nsel = sum(np.linalg.norm(bp[g]) > 1e-8 for g in groups)
print("\n=== sg-LASSO-MIDAS, TEMPORAL split, annual test ===")
print(f"  n_test={te.sum()}  topics_selected={nsel}/{len(topics)}")
print(f"  MAPE={mape:.2f}%   RMSE={rmse:.2f}   R2={r2:.3f}")
print("  reference: NN-AGT 15.62% | MIDAS 31.20% | U-MIDAS 31.92% | RW(L=3) 12.04%")

# ---------- save predictions + Diebold-Mariano vs RW(L=3) and NN-AGT ----------
out = pd.DataFrame({"Country": P.Country.values[te], "Year": P.Year.values[te].astype(int), "SGL": lvl})
out.to_csv(os.path.join(OUT, "sg_lasso_midas_pred.csv"), index=False)
print("saved sg_lasso_midas_pred.csv")
ref = pd.read_csv(os.path.join(OUT, "temporal_annual_all.csv"))
m = out.assign(GERD=P.GERD.values[te]).merge(ref[["Country", "Year", "NN_AGT", "RW3"]], on=["Country", "Year"])
def dm(t, p1, p2):
    t, p1, p2 = map(lambda a: np.asarray(a, float), (t, p1, p2)); d = (t-p1)**2 - (t-p2)**2; n = len(d)
    s = d.mean()/np.sqrt(np.var(d, ddof=1)/n)*np.sqrt((n+1)/n); return s, 2*(1-stats.t.cdf(abs(s), df=n-1))
s1, p1 = dm(m.GERD, m.SGL, m.RW3); s2, p2 = dm(m.GERD, m.NN_AGT, m.SGL)
print(f"  DM sg-LASSO-MIDAS vs RW(L=3): DM={s1:.2f}  p={p1:.3f}  (neg=>SGL better)")
print(f"  DM NN-AGT vs sg-LASSO-MIDAS : DM={s2:.2f}  p={p2:.3f}  (neg=>NN better)")
