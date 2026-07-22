"""
Paper artifacts for the TEMPORAL split (main body): NN-vs-OLS box plots, US fan
chart, ensemble coverage, and the benchmark+Diebold-Mariano table including
publication-lag-consistent random-walk / AR benchmarks (RW/AR at lags L=1,2,3,
reflecting that GERD is released with a 2-3 year delay).

Reads out/temporal_split_predictions.csv (per-member preds from
temporal_predictions.py) + out/merged_features.csv.
Figures -> Nowcasting_Oxford_submission/figures/Nowcast_Model_Temporal/
Tables  -> Nowcasting_Oxford_submission/tables/
"""
import os, numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/Users/atin/Nowcasting/Nowcasting_github"; OUT = os.path.join(ROOT, "additional_analysis", "out")
FIGDIR = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Nowcast_Model_Temporal"
TABDIR = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/tables"
os.makedirs(FIGDIR, exist_ok=True); os.makedirs(TABDIR, exist_ok=True)
CONFIGS = ["LagRD","Macros","AGT","MGT","AGTwRD","MGTwRD","AllVar"]

pred = pd.read_csv(os.path.join(OUT, "temporal_split_predictions.csv"))
member_cols = [c for c in pred.columns if c.startswith("m") and c[1:].isdigit()]
feat = pd.read_csv(os.path.join(OUT, "merged_features.csv")); feat = feat[feat.Year >= 2004].copy()
feat.sort_values(["Country","Year","Month"], inplace=True)

# replicate temporal split + logstd target stats
split = {}
for ctry, g in feat.groupby("Country"):
    yrs = np.array(sorted(g.Year.unique())); n=len(yrs); n_tr=int(round(n*0.64)); n_va=int(round(n*0.16))
    for i, y in enumerate(yrs): split[(ctry,int(y))] = "train" if i<n_tr else ("val" if i<n_tr+n_va else "test")
feat["split"] = [split[(c,int(y))] for c,y in zip(feat.Country, feat.Year)]
masks = {s:(feat.split==s).values for s in ("train","val","test")}
fitmask = masks["train"]|masks["val"]
logy = np.log(feat.rd_expenditure.values.astype(float))
cmean,cstd={},{}
for ctry,g in feat.groupby("Country"):
    v=np.log(g.rd_expenditure.values[(g.split=="train").values]); cmean[ctry]=v.mean(); cstd[ctry]=max(v.std(),0.05)
cm=feat.Country.map(cmean).values; cs=feat.Country.map(cstd).values; ystd=(logy-cm)/cs
months=pd.get_dummies(feat.Month,prefix="M").astype(float); cdum=pd.get_dummies(feat.Country,prefix="c").astype(float)
y=feat.rd_expenditure.values.astype(float)

AR=[f"rd_expenditure_lag{l}" for l in (1,2,3)]
MAC=[f"{v}_lag{l}" for v in ["gdpca","unemp_rate","population","inflation","export_vol","import_vol"] for l in (1,2,3)]
AGTc=[c for c in feat.columns if "_yearly_avg_lag" in c]; YTD=[c for c in feat.columns if c.endswith("_mean_YTD")]
FEATS={"LagRD":AR,"Macros":AR+MAC,"AGT":AGTc,"MGT":AGTc+YTD,"AGTwRD":AR+AGTc,"MGTwRD":AR+AGTc+YTD,"AllVar":AR+MAC+AGTc+YTD}
def ols_level(cols):
    Xc=feat[cols].fillna(0).astype(float).values; mu,sd=Xc[masks["train"]].mean(0),Xc[masks["train"]].std(0); sd[sd==0]=1
    X=np.column_stack([np.ones(len(feat)),(Xc-mu)/sd,months.values,cdum.values])
    b=np.linalg.solve(X[fitmask].T@X[fitmask]+1e-2*np.eye(X.shape[1]),X[fitmask].T@ystd[fitmask])
    return np.exp((X@b)*cs+cm)

# ---------- grouped bar charts: pooled NN vs OLS metrics per config ----------
def pooled(cfg):
    d=pred[pred.Config==cfg].copy()
    tv=d.True_Values.values; nnp=d.pred_mean.values
    ols=ols_level(FEATS[cfg])[masks["test"]]; ft=feat[masks["test"]].reset_index(drop=True)
    omap={(c,int(yr),int(mo)):p for c,yr,mo,p in zip(ft.Country,ft.Year,ft.Month,ols)}
    ovv=np.array([omap[(c,int(yr),int(mo))] for c,yr,mo in zip(d.Country,d.Year,d.Month)])
    def mape(t,p): return np.mean(np.abs((t-p)/t))*100
    def rmse(t,p): return np.sqrt(np.mean((t-p)**2))
    return mape(tv,nnp),mape(tv,ovv),rmse(tv,nnp),rmse(tv,ovv)
M=np.array([pooled(c) for c in CONFIGS])  # cols: nn_mape, ols_mape, nn_rmse, ols_rmse
from matplotlib.patches import Patch
# Tableau-10 qualitative palette: one harmonious colour per configuration
CONFIG_COLORS={"LagRD":"#bab0ac","Macros":"#4e79a7","AGT":"#f28e2b","MGT":"#59a14f",
               "AGTwRD":"#e15759","MGTwRD":"#b07aa1","AllVar":"#76b7b2"}
ccols=[CONFIG_COLORS[c] for c in CONFIGS]
def _shade(hexs,f):  # lighten a hex colour toward white by fraction f (for the OLS bars)
    import matplotlib.colors as mc; r,g,b=mc.to_rgb(hexs); return (r+(1-r)*f,g+(1-g)*f,b+(1-b)*f)
olscols=[_shade(c,0.55) for c in ccols]
with plt.rc_context({"font.size":11,"axes.titlesize":12,"axes.labelsize":11.5,
                     "xtick.labelsize":10.5,"ytick.labelsize":10,"axes.linewidth":0.8}):
    fig,ax=plt.subplots(1,2,figsize=(12.5,4.8)); x=np.arange(len(CONFIGS)); w=0.40
    for k,(ni,oi,lab,fmt) in enumerate([(0,1,"MAPE (%)","%.1f"),(2,3,"RMSE (USD bn)","%.0f")]):
        a=ax[k]
        b1=a.bar(x-w/2,M[:,ni],w,color=ccols,edgecolor="white",linewidth=0.8,zorder=3)                 # NN: solid, full colour
        b2=a.bar(x+w/2,M[:,oi],w,color=olscols,edgecolor=ccols,linewidth=1.0,zorder=3)                 # OLS: same hue, lightened
        top=max(M[:,ni].max(),M[:,oi].max())
        for rects in (b1,b2):                                                                          # value labels above each bar
            for r in rects:
                a.annotate(fmt%r.get_height(),(r.get_x()+r.get_width()/2,r.get_height()),
                           xytext=(0,2),textcoords="offset points",ha="center",va="bottom",
                           fontsize=7.3,color="#333")
        a.set_xticks(x); a.set_xticklabels(CONFIGS,rotation=20,ha="right",color="black")
        a.set_ylabel(lab); a.set_axisbelow(True)
        a.grid(axis="y",color="#e3e3e3",lw=0.8,zorder=0); a.set_xlim(-0.6,len(CONFIGS)-0.4)
        a.set_ylim(0,top*1.16)
        a.spines[["top","right"]].set_visible(False); a.spines[["left","bottom"]].set_color("#999")
        a.tick_params(colors="#444",length=0)
    # estimator legend (solid = NN, lightened = OLS), neutral grey so it reads as a pattern key
    patt=[Patch(facecolor="#6b6f76",edgecolor="white",label="Neural network"),
          Patch(facecolor=_shade("#6b6f76",0.55),edgecolor="#6b6f76",label="OLS, same inputs")]
    ax[0].legend(handles=patt,frameon=False,loc="upper right",fontsize=10,handlelength=1.4)
    # configuration colour key, one row above the panels
    cfgleg=[Patch(facecolor=CONFIG_COLORS[c],edgecolor="white",label=c) for c in CONFIGS]
    fig.legend(handles=cfgleg,frameon=False,loc="upper center",ncol=len(CONFIGS),
               fontsize=10,bbox_to_anchor=(0.5,1.02),columnspacing=1.1,handlelength=1.1,handletextpad=0.5)
    fig.subplots_adjust(top=0.85,wspace=0.22)
    plt.savefig(os.path.join(FIGDIR,"NNvsOLS_Temporal.png"),dpi=220,bbox_inches="tight"); plt.close()
    print("saved NNvsOLS_Temporal")

# ---------- fan chart US AGT ----------
d=pred[(pred.Config=="AGT")&(pred.Country=="US")]
if len(d):
    yrs=np.array(sorted(d.Year.unique()))
    mu=np.array([d[d.Year==yy][member_cols].values.mean() for yy in yrs])
    lo=np.array([np.quantile(d[d.Year==yy][member_cols].values,.025) for yy in yrs])
    hi=np.array([np.quantile(d[d.Year==yy][member_cols].values,.975) for yy in yrs])
    tv=np.array([d[d.Year==yy].True_Values.mean() for yy in yrs])
    fig,ax=plt.subplots(figsize=(7,4)); ax.fill_between(yrs,lo,hi,alpha=.3,color="#2c7fb8",label="95% ensemble interval")
    ax.plot(yrs,mu,"o-",color="#2c7fb8",label="AGT ensemble mean"); ax.plot(yrs,tv,"ks",label="True GERD")
    ax.set_xlabel("Year"); ax.set_ylabel("GERD (USD bn)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR,"fanchart_US_AGT.png"),dpi=200); plt.close(); print("saved fanchart")

# ---------- benchmarks (annual) ----------
panel=feat.groupby(["Country","Year"]).rd_expenditure.mean().reset_index().rename(columns={"rd_expenditure":"GERD"})
panel["istest"]=[split[(c,int(y))]=="test" for c,y in zip(panel.Country,panel.Year)]
def lagval(c,y,L):
    g=panel[(panel.Country==c)&(panel.Year<=y-L)].sort_values("Year"); return g.GERD.iloc[-1] if len(g) else np.nan
def ar_lag(c,y,L):
    s=panel[(panel.Country==c)&(~panel.istest)&(panel.Year<y)].sort_values("Year").GERD.values
    if len(s)<4: return lagval(c,y,L)
    ly=np.log(s); b=np.linalg.lstsq(np.column_stack([np.ones(len(ly)-1),ly[:-1]]),ly[1:],rcond=None)[0]
    p=lagval(c,y,L); 
    if not (p and p>0): return np.nan
    lp=np.log(p)
    for _ in range(L): lp=b[0]+b[1]*lp     # iterate AR forward L steps from last available
    return np.exp(lp)
def hmean(c,y):
    g=panel[(panel.Country==c)&(~panel.istest)&(panel.Year<y)]; return g.GERD.mean() if len(g) else np.nan

# GT-only MIDAS / U-MIDAS (logstd target)
gt=pd.read_csv(os.path.join(ROOT,"data","GT","trends_data_by_topic_filtered.csv"))
gt["date"]=pd.to_datetime(gt["date"]); gt["Year"]=gt.date.dt.year; gt["Month"]=gt.date.dt.month
lo2=gt.melt(id_vars=["date","Year","Month"],var_name="ck",value_name="v"); lo2[["Country","topic"]]=lo2.ck.str.split("_",n=1,expand=True)
comp=lo2.groupby(["Country","Year","Month"]).v.mean().reset_index(); comp["z"]=comp.groupby("Country").v.transform(lambda s:(s-s.mean())/s.std())
wide=comp.pivot_table(index=["Country","Year"],columns="Month",values="z").reset_index(); wide.columns=["Country","Year"]+[f"gt{m}" for m in range(1,13)]
reg=panel.merge(wide,on=["Country","Year"],how="left").dropna(subset=[f"gt{m}" for m in range(1,13)]).reset_index(drop=True)
reg["ystd"]=[(np.log(G)-cmean[c])/cstd[c] for c,G in zip(reg.Country,reg.GERD)]; reg["istest"]=[split[(c,int(y))]=="test" for c,y in zip(reg.Country,reg.Year)]
cdr=pd.get_dummies(reg.Country,prefix="c").astype(float).values; GT=reg[[f"gt{m}" for m in range(1,13)]].values; trm=~reg.istest.values
def lvlz(z): return np.array([np.exp(zz*cstd[c]+cmean[c]) for zz,c in zip(z,reg.Country)])
Xu=np.column_stack([np.ones(len(reg)),cdr,GT]); bu=np.linalg.solve(Xu[trm].T@Xu[trm]+1e-3*np.eye(Xu.shape[1]),Xu[trm].T@reg.ystd.values[trm]); reg["UMIDAS"]=lvlz(Xu@bu)
def bw(p): t1,t2=np.exp(p); x=np.linspace(1e-3,1,12); w=x**(t1-1)*(1-x)**(t2-1); return w/w.sum()
def mres(p):
    idx=GT@bw(p); X=np.column_stack([np.ones(len(reg)),cdr,idx]); b=np.linalg.solve(X[trm].T@X[trm]+1e-3*np.eye(X.shape[1]),X[trm].T@reg.ystd.values[trm]); return reg.ystd.values[trm]-X[trm]@b
rr=optimize.least_squares(mres,np.log([2.,2.]),method="lm",max_nfev=2000); idx=GT@bw(rr.x)
Xm=np.column_stack([np.ones(len(reg)),cdr,idx]); bm=np.linalg.solve(Xm[trm].T@Xm[trm]+1e-3*np.eye(Xm.shape[1]),Xm[trm].T@reg.ystd.values[trm]); reg["MIDAS"]=lvlz(Xm@bm)
mmap={(c,int(y)):(u,m) for c,y,u,m in zip(reg.Country,reg.Year,reg.UMIDAS,reg.MIDAS)}

# annual NN
nn_ann=pred.groupby(["Config","Country","Year"]).agg(True_Values=("True_Values","mean"),p=("pred_mean","mean")).reset_index()
A=nn_ann[nn_ann.Config=="AGT"][["Country","Year","True_Values"]].rename(columns={"True_Values":"GERD"}).reset_index(drop=True)
for cfg in CONFIGS:
    s=nn_ann[nn_ann.Config==cfg].set_index(["Country","Year"]).p; A[f"NN_{cfg}"]=[s.get((c,y),np.nan) for c,y in zip(A.Country,A.Year)]
for L in (1,2,3): A[f"RW{L}"]=[lagval(c,int(y),L) for c,y in zip(A.Country,A.Year)]
for L in (1,2,3): A[f"AR{L}"]=[ar_lag(c,int(y),L) for c,y in zip(A.Country,A.Year)]
A["HistMean"]=[hmean(c,int(y)) for c,y in zip(A.Country,A.Year)]
A["UMIDAS"]=[mmap.get((c,int(y)),(np.nan,np.nan))[0] for c,y in zip(A.Country,A.Year)]
A["MIDAS"]=[mmap.get((c,int(y)),(np.nan,np.nan))[1] for c,y in zip(A.Country,A.Year)]
A.to_csv(os.path.join(OUT,"temporal_annual_all.csv"),index=False)

# sg-LASSO-MIDAS (full-topic-set high-dimensional linear benchmark), if available
sgl_path=os.path.join(OUT,"sg_lasso_midas_pred.csv")
if os.path.exists(sgl_path):
    _s=pd.read_csv(sgl_path); _m={(c,int(y)):v for c,y,v in zip(_s.Country,_s.Year,_s.SGL)}
    A["SGL"]=[_m.get((c,int(y)),np.nan) for c,y in zip(A.Country,A.Year)]

def sk(t,p):
    t=np.asarray(t,float);p=np.asarray(p,float);m=np.isfinite(t)&np.isfinite(p);t,p=t[m],p[m];e=t-p
    return np.sqrt(np.mean(e**2)),np.mean(np.abs(e/t))*100,1-np.sum(e**2)/np.sum((t-t.mean())**2)
def dmv(t,p1,p2):
    t=np.asarray(t,float);p1=np.asarray(p1,float);p2=np.asarray(p2,float);m=np.isfinite(t)&np.isfinite(p1)&np.isfinite(p2)
    t,p1,p2=t[m],p1[m],p2[m];d=(t-p1)**2-(t-p2)**2;n=len(d)
    if n<5 or np.var(d)==0: return np.nan,np.nan
    s=d.mean()/np.sqrt(np.var(d,ddof=1)/n)*np.sqrt((n+1)/n); return s,2*(1-stats.t.cdf(abs(s),df=n-1))

order=[f"NN_{c}" for c in CONFIGS]+["RW1","RW2","RW3","AR3","MIDAS","UMIDAS"]+(["SGL"] if "SGL" in A.columns else [])+["HistMean"]
labels={**{f"NN_{c}":c for c in CONFIGS},"RW1":"RW (L=1)","RW2":"RW (L=2)","RW3":"RW (L=3, feasible)","AR3":"AR(1) (L=3)","MIDAS":"MIDAS (GT)","UMIDAS":"U-MIDAS (GT)","SGL":"sg-LASSO-MIDAS (GT, full)","HistMean":"Hist. mean"}
rows=[]
for k in order:
    rmse,mape,r2=sk(A.GERD,A[k]); dm,p=dmv(A.GERD,A[k],A["RW3"])
    rows.append({"Model":labels[k],"MAPE":mape,"RMSE":rmse,"R2":r2,"DMvsRW3":dm,"p":p})
tab=pd.DataFrame(rows).set_index("Model"); print(tab.round(2).to_string()); tab.round(3).to_csv(os.path.join(OUT,"temporal_skill_full.csv"))

def fr(r):
    dm="" if pd.isna(r.DMvsRW3) else f"{r.DMvsRW3:.2f}"; pp="" if pd.isna(r.p) else f"{r.p:.2f}"
    return f"\\textit{{{r.name}}} & {r.MAPE:.2f} & {r.RMSE:.2f} & {r.R2:.2f} & {dm} & {pp} \\\\"
L=[fr(tab.loc[i]) for i in tab.index]
latex=("% Source: additional_analysis/temporal_artifacts.py\n\\begin{table}[!htb]\n\\centering\n"
 "\\caption{Out-of-sample accuracy under the temporal (chronological) split with within-country log-standardized targets "
 "(annual test points, $n=23$). RW(L)/AR(1)(L) use the most recent annual value available under a publication lag of $L$ years; "
 "since GERD is released with a 2--3 year delay, RW(L$=$3) is the feasible real-time naive benchmark. MIDAS and U-MIDAS use a single "
 "cross-topic Google-trends composite; sg-LASSO-MIDAS uses the full set of $47$ topics with a sparse-group LASSO (Legendre-MIDAS weights, "
 "degree and penalties selected on the validation fold). The Diebold--Mariano statistic compares each model to RW(L$=$3) (negative $\\Rightarrow$ model more accurate; HLN correction).}\n"
 "\\label{tab:temporal_benchmarks}\n\\begin{tabular}{l c c c c c}\n\\toprule\nModel & MAPE (\\%) & RMSE & $R^2$ & DM vs RW(3) & $p$ \\\\\n\\midrule\n"
 +"\n".join(L[:7])+"\n\\midrule\n"+"\n".join(L[7:])+"\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(TABDIR,"temporal_benchmarks_table.tex"),"w").write(latex); print("saved temporal_benchmarks_table.tex")

# NN vs OLS table
nvo=pd.DataFrame({"Config":CONFIGS,
  "NN_MAPE":[sk(A.GERD,A[f"NN_{c}"])[1] for c in CONFIGS],
  "OLS_MAPE":[sk(feat.rd_expenditure.values[masks["test"]], ols_level(FEATS[c])[masks["test"]])[1] for c in CONFIGS]}).set_index("Config")
nvo.round(2).to_csv(os.path.join(OUT,"temporal_nn_vs_ols_full.csv")); print(nvo.round(2).to_string())

# coverage
cov=[]
for cfg in CONFIGS:
    d=pred[pred.Config==cfg]; M=d[member_cols].values
    lo=np.quantile(M,.025,axis=1); hi=np.quantile(M,.975,axis=1); mu=M.mean(1); sd=M.std(1)
    cq=np.mean((d.True_Values.values>=lo)&(d.True_Values.values<=hi))*100
    cg=np.mean((d.True_Values.values>=mu-1.96*sd)&(d.True_Values.values<=mu+1.96*sd))*100
    cov.append({"Config":cfg,"q":cq,"g":cg,"w":np.mean((hi-lo)/np.abs(mu))*100})
covdf=pd.DataFrame(cov).set_index("Config")
clatex=("% Source: additional_analysis/temporal_artifacts.py\n\\begin{table}[!htb]\n\\centering\n"
 "\\caption{Empirical coverage of 95\\% ensemble prediction intervals on the temporal-split test set. Bands reflect across-member dispersion only.}\n"
 "\\label{tab:temporal_coverage}\n\\begin{tabular}{l c c c}\n\\toprule\nConfiguration & Quantile cov. (\\%) & Gaussian cov. (\\%) & Avg.\\ rel.\\ width (\\%) \\\\\n\\midrule\n"
 +"\n".join(f"\\textit{{{i}}} & {covdf.loc[i,'q']:.0f} & {covdf.loc[i,'g']:.0f} & {covdf.loc[i,'w']:.0f} \\\\" for i in CONFIGS)
 +"\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(TABDIR,"temporal_coverage_table.tex"),"w").write(clatex); print("saved temporal_coverage_table.tex")

# AllVar true-vs-pred scatter
d=pred[pred.Config=="AllVar"]; fig,ax=plt.subplots(figsize=(5.5,5.5))
for ctry,g in d.groupby("Country"): ax.scatter(g.True_Values,g.pred_mean,s=16,alpha=0.7,label=ctry)
lim=[d.True_Values.min()*0.8,d.True_Values.max()*1.2]; ax.plot(lim,lim,"k--",lw=1); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("True GERD (USD bn, log)"); ax.set_ylabel("Predicted GERD (USD bn, log)"); ax.legend(ncol=2,fontsize=7,title="Country")
plt.tight_layout(); plt.savefig(os.path.join(FIGDIR,"AllVar_TrueVsPred_Temporal.png"),dpi=200); plt.close(); print("saved AllVar scatter")
print("DONE")
