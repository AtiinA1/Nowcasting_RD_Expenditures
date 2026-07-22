"""
Step B strengthening (options A + C): validate the within-year allocation against
an OBSERVED monthly series and quantify value-added over a naive uniform split.

There is no monthly R&D ground truth, so we test whether each disaggregation
method's within-year *profile* recovers the within-year profile of an observed,
economically related monthly series -- employment in scientific R&D services (US,
Data USA) -- better than a naive uniform (1/12) allocation. We additionally
disaggregate the annual employment total directly from Google Trends (a Denton-
style GT-composite allocation) as a same-target reference.

Metrics (over the overlapping months): within-year share-deviation correlation
(the discriminating, non-mechanical metric), month-on-month growth correlation,
and normalized RMSE of the recovered monthly level.
"""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")

est=pd.read_csv(os.path.join(OUT,"combined_estimates_temporal_level.csv"))   # NN, Sax, Mosley monthly R&D (US)
est=est.rename(columns={"NN":"NN_temporal"}) if "NN" in est.columns else est
nn_col="NN_temporal" if "NN_temporal" in est.columns else "NN"
emp=pd.read_csv(os.path.join(ROOT,"data","datausa.io","Monthly Employment.csv"))
emp["date"]=pd.to_datetime(emp["Date"]); emp["Year"]=emp.date.dt.year; emp["Month"]=emp.date.dt.month
emp=emp[["Year","Month","date","NSA Employees"]].rename(columns={"NSA Employees":"emp"})
gt=pd.read_csv(os.path.join(ROOT,"data","GT","trends_data_by_topic_filtered.csv"))
gt["date"]=pd.to_datetime(gt["date"]); gt["Year"]=gt.date.dt.year; gt["Month"]=gt.date.dt.month
ustopics=[c for c in gt.columns if c.startswith("US_")]
gt["gt_composite"]=gt[ustopics].mean(axis=1)

df=est.merge(emp,on=["Year","Month"],how="inner").merge(gt[["Year","Month","gt_composite"]],on=["Year","Month"],how="inner")
# keep only FULL years (12 months) so annual employment total is well defined
full=df.groupby("Year").Month.nunique(); full_years=full[full==12].index
df=df[df.Year.isin(full_years)].sort_values(["Year","Month"]).reset_index(drop=True)
print(f"Overlap: {df.Year.min()}-{df.Year.max()}, full years={len(full_years)}, N={len(df)}")

df["E_year"]=df.groupby("Year").emp.transform("sum")
df["emp_share"]=df.emp/df.E_year                                  # TRUE within-year employment shares

methods={"NN-elasticity":nn_col,"Sparse (Mosley)":"Mosley","Chow-Lin (Sax)":"Sax","Denton (GT composite)":"gt_composite"}
# within-year shares for each method
for name,col in methods.items():
    df[name+"_share"]=df[col]/df.groupby("Year")[col].transform("sum")
df["Uniform_share"]=1.0/12.0

def metrics(share_col):
    s=df[share_col].values; e=df.emp_share.values
    # recovered monthly level = E_year * method share
    rec=df.E_year.values*s; true=df.emp.values
    nrmse=np.sqrt(np.mean((rec-true)**2))/true.mean()
    # within-year share-deviation correlation (remove the 1/12 mechanical part)
    dev_s=s-1/12; dev_e=e-1/12
    sdev_corr=np.corrcoef(dev_s,dev_e)[0,1] if np.std(dev_s)>1e-9 else np.nan
    # month-on-month growth correlation of recovered vs true
    gr_rec=np.diff(rec)/rec[:-1]; gr_true=np.diff(true)/true[:-1]
    gcorr=np.corrcoef(gr_rec,gr_true)[0,1]
    return sdev_corr,gcorr,nrmse

rows=[]
for name in list(methods)+["Uniform"]:
    sc=name+"_share"
    sdev,gcorr,nrmse=metrics(sc)
    rows.append({"Method":name,"share_dev_corr":sdev,"growth_corr":gcorr,"nRMSE":nrmse})
res=pd.DataFrame(rows).set_index("Method")
print("\n=== Recovery of observed monthly R&D-services employment (US) ===")
print(res.round(3).to_string())
res.round(4).to_csv(os.path.join(OUT,"stepB_groundtruth.csv"))

# figure: true vs recovered (NN, Mosley, Uniform), shares within a representative window
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(df.date,df.emp_share,"k-",lw=1.6,label="True employment share")
for name,c in [("NN-elasticity","#2c7fb8"),("Sparse (Mosley)","#31a354"),("Uniform","#999999")]:
    ax.plot(df.date,df[name+"_share"],lw=1,alpha=0.8,label=name)
ax.axhline(1/12,color="grey",ls=":",lw=0.8)
ax.set_ylabel("within-year monthly share"); ax.set_title("Within-year allocation: methods vs observed employment (US)")
ax.legend(fontsize=8,ncol=2); plt.tight_layout()
plt.savefig(os.path.join(OUT,"stepB_groundtruth.png"),dpi=150)
print("saved out/stepB_groundtruth.csv, out/stepB_groundtruth.png")
