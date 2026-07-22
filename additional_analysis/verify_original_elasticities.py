"""Reproduce the US NN-elasticity disaggregation using the ORIGINAL (level-target)
elasticities (results/AGT/elasticities_avg_ensemble_woyr.csv) through clean code,
to (a) verify the original Step B numbers, and (b) compare elasticity scenarios.
"""
import os, numpy as np, pandas as pd
from scipy import stats
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
el=pd.read_csv(os.path.join(ROOT,"nn_mlp_nowcasting_model","results","AGT","elasticities_avg_ensemble_woyr.csv"))
el=el[el.Feature.str.contains("_yearly_avg_lag")].copy()
el["topic"]=el.Feature.str.replace("_yearly_avg_lag[123]","",regex=True)
eta=el.groupby(["Country","topic"]).Elasticity.mean().reset_index()      # avg over 3 lags
COUNTRY="US"; etaU=eta[eta.Country==COUNTRY].set_index("topic").Elasticity.to_dict()

gt=pd.read_csv(os.path.join(ROOT,"data","GT","trends_data_by_topic_filtered.csv"))
gt["date"]=pd.to_datetime(gt["date"]); gt["Year"]=gt.date.dt.year; gt["Month"]=gt.date.dt.month
mf=pd.read_csv(os.path.join(OUT,"merged_features.csv")); mf=mf[mf.Year>=2004]
rd=mf[mf.Country==COUNTRY].groupby("Year").rd_expenditure.mean()
topics=[t for t in etaU if f"{COUNTRY}_{t}" in gt.columns]
g=gt[["Year","Month"]+[f"{COUNTRY}_{t}" for t in topics]].copy()
g=g[g.Year.isin([y for y in g.Year.unique() if y in rd.index])].copy()
adj=np.zeros(len(g))
for t in topics:
    gc=f"{COUNTRY}_{t}"; denom=g.groupby("Year")[gc].transform("sum").values
    p=np.where(denom>0,g[gc].values/np.where(denom>0,denom,1.0),0.0)
    if np.isfinite(etaU[t]): adj=adj+etaU[t]*p
g["adj"]=adj; g["adj_year"]=g.groupby("Year")["adj"].transform("sum"); g["rd_year"]=g.Year.map(rd)
g["NN_orig"]=g["rd_year"]*g["adj"]/g["adj_year"]
ce=pd.read_csv(os.path.join(ROOT,"temporal_disaggregation","results","combined_estimates.csv"))
ce=ce[ce.Country==COUNTRY][["Year","Month","Monthly_RD_Expenditure_Tempdisagg_Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley"]]
df=g[["Year","Month","NN_orig"]].merge(ce,on=["Year","Month"],how="inner").dropna().reset_index(drop=True)
df=df.rename(columns={"Monthly_RD_Expenditure_Tempdisagg_Sax":"Sax","Monthly_RD_Expenditure_Tempdisagg_Mosley":"Mosley"})
N=len(df); print(f"US N={N}")
def c(a,b): r,p=stats.pearsonr(a,b); return r,p
def gr(x): return np.diff(x)/x[:-1]
for nm in ["Mosley","Sax"]:
    rl,pl=c(df.NN_orig,df[nm]); rg,pg=c(gr(df.NN_orig.values),gr(df[nm].values))
    print(f"NN(original) vs {nm:7s}: level r({N-2})={rl:.2f} (p={pl:.1e}) | growth r={rg:.2f} (p={pg:.2f})")
