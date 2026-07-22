"""Generate temporal-split Step-B figures (NN-elasticity series; employment
overlay) + CCF, from combined_estimates_temporal.csv. Figures written to the
Oxford submission Disaggregation folder."""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/atin/Nowcasting/Nowcasting_github"; OUT=os.path.join(ROOT,"additional_analysis","out")
FIG="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Disaggregation"
df=pd.read_csv(os.path.join(OUT,"combined_estimates_temporal.csv"))
df["date"]=pd.to_datetime(dict(year=df.Year,month=df.Month,day=1)); df=df.sort_values("date").reset_index(drop=True)

# 8a: NN-elasticity series (temporal)
plt.figure(figsize=(7,4))
plt.plot(df.date,df.NN_temporal,color="blue",marker="o",ms=3,lw=1)
plt.ylabel("Monthly R&D (USD bn)"); plt.title("NN-driven elasticity-based estimate (temporal split)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"NNelasticity_temporal.png"),dpi=200); plt.close()

# 8a-combined: three methods overlaid (temporal)
plt.figure(figsize=(7,4))
plt.plot(df.date,df.NN_temporal,label="NN-elasticity (temporal)",color="blue",marker="o",ms=3,lw=1)
plt.plot(df.date,df.Mosley,label="Sparse (Mosley)",color="green",marker="^",ms=3,lw=1)
plt.plot(df.date,df.Sax,label="Chow-Lin (Sax)",color="purple",marker="x",ms=3,lw=1)
plt.ylabel("Monthly R&D (USD bn)"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG,"all_methods_temporal.png"),dpi=200); plt.close()

# employment overlay (temporal NN)
emp=pd.read_csv(os.path.join(ROOT,"data","datausa.io","Monthly Employment.csv"))
emp["date"]=pd.to_datetime(emp["Date"]); emp=emp[["date","NSA Employees"]].rename(columns={"NSA Employees":"emp"})
m=df.merge(emp,on="date",how="inner").sort_values("date").reset_index(drop=True)
fig,ax1=plt.subplots(figsize=(7,4))
ax1.plot(m.date,m.NN_temporal,color="blue",label="NN-elasticity R&D (temporal)"); ax1.set_ylabel("Monthly R&D (USD bn)",color="blue")
ax2=ax1.twinx(); ax2.plot(m.date,m.emp,color="red",alpha=.7,label="R&D-services employment"); ax2.set_ylabel("Employees",color="red")
plt.title("US monthly R&D (temporal NN) vs R&D-services employment"); fig.tight_layout()
plt.savefig(os.path.join(FIG,"employee_temporal.png"),dpi=200); plt.close()

# CCF (growth) NN_temporal vs employment, with bands
me=np.diff(m.emp.values)/m.emp.values[:-1]
def ccf(series):
    sg=np.diff(series)/series[:-1]; out={}
    for L in range(-12,13):
        if L<0: a,b=sg[-L:],me[:L]
        elif L>0: a,b=sg[:-L],me[L:]
        else: a,b=sg,me
        if len(a)>3: out[L]=np.corrcoef(a,b)[0,1]
    return out
N=len(m)-1; band=1.96/np.sqrt(N)
fig,ax=plt.subplots(figsize=(8,4))
for nm,col,c in [("NN-elasticity (temporal)","NN_temporal","#2c7fb8"),("Sparse (Mosley)","Mosley","#31a354"),("Chow-Lin (Sax)","Sax","#d95f0e")]:
    cc=ccf(m[col].values); ax.plot(list(cc.keys()),list(cc.values()),marker="o",ms=3,label=nm,color=c)
ax.axhspan(-band,band,color="grey",alpha=.2); ax.axhline(0,color="k",lw=.6)
ax.set_xlabel("Lag (months); negative = R&D leads employment"); ax.set_ylabel("Cross-correlation (growth)")
ax.set_title("CCF: monthly R&D estimates vs R&D-services employment (temporal)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"temporal_ccf.png"),dpi=150); plt.close()
# summary
print(f"CCF band +-{band:.3f}  N={N}")
for nm,col in [("NN_temporal","NN_temporal"),("Mosley","Mosley"),("Sax","Sax")]:
    cc=ccf(m[col].values); peak=max(cc,key=lambda k:abs(cc[k])); nsig=sum(abs(v)>band for v in cc.values())
    print(f"  {nm:10s}: peak |r|={abs(cc[peak]):.2f} at lag {peak}, lag0={cc[0]:.2f}, #sig={nsig}/25")
print("saved figures: NNelasticity_temporal.png, all_methods_temporal.png, employee_temporal.png; out/temporal_ccf.png")
