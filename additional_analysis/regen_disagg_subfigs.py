"""Regenerate the three Step-B sub-figures (NN / Mosley / Sax) with IDENTICAL
styling (gray Min-Max band, same grid, same axes) for a uniform panel.
Data: out/combined_estimates_temporal_level.csv (Sax/Mosley are split-independent)."""
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
FIG="/Users/atin/Nowcasting/Nowcasting_Oxford_submission/figures/Disaggregation"
df=pd.read_csv("out/combined_estimates_temporal_level.csv"); df["date"]=pd.to_datetime(df["date"])
mn=df[["NN","Sax","Mosley"]].min(axis=1); mx=df[["NN","Sax","Mosley"]].max(axis=1)

def panel(col,color,title,outfile,label):
    fig,ax=plt.subplots(figsize=(7,4))
    ax.fill_between(df.date,mn,mx,color="gray",alpha=0.3,label="Range (Min-Max)")
    ax.plot(df.date,df[col],color=color,marker="o",ms=3,lw=1,label=label)
    ax.set_ylabel("Monthly R&D (USD bn)"); ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True,alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/{outfile}"); plt.close(); print("saved",outfile)

panel("NN","#2c6fbb","NN-driven elasticity-based estimate (temporal split, level target)",
      "NNelasticity_temporal_level.png","NN-elasticity")
panel("Mosley","green","Sparse temporal disaggregation (Mosley)",
      "all_time_series_plot_on_Tempdisagg_Mosley.png","Sparse (Mosley)")
panel("Sax","purple","Chow-Lin temporal disaggregation (Sax)",
      "all_time_series_plot_on_Tempdisagg_Sax.png","Chow-Lin (Sax)")
