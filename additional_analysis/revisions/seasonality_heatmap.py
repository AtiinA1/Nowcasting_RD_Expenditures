"""
Within-year search-intensity seasonality heatmap. Step B allocates the annual
R&D total across months using the within-year *timing* of search intensity. This
figure shows that timing signal directly: for each country and calendar month,
the mean share of annual search intensity (averaged over R&D-relevant topics and
over years). Flat rows => no within-year information (uniform allocation); humped
rows => the search data carry genuine seasonal structure for Step B to exploit.

Inputs (read-only): data/GT/trends_data_by_topic_{CC}.csv  (monthly, per topic)
Output: revisions/figs/seasonality_heatmap.png
Prints a concentration statistic per country.
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GTDIR = os.path.join(ROOT, "data", "GT")

countries = ["US", "GB", "DE", "CH", "CA", "JP", "KR", "CN"]
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def country_month_shares(cc):
    df = pd.read_csv(os.path.join(GTDIR, f"trends_data_by_topic_{cc}.csv"))
    df["date"] = pd.to_datetime(df["date"]); df["Year"] = df.date.dt.year; df["Month"] = df.date.dt.month
    avg_cols = [c for c in df.columns if c.endswith("_average")]
    if not avg_cols:                                   # fall back to sample1 columns
        avg_cols = [c for c in df.columns if c.endswith("_sample1")]
    # total search intensity proxy = sum across topics, per month
    df["total"] = df[avg_cols].sum(axis=1)
    # within-year share: month total / that year's annual total, then average over years
    df["yr_total"] = df.groupby("Year")["total"].transform("sum")
    df = df[df.yr_total > 0]
    df["share"] = df["total"] / df["yr_total"]
    sh = df.groupby("Month")["share"].mean().reindex(range(1, 13)).values
    return sh / np.nansum(sh)                          # renormalize to sum 1

H = np.vstack([country_month_shares(cc) for cc in countries])

print("Within-year concentration (max month share; 1/12=0.083 is uniform):")
for cc, row in zip(countries, H):
    peak = months[int(np.nanargmax(row))]
    print(f"  {cc}: max share {np.nanmax(row):.3f} in {peak}; std across months {np.nanstd(row):.3f}")

fig, ax = plt.subplots(figsize=(7.6, 4.2))
im = ax.imshow(H, aspect="auto", cmap="Blues", vmin=H.min(), vmax=H.max())
ax.set_xticks(range(12)); ax.set_xticklabels(months)
ax.set_yticks(range(len(countries))); ax.set_yticklabels(countries)
ax.set_title("Within-year share of R&D-topic search intensity")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("mean monthly share of annual total")
# Mark the uniform reference below the heatmap so it does not collide with the title.
ax.text(
    0.0,
    -0.20,
    "uniform allocation = 0.083 per month",
    transform=ax.transAxes,
    fontsize=8,
    color="#5F5E5A",
    ha="left",
    va="top",
)
fig.subplots_adjust(top=0.86, bottom=0.22, left=0.08, right=0.92)
plt.savefig(os.path.join(FIGDIR, "seasonality_heatmap.png"), dpi=200)
print("\nsaved figs/seasonality_heatmap.png")
