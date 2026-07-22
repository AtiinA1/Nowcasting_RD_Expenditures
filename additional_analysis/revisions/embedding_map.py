"""
Learned country-embedding similarity map. The MLP learns a 2-D embedding per
country (country_embedding.weight, shape 8x2). Embeddings are identified only up
to rotation/reflection/scale and differ across ensemble members, so we work with
the rotation-invariant pairwise-distance structure: for each of the 10 members we
compute the 8x8 Euclidean distance matrix, scale it to unit mean, average across
members, and visualize with metric MDS. We then check whether the learned
similarities line up with anything economically sensible (R&D scale / region).

Inputs (read-only): nn_mlp_nowcasting_model/results/{cfg}/best_model_{0..9}.pt
                    additional_analysis/out/merged_features.csv  (for mean GERD)
Output: revisions/figs/embedding_map.png
"""
import os, glob
import numpy as np, pandas as pd
import torch
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.normpath(os.path.join(HERE, "..", "out"))

# LabelEncoder on the 8 country codes -> alphabetical
countries = ["CA", "CH", "CN", "DE", "GB", "JP", "KR", "US"]
CFG = "AGT"

dmats = []
embs_for_proc = []
for f in sorted(glob.glob(os.path.join(ROOT, "nn_mlp_nowcasting_model", "results", CFG, "best_model_*.pt"))):
    sd = torch.load(f, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    E = sd["country_embedding.weight"].numpy()       # (8, 2)
    d = pdist(E)                                       # condensed pairwise distances
    d = d / d.mean()                                  # unit-mean scale (remove per-member scale)
    dmats.append(squareform(d))
    embs_for_proc.append(E)
print(f"loaded {len(dmats)} ensemble members for config {CFG}")
D = np.mean(dmats, axis=0)                             # averaged 8x8 distance matrix

# stability: how consistent are member distance matrices?
flat = np.array([squareform(m) for m in dmats])
corrs = np.corrcoef(flat)
iu = np.triu_indices_from(corrs, 1)
print(f"mean pairwise correlation of member distance matrices: {corrs[iu].mean():.2f} "
      f"(min {corrs[iu].min():.2f}) -- high => embedding geometry is stable across seeds")

coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0).fit_transform(D)

# nearest neighbour of each country in the averaged distance space
print("\nNearest neighbour by learned distance:")
for i, c in enumerate(countries):
    order = np.argsort(D[i]); j = order[1]
    print(f"  {c} -> {countries[j]}")

# context: mean GERD level per country (bn)
mf = pd.read_csv(os.path.join(SRC, "merged_features.csv"))
meang = (mf.groupby("Country").rd_expenditure.mean())

fig, ax = plt.subplots(figsize=(6.2, 5.0))
sizes = (meang.reindex(countries).values)
ax.scatter(coords[:, 0], coords[:, 1], s=np.sqrt(sizes) * 12,
           color="#85B7EB", edgecolor="#185FA5", zorder=3)
for i, c in enumerate(countries):
    ax.annotate(f"{c}", (coords[i, 0], coords[i, 1]), fontsize=11,
                ha="center", va="center", zorder=4)
ax.set_title(f"Learned country-embedding similarity ({CFG}, MDS of averaged distances)")
ax.set_xlabel("MDS dim 1"); ax.set_ylabel("MDS dim 2")
ax.text(0.99, 0.01, "marker size ~ mean GERD level", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8, color="#5F5E5A")
ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "embedding_map.png"), dpi=200)
print("\nsaved figs/embedding_map.png")
