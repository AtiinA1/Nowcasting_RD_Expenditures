"""Pure AGT neural-network search.

This script excludes residual learning and uses only the Step-B-compatible AGT
feature space. It tests whether the best pure NN architecture can be improved by
larger seed ensembles and nearby regularization settings.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "out", "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.dirname(__file__))
from common import FIG, OUT, annual_metrics, metrics, prepare_data  # noqa: E402


PURE_OUT = os.path.join(OUT, "agt_pure_nn_seed_search")
os.makedirs(PURE_OUT, exist_ok=True)


@dataclass(frozen=True)
class Variant:
    name: str
    hidden: tuple[int, ...]
    dropout: float
    weight_decay: float
    lr: float
    loss: str
    wide: bool
    seeds: int


VARIANTS = [
    Variant("wide64_huber_wd1e3_s15", (64, 16), 0.15, 1e-3, 3e-3, "huber", True, 15),
    Variant("wide64_mse_wd1e3_s15", (64, 16), 0.15, 1e-3, 3e-3, "mse", True, 15),
    Variant("medium_ln_mse_s15", (128, 32), 0.15, 1e-3, 3e-3, "mse", False, 15),
    Variant("wide48_huber_wd1e3_s15", (48, 12), 0.12, 1e-3, 3e-3, "huber", True, 15),
    Variant("wide64_huber_wd5e4_s15", (64, 16), 0.12, 5e-4, 3e-3, "huber", True, 15),
]


class Net(nn.Module):
    def __init__(self, d: int, hidden: tuple[int, ...], n_countries: int, dropout: float, wide: bool):
        super().__init__()
        self.emb = nn.Embedding(n_countries, 4)
        self.wide = wide
        dims = [d + 4] + list(hidden)
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden)))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for h in hidden)
        self.out = nn.Linear(dims[-1], 1)
        self.skip = nn.Linear(d, 1) if wide else None
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, c):
        z = torch.cat([x, self.emb(c)], dim=1)
        for layer, norm in zip(self.layers, self.norms):
            z = self.drop(self.act(norm(layer(z))))
        y = self.out(z)
        if self.skip is not None:
            y = y + self.skip(x)
        return y


def tensor(a):
    return torch.FloatTensor(a)


def level_from_z(prep, mask, z):
    return np.exp(z * prep.country_std_vec[mask] + prep.country_mean_vec[mask])


def annual_metric(prep, mask, pred, metric="MAPE"):
    tmp = prep.frame[mask][["Country", "Year", "Month", "rd_expenditure"]].copy()
    tmp["pred"] = pred
    return annual_metrics(tmp, "pred")[metric]


def run_variant(prep, variant: Variant, epochs=500, patience=65):
    df = prep.frame
    cols = prep.configs["AGT"]
    le = LabelEncoder()
    cc = le.fit_transform(df.Country)
    feat = df[cols].fillna(0).astype(float).values
    sc = StandardScaler().fit(feat[prep.masks["train"]])
    X = np.column_stack([sc.transform(feat), prep.months.values])

    Xtr, Xva, Xte = tensor(X[prep.masks["train"]]), tensor(X[prep.masks["val"]]), tensor(X[prep.masks["test"]])
    ctr = torch.LongTensor(cc[prep.masks["train"]])
    cva = torch.LongTensor(cc[prep.masks["val"]])
    cte = torch.LongTensor(cc[prep.masks["test"]])
    ytr = tensor(prep.ystd[prep.masks["train"]].reshape(-1, 1))
    crit = nn.SmoothL1Loss(beta=0.5) if variant.loss == "huber" else nn.MSELoss()

    val_members = []
    test_members = []
    histories = []
    member_rows = []
    for seed in range(variant.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = Net(X.shape[1], variant.hidden, len(le.classes_), variant.dropout, variant.wide)
        opt = optim.AdamW(net.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
        best = np.inf
        best_state = None
        bad = 0
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr))
            for start in range(0, len(Xtr), 64):
                idx = perm[start:start + 64]
                opt.zero_grad()
                loss = crit(net(Xtr[idx], ctr[idx]), ytr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                zva = net(Xva, cva).numpy().ravel()
            pred_val = level_from_z(prep, prep.masks["val"], zva)
            score = annual_metric(prep, prep.masks["val"], pred_val)
            histories.append({"Variant": variant.name, "seed": seed, "epoch": ep + 1, "val_MAPE": score})
            if score < best - 1e-5:
                best = score
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            zva = net(Xva, cva).numpy().ravel()
            zte = net(Xte, cte).numpy().ravel()
        val_pred = level_from_z(prep, prep.masks["val"], zva)
        test_pred = level_from_z(prep, prep.masks["test"], zte)
        val_members.append(val_pred)
        test_members.append(test_pred)
        one_test = annual_metrics(prep.frame[prep.masks["test"]].assign(pred=test_pred), "pred")
        member_rows.append({
            "Variant": variant.name,
            "seed": seed,
            "best_val_MAPE": best,
            "test_MAPE": one_test["MAPE"],
            "test_RMSE": one_test["RMSE"],
            "test_R2": one_test["R2"],
        })
        print(f"{variant.name}: seed {seed + 1}/{variant.seeds}, best val MAPE={best:.2f}", flush=True)

    rows = []
    for k in [5, 10, variant.seeds]:
        if k > variant.seeds:
            continue
        val_pred = np.column_stack(val_members[:k]).mean(axis=1)
        test_pred = np.column_stack(test_members[:k]).mean(axis=1)
        val_m = annual_metrics(prep.frame[prep.masks["val"]].assign(pred=val_pred), "pred")
        test_m = annual_metrics(prep.frame[prep.masks["test"]].assign(pred=test_pred), "pred")
        rows.append({
            "Variant": variant.name,
            "ensemble_size": k,
            "hidden": "-".join(map(str, variant.hidden)),
            "dropout": variant.dropout,
            "weight_decay": variant.weight_decay,
            "loss": variant.loss,
            "wide": variant.wide,
            "val_MAPE": val_m["MAPE"],
            "test_MAPE": test_m["MAPE"],
            "test_RMSE": test_m["RMSE"],
            "test_R2": test_m["R2"],
        })

    pred = prep.frame[prep.masks["test"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    pred["Variant"] = variant.name
    pred["pred"] = np.column_stack(test_members).mean(axis=1)
    return pd.DataFrame(rows), pd.DataFrame(histories), pd.DataFrame(member_rows), pred


def reference_metrics():
    rows = []
    ann = pd.read_csv("/Users/atin/Nowcasting/Nowcasting_github/additional_analysis/out/temporal_annual_all.csv")
    for col in ["NN_AGT", "RW3", "AR3", "MIDAS", "UMIDAS"]:
        if col in ann.columns:
            m = metrics(ann.GERD.values, ann[col].values)
            rows.append({"Model": col, "MAPE": m["MAPE"], "RMSE": m["RMSE"], "R2": m["R2"]})
    reg = pd.read_csv(os.path.join(OUT, "regularized_linear_benchmarks.csv"))
    for _, r in reg[reg.Config == "AGT"].iterrows():
        rows.append({"Model": r.Model, "MAPE": r.MAPE, "RMSE": r.RMSE, "R2": r.R2})
    return pd.DataFrame(rows).sort_values("MAPE")


def main():
    prep = prepare_data()
    all_rows, all_hist, all_members, all_preds = [], [], [], []
    for variant in VARIANTS:
        rows, hist, members, pred = run_variant(prep, variant)
        all_rows.append(rows)
        all_hist.append(hist)
        all_members.append(members)
        all_preds.append(pred)
    results = pd.concat(all_rows, ignore_index=True).sort_values("test_MAPE")
    history = pd.concat(all_hist, ignore_index=True)
    members = pd.concat(all_members, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    refs = reference_metrics()

    results.to_csv(os.path.join(PURE_OUT, "agt_pure_nn_seed_search_results.csv"), index=False)
    history.to_csv(os.path.join(PURE_OUT, "agt_pure_nn_seed_search_history.csv"), index=False)
    members.to_csv(os.path.join(PURE_OUT, "agt_pure_nn_seed_member_metrics.csv"), index=False)
    preds.to_csv(os.path.join(PURE_OUT, "agt_pure_nn_seed_search_predictions.csv"), index=False)
    refs.to_csv(os.path.join(PURE_OUT, "agt_pure_reference_metrics.csv"), index=False)

    top = results.head(10).copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(np.arange(len(top)), top.test_MAPE, color="#2c6fbb")
    ax.axhline(refs.loc[refs.Model == "NN_AGT", "MAPE"].iloc[0], color="#777", ls="--", label="Existing NN AGT")
    ax.axhline(refs.loc[refs.Model == "Ridge", "MAPE"].iloc[0], color="#20845c", ls="--", label="Ridge AGT")
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(top.Variant + "\nN=" + top.ensemble_size.astype(str), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Temporal test MAPE (%)")
    ax.set_title("Pure AGT NN seed/regularization search")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "agt_pure_nn_seed_search.png"), dpi=220)
    plt.close(fig)

    print("\nPure AGT NN results")
    print(results[["Variant", "ensemble_size", "val_MAPE", "test_MAPE", "test_RMSE", "test_R2"]].to_string(index=False))
    print("\nAGT references")
    print(refs.to_string(index=False))
    print(f"\nsaved pure AGT NN outputs to {PURE_OUT}")


if __name__ == "__main__":
    main()
