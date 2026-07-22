"""AllVar original-MLP hyperparameter search under the temporal split.

This keeps the full feature set and the original MLP family used in the paper:
Google Trends, macro variables, lagged R&D, month indicators, and country
embeddings. The goal is to improve the NN by tuning training hyperparameters,
not by changing the feature engineering or adding residual/trend targets.
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
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(os.path.dirname(__file__))
from common import FIG, OUT, annual_metrics, metrics, prepare_data  # noqa: E402


SEARCH_OUT = os.path.join(OUT, "allvar_original_mlp_hparam_search")
os.makedirs(SEARCH_OUT, exist_ok=True)


@dataclass(frozen=True)
class Variant:
    name: str
    hidden: tuple[int, ...]
    dropout: float
    weight_decay: float
    lr: float
    batch_size: int
    loss: str
    seeds: int = 7


VARIANTS = [
    Variant("orig_lr01_wd1e4_do10_mse", (200, 20, 20), 0.10, 1e-4, 1e-2, 64, "mse"),
    Variant("orig_lr003_wd1e3_do15_mse", (200, 20, 20), 0.15, 1e-3, 3e-3, 64, "mse"),
    Variant("orig_lr003_wd3e3_do20_mse", (200, 20, 20), 0.20, 3e-3, 3e-3, 64, "mse"),
    Variant("orig_lr003_wd1e3_do15_huber", (200, 20, 20), 0.15, 1e-3, 3e-3, 64, "huber"),
    Variant("orig_lr002_wd3e3_do20_huber", (200, 20, 20), 0.20, 3e-3, 2e-3, 64, "huber"),
    Variant("orig_lr003_wd1e3_do25_huber", (200, 20, 20), 0.25, 1e-3, 3e-3, 64, "huber"),
    Variant("orig_wide_lr003_wd1e3_do15_mse", (256, 64, 16), 0.15, 1e-3, 3e-3, 64, "mse"),
    Variant("orig_compact_lr003_wd1e3_do15_mse", (128, 32, 16), 0.15, 1e-3, 3e-3, 64, "mse"),
]


class MLP(nn.Module):
    def __init__(self, d: int, hidden: tuple[int, ...], n_countries: int, emb: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(n_countries, emb)
        dims = [d + emb] + list(hidden)
        self.lin = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden)))
        self.bn = nn.ModuleList(nn.BatchNorm1d(h) for h in hidden)
        self.out = nn.Linear(dims[-1], 1)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, c):
        z = torch.cat([x, self.emb(c)], dim=1)
        for lin, bn in zip(self.lin, self.bn):
            z = self.drop(self.relu(bn(lin(z))))
        return self.out(z)


def tensor(a):
    return torch.FloatTensor(a)


def level_from_z(prep, mask, z):
    return np.exp(z * prep.country_std_vec[mask] + prep.country_mean_vec[mask])


def annual_metric(prep, mask, pred, metric="MAPE"):
    tmp = prep.frame[mask][["Country", "Year", "Month", "rd_expenditure"]].copy()
    tmp["pred"] = pred
    return annual_metrics(tmp, "pred")[metric]


def run_variant(prep, variant: Variant, epochs=550, patience=75):
    df = prep.frame
    cols = prep.configs["AllVar"]
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
        net = MLP(X.shape[1], variant.hidden, len(le.classes_), dropout=variant.dropout)
        opt = optim.AdamW(net.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
        scheduler = MultiStepLR(opt, [300], gamma=0.1)
        best = np.inf
        best_state = None
        bad = 0
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr))
            for start in range(0, len(Xtr), variant.batch_size):
                idx = perm[start:start + variant.batch_size]
                opt.zero_grad()
                loss = crit(net(Xtr[idx], ctr[idx]), ytr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                scheduler.step()
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
        one = annual_metrics(prep.frame[prep.masks["test"]].assign(pred=test_pred), "pred")
        member_rows.append({
            "Variant": variant.name,
            "seed": seed,
            "best_val_MAPE": best,
            "test_MAPE": one["MAPE"],
            "test_RMSE": one["RMSE"],
            "test_R2": one["R2"],
        })
        print(f"{variant.name}: seed {seed + 1}/{variant.seeds}, best val MAPE={best:.2f}", flush=True)

    rows = []
    for k in [5, variant.seeds]:
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
            "lr": variant.lr,
            "loss": variant.loss,
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
    for col in ["NN_AllVar", "NN_AGT", "RW3", "AR3"]:
        if col in ann.columns:
            m = metrics(ann.GERD.values, ann[col].values)
            rows.append({"Model": col, "MAPE": m["MAPE"], "RMSE": m["RMSE"], "R2": m["R2"]})
    reg = pd.read_csv(os.path.join(OUT, "regularized_linear_benchmarks.csv"))
    for _, r in reg[reg.Config == "AllVar"].iterrows():
        rows.append({"Model": f"AllVar {r.Model}", "MAPE": r.MAPE, "RMSE": r.RMSE, "R2": r.R2})
    return pd.DataFrame(rows).sort_values("MAPE")


def main():
    prep = prepare_data()
    result_parts, history_parts, member_parts, pred_parts = [], [], [], []
    for variant in VARIANTS:
        rows, hist, members, pred = run_variant(prep, variant)
        result_parts.append(rows)
        history_parts.append(hist)
        member_parts.append(members)
        pred_parts.append(pred)
    results = pd.concat(result_parts, ignore_index=True).sort_values("test_MAPE")
    history = pd.concat(history_parts, ignore_index=True)
    members = pd.concat(member_parts, ignore_index=True)
    preds = pd.concat(pred_parts, ignore_index=True)
    refs = reference_metrics()

    results.to_csv(os.path.join(SEARCH_OUT, "allvar_original_mlp_hparam_results.csv"), index=False)
    history.to_csv(os.path.join(SEARCH_OUT, "allvar_original_mlp_hparam_history.csv"), index=False)
    members.to_csv(os.path.join(SEARCH_OUT, "allvar_original_mlp_hparam_member_metrics.csv"), index=False)
    preds.to_csv(os.path.join(SEARCH_OUT, "allvar_original_mlp_hparam_predictions.csv"), index=False)
    refs.to_csv(os.path.join(SEARCH_OUT, "allvar_reference_metrics.csv"), index=False)

    top = results.head(10)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(np.arange(len(top)), top.test_MAPE, color="#2c6fbb")
    ax.axhline(refs.loc[refs.Model == "NN_AllVar", "MAPE"].iloc[0], color="#777", ls="--", label="Existing AllVar NN")
    ax.axhline(refs.loc[refs.Model == "AllVar Ridge", "MAPE"].iloc[0], color="#20845c", ls="--", label="AllVar Ridge")
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(top.Variant + "\nN=" + top.ensemble_size.astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Temporal test MAPE (%)")
    ax.set_title("AllVar original-MLP hyperparameter search")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "allvar_original_mlp_hparam_search.png"), dpi=220)
    plt.close(fig)

    print("\nAllVar original-MLP hyperparameter results")
    print(results[["Variant", "ensemble_size", "val_MAPE", "test_MAPE", "test_RMSE", "test_R2"]].to_string(index=False))
    print("\nReferences")
    print(refs.to_string(index=False))
    print(f"\nsaved AllVar hparam search outputs to {SEARCH_OUT}")


if __name__ == "__main__":
    main()
