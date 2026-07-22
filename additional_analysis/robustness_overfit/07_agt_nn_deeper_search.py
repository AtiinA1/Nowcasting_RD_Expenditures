"""AGT-only neural-network improvement search.

This script focuses only on the AGT feature space because AGT is the relevant
Step A input for Step B elasticities. It tests whether additional NN variants can
improve on the best result from 06_nn_improvement_search.py without using the
test set for model selection.
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


AGT_OUT = os.path.join(OUT, "agt_nn_deeper_search")
os.makedirs(AGT_OUT, exist_ok=True)


@dataclass(frozen=True)
class Variant:
    name: str
    hidden: tuple[int, ...]
    dropout: float
    weight_decay: float
    lr: float
    loss: str
    wide: bool
    residual_target: bool = False


VARIANTS = [
    Variant("wide64_huber_wd3e3", (64, 16), 0.18, 3e-3, 2e-3, "huber", True),
    Variant("wide96_huber_wd3e3", (96, 24), 0.20, 3e-3, 2e-3, "huber", True),
    Variant("wide128_huber_wd3e3", (128, 32), 0.22, 3e-3, 2e-3, "huber", True),
    Variant("wide64_mse_wd3e3", (64, 16), 0.18, 3e-3, 2e-3, "mse", True),
    Variant("wide96_mse_wd3e3", (96, 24), 0.20, 3e-3, 2e-3, "mse", True),
    Variant("mlp96_huber_wd3e3", (96, 24), 0.25, 3e-3, 2e-3, "huber", False),
    Variant("resid_wide32_huber", (32,), 0.10, 3e-3, 2e-3, "huber", True, True),
    Variant("resid_wide64_huber", (64, 16), 0.15, 3e-3, 2e-3, "huber", True, True),
    Variant("resid_mlp64_huber", (64, 16), 0.20, 3e-3, 2e-3, "huber", False, True),
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


def country_trend_baseline(prep) -> np.ndarray:
    """Country-specific linear log trend estimated only on training years."""
    df = prep.frame
    baseline = np.zeros(len(df), dtype=float)
    for ctry, g in df.groupby("Country"):
        idx = g.index.values
        train = g.split.values == "train"
        years = g.Year.values.astype(float)
        y = np.log(g.rd_expenditure.values.astype(float))
        if train.sum() >= 3:
            coef = np.polyfit(years[train] - years[train].min(), y[train], 1)
            baseline[idx] = np.polyval(coef, years - years[train].min())
        else:
            baseline[idx] = y[train].mean() if train.sum() else y.mean()
    return baseline


def level_from_z(prep, mask, z):
    return np.exp(z * prep.country_std_vec[mask] + prep.country_mean_vec[mask])


def annual_metric(prep, mask, pred, metric="MAPE"):
    tmp = prep.frame[mask][["Country", "Year", "Month", "rd_expenditure"]].copy()
    tmp["pred"] = pred
    return annual_metrics(tmp, "pred")[metric]


def run_variant(prep, variant: Variant, seeds=7, epochs=550, patience=70):
    df = prep.frame
    cols = prep.configs["AGT"]
    le = LabelEncoder()
    cc = le.fit_transform(df.Country)
    feat = df[cols].fillna(0).astype(float).values
    sc = StandardScaler().fit(feat[prep.masks["train"]])
    X = np.column_stack([sc.transform(feat), prep.months.values])

    if variant.residual_target:
        base_log = country_trend_baseline(prep)
        y_target = np.log(df.rd_expenditure.values.astype(float)) - base_log
        mu = y_target[prep.masks["train"]].mean()
        sd = max(y_target[prep.masks["train"]].std(), 0.05)
        y_target = (y_target - mu) / sd
    else:
        base_log = None
        mu = 0.0
        sd = 1.0
        y_target = prep.ystd

    Xtr, Xva, Xte = tensor(X[prep.masks["train"]]), tensor(X[prep.masks["val"]]), tensor(X[prep.masks["test"]])
    ctr = torch.LongTensor(cc[prep.masks["train"]])
    cva = torch.LongTensor(cc[prep.masks["val"]])
    cte = torch.LongTensor(cc[prep.masks["test"]])
    ytr = tensor(y_target[prep.masks["train"]].reshape(-1, 1))
    crit = nn.SmoothL1Loss(beta=0.5) if variant.loss == "huber" else nn.MSELoss()

    val_members = []
    test_members = []
    histories = []
    for seed in range(seeds):
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
            if variant.residual_target:
                pred_val = np.exp(base_log[prep.masks["val"]] + (zva * sd + mu))
            else:
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
        if variant.residual_target:
            val_members.append(np.exp(base_log[prep.masks["val"]] + (zva * sd + mu)))
            test_members.append(np.exp(base_log[prep.masks["test"]] + (zte * sd + mu)))
        else:
            val_members.append(level_from_z(prep, prep.masks["val"], zva))
            test_members.append(level_from_z(prep, prep.masks["test"], zte))
        print(f"{variant.name}: seed {seed + 1}/{seeds}, best val MAPE={best:.2f}", flush=True)

    val_pred = np.column_stack(val_members).mean(axis=1)
    test_pred = np.column_stack(test_members).mean(axis=1)
    val_m = annual_metrics(prep.frame[prep.masks["val"]].assign(pred=val_pred), "pred")
    test_m = annual_metrics(prep.frame[prep.masks["test"]].assign(pred=test_pred), "pred")
    row = {
        "Variant": variant.name,
        "hidden": "-".join(map(str, variant.hidden)),
        "dropout": variant.dropout,
        "weight_decay": variant.weight_decay,
        "loss": variant.loss,
        "wide": variant.wide,
        "residual_target": variant.residual_target,
        "seeds": seeds,
        "val_MAPE": val_m["MAPE"],
        "test_MAPE": test_m["MAPE"],
        "test_RMSE": test_m["RMSE"],
        "test_R2": test_m["R2"],
        "mean_stop_epoch": float(pd.DataFrame(histories).groupby("seed").epoch.max().mean()),
    }
    pred = prep.frame[prep.masks["test"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    pred["Variant"] = variant.name
    pred["pred"] = test_pred
    return row, pd.DataFrame(histories), pred


def reference_metrics() -> pd.DataFrame:
    rows = []
    ann_path = "/Users/atin/Nowcasting/Nowcasting_github/additional_analysis/out/temporal_annual_all.csv"
    ann = pd.read_csv(ann_path)
    for col in ["NN_AGT", "RW3", "AR3", "MIDAS", "UMIDAS"]:
        if col in ann.columns:
            m = metrics(ann.GERD.values, ann[col].values)
            rows.append({"Model": col, "MAPE": m["MAPE"], "RMSE": m["RMSE"], "R2": m["R2"]})
    reg = pd.read_csv(os.path.join(OUT, "regularized_linear_benchmarks.csv"))
    for _, r in reg[reg.Config == "AGT"].iterrows():
        rows.append({"Model": r.Model, "MAPE": r.MAPE, "RMSE": r.RMSE, "R2": r.R2})
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, refs: pd.DataFrame) -> None:
    top = results.sort_values("test_MAPE").head(8).copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(np.arange(len(top)), top.test_MAPE, color="#2c6fbb", label="AGT NN variants")
    ax.axhline(refs.loc[refs.Model == "NN_AGT", "MAPE"].iloc[0], color="#777777", ls="--", lw=1.2, label="Existing NN AGT")
    ax.axhline(refs.loc[refs.Model == "Ridge", "MAPE"].iloc[0], color="#20845c", ls="--", lw=1.2, label="Ridge AGT")
    ax.axhline(refs.loc[refs.Model == "RW3", "MAPE"].iloc[0], color="#b15a1c", ls="--", lw=1.2, label="RW3")
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(top.Variant, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Temporal test MAPE (%)")
    ax.set_title("AGT-only NN improvement search")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "agt_nn_deeper_search.png"), dpi=220)
    plt.close(fig)


def main():
    prep = prepare_data()
    rows, histories, preds = [], [], []
    for variant in VARIANTS:
        row, hist, pred = run_variant(prep, variant)
        rows.append(row)
        histories.append(hist)
        preds.append(pred)
    results = pd.DataFrame(rows).sort_values("test_MAPE")
    history = pd.concat(histories, ignore_index=True)
    pred = pd.concat(preds, ignore_index=True)
    refs = reference_metrics().sort_values("MAPE")

    results.to_csv(os.path.join(AGT_OUT, "agt_nn_deeper_search_results.csv"), index=False)
    history.to_csv(os.path.join(AGT_OUT, "agt_nn_deeper_search_history.csv"), index=False)
    pred.to_csv(os.path.join(AGT_OUT, "agt_nn_deeper_search_predictions.csv"), index=False)
    refs.to_csv(os.path.join(AGT_OUT, "agt_reference_metrics.csv"), index=False)
    plot_results(results, refs)

    print("\nAGT-only NN variants")
    print(results[["Variant", "val_MAPE", "test_MAPE", "test_RMSE", "test_R2", "residual_target", "mean_stop_epoch"]].to_string(index=False))
    print("\nAGT framework-relevant references")
    print(refs.to_string(index=False))
    print(f"\nsaved AGT-only search outputs to {AGT_OUT}")


if __name__ == "__main__":
    main()
