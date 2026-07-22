"""Training/validation loss diagnostics for the temporal-split NN.

This script retrains the key AGT and AllVar configurations using the same
chronological split and target transform as the paper pipeline. It writes all
outputs inside this robustness_overfit folder.
"""

from __future__ import annotations

import os
import sys

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
from common import FIG, OUT, annual_metrics, prepare_data  # noqa: E402


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)


class MLP(nn.Module):
    def __init__(self, d: int, hidden: list[int], n_countries: int, emb: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(n_countries, emb)
        dims = [d + emb] + hidden
        self.lin = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden)))
        self.bn = nn.ModuleList(nn.BatchNorm1d(h) for h in hidden)
        self.out = nn.Linear(dims[-1], 1)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, self.emb(c)], dim=1)
        for lin, bn in zip(self.lin, self.bn):
            z = self.drop(self.relu(bn(lin(z))))
        return self.out(z)


def tensor(a: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(a)


def run_config(name: str, cols: list[str], ensemble_size: int = 5, epochs: int = 600, patience: int = 70) -> tuple[pd.DataFrame, pd.DataFrame]:
    prep = prepare_data()
    df = prep.frame
    masks = prep.masks
    le = LabelEncoder()
    cc = le.fit_transform(df.Country)
    months = pd.get_dummies(df.Month.astype(int), prefix="M").astype(float).values

    feat = df[cols].fillna(0).astype(float).values
    scaler = StandardScaler().fit(feat[masks["train"]])
    X = np.hstack([scaler.transform(feat), months])

    Xtr, Xva, Xte = tensor(X[masks["train"]]), tensor(X[masks["val"]]), tensor(X[masks["test"]])
    ctr = torch.LongTensor(cc[masks["train"]])
    cva = torch.LongTensor(cc[masks["val"]])
    cte = torch.LongTensor(cc[masks["test"]])
    ytr = tensor(prep.ystd[masks["train"]].reshape(-1, 1))
    yva = tensor(prep.ystd[masks["val"]].reshape(-1, 1))

    histories = []
    zmembers = []
    for member in range(ensemble_size):
        torch.manual_seed(member)
        net = MLP(X.shape[1], [200, 20, 20], len(le.classes_))
        opt = optim.AdamW(net.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = MultiStepLR(opt, [300], 0.1)
        crit = nn.MSELoss()
        best = np.inf
        bad = 0
        best_state = None

        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr))
            for start in range(0, len(Xtr), 64):
                idx = perm[start:start + 64]
                opt.zero_grad()
                loss = crit(net(Xtr[idx], ctr[idx]), ytr[idx])
                loss.backward()
                opt.step()
                scheduler.step()

            net.eval()
            with torch.no_grad():
                train_loss = crit(net(Xtr, ctr), ytr).item()
                val_loss = crit(net(Xva, cva), yva).item()
            histories.append({
                "Config": name,
                "member": member,
                "epoch": ep + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            if val_loss < best - 1e-7:
                best = val_loss
                bad = 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            zmembers.append(net(Xte, cte).numpy().ravel())
        print(f"{name}: member {member + 1}/{ensemble_size} finished at epoch {histories[-1]['epoch']}", flush=True)

    Z = np.column_stack(zmembers)
    test = df[masks["test"]][["Country", "Year", "Month", "rd_expenditure"]].copy().reset_index(drop=True)
    test["pred"] = np.exp(Z.mean(axis=1) * prep.country_std_vec[masks["test"]] + prep.country_mean_vec[masks["test"]])
    summary = annual_metrics(test, "pred")
    summary.update({
        "Config": name,
        "ensemble_size": ensemble_size,
        "mean_stop_epoch": float(pd.DataFrame(histories).groupby("member").epoch.max().mean()),
        "median_stop_epoch": float(pd.DataFrame(histories).groupby("member").epoch.max().median()),
    })
    return pd.DataFrame(histories), pd.DataFrame([summary])


def make_plot(history: pd.DataFrame) -> None:
    agg = (
        history.groupby(["Config", "epoch"], as_index=False)
        .agg(train_loss=("train_loss", "mean"), val_loss=("val_loss", "mean"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, cfg in zip(axes, ["AGT", "AllVar"]):
        d = agg[agg.Config == cfg]
        ax.plot(d.epoch, d.train_loss, label="Training", color="#2c7fb8", lw=1.8)
        ax.plot(d.epoch, d.val_loss, label="Validation", color="#d95f0e", lw=1.8)
        ax.set_title(cfg)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE on standardized log target")
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "training_diagnostics_loss_curves.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    prep = prepare_data()
    configs = {"AGT": prep.configs["AGT"], "AllVar": prep.configs["AllVar"]}
    histories = []
    summaries = []
    for name, cols in configs.items():
        h, s = run_config(name, cols)
        histories.append(h)
        summaries.append(s)
    hist = pd.concat(histories, ignore_index=True)
    summ = pd.concat(summaries, ignore_index=True)
    hist.to_csv(os.path.join(OUT, "training_diagnostics_loss_history.csv"), index=False)
    summ.to_csv(os.path.join(OUT, "training_diagnostics_summary.csv"), index=False)
    make_plot(hist)
    print(summ.to_string(index=False))
    print("saved training diagnostics")


if __name__ == "__main__":
    main()

