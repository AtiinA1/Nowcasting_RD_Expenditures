"""Updated pure-NN results for all configurations.

This aligns all Step A configurations with the best non-residual, pure-NN setup
identified for AGT. No residual target, trend baseline, or new feature
engineering is introduced. The same architecture/training recipe is applied to
each feature configuration so the NN comparison is internally consistent.
"""

from __future__ import annotations

import os
import sys

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
from common import CONFIG_ORDER, FIG, OUT, SOURCE_OUT, annual_metrics, metrics, prepare_data  # noqa: E402


ALIGN_OUT = os.path.join(OUT, "all_configs_updated_pure_nn")
os.makedirs(ALIGN_OUT, exist_ok=True)


HIDDEN = (64, 16)
DROPOUT = 0.12
WEIGHT_DECAY = 5e-4
LR = 3e-3
LOSS = "huber"
ENSEMBLE_SIZE = 15


class WideDeepNN(nn.Module):
    def __init__(self, d: int, n_countries: int):
        super().__init__()
        self.emb = nn.Embedding(n_countries, 4)
        dims = [d + 4] + list(HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(HIDDEN)))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for h in HIDDEN)
        self.out = nn.Linear(dims[-1], 1)
        self.skip = nn.Linear(d, 1)
        self.drop = nn.Dropout(DROPOUT)
        self.act = nn.SiLU()

    def forward(self, x, c):
        z = torch.cat([x, self.emb(c)], dim=1)
        for layer, norm in zip(self.layers, self.norms):
            z = self.drop(self.act(norm(layer(z))))
        return self.out(z) + self.skip(x)


def tensor(a):
    return torch.FloatTensor(a)


def level_from_z(prep, mask, z):
    return np.exp(z * prep.country_std_vec[mask] + prep.country_mean_vec[mask])


def annual_metric(prep, mask, pred):
    tmp = prep.frame[mask][["Country", "Year", "Month", "rd_expenditure"]].copy()
    tmp["pred"] = pred
    return annual_metrics(tmp, "pred")["MAPE"]


def run_config(prep, cfg: str, epochs=500, patience=65):
    df = prep.frame
    cols = prep.configs[cfg]
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
    crit = nn.SmoothL1Loss(beta=0.5) if LOSS == "huber" else nn.MSELoss()

    val_members = []
    test_members = []
    histories = []
    for seed in range(ENSEMBLE_SIZE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = WideDeepNN(X.shape[1], len(le.classes_))
        opt = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
            val_mape = annual_metric(prep, prep.masks["val"], pred_val)
            histories.append({"Config": cfg, "seed": seed, "epoch": ep + 1, "val_MAPE": val_mape})
            if val_mape < best - 1e-5:
                best = val_mape
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
        val_members.append(level_from_z(prep, prep.masks["val"], zva))
        test_members.append(level_from_z(prep, prep.masks["test"], zte))
        print(f"{cfg}: seed {seed + 1}/{ENSEMBLE_SIZE}, best val MAPE={best:.2f}", flush=True)

    rows = []
    pred_parts = []
    for k in [5, 10, 15]:
        val_pred = np.column_stack(val_members[:k]).mean(axis=1)
        test_pred = np.column_stack(test_members[:k]).mean(axis=1)
        val_m = annual_metrics(prep.frame[prep.masks["val"]].assign(pred=val_pred), "pred")
        test_m = annual_metrics(prep.frame[prep.masks["test"]].assign(pred=test_pred), "pred")
        rows.append({
            "Config": cfg,
            "ensemble_size": k,
            "val_MAPE": val_m["MAPE"],
            "test_MAPE": test_m["MAPE"],
            "test_RMSE": test_m["RMSE"],
            "test_R2": test_m["R2"],
        })
    pred = prep.frame[prep.masks["test"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    pred["Config"] = cfg
    pred["pred"] = np.column_stack(test_members).mean(axis=1)
    pred_parts.append(pred)
    member_parts = []
    base = prep.frame[prep.masks["test"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    for seed, member_pred in enumerate(test_members):
        mp = base.copy()
        mp["Config"] = cfg
        mp["seed"] = seed
        mp["pred"] = member_pred
        member_parts.append(mp)
    return (
        pd.DataFrame(rows),
        pd.DataFrame(histories),
        pd.concat(pred_parts, ignore_index=True),
        pd.concat(member_parts, ignore_index=True),
    )


def reference_metrics():
    rows = []
    ann = pd.read_csv(os.path.join(SOURCE_OUT, "temporal_annual_all.csv"))
    for cfg in CONFIG_ORDER:
        col = f"NN_{cfg}"
        if col in ann.columns:
            m = metrics(ann.GERD.values, ann[col].values)
            rows.append({"Config": cfg, "Model": "Original NN", "MAPE": m["MAPE"], "RMSE": m["RMSE"], "R2": m["R2"]})
    reg_path = os.path.join(OUT, "regularized_linear_benchmarks.csv")
    if os.path.exists(reg_path):
        reg = pd.read_csv(reg_path)
        for _, r in reg.iterrows():
            rows.append({"Config": r.Config, "Model": r.Model, "MAPE": r.MAPE, "RMSE": r.RMSE, "R2": r.R2})
    return pd.DataFrame(rows)


def main():
    prep = prepare_data()
    result_parts, history_parts, pred_parts, member_parts = [], [], [], []
    for cfg in CONFIG_ORDER:
        rows, hist, pred, member_pred = run_config(prep, cfg)
        result_parts.append(rows)
        history_parts.append(hist)
        pred_parts.append(pred)
        member_parts.append(member_pred)

    results = pd.concat(result_parts, ignore_index=True)
    history = pd.concat(history_parts, ignore_index=True)
    preds = pd.concat(pred_parts, ignore_index=True)
    member_preds = pd.concat(member_parts, ignore_index=True)
    refs = reference_metrics()

    results.to_csv(os.path.join(ALIGN_OUT, "all_configs_updated_pure_nn_results.csv"), index=False)
    history.to_csv(os.path.join(ALIGN_OUT, "all_configs_updated_pure_nn_history.csv"), index=False)
    preds.to_csv(os.path.join(ALIGN_OUT, "all_configs_updated_pure_nn_predictions.csv"), index=False)
    member_preds.to_csv(os.path.join(ALIGN_OUT, "all_configs_updated_pure_nn_member_predictions.csv"), index=False)
    refs.to_csv(os.path.join(ALIGN_OUT, "all_configs_reference_metrics.csv"), index=False)

    best = results.sort_values(["Config", "test_MAPE"]).groupby("Config", as_index=False).head(1)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(np.arange(len(best)), best.test_MAPE, color="#2c6fbb", label="Updated pure NN")
    orig = refs[refs.Model == "Original NN"].set_index("Config").loc[best.Config]
    ax.scatter(np.arange(len(best)), orig.MAPE.values, color="#333333", marker="x", s=50, label="Original NN")
    ax.set_xticks(np.arange(len(best)))
    ax.set_xticklabels(best.Config, rotation=30, ha="right")
    ax.set_ylabel("Temporal test MAPE (%)")
    ax.set_title("Aligned updated pure-NN results across configurations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "all_configs_updated_pure_nn.png"), dpi=220)
    plt.close(fig)

    print("\nBest updated pure NN per configuration")
    print(best[["Config", "ensemble_size", "val_MAPE", "test_MAPE", "test_RMSE", "test_R2"]].to_string(index=False))
    print("\nOriginal NN references")
    print(refs[refs.Model == "Original NN"].sort_values("Config").to_string(index=False))
    print(f"\nsaved aligned updated pure NN outputs to {ALIGN_OUT}")


if __name__ == "__main__":
    main()
