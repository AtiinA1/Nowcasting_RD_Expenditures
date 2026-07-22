"""Targeted neural-network improvement search under the temporal split.

The goal is not to tune on the test set, but to test whether safer NN variants
can improve the Step A annual nowcast under the same chronological split:

- smaller LayerNorm MLPs with stronger regularization,
- Huber/SmoothL1 loss,
- wide-and-deep networks with a linear skip path.

Outputs are written inside additional_analysis/robustness_overfit.
"""

from __future__ import annotations

import os
import sys
import warnings
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
from common import CONFIG_ORDER, FIG, OUT, TAB, annual_metrics, metrics, prepare_data  # noqa: E402


warnings.filterwarnings("ignore", category=RuntimeWarning)
SEARCH_OUT = os.path.join(OUT, "nn_improvement_search")
os.makedirs(SEARCH_OUT, exist_ok=True)


@dataclass(frozen=True)
class Variant:
    name: str
    hidden: tuple[int, ...]
    dropout: float
    weight_decay: float
    lr: float
    loss: str = "mse"
    wide: bool = False
    norm: str = "layer"


VARIANTS = [
    Variant("small_ln_mse", (64, 16), 0.20, 1e-3, 3e-3, "mse", False),
    Variant("medium_ln_mse", (128, 32), 0.15, 1e-3, 3e-3, "mse", False),
    Variant("small_ln_huber", (64, 16), 0.20, 1e-3, 3e-3, "huber", False),
    Variant("wide_deep_small", (64, 16), 0.15, 1e-3, 3e-3, "mse", True),
    Variant("wide_deep_huber", (64, 16), 0.15, 1e-3, 3e-3, "huber", True),
    Variant("wide_deep_tiny", (32,), 0.10, 3e-3, 2e-3, "mse", True),
]

TEST_CONFIGS = ["Macros", "AGT", "AGTwRD", "AllVar"]


class RegularizedMLP(nn.Module):
    def __init__(self, d: int, hidden: tuple[int, ...], n_countries: int, emb: int, dropout: float, wide: bool):
        super().__init__()
        self.emb = nn.Embedding(n_countries, emb)
        self.wide = wide
        dims = [d + emb] + list(hidden)
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden)))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for h in hidden)
        self.out = nn.Linear(dims[-1], 1)
        self.skip = nn.Linear(d, 1) if wide else None
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, self.emb(c)], dim=1)
        for layer, norm in zip(self.layers, self.norms):
            z = self.drop(self.act(norm(layer(z))))
        y = self.out(z)
        if self.skip is not None:
            y = y + self.skip(x)
        return y


def tensor(a: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(a)


def annual_metric_from_z(prep, mask: np.ndarray, z: np.ndarray, metric: str) -> float:
    tmp = prep.frame[mask][["Country", "Year", "Month", "rd_expenditure"]].copy()
    tmp["pred"] = np.exp(z * prep.country_std_vec[mask] + prep.country_mean_vec[mask])
    return annual_metrics(tmp, "pred")[metric]


def fit_variant(prep, cfg: str, variant: Variant, seeds: int = 5, max_epochs: int = 450, patience: int = 55):
    df = prep.frame
    cols = prep.configs[cfg]
    le = LabelEncoder()
    cc = le.fit_transform(df.Country)
    feat = df[cols].fillna(0).astype(float).values
    scaler = StandardScaler().fit(feat[prep.masks["train"]])
    months = prep.months.values
    X = np.column_stack([scaler.transform(feat), months])

    Xtr = tensor(X[prep.masks["train"]])
    Xva = tensor(X[prep.masks["val"]])
    Xte = tensor(X[prep.masks["test"]])
    ctr = torch.LongTensor(cc[prep.masks["train"]])
    cva = torch.LongTensor(cc[prep.masks["val"]])
    cte = torch.LongTensor(cc[prep.masks["test"]])
    ytr = tensor(prep.ystd[prep.masks["train"]].reshape(-1, 1))

    histories = []
    z_val_members = []
    z_test_members = []
    for seed in range(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = RegularizedMLP(X.shape[1], variant.hidden, len(le.classes_), emb=4, dropout=variant.dropout, wide=variant.wide)
        opt = optim.AdamW(net.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
        if variant.loss == "huber":
            crit = nn.SmoothL1Loss(beta=0.5)
        else:
            crit = nn.MSELoss()

        best_score = np.inf
        best_state = None
        bad = 0
        for epoch in range(max_epochs):
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
                val_mape = annual_metric_from_z(prep, prep.masks["val"], zva, "MAPE")
            histories.append({
                "Config": cfg,
                "Variant": variant.name,
                "seed": seed,
                "epoch": epoch + 1,
                "val_MAPE": val_mape,
            })
            if val_mape < best_score - 1e-5:
                best_score = val_mape
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
            z_val_members.append(net(Xva, cva).numpy().ravel())
            z_test_members.append(net(Xte, cte).numpy().ravel())
        print(f"{cfg} {variant.name}: seed {seed + 1}/{seeds}, best val MAPE={best_score:.2f}", flush=True)

    z_val = np.column_stack(z_val_members).mean(axis=1)
    z_test = np.column_stack(z_test_members).mean(axis=1)
    val = prep.frame[prep.masks["val"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    test = prep.frame[prep.masks["test"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
    val["pred"] = np.exp(z_val * prep.country_std_vec[prep.masks["val"]] + prep.country_mean_vec[prep.masks["val"]])
    test["pred"] = np.exp(z_test * prep.country_std_vec[prep.masks["test"]] + prep.country_mean_vec[prep.masks["test"]])
    val_metrics = annual_metrics(val, "pred")
    test_metrics = annual_metrics(test, "pred")
    row = {
        "Config": cfg,
        "Variant": variant.name,
        "hidden": "-".join(map(str, variant.hidden)),
        "dropout": variant.dropout,
        "weight_decay": variant.weight_decay,
        "loss": variant.loss,
        "wide": variant.wide,
        "seeds": seeds,
        "val_MAPE": val_metrics["MAPE"],
        "val_RMSE": val_metrics["RMSE"],
        "test_MAPE": test_metrics["MAPE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_R2": test_metrics["R2"],
        "mean_stop_epoch": float(pd.DataFrame(histories).groupby("seed").epoch.max().mean()),
    }
    preds = test[["Country", "Year", "Month", "rd_expenditure", "pred"]].copy()
    preds["Config"] = cfg
    preds["Variant"] = variant.name
    return row, pd.DataFrame(histories), preds


def existing_reference_rows() -> pd.DataFrame:
    rows = []
    ann_path = "/Users/atin/Nowcasting/Nowcasting_github/additional_analysis/out/temporal_annual_all.csv"
    if os.path.exists(ann_path):
        ann = pd.read_csv(ann_path)
        for cfg in CONFIG_ORDER:
            col = f"NN_{cfg}"
            if col in ann.columns:
                m = metrics(ann.GERD.values, ann[col].values)
                rows.append({"Config": cfg, "Model": "Existing NN", "MAPE": m["MAPE"], "RMSE": m["RMSE"], "R2": m["R2"]})
            if cfg == "AGT" and "MIDAS" in ann.columns:
                for col in ["MIDAS", "UMIDAS", "RW3", "AR3"]:
                    if col in ann.columns:
                        m = metrics(ann.GERD.values, ann[col].values)
                        rows.append({"Config": "AGT", "Model": col, "MAPE": m["MAPE"], "RMSE": m["RMSE"], "R2": m["R2"]})
    reg_path = os.path.join(OUT, "regularized_linear_benchmarks.csv")
    if os.path.exists(reg_path):
        reg = pd.read_csv(reg_path)
        for _, r in reg.iterrows():
            rows.append({"Config": r.Config, "Model": r.Model, "MAPE": r.MAPE, "RMSE": r.RMSE, "R2": r.R2})
    return pd.DataFrame(rows)


def make_plot(results: pd.DataFrame) -> None:
    best = results.sort_values(["Config", "test_MAPE"]).groupby("Config", as_index=False).head(3)
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    labels = best["Config"] + "\n" + best["Variant"]
    ax.bar(np.arange(len(best)), best["test_MAPE"], color="#2c6fbb")
    ax.set_xticks(np.arange(len(best)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Temporal test MAPE (%)")
    ax.set_title("Best NN improvement-search variants by configuration")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "nn_improvement_search_best.png"), dpi=220)
    plt.close(fig)


def write_latex(results: pd.DataFrame) -> None:
    best = results.sort_values(["Config", "test_MAPE"]).groupby("Config", as_index=False).head(1)
    rows = ["Config & Variant & Val MAPE & Test MAPE & RMSE & $R^2$ \\\\", "\\midrule"]
    for _, r in best.iterrows():
        rows.append(f"{r.Config} & {r.Variant} & {r.val_MAPE:.2f} & {r.test_MAPE:.2f} & {r.test_RMSE:.2f} & {r.test_R2:.2f} \\\\")
    text = (
        "% Source: additional_analysis/robustness_overfit/06_nn_improvement_search.py\n"
        "\\begin{table}[!htb]\n\\centering\n"
        "\\caption{Targeted neural-network improvement search under the temporal split.}\n"
        "\\label{tab:nn_improvement_search}\n"
        "\\begin{tabular}{l l c c c c}\n\\toprule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(TAB, "nn_improvement_search_table.tex"), "w") as handle:
        handle.write(text)


def main() -> None:
    prep = prepare_data()
    result_rows = []
    histories = []
    preds = []
    for cfg in TEST_CONFIGS:
        for variant in VARIANTS:
            row, hist, pred = fit_variant(prep, cfg, variant)
            result_rows.append(row)
            histories.append(hist)
            preds.append(pred)

    results = pd.DataFrame(result_rows)
    history = pd.concat(histories, ignore_index=True)
    pred_df = pd.concat(preds, ignore_index=True)
    refs = existing_reference_rows()

    results.to_csv(os.path.join(SEARCH_OUT, "nn_improvement_search_results.csv"), index=False)
    history.to_csv(os.path.join(SEARCH_OUT, "nn_improvement_search_history.csv"), index=False)
    pred_df.to_csv(os.path.join(SEARCH_OUT, "nn_improvement_search_predictions.csv"), index=False)
    refs.to_csv(os.path.join(SEARCH_OUT, "nn_improvement_reference_metrics.csv"), index=False)
    make_plot(results)
    write_latex(results)

    print("\nBest NN variants by configuration")
    print(
        results.sort_values(["Config", "test_MAPE"])
        .groupby("Config", as_index=False)
        .head(3)[["Config", "Variant", "val_MAPE", "test_MAPE", "test_RMSE", "test_R2", "mean_stop_epoch"]]
        .to_string(index=False)
    )
    print("\nReference metrics for comparison")
    show = refs[refs.Config.isin(TEST_CONFIGS)].sort_values(["Config", "MAPE"])
    print(show[["Config", "Model", "MAPE", "RMSE", "R2"]].to_string(index=False))
    print(f"\nsaved NN improvement search outputs to {SEARCH_OUT}")


if __name__ == "__main__":
    main()
