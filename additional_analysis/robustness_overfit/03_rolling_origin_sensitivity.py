"""Lightweight rolling-origin sensitivity for the AGT configuration.

For each origin year, train on years < origin-2, validate on the two years
immediately before the origin, and test on the origin year. Countries are
included only if they have enough training and validation history. This is not
a large-sample time-series CV; it is a small-sample sensitivity check.
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(os.path.dirname(__file__))
from common import FIG, OUT, TAB, feature_configs, load_features, metrics  # noqa: E402


warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(0)
torch.manual_seed(0)

ORIGINS = [2017, 2018, 2019]
MIN_TRAIN_YEARS = 6
VAL_WINDOW = 2
ENSEMBLE_SIZE = 3
EPOCHS = 400
PATIENCE = 50


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

    def forward(self, x, c):
        z = torch.cat([x, self.emb(c)], dim=1)
        for lin, bn in zip(self.lin, self.bn):
            z = self.drop(self.relu(bn(lin(z))))
        return self.out(z)


def tensor(a: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(a)


def build_origin_frame(full: pd.DataFrame, origin: int) -> pd.DataFrame:
    countries = []
    for ctry, g in full.groupby("Country"):
        years = set(int(y) for y in g.Year.unique())
        train_years = [y for y in years if y < origin - VAL_WINDOW]
        val_years = [y for y in years if origin - VAL_WINDOW <= y < origin]
        if len(train_years) >= MIN_TRAIN_YEARS and len(val_years) >= 1 and origin in years:
            countries.append(ctry)
    df = full[full.Country.isin(countries)].copy()
    df["split"] = "unused"
    df.loc[df.Year < origin - VAL_WINDOW, "split"] = "train"
    df.loc[(df.Year >= origin - VAL_WINDOW) & (df.Year < origin), "split"] = "val"
    df.loc[df.Year == origin, "split"] = "test"
    df = df[df.split != "unused"].copy()
    return df


def target_stats(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    cmean, cstd = {}, {}
    for ctry, g in df.groupby("Country"):
        v = np.log(g.loc[g.split == "train", "rd_expenditure"].values.astype(float))
        cmean[ctry] = float(v.mean())
        cstd[ctry] = float(max(v.std(), 0.05))
    logy = np.log(df.rd_expenditure.values.astype(float))
    ystd = (logy - df.Country.map(cmean).values) / df.Country.map(cstd).values
    return ystd, cmean, cstd


def to_level(df: pd.DataFrame, z: np.ndarray, cmean: dict[str, float], cstd: dict[str, float]) -> np.ndarray:
    return np.exp(z * df.Country.map(cstd).values + df.Country.map(cmean).values)


def annual_test_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    t = df[df.split == "test"][["Country", "Year", "Month", "rd_expenditure"]].copy()
    t["pred"] = pred[df.split.values == "test"]
    ann = t.groupby(["Country", "Year"], as_index=False).agg(Actual=("rd_expenditure", "mean"), Pred=("pred", "mean"))
    m = metrics(ann["Actual"].values, ann["Pred"].values)
    m["n_country_years"] = float(len(ann))
    return m


def nn_agt(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, float]:
    masks = {s: (df.split == s).values for s in ("train", "val", "test")}
    ystd, cmean, cstd = target_stats(df)
    le = LabelEncoder()
    cc = le.fit_transform(df.Country)
    months = pd.get_dummies(df.Month.astype(int), prefix="M").astype(float)
    Xraw = df[cols].fillna(0).astype(float).values
    scaler = StandardScaler().fit(Xraw[masks["train"]])
    X = np.hstack([scaler.transform(Xraw), months.values])
    Xtr, Xva, Xte = tensor(X[masks["train"]]), tensor(X[masks["val"]]), tensor(X[masks["test"]])
    ctr = torch.LongTensor(cc[masks["train"]])
    cva = torch.LongTensor(cc[masks["val"]])
    cte = torch.LongTensor(cc[masks["test"]])
    ytr = tensor(ystd[masks["train"]].reshape(-1, 1))
    yva = tensor(ystd[masks["val"]].reshape(-1, 1))
    preds = []
    stop_epochs = []
    for member in range(ENSEMBLE_SIZE):
        torch.manual_seed(member)
        net = MLP(X.shape[1], [200, 20, 20], len(le.classes_))
        opt = optim.AdamW(net.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = MultiStepLR(opt, [250], 0.1)
        crit = nn.MSELoss()
        best = np.inf
        bad = 0
        best_state = None
        stop = EPOCHS
        for ep in range(EPOCHS):
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
                val_loss = crit(net(Xva, cva), yva).item()
            if val_loss < best - 1e-7:
                best = val_loss
                bad = 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    stop = ep + 1
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            preds.append(net(tensor(X), torch.LongTensor(cc)).numpy().ravel())
        stop_epochs.append(stop)
    z = np.column_stack(preds).mean(axis=1)
    return to_level(df, z, cmean, cstd), float(np.mean(stop_epochs))


def elastic_net_agt(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, str]:
    masks = {s: (df.split == s).values for s in ("train", "val", "test")}
    ystd, cmean, cstd = target_stats(df)
    months = pd.get_dummies(df.Month.astype(int), prefix="M").astype(float)
    countries = pd.get_dummies(df.Country, prefix="c").astype(float)
    Xraw = df[cols].fillna(0).astype(float).values
    scaler = StandardScaler().fit(Xraw[masks["train"]])
    X = np.column_stack([scaler.transform(Xraw), months.values, countries.values])
    best = None
    for l1_ratio in [0.05, 0.2, 0.5, 0.8]:
        for alpha in np.logspace(-4, 1, 18):
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000, tol=1e-5)
            model.fit(X[masks["train"]], ystd[masks["train"]])
            pred = to_level(df, model.predict(X), cmean, cstd)
            val = df[masks["val"]][["Country", "Year", "Month", "rd_expenditure"]].copy()
            val["pred"] = pred[masks["val"]]
            ann = val.groupby(["Country", "Year"], as_index=False).agg(Actual=("rd_expenditure", "mean"), Pred=("pred", "mean"))
            score = metrics(ann["Actual"].values, ann["Pred"].values)["MAPE"]
            if best is None or score < best[0]:
                best = (score, alpha, l1_ratio)
    _, alpha, l1_ratio = best
    fit = masks["train"] | masks["val"]
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000, tol=1e-5)
    model.fit(X[fit], ystd[fit])
    return to_level(df, model.predict(X), cmean, cstd), f"alpha={alpha:.3g}, l1={l1_ratio:.2g}"


def rw3(df: pd.DataFrame) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    annual = (
        load_features()
        .groupby(["Country", "Year"], as_index=False)
        .agg(GERD=("rd_expenditure", "mean"))
    )
    lookup = {(c, int(y)): v for c, y, v in zip(annual.Country, annual.Year, annual.GERD)}
    for i, (c, y) in enumerate(zip(df.Country, df.Year)):
        pred[i] = lookup.get((c, int(y) - 3), np.nan)
    return pred


def latex_table(results: pd.DataFrame) -> None:
    rows = [
        "Origin & Model & Countries & MAPE (\\%) & RMSE & $R^2$ \\\\",
        "\\midrule",
    ]
    for _, r in results.iterrows():
        rows.append(
            f"{int(r.origin)} & {r.Model} & {int(r.n_country_years)} & {r.MAPE:.2f} & {r.RMSE:.2f} & {r.R2:.2f} \\\\"
        )
    text = (
        "% Source: additional_analysis/robustness_overfit/03_rolling_origin_sensitivity.py\n"
        "\\begin{table}[!htb]\n\\centering\n"
        "\\caption{Rolling-origin sensitivity for the AGT configuration. For each origin, models are trained on years before the two-year validation window, tuned or early-stopped on the validation window, and evaluated on the origin year. The exercise is a small-sample robustness check rather than a full time-series cross-validation design.}\n"
        "\\label{tab:rolling_origin_sensitivity}\n"
        "\\begin{tabular}{l l c c c c}\n\\toprule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(TAB, "rolling_origin_sensitivity_table.tex"), "w") as handle:
        handle.write(text)


def plot_results(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4))
    models = ["NN-AGT", "Elastic Net", "RW(3)"]
    width = 0.24
    x = np.arange(len(ORIGINS))
    colors = {"NN-AGT": "#2c7fb8", "Elastic Net": "#59a14f", "RW(3)": "#d95f0e"}
    for i, model in enumerate(models):
        vals = [results[(results.origin == o) & (results.Model == model)].MAPE.iloc[0] for o in ORIGINS]
        ax.bar(x + (i - 1) * width, vals, width, label=model, color=colors[model])
    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in ORIGINS])
    ax.set_xlabel("Test origin year")
    ax.set_ylabel("MAPE (%)")
    ax.grid(axis="y", color="#e6e6e6")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "rolling_origin_sensitivity.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    full = load_features()
    cols = feature_configs(full)["AGT"]
    rows = []
    detail_rows = []
    for origin in ORIGINS:
        df = build_origin_frame(full, origin)
        print(f"origin {origin}: countries={sorted(df.Country.unique())}", flush=True)
        model_preds = {}
        pred, stop_epoch = nn_agt(df, cols)
        model_preds["NN-AGT"] = (pred, f"mean_stop_epoch={stop_epoch:.1f}")
        pred, tuning = elastic_net_agt(df, cols)
        model_preds["Elastic Net"] = (pred, tuning)
        model_preds["RW(3)"] = (rw3(df), "")

        for model, (pred, note) in model_preds.items():
            m = annual_test_metrics(df, pred)
            rows.append({"origin": origin, "Model": model, "note": note, **m})
            t = df[df.split == "test"][["Country", "Year", "Month", "rd_expenditure"]].copy()
            t["pred"] = pred[df.split.values == "test"]
            ann = t.groupby(["Country", "Year"], as_index=False).agg(Actual=("rd_expenditure", "mean"), Pred=("pred", "mean"))
            ann["origin"] = origin
            ann["Model"] = model
            detail_rows.append(ann)

    results = pd.DataFrame(rows)
    details = pd.concat(detail_rows, ignore_index=True)
    results.to_csv(os.path.join(OUT, "rolling_origin_sensitivity.csv"), index=False)
    details.to_csv(os.path.join(OUT, "rolling_origin_predictions.csv"), index=False)
    latex_table(results)
    plot_results(results)
    print(results.to_string(index=False))
    print("saved rolling-origin sensitivity")


if __name__ == "__main__":
    main()
