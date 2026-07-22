"""Country-year robustness run using the main temporal NN specification.

All outputs are isolated in this directory's ``out`` folder. The country-year
split is identical to the existing robustness experiment; only the NN recipe is
aligned with the current main-body specification.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nowcasting-mpl")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_OUT = ROOT / "additional_analysis" / "out"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
HIDDEN = (64, 16)
DROPOUT = 0.12
WEIGHT_DECAY = 5e-4
LR = 3e-3
LOSS_BETA = 0.5
ENSEMBLE_SIZE = 15
EPOCHS = 500
PATIENCE = 65

CONFIG_ORDER = ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]


class WideDeepNN(nn.Module):
    def __init__(self, d: int, n_countries: int):
        super().__init__()
        self.embedding = nn.Embedding(n_countries, 4)
        dims = [d + 4] + list(HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(HIDDEN)))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for h in HIDDEN)
        self.output = nn.Linear(dims[-1], 1)
        self.skip = nn.Linear(d, 1)
        self.dropout = nn.Dropout(DROPOUT)
        self.activation = nn.SiLU()

    def forward(self, x, country):
        z = torch.cat([x, self.embedding(country)], dim=1)
        for layer, norm in zip(self.layers, self.norms):
            z = self.dropout(self.activation(norm(layer(z))))
        return self.output(z) + self.skip(x)


def annual_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    annual = frame.groupby(["Country", "Year"], as_index=False).agg(
        true=("rd_expenditure", "mean"), pred=(pred_col, "mean")
    )
    err = annual.true.to_numpy() - annual.pred.to_numpy()
    return {
        "MAPE": np.mean(np.abs(err / annual.true.to_numpy())) * 100,
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAE": np.mean(np.abs(err)),
        "n": len(annual),
    }


def dm_test(true, p1, p2):
    true, p1, p2 = map(lambda x: np.asarray(x, float), (true, p1, p2))
    d = (true - p1) ** 2 - (true - p2) ** 2
    n = len(d)
    stat = d.mean() / np.sqrt(np.var(d, ddof=1) / n)
    stat *= np.sqrt((n - 1) / n)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=n - 1)), n


def prepare_data():
    frame = pd.read_csv(SOURCE_OUT / "merged_features.csv")
    frame = frame[frame.Year >= 2004].copy().sort_values(["Country", "Year", "Month"])

    ar = [f"rd_expenditure_lag{lag}" for lag in (1, 2, 3)]
    macro = [
        f"{var}_lag{lag}"
        for var in ["gdpca", "unemp_rate", "population", "inflation", "export_vol", "import_vol"]
        for lag in (1, 2, 3)
    ]
    agt = [c for c in frame.columns if "_yearly_avg_lag" in c]
    ytd = [c for c in frame.columns if c.endswith("_mean_YTD")]
    configs = {
        "LagRD": ar,
        "Macros": ar + macro,
        "AGT": agt,
        "MGT": agt + ytd,
        "AGTwRD": ar + agt,
        "MGTwRD": ar + agt + ytd,
        "AllVar": ar + macro + agt + ytd,
    }

    rng = np.random.default_rng(SEED)
    split_map = {}
    for country, group in frame.groupby("Country"):
        years = np.array(sorted(group.Year.unique()))
        rng.shuffle(years)
        n_train = int(round(len(years) * 0.64))
        n_val = int(round(len(years) * 0.16))
        for i, year in enumerate(years):
            split_map[(country, int(year))] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    frame["split"] = [split_map[(c, int(y))] for c, y in zip(frame.Country, frame.Year)]
    masks = {name: frame.split.eq(name).to_numpy() for name in ("train", "val", "test")}

    # Identical train-only, country-specific log standardization to the main run.
    log_y = np.log(frame.rd_expenditure.to_numpy(float))
    train_stats = frame.loc[masks["train"], ["Country"]].copy()
    train_stats["log_y"] = log_y[masks["train"]]
    means = train_stats.groupby("Country").log_y.mean().to_dict()
    stds = train_stats.groupby("Country").log_y.std().replace(0, 1).to_dict()
    mean_vec = np.array([means[c] for c in frame.Country])
    std_vec = np.array([stds[c] for c in frame.Country])
    y_std = (log_y - mean_vec) / std_vec

    months = pd.get_dummies(frame.Month, prefix="M").astype(float)
    labels = LabelEncoder().fit_transform(frame.Country)
    return frame, configs, masks, y_std, mean_vec, std_vec, months, labels


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    frame, configs, masks, y_std, mean_vec, std_vec, months, labels = prepare_data()
    split_counts = frame.groupby("split").size().to_dict()
    cy_counts = frame.groupby("split").apply(lambda x: x[["Country", "Year"]].drop_duplicates().shape[0]).to_dict()
    print("rows:", split_counts, "country-years:", cy_counts, flush=True)

    all_predictions, all_histories, metric_rows, coverage_rows = [], [], [], []
    for config in CONFIG_ORDER:
        columns = configs[config]
        raw = frame[columns].fillna(0).astype(float).to_numpy()
        scaler = StandardScaler().fit(raw[masks["train"]])
        x = np.column_stack([scaler.transform(raw), months.to_numpy()])
        tensors = {
            split: (
                torch.FloatTensor(x[mask]),
                torch.LongTensor(labels[mask]),
            )
            for split, mask in masks.items()
        }
        y_train = torch.FloatTensor(y_std[masks["train"]].reshape(-1, 1))
        criterion = nn.SmoothL1Loss(beta=LOSS_BETA)
        members, histories = [], []

        for seed in range(ENSEMBLE_SIZE):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = WideDeepNN(x.shape[1], len(np.unique(labels)))
            optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            best, bad, best_state = np.inf, 0, None
            xtr, ctr = tensors["train"]
            xval, cval = tensors["val"]
            for epoch in range(EPOCHS):
                net.train()
                perm = torch.randperm(len(xtr))
                for start in range(0, len(xtr), 64):
                    idx = perm[start : start + 64]
                    optimizer.zero_grad()
                    loss = criterion(net(xtr[idx], ctr[idx]), y_train[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()
                net.eval()
                with torch.no_grad():
                    z_val = net(xval, cval).numpy().ravel()
                val_pred = np.exp(z_val * std_vec[masks["val"]] + mean_vec[masks["val"]])
                val_frame = frame.loc[masks["val"], ["Country", "Year", "rd_expenditure"]].copy()
                val_frame["pred"] = val_pred
                val_mape = annual_metrics(val_frame, "pred")["MAPE"]
                histories.append({"Config": config, "seed": seed, "epoch": epoch + 1, "val_MAPE": val_mape})
                if val_mape < best - 1e-5:
                    best, bad = val_mape, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                z_test = net(*tensors["test"]).numpy().ravel()
            members.append(np.exp(z_test * std_vec[masks["test"]] + mean_vec[masks["test"]]))
            print(f"{config} member {seed + 1:02d}/{ENSEMBLE_SIZE}: epoch={epoch + 1}, val MAPE={best:.2f}", flush=True)

        member_matrix = np.column_stack(members)
        test = frame.loc[masks["test"], ["Country", "Year", "Month", "rd_expenditure"]].copy()
        test["Config"] = config
        test["pred_mean"] = member_matrix.mean(axis=1)
        for i in range(ENSEMBLE_SIZE):
            test[f"m{i}"] = member_matrix[:, i]
        all_predictions.append(test)
        all_histories.extend(histories)
        metrics = annual_metrics(test, "pred_mean")
        metric_rows.append({"Config": config, **metrics})
        lo, hi = np.quantile(member_matrix, [0.025, 0.975], axis=1)
        mu, sd = member_matrix.mean(axis=1), member_matrix.std(axis=1)
        truth = test.rd_expenditure.to_numpy()
        coverage_rows.append({
            "Config": config,
            "quantile_coverage": np.mean((truth >= lo) & (truth <= hi)) * 100,
            "gaussian_coverage": np.mean((truth >= mu - 1.96 * sd) & (truth <= mu + 1.96 * sd)) * 100,
            "relative_width": np.mean((hi - lo) / np.abs(mu)) * 100,
        })

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    coverage = pd.DataFrame(coverage_rows)
    predictions.to_csv(OUT / "cy_current_nn_predictions.csv", index=False)
    pd.DataFrame(all_histories).to_csv(OUT / "cy_current_nn_history.csv", index=False)
    metrics.to_csv(OUT / "cy_current_nn_metrics.csv", index=False)
    coverage.to_csv(OUT / "cy_current_nn_coverage.csv", index=False)

    old = pd.read_csv(SOURCE_OUT / "cy_skill_scores.csv").rename(columns={"Model": "Config"})
    comparison = metrics.merge(old[["Config", "MAPE", "RMSE", "OOS_R2_vs_RW"]], on="Config", suffixes=("_current", "_old"))
    comparison.to_csv(OUT / "comparison_with_existing_cy.csv", index=False)
    print("\nCURRENT NN METRICS\n", metrics.round(3).to_string(index=False), flush=True)
    print("\nCOMPARISON\n", comparison.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
