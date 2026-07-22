"""Shared utilities for additive robustness analyses.

This folder is intentionally self-contained: scripts read the existing paper
inputs under additional_analysis/out, but write all new outputs under
additional_analysis/robustness_overfit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = str(Path(os.environ.get("NOWCASTING_ROOT", Path(__file__).resolve().parents[2])))
PAPER_ROOT = str(Path(os.environ.get("NOWCASTING_PAPER_DIR", Path(ROOT) / "paper")))
SOURCE_OUT = os.path.join(ROOT, "additional_analysis", "out")
ROBUST_ROOT = os.path.join(ROOT, "additional_analysis", "robustness_overfit")
OUT = os.path.join(ROBUST_ROOT, "out")
FIG = os.path.join(ROBUST_ROOT, "figures")
TAB = os.path.join(ROBUST_ROOT, "tables")

for path in (OUT, FIG, TAB):
    os.makedirs(path, exist_ok=True)


CONFIG_ORDER = ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    masks: dict[str, np.ndarray]
    configs: dict[str, list[str]]
    months: pd.DataFrame
    countries: pd.DataFrame
    logy: np.ndarray
    ystd: np.ndarray
    cmean: dict[str, float]
    cstd: dict[str, float]
    country_mean_vec: np.ndarray
    country_std_vec: np.ndarray


def load_features() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(SOURCE_OUT, "merged_features.csv"))
    df = df[df.Year >= 2004].copy()
    df.sort_values(["Country", "Year", "Month"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def feature_configs(df: pd.DataFrame) -> dict[str, list[str]]:
    ar = [f"rd_expenditure_lag{lag}" for lag in (1, 2, 3)]
    mac = [
        f"{var}_lag{lag}"
        for var in ["gdpca", "unemp_rate", "population", "inflation", "export_vol", "import_vol"]
        for lag in (1, 2, 3)
    ]
    agt = [c for c in df.columns if "_yearly_avg_lag" in c]
    ytd = [c for c in df.columns if c.endswith("_mean_YTD")]
    return {
        "LagRD": ar,
        "Macros": ar + mac,
        "AGT": agt,
        "MGT": agt + ytd,
        "AGTwRD": ar + agt,
        "MGTwRD": ar + agt + ytd,
        "AllVar": ar + mac + agt + ytd,
    }


def add_chronological_split(df: pd.DataFrame, train_share: float = 0.64, val_share: float = 0.16) -> pd.DataFrame:
    split = {}
    for ctry, g in df.groupby("Country"):
        years = np.array(sorted(g.Year.unique()))
        n = len(years)
        n_train = int(round(n * train_share))
        n_val = int(round(n * val_share))
        for i, year in enumerate(years):
            split[(ctry, int(year))] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    out = df.copy()
    out["split"] = [split[(c, int(y))] for c, y in zip(out.Country, out.Year)]
    return out


def prepare_data() -> PreparedData:
    df = add_chronological_split(load_features())
    masks = {name: (df.split == name).values for name in ("train", "val", "test")}
    months = pd.get_dummies(df.Month.astype(int), prefix="M").astype(float)
    countries = pd.get_dummies(df.Country, prefix="c").astype(float)

    logy = np.log(df.rd_expenditure.values.astype(float))
    cmean: dict[str, float] = {}
    cstd: dict[str, float] = {}
    for ctry, g in df.groupby("Country"):
        v = np.log(g.rd_expenditure.values[(g.split == "train").values])
        cmean[ctry] = float(v.mean())
        cstd[ctry] = float(max(v.std(), 0.05))
    cm = df.Country.map(cmean).values
    cs = df.Country.map(cstd).values
    ystd = (logy - cm) / cs
    return PreparedData(
        frame=df,
        masks=masks,
        configs=feature_configs(df),
        months=months,
        countries=countries,
        logy=logy,
        ystd=ystd,
        cmean=cmean,
        cstd=cstd,
        country_mean_vec=cm,
        country_std_vec=cs,
    )


def level_from_z(prep: PreparedData, z: np.ndarray) -> np.ndarray:
    return np.exp(z * prep.country_std_vec + prep.country_mean_vec)


def design_matrix(prep: PreparedData, cols: list[str], fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xraw = prep.frame[cols].fillna(0).astype(float).values
    mu = Xraw[fit_mask].mean(axis=0)
    sd = Xraw[fit_mask].std(axis=0)
    sd[sd == 0] = 1.0
    X = np.column_stack([
        (Xraw - mu) / sd,
        prep.months.values,
        prep.countries.values,
    ])
    return X, mu, sd


def annualize(frame: pd.DataFrame, pred_col: str, true_col: str = "rd_expenditure") -> pd.DataFrame:
    return (
        frame.groupby(["Country", "Year"], as_index=False)
        .agg(Actual=(true_col, "mean"), Pred=(pred_col, "mean"))
    )


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    err = true - pred
    return {
        "n": float(len(true)),
        "MAPE": float(np.mean(np.abs(err / true)) * 100),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2)) if len(true) > 1 else np.nan,
    }


def annual_metrics(df: pd.DataFrame, pred_col: str, true_col: str = "rd_expenditure") -> dict[str, float]:
    ann = annualize(df, pred_col, true_col)
    return metrics(ann["Actual"].values, ann["Pred"].values)


def write_latex_table(path: str, caption: str, label: str, rows: list[str], columns: str) -> None:
    body = "\n".join(rows)
    text = (
        "% Source: additional_analysis/robustness_overfit\n"
        "\\begin{table}[!htb]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{columns}}}\n"
        "\\toprule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(path, "w") as handle:
        handle.write(text)
