"""Regenerate Step B from the matched-refit AGT Step A model.

The neural-network ensemble uses the validation-selected training duration for
each member, then refits on train + validation with preprocessing recomputed on
that combined sample. Model-implied elasticities perturb each lag-specific
Google Trends feature by one percent on its original scale before applying the
refit StandardScaler. Annual totals remain observed GERD totals, as in the
existing retrospective Step B comparison.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "out", "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.dirname(__file__))
from common import prepare_data  # noqa: E402


CODE = Path(os.environ.get("NOWCASTING_ROOT", Path(__file__).resolve().parents[2]))
ROBUST = CODE / "additional_analysis" / "robustness_overfit"
STEP_A_OUT = ROBUST / "out" / "all_configs_updated_pure_nn"
OUT = ROBUST / "out" / "stepB_matched_refit"
DECK_ASSETS = Path(
    os.environ.get(
        "NOWCASTING_DECK_ASSETS",
        CODE / "paper" / "figures" / "UpdatedResults",
    )
)

HIDDEN = (64, 16)
DROPOUT = 0.12
WEIGHT_DECAY = 5e-4
LR = 3e-3
ENSEMBLE_SIZE = 15
RIDGE_ALPHAS = np.logspace(-3, 3, 13)


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


@dataclass(frozen=True)
class TargetTransform:
    ystd: np.ndarray
    country_mean_vec: np.ndarray
    country_std_vec: np.ndarray


def target_transform(frame: pd.DataFrame, fit_mask: np.ndarray) -> TargetTransform:
    logy = np.log(frame.rd_expenditure.to_numpy(dtype=float))
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for country in frame.Country.unique():
        country_mask = frame.Country.eq(country).to_numpy()
        values = logy[country_mask & fit_mask]
        means[country] = float(values.mean())
        stds[country] = float(max(values.std(), 0.05))
    mean_vec = frame.Country.map(means).to_numpy(dtype=float)
    std_vec = frame.Country.map(stds).to_numpy(dtype=float)
    return TargetTransform(
        ystd=(logy - mean_vec) / std_vec,
        country_mean_vec=mean_vec,
        country_std_vec=std_vec,
    )


def selected_agt_epochs() -> dict[int, int]:
    history = pd.read_csv(STEP_A_OUT / "all_configs_updated_pure_nn_history.csv")
    history = history[history.Config == "AGT"].copy()
    best = history.loc[history.groupby("seed").val_MAPE.idxmin()]
    epochs = {int(row.seed): int(row.epoch) for row in best.itertuples()}
    if sorted(epochs) != list(range(ENSEMBLE_SIZE)):
        raise RuntimeError("The saved AGT history does not contain all 15 ensemble members.")
    return epochs


def topics_from_columns(columns: list[str]) -> list[str]:
    topics = set()
    for column in columns:
        for lag in (1, 2, 3):
            suffix = f"_yearly_avg_lag{lag}"
            if column.endswith(suffix):
                topics.add(column[: -len(suffix)])
    return sorted(topics)


def train_matched_agt(prep, fit_mask: np.ndarray, target: TargetTransform):
    columns = prep.configs["AGT"]
    raw = prep.frame[columns].fillna(0).astype(float).to_numpy()
    scaler = StandardScaler().fit(raw[fit_mask])
    x = np.column_stack([scaler.transform(raw), prep.months.to_numpy()])
    encoder = LabelEncoder()
    country_codes = encoder.fit_transform(prep.frame.Country)

    x_fit = torch.FloatTensor(x[fit_mask])
    c_fit = torch.LongTensor(country_codes[fit_mask])
    y_fit = torch.FloatTensor(target.ystd[fit_mask].reshape(-1, 1))
    criterion = nn.SmoothL1Loss(beta=0.5)
    epochs = selected_agt_epochs()
    models = []
    training_rows = []

    for seed in range(ENSEMBLE_SIZE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = WideDeepNN(x.shape[1], len(encoder.classes_))
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for epoch in range(epochs[seed]):
            model.train()
            permutation = torch.randperm(len(x_fit))
            running_loss = 0.0
            for start in range(0, len(x_fit), 64):
                idx = permutation[start : start + 64]
                optimizer.zero_grad()
                loss = criterion(model(x_fit[idx], c_fit[idx]), y_fit[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running_loss += float(loss.detach()) * len(idx)
            training_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "fit_loss": running_loss / len(x_fit),
                    "selected_epochs": epochs[seed],
                }
            )
        model.eval()
        models.append(model)
        print(f"AGT matched refit: member {seed + 1}/{ENSEMBLE_SIZE}, epochs={epochs[seed]}", flush=True)

    return models, raw, scaler, x, country_codes, columns, pd.DataFrame(training_rows)


def nn_level_predictions(
    models: list[nn.Module],
    x: np.ndarray,
    country_codes: np.ndarray,
    target: TargetTransform,
) -> np.ndarray:
    tx = torch.FloatTensor(x)
    tc = torch.LongTensor(country_codes)
    members = []
    for model in models:
        model.eval()
        with torch.no_grad():
            z = model(tx, tc).numpy().ravel()
        members.append(np.exp(z * target.country_std_vec + target.country_mean_vec))
    return np.column_stack(members).mean(axis=1)


def nn_raw_scale_elasticities(
    models: list[nn.Module],
    x: np.ndarray,
    raw: np.ndarray,
    scaler: StandardScaler,
    country_codes: np.ndarray,
    target: TargetTransform,
) -> np.ndarray:
    """Mean member-level elasticity to a 1% raw-input change."""
    tx = torch.FloatTensor(x)
    tc = torch.LongTensor(country_codes)
    output = np.zeros((len(x), raw.shape[1]), dtype=float)
    for member, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            base_z = model(tx, tc).numpy().ravel()
        base = np.exp(base_z * target.country_std_vec + target.country_mean_vec)
        denominator = np.where(np.abs(base) > 1e-12, base, np.nan)
        for j in range(raw.shape[1]):
            xp = x.copy()
            xp[:, j] = (raw[:, j] * 1.01 - scaler.mean_[j]) / scaler.scale_[j]
            with torch.no_grad():
                perturbed_z = model(torch.FloatTensor(xp), tc).numpy().ravel()
            perturbed = np.exp(
                perturbed_z * target.country_std_vec + target.country_mean_vec
            )
            output[:, j] += ((perturbed - base) / denominator) / 0.01
        print(
            f"NN raw-scale sensitivities: member {member + 1}/{len(models)}",
            flush=True,
        )
    return output / len(models)


def nn_standardized_input_sensitivities(
    models: list[nn.Module],
    x: np.ndarray,
    country_codes: np.ndarray,
    target: TargetTransform,
    n_gt: int,
) -> np.ndarray:
    """Legacy diagnostic: perturb the already-standardized model input by 1%."""
    tx = torch.FloatTensor(x)
    tc = torch.LongTensor(country_codes)
    output = np.zeros((len(x), n_gt), dtype=float)
    for member, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            base_z = model(tx, tc).numpy().ravel()
        base = np.exp(base_z * target.country_std_vec + target.country_mean_vec)
        denominator = np.where(np.abs(base) > 1e-12, base, np.nan)
        for j in range(n_gt):
            xp = x.copy()
            xp[:, j] *= 1.01
            with torch.no_grad():
                perturbed_z = model(torch.FloatTensor(xp), tc).numpy().ravel()
            perturbed = np.exp(
                perturbed_z * target.country_std_vec + target.country_mean_vec
            )
            output[:, j] += ((perturbed - base) / denominator) / 0.01
        print(
            f"NN standardized-input diagnostic: member {member + 1}/{len(models)}",
            flush=True,
        )
    return output / len(models)


def nn_additive_standardized_sensitivities(
    models: list[nn.Module],
    x: np.ndarray,
    country_codes: np.ndarray,
    target: TargetTransform,
    n_gt: int,
    delta: float = 0.01,
) -> np.ndarray:
    """Semi-elasticity to a common additive change in standardized input space."""
    tx = torch.FloatTensor(x)
    tc = torch.LongTensor(country_codes)
    output = np.zeros((len(x), n_gt), dtype=float)
    for member, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            base_z = model(tx, tc).numpy().ravel()
        base = np.exp(base_z * target.country_std_vec + target.country_mean_vec)
        denominator = np.where(np.abs(base) > 1e-12, base, np.nan)
        for j in range(n_gt):
            xp = x.copy()
            xp[:, j] += delta
            with torch.no_grad():
                perturbed_z = model(torch.FloatTensor(xp), tc).numpy().ravel()
            perturbed = np.exp(
                perturbed_z * target.country_std_vec + target.country_mean_vec
            )
            output[:, j] += ((perturbed - base) / denominator) / delta
        print(
            f"NN additive-standardized sensitivities: member {member + 1}/{len(models)}",
            flush=True,
        )
    return output / len(models)


def nn_raw_plus_one_sensitivities(
    models: list[nn.Module],
    x: np.ndarray,
    raw: np.ndarray,
    scaler: StandardScaler,
    country_codes: np.ndarray,
    target: TargetTransform,
) -> np.ndarray:
    """Semi-elasticity to a one-point increase in each original GT series."""
    tx = torch.FloatTensor(x)
    tc = torch.LongTensor(country_codes)
    output = np.zeros((len(x), raw.shape[1]), dtype=float)
    for member, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            base_z = model(tx, tc).numpy().ravel()
        base = np.exp(base_z * target.country_std_vec + target.country_mean_vec)
        denominator = np.where(np.abs(base) > 1e-12, base, np.nan)
        for j in range(raw.shape[1]):
            xp = x.copy()
            xp[:, j] = (raw[:, j] + 1.0 - scaler.mean_[j]) / scaler.scale_[j]
            with torch.no_grad():
                perturbed_z = model(torch.FloatTensor(xp), tc).numpy().ravel()
            perturbed = np.exp(
                perturbed_z * target.country_std_vec + target.country_mean_vec
            )
            output[:, j] += (perturbed - base) / denominator
        print(
            f"NN raw-plus-one sensitivities: member {member + 1}/{len(models)}",
            flush=True,
        )
    return output / len(models)


def nn_central_standardized_sensitivities(
    models: list[nn.Module],
    x: np.ndarray,
    country_codes: np.ndarray,
    target: TargetTransform,
    n_gt: int,
    delta: float = 0.10,
) -> np.ndarray:
    """Symmetric semi-elasticity for a common standardized-input perturbation."""
    tx = torch.FloatTensor(x)
    tc = torch.LongTensor(country_codes)
    output = np.zeros((len(x), n_gt), dtype=float)
    for member, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            base_z = model(tx, tc).numpy().ravel()
        base = np.exp(base_z * target.country_std_vec + target.country_mean_vec)
        denominator = np.where(np.abs(base) > 1e-12, base, np.nan)
        for j in range(n_gt):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[:, j] += delta
            x_minus[:, j] -= delta
            with torch.no_grad():
                plus_z = model(torch.FloatTensor(x_plus), tc).numpy().ravel()
                minus_z = model(torch.FloatTensor(x_minus), tc).numpy().ravel()
            plus = np.exp(plus_z * target.country_std_vec + target.country_mean_vec)
            minus = np.exp(minus_z * target.country_std_vec + target.country_mean_vec)
            output[:, j] += ((plus - minus) / denominator) / (2.0 * delta)
        print(
            f"NN central-standardized sensitivities: member {member + 1}/{len(models)}",
            flush=True,
        )
    return output / len(models)


def annual_mape(frame: pd.DataFrame, predictions: np.ndarray, mask: np.ndarray) -> float:
    annual = frame.loc[mask, ["Country", "Year", "rd_expenditure"]].copy()
    annual["pred"] = predictions
    annual = annual.groupby(["Country", "Year"], as_index=False).agg(
        actual=("rd_expenditure", "mean"), pred=("pred", "mean")
    )
    return float(np.mean(np.abs(annual.actual - annual.pred) / annual.actual) * 100)


def ridge_design(prep, raw: np.ndarray, fit_mask: np.ndarray):
    scaler = StandardScaler().fit(raw[fit_mask])
    x = np.column_stack(
        [scaler.transform(raw), prep.months.to_numpy(), prep.countries.to_numpy()]
    )
    return x, scaler


def select_ridge_alpha(prep, raw: np.ndarray) -> tuple[float, pd.DataFrame]:
    train = prep.masks["train"]
    val = prep.masks["val"]
    target = target_transform(prep.frame, train)
    x, _ = ridge_design(prep, raw, train)
    rows = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(x[train], target.ystd[train])
        z = model.predict(x[val])
        prediction = np.exp(
            z * target.country_std_vec[val] + target.country_mean_vec[val]
        )
        score = annual_mape(prep.frame, prediction, val)
        rows.append({"alpha": alpha, "validation_MAPE": score})
    table = pd.DataFrame(rows)
    best = float(table.loc[table.validation_MAPE.idxmin(), "alpha"])
    return best, table


def fit_matched_ridge(prep, raw: np.ndarray, fit_mask: np.ndarray, target: TargetTransform):
    alpha, tuning = select_ridge_alpha(prep, raw)
    x, scaler = ridge_design(prep, raw, fit_mask)
    model = Ridge(alpha=alpha).fit(x[fit_mask], target.ystd[fit_mask])
    print(f"Ridge matched refit: selected alpha={alpha:g}", flush=True)
    return model, x, scaler, alpha, tuning


def ridge_level_predictions(model, x: np.ndarray, target: TargetTransform) -> np.ndarray:
    z = model.predict(x)
    return np.exp(z * target.country_std_vec + target.country_mean_vec)


def ridge_raw_scale_elasticities(
    model,
    x: np.ndarray,
    raw: np.ndarray,
    scaler: StandardScaler,
    target: TargetTransform,
) -> np.ndarray:
    base = ridge_level_predictions(model, x, target)
    denominator = np.where(np.abs(base) > 1e-12, base, np.nan)
    output = np.zeros((len(x), raw.shape[1]), dtype=float)
    for j in range(raw.shape[1]):
        xp = x.copy()
        xp[:, j] = (raw[:, j] * 1.01 - scaler.mean_[j]) / scaler.scale_[j]
        perturbed = ridge_level_predictions(model, xp, target)
        output[:, j] = ((perturbed - base) / denominator) / 0.01
    return output


def topic_weights(
    frame: pd.DataFrame,
    columns: list[str],
    topics: list[str],
    elasticities: np.ndarray,
    fit_mask: np.ndarray,
) -> dict[str, float]:
    values = pd.DataFrame(elasticities, columns=columns)
    values["Country"] = frame.Country.to_numpy()
    values["fit"] = fit_mask
    by_country = values[values.fit].groupby("Country")[columns].mean()
    result = {}
    for topic in topics:
        lag_columns = [
            f"{topic}_yearly_avg_lag{lag}"
            for lag in (1, 2, 3)
            if f"{topic}_yearly_avg_lag{lag}" in by_country.columns
        ]
        result[topic] = float(np.nanmean(by_country.loc["US", lag_columns]))
    return result


def load_stepb_inputs(prep):
    gt = pd.read_csv(CODE / "data" / "GT" / "trends_data_by_topic_filtered.csv")
    gt["date"] = pd.to_datetime(gt.date)
    gt["Year"] = gt.date.dt.year
    gt["Month"] = gt.date.dt.month

    references = pd.read_csv(CODE / "temporal_disaggregation" / "results" / "combined_estimates.csv")
    references = references[references.Country == "US"][
        [
            "Year",
            "Month",
            "Monthly_RD_Expenditure_Tempdisagg_Sax",
            "Monthly_RD_Expenditure_Tempdisagg_Mosley",
        ]
    ].copy()
    references.rename(
        columns={
            "Monthly_RD_Expenditure_Tempdisagg_Sax": "Sax",
            "Monthly_RD_Expenditure_Tempdisagg_Mosley": "Mosley",
        },
        inplace=True,
    )
    annual_us = prep.frame[prep.frame.Country == "US"].groupby("Year").rd_expenditure.mean()
    return gt, references, annual_us


def disaggregate_us(
    weights: dict[str, float],
    topics: list[str],
    gt: pd.DataFrame,
    annual_us: pd.Series,
) -> pd.DataFrame:
    usable = [
        topic
        for topic in topics
        if f"US_{topic}" in gt.columns and np.isfinite(weights.get(topic, np.nan))
    ]
    frame = gt[["Year", "Month"] + [f"US_{topic}" for topic in usable]].copy()
    frame = frame[frame.Year.isin(annual_us.index)].copy()
    timing = np.zeros(len(frame), dtype=float)
    for topic in usable:
        column = f"US_{topic}"
        denominator = frame.groupby("Year")[column].transform("sum").to_numpy(dtype=float)
        share = np.where(
            denominator > 0,
            frame[column].to_numpy(dtype=float) / np.where(denominator > 0, denominator, 1.0),
            0.0,
        )
        timing += weights[topic] * share
    frame["timing"] = timing
    frame["annual_timing"] = frame.groupby("Year").timing.transform("sum")
    if np.any(np.abs(frame.annual_timing) < 1e-12):
        raise RuntimeError("At least one year has a zero Step B normalization denominator.")
    frame["estimate"] = frame.Year.map(annual_us) * frame.timing / frame.annual_timing
    return frame[["Year", "Month", "estimate"]]


def pearson(a, b) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(a[mask], b[mask])
    return float(r), float(p)


def growth(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.diff(values) / values[:-1]


def agreement(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark in ["Mosley", "Sax", "Ridge"]:
        rows.append(
            {
                "Benchmark": benchmark,
                "N_months": len(monthly),
                "level_correlation": pearson(monthly.NN, monthly[benchmark])[0],
                "level_p_value": pearson(monthly.NN, monthly[benchmark])[1],
                "monthly_growth_correlation": pearson(
                    growth(monthly.NN), growth(monthly[benchmark])
                )[0],
                "monthly_growth_p_value": pearson(
                    growth(monthly.NN), growth(monthly[benchmark])
                )[1],
            }
        )
    return pd.DataFrame(rows)


def quarterly_outputs(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = monthly.copy()
    frame["Quarter"] = ((frame.Month - 1) // 3) + 1
    quarterly = (
        frame.groupby(["Year", "Quarter"], as_index=False)[["NN", "Mosley", "Sax", "Ridge"]]
        .sum()
        .sort_values(["Year", "Quarter"])
        .reset_index(drop=True)
    )
    rows = []
    for benchmark in ["Mosley", "Sax", "Ridge"]:
        rows.append(
            {
                "Benchmark": benchmark,
                "N_quarters": len(quarterly),
                "level_correlation": pearson(quarterly.NN, quarterly[benchmark])[0],
                "level_p_value": pearson(quarterly.NN, quarterly[benchmark])[1],
                "quarterly_growth_correlation": pearson(
                    growth(quarterly.NN), growth(quarterly[benchmark])
                )[0],
                "quarterly_growth_p_value": pearson(
                    growth(quarterly.NN), growth(quarterly[benchmark])
                )[1],
            }
        )
    return quarterly, pd.DataFrame(rows)


def employment_lags(monthly: pd.DataFrame) -> pd.DataFrame:
    employment = pd.read_csv(CODE / "data" / "datausa.io" / "Monthly Employment.csv")
    employment["date"] = pd.to_datetime(employment.Date)
    employment = employment[["date", "NSA Employees"]].rename(columns={"NSA Employees": "employment"})
    frame = monthly.copy()
    frame["date"] = pd.to_datetime(dict(year=frame.Year, month=frame.Month, day=1))
    frame = frame.merge(employment, on="date", how="inner").sort_values("date").reset_index(drop=True)
    rows = []
    for method in ["NN", "Mosley", "Sax", "Ridge"]:
        x = growth(frame[method])
        y = growth(frame.employment)
        for lag in range(-12, 13):
            if lag < 0:
                a, b = x[-lag:], y[:lag]
            elif lag > 0:
                a, b = x[:-lag], y[lag:]
            else:
                a, b = x, y
            r, p = pearson(a, b)
            rows.append(
                {"Method": method, "Lag": lag, "Correlation": r, "P_Value": p, "N": len(a)}
            )
    return pd.DataFrame(rows)


def annual_sum_check(monthly: pd.DataFrame, annual_us: pd.Series) -> pd.DataFrame:
    sums = monthly.groupby("Year", as_index=False)[["NN", "Mosley", "Sax", "Ridge"]].sum()
    sums["annual_GERD"] = sums.Year.map(annual_us)
    for column in ["NN", "Mosley", "Sax", "Ridge"]:
        sums[f"{column}_difference"] = sums[column] - sums.annual_GERD
    return sums


def transformed_weights(weights: dict[str, float], transform: str) -> dict[str, float]:
    if transform == "signed":
        return dict(weights)
    if transform == "positive_part":
        return {topic: max(value, 0.0) for topic, value in weights.items()}
    if transform == "absolute":
        return {topic: abs(value) for topic, value in weights.items()}
    raise ValueError(f"Unknown weight transform: {transform}")


def perturbation_grid(
    strategies: dict[str, dict[str, float]],
    topics: list[str],
    gt: pd.DataFrame,
    annual_us: pd.Series,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare perturbation definitions and signed/nonnegative relevance rules."""
    summary_rows = []
    series_rows = []
    for strategy, base_weights in strategies.items():
        finite_base = np.asarray(
            [value for value in base_weights.values() if np.isfinite(value)], dtype=float
        )
        transforms = (
            ("signed",)
            if strategy.endswith("_mean_abs_local")
            else ("signed", "positive_part", "absolute")
        )
        for transform in transforms:
            weights = transformed_weights(base_weights, transform)
            monthly = disaggregate_us(weights, topics, gt, annual_us).rename(
                columns={"estimate": "NN"}
            )
            monthly = (
                monthly.merge(references, on=["Year", "Month"], how="inner")
                .dropna()
                .sort_values(["Year", "Month"])
                .reset_index(drop=True)
            )
            quarterly = monthly.copy()
            quarterly["Quarter"] = ((quarterly.Month - 1) // 3) + 1
            quarterly = (
                quarterly.groupby(["Year", "Quarter"], as_index=False)[
                    ["NN", "Mosley", "Sax"]
                ]
                .sum()
                .sort_values(["Year", "Quarter"])
                .reset_index(drop=True)
            )
            for benchmark in ("Mosley", "Sax"):
                summary_rows.append(
                    {
                        "strategy": strategy,
                        "weight_transform": transform,
                        "benchmark": benchmark,
                        "level_correlation": pearson(monthly.NN, monthly[benchmark])[0],
                        "monthly_growth_correlation": pearson(
                            growth(monthly.NN), growth(monthly[benchmark])
                        )[0],
                        "quarterly_growth_correlation": pearson(
                            growth(quarterly.NN), growth(quarterly[benchmark])
                        )[0],
                        "estimate_min": float(monthly.NN.min()),
                        "estimate_max": float(monthly.NN.max()),
                        "monthly_growth_sd": float(np.nanstd(growth(monthly.NN))),
                        "negative_months": int((monthly.NN < 0).sum()),
                        "base_weight_sum": float(np.nansum(finite_base)),
                        "base_negative_share": float(np.mean(finite_base < 0)),
                        "transformed_weight_sum": float(
                            np.nansum(list(weights.values()))
                        ),
                    }
                )
            for row in monthly.itertuples(index=False):
                series_rows.append(
                    {
                        "strategy": strategy,
                        "weight_transform": transform,
                        "Year": row.Year,
                        "Month": row.Month,
                        "estimate": row.NN,
                        "Mosley": row.Mosley,
                        "Sax": row.Sax,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(series_rows)


def render_panels(monthly: pd.DataFrame) -> None:
    frame = monthly.copy()
    frame["date"] = pd.to_datetime(dict(year=frame.Year, month=frame.Month, day=1))
    lower = frame[["NN", "Sax", "Mosley"]].min(axis=1)
    upper = frame[["NN", "Sax", "Mosley"]].max(axis=1)
    panels = [
        (
            "NN",
            "#2c6fbb",
            "NN-driven elasticity-based estimate (AGT)",
            "NN-elasticity",
            "c1_disagg_nn_matched_refit.png",
        ),
        (
            "Mosley",
            "green",
            "Sparse temporal disaggregation (Mosley)",
            "Sparse (Mosley)",
            "c1_disagg_mosley_matched_refit.png",
        ),
        (
            "Sax",
            "purple",
            "Chow-Lin temporal disaggregation (Sax)",
            "Chow-Lin (Sax)",
            "c1_disagg_sax_matched_refit.png",
        ),
    ]
    for column, color, title, label, filename in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(frame.date, lower, upper, color="gray", alpha=0.30, label="Range across methods")
        ax.plot(frame.date, frame[column], color=color, marker="o", ms=3, lw=1, label=label)
        ax.set_ylabel("Monthly R&D (USD bn)")
        ax.set_title(title)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(DECK_ASSETS / filename, dpi=220)
        plt.close(fig)
        print(DECK_ASSETS / filename, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DECK_ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    prep = prepare_data()
    fit_mask = prep.masks["train"] | prep.masks["val"]
    target = target_transform(prep.frame, fit_mask)
    models, raw, scaler, x, country_codes, columns, training = train_matched_agt(
        prep, fit_mask, target
    )
    training.to_csv(OUT / "matched_refit_agt_training_history.csv", index=False)

    topics = topics_from_columns(columns)
    nn_elasticities = nn_raw_scale_elasticities(
        models, x, raw, scaler, country_codes, target
    )
    nn_weights = topic_weights(
        prep.frame, columns, topics, nn_elasticities, fit_mask
    )
    nn_weight_frame = pd.DataFrame(
        {"topic": list(nn_weights), "elasticity": list(nn_weights.values())}
    )
    nn_weight_frame["abs_elasticity"] = nn_weight_frame.elasticity.abs()
    nn_weight_frame.sort_values("abs_elasticity", ascending=False).to_csv(
        OUT / "matched_refit_agt_topic_elasticities.csv", index=False
    )

    standardized_sensitivities = nn_standardized_input_sensitivities(
        models, x, country_codes, target, len(columns)
    )
    standardized_weights = topic_weights(
        prep.frame, columns, topics, standardized_sensitivities, fit_mask
    )
    standardized_weight_frame = pd.DataFrame(
        {"topic": list(standardized_weights), "sensitivity": list(standardized_weights.values())}
    )
    standardized_weight_frame["abs_sensitivity"] = standardized_weight_frame.sensitivity.abs()
    standardized_weight_frame.sort_values("abs_sensitivity", ascending=False).to_csv(
        OUT / "matched_refit_agt_standardized_input_sensitivities.csv", index=False
    )

    additive_standardized_sensitivities = nn_additive_standardized_sensitivities(
        models, x, country_codes, target, len(columns)
    )
    additive_standardized_weights = topic_weights(
        prep.frame,
        columns,
        topics,
        additive_standardized_sensitivities,
        fit_mask,
    )
    additive_standardized_weight_frame = pd.DataFrame(
        {
            "topic": list(additive_standardized_weights),
            "sensitivity": list(additive_standardized_weights.values()),
        }
    )
    additive_standardized_weight_frame["abs_sensitivity"] = (
        additive_standardized_weight_frame.sensitivity.abs()
    )
    additive_standardized_weight_frame.sort_values(
        "abs_sensitivity", ascending=False
    ).to_csv(
        OUT / "matched_refit_agt_additive_standardized_sensitivities.csv",
        index=False,
    )

    raw_plus_one_sensitivities = nn_raw_plus_one_sensitivities(
        models, x, raw, scaler, country_codes, target
    )
    raw_plus_one_weights = topic_weights(
        prep.frame, columns, topics, raw_plus_one_sensitivities, fit_mask
    )
    raw_plus_one_weight_frame = pd.DataFrame(
        {
            "topic": list(raw_plus_one_weights),
            "sensitivity": list(raw_plus_one_weights.values()),
        }
    )
    raw_plus_one_weight_frame["abs_sensitivity"] = (
        raw_plus_one_weight_frame.sensitivity.abs()
    )
    raw_plus_one_weight_frame.sort_values("abs_sensitivity", ascending=False).to_csv(
        OUT / "matched_refit_agt_raw_plus_one_sensitivities.csv", index=False
    )

    central_standardized_sensitivities = nn_central_standardized_sensitivities(
        models, x, country_codes, target, len(columns)
    )
    central_standardized_weights = topic_weights(
        prep.frame,
        columns,
        topics,
        central_standardized_sensitivities,
        fit_mask,
    )
    central_standardized_mean_abs_weights = topic_weights(
        prep.frame,
        columns,
        topics,
        np.abs(central_standardized_sensitivities),
        fit_mask,
    )
    pd.DataFrame(
        {
            "topic": list(central_standardized_weights),
            "signed_mean_sensitivity": list(central_standardized_weights.values()),
            "mean_absolute_local_sensitivity": [
                central_standardized_mean_abs_weights[topic]
                for topic in central_standardized_weights
            ],
        }
    ).sort_values("mean_absolute_local_sensitivity", ascending=False).to_csv(
        OUT / "matched_refit_agt_central_standardized_sensitivities.csv",
        index=False,
    )

    raw_mean_abs_weights = topic_weights(
        prep.frame, columns, topics, np.abs(nn_elasticities), fit_mask
    )
    legacy_mean_abs_weights = topic_weights(
        prep.frame, columns, topics, np.abs(standardized_sensitivities), fit_mask
    )
    additive_standardized_mean_abs_weights = topic_weights(
        prep.frame,
        columns,
        topics,
        np.abs(additive_standardized_sensitivities),
        fit_mask,
    )
    raw_plus_one_mean_abs_weights = topic_weights(
        prep.frame, columns, topics, np.abs(raw_plus_one_sensitivities), fit_mask
    )

    ridge, ridge_x, ridge_scaler, ridge_alpha, ridge_tuning = fit_matched_ridge(
        prep, raw, fit_mask, target
    )
    ridge_tuning.to_csv(OUT / "matched_refit_ridge_tuning.csv", index=False)
    ridge_elasticities = ridge_raw_scale_elasticities(
        ridge, ridge_x, raw, ridge_scaler, target
    )
    ridge_weights = topic_weights(
        prep.frame, columns, topics, ridge_elasticities, fit_mask
    )
    ridge_weight_frame = pd.DataFrame(
        {"topic": list(ridge_weights), "elasticity": list(ridge_weights.values())}
    )
    ridge_weight_frame["abs_elasticity"] = ridge_weight_frame.elasticity.abs()
    ridge_weight_frame.sort_values("abs_elasticity", ascending=False).to_csv(
        OUT / "matched_refit_ridge_topic_elasticities.csv", index=False
    )

    gt, references, annual_us = load_stepb_inputs(prep)
    grid_summary, grid_series = perturbation_grid(
        {
            "raw_multiplicative_1pct": nn_weights,
            "legacy_standardized_multiplicative_1pct": standardized_weights,
            "additive_standardized_0.01sd": additive_standardized_weights,
            "raw_additive_one_gt_point": raw_plus_one_weights,
            "raw_multiplicative_1pct_mean_abs_local": raw_mean_abs_weights,
            "legacy_standardized_multiplicative_1pct_mean_abs_local": legacy_mean_abs_weights,
            "additive_standardized_0.01sd_mean_abs_local": additive_standardized_mean_abs_weights,
            "raw_additive_one_gt_point_mean_abs_local": raw_plus_one_mean_abs_weights,
            "central_standardized_0.10sd": central_standardized_weights,
            "central_standardized_0.10sd_mean_abs_local": central_standardized_mean_abs_weights,
        },
        topics,
        gt,
        annual_us,
        references,
    )
    grid_summary.to_csv(OUT / "stepB_perturbation_grid_summary.csv", index=False)
    grid_series.to_csv(OUT / "stepB_perturbation_grid_monthly_series.csv", index=False)
    nn_monthly = disaggregate_us(nn_weights, topics, gt, annual_us).rename(
        columns={"estimate": "NN"}
    )
    standardized_monthly = disaggregate_us(
        standardized_weights, topics, gt, annual_us
    ).rename(columns={"estimate": "NN_standardized_input"})
    ridge_monthly = disaggregate_us(ridge_weights, topics, gt, annual_us).rename(
        columns={"estimate": "Ridge"}
    )
    monthly = (
        nn_monthly.merge(ridge_monthly, on=["Year", "Month"])
        .merge(references, on=["Year", "Month"], how="inner")
        .dropna()
        .sort_values(["Year", "Month"])
        .reset_index(drop=True)
    )
    monthly.to_csv(OUT / "matched_refit_stepB_monthly_estimates.csv", index=False)

    standardized_comparison = (
        standardized_monthly.merge(references, on=["Year", "Month"], how="inner")
        .merge(nn_monthly, on=["Year", "Month"], how="inner")
        .dropna()
        .sort_values(["Year", "Month"])
        .reset_index(drop=True)
    )
    standardized_comparison.to_csv(
        OUT / "matched_refit_stepB_standardized_input_monthly_estimates.csv", index=False
    )
    standardized_rows = []
    for benchmark in ["Mosley", "Sax", "NN"]:
        standardized_rows.append(
            {
                "Benchmark": benchmark,
                "level_correlation": pearson(
                    standardized_comparison.NN_standardized_input,
                    standardized_comparison[benchmark],
                )[0],
                "monthly_growth_correlation": pearson(
                    growth(standardized_comparison.NN_standardized_input),
                    growth(standardized_comparison[benchmark]),
                )[0],
            }
        )
    standardized_agreement = pd.DataFrame(standardized_rows)
    standardized_agreement.to_csv(
        OUT / "matched_refit_stepB_standardized_input_agreement.csv", index=False
    )

    monthly_agreement = agreement(monthly)
    monthly_agreement["ridge_alpha"] = ridge_alpha
    monthly_agreement.to_csv(OUT / "matched_refit_stepB_monthly_agreement.csv", index=False)
    quarterly, quarterly_agreement = quarterly_outputs(monthly)
    quarterly.to_csv(OUT / "matched_refit_stepB_quarterly_estimates.csv", index=False)
    quarterly_agreement.to_csv(OUT / "matched_refit_stepB_quarterly_agreement.csv", index=False)
    employment_lags(monthly).to_csv(OUT / "matched_refit_stepB_employment_lags.csv", index=False)
    sums = annual_sum_check(monthly, annual_us)
    sums.to_csv(OUT / "matched_refit_stepB_annual_sum_check.csv", index=False)

    render_panels(monthly)

    print("\n=== matched-refit Step B monthly agreement ===")
    print(monthly_agreement.to_string(index=False))
    print("\n=== matched-refit Step B quarterly agreement ===")
    print(quarterly_agreement.to_string(index=False))
    print("\n=== leading AGT topic elasticities ===")
    print(nn_weight_frame.nlargest(10, "abs_elasticity").to_string(index=False))
    print("\n=== standardized-input diagnostic agreement ===")
    print(standardized_agreement.to_string(index=False))
    print("\n=== perturbation and relevance-weight grid ===")
    print(grid_summary.to_string(index=False))
    print("\n=== annual adding-up maximum absolute errors ===")
    print(
        sums[[c for c in sums.columns if c.endswith("_difference")]]
        .abs()
        .max()
        .to_string()
    )
    print(f"\nOutputs saved under {OUT}")


if __name__ == "__main__":
    main()
