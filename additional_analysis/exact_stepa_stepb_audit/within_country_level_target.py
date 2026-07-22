"""Evaluate a within-country level-standardized AGT target in Steps A and B.

This additive experiment holds the temporal split, architecture, optimization,
predictors, ensemble, and Step B allocation fixed. Only the Step A target
transformation changes from within-country log-standardized GERD to
within-country level-standardized GERD.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "additional_analysis" / "robustness_overfit"))
sys.path.insert(0, str(HERE))

from common import annual_metrics, prepare_data  # noqa: E402
from run_audit import (  # noqa: E402
    DROPOUT,
    ENSEMBLE_SIZE,
    HIDDEN,
    LR,
    WEIGHT_DECAY,
    WideDeepNN,
    allocate,
    agreement_rows,
    load_references,
    tensor,
    us_topic_weights,
)


EPSILON = 0.01


def country_level_target(prep):
    frame = prep.frame
    target = frame.rd_expenditure.to_numpy(dtype=float)
    means = {}
    standard_deviations = {}
    for country, group in frame.groupby("Country"):
        values = group.loc[group.split.eq("train"), "rd_expenditure"].to_numpy(dtype=float)
        means[country] = float(values.mean())
        standard_deviations[country] = float(max(values.std(), 1e-6))
    mean_vector = frame.Country.map(means).to_numpy(dtype=float)
    sd_vector = frame.Country.map(standard_deviations).to_numpy(dtype=float)
    transformed = (target - mean_vector) / sd_vector
    return transformed, mean_vector, sd_vector, means, standard_deviations


def to_level(z, mean_vector, sd_vector, rows=None):
    if rows is None:
        return z * sd_vector + mean_vector
    return z * sd_vector[rows] + mean_vector[rows]


def train_models():
    prep = prepare_data()
    frame = prep.frame
    agt_columns = prep.configs["AGT"]
    train, validation, test = (prep.masks[name] for name in ("train", "val", "test"))
    raw = frame[agt_columns].fillna(0).astype(float).to_numpy()
    scaler = StandardScaler().fit(raw[train])
    standardized = scaler.transform(raw)
    x = np.column_stack([standardized, prep.months.to_numpy()])
    encoder = LabelEncoder()
    countries = encoder.fit_transform(frame.Country)
    target, target_mean, target_sd, means, standard_deviations = country_level_target(prep)

    xtrain, xvalidation = tensor(x[train]), tensor(x[validation])
    ctrain = torch.as_tensor(countries[train], dtype=torch.long)
    cvalidation = torch.as_tensor(countries[validation], dtype=torch.long)
    ytrain = tensor(target[train].reshape(-1, 1))
    criterion = nn.SmoothL1Loss(beta=0.5)
    models = []
    histories = []

    for seed in range(ENSEMBLE_SIZE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = WideDeepNN(x.shape[1], len(encoder.classes_))
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best = np.inf
        best_state = None
        bad = 0
        for epoch in range(500):
            model.train()
            permutation = torch.randperm(len(xtrain))
            for start in range(0, len(xtrain), 64):
                index = permutation[start : start + 64]
                optimizer.zero_grad()
                loss = criterion(model(xtrain[index], ctrain[index]), ytrain[index])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                zvalidation = model(xvalidation, cvalidation).numpy().ravel()
            prediction = to_level(zvalidation, target_mean, target_sd, validation)
            score = annual_metrics(frame[validation].assign(pred=prediction), "pred")["MAPE"]
            histories.append({"seed": seed, "epoch": epoch + 1, "val_MAPE": score})
            if score < best - 1e-5:
                best = score
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 65:
                    break
        model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        print(f"Country-level seed {seed + 1}/{ENSEMBLE_SIZE}: best validation MAPE={best:.4f}", flush=True)

    with torch.no_grad():
        member_z = np.column_stack(
            [
                model(tensor(x[test]), torch.as_tensor(countries[test], dtype=torch.long)).numpy().ravel()
                for model in models
            ]
        )
    member_levels = np.column_stack(
        [to_level(member_z[:, member], target_mean, target_sd, test) for member in range(ENSEMBLE_SIZE)]
    )
    predictions = frame[test][["Country", "Year", "Month", "rd_expenditure"]].copy()
    predictions["prediction"] = member_levels.mean(axis=1)
    for member in range(ENSEMBLE_SIZE):
        predictions[f"member_{member}"] = member_levels[:, member]
    metrics = annual_metrics(predictions, "prediction")
    pd.DataFrame(histories).to_csv(OUT / "within_country_level_training_history.csv", index=False)
    predictions.to_csv(OUT / "within_country_level_agt_predictions.csv", index=False)
    pd.DataFrame([{**metrics, "target": "within-country level-standardized GERD"}]).to_csv(
        OUT / "within_country_level_agt_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "Country": list(means),
            "training_level_mean": [means[country] for country in means],
            "training_level_sd": [standard_deviations[country] for country in means],
        }
    ).to_csv(OUT / "within_country_level_target_statistics.csv", index=False)
    print("Within-country level Step A metrics:", metrics, flush=True)
    return prep, agt_columns, raw, standardized, scaler, x, countries, models, target_mean, target_sd


def sensitivities(
    prep,
    agt_columns,
    raw,
    standardized,
    scaler,
    x,
    countries,
    models,
    target_mean,
    target_sd,
    definition,
):
    n_topics = len(agt_columns)
    country_tensor = torch.as_tensor(countries, dtype=torch.long)
    baseline_x = tensor(x)
    output = np.zeros((len(x), n_topics), dtype=float)
    for member, model in enumerate(models):
        with torch.no_grad():
            base_z = model(baseline_x, country_tensor).numpy().ravel()
        base = to_level(base_z, target_mean, target_sd)
        denominator = np.where(np.abs(base) > 1e-12, base, np.nan)
        for feature in range(n_topics):
            perturbed = x.copy()
            if definition == "raw_scale_elasticity":
                changed_raw = raw[:, feature] * (1 + EPSILON)
                perturbed[:, feature] = (changed_raw - scaler.mean_[feature]) / scaler.scale_[feature]
            elif definition == "standardized_input_sensitivity":
                perturbed[:, feature] = standardized[:, feature] * (1 + EPSILON)
            else:
                raise ValueError(definition)
            with torch.no_grad():
                changed_z = model(tensor(perturbed), country_tensor).numpy().ravel()
            changed = to_level(changed_z, target_mean, target_sd)
            output[:, feature] += ((changed - base) / denominator) / EPSILON
        print(f"Within-country level {definition}: member {member + 1}/{len(models)}", flush=True)
    return output / len(models)


def employment_lags(data):
    employment = pd.read_csv(ROOT / "data" / "datausa.io" / "Monthly Employment.csv")
    employment["date"] = pd.to_datetime(employment.Date)
    employment = employment[["date", "NSA Employees"]].rename(columns={"NSA Employees": "employment"})
    frame = data.copy()
    frame["date"] = pd.to_datetime(dict(year=frame.Year, month=frame.Month, day=1))
    frame = frame.merge(employment, on="date", how="inner").sort_values("date")
    rows = []
    for method in ["Raw_scale", "Standardized_input"]:
        rd_growth = frame[method].pct_change().iloc[1:].to_numpy()
        employment_growth = frame.employment.pct_change().iloc[1:].to_numpy()
        for lag in range(-12, 13):
            if lag < 0:
                first, second = rd_growth[-lag:], employment_growth[:lag]
            elif lag > 0:
                first, second = rd_growth[:-lag], employment_growth[lag:]
            else:
                first, second = rd_growth, employment_growth
            correlation, p_value = stats.pearsonr(first, second)
            rows.append(
                {
                    "method": method,
                    "lag": lag,
                    "correlation": correlation,
                    "p_value_unadjusted": p_value,
                    "n": len(first),
                }
            )
    return pd.DataFrame(rows)


def main():
    prep, agt_columns, raw, standardized, scaler, x, countries, models, target_mean, target_sd = train_models()
    trends, annual, references = load_references(prep)
    allocations = []
    for definition, label in [
        ("raw_scale_elasticity", "Raw_scale"),
        ("standardized_input_sensitivity", "Standardized_input"),
    ]:
        sensitivity = sensitivities(
            prep,
            agt_columns,
            raw,
            standardized,
            scaler,
            x,
            countries,
            models,
            target_mean,
            target_sd,
            definition,
        )
        weights = us_topic_weights(prep, agt_columns, sensitivity)
        weights.to_csv(OUT / f"within_country_level_{definition}_us_topic_weights.csv", index=False)
        allocations.append(allocate(weights, trends, annual, references, label))

    combined = allocations[0].merge(
        allocations[1][["Year", "Month", "Standardized_input"]], on=["Year", "Month"], how="inner"
    )
    combined.to_csv(OUT / "within_country_level_stepb_allocations.csv", index=False)
    agreement = agreement_rows(combined, ["Raw_scale", "Standardized_input"])
    agreement.to_csv(OUT / "within_country_level_stepb_agreement.csv", index=False)
    employment = employment_lags(combined)
    employment.to_csv(OUT / "within_country_level_employment_lags.csv", index=False)

    diagnostics = []
    for method in ["Raw_scale", "Standardized_input"]:
        diagnostics.append(
            {
                "method": method,
                "minimum_monthly_estimate": combined[method].min(),
                "maximum_monthly_estimate": combined[method].max(),
                "negative_months": int((combined[method] < 0).sum()),
                "employment_lag0_r": employment.loc[
                    employment.method.eq(method) & employment.lag.eq(0), "correlation"
                ].iloc[0],
                "employment_lags_p_lt_01": int(
                    (employment.loc[employment.method.eq(method), "p_value_unadjusted"] < 0.01).sum()
                ),
            }
        )
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(OUT / "within_country_level_stepb_diagnostics.csv", index=False)
    print("\nWithin-country level Step B agreement:\n", agreement.to_string(index=False), flush=True)
    print("\nWithin-country level diagnostics:\n", diagnostics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
