"""Correct raw-scale elasticities for the Step B linear robustness models."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


CODE = Path(__file__).resolve().parents[2]
SOURCE = CODE / "additional_analysis" / "robustness_overfit" / "04_stepb_model_agnostic_elasticities.py"
OUT = CODE / "additional_analysis" / "stepb_raw_scale_elasticity_refresh" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def load_module():
    spec = importlib.util.spec_from_file_location("linear_stepb", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_scale_elasticities(module, model, X, raw_gt, scaler, y_mean, y_sd):
    base = module.predict_level(model, X, y_mean, y_sd)
    denom = np.where(np.abs(base) > 1e-12, base, np.nan)
    out = np.zeros((len(X), raw_gt.shape[1]), dtype=float)
    for j in range(raw_gt.shape[1]):
        Xp = X.copy()
        Xp[:, j] = (raw_gt[:, j] * 1.01 - scaler.mean_[j]) / scaler.scale_[j]
        perturbed = module.predict_level(model, Xp, y_mean, y_sd)
        out[:, j] = ((perturbed - base) / denom) / 0.01
    return out


def topic_weights(df, agt_cols, topics, elasticities):
    frame = pd.DataFrame(elasticities, columns=agt_cols)
    frame["Country"] = df.Country.values
    frame["split"] = df.split.values
    eta = frame[frame.split == "train"].groupby("Country")[agt_cols].mean()
    result = {}
    for topic in topics:
        lag_cols = [
            f"{topic}_yearly_avg_lag{lag}"
            for lag in (1, 2, 3)
            if f"{topic}_yearly_avg_lag{lag}" in eta.columns
        ]
        result[topic] = float(np.nanmean(eta.loc["US", lag_cols])) if lag_cols else np.nan
    return result


def quarterly_summary(monthly: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.merge(references[["Year", "Month", "NN", "Mosley", "Sax"]], on=["Year", "Month"], how="inner")
    frame["Quarter"] = ((frame.Month - 1) // 3) + 1
    estimates = frame.groupby(["Model", "Variant", "Year", "Quarter"], as_index=False).estimate.sum()
    refs = frame.groupby(["Year", "Quarter"], as_index=False)[["NN", "Mosley", "Sax"]].first()
    estimates = estimates.merge(refs, on=["Year", "Quarter"], how="left")
    rows = []
    for (model, variant), group in estimates.groupby(["Model", "Variant"]):
        group = group.sort_values(["Year", "Quarter"])
        row = {"Model": model, "Variant": variant, "N_quarters": len(group)}
        for benchmark in ["NN", "Mosley", "Sax"]:
            row[f"{benchmark}_level_corr"] = module.pearson(group.estimate.values, group[benchmark].values)
            row[f"{benchmark}_growth_corr"] = module.pearson(
                module.growth(group.estimate.values), module.growth(group[benchmark].values)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    global module
    module = load_module()
    df = module.chronological_split(module.load_features())
    agt_cols = [col for col in df.columns if "_yearly_avg_lag" in col]
    topics = module.agt_topics(agt_cols)
    train = (df.split == "train").values
    fit = df.split.isin(["train", "val"]).values

    raw_gt = df[agt_cols].fillna(0).astype(float).values
    scaler = StandardScaler().fit(raw_gt[train])
    X_gt = scaler.transform(raw_gt)
    months = pd.get_dummies(df.Month.astype(int), prefix="M").astype(float)
    countries = pd.get_dummies(df.Country, prefix="c").astype(float)
    X = np.column_stack([X_gt, months.values, countries.values])

    y_level = df.rd_expenditure.values.astype(float)
    y_mean = float(y_level[train].mean())
    y_sd = float(y_level[train].std())
    y = (y_level - y_mean) / y_sd

    ols = LinearRegression().fit(X[fit], y[fit])
    ridge, ridge_tuning, ridge_val = module.tune_ridge(X, y, df, y_mean, y_sd)
    enet, enet_tuning, enet_val = module.tune_elastic_net(X, y, df, y_mean, y_sd)
    models = [
        ("OLS", ols, "", np.nan),
        ("Ridge", ridge, ridge_tuning, ridge_val),
        ("Elastic Net", enet, enet_tuning, enet_val),
    ]

    gt, _, references, employment = module.load_monthly_references()
    corrected_nn = pd.read_csv(OUT / "corrected_temporal_monthly_estimates.csv")[["Year", "Month", "NN"]]
    references = references.drop(columns=["NN"]).merge(corrected_nn, on=["Year", "Month"], how="inner")
    rd_us = df[df.Country == "US"].groupby("Year").rd_expenditure.mean()

    summaries = []
    monthly_outputs = []
    elasticity_outputs = []
    top_outputs = []
    for model_name, model, tuning, val_mape in models:
        elasticities = raw_scale_elasticities(module, model, X, raw_gt, scaler, y_mean, y_sd)
        eta = topic_weights(df, agt_cols, topics, elasticities)
        for topic, value in eta.items():
            elasticity_outputs.append({"Model": model_name, "topic": topic, "elasticity": value})
        ranked = sorted(eta.items(), key=lambda item: -abs(item[1]) if np.isfinite(item[1]) else -np.inf)
        for rank, (topic, value) in enumerate(ranked[:20], 1):
            top_outputs.append(
                {"Model": model_name, "rank": rank, "topic": topic, "elasticity": value, "abs_elasticity": abs(value)}
            )

        records, series = module.disaggregate_us(model_name, eta, topics, gt, rd_us, references, employment)
        for record in records:
            record["tuning"] = tuning
            record["val_MAPE"] = val_mape
            record["negative_topic_elasticity_share"] = float(
                np.mean([value < 0 for value in eta.values() if np.isfinite(value)])
            )
        summaries.extend(records)
        monthly_outputs.append(series)
        print(f"finished corrected {model_name}: {tuning or 'unregularized'}", flush=True)

    summary = pd.DataFrame(summaries)
    monthly = pd.concat(monthly_outputs, ignore_index=True)
    elasticities = pd.DataFrame(elasticity_outputs)
    top = pd.DataFrame(top_outputs)
    summary.to_csv(OUT / "corrected_linear_stepb_summary.csv", index=False)
    monthly.to_csv(OUT / "corrected_linear_stepb_monthly_estimates.csv", index=False)
    elasticities.to_csv(OUT / "corrected_linear_topic_elasticities.csv", index=False)
    top.to_csv(OUT / "corrected_linear_top_topics.csv", index=False)
    quarterly = quarterly_summary(monthly, references)
    quarterly.to_csv(OUT / "corrected_linear_quarterly_summary.csv", index=False)

    print("\n=== corrected linear Step B summary ===")
    print(summary.to_string(index=False))
    print("\n=== corrected signed and positive-part quarterly summary ===")
    print(quarterly[quarterly.Variant.isin(["signed", "positive_part"])].to_string(index=False))


if __name__ == "__main__":
    main()
