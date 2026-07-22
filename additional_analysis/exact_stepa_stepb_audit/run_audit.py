"""Audit Step B using the exact reported Step A AGT neural network.

This script is intentionally additive. It retrains the published temporal AGT
ensemble (within-country log-standardized target), computes finite-difference
sensitivities on both the raw and standardized predictor scales, constructs the
two US monthly allocations, and writes comparison diagnostics. It also derives
an updated topic-level SHAP ranking for the Chow-Lin indicator refresh.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "out" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_OUT = ROOT / "additional_analysis" / "out"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "additional_analysis" / "robustness_overfit"))
from common import annual_metrics, prepare_data  # noqa: E402


HIDDEN = (64, 16)
DROPOUT = 0.12
WEIGHT_DECAY = 5e-4
LR = 3e-3
ENSEMBLE_SIZE = 15
EPSILON = 0.01


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

    def forward(self, x: torch.Tensor, country: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, self.emb(country)], dim=1)
        for layer, norm in zip(self.layers, self.norms):
            z = self.drop(self.act(norm(layer(z))))
        return self.out(z) + self.skip(x)


def tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


def topic_from_feature(feature: str) -> str:
    for lag in (1, 2, 3):
        suffix = f"_yearly_avg_lag{lag}"
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature


def level_predictions(prep, z: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    if rows is None:
        means = prep.country_mean_vec
        stds = prep.country_std_vec
    else:
        means = prep.country_mean_vec[rows]
        stds = prep.country_std_vec[rows]
    return np.exp(z * stds + means)


def train_exact_agt():
    prep = prepare_data()
    frame = prep.frame
    agt_cols = prep.configs["AGT"]
    raw = frame[agt_cols].fillna(0).astype(float).to_numpy()
    scaler = StandardScaler().fit(raw[prep.masks["train"]])
    standardized = scaler.transform(raw)
    x = np.column_stack([standardized, prep.months.to_numpy()])
    encoder = LabelEncoder()
    countries = encoder.fit_transform(frame.Country)

    train = prep.masks["train"]
    val = prep.masks["val"]
    xtr, xva = tensor(x[train]), tensor(x[val])
    ctr = torch.as_tensor(countries[train], dtype=torch.long)
    cva = torch.as_tensor(countries[val], dtype=torch.long)
    ytr = tensor(prep.ystd[train].reshape(-1, 1))
    criterion = nn.SmoothL1Loss(beta=0.5)
    models = []
    history = []

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
            permutation = torch.randperm(len(xtr))
            for start in range(0, len(xtr), 64):
                idx = permutation[start : start + 64]
                optimizer.zero_grad()
                loss = criterion(model(xtr[idx], ctr[idx]), ytr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                zval = model(xva, cva).numpy().ravel()
            pred = level_predictions(prep, zval, val)
            score = annual_metrics(frame[val].assign(pred=pred), "pred")["MAPE"]
            history.append({"seed": seed, "epoch": epoch + 1, "val_MAPE": score})
            if score < best - 1e-5:
                best = score
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 65:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        print(f"AGT seed {seed + 1}/{ENSEMBLE_SIZE}: best validation MAPE={best:.4f}", flush=True)

    pd.DataFrame(history).to_csv(OUT / "exact_stepa_training_history.csv", index=False)
    test = prep.masks["test"]
    with torch.no_grad():
        member_z = np.column_stack(
            [model(tensor(x[test]), torch.as_tensor(countries[test], dtype=torch.long)).numpy().ravel() for model in models]
        )
    member_level = np.column_stack([level_predictions(prep, member_z[:, j], test) for j in range(ENSEMBLE_SIZE)])
    test_frame = frame[test][["Country", "Year", "Month", "rd_expenditure"]].copy()
    test_frame["prediction"] = member_level.mean(axis=1)
    metrics = annual_metrics(test_frame, "prediction")
    pd.DataFrame([metrics]).to_csv(OUT / "exact_stepa_agt_metrics.csv", index=False)
    test_frame.to_csv(OUT / "exact_stepa_agt_predictions.csv", index=False)
    print("Exact Step A AGT metrics:", metrics, flush=True)
    return prep, agt_cols, raw, standardized, scaler, x, countries, encoder, models


def finite_difference_sensitivities(
    prep,
    agt_cols,
    raw,
    standardized,
    scaler,
    x,
    countries,
    models,
    definition: str,
) -> np.ndarray:
    n_gt = len(agt_cols)
    country_tensor = torch.as_tensor(countries, dtype=torch.long)
    baseline_x = tensor(x)
    output = np.zeros((len(x), n_gt), dtype=float)
    for member, model in enumerate(models):
        with torch.no_grad():
            base_z = model(baseline_x, country_tensor).numpy().ravel()
        base_level = level_predictions(prep, base_z)
        denominator = np.where(np.abs(base_level) > 1e-12, base_level, np.nan)
        for j in range(n_gt):
            perturbed_x = x.copy()
            if definition == "raw_scale_elasticity":
                perturbed_raw = raw[:, j] * (1.0 + EPSILON)
                perturbed_x[:, j] = (perturbed_raw - scaler.mean_[j]) / scaler.scale_[j]
            elif definition == "standardized_input_sensitivity":
                perturbed_x[:, j] = standardized[:, j] * (1.0 + EPSILON)
            else:
                raise ValueError(definition)
            with torch.no_grad():
                perturbed_z = model(tensor(perturbed_x), country_tensor).numpy().ravel()
            perturbed_level = level_predictions(prep, perturbed_z)
            output[:, j] += ((perturbed_level - base_level) / denominator) / EPSILON
        print(f"{definition}: member {member + 1}/{len(models)}", flush=True)
    return output / len(models)


def us_topic_weights(prep, agt_cols, sensitivities):
    values = pd.DataFrame(sensitivities, columns=agt_cols)
    values["Country"] = prep.frame.Country.to_numpy()
    values["train"] = prep.masks["train"]
    country_feature = values[values.train].groupby("Country")[agt_cols].mean()
    rows = []
    for topic in sorted({topic_from_feature(col) for col in agt_cols}):
        lag_cols = [col for col in agt_cols if topic_from_feature(col) == topic]
        rows.append(
            {
                "topic": topic,
                "weight": float(country_feature.loc["US", lag_cols].mean()),
                "lag_count": len(lag_cols),
            }
        )
    return pd.DataFrame(rows)


def load_references(prep):
    trends = pd.read_csv(ROOT / "data" / "GT" / "trends_data_by_topic_filtered.csv")
    trends["date"] = pd.to_datetime(trends.date)
    trends["Year"] = trends.date.dt.year
    trends["Month"] = trends.date.dt.month
    annual = prep.frame[prep.frame.Country == "US"].groupby("Year").rd_expenditure.mean()
    refs = pd.read_csv(ROOT / "temporal_disaggregation" / "results" / "combined_estimates.csv")
    refs = refs[refs.Country == "US"][[
        "Year",
        "Month",
        "Monthly_RD_Expenditure_Tempdisagg_Sax",
        "Monthly_RD_Expenditure_Tempdisagg_Mosley",
    ]].rename(
        columns={
            "Monthly_RD_Expenditure_Tempdisagg_Sax": "Chow_Lin_old",
            "Monthly_RD_Expenditure_Tempdisagg_Mosley": "Sparse",
        }
    )
    return trends, annual, refs


def allocate(weights, trends, annual, refs, label):
    topics = [row.topic for row in weights.itertuples() if f"US_{row.topic}" in trends.columns]
    selected = trends[["Year", "Month"] + [f"US_{topic}" for topic in topics]].copy()
    selected = selected[selected.Year.isin(annual.index)].copy()
    timing = np.zeros(len(selected), dtype=float)
    weight_map = weights.set_index("topic").weight.to_dict()
    for topic in topics:
        values = selected[f"US_{topic}"].to_numpy(dtype=float)
        totals = selected.groupby("Year")[f"US_{topic}"].transform("sum").to_numpy(dtype=float)
        shares = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
        timing += weight_map[topic] * shares
    selected["timing_score"] = timing
    selected["annual_timing_score"] = selected.groupby("Year").timing_score.transform("sum")
    selected[label] = selected.Year.map(annual) * selected.timing_score / selected.annual_timing_score
    result = selected[["Year", "Month", label]].merge(refs, on=["Year", "Month"], how="inner")
    return result


def corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan, np.nan
    return stats.pearsonr(np.asarray(a)[mask], np.asarray(b)[mask])


def agreement_rows(data, estimate_columns):
    rows = []
    for estimate in estimate_columns:
        for reference in ["Sparse", "Chow_Lin_old"]:
            level_r, level_p = corr(data[estimate].to_numpy(), data[reference].to_numpy())
            estimate_growth = data[estimate].pct_change().to_numpy()[1:]
            reference_growth = data[reference].pct_change().to_numpy()[1:]
            growth_r, growth_p = corr(estimate_growth, reference_growth)
            rows.append(
                {
                    "estimate": estimate,
                    "reference": reference,
                    "level_r": level_r,
                    "level_p": level_p,
                    "growth_r": growth_r,
                    "growth_p": growth_p,
                    "n_level": len(data),
                    "n_growth": len(data) - 1,
                }
            )
    raw_vs_standard_r, raw_vs_standard_p = corr(data.Raw_scale.to_numpy(), data.Standardized_input.to_numpy())
    raw_growth_r, raw_growth_p = corr(
        data.Raw_scale.pct_change().to_numpy()[1:], data.Standardized_input.pct_change().to_numpy()[1:]
    )
    rows.append(
        {
            "estimate": "Raw_scale",
            "reference": "Standardized_input",
            "level_r": raw_vs_standard_r,
            "level_p": raw_vs_standard_p,
            "growth_r": raw_growth_r,
            "growth_p": raw_growth_p,
            "n_level": len(data),
            "n_growth": len(data) - 1,
        }
    )
    return pd.DataFrame(rows)


def current_topic_shap(prep, agt_cols, x, countries, encoder, models):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required for the current indicator ranking") from exc

    feature_names = agt_cols + list(prep.months.columns)
    all_values = []
    all_features = []
    for country in encoder.classes_:
        code = int(encoder.transform([country])[0])
        train_rows = prep.masks["train"] & (prep.frame.Country.to_numpy() == country)
        test_rows = prep.masks["test"] & (prep.frame.Country.to_numpy() == country)
        if not test_rows.any() or train_rows.sum() < 3:
            continue
        background = shap.kmeans(x[train_rows], min(5, int(train_rows.sum())))
        indices = np.where(test_rows)[0][:5]
        explained = x[indices]

        def prediction_function(values):
            values_tensor = tensor(np.asarray(values))
            country_tensor = torch.full((len(values),), code, dtype=torch.long)
            with torch.no_grad():
                predictions = torch.stack([model(values_tensor, country_tensor) for model in models]).mean(0)
            return predictions.numpy().ravel()

        explainer = shap.KernelExplainer(prediction_function, background)
        shap_values = np.asarray(explainer.shap_values(explained, nsamples=200, silent=True))
        all_values.append(shap_values)
        all_features.append(explained)
        print(f"SHAP: {country}", flush=True)

    shap_values = np.vstack(all_values)
    explained_features = np.vstack(all_features)
    feature_ranking = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    feature_ranking.to_csv(OUT / "current_agt_shap_feature_ranking.csv", index=False)
    topic_ranking = (
        feature_ranking[feature_ranking.feature.isin(agt_cols)]
        .assign(topic=lambda data: data.feature.map(topic_from_feature))
        .groupby("topic", as_index=False)
        .agg(mean_abs_shap_sum=("mean_abs_shap", "sum"), mean_abs_shap_mean=("mean_abs_shap", "mean"))
        .sort_values("mean_abs_shap_sum", ascending=False)
    )
    topic_ranking.to_csv(OUT / "current_agt_shap_topic_ranking.csv", index=False)
    trends = pd.read_csv(ROOT / "data" / "GT" / "trends_data_by_topic_filtered.csv")
    trends["Year"] = pd.to_datetime(trends.date).dt.year
    usable = []
    for topic in topic_ranking.topic:
        column = f"US_{topic}"
        period = trends.loc[trends.Year.between(2004, 2021), column] if column in trends else pd.Series(dtype=float)
        if len(period) and period.std() > 0:
            usable.append(topic)
    top_six = usable[:6]
    (OUT / "current_chowlin_top6.json").write_text(json.dumps(top_six, indent=2) + "\n")

    plt.figure(figsize=(8, 5))
    top = topic_ranking.head(15).sort_values("mean_abs_shap_sum")
    plt.barh(top.topic, top.mean_abs_shap_sum, color="#4d6f8c")
    plt.xlabel("Sum of mean absolute SHAP values across three lags")
    plt.tight_layout()
    plt.savefig(OUT / "current_agt_topic_shap_ranking.png", dpi=200)
    plt.close()
    return top_six


def export_chowlin_inputs(prep, trends, annual, top_six):
    old_six = ["Capitalization", "Investment_management", "Patent_office", "Tax_credit", "Cost", "Technology"]
    topics = list(dict.fromkeys(old_six + top_six))
    data = trends[["Year", "Month"] + [f"US_{topic}" for topic in topics]].copy()
    data = data[data.Year.isin(annual.index)].copy()
    data["annual_gerd"] = data.Year.map(annual)
    data.rename(columns={f"US_{topic}": topic for topic in topics}, inplace=True)
    data.to_csv(OUT / "chowlin_inputs.csv", index=False)
    pd.DataFrame({"set": ["old"] * 6 + ["current"] * 6, "topic": old_six + top_six}).to_csv(
        OUT / "chowlin_indicator_sets.csv", index=False
    )


def main():
    prep, agt_cols, raw, standardized, scaler, x, countries, encoder, models = train_exact_agt()
    trends, annual, refs = load_references(prep)
    allocations = []
    for definition, label in [
        ("raw_scale_elasticity", "Raw_scale"),
        ("standardized_input_sensitivity", "Standardized_input"),
    ]:
        sensitivities = finite_difference_sensitivities(
            prep, agt_cols, raw, standardized, scaler, x, countries, models, definition
        )
        weights = us_topic_weights(prep, agt_cols, sensitivities)
        weights.to_csv(OUT / f"{definition}_us_topic_weights.csv", index=False)
        allocations.append(allocate(weights, trends, annual, refs, label))
    combined = allocations[0].merge(
        allocations[1][["Year", "Month", "Standardized_input"]], on=["Year", "Month"], how="inner"
    )
    combined.to_csv(OUT / "exact_stepa_stepb_monthly_allocations.csv", index=False)
    agreement = agreement_rows(combined, ["Raw_scale", "Standardized_input"])
    agreement.to_csv(OUT / "exact_stepa_stepb_agreement.csv", index=False)

    adding_up = []
    for label in ["Raw_scale", "Standardized_input"]:
        annualized = combined.groupby("Year")[label].sum()
        for year in annualized.index:
            adding_up.append(
                {"definition": label, "Year": year, "difference": annualized.loc[year] - annual.loc[year]}
            )
    pd.DataFrame(adding_up).to_csv(OUT / "annual_adding_up_check.csv", index=False)

    top_six = current_topic_shap(prep, agt_cols, x, countries, encoder, models)
    export_chowlin_inputs(prep, trends, annual, top_six)
    print("\nSensitivity agreement:\n", agreement.to_string(index=False), flush=True)
    print("\nCurrent topic-level SHAP top six:", top_six, flush=True)
    print(f"\nSaved audit outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
