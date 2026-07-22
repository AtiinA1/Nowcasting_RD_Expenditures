"""Recompute Step B with finite-difference elasticities on the raw GT scale.

This audit leaves all existing paper-facing files untouched. It reuses the
current level-target AGT architecture and training routine, but perturbs each
raw lag-specific Google Trends feature by one percent before applying the
training-fitted StandardScaler. The resulting quantity is therefore aligned
with the elasticity definition used in the paper.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler


CODE = Path(__file__).resolve().parents[2]
SOURCE = CODE / "additional_analysis" / "robustness_overfit" / "12_refresh_stepB_updated_agt.py"
OUT = CODE / "additional_analysis" / "stepb_raw_scale_elasticity_refresh" / "out"
HISTORY_OUT = OUT / "training_histories"
OLD_OUT = CODE / "additional_analysis" / "robustness_overfit" / "out" / "updated_stepB_agt"

OUT.mkdir(parents=True, exist_ok=True)
HISTORY_OUT.mkdir(parents=True, exist_ok=True)


def load_current_module():
    spec = importlib.util.spec_from_file_location("current_stepb", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROBUST_OUT = HISTORY_OUT
    return module


def raw_scale_elasticities(
    module,
    models,
    X: np.ndarray,
    country_codes: np.ndarray,
    y_mean: float,
    y_sd: float,
    raw_gt: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Perturb raw s by 1%, then transform it with the fitted scaler."""
    n_gt = raw_gt.shape[1]
    countries = torch.LongTensor(country_codes)
    Xall = module.tensor(X)
    elasticities = np.zeros((X.shape[0], n_gt), dtype=float)

    for model in models:
        model.eval()
        with torch.no_grad():
            base = model(Xall, countries).numpy().ravel() * y_sd + y_mean
        output_denom = np.where(np.abs(base) > 1e-12, base, np.nan)

        for j in range(n_gt):
            Xp = X.copy()
            raw_perturbed = raw_gt[:, j] * 1.01
            Xp[:, j] = (raw_perturbed - scaler.mean_[j]) / scaler.scale_[j]
            with torch.no_grad():
                perturbed = model(module.tensor(Xp), countries).numpy().ravel() * y_sd + y_mean
            elasticities[:, j] += ((perturbed - base) / output_denom) / 0.01

    return elasticities / len(models)


def pearson(a, b) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(a[mask], b[mask])
    return float(r), float(p)


def growth(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.diff(x) / x[:-1]


def series_comparison(new: pd.DataFrame, old: pd.DataFrame) -> dict[str, float]:
    merged = new[["Year", "Month", "NN"]].merge(
        old[["Year", "Month", "NN"]], on=["Year", "Month"], suffixes=("_raw", "_standardized")
    )
    raw = merged.NN_raw.values
    standardized = merged.NN_standardized.values
    return {
        "N_months": int(len(merged)),
        "level_correlation": pearson(raw, standardized)[0],
        "growth_correlation": pearson(growth(raw), growth(standardized))[0],
        "mean_absolute_difference_usd_bn": float(np.mean(np.abs(raw - standardized))),
        "mean_absolute_percent_difference": float(np.mean(np.abs(raw - standardized) / standardized) * 100),
        "maximum_absolute_difference_usd_bn": float(np.max(np.abs(raw - standardized))),
    }


def topic_comparison(new_eta: pd.DataFrame, old_eta: pd.DataFrame) -> dict[str, float]:
    merged = new_eta.merge(old_eta, on="topic", suffixes=("_raw", "_standardized")).dropna()
    top_raw = set(merged.nlargest(10, "abs_raw").topic)
    top_old = set(merged.nlargest(10, "abs_standardized").topic)
    sign_match = np.sign(merged.elasticity_raw) == np.sign(merged.elasticity_standardized)
    return {
        "N_topics": int(len(merged)),
        "signed_elasticity_correlation": pearson(merged.elasticity_raw, merged.elasticity_standardized)[0],
        "absolute_elasticity_correlation": pearson(merged.abs_raw, merged.abs_standardized)[0],
        "sign_agreement_share": float(sign_match.mean()),
        "top10_overlap": int(len(top_raw & top_old)),
    }


def quarterly_agreement(monthly: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.copy()
    frame["Quarter"] = ((frame.Month - 1) // 3) + 1
    quarterly = frame.groupby(["Year", "Quarter"], as_index=False)[["NN", "Sax", "Mosley"]].sum()
    rows = []
    for benchmark in ["Mosley", "Sax"]:
        rows.append(
            {
                "benchmark": benchmark,
                "N_quarters": int(len(quarterly)),
                "level_correlation": pearson(quarterly.NN, quarterly[benchmark])[0],
                "growth_correlation": pearson(growth(quarterly.NN), growth(quarterly[benchmark]))[0],
                "growth_p_value": pearson(growth(quarterly.NN), growth(quarterly[benchmark]))[1],
            }
        )
    quarterly.to_csv(OUT / "corrected_temporal_quarterly_estimates.csv", index=False)
    return pd.DataFrame(rows)


def employment_lags(module, monthly: pd.DataFrame, employment: pd.DataFrame) -> pd.DataFrame:
    merged = monthly.merge(employment, on="date", how="inner").sort_values("date").reset_index(drop=True)
    rows = []
    for lag_row in module.lag_correlations(merged.NN.values, merged.emp.values, range(-12, 13)):
        rows.append(lag_row)
    return pd.DataFrame(rows)


def main() -> None:
    module = load_current_module()
    df, agt_cols, topics, gt, rd_us, refs, employment = module.load_inputs()
    all_correlations = []
    all_topics = []
    temporal_monthly = None
    temporal_eta = None

    for split in ["temporal", "random", "alldata"]:
        print(f"=== raw-scale elasticity split: {split} ===", flush=True)
        models, X, country_codes, _le, y_mean, y_sd, train, _n_gt, _X_gt = module.train_split_models(
            df, agt_cols, split
        )
        raw_gt = df[agt_cols].fillna(0).astype(float).values
        scaler = StandardScaler().fit(raw_gt[train])
        corrected = raw_scale_elasticities(
            module, models, X, country_codes, y_mean, y_sd, raw_gt, scaler
        )
        eta = module.topic_elasticities(df, agt_cols, topics, corrected, train)
        eta_frame = pd.DataFrame({"topic": list(eta), "elasticity": list(eta.values())})
        eta_frame["abs_elasticity"] = eta_frame.elasticity.abs()
        eta_frame.to_csv(OUT / f"corrected_{split}_topic_elasticities.csv", index=False)

        monthly = module.disaggregate_us(eta, topics, gt, rd_us, refs)
        monthly.to_csv(OUT / f"corrected_{split}_monthly_estimates.csv", index=False)
        agreement = module.method_agreement(monthly)
        agreement["split"] = split
        agreement["sum_topic_elasticities"] = float(np.nansum(eta_frame.elasticity))
        agreement["negative_topic_share"] = float((eta_frame.elasticity < 0).mean())
        agreement["minimum_monthly_estimate"] = float(monthly.NN.min())
        all_correlations.append(agreement)

        ranked = eta_frame.sort_values("abs_elasticity", ascending=False).head(15)
        for rank, row in enumerate(ranked.itertuples(index=False), 1):
            all_topics.append(
                {
                    "split": split,
                    "rank": rank,
                    "topic": row.topic,
                    "elasticity": row.elasticity,
                    "abs_elasticity": row.abs_elasticity,
                }
            )
        print(
            f"{split}: Mosley level={agreement['Mosley_lvl']:.3f}, growth={agreement['Mosley_gr']:.3f}; "
            f"Chow-Lin level={agreement['Sax_lvl']:.3f}, growth={agreement['Sax_gr']:.3f}",
            flush=True,
        )

        if split == "temporal":
            temporal_monthly = monthly
            temporal_eta = eta_frame

    assert temporal_monthly is not None and temporal_eta is not None
    correlations = pd.DataFrame(all_correlations)
    correlations.to_csv(OUT / "corrected_split_method_agreement.csv", index=False)
    pd.DataFrame(all_topics).to_csv(OUT / "corrected_top_topics_all_splits.csv", index=False)

    old_monthly = pd.read_csv(OLD_OUT / "updated_stepB_temporal_monthly_estimates.csv")
    old_eta = pd.read_csv(OLD_OUT / "updated_stepB_temporal_topic_elasticities.csv")
    old_eta["abs_elasticity"] = old_eta.elasticity.abs()
    old_eta.rename(columns={"abs_elasticity": "abs_standardized"}, inplace=True)
    new_eta = temporal_eta.rename(columns={"abs_elasticity": "abs_raw"})
    series_metrics = series_comparison(temporal_monthly, old_monthly)
    topic_metrics = topic_comparison(new_eta, old_eta)

    quarterly = quarterly_agreement(temporal_monthly)
    quarterly.to_csv(OUT / "corrected_quarterly_method_agreement.csv", index=False)
    employment = employment_lags(module, temporal_monthly, employment)
    employment.to_csv(OUT / "corrected_temporal_employment_lags.csv", index=False)

    annual_check = temporal_monthly.groupby("Year").NN.sum().rename("monthly_sum").to_frame()
    annual_check["annual_GERD"] = annual_check.index.map(rd_us)
    annual_check["difference"] = annual_check.monthly_sum - annual_check.annual_GERD
    annual_check.to_csv(OUT / "corrected_annual_aggregation_check.csv")

    summary = {
        "series_comparison_with_previous_implementation": series_metrics,
        "topic_weight_comparison_with_previous_implementation": topic_metrics,
        "maximum_absolute_annual_aggregation_error": float(annual_check.difference.abs().max()),
        "all_corrected_monthly_estimates_positive": bool((temporal_monthly.NN > 0).all()),
    }
    (OUT / "corrected_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== corrected split agreement ===")
    print(correlations.to_string(index=False))
    print("\n=== corrected vs previous temporal series ===")
    print(pd.Series(series_metrics).to_string())
    print("\n=== corrected vs previous topic weights ===")
    print(pd.Series(topic_metrics).to_string())
    print("\n=== corrected quarterly agreement ===")
    print(quarterly.to_string(index=False))
    print("\n=== corrected significant employment lags (p < 0.01) ===")
    print(employment[employment.P_Value < 0.01].to_string(index=False))
    print(f"\nOutputs saved under {OUT}")


if __name__ == "__main__":
    main()
