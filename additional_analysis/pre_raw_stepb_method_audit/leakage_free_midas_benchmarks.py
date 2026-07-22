"""Leakage-free rerun of the paper's mixed-frequency Step A benchmarks.

The original scripts normalized the raw Google Trends histories over the full
sample before applying the chronological split. This audit estimates those
country-level and country-topic normalization constants on training years only.
All other split, target, tuning, and scoring choices are preserved.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import legendre as legendre
from scipy import optimize, stats


CODE = Path(__file__).resolve().parents[2]
SOURCE_OUT = CODE / "additional_analysis" / "out"
REFRESH_OUT = CODE / "additional_analysis" / "robustness_overfit" / "out" / "paper_stepA_refresh"
OUT = CODE / "additional_analysis" / "pre_raw_stepb_method_audit" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def chronological_split(panel: pd.DataFrame) -> dict[tuple[str, int], str]:
    split: dict[tuple[str, int], str] = {}
    for country, group in panel.groupby("Country"):
        years = np.array(sorted(group.Year.unique()))
        n_train = int(round(len(years) * 0.64))
        n_val = int(round(len(years) * 0.16))
        for index, year in enumerate(years):
            split[(country, int(year))] = (
                "train" if index < n_train else "val" if index < n_train + n_val else "test"
            )
    return split


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    return {
        "n": int(len(actual)),
        "MAPE": float(np.mean(np.abs(error / actual)) * 100),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def dm_test(actual: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    loss_diff = (actual - first) ** 2 - (actual - second) ** 2
    n = len(loss_diff)
    if n < 5 or np.isclose(np.var(loss_diff, ddof=1), 0):
        return math.nan, math.nan
    statistic = np.mean(loss_diff) / np.sqrt(np.var(loss_diff, ddof=1) / n)
    statistic *= np.sqrt((n - 1) / n)
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p_value)


def training_stats(
    values: pd.DataFrame,
    split: dict[tuple[str, int], str],
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    train_mask = np.array(
        [split.get((country, int(year))) == "train" for country, year in zip(values.Country, values.Year)]
    )
    stats_frame = (
        values.loc[train_mask]
        .groupby(group_columns)[value_column]
        .agg(train_mean="mean", train_sd="std")
        .reset_index()
    )
    stats_frame["train_sd"] = stats_frame.train_sd.fillna(1.0).replace(0.0, 1.0)
    return stats_frame


def full_sample_stats(values: pd.DataFrame, group_columns: list[str], value_column: str) -> pd.DataFrame:
    stats_frame = (
        values.groupby(group_columns)[value_column]
        .agg(train_mean="mean", train_sd="std")
        .reset_index()
    )
    stats_frame["train_sd"] = stats_frame.train_sd.fillna(1.0).replace(0.0, 1.0)
    return stats_frame


def prepare_panel():
    features = pd.read_csv(SOURCE_OUT / "merged_features.csv")
    features = features[features.Year >= 2004].copy()
    panel = (
        features.groupby(["Country", "Year"], as_index=False)
        .rd_expenditure.mean()
        .rename(columns={"rd_expenditure": "GERD"})
    )
    split = chronological_split(panel)
    country_mean: dict[str, float] = {}
    country_sd: dict[str, float] = {}
    for country, group in panel.groupby("Country"):
        train = group[[split[(country, int(year))] == "train" for year in group.Year]]
        logged = np.log(train.GERD.to_numpy())
        country_mean[country] = float(logged.mean())
        country_sd[country] = float(max(logged.std(), 0.05))
    return panel, split, country_mean, country_sd


def to_level(country, prediction_z, country_mean, country_sd):
    return np.array(
        [np.exp(value * country_sd[c] + country_mean[c]) for value, c in zip(prediction_z, country)]
    )


def composite_models(panel, split, country_mean, country_sd, normalization_scope="train"):
    gt = pd.read_csv(CODE / "data" / "GT" / "trends_data_by_topic_filtered.csv")
    gt["date"] = pd.to_datetime(gt.date)
    gt["Year"] = gt.date.dt.year
    gt["Month"] = gt.date.dt.month
    long = gt.melt(id_vars=["date", "Year", "Month"], var_name="country_topic", value_name="value")
    long[["Country", "topic"]] = long.country_topic.str.split("_", n=1, expand=True)
    composite = long.groupby(["Country", "Year", "Month"], as_index=False).value.mean()
    stats_frame = (
        training_stats(composite, split, ["Country"], "value")
        if normalization_scope == "train"
        else full_sample_stats(composite, ["Country"], "value")
    )
    composite = composite.merge(stats_frame, on="Country", how="inner")
    composite["z"] = (composite.value - composite.train_mean) / composite.train_sd

    wide = composite.pivot_table(index=["Country", "Year"], columns="Month", values="z").reset_index()
    wide.columns = ["Country", "Year"] + [f"gt{month}" for month in range(1, 13)]
    gt_columns = [f"gt{month}" for month in range(1, 13)]
    regression = panel.merge(wide, on=["Country", "Year"], how="left").dropna(subset=gt_columns).reset_index(drop=True)
    regression["split"] = [split[(c, int(y))] for c, y in zip(regression.Country, regression.Year)]
    regression["ystd"] = [
        (np.log(value) - country_mean[c]) / country_sd[c]
        for c, value in zip(regression.Country, regression.GERD)
    ]
    country_dummies = pd.get_dummies(regression.Country, prefix="c").astype(float).to_numpy()
    monthly = regression[gt_columns].to_numpy()
    fit = regression.split.ne("test").to_numpy()

    unrestricted_x = np.column_stack([np.ones(len(regression)), country_dummies, monthly])
    unrestricted_beta = np.linalg.solve(
        unrestricted_x[fit].T @ unrestricted_x[fit] + 1e-3 * np.eye(unrestricted_x.shape[1]),
        unrestricted_x[fit].T @ regression.ystd.to_numpy()[fit],
    )
    regression["UMIDAS"] = to_level(
        regression.Country, unrestricted_x @ unrestricted_beta, country_mean, country_sd
    )

    def beta_weights(parameters):
        first, second = np.exp(parameters)
        support = np.linspace(1e-3, 1, 12)
        weights = support ** (first - 1) * (1 - support) ** (second - 1)
        return weights / weights.sum()

    def residuals(parameters):
        index = monthly @ beta_weights(parameters)
        design = np.column_stack([np.ones(len(regression)), country_dummies, index])
        beta = np.linalg.solve(
            design[fit].T @ design[fit] + 1e-3 * np.eye(design.shape[1]),
            design[fit].T @ regression.ystd.to_numpy()[fit],
        )
        return regression.ystd.to_numpy()[fit] - design[fit] @ beta

    selected = optimize.least_squares(residuals, np.log([2.0, 2.0]), method="lm", max_nfev=2000)
    index = monthly @ beta_weights(selected.x)
    restricted_x = np.column_stack([np.ones(len(regression)), country_dummies, index])
    restricted_beta = np.linalg.solve(
        restricted_x[fit].T @ restricted_x[fit] + 1e-3 * np.eye(restricted_x.shape[1]),
        restricted_x[fit].T @ regression.ystd.to_numpy()[fit],
    )
    regression["MIDAS"] = to_level(
        regression.Country, restricted_x @ restricted_beta, country_mean, country_sd
    )
    return regression[regression.split.eq("test")][["Country", "Year", "GERD", "MIDAS", "UMIDAS"]]


def sparse_group_lasso_models(panel, split, country_mean, country_sd, normalization_scope="train"):
    gt = pd.read_csv(CODE / "data" / "GT" / "trends_data_by_topic_filtered.csv")
    gt["date"] = pd.to_datetime(gt.date)
    gt["Year"] = gt.date.dt.year
    gt["Month"] = gt.date.dt.month
    long = gt.melt(id_vars=["date", "Year", "Month"], var_name="country_topic", value_name="value")
    long[["Country", "topic"]] = long.country_topic.str.split("_", n=1, expand=True)
    topics = sorted(set.intersection(*[set(group.topic.unique()) for _, group in long.groupby("Country")]))
    long = long[long.topic.isin(topics)].copy()
    stats_frame = (
        training_stats(long, split, ["Country", "topic"], "value")
        if normalization_scope == "train"
        else full_sample_stats(long, ["Country", "topic"], "value")
    )
    long = long.merge(stats_frame, on=["Country", "topic"], how="inner")
    long["z"] = (long.value - long.train_mean) / long.train_sd

    monthly = {}
    for (country, year, topic), group in long.groupby(["Country", "Year", "topic"]):
        if group.Month.nunique() == 12:
            monthly[(country, int(year), topic)] = group.sort_values("Month").z.to_numpy()[:12]
    usable = panel.copy()
    usable["ok"] = [
        all((country, int(year), topic) in monthly for topic in topics)
        for country, year in zip(usable.Country, usable.Year)
    ]
    usable = usable[usable.ok].reset_index(drop=True)
    usable["split"] = [split[(c, int(y))] for c, y in zip(usable.Country, usable.Year)]
    ystd = np.array(
        [(np.log(value) - country_mean[c]) / country_sd[c] for c, value in zip(usable.Country, usable.GERD)]
    )
    country_dummies = pd.get_dummies(usable.Country, prefix="c").astype(float).to_numpy()
    unpenalized = np.column_stack([np.ones(len(usable)), country_dummies])
    train = usable.split.eq("train").to_numpy()
    validation = usable.split.eq("val").to_numpy()
    test = usable.split.eq("test").to_numpy()

    def midas_features(degree):
        support = (np.arange(1, 13) - 0.5) / 12 * 2 - 1
        basis = np.column_stack(
            [legendre.legval(support, [0] * order + [1]) for order in range(degree + 1)]
        )
        basis /= np.sqrt((basis**2).sum(axis=0))
        design = np.zeros((len(usable), len(topics) * (degree + 1)))
        for row, (country, year) in enumerate(zip(usable.Country, usable.Year)):
            for topic_index, topic in enumerate(topics):
                start = topic_index * (degree + 1)
                design[row, start : start + degree + 1] = basis.T @ monthly[(country, int(year), topic)]
        groups = [
            np.arange(topic_index * (degree + 1), (topic_index + 1) * (degree + 1))
            for topic_index in range(len(topics))
        ]
        return design, groups

    def fit_sgl(penalized, mask, groups, penalty, alpha, group_scale, iterations):
        xu, xp, target = unpenalized[mask], penalized[mask], ystd[mask]
        n = mask.sum()
        step = 1.0 / (np.linalg.norm(np.column_stack([xu, xp]), 2) ** 2 / n)
        beta_u = np.zeros(unpenalized.shape[1])
        beta_p = np.zeros(penalized.shape[1])
        for _ in range(iterations):
            residual = target - xu @ beta_u - xp @ beta_p
            beta_u += step * (xu.T @ residual) / n
            z = beta_p + step * (xp.T @ (target - xu @ beta_u - xp @ beta_p)) / n
            z = np.sign(z) * np.maximum(np.abs(z) - step * penalty * alpha, 0.0)
            for group in groups:
                norm = np.linalg.norm(z[group])
                beta_p[group] = (
                    z[group] * max(0.0, 1 - step * penalty * (1 - alpha) * group_scale / norm)
                    if norm > 0 else 0.0
                )
        return beta_u, beta_p

    best = None
    for degree in (2, 3, 4):
        raw_features, groups = midas_features(degree)
        mean = raw_features[train].mean(axis=0)
        sd = raw_features[train].std(axis=0)
        sd[sd == 0] = 1.0
        penalized = (raw_features - mean) / sd
        for alpha in (0.05, 0.2, 0.5):
            for penalty in np.geomspace(0.001, 1.0, 16):
                beta_u, beta_p = fit_sgl(
                    penalized, train, groups, penalty, alpha, np.sqrt(degree + 1), 1200
                )
                prediction = unpenalized[validation] @ beta_u + penalized[validation] @ beta_p
                level = to_level(usable.Country.to_numpy()[validation], prediction, country_mean, country_sd)
                validation_mape = metrics(usable.GERD.to_numpy()[validation], level)["MAPE"]
                candidate = (validation_mape, degree, penalty, alpha)
                if best is None or candidate[0] < best[0]:
                    best = candidate

    _, degree, penalty, alpha = best
    raw_features, groups = midas_features(degree)
    mean = raw_features[train].mean(axis=0)
    sd = raw_features[train].std(axis=0)
    sd[sd == 0] = 1.0
    penalized = (raw_features - mean) / sd
    beta_u, beta_p = fit_sgl(
        penalized, train | validation, groups, penalty, alpha, np.sqrt(degree + 1), 6000
    )
    prediction = unpenalized[test] @ beta_u + penalized[test] @ beta_p
    level = to_level(usable.Country.to_numpy()[test], prediction, country_mean, country_sd)
    selected_topics = int(sum(np.linalg.norm(beta_p[group]) > 1e-8 for group in groups))
    output = usable.loc[test, ["Country", "Year", "GERD"]].copy()
    output["SGL"] = level
    tuning = {
        "topics_available": len(topics),
        "topics_selected": selected_topics,
        "degree": degree,
        "lambda": penalty,
        "alpha": alpha,
        "validation_MAPE": best[0],
    }
    return output, tuning


def main():
    panel, split, country_mean, country_sd = prepare_panel()
    composite = composite_models(panel, split, country_mean, country_sd)
    sparse, tuning = sparse_group_lasso_models(panel, split, country_mean, country_sd)
    predictions = composite.merge(sparse[["Country", "Year", "SGL"]], on=["Country", "Year"], how="inner")

    refreshed = pd.read_csv(REFRESH_OUT / "updated_temporal_annual_all.csv")
    references = refreshed[["Country", "Year", "NN_AGT", "RW3", "MIDAS", "UMIDAS", "SGL"]].rename(
        columns={"MIDAS": "MIDAS_old", "UMIDAS": "UMIDAS_old", "SGL": "SGL_old"}
    )
    predictions = predictions.merge(references, on=["Country", "Year"], how="inner")
    predictions.to_csv(OUT / "leakage_free_midas_predictions.csv", index=False)

    rows = []
    for model, column in [
        ("AGT neural network", "NN_AGT"),
        ("MIDAS, corrected", "MIDAS"),
        ("MIDAS, previous", "MIDAS_old"),
        ("U-MIDAS, corrected", "UMIDAS"),
        ("U-MIDAS, previous", "UMIDAS_old"),
        ("sg-LASSO-MIDAS, corrected", "SGL"),
        ("sg-LASSO-MIDAS, previous", "SGL_old"),
    ]:
        result = metrics(predictions.GERD, predictions[column])
        dm_nn, p_nn = (math.nan, math.nan)
        if column != "NN_AGT":
            dm_nn, p_nn = dm_test(predictions.GERD.to_numpy(), predictions.NN_AGT.to_numpy(), predictions[column].to_numpy())
        rows.append({"Model": model, **result, "DM_NN_vs_model": dm_nn, "p_NN_vs_model": p_nn})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "leakage_free_midas_metrics.csv", index=False)
    pd.DataFrame([tuning]).to_csv(OUT / "leakage_free_sgl_tuning.csv", index=False)

    # Reproduce the previous full-sample normalization inside this audit code.
    # This is a verification run only and is not used as the corrected result.
    composite_replication = composite_models(
        panel, split, country_mean, country_sd, normalization_scope="full"
    )
    sparse_replication, replication_tuning = sparse_group_lasso_models(
        panel, split, country_mean, country_sd, normalization_scope="full"
    )
    replication = composite_replication.merge(
        sparse_replication[["Country", "Year", "SGL"]], on=["Country", "Year"], how="inner"
    ).merge(references, on=["Country", "Year"], how="inner")
    replication_rows = []
    for model, reproduced, saved in [
        ("MIDAS", "MIDAS", "MIDAS_old"),
        ("U-MIDAS", "UMIDAS", "UMIDAS_old"),
        ("sg-LASSO-MIDAS", "SGL", "SGL_old"),
    ]:
        replication_rows.append(
            {
                "Model": model,
                "max_abs_prediction_difference": float(np.max(np.abs(replication[reproduced] - replication[saved]))),
                "reproduced_MAPE": metrics(replication.GERD, replication[reproduced])["MAPE"],
                "saved_MAPE": metrics(replication.GERD, replication[saved])["MAPE"],
            }
        )
    replication_check = pd.DataFrame(replication_rows)
    replication_check.to_csv(OUT / "full_sample_normalization_replication_check.csv", index=False)
    pd.DataFrame([replication_tuning]).to_csv(OUT / "full_sample_sgl_tuning_replication.csv", index=False)

    print("\nLeakage-free mixed-frequency benchmark comparison")
    print(comparison.round(4).to_string(index=False))
    print("\nCorrected sg-LASSO-MIDAS tuning")
    print(pd.Series(tuning).to_string())
    print("\nFull-sample normalization replication check")
    print(replication_check.round(8).to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
