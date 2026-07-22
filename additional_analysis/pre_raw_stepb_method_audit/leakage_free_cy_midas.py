"""Correct country-year MIDAS normalization using the saved NN split design."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


CODE = Path(__file__).resolve().parents[2]
SOURCE = CODE / "additional_analysis" / "out"
OUT = CODE / "additional_analysis" / "pre_raw_stepb_method_audit" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def main():
    frame = pd.read_csv(SOURCE / "merged_features.csv")
    frame = frame[frame.Year >= 2004].copy().sort_values(["Country", "Year", "Month"])
    panel = frame.groupby(["Country", "Year"], as_index=False).rd_expenditure.mean().rename(
        columns={"rd_expenditure": "GERD"}
    )
    rng = np.random.default_rng(SEED)
    split = {}
    for country, group in frame.groupby("Country"):
        years = np.array(sorted(group.Year.unique()))
        rng.shuffle(years)
        n_train = int(round(len(years) * 0.64))
        n_val = int(round(len(years) * 0.16))
        for index, year in enumerate(years):
            split[(country, int(year))] = (
                "train" if index < n_train else "val" if index < n_train + n_val else "test"
            )
    panel["split"] = [split[(c, int(y))] for c, y in zip(panel.Country, panel.Year)]

    gt = pd.read_csv(CODE / "data" / "GT" / "trends_data_by_topic_filtered.csv")
    gt["date"] = pd.to_datetime(gt.date)
    gt["Year"] = gt.date.dt.year
    gt["Month"] = gt.date.dt.month
    long = gt.melt(id_vars=["date", "Year", "Month"], var_name="key", value_name="value")
    long[["Country", "topic"]] = long.key.str.split("_", n=1, expand=True)
    composite = long.groupby(["Country", "Year", "Month"], as_index=False).value.mean()
    composite["split"] = [split.get((c, int(y))) for c, y in zip(composite.Country, composite.Year)]
    train_stats = composite[composite.split.eq("train")].groupby("Country").value.agg(["mean", "std"])
    composite = composite.merge(train_stats, on="Country", how="inner")
    composite["z"] = (composite.value - composite["mean"]) / composite["std"].replace(0, 1)
    wide = composite.pivot_table(index=["Country", "Year"], columns="Month", values="z").reset_index()
    wide.columns = ["Country", "Year"] + [f"gt{month}" for month in range(1, 13)]
    columns = [f"gt{month}" for month in range(1, 13)]
    regression = panel.merge(wide, on=["Country", "Year"], how="left").dropna(subset=columns).reset_index(drop=True)

    target_stats = regression[regression.split.eq("train")].copy()
    target_stats["log_GERD"] = np.log(target_stats.GERD)
    means = target_stats.groupby("Country").log_GERD.mean().to_dict()
    stds = target_stats.groupby("Country").log_GERD.std().replace(0, 1).to_dict()
    regression["ystd"] = [
        (np.log(value) - means[country]) / stds[country]
        for country, value in zip(regression.Country, regression.GERD)
    ]
    country_dummies = pd.get_dummies(regression.Country, prefix="c").astype(float).to_numpy()
    monthly = regression[columns].to_numpy()
    fit = regression.split.ne("test").to_numpy()
    test = regression.split.eq("test").to_numpy()

    def to_level(values):
        return np.array(
            [np.exp(value * stds[country] + means[country]) for value, country in zip(values, regression.Country)]
        )

    unrestricted = np.column_stack([np.ones(len(regression)), country_dummies, monthly])
    beta_u = np.linalg.solve(
        unrestricted[fit].T @ unrestricted[fit] + 1e-3 * np.eye(unrestricted.shape[1]),
        unrestricted[fit].T @ regression.ystd.to_numpy()[fit],
    )
    regression["UMIDAS"] = to_level(unrestricted @ beta_u)

    def weights(parameters):
        first, second = np.exp(parameters)
        support = np.linspace(1e-3, 1, 12)
        result = support ** (first - 1) * (1 - support) ** (second - 1)
        return result / result.sum()

    def residuals(parameters):
        index = monthly @ weights(parameters)
        design = np.column_stack([np.ones(len(regression)), country_dummies, index])
        beta = np.linalg.solve(
            design[fit].T @ design[fit] + 1e-3 * np.eye(design.shape[1]),
            design[fit].T @ regression.ystd.to_numpy()[fit],
        )
        return regression.ystd.to_numpy()[fit] - design[fit] @ beta

    selected = optimize.least_squares(residuals, np.log([2.0, 2.0]), method="lm", max_nfev=2000)
    index = monthly @ weights(selected.x)
    restricted = np.column_stack([np.ones(len(regression)), country_dummies, index])
    beta_m = np.linalg.solve(
        restricted[fit].T @ restricted[fit] + 1e-3 * np.eye(restricted.shape[1]),
        restricted[fit].T @ regression.ystd.to_numpy()[fit],
    )
    regression["MIDAS"] = to_level(restricted @ beta_m)

    result = regression.loc[test, ["Country", "Year", "GERD", "MIDAS", "UMIDAS"]].copy()
    result["RW"] = [
        panel[(panel.Country.eq(country)) & (panel.Year.lt(year))].sort_values("Year").GERD.iloc[-1]
        for country, year in zip(result.Country, result.Year)
    ]
    result.to_csv(OUT / "leakage_free_cy_midas_predictions.csv", index=False)
    scale = np.mean(
        np.abs(
            np.concatenate(
                [np.diff(group.sort_values("Year").GERD.to_numpy()) for _, group in panel.groupby("Country")]
            )
        )
    )
    rows = []
    for model in ["MIDAS", "UMIDAS"]:
        error = result.GERD - result[model]
        rw_error = result.GERD - result.RW
        loss_diff = error**2 - rw_error**2
        n = len(result)
        dm = loss_diff.mean() / np.sqrt(np.var(loss_diff, ddof=1) / n) * np.sqrt((n - 1) / n)
        rows.append(
            {
                "Model": model,
                "n": n,
                "MAPE": np.mean(np.abs(error / result.GERD)) * 100,
                "RMSE": np.sqrt(np.mean(error**2)),
                "MASE": np.mean(np.abs(error)) / scale,
                "OOS_R2": 1 - np.sum(error**2) / np.sum(rw_error**2),
                "DM_vs_RW": dm,
                "p": 2 * stats.t.sf(abs(dm), df=n - 1),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "leakage_free_cy_midas_metrics.csv", index=False)
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
