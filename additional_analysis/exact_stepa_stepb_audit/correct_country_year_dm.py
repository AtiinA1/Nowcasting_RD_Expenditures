"""Recalculate country-year DM tests with the stated h=1 correction."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "cy_current_nn_refresh" / "out"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)


def dm(actual, first, second, power=2):
    actual = np.asarray(actual, dtype=float)
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(first) & np.isfinite(second)
    actual, first, second = actual[mask], first[mask], second[mask]
    first_error, second_error = actual - first, actual - second
    if power == 2:
        differential = first_error**2 - second_error**2
    else:
        differential = np.abs(first_error) - np.abs(second_error)
    n = len(differential)
    uncorrected = differential.mean() / np.sqrt(np.var(differential, ddof=1) / n)
    corrected = uncorrected * np.sqrt((n - 1) / n)
    p_value = 2 * (1 - stats.t.cdf(abs(corrected), df=n - 1))
    return corrected, p_value, n


def main():
    predictions = pd.read_csv(SOURCE / "cy_current_nn_predictions.csv")
    annual = (
        predictions.groupby(["Config", "Country", "Year"], as_index=False)
        .agg(GERD=("rd_expenditure", "mean"), prediction=("pred_mean", "mean"))
        .pivot_table(index=["Country", "Year", "GERD"], columns="Config", values="prediction")
        .reset_index()
    )
    corrected_benchmarks = pd.read_csv(
        HERE.parent / "pre_raw_stepb_method_audit" / "out" / "leakage_free_cy_midas_predictions.csv"
    )
    combined = annual.merge(
        corrected_benchmarks.drop(columns="GERD"), on=["Country", "Year"], how="inner", validate="one_to_one"
    )

    rows = []
    for model in ["LagRD", "Macros", "AGT", "MGT", "AGTwRD", "MGTwRD", "AllVar"]:
        statistic, p_value, n = dm(combined.GERD, combined[model], combined.RW)
        rows.append({"comparison": f"{model} vs RW", "DM": statistic, "p_value": p_value, "n": n})
    for benchmark_name in ["UMIDAS", "MIDAS"]:
        statistic, p_value, n = dm(combined.GERD, combined.AGT, combined[benchmark_name])
        rows.append({"comparison": f"AGT vs {benchmark_name}", "DM": statistic, "p_value": p_value, "n": n})

    # AR(1) predictions were not retained, but the old statistic can be
    # converted exactly because only the multiplicative HLN factor changes.
    old = pd.read_csv(SOURCE / "cy_dm_vs_rw.csv")
    for row in old.itertuples():
        if row.Model != "AR1":
            continue
        n = int(row.n)
        statistic = float(row.DM_vs_RW) * np.sqrt((n - 1) / (n + 1))
        p_value = 2 * (1 - stats.t.cdf(abs(statistic), df=n - 1))
        rows.append({"comparison": "AR1 vs RW", "DM": statistic, "p_value": p_value, "n": n})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "country_year_dm_corrected.csv", index=False)
    combined.to_csv(OUT / "country_year_annual_predictions_audit.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
