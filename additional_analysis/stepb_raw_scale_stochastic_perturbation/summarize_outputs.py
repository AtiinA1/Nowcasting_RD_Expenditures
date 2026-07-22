"""Additional diagnostics for the stochastic raw-scale perturbation experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/Users/atin/Nowcasting")
CODE = ROOT / "Nowcasting_github"
OUT = CODE / "additional_analysis" / "stepb_raw_scale_stochastic_perturbation" / "out"


def growth(values):
    values = np.asarray(values, dtype=float)
    return np.diff(values) / values[:-1]


def corr(a, b):
    return float(stats.pearsonr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0])


def lag_table(name: str, monthly: pd.DataFrame, employment: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.merge(employment, on="date", how="inner").sort_values("date")
    rd, emp = growth(frame.NN), growth(frame.emp)
    rows = []
    for lag in range(-12, 13):
        if lag < 0:
            a, b = rd[-lag:], emp[:lag]
        elif lag > 0:
            a, b = rd[:-lag], emp[lag:]
        else:
            a, b = rd, emp
        r, p = stats.pearsonr(a, b)
        rows.append({"estimator": name, "lag": lag, "correlation": r, "p_value": p, "N": len(a)})
    return pd.DataFrame(rows)


def main() -> None:
    fixed = pd.read_csv(OUT / "fixed_1pct_monthly_estimates.csv")
    normal = pd.read_csv(OUT / "normal_mean_monthly_estimates.csv")
    fixed["date"] = pd.to_datetime(fixed.date)
    normal["date"] = pd.to_datetime(normal.date)
    fixed_eta = pd.read_csv(OUT / "fixed_1pct_topic_elasticities.csv")
    normal_eta = pd.read_csv(OUT / "normal_mean_topic_elasticities.csv")
    eta = fixed_eta.merge(normal_eta, on="topic", suffixes=("_fixed", "_normal"))

    employment = pd.read_csv(CODE / "data" / "datausa.io" / "Monthly Employment.csv")
    employment["date"] = pd.to_datetime(employment.Date)
    employment = employment[["date", "NSA Employees"]].rename(columns={"NSA Employees": "emp"})
    lags = pd.concat(
        [lag_table("fixed_1pct", fixed, employment), lag_table("normal_mean", normal, employment)],
        ignore_index=True,
    )
    lags.to_csv(OUT / "monthly_employment_lags.csv", index=False)

    rd = pd.read_csv(CODE / "additional_analysis" / "out" / "merged_features.csv")
    rd_us = rd[rd.Country == "US"].groupby("Year").rd_expenditure.mean()
    checks = []
    for name, frame in [("fixed_1pct", fixed), ("normal_mean", normal)]:
        annual = frame.groupby("Year").NN.sum()
        error = annual - annual.index.map(rd_us)
        checks.append({"estimator": name, "maximum_absolute_annual_error": float(np.abs(error).max())})
    pd.DataFrame(checks).to_csv(OUT / "annual_aggregation_checks.csv", index=False)

    eta["abs_fixed"] = eta.elasticity_fixed.abs()
    eta["abs_normal"] = eta.elasticity_normal.abs()
    top_fixed = set(eta.nlargest(10, "abs_fixed").topic)
    top_normal = set(eta.nlargest(10, "abs_normal").topic)
    summary = {
        "topic_elasticity_correlation": corr(eta.elasticity_fixed, eta.elasticity_normal),
        "topic_elasticity_mean_absolute_difference": float(np.mean(np.abs(eta.elasticity_fixed - eta.elasticity_normal))),
        "topic_sign_agreement": float((np.sign(eta.elasticity_fixed) == np.sign(eta.elasticity_normal)).mean()),
        "top10_overlap_by_absolute_value": int(len(top_fixed & top_normal)),
        "fixed_significant_monthly_employment_lags_p01": lags[(lags.estimator == "fixed_1pct") & (lags.p_value < 0.01)][
            ["lag", "correlation", "p_value"]
        ].to_dict("records"),
        "normal_significant_monthly_employment_lags_p01": lags[(lags.estimator == "normal_mean") & (lags.p_value < 0.01)][
            ["lag", "correlation", "p_value"]
        ].to_dict("records"),
        "annual_aggregation_checks": checks,
    }
    (OUT / "additional_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
