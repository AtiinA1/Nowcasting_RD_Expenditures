"""Raw-scale Step B robustness with stochastic finite-difference perturbations.

This experiment leaves all existing results untouched. It trains the current
temporal AGT ensemble once, computes raw-scale elasticities with a fixed 1%
perturbation and with repeated N(0.01, 0.005^2) perturbations, and compares the
resulting monthly allocations and validation diagnostics.
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


ROOT = Path("/Users/atin/Nowcasting")
CODE = ROOT / "Nowcasting_github"
SOURCE = CODE / "additional_analysis" / "robustness_overfit" / "12_refresh_stepB_updated_agt.py"
OUT = CODE / "additional_analysis" / "stepb_raw_scale_stochastic_perturbation" / "out"
HISTORY_OUT = OUT / "training_histories"
N_DRAWS = 20
DRAW_SEED = 20260714
PREDICT_BATCH = 8192

OUT.mkdir(parents=True, exist_ok=True)
HISTORY_OUT.mkdir(parents=True, exist_ok=True)


def load_module():
    spec = importlib.util.spec_from_file_location("current_stepb_stochastic", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROBUST_OUT = HISTORY_OUT
    return module


def predict_batched(module, model, x: np.ndarray, countries: np.ndarray, y_mean: float, y_sd: float) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), PREDICT_BATCH):
            stop = min(start + PREDICT_BATCH, len(x))
            pred = model(module.tensor(x[start:stop]), torch.LongTensor(countries[start:stop]))
            values.append(pred.numpy().ravel() * y_sd + y_mean)
    return np.concatenate(values)


def elasticities_by_draw(
    module,
    models,
    x: np.ndarray,
    country_codes: np.ndarray,
    y_mean: float,
    y_sd: float,
    raw_gt: np.ndarray,
    scaler: StandardScaler,
    deltas: np.ndarray,
) -> np.ndarray:
    """Return ensemble-mean raw-scale elasticities by perturbation draw."""
    n_obs, n_gt = raw_gt.shape
    n_draws = len(deltas)
    result = np.zeros((n_draws, n_obs, n_gt), dtype=np.float64)

    for model_number, model in enumerate(models, 1):
        base = predict_batched(module, model, x, country_codes, y_mean, y_sd)
        output_denom = np.where(np.abs(base) > 1e-12, base, np.nan)
        tiled_countries = np.tile(country_codes, n_draws)

        for feature in range(n_gt):
            stacked = np.tile(x, (n_draws, 1))
            perturbed_raw = np.concatenate([raw_gt[:, feature] * (1.0 + delta) for delta in deltas])
            stacked[:, feature] = (perturbed_raw - scaler.mean_[feature]) / scaler.scale_[feature]
            pred = predict_batched(module, model, stacked, tiled_countries, y_mean, y_sd).reshape(n_draws, n_obs)
            result[:, :, feature] += ((pred - base[None, :]) / output_denom[None, :]) / deltas[:, None]

        print(f"elasticities: model {model_number}/{len(models)}", flush=True)

    return result / len(models)


def corr(a, b) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan, np.nan
    value, p_value = stats.pearsonr(a[mask], b[mask])
    return float(value), float(p_value)


def growth(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.diff(values) / values[:-1]


def quarterly(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Quarter"] = ((out.Month - 1) // 3) + 1
    return out.groupby(["Year", "Quarter"], as_index=False)[["NN", "Mosley", "Sax"]].sum()


def quarterly_employment(frame: pd.DataFrame, employment: pd.DataFrame) -> tuple[float, float, int]:
    q = quarterly(frame)
    emp = employment.copy()
    emp["Year"] = emp.date.dt.year
    emp["Quarter"] = emp.date.dt.quarter
    emp_q = emp.groupby(["Year", "Quarter"], as_index=False).agg(emp=("emp", "mean"))
    merged = q.merge(emp_q, on=["Year", "Quarter"], how="inner").sort_values(["Year", "Quarter"])
    r, p = corr(growth(merged.NN), growth(merged.emp))
    return r, p, len(merged) - 1


def diagnostics(module, frame: pd.DataFrame, employment: pd.DataFrame) -> dict[str, float]:
    monthly = module.method_agreement(frame)
    q = quarterly(frame)
    emp_r, emp_p, emp_n = quarterly_employment(frame, employment)
    return {
        **monthly,
        "Mosley_q_lvl": corr(q.NN, q.Mosley)[0],
        "Mosley_q_gr": corr(growth(q.NN), growth(q.Mosley))[0],
        "Sax_q_lvl": corr(q.NN, q.Sax)[0],
        "Sax_q_gr": corr(growth(q.NN), growth(q.Sax))[0],
        "quarterly_employment_r": emp_r,
        "quarterly_employment_p": emp_p,
        "quarterly_employment_n": emp_n,
        "minimum_monthly_estimate": float(frame.NN.min()),
    }


def main() -> None:
    module = load_module()
    df, agt_cols, topics, gt, rd_us, refs, employment = module.load_inputs()
    models, x, country_codes, _le, y_mean, y_sd, train, _n_gt, _x_gt = module.train_split_models(
        df, agt_cols, "temporal"
    )
    raw_gt = df[agt_cols].fillna(0).astype(float).values
    scaler = StandardScaler().fit(raw_gt[train])

    rng = np.random.default_rng(DRAW_SEED)
    deltas = rng.normal(0.01, 0.005, N_DRAWS)
    # Extremely small draws amplify floating-point noise in the finite difference.
    while np.any(np.abs(deltas) < 0.001):
        mask = np.abs(deltas) < 0.001
        deltas[mask] = rng.normal(0.01, 0.005, mask.sum())
    pd.DataFrame({"draw": np.arange(1, N_DRAWS + 1), "delta": deltas}).to_csv(OUT / "perturbation_draws.csv", index=False)

    fixed = elasticities_by_draw(
        module, models, x, country_codes, y_mean, y_sd, raw_gt, scaler, np.array([0.01])
    )[0]
    stochastic = elasticities_by_draw(
        module, models, x, country_codes, y_mean, y_sd, raw_gt, scaler, deltas
    )

    rows = []
    monthly_draws = []
    topic_draws = []

    fixed_eta = module.topic_elasticities(df, agt_cols, topics, fixed, train)
    fixed_monthly = module.disaggregate_us(fixed_eta, topics, gt, rd_us, refs)
    fixed_metrics = diagnostics(module, fixed_monthly, employment)
    rows.append({"estimator": "fixed_1pct", "draw": 0, "delta": 0.01, **fixed_metrics})

    for draw, delta in enumerate(deltas, 1):
        eta = module.topic_elasticities(df, agt_cols, topics, stochastic[draw - 1], train)
        monthly = module.disaggregate_us(eta, topics, gt, rd_us, refs)
        metrics = diagnostics(module, monthly, employment)
        metrics.update(
            {
                "estimator": "normal_draw",
                "draw": draw,
                "delta": delta,
                "fixed_level_corr": corr(monthly.NN, fixed_monthly.NN)[0],
                "fixed_growth_corr": corr(growth(monthly.NN), growth(fixed_monthly.NN))[0],
                "fixed_monthly_mae": float(np.mean(np.abs(monthly.NN - fixed_monthly.NN))),
            }
        )
        rows.append(metrics)
        copy = monthly.copy()
        copy.insert(0, "draw", draw)
        copy.insert(1, "delta", delta)
        monthly_draws.append(copy)
        topic_draws.extend(
            {"draw": draw, "delta": delta, "topic": topic, "elasticity": value}
            for topic, value in eta.items()
        )

    mean_elasticities = stochastic.mean(axis=0)
    mean_eta = module.topic_elasticities(df, agt_cols, topics, mean_elasticities, train)
    mean_monthly = module.disaggregate_us(mean_eta, topics, gt, rd_us, refs)
    mean_metrics = diagnostics(module, mean_monthly, employment)
    mean_metrics.update(
        {
            "estimator": "normal_mean",
            "draw": -1,
            "delta": float(deltas.mean()),
            "fixed_level_corr": corr(mean_monthly.NN, fixed_monthly.NN)[0],
            "fixed_growth_corr": corr(growth(mean_monthly.NN), growth(fixed_monthly.NN))[0],
            "fixed_monthly_mae": float(np.mean(np.abs(mean_monthly.NN - fixed_monthly.NN))),
        }
    )
    rows.append(mean_metrics)

    fixed_monthly.to_csv(OUT / "fixed_1pct_monthly_estimates.csv", index=False)
    mean_monthly.to_csv(OUT / "normal_mean_monthly_estimates.csv", index=False)
    pd.DataFrame({"topic": list(fixed_eta), "elasticity": list(fixed_eta.values())}).to_csv(
        OUT / "fixed_1pct_topic_elasticities.csv", index=False
    )
    pd.DataFrame({"topic": list(mean_eta), "elasticity": list(mean_eta.values())}).to_csv(
        OUT / "normal_mean_topic_elasticities.csv", index=False
    )
    pd.concat(monthly_draws, ignore_index=True).to_csv(OUT / "normal_draw_monthly_estimates.csv", index=False)
    pd.DataFrame(topic_draws).to_csv(OUT / "normal_draw_topic_elasticities.csv", index=False)
    diagnostics_frame = pd.DataFrame(rows)
    diagnostics_frame.to_csv(OUT / "diagnostics_by_draw.csv", index=False)

    draw_only = diagnostics_frame[diagnostics_frame.estimator == "normal_draw"]
    summary = {
        "normal_draws": {
            "n": N_DRAWS,
            "seed": DRAW_SEED,
            "sample_mean": float(deltas.mean()),
            "sample_sd": float(deltas.std(ddof=1)),
            "minimum": float(deltas.min()),
            "maximum": float(deltas.max()),
        },
        "fixed_1pct": fixed_metrics,
        "normal_mean": mean_metrics,
        "draw_dispersion": {
            col: {
                "mean": float(draw_only[col].mean()),
                "sd": float(draw_only[col].std(ddof=1)),
                "min": float(draw_only[col].min()),
                "max": float(draw_only[col].max()),
            }
            for col in [
                "fixed_level_corr",
                "fixed_growth_corr",
                "fixed_monthly_mae",
                "Mosley_lvl",
                "Mosley_gr",
                "Mosley_q_gr",
                "quarterly_employment_r",
            ]
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== perturbation draws ===")
    print(pd.Series(summary["normal_draws"]).to_string())
    print("\n=== fixed 1% versus mean stochastic estimator ===")
    print(diagnostics_frame[diagnostics_frame.estimator.isin(["fixed_1pct", "normal_mean"])].to_string(index=False))
    print("\n=== dispersion across normal draws ===")
    print(pd.DataFrame(summary["draw_dispersion"]).T.to_string())
    print(f"\nOutputs saved under {OUT}")


if __name__ == "__main__":
    main()
