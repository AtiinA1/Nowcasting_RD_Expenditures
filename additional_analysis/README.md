# Analysis map

This directory contains the paper-facing experiments and the robustness work
developed during revision. Saved CSVs, figures, and tables are retained beside
their generating scripts.

## Current paper workflow

| Path | Role |
| --- | --- |
| `out/` | Shared merged features and baseline temporal results |
| `robustness_overfit/` | Current aligned neural-network ensembles, OLS and regularized linear benchmarks, loss curves, rolling-origin checks, lag sensitivity, and paper artifact refresh scripts |
| `pre_raw_stepb_method_audit/` | Leakage-corrected MIDAS benchmarks, uniform allocation, and incremental-timing checks |
| `exact_stepa_stepb_audit/` | Exact Step A architecture audit, two sensitivity definitions, refreshed Chow-Lin inputs, and corrected country-year DM tests |
| `employment_inference_robustness/` | HAC diagnostics and year-block simultaneous bands across employment lags |
| `cy_current_nn_refresh/` | Country-year grouped robustness with the current network architecture |
| `stepb_raw_scale_elasticity_refresh/` | Original-scale elasticity robustness |
| `stepb_raw_scale_stochastic_perturbation/` | Stochastic raw-scale finite-difference robustness |
| `revisions/` | Shock, calibration, seasonality, embedding, and timeliness analyses |

## Current entry points

Run these scripts from the repository root. Unless otherwise noted, they write
their outputs below the folder containing the script.

| Analysis | Script |
| --- | --- |
| Step A, all seven neural-network input configurations | `robustness_overfit/10_all_configs_updated_pure_nn.py` |
| Step A paper figures and tables | `robustness_overfit/11_refresh_stepA_paper_artifacts.py` and `13_fig2_distribution_refresh.py` |
| Corrected temporal MIDAS and sg-LASSO-MIDAS | `pre_raw_stepb_method_audit/leakage_free_midas_benchmarks.py` |
| Principal Step B neural-network allocation | `robustness_overfit/12_refresh_stepB_updated_agt.py` |
| Ridge, Elastic Net, OLS, and model-agnostic Step B | `robustness_overfit/02_regularized_linear_benchmarks.py` and `04_stepb_model_agnostic_elasticities.py` |
| Quarterly Step B checks | `robustness_overfit/05_quarterly_stepb_values.py` |
| Exact Step A-to-Step B sensitivity audit | `exact_stepa_stepb_audit/run_audit.py` |
| Uniform allocation and incremental timing | `pre_raw_stepb_method_audit/uniform_employment_benchmark.py` and `incremental_timing_test.py` |
| Dependence-robust employment diagnostics | `employment_inference_robustness/year_block_hac_inference.py` |
| Country-year grouped robustness | `cy_current_nn_refresh/run_cy_current_nn.py` |

The `out/` directories contain the research snapshot used for the manuscript.
Rerunning neural-network ensembles may produce small last-decimal differences
across PyTorch versions and hardware.

## Naming convention

Several directory and script names preserve the order in which revision checks
were developed. For example, `12_refresh_stepB_updated_agt.py` is the current
paper-facing Step B implementation despite the historical words `refresh` and
`updated` in its filename. The paper refers to it simply as the AGT neural
network. Similarly, `pre_raw_stepb_method_audit/` now contains the corrected
mixed-frequency benchmarks and uniform-allocation diagnostics used by the
current repository snapshot.

## Earlier and supporting scripts

The scripts directly under `additional_analysis/` document the development of
the temporal split, annual prediction tables, Step B estimates, SHAP analyses,
and figures. Some are superseded by the focused folders above and may contain
historical local paths. They are retained for provenance, not as current entry
points. `benchmark_and_dm.py` records the earlier observation-level random-split
exercise that helped identify information leakage; it is not the paper's
primary evaluation.

Use the command order in [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for a
clean rerun of the current paper pipeline.
