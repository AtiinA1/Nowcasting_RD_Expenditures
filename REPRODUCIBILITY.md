# Reproducibility guide

## 1. Environment

The primary analysis uses Python 3.11 with PyTorch, pandas, NumPy, SciPy,
scikit-learn, statsmodels, Matplotlib, seaborn, and SHAP. Install either with
Conda:

```bash
conda env create -f environment.yml
conda activate nowcasting-rd
```

or with an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

Classical temporal disaggregation also requires R and the `tempdisagg` package.
The sparse implementation retained under `temporal_disaggregation/` uses
`DisaggregateTS`.

All paper-facing Python scripts infer the repository root from their own file
location. `NOWCASTING_ROOT` and `NOWCASTING_PAPER_DIR` can override those paths
for the main robustness workflow when needed.

## 2. Data and derived panel

The repository includes the input snapshots used in the current analysis.
Their roles and provenance are summarized in [`data/README.md`](data/README.md).
The main scripts read the merged monthly panel from:

```text
additional_analysis/out/merged_features.csv
```

The panel contains country, year, and month identifiers; annual GERD; monthly
Google Trends topics and their lagged representations; annual macroeconomic
variables; and split-supporting fields. The current temporal split is created
within the scripts, country by country, using earlier years for training and
validation and the most recent years for testing.

## 3. Step A: current neural-network experiments

Run all seven information configurations with the aligned neural-network
architecture:

```bash
python additional_analysis/robustness_overfit/10_all_configs_updated_pure_nn.py
```

The script uses a 15-member ensemble, hidden widths 64 and 16, SiLU activation,
layer normalization, dropout, AdamW, Huber loss, a country embedding, and a
linear skip connection. Its primary result file is:

```text
additional_analysis/robustness_overfit/out/all_configs_updated_pure_nn/
  all_configs_updated_pure_nn_results.csv
```

Generate the Step A paper artifacts with:

```bash
python additional_analysis/robustness_overfit/11_refresh_stepA_paper_artifacts.py
python additional_analysis/robustness_overfit/13_fig2_distribution_refresh.py
```

## 4. Step A benchmarks

The corrected temporal MIDAS, U-MIDAS, and sparse-group-LASSO MIDAS analysis
fits Google Trends normalization on the training period only:

```bash
python additional_analysis/pre_raw_stepb_method_audit/leakage_free_midas_benchmarks.py
```

The country-year grouped counterpart is:

```bash
python additional_analysis/pre_raw_stepb_method_audit/leakage_free_cy_midas.py
```

Regularized linear benchmarks, rolling-origin sensitivity, training-loss
diagnostics, and lag-order checks are in
`additional_analysis/robustness_overfit/`. The centered minimum-norm OLS,
Ridge, Elastic Net, persistence benchmarks, and Diebold-Mariano comparisons
are represented in the checked-in outputs and paper tables.

## 5. Step B: model-implied allocation

The principal paper results use standardized-input predictive sensitivities
from the AGT neural-network architecture and a dedicated level-target
sensitivity fit:

```bash
python additional_analysis/robustness_overfit/12_refresh_stepB_updated_agt.py
```

Despite the historical filename, this is the paper's current neural-network
Step B implementation. The script trains ensembles and is computationally
expensive.

The exact Step A-to-Step B audit computes both original-scale elasticities and
standardized-input predictive sensitivities:

```bash
python additional_analysis/exact_stepa_stepb_audit/run_audit.py
Rscript additional_analysis/exact_stepa_stepb_audit/run_chowlin_refresh.R
python additional_analysis/exact_stepa_stepb_audit/correct_country_year_dm.py
```

Model-agnostic linear allocations and quarterly aggregation checks are run by:

```bash
python additional_analysis/robustness_overfit/04_stepb_model_agnostic_elasticities.py
python additional_analysis/robustness_overfit/05_quarterly_stepb_values.py
```

The raw-scale and stochastic-perturbation folders retain sensitivity analyses
for alternative finite-difference definitions. These are appendix robustness
results rather than the principal Step B specification.

## 6. External diagnostics

Uniform allocation and employment comparisons:

```bash
python additional_analysis/pre_raw_stepb_method_audit/uniform_employment_benchmark.py
python additional_analysis/pre_raw_stepb_method_audit/incremental_timing_test.py
python additional_analysis/employment_inference_robustness/year_block_hac_inference.py
```

The employment analysis searches lags from -12 to +12 months and uses
year-block resampling for simultaneous bands. No method crosses the global
simultaneous band in the saved results. Accordingly, the paper treats these
correlations as descriptive checks on seasonal face validity, not as estimator
rankings or validation against observed monthly GERD.

## 7. Country-year robustness

Run the grouped split with the current architecture, evaluate annual results,
and regenerate its appendix artifacts:

```bash
python additional_analysis/cy_current_nn_refresh/run_cy_current_nn.py
python additional_analysis/cy_current_nn_refresh/evaluate_current_nn.py
python additional_analysis/cy_current_nn_refresh/paper_artifacts_current.py
```

The corrected one-step Diebold-Mariano small-sample factor is
`sqrt((n - 1) / n)`.

## 8. Repository checks and paper artifacts

```bash
make check
```

`make check` verifies required artifacts, benchmark rows, Step B adding-up,
portable paths in the primary scripts, the curated paper-artifact inventory,
and GitHub's file-size constraint. The public repository excludes the LaTeX
manuscript source and current revision PDF. The 28 current paper-ready figures
and 8 generated table fragments are retained under `paper/`; the public
manuscript record is available at https://arxiv.org/abs/2407.11765.

## 9. Saved outputs and numerical variation

Neural-network outputs depend on PyTorch and platform versions even when seeds
are fixed. Minor last-decimal differences are therefore possible. The checked-in
CSVs and curated paper artifacts constitute the paper's reported research snapshot.
Large raw downloads, training caches, W&B runs, and model checkpoints are
excluded. They are not necessary to inspect or regenerate the reported tables
from the curated panel.
