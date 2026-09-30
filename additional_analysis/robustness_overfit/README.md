# Robustness analyses for overfitting and small-sample concerns

This folder is additive and self-contained. It reads the existing paper inputs
from `Nowcasting_github/additional_analysis/out/` but writes all new outputs
inside this folder.

## Scripts

- `01_training_diagnostics.py`
  - Retrains AGT and AllVar under the temporal split.
  - Saves train/validation loss histories and a loss-curve figure.

- `02_regularized_linear_benchmarks.py`
  - Adds validation-tuned Ridge and Elastic Net benchmarks on the same feature
    spaces as the neural network.
  - Saves CSV results and a standalone LaTeX table.

- `03_rolling_origin_sensitivity.py`
  - Runs a lightweight rolling-origin sensitivity check for AGT.
  - Compares NN-AGT, Elastic Net, and RW(3) over three origin years.

- `14_regenerate_fair_fig2_candidates.py` and `19_stepA_plot_with_ridge.py`
  - Apply the matched final-refit protocol after validation selects each
    ensemble member's training duration.
  - Compare the neural network with centered minimum-norm OLS and
    validation-tuned Ridge on the same estimation sample.

- `16_regenerate_stepB_matched_refit.py` and `18_stepB_gaussian_relevance.py`
  - Refit the AGT Step A model on training and validation observations.
  - Generate the principal Step B allocation using Gaussian perturbation
    relevance and check its stability across perturbation scales.

- `20_stepB_gaussian_linear_checks.py`
  - Applies the same Gaussian-relevance procedure to matched OLS and Ridge
    models.

- `run_all.py`
  - Runs the first three diagnostic analyses in sequence.

## Outputs

- `out/`
  - Raw CSV results, including `paper_stepA_refresh/` and
    `stepB_matched_refit/` for the current paper-facing analyses.

- `figures/`
  - `training_diagnostics_loss_curves.png`
  - `rolling_origin_sensitivity.png`

- `tables/`
  - `regularized_linear_benchmarks_table.tex`
  - `rolling_origin_sensitivity_table.tex`

## Interpretation

These analyses are designed as robustness checks rather than replacements for
the main benchmark table. They address reviewer-facing concerns about
overfitting by documenting training behavior, comparing against validation-tuned
regularized linear models, and checking whether the AGT result is sensitive to a
single temporal split.
