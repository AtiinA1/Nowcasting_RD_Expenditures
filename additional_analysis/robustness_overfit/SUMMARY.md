# Final decisions and interpretation

## What was added

All outputs are contained in this folder and no existing paper or analysis files
were modified.

1. Training diagnostics
   - Script: `01_training_diagnostics.py`
   - CSVs:
     - `out/training_diagnostics_loss_history.csv`
     - `out/training_diagnostics_summary.csv`
   - Figure:
     - `figures/training_diagnostics_loss_curves.png`

2. Regularized linear benchmarks
   - Script: `02_regularized_linear_benchmarks.py`
   - CSV:
     - `out/regularized_linear_benchmarks.csv`
   - LaTeX:
     - `tables/regularized_linear_benchmarks_table.tex`

3. Rolling-origin sensitivity
   - Script: `03_rolling_origin_sensitivity.py`
   - CSVs:
     - `out/rolling_origin_sensitivity.csv`
     - `out/rolling_origin_predictions.csv`
   - Figure:
     - `figures/rolling_origin_sensitivity.png`
   - LaTeX:
     - `tables/rolling_origin_sensitivity_table.tex`

## Main findings

### Training diagnostics

The diagnostic retraining exercise reproduces the main temporal-split NN results
closely with a smaller 5-member ensemble:

- AGT: 15.61% MAPE, mean early-stop epoch 90.8.
- AllVar: 14.61% MAPE, mean early-stop epoch 87.0.

The train/validation curves and early stopping behavior support the claim that
the NN is not simply trained to convergence on the training sample. This is good
evidence for an appendix figure.

### Regularized linear benchmarks

Validation-tuned Ridge and Elastic Net are much stronger than the OLS-like
same-input linear benchmark. This changes the safest interpretation:

- The NN clearly improves over unregularized OLS in high-dimensional Google
  Trends spaces.
- However, properly regularized linear models can be competitive or better on
  annual levels and MAPE.
- The NN remains useful as the nonlinear model from which Step B extracts
  model-implied elasticities, but the paper should avoid claiming broad
  dominance over all regularized linear alternatives.

Selected examples from the temporal test set:

- AGT:
  - NN: 15.62% MAPE, RMSE 79.30.
  - Ridge: 14.99% MAPE, RMSE 97.69.
  - Elastic Net: 15.43% MAPE, RMSE 122.84.
  - sg-LASSO-MIDAS: 22.15% MAPE, RMSE 77.03.

- AllVar:
  - NN: 14.76% MAPE, RMSE 69.21.
  - Ridge: 11.56% MAPE, RMSE 70.17.
  - Elastic Net: 9.79% MAPE, RMSE 52.92.

This is informative but should be discussed carefully because the annual test
set is only 23 country-years.

### Rolling-origin sensitivity

The AGT rolling-origin check shows that NN-AGT is not an artifact of one split,
but it also confirms that regularized linear models are strong:

- 2017 origin:
  - NN-AGT: 10.16% MAPE.
  - Elastic Net: 5.31% MAPE.
  - RW(3): 10.19% MAPE.

- 2018 origin:
  - NN-AGT: 13.93% MAPE.
  - Elastic Net: 9.93% MAPE.
  - RW(3): 14.04% MAPE.

- 2019 origin:
  - NN-AGT: 8.50% MAPE.
  - Elastic Net: 2.51% MAPE.
  - RW(3): 13.56% MAPE.

Switzerland is excluded from these rolling-origin folds because its early annual
history is too sparse for a meaningful train/validation/test construction.

## Recommended paper use

1. Add the training diagnostic figure to the appendix.
   - Use it to support the overfitting mitigation statement.

2. Add the regularized linear benchmark table to the appendix.
   - In the main text, say that validation-tuned Ridge/Elastic Net are strong
     and sometimes outperform the NN on annual levels.
   - Reframe the NN contribution as: it improves over unregularized OLS and
     parsimonious MIDAS, remains competitive with high-dimensional regularized
     alternatives, and supplies the elasticities required for Step B.

3. Add the rolling-origin table or figure to the appendix.
   - Use it as a sensitivity check, not as a definitive cross-validation study.
   - Emphasize small fold sizes.

## Suggested wording

The additional checks reinforce the need for a cautious interpretation. The
neural network is not merely overfitting the training sample: early stopping
occurs well before the epoch cap and the temporal/rolling-origin test errors
remain stable. At the same time, validation-tuned regularized linear models
recover a substantial share of the high-dimensional search signal and in some
configurations outperform the network on annual levels. We therefore interpret
Step A as establishing that high-dimensional search data contain recoverable
nowcasting information, rather than as proving unconditional dominance of neural
networks over all regularized linear alternatives. The distinctive role of the
network in the paper remains its ability to provide model-implied nonlinear
elasticities for the temporal-disaggregation step.

