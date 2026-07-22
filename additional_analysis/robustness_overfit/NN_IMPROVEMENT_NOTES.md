# Neural-Network Improvement Search Notes

Source script: `06_nn_improvement_search.py`

Outputs:

- `out/nn_improvement_search/nn_improvement_search_results.csv`
- `out/nn_improvement_search/nn_improvement_search_predictions.csv`
- `out/nn_improvement_search/nn_improvement_search_history.csv`
- `out/nn_improvement_search/nn_improvement_reference_metrics.csv`
- `out/nn_improvement_search/nn_improvement_variant_ensembles.csv`
- `figures/nn_improvement_search_best.png`
- `tables/nn_improvement_search_table.tex`

Main result:

The original neural network can be improved materially under the same temporal
split, but the improved NN still does not beat the strongest autoregressive
trend benchmark for annual GERD levels. This is consistent with the paper's
current framing: GERD is very smooth, and annual level prediction is dominated
by persistence.

Best improved NN variants:

- `AGT`: `wide_deep_huber`, MAPE 14.17, RMSE 65.34, R2 0.91.
- `AGTwRD`: `medium_ln_mse`, MAPE 12.41, RMSE 58.25, R2 0.93.
- `AllVar`: `wide_deep_huber`, MAPE 12.50, RMSE 58.25, R2 0.93.
- `Macros`: `wide_deep_small`, MAPE 14.43, RMSE 64.80, R2 0.91.

Comparison to existing models:

- Existing `AGT` NN: MAPE 15.62, RMSE 79.30.
- Improved `AGT` NN: MAPE 14.17, RMSE 65.34.
- Existing `AGTwRD` NN: MAPE 14.84, RMSE 69.33.
- Improved `AGTwRD` NN: MAPE 12.41, RMSE 58.25.
- Existing `AllVar` NN: MAPE 14.76, RMSE 69.21.
- Improved `AllVar` NN: MAPE 12.50, RMSE 58.25.

What did not happen:

- The improved NN does not beat AR(1) with a realistic three-year publication
  lag (`AR3`, MAPE 5.18) on annual levels.
- The improved NN nearly matches but does not clearly beat the feasible random
  walk benchmark (`RW3`, MAPE 12.04).
- In `AllVar`, Elastic Net remains stronger on MAPE (9.79) than the improved
  NN (12.50), although the improved NN closes much of the gap relative to the
  original architecture.

Interpretation:

The best architectural direction is smaller, more regularized networks with
LayerNorm, and especially wide-and-deep models that preserve a linear skip path.
This supports the idea that the data benefit from a stabilized linear component
plus optional nonlinear residual learning, rather than from simply increasing
neural-network capacity.

Suggested framing:

If updating the paper, we can say that targeted NN regularization improves the
neural-network Step A results substantially, especially for `AGT`, `AGTwRD`, and
`AllVar`. However, the improved results reinforce rather than overturn the
central interpretation: the annual GERD level is so persistent that
autoregressive trend benchmarks remain difficult to beat, and the main value of
the search-based Step A model is timeliness, robustness, and supplying
elasticities for Step B.
