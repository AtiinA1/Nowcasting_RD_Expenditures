# AGT-Only NN Deeper Search Notes

Source script: `07_agt_nn_deeper_search.py`

Outputs:

- `out/agt_nn_deeper_search/agt_nn_deeper_search_results.csv`
- `out/agt_nn_deeper_search/agt_nn_deeper_search_predictions.csv`
- `out/agt_nn_deeper_search/agt_nn_deeper_search_history.csv`
- `out/agt_nn_deeper_search/agt_reference_metrics.csv`
- `figures/agt_nn_deeper_search.png`

Main result:

Focusing only on the Step-B-relevant AGT feature space, the best improvement is
a residual-learning NN:

- Variant: `resid_wide64_huber`
- Validation MAPE: 3.24
- Temporal test MAPE: 11.94
- Temporal test RMSE: 64.74
- Temporal test R2: 0.91

This improves strongly over the original AGT NN:

- Existing AGT NN: MAPE 15.62, RMSE 79.30
- Improved pure AGT NN from previous search: MAPE 14.17, RMSE 65.34
- Residual-learning AGT NN: MAPE 11.94, RMSE 64.74

Framework-relevant comparison:

- Residual-learning AGT NN: MAPE 11.94
- RW3: MAPE 12.04
- Ridge AGT: MAPE 14.99
- Elastic Net AGT: MAPE 15.43
- Existing AGT NN: MAPE 15.62
- sg-LASSO-MIDAS: MAPE 22.15
- OLS AGT: MAPE 22.38
- MIDAS/U-MIDAS: MAPE 31--32
- AR3: MAPE 5.18

Interpretation:

The residual-learning AGT NN is the first AGT-family NN variant that slightly
beats the feasible random-walk benchmark on MAPE. It does not beat AR3 and it
does not beat RW3 on RMSE, so the annual-level win should be reported carefully.

The model is a hybrid: a country-specific log trend estimated on the training
period supplies the smooth baseline, and the AGT neural network learns the
search-driven residual around that trend. This is scientifically coherent for a
smooth target like GERD because it asks search data to explain deviations from
trend, not the entire persistent level.

Suggested framing:

For the Step-A-to-Step-B framework, the relevant comparison is AGT-only. Under
that comparison, the residual-learning AGT NN is the strongest AGT model on
percentage error and slightly outperforms the feasible RW3 benchmark on MAPE.
However, AR3 remains much stronger on annual levels, and the residual-learning
setup changes the interpretation of Step B elasticities: the AGT component
primarily captures search-driven deviations around trend. For the current paper,
this is best presented as an additional robustness/improvement experiment rather
than replacing the simpler main specification without further Step B validation.
