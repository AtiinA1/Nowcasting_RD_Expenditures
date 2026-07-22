# Exact Step A to Step B audit

This folder is additive and does not overwrite the paper or existing results.

1. `run_audit.py` retrains the exact reported temporal AGT neural-network
   ensemble, computes raw-scale elasticities and standardized-input predictive
   sensitivities, and derives the current topic-level SHAP ranking.
2. `run_chowlin_refresh.R` reproduces the old six-indicator Chow-Lin model and
   fits the same specification with the current top-six SHAP topics.
3. `correct_country_year_dm.py` recomputes the country-year DM statistics with
   the one-step Harvey correction factor `sqrt((n - 1) / n)`.

All generated files are stored in `out/`.
