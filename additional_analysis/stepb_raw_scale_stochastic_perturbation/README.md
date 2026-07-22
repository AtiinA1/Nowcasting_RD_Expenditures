# Stochastic raw-scale Step B perturbation

This isolated robustness experiment compares the current fixed 1% raw-scale finite difference with 20 perturbations drawn from a normal distribution with mean 0.01 and standard deviation 0.005. The same temporal AGT ensemble is used for both estimators. Near-zero draws are resampled to avoid numerical instability when dividing by the perturbation magnitude.

Run:

```bash
python stochastic_raw_scale.py
```

All outputs are written to `out/`. Existing analysis and paper files are not modified.

## Result

The stochastic estimator is effectively identical to the fixed 1% estimator in this sample. Their monthly level and growth correlations both exceed 0.9999999, and the mean absolute difference is 0.00118 USD billion per month. Mosley, Chow-Lin, quarterly-employment, topic-ranking, and annual-add-up diagnostics are unchanged to the reported precision. Sampling around 1% therefore adds Monte Carlo computation but does not improve alignment with the comparison estimators.
