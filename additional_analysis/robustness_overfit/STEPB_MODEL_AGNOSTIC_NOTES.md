# Step B Model-Agnostic Robustness Notes

Source script: `04_stepb_model_agnostic_elasticities.py`

Outputs:

- `out/stepb_model_agnostic/stepb_alt_model_summary.csv`
- `out/stepb_model_agnostic/stepb_alt_monthly_estimates.csv`
- `out/stepb_model_agnostic/stepb_alt_top_topics.csv`
- `out/stepb_model_agnostic/stepb_alt_topic_elasticities.csv`
- `figures/stepb_alt_model_elasticities.png`
- `tables/stepb_model_agnostic_elasticities_table.tex`

Main decision:

The results support a model-agnostic Step B framing more strongly than a claim
that the neural network is uniquely required for temporal disaggregation. The
framework can convert different Step A predictors into model-implied topic
elasticities and then use the same search-share allocation rule.

Key findings:

- Ridge is the cleanest alternative benchmark. Its signed elasticity allocation
  remains positive in every month and is highly aligned with the NN monthly
  series: level correlation 0.93 and growth correlation 0.88.
- Ridge with positive-part weights is even closer to the NN series: level
  correlation 0.98 and growth correlation 0.94. It also remains strongly aligned
  with Mosley: level correlation 0.93 and growth correlation 0.63.
- OLS and Elastic Net have many negative topic elasticities. Their signed
  allocations are much less useful, especially Elastic Net, which produces three
  negative monthly estimates. After positive-part or absolute weighting, they
  become much more similar to NN/Mosley, but that requires an additional
  transformation.
- About 42 percent of Ridge and Elastic Net topic elasticities, and 44 percent
  of OLS topic elasticities, are negative. This is substantively important:
  unconstrained high-dimensional linear elasticities are not always directly
  interpretable as positive allocation weights.

Recommended paper framing:

Use the neural network as the main Step A/Step B implementation because it is a
flexible nonlinear predictor and was the original production model, but avoid
claiming that Step B depends uniquely on neural networks. The stronger claim is
that the proposed disaggregation framework is portable across Step A models. In
the robustness analysis, a validation-tuned Ridge model produces very similar
monthly allocations, suggesting that the monthly R&D signal is not an artifact
of one neural-network specification.

Suggested caveat:

For linear models, signed elasticities can be unstable or negative in a way that
is awkward for monthly allocation. Therefore, the paper should report signed
Ridge as the cleanest linear robustness check and treat positive-part or
absolute-weight variants as diagnostic sensitivity checks rather than the main
estimator.
