# Raw-scale elasticity correction: results

The corrected implementation perturbs one lag-specific Google Trends feature
by 1 percent on its original scale and then applies the StandardScaler fitted
on the training observations. Existing paper files and paper-facing outputs
were not changed.

## Primary temporal NN results

- Corrected versus previous NN series: level correlation 0.907, monthly-growth
  correlation 0.874, mean absolute difference USD 3.400 billion, mean absolute
  percentage difference 8.16 percent.
- Corrected NN versus Mosley: level correlation 0.758, monthly-growth
  correlation 0.159 (p = 0.019).
- Corrected NN versus Chow-Lin: level correlation 0.664, monthly-growth
  correlation -0.069 (p = 0.315).
- All corrected monthly NN estimates are positive.
- Maximum annual adding-up error is 5.68e-14, numerical zero.

## Topic weights and split stability

- Corrected and previous topic elasticities correlate 0.864 in signed values
  and 0.869 in absolute values.
- Elasticity signs agree for 84.2 percent of topics, and 7 of the leading 10
  topics overlap.
- Leading corrected temporal topics are Subsidy, Business loan, Trademark
  attorney, Research and Experimentation Tax Credit, Patent infringement, and
  Capitalization.
- Corrected temporal versus random monthly estimates correlate 0.931 in levels
  and 0.858 in growth. Corrected temporal versus all-data estimates correlate
  0.929 in levels and 0.854 in growth.

## Quarterly results

- Corrected NN versus Mosley: quarterly level correlation 0.872 and quarterly
  growth correlation 0.085 (p = 0.480).
- Corrected NN versus Chow-Lin: quarterly level correlation 0.893 and quarterly
  growth correlation 0.240 (p = 0.044).
- NN monthly growth standard deviation falls from 0.232 to 0.143 after
  quarterly aggregation.
- Corrected NN quarterly growth correlates 0.345 with R&D-services employment
  growth (p = 0.0098, 55 growth observations).

## Corrected linear robustness

- Signed Ridge remains positive and correlates 0.834 with the corrected NN in
  levels and 0.744 in monthly growth. These values are lower than the previous
  standardized-perturbation comparison of 0.958 and 0.934.
- Positive-part Ridge correlates 0.791 with the corrected NN in levels and
  0.385 in growth, and 0.933 with Mosley in levels and 0.670 in growth.
- Signed OLS produces 13 negative monthly estimates. Signed Elastic Net
  produces 26 negative monthly estimates. Their positive-part and absolute
  variants remain positive but require an additional sign transformation.

## Interpretation

The corrected estimator remains well-defined in this sample and retains
economically sensible topic weights, exact annual aggregation, positivity, and
strong stability across fitting designs. However, the previous implementation
materially overstated agreement between the NN allocation and Mosley in monthly
and quarterly growth rates, and overstated the similarity between NN and Ridge
allocations. If the paper retains the raw-scale elasticity definition, Step B
must be refreshed with these corrected outputs and the empirical claims must be
revised accordingly.
