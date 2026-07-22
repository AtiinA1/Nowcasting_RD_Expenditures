# Original temporal-disaggregation implementation

This folder retains the original Step B implementation and the classical
temporal-disaggregation estimates used during development.

| Path | Role |
| --- | --- |
| `MLP_AGT_4TempDisagg.py` | Original neural-network elasticity allocation |
| `classical_regression_tempdisagg/` | Chow-Lin and sparse temporal-disaggregation estimates |
| `temp_disagg_combined_data.py` | Combines the original monthly estimators |
| `temp_disagg_analysis_employee.py` | Original employment comparison |
| `results/` | Historical combined estimates and figures |

The current Step B implementation, sensitivity definitions, quarterly checks,
uniform benchmark, and corrected employment inference are under
[`../additional_analysis/`](../additional_analysis/). Use the current entry
points listed in [`../additional_analysis/README.md`](../additional_analysis/README.md)
for the results reported in the revised paper.

