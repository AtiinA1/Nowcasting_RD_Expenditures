# Nowcasting and Temporal Disaggregation of R&D Expenditures

This repository contains the data snapshots, models, benchmarks, and
robustness analyses for **Nowcasting and Temporal Disaggregation of R&D
Expenditures Using Internet Search Data** by Atin Aboutorabi and Gaetan de
Rassenfosse.

The public manuscript record is available on
[arXiv:2407.11765](https://arxiv.org/abs/2407.11765).

The study has two components:

1. **Step A, annual nowcasting.** Neural networks combine annual GERD,
   macroeconomic variables, and monthly Google Trends information under a
   chronological pseudo-out-of-sample design. The search-only AGT
   specification is the principal input to Step B.
2. **Step B, temporal disaggregation.** Model-implied sensitivities and
   within-year search profiles allocate observed annual GERD totals to months,
   subject to an exact annual adding-up constraint. Classical and sparse
   regression-based disaggregation methods provide benchmarks.

## Start here

The current paper-facing workflow lives in
[`additional_analysis/`](additional_analysis/). The two original implementation
folders are retained to document the development history, but they are not the
recommended starting point for reproducing the reported results.

| Task | Current entry point |
| --- | --- |
| Build the merged analysis panel | [`additional_analysis/out/merged_features.csv`](additional_analysis/out/merged_features.csv) is the checked-in research snapshot |
| Train the aligned Step A neural networks | [`additional_analysis/robustness_overfit/10_all_configs_updated_pure_nn.py`](additional_analysis/robustness_overfit/10_all_configs_updated_pure_nn.py) |
| Run corrected MIDAS benchmarks | [`additional_analysis/pre_raw_stepb_method_audit/leakage_free_midas_benchmarks.py`](additional_analysis/pre_raw_stepb_method_audit/leakage_free_midas_benchmarks.py) |
| Reproduce the principal Step B allocation | [`additional_analysis/robustness_overfit/12_refresh_stepB_updated_agt.py`](additional_analysis/robustness_overfit/12_refresh_stepB_updated_agt.py) |
| Audit both Step B sensitivity definitions | [`additional_analysis/exact_stepa_stepb_audit/`](additional_analysis/exact_stepa_stepb_audit/) |
| Run the country-year robustness design | [`additional_analysis/cy_current_nn_refresh/`](additional_analysis/cy_current_nn_refresh/) |
| Run employment inference | [`additional_analysis/employment_inference_robustness/year_block_hac_inference.py`](additional_analysis/employment_inference_robustness/year_block_hac_inference.py) |

Some filenames retain terms such as `updated`, `refresh`, or `pre_raw` because
they record the revision sequence. The map above identifies their status in the
current paper. In manuscript text, the current AGT implementation is simply
called the neural network, not the "updated" model.

## Repository map

| Path | Contents |
| --- | --- |
| [`additional_analysis/`](additional_analysis/) | **Current:** temporal experiments, benchmarks, Step B analyses, saved results, and robustness checks |
| [`data/`](data/) | **Current inputs:** curated source snapshots used to build the analysis panel |
| [`gt_code/`](gt_code/) | **Data preparation:** Google Trends collection and filtering code |
| [`nn_mlp_nowcasting_model/`](nn_mlp_nowcasting_model/) | **Historical:** original model implementations and W&B hyperparameter-search records |
| [`temporal_disaggregation/`](temporal_disaggregation/) | **Historical/supporting:** original temporal-disaggregation scripts and classical estimates |
| [`paper/`](paper/) | **Public artifacts:** the current paper-ready figures and generated tables; manuscript source remains linked through arXiv |

The compact derived panel used by the paper-facing workflows is
[`additional_analysis/out/merged_features.csv`](additional_analysis/out/merged_features.csv).
Saved result CSVs are retained so that the paper's reported results can be
audited without rerunning every neural-network ensemble.

## Main results

Under the temporal split, the 15-member AGT neural-network ensemble records a
test MAPE of **13.96%**, RMSE of **70.16**, and test-set R-squared of **0.894**.
With training-period normalization, the corresponding MIDAS, U-MIDAS, and
sparse-group-LASSO MIDAS MAPEs are 32.94%, 34.20%, and 42.90%, respectively.
The small annual test sample means these numerical differences should not be
read as definitive model-selection evidence.

Step B preserves annual totals to numerical precision. Its employment
comparisons are descriptive external diagnostics: they do not observe true
monthly GERD and do not identify or rank the most accurate allocation method.

## Quick start

Create the environment and verify the checked-in research artifacts:

```bash
conda env create -f environment.yml
conda activate nowcasting-rd
python scripts/validate_repository.py
```

Run the current Step A ensemble and corrected mixed-frequency benchmarks:

```bash
make step-a
make benchmarks
```

These model runs can take substantial time. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the complete workflow, result
provenance, R dependencies, and interpretation notes.

## Reproducibility scope

The paper-facing scripts use paths relative to the repository. Scripts directly
under `nn_mlp_nowcasting_model/`, `temporal_disaggregation/`, and the root of
`additional_analysis/` include development-stage implementations. Some retain
historical random splits, older absolute paths, or superseded model variants;
their folder guides state when they are retained only for provenance. In
particular, naive observation-level random splits can place observations from
the same country-year in both training and test sets and should not be used for
the paper's headline evaluation.

Google Trends series are cached because downloads may vary with sampling,
normalization, and retrieval date. Large USPTO and patent-classification files
from the initial repository are not required by the current paper pipeline and
are intentionally excluded from this research snapshot.

The public repository intentionally excludes the revision's LaTeX source,
bibliography, internal version notes, and compiled draft. The curated figures
and generated table fragments used by the current manuscript remain available
under [`paper/`](paper/), while the manuscript itself is linked through arXiv.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite
the public manuscript record on [arXiv](https://arxiv.org/abs/2407.11765).
