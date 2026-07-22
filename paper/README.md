# Paper-ready artifacts

The public repository intentionally contains the current paper-ready artifacts,
but not the revision's LaTeX source, bibliography, internal notes, or compiled
draft. The public manuscript record is
[arXiv:2407.11765](https://arxiv.org/abs/2407.11765).

| Path | Contents |
| --- | --- |
| `figures/` | The 28 main-text and appendix figures referenced by the current manuscript |
| `tables/` | The 8 generated table fragments referenced by the current manuscript |

The figures cover the framework, Step A performance and diagnostics, Step B
monthly allocations, classical temporal-disaggregation comparisons,
employment diagnostics, calibration, seasonality, and country-year robustness.
The tables report the temporal and country-year benchmarks and coverage checks,
plus the model-agnostic, quarterly, sensitivity-definition, and uniform Step B
analyses.

The generating scripts are identified in
[`../additional_analysis/README.md`](../additional_analysis/README.md). Some of
those scripts also produce exploratory figures; only artifacts referenced by
the current manuscript are committed here.
