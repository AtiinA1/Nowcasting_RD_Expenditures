# Pre-raw Step B methodological audit

This folder contains isolated reruns requested during the review of the
pre-raw Step B manuscript. It does not overwrite existing experiments or paper
artifacts.

- `leakage_free_midas_benchmarks.py` reruns composite MIDAS, U-MIDAS, and
  sg-LASSO-MIDAS after estimating every Google Trends normalization constant
  from the chronological training period only.
- `uniform_employment_benchmark.py` compares a uniform `1/12` allocation with
  the current Step B estimators on the same employment overlap, growth-rate
  definition, and lag convention.

All generated files are written to `out/`.
