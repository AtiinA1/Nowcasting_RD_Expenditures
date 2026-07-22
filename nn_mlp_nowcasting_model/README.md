# Original neural-network implementations

This folder preserves the original neural-network code and the subsequent
Weights & Biases hyperparameter-search infrastructure. It documents the model's
development, but it is **not the primary entry point for the current paper**.

For the aligned architecture, chronological pseudo-out-of-sample evaluation,
current benchmarks, and paper-reported metrics, start with:

```text
additional_analysis/robustness_overfit/10_all_configs_updated_pure_nn.py
```

## Contents

| Files | Status |
| --- | --- |
| `MLP_AGT.py`, `MLP_MGT.py`, and related configuration scripts | Original observation-level implementations, retained for provenance |
| `*_temporalsplit.py` | First chronological-split adaptations of the original architecture |
| `*_temporalsplit_wandb.py` and `sweep_*.yaml` | W&B sweep infrastructure used during model development |
| `best_configs/`, `top_configs/`, and `fixed_results/` | Saved hyperparameter-search summaries |

The current paper architecture differs from the original three-layer,
BatchNorm-based scripts in this folder. It uses hidden widths 64 and 16, SiLU
activation, layer normalization, dropout, AdamW, Huber loss, a country
embedding, a linear skip connection, and a 15-member ensemble.

## Historical caveats

Some scripts in this folder retain machine-specific paths or expect raw data
that were part of the development environment but are not needed by the current
paper workflow. Large checkpoints, W&B caches, and obsolete generated results
are intentionally excluded from the public repository. Do not use these files
to reproduce the headline tables unless a historical replication is the goal.

The complete current run order is documented in
[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

