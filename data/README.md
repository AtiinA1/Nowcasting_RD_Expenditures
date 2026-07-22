# Data guide

This directory contains the curated snapshots used to construct the paper's
analysis panel.

| Directory | Contents |
| --- | --- |
| `GERD/` | Annual gross domestic expenditure on R&D inputs |
| `GT/` | Monthly Google Trends topic panels and topic metadata |
| `IMF/` | April 2023 World Economic Outlook CSV snapshot |
| `OECD/` | GERD, R&D personnel, and education-finance extracts used by the analysis |
| `FRED/` | GDP and real-GDP series used in data preparation |
| `WB/` | World Bank R&D and GDP-derived inputs |
| `datausa.io/` | Monthly employment in scientific R&D services used as an external diagnostic |

`country_code.csv` contains country-code mappings.

The primary reproducibility scripts use the derived panel at
`additional_analysis/out/merged_features.csv`. Source snapshots are retained to
make the transformations inspectable. Google Trends values can vary across
retrievals because of Google sampling and normalization, so the cached files
are the relevant research vintage.

The original repository also contained multi-gigabyte USPTO tables and a large
OECD patent-classification extract. Those files are not used by the current
paper-facing models or tables and are excluded here to keep the repository
cloneable. The `.gitignore` prevents accidental recommits of those raw files.

