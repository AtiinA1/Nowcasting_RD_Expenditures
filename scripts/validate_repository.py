#!/usr/bin/env python3
"""Validate the public research snapshot and primary reproducibility paths."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_BYTES = 100 * 1024 * 1024

REQUIRED = [
    "paper/README.md",
    "paper/Nowcasting_Temporal_Disaggregation_RD.pdf",
    "paper/figures/Framework_NowcastingRD2024.pdf",
    "paper/figures/Nowcast_Model_Temporal/NNvsOLS_Temporal_distribution.png",
    "paper/figures/Disaggregation/NNelasticity_temporal_level.png",
    "paper/tables/temporal_benchmarks_table.tex",
    "paper/tables/stepb_sensitivity_definition_table.tex",
    "additional_analysis/out/merged_features.csv",
    "additional_analysis/robustness_overfit/out/all_configs_updated_pure_nn/all_configs_updated_pure_nn_results.csv",
    "additional_analysis/pre_raw_stepb_method_audit/out/leakage_free_midas_metrics.csv",
    "additional_analysis/exact_stepa_stepb_audit/out/annual_adding_up_check.csv",
    "additional_analysis/employment_inference_robustness/out/strongest_lag_by_method.csv",
]

PUBLIC_EXCLUSIONS = {
    "paper/template-arxiv.tex",
    "paper/annex-arxiv.tex",
    "paper/references.bib",
    "paper/arxiv.sty",
    "paper/VERSION_NOTE.md",
    "paper/template-arxiv.pdf",
}

PRIMARY_SCRIPTS = [
    "additional_analysis/robustness_overfit/common.py",
    "additional_analysis/robustness_overfit/10_all_configs_updated_pure_nn.py",
    "additional_analysis/robustness_overfit/11_refresh_stepA_paper_artifacts.py",
    "additional_analysis/robustness_overfit/12_refresh_stepB_updated_agt.py",
    "additional_analysis/robustness_overfit/13_fig2_distribution_refresh.py",
    "additional_analysis/pre_raw_stepb_method_audit/leakage_free_midas_benchmarks.py",
    "additional_analysis/pre_raw_stepb_method_audit/leakage_free_cy_midas.py",
    "additional_analysis/cy_current_nn_refresh/run_cy_current_nn.py",
    "additional_analysis/cy_current_nn_refresh/evaluate_current_nn.py",
    "additional_analysis/exact_stepa_stepb_audit/run_audit.py",
    "additional_analysis/employment_inference_robustness/year_block_hac_inference.py",
]


def read_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"ERROR: {message}")


def main() -> int:
    errors: list[str] = []

    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines()
    )

    for relative in sorted(PUBLIC_EXCLUSIONS & tracked):
        fail(f"private manuscript material is tracked: {relative}", errors)

    tracked_figures = {path for path in tracked if path.startswith("paper/figures/")}
    tracked_tables = {path for path in tracked if path.startswith("paper/tables/")}
    if len(tracked_figures) != 28:
        fail(f"expected 28 curated paper figures, found {len(tracked_figures)}", errors)
    if len(tracked_tables) != 8:
        fail(f"expected 8 generated paper tables, found {len(tracked_tables)}", errors)

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            fail(f"missing required artifact: {relative}", errors)

    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.stat().st_size >= MAX_GITHUB_BYTES:
            fail(f"file exceeds GitHub's 100 MiB limit: {path.relative_to(ROOT)}", errors)

    for relative in PRIMARY_SCRIPTS:
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text or "Nowcasting_github" in text:
            fail(f"machine-specific path remains in primary script: {relative}", errors)

    if not errors:
        nn_rows = read_rows(
            "additional_analysis/robustness_overfit/out/all_configs_updated_pure_nn/"
            "all_configs_updated_pure_nn_results.csv"
        )
        agt = [row for row in nn_rows if row["Config"] == "AGT" and row["ensemble_size"] == "15"]
        if len(agt) != 1 or abs(float(agt[0]["test_MAPE"]) - 13.959167) > 0.01:
            fail("the saved 15-member AGT result does not match the paper snapshot", errors)

        benchmark_rows = read_rows(
            "additional_analysis/pre_raw_stepb_method_audit/out/leakage_free_midas_metrics.csv"
        )
        required_models = {
            "AGT neural network",
            "MIDAS, corrected",
            "U-MIDAS, corrected",
            "sg-LASSO-MIDAS, corrected",
        }
        found_models = {row["Model"] for row in benchmark_rows}
        if not required_models.issubset(found_models):
            fail("corrected Step A benchmark rows are incomplete", errors)

        adding_up = read_rows("additional_analysis/exact_stepa_stepb_audit/out/annual_adding_up_check.csv")
        maximum_gap = max(abs(float(row["difference"])) for row in adding_up)
        if maximum_gap > 1e-6:
            fail(f"Step B annual adding-up gap is {maximum_gap:.3g}", errors)

    if errors:
        print(f"\nRepository validation failed with {len(errors)} error(s).")
        return 1

    print("Repository validation passed.")
    print("AGT temporal test MAPE: 13.96%")
    print("Step B maximum annual adding-up gap: < 1e-6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
