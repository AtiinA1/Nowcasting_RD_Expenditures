"""Render Step B candidate panels for symmetric, absolute relevance weights."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "out" / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROBUST = Path(__file__).resolve().parent
INPUT = ROBUST / "out" / "stepB_matched_refit" / "stepB_perturbation_grid_monthly_series.csv"
CODE = Path(__file__).resolve().parents[2]
ASSETS = Path(
    os.environ.get(
        "NOWCASTING_DECK_ASSETS",
        CODE / "paper" / "figures" / "UpdatedResults",
    )
)
STRATEGY = "central_standardized_0.10sd_mean_abs_local"
TRANSFORM = "signed"


def main() -> None:
    data = pd.read_csv(INPUT)
    frame = data[
        data.strategy.eq(STRATEGY) & data.weight_transform.eq(TRANSFORM)
    ].copy()
    if len(frame) != 216:
        raise RuntimeError(f"Expected 216 monthly observations, found {len(frame)}")

    frame.rename(columns={"estimate": "NN"}, inplace=True)
    frame["date"] = pd.to_datetime(dict(year=frame.Year, month=frame.Month, day=1))
    lower = frame[["NN", "Sax", "Mosley"]].min(axis=1)
    upper = frame[["NN", "Sax", "Mosley"]].max(axis=1)

    panels = [
        (
            "NN",
            "#2c6fbb",
            "NN-driven relevance-based estimate (AGT)",
            "NN relevance",
            "c1_disagg_nn_matched_refit_symmetric_abs.png",
        ),
        (
            "Mosley",
            "green",
            "Sparse temporal disaggregation (Mosley)",
            "Sparse (Mosley)",
            "c1_disagg_mosley_matched_refit_symmetric_abs.png",
        ),
        (
            "Sax",
            "purple",
            "Chow-Lin temporal disaggregation (Sax)",
            "Chow-Lin (Sax)",
            "c1_disagg_sax_matched_refit_symmetric_abs.png",
        ),
    ]

    ASSETS.mkdir(parents=True, exist_ok=True)
    for column, color, title, label, filename in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(
            frame.date,
            lower,
            upper,
            color="gray",
            alpha=0.30,
            label="Range across methods",
        )
        ax.plot(
            frame.date,
            frame[column],
            color=color,
            marker="o",
            ms=3,
            lw=1,
            label=label,
        )
        ax.set_ylabel("Monthly R&D (USD bn)")
        ax.set_title(title)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        output = ASSETS / filename
        fig.savefig(output, dpi=220)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
