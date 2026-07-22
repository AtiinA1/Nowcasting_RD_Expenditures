"""Generate the pre-raw manuscript CCF figure including uniform allocation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parents[2]
PAPER = CODE / "paper"
AUDIT_OUT = CODE / "additional_analysis" / "pre_raw_stepb_method_audit" / "out"


def main():
    data = pd.read_csv(AUDIT_OUT / "uniform_employment_monthly_lags.csv")
    styles = {
        "NN": ("#35658A", "-", "NN sensitivity"),
        "Mosley": ("#5B8E55", "-", "Sparse temporal disaggregation"),
        "Sax": ("#B66A45", "-", "Chow-Lin"),
        "Uniform": ("#666666", "--", "Uniform allocation"),
    }
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for method, (color, linestyle, label) in styles.items():
        subset = data[data.Method.eq(method)].sort_values("Lag")
        ax.plot(
            subset.Lag, subset.Correlation, color=color, linestyle=linestyle,
            marker="o", markersize=3, linewidth=1.3, label=label,
        )
    n = int(data[(data.Method.eq("NN")) & (data.Lag.eq(0))].N.iloc[0])
    band = 1.96 / np.sqrt(n)
    ax.axhspan(-band, band, color="#BDBDBD", alpha=0.25, label="Approximate 95% band")
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_xlim(-12, 12)
    ax.set_xticks(np.arange(-12, 13, 2))
    ax.set_xlabel("Lag k (months): corr(R&D growth at t, employment growth at t+k)")
    ax.set_ylabel("Growth-rate cross-correlation")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    fig.tight_layout()
    output = PAPER / "figures" / "Revision" / "disagg_ccf_with_uniform.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
