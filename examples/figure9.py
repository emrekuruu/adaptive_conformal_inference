"""
Reproduce Figure 9 from Gibbs & Candès (2021), Appendix A.4.

Local coverage frequencies for adaptive conformal (blue) and fixed-alpha (red)
for volatility prediction using the **normalized** conformity score on 8
additional stocks/indices.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path
import sys

_EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))

from common.data import local_coverage, fetch_stock_data
from common.volatility_pipeline import garch_conformal_forecasting

# Paper parameters (same as Figures 1 & 2)
ALPHA = 0.1
GAMMA = 0.005
LOOKBACK = 1250
COVERAGE_WINDOW = 500
PLOT_START = "2008-01-01"
PLOT_END = "2020-12-31"

# 8 additional stocks from Appendix A.4, Figure 9.
STOCKS = {
    "S&P 500": {"ticker": "^GSPC", "start": "2001-01-01", "end": "2020-12-31"},
    "Nasdaq": {"ticker": "^IXIC", "start": "2001-01-01", "end": "2020-12-31"},
    "Dow Jones": {"ticker": "^DJI", "start": "2001-01-01", "end": "2020-12-31"},
    "Apple": {"ticker": "AAPL", "start": "2001-01-01", "end": "2020-12-31"},
    "Illumina": {"ticker": "ILMN", "start": "2000-07-01", "end": "2020-12-31"},
    "Exxon Mobil": {"ticker": "XOM", "start": "2001-01-01", "end": "2020-12-31"},
    "Toyota": {"ticker": "TM", "start": "2001-01-01", "end": "2020-12-31"},
    "Goldman Sachs": {"ticker": "GS", "start": "2001-01-01", "end": "2020-12-31"},
}

def run_single_stock(name: str, config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Processing {name} ({config['ticker']})")
    print(f"{'='*60}")

    data = fetch_stock_data(
        config["ticker"],
        config["start"],
        config["end"],
    )
    returns = data["returns"]
    dates = data["dates"]
    print(f"  Downloaded {len(returns)} daily returns")

    _, err_aci, err_fixed = garch_conformal_forecasting(
        returns,
        alpha=ALPHA,
        gamma=GAMMA,
        lookback=LOOKBACK,
        update_method="simple",
        score="normalized",
    )

    # Compute local coverage
    cov_aci = local_coverage(err_aci, window=COVERAGE_WINDOW)
    cov_fixed = local_coverage(err_fixed, window=COVERAGE_WINDOW)

    # Align dates
    start_up = max(100, LOOKBACK)
    aci_dates = dates[start_up - 1:]
    offset = (COVERAGE_WINDOW - 1) // 2
    cov_dates = aci_dates[offset: offset + len(cov_aci)]

    return {
        "name": name,
        "cov_aci": cov_aci,
        "cov_fixed": cov_fixed,
        "cov_dates": cov_dates,
        "avg_cov_aci": 1 - err_aci.mean(),
        "avg_cov_fixed": 1 - err_fixed.mean(),
    }


def plot_results(results: list[dict], output_path: str):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 17,
        "axes.labelsize": 17,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
    })

    fig, axes = plt.subplots(4, 2, figsize=(12.8, 17.6), sharex=False, sharey=True)
    axes = axes.flatten()

    for i, (ax, res) in enumerate(zip(axes, results)):
        dates = res["cov_dates"]
        ax.set_facecolor("#EBEBEB")
        ax.plot(dates, res["cov_aci"], color="#1f3aff", linewidth=1.1)
        ax.plot(dates, res["cov_fixed"], color="#ff3b30", linewidth=1.1)

        ax.axhline(res["avg_cov_aci"], color="#1f3aff", linestyle=(0, (4, 4)), linewidth=2.0)
        ax.axhline(res["avg_cov_fixed"], color="#ff3b30", linestyle=(0, (4, 4)), linewidth=2.0)
        ax.axhline(1 - ALPHA, color="black", linestyle=(0, (4, 4)), linewidth=2.0)

        ax.text(
            0.02, 0.96, res["name"],
            transform=ax.transAxes, va="top", ha="left",
            fontsize=15, fontweight="bold",
        )
        if i % 2 == 0:
            ax.set_ylabel("Local Coverage Level")
        ax.set_ylim(0.84, 0.96)
        ax.set_yticks([0.84, 0.88, 0.92, 0.96])
        ax.set_xlim(np.datetime64(PLOT_START), np.datetime64(PLOT_END))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True, color="white", linewidth=1.1)
        ax.minorticks_off()
        ax.tick_params(axis="x", rotation=0, top=False)
        ax.tick_params(axis="y", right=False)

    legend_handles = [
        Line2D([0], [0], color="#1f3aff", lw=2, label="Adaptive Alpha"),
        Line2D([0], [0], color="#ff3b30", lw=2, label="Fixed Alpha"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.99),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.supxlabel("Time", y=0.04, fontsize=19)
    fig.tight_layout(rect=[0.03, 0.05, 0.995, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    all_results = []
    for name, config in STOCKS.items():
        result = run_single_stock(name, config)
        all_results.append(result)
        print(f"  Avg coverage — ACI: {result['avg_cov_aci']:.3f}, "
              f"Fixed: {result['avg_cov_fixed']:.3f}")

    plot_results(all_results, "figures/figure9.png")
