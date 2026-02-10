"""
Reproduce Figure 4 from Gibbs & Candès (2021), Appendix A.2.

Daily open prices for the four stocks used in Section 2.2:
Nvidia, AMD, BlackBerry, Fannie Mae.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

_EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))

from common.data import fetch_stock_data

STOCKS = {
    "Nvidia": {"ticker": "NVDA", "start": "1999-01-01", "end": "2020-12-31"},
    "AMD": {"ticker": "AMD", "start": "1994-01-01", "end": "2020-12-31"},
    "BlackBerry": {"ticker": "BB", "start": "1999-01-01", "end": "2020-12-31"},
    "Fannie Mae": {"ticker": "FNMA", "start": "1994-01-01", "end": "2020-12-31"},
}

X_LIMITS = {
    "Nvidia": ("1999-01-01", "2020-12-31"),
    "AMD": ("1994-01-01", "2020-12-31"),
    "BlackBerry": ("1999-01-01", "2020-12-31"),
    "Fannie Mae": ("1994-01-01", "2020-12-31"),
}


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.labelsize": 17,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, (ax, (name, config)) in enumerate(zip(axes, STOCKS.items())):
        print(f"Processing {name} ({config['ticker']})")
        data = fetch_stock_data(
            config["ticker"],
            config["start"],
            config["end"],
        )
        dates = data["dates"]
        prices = data["prices"][1:]

        ax.set_facecolor("#EBEBEB")
        ax.plot(dates, prices, color="#6b6b6b", linewidth=1.1)

        xmin, xmax = X_LIMITS[name]
        ax.set_xlim(np.datetime64(xmin), np.datetime64(xmax))
        years = (2000, 2005, 2010, 2015, 2020)
        xticks = [np.datetime64(f"{year}-01-01") for year in years]
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        if i % 2 == 0:
            ax.set_ylabel("Open Price")

        ax.text(
            0.02,
            0.96,
            name,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=15,
            fontweight="bold",
        )

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True, color="white", linewidth=1.1)
        ax.minorticks_off()
        ax.tick_params(axis="x", rotation=0, top=False)
        ax.tick_params(axis="y", right=False)

    fig.supxlabel("Time", y=0.07, fontsize=19)
    fig.tight_layout(rect=[0.03, 0.09, 0.995, 0.98])
    output_path = "figures/figure4.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
