"""
Reproduce Figure 5 from Gibbs & Candès (2021), Appendix A.3.

Realized alpha_t trajectories for stock volatility prediction (Section 2.2)
using update (2) a.k.a. the simple update.
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
from common.volatility_pipeline import garch_conformal_forecasting

ALPHA = 0.1
GAMMA = 0.005
LOOKBACK = 1250
GARCH_P = 1
GARCH_Q = 1
START_UP = 100

STOCKS = {
    "Nvidia": {"ticker": "NVDA", "start": "1999-01-01", "end": "2020-12-31"},
    "AMD": {"ticker": "AMD", "start": "1994-01-01", "end": "2020-12-31"},
    "BlackBerry": {"ticker": "BB", "start": "1999-01-01", "end": "2020-12-31"},
    "Fannie Mae": {"ticker": "FNMA", "start": "1994-01-01", "end": "2020-12-31"},
}

X_LIMITS = {
    "Nvidia": ("2004-06-01", "2020-12-31"),
    "AMD": ("1999-01-01", "2020-12-31"),
    "BlackBerry": ("2004-06-01", "2020-12-31"),
    "Fannie Mae": ("1999-01-01", "2020-12-31"),
}


def run_single_stock(name: str, config: dict) -> dict:
    print(f"Processing {name} ({config['ticker']})")
    data = fetch_stock_data(
        config["ticker"],
        config["start"],
        config["end"],
    )
    returns = data["returns"]
    dates = data["dates"]

    alpha_seq, _, _ = garch_conformal_forecasting(
        returns,
        alpha=ALPHA,
        gamma=GAMMA,
        lookback=LOOKBACK,
        garch_p=GARCH_P,
        garch_q=GARCH_Q,
        start_up=START_UP,
        update_method="simple",
        score="normalized",
    )

    start_up = max(START_UP, LOOKBACK)
    alpha_dates = dates[start_up - 1:]

    return {
        "name": name,
        "alpha_seq": alpha_seq,
        "alpha_dates": alpha_dates,
    }


def plot_results(results: list[dict], output_path: str):
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

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), sharex=False, sharey=True)
    axes = axes.flatten()

    for i, (ax, res) in enumerate(zip(axes, results)):
        ax.set_facecolor("#EBEBEB")
        ax.plot(res["alpha_dates"], res["alpha_seq"], color="black", linewidth=1.1)

        xmin, xmax = X_LIMITS[res["name"]]
        ax.set_xlim(np.datetime64(xmin), np.datetime64(xmax))
        if res["name"] in {"AMD", "Fannie Mae"}:
            years = (2000, 2005, 2010, 2015, 2020)
        else:
            years = (2005, 2010, 2015, 2020)
        xticks = [np.datetime64(f"{year}-01-01") for year in years]
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.set_ylim(0.025, 0.18)
        ax.set_yticks([0.05, 0.10, 0.15])

        if i % 2 == 0:
            ax.set_ylabel(r"$\alpha_t$")

        ax.text(
            0.02,
            0.96,
            res["name"],
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
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    all_results = []
    for name, config in STOCKS.items():
        all_results.append(run_single_stock(name, config))
    plot_results(all_results, "figures/figure5.png")
