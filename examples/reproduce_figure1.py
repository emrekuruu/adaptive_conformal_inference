"""
Reproduce Figure 1 from Gibbs & Candès (2021).

Local coverage frequencies for adaptive conformal (blue), fixed-alpha (red),
and i.i.d. Bernoulli(0.1) sequence (grey) for volatility prediction using
the **normalized** conformity score S_t = |V_t - σ̂²_t| / σ̂²_t.

Stocks: Nvidia, AMD, BlackBerry, Fannie Mae.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from volatility import garch_conformal_forecasting
from utils import local_coverage, fetch_stock_data

# Paper parameters
ALPHA = 0.1
GAMMA = 0.005
LOOKBACK = 1250
COVERAGE_WINDOW = 500

STOCKS = {
    "Nvidia": {"ticker": "NVDA", "start": "1999-01-01", "end": "2021-06-01"},
    "AMD": {"ticker": "AMD", "start": "1994-01-01", "end": "2021-06-01"},
    "BlackBerry": {"ticker": "BB", "start": "1999-01-01", "end": "2021-06-01"},
    "Fannie Mae": {"ticker": "FNMA", "start": "1994-01-01", "end": "2021-06-01"},
}

# Fixed panel x-axis ranges to keep Figure 1 and Figure 2 aligned.
X_LIMITS = {
    "Nvidia": ("2004-06-01", "2020-12-31"),
    "AMD": ("1999-01-01", "2020-12-31"),
    "BlackBerry": ("2004-06-01", "2020-12-31"),
    "Fannie Mae": ("1999-01-01", "2020-12-31"),
}

# Deterministic Bernoulli baseline for reproducible visuals.
RNG = np.random.default_rng(42)


def run_single_stock(name: str, config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Processing {name} ({config['ticker']})")
    print(f"{'='*60}")

    data = fetch_stock_data(config["ticker"], config["start"], config["end"])
    returns = data["returns"]
    dates = data["dates"]
    print(f"  Downloaded {len(returns)} daily returns")

    alpha_seq, err_aci, err_fixed = garch_conformal_forecasting(
        returns,
        alpha=ALPHA,
        gamma=GAMMA,
        lookback=LOOKBACK,
        score="normalized",
    )

    # Compute local coverage
    cov_aci = local_coverage(err_aci, window=COVERAGE_WINDOW)
    cov_fixed = local_coverage(err_fixed, window=COVERAGE_WINDOW)

    # Bernoulli(alpha) baseline for visual comparison
    bernoulli_seq = RNG.binomial(1, ALPHA, size=len(err_aci))
    cov_bernoulli = local_coverage(bernoulli_seq, window=COVERAGE_WINDOW)

    # Align dates: ACI starts at index `lookback` in returns,
    # then local_coverage trims COVERAGE_WINDOW//2 from each end.
    start_up = max(100, LOOKBACK)
    # Dates corresponding to ACI output
    aci_dates = dates[start_up - 1:]  # length = T - start_up + 1
    # Dates for local coverage (centered window trims (window-1) total)
    offset = (COVERAGE_WINDOW - 1) // 2
    cov_dates = aci_dates[offset: offset + len(cov_aci)]

    return {
        "name": name,
        "cov_aci": cov_aci,
        "cov_fixed": cov_fixed,
        "cov_bernoulli": cov_bernoulli,
        "cov_dates": cov_dates,
        "alpha_seq": alpha_seq,
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

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), sharex=False, sharey=True)
    axes = axes.flatten()

    for i, (ax, res) in enumerate(zip(axes, results)):
        dates = res["cov_dates"]
        ax.set_facecolor("#EBEBEB")
        ax.plot(dates, res["cov_aci"], color="#1f3aff", linewidth=1.1, label="Adaptive Alpha")
        ax.plot(dates, res["cov_fixed"], color="#ff3b30", linewidth=1.1, label="Fixed Alpha")
        ax.plot(dates, res["cov_bernoulli"], color="#bfbfbf", linewidth=1.0, alpha=0.9, label="Bernoulli Sequence")

        ax.axhline(res["avg_cov_aci"], color="#1f3aff", linestyle=(0, (4, 4)), linewidth=2.0)
        ax.axhline(res["avg_cov_fixed"], color="#ff3b30", linestyle=(0, (4, 4)), linewidth=2.0)

        ax.axhline(1 - ALPHA, color="black", linestyle=(0, (4, 4)), linewidth=2.0)

        ax.set_title(res["name"], loc="left", fontsize=17, fontweight="bold", pad=10)
        if i % 2 == 0:
            ax.set_ylabel("Local Coverage Level")
        xmin, xmax = X_LIMITS[res["name"]]
        ax.set_xlim(np.datetime64(xmin), np.datetime64(xmax))
        ax.set_ylim(0.80, 0.975)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, color="white", linewidth=1.1)
        ax.minorticks_on()
        ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.7)
        ax.tick_params(axis="x", rotation=0)

    legend_handles = [
        Line2D([0], [0], color="#1f3aff", lw=2, label="Adaptive Alpha"),
        Line2D([0], [0], color="#ff3b30", lw=2, label="Fixed Alpha"),
        Line2D([0], [0], color="#bfbfbf", lw=2, label="Bernoulli Sequence"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.985),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.supxlabel("Time", y=0.07, fontsize=19)
    fig.tight_layout(rect=[0.03, 0.09, 0.995, 0.93])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    all_results = []
    for name, config in STOCKS.items():
        result = run_single_stock(name, config)
        all_results.append(result)
        print(f"  Avg coverage — ACI: {result['avg_cov_aci']:.3f}, "
              f"Fixed: {result['avg_cov_fixed']:.3f}")

    plot_results(all_results, "figures/figure1.png")
