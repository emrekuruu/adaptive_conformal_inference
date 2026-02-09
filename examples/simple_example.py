"""
Simple ACI showcase: ACI vs fixed intervals with hits/misses over time.

Run:
    python examples/simple_example.py
"""

import matplotlib.pyplot as plt
import numpy as np

from aci import ACITracker


def plot_observation_intervals(
    ax,
    history: list[dict],
    title: str,
    coverage: float,
) -> None:
    """Plot intervals + actual values; green=hit, red=miss."""
    rounds = np.array([d["round"] for d in history])
    actuals = np.array([d["actual"] for d in history])
    lowers = np.array([d["lower_bound"] for d in history])
    uppers = np.array([d["upper_bound"] for d in history])
    predicted = np.array([d["predicted"] for d in history])
    hits = np.array([d["hit"] for d in history], dtype=bool)

    ax.fill_between(rounds, predicted, uppers, alpha=0.25, color="blue", label="Upper interval")
    ax.fill_between(rounds, lowers, predicted, alpha=0.25, color="orange", label="Lower interval")
    ax.plot(rounds, predicted, color="black", linewidth=0.8, alpha=0.6, label="Predicted")

    hit_rounds = rounds[hits]
    hit_actuals = actuals[hits]
    miss_rounds = rounds[~hits]
    miss_actuals = actuals[~hits]

    if len(hit_rounds) > 0:
        ax.scatter(hit_rounds, hit_actuals, c="green", s=14, alpha=0.7, label=f"Hit ({len(hit_rounds)})", zorder=3)
    if len(miss_rounds) > 0:
        ax.scatter(miss_rounds, miss_actuals, c="red", s=14, alpha=0.7, label=f"Miss ({len(miss_rounds)})", zorder=3)

    ax.set_title(f"{title} | overall coverage={coverage:.1%}")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    rng = np.random.default_rng(7)

    alpha = 0.05
    gamma = 0.01
    lookback = 120
    warmup = 40
    T = 1200

    # Simple dataset with clear regime shifts in noise.
    t = np.arange(T)
    trend = 0.002 * t
    seasonal = 0.5 * np.sin(t / 25.0)
    sigma = np.where(t < 400, 0.18, np.where(t < 800, 0.70, 0.28))
    y = trend + seasonal + sigma * rng.standard_normal(T)

    tracker = ACITracker(alpha=alpha, gamma=gamma, method="simple")

    hist_aci: list[dict] = []
    hist_fixed: list[dict] = []
    residuals: list[float] = []

    # Simple model: trailing moving-average predictor.
    for i in range(warmup, T):
        past = y[max(0, i - 20):i]
        y_pred = float(np.mean(past))
        score_t = abs(y[i] - y_pred)

        cal_scores = np.array(residuals[max(0, len(residuals) - lookback):], dtype=float)
        if len(cal_scores) == 0:
            q_aci = 0.0
            q_fixed = 0.0
        else:
            q_aci = float(np.quantile(cal_scores, 1 - tracker.alpha_t))
            q_fixed = float(np.quantile(cal_scores, 1 - alpha))

        lower_aci, upper_aci = y_pred - q_aci, y_pred + q_aci
        lower_fixed, upper_fixed = y_pred - q_fixed, y_pred + q_fixed

        hit_aci = lower_aci <= y[i] <= upper_aci
        hit_fixed = lower_fixed <= y[i] <= upper_fixed

        hist_aci.append(
            {
                "round": i,
                "actual": float(y[i]),
                "predicted": y_pred,
                "lower_bound": lower_aci,
                "upper_bound": upper_aci,
                "hit": bool(hit_aci),
            }
        )
        hist_fixed.append(
            {
                "round": i,
                "actual": float(y[i]),
                "predicted": y_pred,
                "lower_bound": lower_fixed,
                "upper_bound": upper_fixed,
                "hit": bool(hit_fixed),
            }
        )

        tracker.update(float(not hit_aci))
        residuals.append(score_t)

    cov_aci = np.mean([d["hit"] for d in hist_aci]) if hist_aci else 0.0
    cov_fixed = np.mean([d["hit"] for d in hist_fixed]) if hist_fixed else 0.0
    rounds = np.array([d["round"] for d in hist_aci])
    alpha_aci = np.array(tracker.alpha_history)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plot_observation_intervals(axes[0], hist_aci, "ACI Interval", cov_aci)
    plot_observation_intervals(axes[1], hist_fixed, "Fixed Interval", cov_fixed)
    axes[1].set_xlabel("Round")
    fig.suptitle(
        f"ACI vs Fixed Intervals (target alpha={alpha:.2f}, target coverage={1-alpha:.0%})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("figures/simple_example.png", dpi=300)
    plt.close(fig)

    # Separate alpha trajectory plot.
    fig_alpha, ax_alpha = plt.subplots(figsize=(14, 3.5))
    ax_alpha.plot(rounds, alpha_aci, color="#1f3aff", linewidth=1.5, label=r"ACI $\alpha_t$")
    ax_alpha.axhline(alpha, color="#ff3b30", linestyle="--", linewidth=1.3, label=f"Fixed alpha={alpha:.2f}")
    ax_alpha.set_title("ACI Alpha Trajectory vs Fixed Alpha")
    ax_alpha.set_xlabel("Round")
    ax_alpha.set_ylabel(r"$\alpha$")
    ax_alpha.grid(True, alpha=0.3)
    ax_alpha.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/simple_example_alpha.png", dpi=300)
    plt.close(fig_alpha)

if __name__ == "__main__":
    main()
