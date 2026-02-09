"""
Simple ACI showcase: ACI vs fixed intervals with hits/misses over time.

Run:
    python examples/simple_example.py
"""

import matplotlib.pyplot as plt
import numpy as np

from aci import ACI


def plot_observation_intervals(
    ax,
    history: list[dict],
    title: str,
    coverage: float,
    split_round: int,
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
    ax.axvline(split_round, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    rng = np.random.default_rng(7)

    alpha = 0.10
    gamma = 0.05
    lookback = 700
    warmup = 120
    T = 1600
    split_round = warmup + (T - warmup) // 2

    # Two explicit regimes to showcase ACI adaptation:
    # 1) hard regime (frequent misses): high volatility + occasional shocks
    # 2) easy regime (rare misses): low volatility, smooth dynamics
    t = np.arange(T)
    base = 0.45 * np.sin(t / 32.0) + 0.14 * np.cos(t / 70.0)

    easy_noise = 0.03 * rng.standard_normal(T)
    y_easy = base + easy_noise

    hard_noise = 1.20 * rng.standard_normal(T)
    shock_mask = rng.random(T) < 0.18
    shock_sign = np.where(rng.random(T) < 0.5, -1.0, 1.0)
    shocks = shock_mask * shock_sign * (4.0 + 2.5 * rng.random(T))
    regime_shift = 0.8 * np.sign(np.sin(t / 7.5))
    y_hard = base + regime_shift + hard_noise + shocks

    y = y_easy.copy()
    y[warmup:split_round] = y_hard[warmup:split_round]

    aci = ACI(alpha=alpha, gamma=gamma, lookback=lookback, method="simple")
    fixed = ACI(alpha=alpha, gamma=0.0, lookback=lookback, method="simple")

    hist_aci: list[dict] = []
    hist_fixed: list[dict] = []

    # Simple model: trailing moving-average predictor.
    for i in range(warmup, T):
        past = y[max(0, i - 40):i]
        y_pred = float(np.mean(past))
        aci.issue(y_pred)
        fixed.issue(y_pred)

        out_aci = aci.observe(float(y[i]))
        out_fixed = fixed.observe(float(y[i]))

        hist_aci.append(
            {
                "round": i,
                "actual": out_aci["y_true"],
                "predicted": out_aci["y_pred"],
                "lower_bound": out_aci["lower"],
                "upper_bound": out_aci["upper"],
                "hit": bool(out_aci["hit"]),
            }
        )
        hist_fixed.append(
            {
                "round": i,
                "actual": out_fixed["y_true"],
                "predicted": out_fixed["y_pred"],
                "lower_bound": out_fixed["lower"],
                "upper_bound": out_fixed["upper"],
                "hit": bool(out_fixed["hit"]),
            }
        )

    cov_aci = np.mean([d["hit"] for d in hist_aci]) if hist_aci else 0.0
    cov_fixed = np.mean([d["hit"] for d in hist_fixed]) if hist_fixed else 0.0
    rounds = np.array([d["round"] for d in hist_aci])
    alpha_aci = np.array(aci.alpha_history)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plot_observation_intervals(axes[0], hist_aci, "ACI Interval", cov_aci, split_round)
    plot_observation_intervals(axes[1], hist_fixed, "Fixed Interval", cov_fixed, split_round)
    axes[1].set_xlabel("Round")
    fig.suptitle(
        (
            "ACI vs Fixed Intervals "
            f"(target alpha={alpha:.2f}, target coverage={1-alpha:.0%}; "
            "left regime: hard, right regime: easy)"
        ),
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("figures/simple_example.png", dpi=300)
    plt.close(fig)

    # Separate alpha trajectory plot.
    fig_alpha, ax_alpha = plt.subplots(figsize=(14, 3.5))
    ax_alpha.plot(rounds, alpha_aci, color="#1f3aff", linewidth=1.5, label=r"ACI $\alpha_t$")
    ax_alpha.axhline(alpha, color="#ff3b30", linestyle="--", linewidth=1.3, label=f"Fixed alpha={alpha:.2f}")
    ax_alpha.axvline(split_round, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
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
