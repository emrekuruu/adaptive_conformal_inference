"""
CQR-style ACI demo with asymmetric intervals.

Run:
    python examples/cqr_asymmetric_example.py
"""

import matplotlib.pyplot as plt
import numpy as np

from aci import ACI, cqr_interval_set, cqr_score


def simulate_asymmetric_series(t_total: int, seed: int = 13):
    """Create synthetic data with asymmetric predictive quantiles."""
    rng = np.random.default_rng(seed)
    t = np.arange(t_total)

    median = 0.45 + 0.18 * np.sin(t / 55.0) + 0.05 * np.cos(t / 17.0)
    # Intentionally strong asymmetry: upper side much wider than lower side.
    lower_spread = 0.05 + 0.02 * (1.0 + np.sin(t / 40.0))
    upper_spread = 0.30 + 0.08 * (1.0 + np.cos(t / 60.0))
    q_low = median - lower_spread
    q_high = median + upper_spread

    # Keep residual noise modest so asymmetry from (q_low, q_high) is clearly visible.
    base_noise = 0.02 * rng.standard_normal(t_total)
    pos_jump = (rng.random(t_total) < 0.05) * (0.12 + 0.10 * rng.random(t_total))
    y_true = median + base_noise + pos_jump

    return t, y_true, median, q_low, q_high


def run_method(
    y_true: np.ndarray,
    center: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    alpha: float,
    gamma: float,
    lookback: int,
) -> tuple[list[dict], float, float, float, np.ndarray]:
    """Run one method (ACI or fixed-alpha baseline)."""
    aci = ACI(
        alpha=alpha,
        gamma=gamma,
        lookback=lookback,
        method="simple",
        score_fn=cqr_score,
        set_fn=cqr_interval_set,
    )

    history: list[dict] = []
    for i in range(len(y_true)):
        pred = (float(q_low[i]), float(q_high[i]))
        lo, hi = aci.issue(pred)
        out = aci.observe(float(y_true[i]))
        history.append(
            {
                "round": i,
                "y_true": out["y_true"],
                "center": float(center[i]),
                "low": float(lo),
                "high": float(hi),
                "hit": bool(out["hit"]),
            }
        )

    coverage = float(np.mean([d["hit"] for d in history]))
    avg_low_half = float(np.mean([d["center"] - d["low"] for d in history]))
    avg_high_half = float(np.mean([d["high"] - d["center"] for d in history]))
    return history, coverage, avg_low_half, avg_high_half, np.array(aci.alpha_history)


def plot_panel(ax, history: list[dict], label: str, lower_width: float, upper_width: float):
    rounds = np.array([d["round"] for d in history])
    y_true = np.array([d["y_true"] for d in history])
    center = np.array([d["center"] for d in history])
    low = np.array([d["low"] for d in history])
    high = np.array([d["high"] for d in history])
    hit = np.array([d["hit"] for d in history], dtype=bool)

    # Split fill around model center (median), not interval midpoint.
    ax.fill_between(rounds, low, center, color="#f5a623", alpha=0.25, label="Lower side")
    ax.fill_between(rounds, center, high, color="#4a67ff", alpha=0.25, label="Upper side")
    ax.plot(rounds, center, color="black", linewidth=0.9, alpha=0.65, label="Predicted center")

    ax.scatter(rounds[hit], y_true[hit], s=14, c="green", alpha=0.7, label=f"Hit ({int(hit.sum())})", zorder=3)
    ax.scatter(rounds[~hit], y_true[~hit], s=14, c="red", alpha=0.7, label=f"Miss ({int((~hit).sum())})", zorder=3)

    ax.set_title(f"{label} | Lower width={lower_width:.3f} | Upper width={upper_width:.3f}")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    alpha = 0.05
    gamma = 0.01
    lookback = 200
    t_total = 1000

    _, y_true, center, q_low, q_high = simulate_asymmetric_series(t_total=t_total)

    # Method 1: CQR score + CQR set (asymmetric by design).
    hist_aci, cov_aci, low_aci, high_aci, alpha_aci = run_method(
        y_true, center, q_low, q_high, alpha, gamma, lookback
    )

    # Method 2: default ACI score/set on the same model center (median).
    # This keeps baseline symmetric around center, making CQR asymmetry explicit.
    q_center = center
    default = ACI(alpha=alpha, gamma=gamma, lookback=lookback, method="simple")
    hist_default: list[dict] = []
    for i in range(len(y_true)):
        lo, hi = default.issue(float(q_center[i]))
        out = default.observe(float(y_true[i]))
        hist_default.append(
            {
                "round": i,
                "y_true": out["y_true"],
                "center": float(q_center[i]),
                "low": float(lo),
                "high": float(hi),
                "hit": bool(out["hit"]),
            }
        )
    cov_fix = float(np.mean([d["hit"] for d in hist_default]))
    low_fix = float(np.mean([d["center"] - d["low"] for d in hist_default]))
    high_fix = float(np.mean([d["high"] - d["center"] for d in hist_default]))
    alpha_default = np.array(default.alpha_history)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plot_panel(axes[0], hist_aci, "CQR Functions", low_aci, high_aci)
    plot_panel(axes[1], hist_default, "Default Functions", low_fix, high_fix)
    axes[1].set_xlabel("Round")
    plt.tight_layout()
    plt.savefig("figures/cqr_asymmetric_example.png", dpi=300)
    plt.close(fig)
    print("Saved: figures/cqr_asymmetric_example.png")

    # Alpha trajectory plot (single shared alpha_t per method, not separate per quantile).
    rounds = np.array([d["round"] for d in hist_aci])
    fig_alpha, ax_alpha = plt.subplots(figsize=(14, 3.8))
    ax_alpha.plot(rounds, alpha_aci, color="#1f3aff", linewidth=1.5, label=r"CQR functions: $\alpha_t$")
    ax_alpha.plot(rounds, alpha_default, color="#ff3b30", linewidth=1.3, label=r"Default functions: $\alpha_t$")
    ax_alpha.axhline(alpha, color="black", linestyle="--", linewidth=1.0, label=f"Target alpha={alpha:.2f}")
    ax_alpha.set_title("ACI Alpha Trajectory (single shared alpha, not per quantile)")
    ax_alpha.set_xlabel("Round")
    ax_alpha.set_ylabel(r"$\alpha_t$")
    ax_alpha.grid(True, alpha=0.3)
    ax_alpha.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/cqr_asymmetric_example_alpha.png", dpi=300)
    plt.close(fig_alpha)
    print("Saved: figures/cqr_asymmetric_example_alpha.png")


if __name__ == "__main__":
    main()
