"""
GARCH-based adaptive conformal inference for volatility prediction.

Example pipeline used to reproduce Figures 1 and 2 from Gibbs & Candès (2021).
This module is intentionally outside core `aci` to keep the library model-agnostic.
"""

import warnings

import numpy as np
from arch import arch_model

from aci import ACITracker


def garch_conformal_forecasting(
    returns: np.ndarray,
    alpha: float,
    gamma: float,
    lookback: int = 1250,
    garch_p: int = 1,
    garch_q: int = 1,
    start_up: int = 100,
    verbose: bool = False,
    update_method: str = "simple",
    momentum_bw: float = 0.95,
    score: str = "normalized",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ACI with GARCH(p,q) volatility model."""
    if score not in ("normalized", "unnormalized"):
        raise ValueError(f"score must be 'normalized' or 'unnormalized', got '{score}'")

    T = len(returns)
    start_up = max(start_up, lookback)
    n_steps = T - start_up + 1

    tracker = ACITracker(
        alpha,
        gamma,
        method=update_method,
        momentum_bw=momentum_bw,
        clip_alpha=False,
    )
    scores = np.zeros(n_steps)
    err_seq_aci = np.zeros(n_steps)
    err_seq_fixed = np.zeros(n_steps)

    for t in range(start_up, T + 1):
        if verbose:
            print(t)

        idx = t - start_up
        window = returns[t - lookback: t - 1]

        am = arch_model(
            window, mean="Zero", vol="GARCH", p=garch_p, q=garch_q,
            dist="normal", rescale=False
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            res = am.fit(disp="off")

        forecast = res.forecast(horizon=1)
        sigma2_hat = forecast.variance.values[-1, 0]
        v_t = returns[t - 1] ** 2

        if score == "normalized":
            scores[idx] = abs(v_t - sigma2_hat) / sigma2_hat
        else:
            scores[idx] = abs(v_t - sigma2_hat)

        if idx > 0:
            recent_start = max(idx - lookback + 1, 0)
            recent_scores = scores[recent_start: idx]
            alphat = tracker.alpha_t
            if alphat >= 1:
                err_seq_aci[idx] = 1.0
            elif alphat <= 0:
                err_seq_aci[idx] = 0.0
            else:
                err_seq_aci[idx] = float(scores[idx] > np.quantile(recent_scores, 1 - alphat))
            err_seq_fixed[idx] = float(scores[idx] > np.quantile(recent_scores, 1 - alpha))

        tracker.update(err_seq_aci[idx])

        if t % 100 == 0:
            print(f"Done {t} steps")

    alpha_sequence = tracker.alpha_history
    return alpha_sequence, err_seq_aci, err_seq_fixed
