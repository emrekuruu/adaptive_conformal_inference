"""
GARCH-based adaptive conformal inference for volatility prediction.

Example pipeline used to reproduce Figures 1 and 2 from Gibbs & Candès (2021).
This module is intentionally outside core `aci` to keep the library model-agnostic.
"""

import warnings

import numpy as np
from arch import arch_model

from aci import ACI


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

    if score == "normalized":
        # Run ACI in transformed space:
        # z_t = V_t / sigma2_hat_t, prediction is 1.0
        # so |z_t - 1| equals the normalized conformity score.
        score_fn = lambda y_true, y_pred: abs(y_true - y_pred)
    else:
        score_fn = lambda y_true, y_pred: abs(y_true - y_pred)

    aci = ACI(
        alpha=alpha,
        gamma=gamma,
        lookback=lookback,
        method=update_method,
        momentum_bw=momentum_bw,
        score_fn=score_fn,
        clip_alpha=False,
    )
    fixed = ACI(
        alpha=alpha,
        gamma=0.0,
        lookback=lookback,
        method="simple",
        score_fn=score_fn,
        clip_alpha=False,
    )

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
            y_pred_ci = 1.0
            y_true_ci = float(v_t / max(float(sigma2_hat), 1e-12))
        else:
            y_pred_ci = float(sigma2_hat)
            y_true_ci = float(v_t)

        aci.issue(y_pred_ci)
        fixed.issue(y_pred_ci)

        # Match original reproduction initialization (first err_t set to 0).
        if idx == 0:
            out_aci = aci.observe(y_true_ci, err_t_override=0.0)
            out_fixed = fixed.observe(y_true_ci, err_t_override=0.0)
        else:
            out_aci = aci.observe(y_true_ci)
            out_fixed = fixed.observe(y_true_ci)

        err_seq_aci[idx] = float(out_aci["err_t"])
        err_seq_fixed[idx] = float(out_fixed["err_t"])

        if t % 100 == 0:
            print(f"Done {t} steps")

    alpha_sequence = aci.alpha_history
    return alpha_sequence, err_seq_aci, err_seq_fixed
