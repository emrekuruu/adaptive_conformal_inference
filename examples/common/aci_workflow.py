"""ACI evaluation workflow helpers used by examples."""

import numpy as np

from aci import ACI


def run_aci_vs_fixed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
    gamma: float,
    lookback: int,
    update_method: str = "simple",
    momentum_bw: float = 0.95,
    score: str = "normalized",
    first_err_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run adaptive ACI and fixed-alpha baseline on aligned predictions/outcomes."""
    if score not in ("normalized", "unnormalized"):
        raise ValueError(f"score must be 'normalized' or 'unnormalized', got '{score}'")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # In normalized mode we transform to z_t = V_t / sigma2_hat_t and predict 1.0.
    score_fn = lambda a, b: abs(a - b)

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

    n_steps = len(y_true)
    err_seq_aci = np.zeros(n_steps)
    err_seq_fixed = np.zeros(n_steps)

    for idx in range(n_steps):
        if score == "normalized":
            y_pred_ci = 1.0
            y_true_ci = float(y_true[idx] / max(float(y_pred[idx]), 1e-12))
        else:
            y_pred_ci = float(y_pred[idx])
            y_true_ci = float(y_true[idx])

        aci.issue(y_pred_ci)
        fixed.issue(y_pred_ci)

        if first_err_zero and idx == 0:
            out_aci = aci.observe(y_true_ci, err_t_override=0.0)
            out_fixed = fixed.observe(y_true_ci, err_t_override=0.0)
        else:
            out_aci = aci.observe(y_true_ci)
            out_fixed = fixed.observe(y_true_ci)

        err_seq_aci[idx] = float(out_aci["err_t"])
        err_seq_fixed[idx] = float(out_fixed["err_t"])

    alpha_sequence = aci.alpha_history
    return alpha_sequence, err_seq_aci, err_seq_fixed

