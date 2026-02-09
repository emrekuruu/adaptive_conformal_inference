"""
Conformalized Quantile Regression (CQR) with adaptive conformal inference.

Example pipeline for election-night style regression experiments.
This module is intentionally outside core `aci` to keep the library model-agnostic.
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from aci import ACITracker


def run_election_night_pred(
    Y: np.ndarray,
    X: np.ndarray,
    alpha: float,
    gamma: float,
    tinit: int = 500,
    split_size: float = 0.75,
    update_method: str = "simple",
    momentum_bw: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ACI with conformalized quantile regression (CQR)."""
    T = len(Y)
    n_steps = T - tinit + 1

    tracker = ACITracker(alpha, gamma, method=update_method, momentum_bw=momentum_bw)
    adapt_err_seq = np.zeros(n_steps)
    no_adapt_err_seq = np.zeros(n_steps)

    for t in range(tinit - 1, T):
        idx = t - (tinit - 1)

        history_indices = np.arange(t)
        n_train = round(split_size * t)
        train_idx = np.sort(np.random.choice(history_indices, size=n_train, replace=False))
        cal_idx = np.setdiff1d(history_indices, train_idx)

        X_train = X[train_idx, :]
        Y_train = Y[train_idx]
        X_cal = X[cal_idx, :]
        Y_cal = Y[cal_idx]

        X_train_c = sm.add_constant(X_train)
        fit_upper = QuantReg(Y_train, X_train_c).fit(q=1 - alpha / 2)
        fit_lower = QuantReg(Y_train, X_train_c).fit(q=alpha / 2)

        X_cal_c = sm.add_constant(X_cal)
        pred_low = X_cal_c @ fit_lower.params
        pred_up = X_cal_c @ fit_upper.params
        scores = np.maximum(Y_cal - pred_up, pred_low - Y_cal)

        x_new = np.concatenate(([1.0], X[t, :]))
        q_up = x_new @ fit_upper.params
        q_low = x_new @ fit_lower.params
        new_score = max(Y[t] - q_up, q_low - Y[t])

        conf_quant_naive = np.quantile(scores, 1 - alpha)
        no_adapt_err_seq[idx] = float(conf_quant_naive < new_score)

        alphat = tracker.alpha_t
        if alphat >= 1:
            adapt_err_seq[idx] = 1.0
        elif alphat <= 0:
            adapt_err_seq[idx] = 0.0
        else:
            conf_quant_adapt = np.quantile(scores, 1 - alphat)
            adapt_err_seq[idx] = float(conf_quant_adapt < new_score)

        tracker.update(adapt_err_seq[idx])

        if (t + 1) % 100 == 0:
            print(f"Done {t + 1} time steps")

    alpha_trajectory = tracker.alpha_history
    return alpha_trajectory, adapt_err_seq, no_adapt_err_seq
