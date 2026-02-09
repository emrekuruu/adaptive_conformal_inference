"""Shared volatility + ACI pipeline for Figure 1 and Figure 2 examples."""

import numpy as np

from common.aci_workflow import run_aci_vs_fixed
from common.garch_forecaster import rolling_garch_volatility_predictions


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
    """Run GARCH forecasting + ACI evaluation for a volatility series."""
    sigma2_hat, realized_v = rolling_garch_volatility_predictions(
        returns=returns,
        lookback=lookback,
        garch_p=garch_p,
        garch_q=garch_q,
        start_up=start_up,
        verbose=verbose,
    )

    return run_aci_vs_fixed(
        y_true=realized_v,
        y_pred=sigma2_hat,
        alpha=alpha,
        gamma=gamma,
        lookback=lookback,
        update_method=update_method,
        momentum_bw=momentum_bw,
        score=score,
        first_err_zero=True,
    )

