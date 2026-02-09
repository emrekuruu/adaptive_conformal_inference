"""GARCH forecaster helpers used by example scripts."""

import warnings

import numpy as np
from arch import arch_model


def rolling_garch_volatility_predictions(
    returns: np.ndarray,
    lookback: int = 1250,
    garch_p: int = 1,
    garch_q: int = 1,
    start_up: int = 100,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit rolling GARCH models and return (predicted_variance, realized_variance)."""
    t_total = len(returns)
    start_up = max(start_up, lookback)
    n_steps = t_total - start_up + 1

    sigma2_hat = np.zeros(n_steps)
    realized_v = np.zeros(n_steps)

    for t in range(start_up, t_total + 1):
        if verbose:
            print(t)

        idx = t - start_up
        window = returns[t - lookback:t - 1]

        am = arch_model(
            window,
            mean="Zero",
            vol="GARCH",
            p=garch_p,
            q=garch_q,
            dist="normal",
            rescale=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            res = am.fit(disp="off")

        forecast = res.forecast(horizon=1)
        sigma2_hat[idx] = forecast.variance.values[-1, 0]
        realized_v[idx] = returns[t - 1] ** 2

        if t % 100 == 0:
            print(f"Done {t} steps")

    return sigma2_hat, realized_v

