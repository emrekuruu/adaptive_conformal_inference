"""
Core ACI (Adaptive Conformal Inference) tracker.

Implements the alpha update rules from Gibbs & Candès (2021):
  - Simple update (Eq. 2):   α_{t+1} = α_t + γ(α − err_t)
  - Momentum update (Eq. 3): α_{t+1} = α_t + γ(α − Σ w_s · err_s)
"""

import numpy as np


class ACITracker:
    """Tracks the adaptive miscoverage level α_t for conformal inference.

    This is the core of ACI: a model-agnostic online update that adjusts
    the quantile level based on observed coverage errors.

    Parameters
    ----------
    alpha : float
        Target miscoverage level (e.g. 0.1 for 90% coverage).
    gamma : float
        Step size for the online update. Larger values adapt faster
        but oscillate more. Paper uses γ = 0.005.
    method : {"simple", "momentum"}
        Update rule. "simple" uses Eq. (2), "momentum" uses Eq. (3).
    momentum_bw : float
        Bandwidth for exponential weighting in momentum update.
        Paper uses 0.95.
    """

    def __init__(self, alpha: float, gamma: float, method: str = "simple",
                 momentum_bw: float = 0.95):
        if method not in ("simple", "momentum"):
            raise ValueError(f"method must be 'simple' or 'momentum', got '{method}'")

        self._alpha = alpha
        self._gamma = gamma
        self._method = method
        self._momentum_bw = momentum_bw

        self._alphat = alpha
        self._err_history: list[float] = []
        self._alpha_history: list[float] = []

    @property
    def alpha_t(self) -> float:
        """Current adaptive miscoverage level."""
        return self._alphat

    @property
    def alpha(self) -> float:
        """Target miscoverage level."""
        return self._alpha

    @property
    def gamma(self) -> float:
        """Step size."""
        return self._gamma

    @property
    def err_history(self) -> np.ndarray:
        """Sequence of coverage errors observed so far."""
        return np.array(self._err_history)

    @property
    def alpha_history(self) -> np.ndarray:
        """Sequence of α_t values used at each step."""
        return np.array(self._alpha_history)

    def update(self, err_t: float) -> float:
        """Record a coverage error and update α_t.

        Parameters
        ----------
        err_t : float
            1.0 if Y_t was NOT covered (miscoverage), 0.0 if covered.

        Returns
        -------
        float
            The updated α_{t+1}.
        """
        self._alpha_history.append(self._alphat)
        self._err_history.append(err_t)

        if self._method == "simple":
            self._alphat = self._alphat + self._gamma * (self._alpha - err_t)
        elif self._method == "momentum":
            t = len(self._err_history)
            w = np.array([self._momentum_bw ** i for i in range(t, 0, -1)])
            w = w / w.sum()
            weighted_err = np.dot(w, np.array(self._err_history))
            self._alphat = self._alphat + self._gamma * (self._alpha - weighted_err)

        # Keep alpha_t in a valid quantile range for downstream conformal logic.
        self._alphat = float(np.clip(self._alphat, 0.0, 1.0))

        return self._alphat

    def reset(self) -> None:
        """Reset the tracker to its initial state."""
        self._alphat = self._alpha
        self._err_history.clear()
        self._alpha_history.clear()
