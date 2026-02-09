"""
High-level Adaptive Conformal Inference (ACI) object.

This class wraps score calibration (Q-hat), prediction-set issuance, and
online alpha adaptation in a single object, aligned with the paper workflow.
"""

from collections.abc import Callable

import numpy as np


class ACI:
    """Single-object online ACI workflow.

    Lifecycle per round:
      1) issue(y_pred): emit C_hat_t(alpha_t) using recent conformity scores
      2) observe(y_true): compute S_t, err_t, update alpha_t

    Parameters
    ----------
    alpha : float
        Target miscoverage level.
    gamma : float
        ACI step size.
    lookback : int
        Number of recent conformity scores used to estimate Q-hat_t.
    method : {"simple", "momentum"}
        ACI alpha update rule.
    momentum_bw : float
        Momentum bandwidth (used when method="momentum").
    score_fn : callable or None
        Conformity score S_t(y_true, y_pred). Defaults to abs(y_true - y_pred).
    clip_alpha : bool
        Whether to clip alpha_t to [0, 1].
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        lookback: int = 500,
        method: str = "simple",
        momentum_bw: float = 0.95,
        score_fn: Callable[[float, float], float] | None = None,
        clip_alpha: bool = True,
    ):
        if method not in ("simple", "momentum"):
            raise ValueError(f"method must be 'simple' or 'momentum', got '{method}'")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")

        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._method = method
        self._momentum_bw = float(momentum_bw)
        self._clip_alpha = bool(clip_alpha)
        self._alphat = float(alpha)

        self._lookback = int(lookback)
        self._score_fn = score_fn if score_fn is not None else self._default_score

        self._score_history: list[float] = []
        self._err_history: list[float] = []
        self._alpha_history: list[float] = []
        self._issued: dict | None = None
        self._round = 0

    @staticmethod
    def _default_score(y_true: float, y_pred: float) -> float:
        return abs(y_true - y_pred)

    @property
    def alpha_t(self) -> float:
        """Current adaptive miscoverage level alpha_t."""
        return self._alphat

    @property
    def alpha(self) -> float:
        """Target miscoverage level alpha."""
        return self._alpha

    @property
    def gamma(self) -> float:
        """ACI update step size."""
        return self._gamma

    @property
    def method(self) -> str:
        """ACI update method: 'simple' or 'momentum'."""
        return self._method

    @property
    def lookback(self) -> int:
        """Calibration lookback window for Q-hat_t."""
        return self._lookback

    @property
    def score_history(self) -> np.ndarray:
        """Observed conformity score history."""
        return np.array(self._score_history)

    @property
    def err_history(self) -> np.ndarray:
        """Observed miscoverage history."""
        return np.array(self._err_history)

    @property
    def alpha_history(self) -> np.ndarray:
        """Alpha trajectory used across issued rounds."""
        return np.array(self._alpha_history)

    @property
    def has_pending_prediction(self) -> bool:
        """Whether a prediction set is currently pending observation."""
        return self._issued is not None

    def _recent_scores(self) -> np.ndarray:
        if not self._score_history:
            return np.array([])
        return np.array(self._score_history[-self._lookback:])

    def _qhat(self) -> float:
        scores = self._recent_scores()
        if scores.size == 0:
            return 0.0
        q_level = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
        return float(np.quantile(scores, q_level))

    def _update_alpha(self, err_t: float) -> float:
        self._alpha_history.append(self._alphat)
        self._err_history.append(err_t)

        if self._method == "simple":
            self._alphat = self._alphat + self._gamma * (self._alpha - err_t)
        else:
            t = len(self._err_history)
            w = np.array([self._momentum_bw ** i for i in range(t, 0, -1)])
            w = w / w.sum()
            weighted_err = float(np.dot(w, np.array(self._err_history)))
            self._alphat = self._alphat + self._gamma * (self._alpha - weighted_err)

        if self._clip_alpha:
            self._alphat = float(np.clip(self._alphat, 0.0, 1.0))

        return self._alphat

    def issue(self, y_pred: float) -> tuple[float, float]:
        """Issue prediction interval C-hat_t(alpha_t) around y_pred.

        Returns
        -------
        (lower, upper) : tuple[float, float]
            Issued interval bounds for the current round.
        """
        if self._issued is not None:
            raise RuntimeError("A prediction is already pending. Call observe(y_true) first.")

        qhat = self._qhat()
        y_pred = float(y_pred)
        lower = y_pred - qhat
        upper = y_pred + qhat

        self._issued = {
            "round": self._round,
            "y_pred": y_pred,
            "lower": lower,
            "upper": upper,
            "alpha_used": self.alpha_t,
            "qhat": qhat,
        }
        return lower, upper

    def observe(self, y_true: float) -> dict:
        """Observe truth for last issued interval and update alpha_t.

        Returns
        -------
        dict
            Round outcome including score, err_t, hit flag, alpha_used, alpha_next.
        """
        if self._issued is None:
            raise RuntimeError("No pending prediction. Call issue(y_pred) before observe(y_true).")

        y_true = float(y_true)
        y_pred = float(self._issued["y_pred"])
        lower = float(self._issued["lower"])
        upper = float(self._issued["upper"])
        alpha_used = float(self._issued["alpha_used"])
        qhat = float(self._issued["qhat"])
        round_id = int(self._issued["round"])

        score = float(self._score_fn(y_true, y_pred))
        if not np.isfinite(score):
            raise ValueError(f"score_fn returned non-finite value: {score}")
        if score < 0:
            raise ValueError(f"score_fn must return a non-negative score, got {score}")

        # Paper-consistent miscoverage event:
        # err_t = 1{ S_t > Q_hat_t(1 - alpha_t) }.
        # If alpha_t is allowed outside [0, 1], mirror original implementation
        # boundary behavior for quantile-level saturation.
        if alpha_used >= 1.0:
            err_t = 1.0
        elif alpha_used <= 0.0:
            err_t = 0.0
        else:
            err_t = float(score > qhat)
        hit = err_t == 0.0
        alpha_next = float(self._update_alpha(err_t))
        self._score_history.append(score)

        self._issued = None
        self._round += 1

        return {
            "round": round_id,
            "y_true": y_true,
            "y_pred": y_pred,
            "lower": lower,
            "upper": upper,
            "hit": hit,
            "err_t": err_t,
            "score_t": score,
            "qhat_t": qhat,
            "alpha_used": alpha_used,
            "alpha_next": alpha_next,
        }

    def reset(self) -> None:
        """Reset all ACI state."""
        self._alphat = self._alpha
        self._score_history.clear()
        self._err_history.clear()
        self._alpha_history.clear()
        self._issued = None
        self._round = 0
