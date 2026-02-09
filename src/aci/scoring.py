"""Built-in conformity scores and set builders for common ACI workflows."""


def absolute_error_score(y_true: float, y_pred: float) -> float:
    """Absolute-error conformity score |y_true - y_pred|."""
    return abs(float(y_true) - float(y_pred))


def relative_error_score(y_true: float, y_pred: float, eps: float = 1e-12) -> float:
    """Relative conformity score |y_true - y_pred| / max(|y_pred|, eps)."""
    denom = max(abs(float(y_pred)), float(eps))
    return abs(float(y_true) - float(y_pred)) / denom


def cqr_score(y_true: float, y_pred: tuple[float, float]) -> float:
    """CQR conformity score max(q_low - y_true, y_true - q_high)."""
    q_low, q_high = y_pred
    return max(float(q_low) - float(y_true), float(y_true) - float(q_high))


def symmetric_interval_set(y_pred: float, qhat: float) -> tuple[float, float]:
    """Build symmetric interval [y_pred - qhat, y_pred + qhat]."""
    y = float(y_pred)
    return y - float(qhat), y + float(qhat)


def cqr_interval_set(y_pred: tuple[float, float], qhat: float) -> tuple[float, float]:
    """Build CQR-adjusted interval [q_low - qhat, q_high + qhat]."""
    q_low, q_high = y_pred
    return float(q_low) - float(qhat), float(q_high) + float(qhat)


def relative_interval_set(y_pred: float, qhat: float) -> tuple[float, float]:
    """Build relative interval [y_pred(1-qhat), y_pred(1+qhat)]."""
    y = float(y_pred)
    q = float(qhat)
    return y * (1.0 - q), y * (1.0 + q)

