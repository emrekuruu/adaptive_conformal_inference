"""
aci-python: Adaptive Conformal Inference Under Distribution Shift.

Implementation of Gibbs & Candès (NeurIPS 2021).
https://arxiv.org/abs/2106.00170
"""

from importlib.metadata import PackageNotFoundError, version

from aci.aci import ACI
from aci.scoring import (
    absolute_error_score,
    cqr_interval_set,
    cqr_score,
    relative_error_score,
    relative_interval_set,
    symmetric_interval_set,
)

try:
    __version__ = version("adaptive-conformal-inference")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "ACI",
    "absolute_error_score",
    "relative_error_score",
    "cqr_score",
    "symmetric_interval_set",
    "relative_interval_set",
    "cqr_interval_set",
]
