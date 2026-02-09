"""
aci-python: Adaptive Conformal Inference Under Distribution Shift.

Implementation of Gibbs & Candès (NeurIPS 2021).
https://arxiv.org/abs/2106.00170
"""

from importlib.metadata import PackageNotFoundError, version

from aci.tracker import ACITracker

try:
    __version__ = version("adaptive-conformal-inference")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "ACITracker",
]
