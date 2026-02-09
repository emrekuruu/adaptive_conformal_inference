"""
aci-python: Adaptive Conformal Inference Under Distribution Shift.

Implementation of Gibbs & Candès (NeurIPS 2021).
https://arxiv.org/abs/2106.00170
"""

from aci.tracker import ACITracker

__version__ = "0.2.0"

__all__ = [
    "ACITracker",
]
