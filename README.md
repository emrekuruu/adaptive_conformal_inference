# adaptive-conformal-inference

Python implementation of **Adaptive Conformal Inference (ACI)** from:

> Isaac Gibbs and Emmanuel J. Cand&egrave;s. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021.
> [arXiv:2106.00170](https://arxiv.org/abs/2106.00170)

ACI adapts conformal prediction sets to distribution shift by tracking a single parameter that adjusts the quantile level online, provably achieving target coverage over long time intervals.

## Installation

```bash
pip install adaptive-conformal-inference
```

For running the example scripts (stock data + plots):

```bash
pip install adaptive-conformal-inference[examples]
```

## Quick Start

```python
from aci import ACITracker

# Core: model-agnostic alpha tracker
tracker = ACITracker(alpha=0.1, gamma=0.005)

for t in range(T):
    # Your conformal prediction logic here...
    err_t = 1.0 if y_true not in prediction_set else 0.0
    tracker.update(err_t)
    next_alpha = tracker.alpha_t  # use this for the next prediction set
```

## Modules

| Module | Description |
|--------|-------------|
| `aci.tracker.ACITracker` | Core alpha update (Simple and Momentum rules) |

Non-essential forecasting/data pipelines are intentionally kept in `examples/`:
- `examples/volatility.py` (GARCH volatility example)
- `examples/quantile_reg.py` (CQR regression example)
- `examples/utils.py` (example-only data and plotting utilities)

## Reproducing Paper Figures

```bash
# Figure 1: Volatility prediction with normalized score (4 stocks)
python examples/reproduce_figure1.py

# Figure 2: Volatility prediction with unnormalized score (4 stocks)
python examples/reproduce_figure2.py
```

## Citation

```bibtex
@inproceedings{gibbs2021adaptive,
  title={Adaptive Conformal Inference Under Distribution Shift},
  author={Gibbs, Isaac and Cand{\`e}s, Emmanuel J},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
