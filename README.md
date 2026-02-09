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
- `examples/figure1/` (self-contained Figure 1 example files)
- `examples/figure2/` (self-contained Figure 2 example files)
- `examples/quantile_reg.py` (CQR regression example)

## Reproducing Paper Figures

```bash
# Figure 1: Volatility prediction with normalized score (4 stocks)
python examples/figure1/reproduce.py

# Figure 2: Volatility prediction with unnormalized score (4 stocks)
python examples/figure2/reproduce.py
```

## Reproduced Figures From the Paper 

### Figure 1 (Normalized Score)

| Our reproduction | Paper figure |
|---|---|
| <img src="figures/figure1.png" alt="Our Figure 1" width="520"> | <img src="figures/figure_1_original.png" alt="Paper Figure 1" width="640"> |

### Figure 2 (Unnormalized Score)

| Our reproduction | Paper figure |
|---|---|
| <img src="figures/figure2.png" alt="Our Figure 2" width="520"> | <img src="figures/figure_2_original.png" alt="Paper Figure 2" width="640"> |

## Citation

```bibtex
  @article{gibbs2021adaptive,
    title={Adaptive conformal inference under distribution shift},
    author={Gibbs, Isaac and Candes, Emmanuel},
    journal={Advances in Neural Information Processing Systems},
    volume={34},
    pages={1660--1672},
    year={2021}
  }
```
