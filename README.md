# adaptive-conformal-inference

`adaptive-conformal-inference` is a Python library for **Adaptive Conformal Inference (ACI)**.

Its main job is to provide a clean, model-agnostic **online ACI object** (`ACI`) that:
- takes point predictions,
- issues conformal intervals, and
- updates adaptively from feedback (`y_true`).

This repository also contains example scripts to:
- reproduce figures from the original ACI paper, and
- demonstrate ACI behavior versus a fixed conformal baseline.

Implementation is based on:

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

## Core Usage

### Initialization Parameters

| Parameter | Default | What it controls and how to tune it |
|---|---|---|
| `alpha` | required | Target miss rate. `alpha=0.1` means target coverage is about `90%`. Lower `alpha` makes intervals more conservative (wider on average). |
| `gamma` | required | Adaptation speed for `alpha_t`. Increase if environment changes quickly and coverage lags. Decrease if `alpha_t` oscillates too much. |
| `lookback` | `500` | Calibration memory for score quantiles. Larger is more stable but slower to react. Smaller reacts faster but can be noisy. |
| `method` | `"simple"` | Alpha update rule. `"simple"` reacts directly to each miss/hit. `"momentum"` smooths updates using recent history. |
| `momentum_bw` | `0.95` | Used only for `method="momentum"`. Closer to `1.0` means smoother, longer memory. Lower values emphasize very recent rounds. |
| `score_fn` | `None` | Conformity score function `score_fn(y_true, y_pred) -> nonnegative float`. Default is absolute error `abs(y_true - y_pred)`. Choose this to match your task's notion of error. |
| `clip_alpha` | `True` | Keeps `alpha_t` in `[0, 1]`. Recommended `True` for most users to avoid invalid quantile levels. |

### Conformity Score (`score_fn`)

`score_fn` defines what "bad prediction" means for your problem. ACI adapts coverage for that score.

- Default (`None`): absolute error, good for many regression tasks.
- Relative/percentage error:
  - useful when prediction scale changes a lot over time.
  - often better for volatility-like tasks.
- Asymmetric score:
  - useful if underprediction is more costly than overprediction (or vice versa).

`score_fn` must return a nonnegative finite number.

### Tuning Guide

- Coverage below target for long periods:
  - increase `gamma` (faster correction), or decrease `lookback` (faster calibration updates).
- Coverage far above target (intervals too wide):
  - decrease `gamma`, or increase `lookback` for less reactive widening.
- `alpha_t` is very jumpy:
  - decrease `gamma`, or use `method="momentum"` with high `momentum_bw` (for example `0.95` to `0.99`).
- Regime changes happen often:
  - use smaller `lookback` and/or larger `gamma`.
- Regime is mostly stable:
  - use larger `lookback` and smaller `gamma` for smoother intervals.


### API Reference

The package exposes one class: `aci.ACI`.

| Method / Property | What it does in practice |
|---|---|
| `issue(y_pred)` | Call this once per round after producing your model prediction. Returns `(lower, upper)` interval for that round, using current `alpha_t` and recent calibration scores. |
| `observe(y_true, err_t_override=None)` | Call this once the true outcome is available. Computes hit/miss, updates `alpha_t`, and returns diagnostics (`hit`, `err_t`, `score_t`, `alpha_used`, `alpha_next`, `qhat_t`). |
| `reset()` | Restarts the object to its initial state (`alpha_t=alpha`, empty histories). Useful between datasets/episodes. |
| `alpha_t` | Current adaptive miss-rate level that will be used for the next `issue(...)`. If it goes down, intervals usually widen; if it goes up, intervals usually narrow. |
| `alpha_history` | Historical `alpha_t` values. Use this to inspect adaptation behavior and tune `gamma`/`lookback`. |
| `err_history` | Historical miss indicators (`1=miss`, `0=hit`). Use mean of this to estimate realized miss rate. |
| `score_history` | Historical conformity scores from your `score_fn`. Useful for debugging whether score design matches your task. |
| `has_pending_prediction` | `True` after `issue(...)` and before `observe(...)`. Helps enforce correct online call order. |

### Online Workflow

Use this order every round:

1. Compute your point prediction `y_pred`.
2. Call `issue(y_pred)` to get interval bounds.
3. Observe the true value `y_true`.
4. Call `observe(y_true)` to update ACI state.


```python
from aci import ACI

# One object handles interval issuance + online alpha adaptation
aci = ACI(alpha=0.1, gamma=0.005, lookback=500, method="simple")

for t in range(T):
    y_pred_t = model_predict(x_t)      # your forecaster
    lower, upper = aci.issue(y_pred_t) # C_hat_t(alpha_t)

    y_true_t = observe_truth()
    out = aci.observe(y_true_t)        # updates alpha_t online

    hit = out["hit"]                   # per-round coverage signal
    alpha_next = aci.alpha_t           # alpha to be used on the NEXT round
```


## Reproduce Paper Results

```bash
# Figure 1: Volatility prediction with normalized score (4 stocks)
python examples/figure1/reproduce.py

# Figure 2: Volatility prediction with unnormalized score (4 stocks)
python examples/figure2/reproduce.py
```

### Figure 1 (Normalized Score)

| Our reproduction | Paper figure |
|---|---|
| <img src="figures/figure1.png" alt="Our Figure 1" width="520"> | <img src="figures/figure_1_original.png" alt="Paper Figure 1" width="640"> |

### Figure 2 (Unnormalized Score)

| Our reproduction | Paper figure |
|---|---|
| <img src="figures/figure2.png" alt="Our Figure 2" width="520"> | <img src="figures/figure_2_original.png" alt="Paper Figure 2" width="640"> |

## ACI Demo (vs Fixed Baseline)

Run:

```bash
python examples/simple_example.py
```

This demo uses a simple synthetic dataset and simple predictor, then compares:
- ACI adaptive intervals (online-updated `alpha_t`)
- Fixed intervals (constant `alpha`)
- Left half of the timeline is intentionally hard (frequent misses).
- Right half is intentionally easy (rare misses).

Green points are hits, red points are misses.

<img src="figures/simple_example.png" alt="Simple ACI vs Fixed demo" width="980">

The script also saves the alpha trajectory:

<img src="figures/simple_example_alpha.png" alt="ACI alpha trajectory vs fixed alpha" width="980">

How to interpret alpha movement:
- In the hard half, `alpha_t` should go down: misses are frequent, so intervals widen.
- In the easy half, `alpha_t` should go up: misses are rare, so intervals shrink.

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
