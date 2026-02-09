"""Utility functions for ACI examples."""

import numpy as np


def local_coverage(err_seq: np.ndarray, window: int = 500) -> np.ndarray:
    """Compute local coverage frequency using a centered rolling window."""
    kernel = np.ones(window) / window
    rolling_err = np.convolve(err_seq, kernel, mode="valid")
    return 1.0 - rolling_err


def fetch_stock_data(ticker: str, start: str, end: str) -> dict:
    """Download stock data and compute returns and realized volatility."""
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Open"].values.flatten()
    dates = data.index
    returns = np.diff(prices) / prices[:-1]
    volatility = returns ** 2

    return {
        "prices": prices,
        "returns": returns,
        "volatility": volatility,
        "dates": dates[1:],
    }
