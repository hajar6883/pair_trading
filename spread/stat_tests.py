import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def hurst(ts: pd.Series, max_lag: int = 100) -> float:
    """
    Hurst exponent via R/S analysis
    H < 0.5  → mean-reverting
    H = 0.5  → random walk
    H > 0.5  → trending
    """
    ts = ts.dropna().values
    lags = range(2, min(max_lag, len(ts) // 2))
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if len(tau) < 2 or np.any(np.array(tau) <= 0):
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]


def adf_pvalue(spread: pd.Series, maxlag: int = 1) -> float:
    return adfuller(spread.dropna(), maxlag=maxlag)[1]


def is_cointegrated(p, q, significance=0.05):
    _, p_value, _ = coint(p, q)
    return p_value < significance

