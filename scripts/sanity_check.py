from statsmodels.tsa.stattools import adfuller, coint

def is_cointegrated(p, q, significance=0.05):
    _, p_value, _ = coint(p, q)
    return p_value < significance

def is_mean_reverting(spread, significance=0.05):
    p_value = adfuller(spread)[1]
    return p_value < significance