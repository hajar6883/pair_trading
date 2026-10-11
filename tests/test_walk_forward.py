import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import walk_forward_backtest
from evaluation.metrics import compute_metrics

TRAIN, TEST = 252, 63


def cointegrated_pair(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.normal(0, 0.01, n)) + 4
    x = np.zeros(n)
    for t in range(1, n):                       # OU spread, half-life ~ 14 days
        x[t] = 0.95 * x[t - 1] + rng.normal(0, 0.005)
    idx = pd.bdate_range('2015-01-01', periods=n)
    return pd.Series(0.8 * q + x, index=idx), pd.Series(q, index=idx)


def random_walks(n, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2015-01-01', periods=n)
    return (pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4, index=idx),
            pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4, index=idx))


@pytest.mark.parametrize('hedge', ['ols', 'tls', 'kalman'])
def test_no_lookahead(hedge):
    """Appending future data must not change any result already computed."""
    P, Q = cointegrated_pair()
    n = TRAIN + 15 * TEST                       # end on a window boundary
    short = walk_forward_backtest(P.iloc[:n], Q.iloc[:n], hedge=hedge)
    long  = walk_forward_backtest(P.iloc[:n + 100], Q.iloc[:n + 100], hedge=hedge)
    pd.testing.assert_frame_equal(short, long.iloc[:len(short)])


def test_finds_edge_on_true_cointegrated_pair():
    P, Q = cointegrated_pair()
    res = walk_forward_backtest(P, Q, hedge='ols', tc=0)
    assert res['traded'].any()
    assert res['equity'].iloc[-1] > 0


@pytest.mark.parametrize('hedge', ['ols', 'tls', 'kalman'])
def test_noise_has_no_edge(hedge, capsys):
    """Independent random walks: average gross Sharpe across seeds should be ~0."""
    sharpes = []
    for seed in range(30):
        P, Q = random_walks(TRAIN + 20 * TEST, seed)
        res = walk_forward_backtest(P, Q, hedge=hedge, tc=0)
        if res['traded'].any() and res['net_pnl'].std() > 0:
            sharpes.append(compute_metrics(res)['sharpe'])
    capsys.readouterr()
    assert len(sharpes) >= 5
    assert abs(np.mean(sharpes)) < 0.3
