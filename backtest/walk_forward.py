import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from spread.OU_process import OUProcess
from spread.hedge_ratio import ols_hedge_ratio, tls_hedge_ratio
from spread.kalman import KalmanHedge
from signals.signal import generate_signals

STATIC_HEDGES = {'ols': ols_hedge_ratio, 'tls': tls_hedge_ratio}


def walk_forward_backtest(P, Q, hedge='ols', train_window=252, test_window=63, window='rolling', tc=.001,
                          min_halflife=5, max_halflife=150, adf_thresh=0.05, kalman_kwargs=None):
    """
    P, Q   : pd.Series of log prices (same index)
    hedge  : 'ols' | 'tls' -> beta re-estimated on each training window only
             'kalman'      -> causal filter initialised on the first training window,
                              spread = prior innovation, hedge = beta_{t-1}

    tc     : cost per leg per unit of notional traded (0.001 = 10 bp)

    P&L is that of holding +1 P / -beta Q:  pos_{t-1} * (dP_t - beta_{t-1} * dQ_t).
    Costs: |dpos_t| * tc * (1 + |beta|)          entering/exiting both legs
         + |pos_{t-1}| * |dbeta_t| * tc          re-hedging the Q leg (Kalman only)
    All P&L columns are divided by the gross notional 1 + |beta|, i.e. they are
    daily returns on capital deployed, so Sharpe and drawdown are in % of gross notional.
    Positions are flattened at the end of each test window (exit cost charged).

    Returns a DataFrame over the whole out-of-sample period (from the first test day);
    windows rejected by the filters are flat with zero P&L.
    """
    n = len(P)
    dP, dQ = P.diff(), Q.diff()

    if hedge == 'kalman':
        kalman_spread, kalman_beta = KalmanHedge(**(kalman_kwargs or {})).fit(P, Q, init_window=train_window)

    oos_index = P.index[train_window:]
    out = pd.DataFrame({
        'spread'   : np.nan,
        'beta'     : np.nan,
        'position' : 0.0,
        'raw_pnl'  : 0.0,
        'costs'    : 0.0,
        'net_pnl'  : 0.0,
        'traded'   : False,
    }, index=oos_index)

    for t in range(0, n - train_window, test_window):
        start = t if window == 'rolling' else 0
        train = slice(start, t + train_window)
        test  = slice(t + train_window, t + train_window + test_window)

        if hedge == 'kalman':
            train_spread = kalman_spread.iloc[train]
            test_spread  = kalman_spread.iloc[test]
            test_beta    = kalman_beta.iloc[test]
        else:
            beta = STATIC_HEDGES[hedge](P.iloc[train], Q.iloc[train])
            train_spread = P.iloc[train] - beta * Q.iloc[train]
            test_spread  = P.iloc[test]  - beta * Q.iloc[test]
            test_beta    = pd.Series(beta, index=test_spread.index)

        idx = test_spread.index
        out.loc[idx, 'spread'] = test_spread
        out.loc[idx, 'beta']   = test_beta

        ou_temp = OUProcess()
        ou_temp.fit(train_spread.values)

        if ou_temp.kappa <= 0 or np.isnan(ou_temp.kappa):
            continue

        hl = ou_temp.half_life()
        if hl < min_halflife or hl > max_halflife:
            continue

        if adfuller(train_spread)[1] > adf_thresh:
            continue

        positions = generate_signals(ou_temp.zscore(test_spread))
        positions.iloc[-1] = 0.0          # close any open trade at the window boundary

        held     = positions.shift(1).fillna(0)
        gross    = 1 + test_beta.abs()
        ret      = held * (dP.loc[idx] - test_beta * dQ.loc[idx]) / gross
        trade    = positions.diff().fillna(positions.iloc[0]).abs()
        rehedge  = held.abs() * test_beta.diff().fillna(0).abs()
        cost     = (trade * gross + rehedge) * tc / gross

        out.loc[idx, 'position'] = positions
        out.loc[idx, 'raw_pnl']  = ret
        out.loc[idx, 'costs']    = cost
        out.loc[idx, 'net_pnl']  = ret - cost
        out.loc[idx, 'traded']   = True

    out['equity'] = out['net_pnl'].cumsum()
    return out
