import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetch import fetch_2tick
from spread.hedge_ratio import ols_hedge_ratio, tls_hedge_ratio
from spread.kalman import KalmanHedge
from backtest.walk_forward import walk_forward_backtest
from evaluation.metrics import compute_metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--t1',    default='GLD')
parser.add_argument('--t2',    default='GDX')
parser.add_argument('--start', default='2020-01-01')
parser.add_argument('--end',   default='2026-01-01')
parser.add_argument('--train', type=int, default=252)
parser.add_argument('--test',  type=int, default=63)
args = parser.parse_args()

TICKER_1, TICKER_2 = args.t1, args.t2
START, END         = args.start, args.end
TRAIN_WINDOW       = args.train
TEST_WINDOW        = args.test




data = fetch_2tick(TICKER_1, TICKER_2, START, END)
log_prices = np.log(data)
P = log_prices.iloc[:, 0]  #1st ticker
Q = log_prices.iloc[:, 1]   


ols_beta   = ols_hedge_ratio(P, Q)
ols_spread = P - ols_beta * Q

tls_beta   = tls_hedge_ratio(P, Q)
tls_spread = P - tls_beta * Q

kh = KalmanHedge()
# kh.W = 1e-6
kalman_spread, kalman_beta = kh.fit(P, Q)


print("── OLS (static β) ─────────────────────────────────")
wf_ols = walk_forward_backtest(ols_spread, train_window=TRAIN_WINDOW, test_window=TEST_WINDOW, min_halflife=0.1)

full_index = ols_spread.index[TRAIN_WINDOW:]
wf_ols = wf_ols.reindex(full_index)
wf_ols['equity'] = wf_ols['equity'].ffill().fillna(0)

compute_metrics(wf_ols)

print("\n── TLS (symmetric static β) ────────────────────────")
wf_tls = walk_forward_backtest(tls_spread, train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,min_halflife=0.1)
full_index = tls_spread.index[TRAIN_WINDOW:]
wf_tls = wf_tls.reindex(full_index)
wf_tls['equity'] = wf_tls['equity'].ffill().fillna(0)
compute_metrics(wf_tls)

from statsmodels.tsa.stattools import adfuller
from spread.OU_process import OUProcess

# n = len(kalman_spread)
# for t in range(0, n - TRAIN_WINDOW, TEST_WINDOW):
#     train_slice = kalman_spread.iloc[t : t + TRAIN_WINDOW]
#     ou = OUProcess()
#     ou.fit(train_slice.values)
#     hl = ou.half_life() if ou.kappa > 0 else None
#     adf = adfuller(train_slice)[1]
#     print(f"t={t:4d}  kappa={ou.kappa:.4f}  hl={hl}  adf={adf:.4f}")

"""the hl filter kills is killing al the windows so either increase min_half life or change it conceptually : process noise (W )making beta adapt too aggressively and overcleaning the spread ( tradeoff .. slow)"""

print("\n── Kalman (time-varying β) ─────────────────────────")
wf_kalman = walk_forward_backtest(kalman_spread, train_window=TRAIN_WINDOW, test_window=TEST_WINDOW, min_halflife=0.1)

# wf_kalman = walk_forward_backtest(kalman_spread, train_window=TRAIN_WINDOW, test_window=TEST_WINDOW)

compute_metrics(wf_kalman)

# ── Plots

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        f'{TICKER_1}/{TICKER_2} — Walk-Forward Equity: OLS vs TLS vs Kalman',
        'Hedge Ratio β Over Time',
    ],
    vertical_spacing=0.12,
    row_heights=[0.6, 0.4],
)

# Equity curves
fig.add_trace(go.Scatter(
    x=wf_ols.index, y=wf_ols['equity'],
    mode='lines', name='OLS (static)',
    line=dict(color='#f77f00', width=1.5)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=wf_tls.index, y=wf_tls['equity'],
    mode='lines', name='TLS (symmetric static)',
    line=dict(color='#06d6a0', width=1.5)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=wf_kalman.index, y=wf_kalman['equity'],
    mode='lines', name='Kalman (time-varying)',
    line=dict(color='#00b4d8', width=1.5)
), row=1, col=1)

fig.add_hline(y=0, line_color='white', line_dash='dot', row=1, col=1)

# Beta over time
fig.add_trace(go.Scatter(
    x=kalman_beta.index, y=kalman_beta,
    mode='lines', name='Kalman β',
    line=dict(color='#00b4d8', width=1.5)
), row=2, col=1)

fig.add_hline(y=ols_beta, line_color='#f77f00', line_dash='dash',
              annotation_text=f'OLS β = {ols_beta:.3f}',
              annotation_position='right', row=2, col=1)

fig.add_hline(y=tls_beta, line_color='#06d6a0', line_dash='dash',
              annotation_text=f'TLS β = {tls_beta:.3f}',
              annotation_position='right', row=2, col=1)

fig.update_layout(
    template='plotly_dark',
    height=700,
    title=f'{TICKER_1}/{TICKER_2} — Hedge Ratio Method Comparison',
)

fig.show()
