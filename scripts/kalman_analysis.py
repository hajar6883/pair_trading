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
parser.add_argument('--W',     type=float, default=1e-5, help='Kalman state noise variance')
args = parser.parse_args()

TICKER_1, TICKER_2 = args.t1, args.t2
START, END         = args.start, args.end
TRAIN_WINDOW       = args.train
TEST_WINDOW        = args.test




data = fetch_2tick(TICKER_1, TICKER_2, START, END)
log_prices = np.log(data)
P = log_prices.iloc[:, 0]  #1st ticker
Q = log_prices.iloc[:, 1]   


# full-sample betas / Kalman path are for the β plot only; the backtests re-estimate per window
ols_beta = ols_hedge_ratio(P, Q)
tls_beta = tls_hedge_ratio(P, Q)
_, kalman_beta = KalmanHedge(W=args.W).fit(P, Q, init_window=TRAIN_WINDOW)

wf_kwargs = dict(train_window=TRAIN_WINDOW, test_window=TEST_WINDOW)

print("── OLS (static β, refit per window) ───────────────")
wf_ols = walk_forward_backtest(P, Q, hedge='ols', **wf_kwargs)
compute_metrics(wf_ols)

print("\n── TLS (symmetric static β, refit per window) ──────")
wf_tls = walk_forward_backtest(P, Q, hedge='tls', **wf_kwargs)
compute_metrics(wf_tls)

print("\n── Kalman (time-varying β) ─────────────────────────")
wf_kalman = walk_forward_backtest(P, Q, hedge='kalman', kalman_kwargs={'W': args.W}, **wf_kwargs)
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
              annotation_text=f'OLS β (full sample) = {ols_beta:.3f}',
              annotation_position='right', row=2, col=1)

fig.add_hline(y=tls_beta, line_color='#06d6a0', line_dash='dash',
              annotation_text=f'TLS β (full sample) = {tls_beta:.3f}',
              annotation_position='right', row=2, col=1)

fig.update_layout(
    template='plotly_dark',
    height=700,
    title=f'{TICKER_1}/{TICKER_2} — Hedge Ratio Method Comparison',
)

fig.show()
