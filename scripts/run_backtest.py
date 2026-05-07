import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetch import fetch_2tick, compute_pair_spread
from spread.OU_process import OUProcess
from signals.signal import generate_signals
from backtest.engine import in_sample_backtest
from backtest.walk_forward import walk_forward_backtest
from evaluation.metrics import compute_metrics

ticker_1, ticker_2 =  'UCG.MI','ISP.MI'
start_date , end_date =  '2023-01-01', '2026-01-01'

data   = fetch_2tick(ticker_1, ticker_2,start_date , end_date)
spread = compute_pair_spread(data)

ou = OUProcess()
ou.fit(spread.values)
zscores   = pd.Series(ou.zscore(spread.values), index=spread.index)
positions = generate_signals(zscores)
is_results = in_sample_backtest(spread, positions, tc=0.001)

print("── IN-SAMPLE ──────────────────────────────")
compute_metrics(is_results)

wf_results = walk_forward_backtest(spread, train_window=126, test_window=21)

print("\n── WALK-FORWARD (out-of-sample) ───────────────────────────────────")
compute_metrics(wf_results)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['In-Sample Equity', 'Walk-Forward Equity (out-of-sample)']
)

fig.add_trace(go.Scatter(
    x=is_results.index, y=is_results['equity'],
    mode='lines', name='In-Sample',
    line=dict(color='#f77f00', width=1.5)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=wf_results.index, y=wf_results['equity'],
    mode='lines', name='Walk-Forward',
    line=dict(color='#00b4d8', width=1.5)
), row=1, col=2)

for col in [1, 2]:
    fig.add_hline(y=0, line_color='white', line_dash='dot', row=1, col=col)

fig.update_layout(
    template='plotly_dark',
    title=f"{ticker_1}/{ticker_2} — In-Sample vs Walk-Forward Equity",
    height=450
)

fig.show()
