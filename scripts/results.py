"""
Out-of-sample results across pairs, hedge methods and cost levels.
Writes results/results.md (tables) and results/equity.png.

python3 scripts/results.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.fetch import fetch_2tick
from backtest.walk_forward import walk_forward_backtest
from evaluation.metrics import compute_metrics

PAIRS = [
    ('GLD', 'GDX'),          # gold vs gold miners
    ('LQD', 'HYG'),          # IG vs HY credit, breaks down 2022-23
    ('WFC', 'JPM'),          # from scripts/find_pairs.py
    ('UCG.MI', 'ISP.MI'),    # from scripts/find_pairs.py
]
START, END   = '2018-01-01', '2026-01-01'
METHODS      = {'OLS': 'ols', 'TLS': 'tls', 'Kalman': 'kalman'}
COSTS_BP     = [0, 5, 10, 20]          # per leg
BASE_COST_BP = 10
OUT_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def run(P, Q, hedge, cost_bp):
    return walk_forward_backtest(P, Q, hedge=hedge, tc=cost_bp / 1e4)


def summarise(P, Q, hedge):
    base = run(P, Q, hedge, BASE_COST_BP)
    m    = compute_metrics(base, verbose=False)
    years = len(base) / 252

    # costs are linear in tc -> break-even = gross P&L / cost per bp
    gross    = base['raw_pnl'].sum()
    cost_1bp = base['costs'].sum() / BASE_COST_BP
    breakeven = gross / cost_1bp if cost_1bp > 0 and gross > 0 else np.nan

    row = {
        'Sharpe':        m['sharpe'],
        'Ann. return':   base['net_pnl'].sum() / years,
        'Max DD':        m['max_drawdown'],
        'Trades':        m['n_trades'],
        'Win rate':      m['win_rate'] if m['n_trades'] else np.nan,
        'In market':     (base['position'] != 0).mean(),
        'Active windows': base['traded'].mean(),
        'Break-even bp': breakeven,
    }
    sharpes = {f'{c} bp': compute_metrics(run(P, Q, hedge, c), verbose=False)['sharpe'] for c in COSTS_BP}
    return row, sharpes, base['equity']


def fmt(df, pct_cols=(), int_cols=(), dec=2):
    out = df.copy().astype(object)
    for c in df.columns:
        for i in df.index:
            v = df.loc[i, c]
            if pd.isna(v):
                out.loc[i, c] = '—'
            elif c in pct_cols:
                out.loc[i, c] = f'{v:.1%}'
            elif c in int_cols:
                out.loc[i, c] = f'{int(v)}'
            else:
                out.loc[i, c] = f'{v:.{dec}f}'
    names  = list(out.index.names)
    header = '| ' + ' | '.join(names + list(out.columns)) + ' |'
    sep    = '|' + '---|' * (len(names) + len(out.columns))
    body   = ['| ' + ' | '.join(list(i) + list(r)) + ' |' for i, r in zip(out.index, out.values)]
    return '\n'.join([header, sep] + body)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, cost_rows, curves = {}, {}, {}

    for t1, t2 in PAIRS:
        print(f'{t1}/{t2}...')
        log_prices = np.log(fetch_2tick(t1, t2, START, END).dropna())
        P, Q = log_prices[t1], log_prices[t2]
        for name, hedge in METHODS.items():
            key = (f'{t1}/{t2}', name)
            rows[key], cost_rows[key], curves[key] = summarise(P, Q, hedge)

    main  = pd.DataFrame(rows).T
    costs = pd.DataFrame(cost_rows).T
    main.index.names = costs.index.names = ['Pair', 'Hedge']

    md = (
        f'# Walk-forward results\n\n'
        f'{START} to {END}, train 252 / test 63 days, rolling. Net of {BASE_COST_BP} bp per leg plus re-hedging. '
        f'Returns and drawdown are on gross notional (1 + |β|). Flat days count as zero. '
        f'*In market* = share of days with an open position; *Active windows* = share of days in a window '
        f'that passed the κ / half-life / ADF filters.\n\n'
        + fmt(main, pct_cols=('Ann. return', 'Max DD', 'Win rate', 'In market', 'Active windows'), int_cols=('Trades',))
        + '\n\n## Sharpe vs cost per leg\n\n'
        + fmt(costs)
        + '\n'
    )
    with open(os.path.join(OUT_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('\n' + md)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    colors = {'OLS': '#f77f00', 'TLS': '#06d6a0', 'Kalman': '#00b4d8'}
    for ax, (t1, t2) in zip(axes.flat, PAIRS):
        pair = f'{t1}/{t2}'
        for name in METHODS:
            (curves[(pair, name)] * 100).plot(ax=ax, label=name, color=colors[name], lw=1.2)
        ax.axhline(0, color='grey', lw=0.6, ls=':')
        ax.set_title(pair)
        ax.set_ylabel('cum. return, % of gross')
    axes[0, 0].legend()
    fig.suptitle(f'Walk-forward equity, net of {BASE_COST_BP} bp per leg')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'equity.png'), dpi=120)
    print(f'saved results/results.md and results/equity.png')
