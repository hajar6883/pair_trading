import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch import fetch_2tick, compute_pair_spread
from spread.OU_process import OUProcess
from signals.signal import generate_signals
from evaluation.metrics import compute_metrics
from plots.plot_results import plot_results
import pandas as pd



def in_sample_backtest(spread, positions, tc=0.001):
    """
    spread    : pd.Series — log spread
    positions : pd.Series — +1, -1, 0
    tc        : transaction cost per unit traded (in spread units)

    Returns a DataFrame with daily pnl, costs, net pnl, equity curve.
    """
    delta_spread = spread.diff()

    raw_pnl = positions.shift(1) * delta_spread

    costs = positions.diff().abs()* tc     # cost paid whenever position changes


    net_pnl = raw_pnl - costs
    equity  = net_pnl.cumsum()

    results = pd.DataFrame({
        'spread'   : spread,
        'position' : positions,
        'raw_pnl'  : raw_pnl,
        'costs'    : costs,
        'net_pnl'  : net_pnl,
        'equity'   : equity,
    })

    return results




# if __name__ == '__main__':
#     data    = fetch_2tick('XOM', 'CVX', '2020-01-01', '2026-01-01')
#     spread  = compute_pair_spread(data)

#     ou = OUProcess()
#     ou.fit(spread.values)
#     zscores   = ou.zscore(spread.values)
#     zscores   = pd.Series(zscores, index=spread.index)

#     positions = generate_signals(zscores)
#     results   = in_sample_backtest(spread, positions, tc=0.001)

#     print(f"\nOU params — kappa: {ou.kappa:.4f}  mu: {ou.mu:.4f}  sigma: {ou.sigma:.4f}")
#     print(f"Half-life: {ou.half_life():.1f} days\n")
#     compute_metrics(results)
#     plot_results(results, title='XOM/CVX Pairs Trading')
