import numpy as np


def compute_metrics(results, risk_free_annual=0.04):
    """Print and return a summary of strategy performance."""
    pnl    = results['net_pnl'].dropna()
    equity = results['equity'].dropna()

    # Sharpe (annualised, 252 trading days)
    rf_daily = risk_free_annual / 252
    sharpe = ((pnl.mean() - rf_daily) / pnl.std()) * np.sqrt(252)

    # max drawdown
    running_max = equity.cummax()
    drawdown    = equity - running_max
    max_dd      = drawdown.min()

    # number of round-trip trades
    position_changes = results['position'].diff().abs()
    n_trades = int((position_changes > 0).sum() / 2)

    # average holding period (days in non-zero position)
    in_trade = (results['position'] != 0).sum()
    avg_hold = in_trade / n_trades if n_trades > 0 else 0

    # win rate
    trade_starts = results.index[results['position'].diff().abs() > 0]
    wins = 0
    for i in range(0, len(trade_starts) - 1, 2):
        trade_pnl = results.loc[trade_starts[i]:trade_starts[i+1], 'net_pnl'].sum()
        if trade_pnl > 0:
            wins += 1
    win_rate = wins / n_trades if n_trades > 0 else 0

    print(f"{'Sharpe ratio':<25} {sharpe:.3f}")
    print(f"{'Max drawdown':<25} {max_dd:.5f}")
    print(f"{'Total P&L':<25} {equity.iloc[-1]:.5f}")
    print(f"{'Number of trades':<25} {n_trades}")
    print(f"{'Avg holding period':<25} {avg_hold:.1f} days")
    print(f"{'Win rate':<25} {win_rate:.1%}")

    return {
        'sharpe': sharpe, 'max_drawdown': max_dd,
        'n_trades': n_trades, 'avg_holding': avg_hold,
        'win_rate': win_rate
    }
