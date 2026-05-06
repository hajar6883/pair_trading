# Pairs Trading — Statistical Arbitrage Framework

A quant-style pairs trading system built from scratch. Implements cointegration-based spread construction, OU process signal generation, and walk-forward backtesting with three hedge ratio methods.

## Project Structure

```
pairs_trading/
├── data/           fetch.py          — yfinance data download
├── spread/         OU_process.py     — OU parameter estimation (AR(1) / OLS)
│                   hedge_ratio.py    — OLS and TLS hedge ratios
│                   kalman.py         — Kalman filter for dynamic hedge ratio
│                   stat_tests.py     — Hurst exponent, ADF p-value
├── signals/        signal.py         — entry/exit state machine (±2σ / 0 / ±3σ)
├── backtest/       engine.py         — in-sample backtest
│                   walk_forward.py   — rolling walk-forward backtest
├── evaluation/     metrics.py        — Sharpe, max drawdown, win rate, avg holding
├── plots/          plot_results.py   — Plotly dark theme visualizations
├── scripts/        run_backtest.py   — in-sample vs walk-forward comparison
│                   kalman_analysis.py — OLS vs TLS vs Kalman comparison (CLI)
│                   find_pairs.py     — cointegration screener across sectors
```

## Methodology

**Spread construction**: log spread `S_t = log(P_t) - β · log(Q_t)` where β is estimated via OLS, TLS (orthogonal regression), or Kalman filter.

**OU process**: spread is modelled as `dX_t = κ(μ − X_t)dt + σdW_t`. Parameters fitted via AR(1) regression. Half-life = `log(2)/κ`.

**Signals**: z-score = `(S_t − μ) / σ_eq` where `σ_eq = σ/√(2κ)`. Entry at ±2σ, exit at 0, stop-loss at ±3σ.

**Walk-forward**: rolling 252-day training window, 63-day test window. Per-window filters: `κ > 0`, half-life 5–150 days, ADF p-value < 0.05.

## Hedge Ratio Comparison

Three methods compared on GLD/GDX (2020–2026):

| Method | Sharpe | Max DD | Trades | Win Rate |
|--------|--------|--------|--------|----------|
| OLS (static) | 2.08 | -0.053 | 10 | 70% |
| TLS (symmetric) | 2.31 | -0.056 | 11 | 73% |
| Kalman (dynamic) | 3.88 | -0.032 | 180 | 73% |

TLS outperforms OLS on this pair — the symmetric hedge ratio avoids the ordering bias of OLS. Kalman β adapts over time (0.65–0.75 vs static OLS 1.02), producing more trading opportunities and better risk-adjusted returns.

## Next Steps — Regime Detection

Running the following exposes a key weakness of the current framework:

```bash
python3 scripts/kalman_analysis.py --t1 LQD --t2 HYG --start 2018-01-01 --end 2026-01-01
```

LQD/HYG (investment grade vs high yield credit) appears cointegrated pre-2022 but breaks down during the Fed rate hiking cycle (2022–2023). Static hedge ratios (OLS/TLS) keep entering trades expecting mean reversion that never comes, resulting in sustained drawdowns of ~-0.25. The Kalman filter partially adapts β to the shifting relationship, significantly reducing losses — but cannot fully rescue a pair where cointegration itself breaks down.

This motivates the next step: a **regime detection layer** that suspends trading when the spread signals a structural break — via rolling ADF stability, VIX threshold, or a hidden Markov model on spread volatility.

## Run

```bash
# In-sample vs walk-forward backtest
python3 scripts/run_backtest.py

# Compare hedge ratio methods (CLI)
python3 scripts/kalman_analysis.py --t1 GLD --t2 GDX
python3 scripts/kalman_analysis.py --t1 LQD --t2 HYG --start 2018-01-01 --end 2026-01-01

# Screen for cointegrated pairs
python3 scripts/find_pairs.py
```
