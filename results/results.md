# Walk-forward results

2018-01-01 to 2026-01-01, train 252 / test 63 days, rolling. Net of 10 bp per leg plus re-hedging. Returns and drawdown are on gross notional (1 + |β|). Flat days count as zero. *In market* = share of days with an open position; *Active windows* = share of days in a window that passed the κ / half-life / ADF filters.

| Pair | Hedge | Sharpe | Ann. return | Max DD | Trades | Win rate | In market | Active windows | Break-even bp |
|---|---|---|---|---|---|---|---|---|---|
| GLD/GDX | OLS | -0.46 | -0.8% | -6.6% | 18 | 33.3% | 5.9% | 10.7% | — |
| GLD/GDX | TLS | — | 0.0% | 0.0% | 0 | — | 0.0% | 3.6% | — |
| GLD/GDX | Kalman | -0.48 | -2.3% | -24.3% | 12 | 33.3% | 8.5% | 43.0% | — |
| LQD/HYG | OLS | -0.73 | -1.7% | -12.8% | 44 | 27.3% | 11.3% | 21.2% | — |
| LQD/HYG | TLS | -1.78 | -1.5% | -10.8% | 32 | 9.4% | 3.3% | 10.5% | — |
| LQD/HYG | Kalman | — | 0.0% | 0.0% | 0 | — | 0.0% | 3.6% | — |
| WFC/JPM | OLS | 0.04 | 0.1% | -6.6% | 31 | 38.7% | 5.0% | 10.5% | 11.07 |
| WFC/JPM | TLS | 0.12 | 0.2% | -2.3% | 9 | 33.3% | 2.8% | 3.6% | 17.21 |
| WFC/JPM | Kalman | -0.33 | -0.9% | -9.1% | 12 | 50.0% | 4.1% | 21.5% | — |
| UCG.MI/ISP.MI | OLS | -0.72 | -1.8% | -12.9% | 33 | 24.2% | 7.6% | 17.7% | — |
| UCG.MI/ISP.MI | TLS | -0.38 | -0.6% | -5.7% | 24 | 41.7% | 2.1% | 3.5% | 1.59 |
| UCG.MI/ISP.MI | Kalman | 0.10 | 0.1% | -1.2% | 1 | 100.0% | 0.6% | 11.4% | 41.05 |

## Sharpe vs cost per leg

| Pair | Hedge | 0 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|---|
| GLD/GDX | OLS | -0.17 | -0.32 | -0.46 | -0.74 |
| GLD/GDX | TLS | — | — | — | — |
| GLD/GDX | Kalman | -0.41 | -0.45 | -0.48 | -0.55 |
| LQD/HYG | OLS | -0.18 | -0.46 | -0.73 | -1.22 |
| LQD/HYG | TLS | -0.80 | -1.35 | -1.78 | -2.33 |
| LQD/HYG | Kalman | — | — | — | — |
| WFC/JPM | OLS | 0.39 | 0.21 | 0.04 | -0.31 |
| WFC/JPM | TLS | 0.30 | 0.21 | 0.12 | -0.05 |
| WFC/JPM | Kalman | -0.21 | -0.27 | -0.33 | -0.45 |
| UCG.MI/ISP.MI | OLS | -0.36 | -0.55 | -0.72 | -1.06 |
| UCG.MI/ISP.MI | TLS | 0.07 | -0.16 | -0.38 | -0.80 |
| UCG.MI/ISP.MI | Kalman | 0.14 | 0.12 | 0.10 | 0.07 |
