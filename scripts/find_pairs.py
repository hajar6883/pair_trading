"""
Cointegration Pair Screener
----------------------------
Screens a universe of stocks for cointegrated pairs using:
  - Engle-Granger cointegration test
  - Half-life of mean reversion (Ornstein-Uhlenbeck)
  - Hurst exponent (mean-reversion strength)

Output: ranked DataFrame saved to pair_screener_results.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta

from spread.OU_process import OUProcess
from spread.hedge_ratio import ols_hedge_ratio
from spread.stat_tests import hurst, adf_pvalue


# ─── Universe 

UNIVERSE = {
    "French Banks":       ["BNP.PA", "GLE.PA", "ACA.PA"],
    "Euro Banks":         ["SAN.MC", "BBVA.MC", "ISP.MI", "UCG.MI"],
    "US Banks":           ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "Oil & Gas":          ["XOM", "CVX", "COP", "BP", "SHEL", "TTE.PA"],
    "Euro Oil":           ["TTE.PA", "ENI.MI", "REP.MC"],
    "US Tech":            ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "Semis":              ["NVDA", "AMD", "INTC", "QCOM", "AVGO"],
    "Euro Luxury":        ["MC.PA", "RMS.PA", "KER.PA", "CFR.SW"],
    "US Airlines":        ["AAL", "DAL", "UAL", "LUV"],
    "Mining":             ["RIO", "BHP", "GLEN.L", "AAL.L"],
    "Pharma":             ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
    "Euro Pharma":        ["SAN.PA", "NOVN.SW", "ROG.SW"],
}

START       = (datetime.today() - timedelta(days=3*365)).strftime("%Y-%m-%d")
END         = datetime.today().strftime("%Y-%m-%d")
PVAL_THRESH  = 0.05
MIN_HALFLIFE = 5
MAX_HALFLIFE = 126


def download_prices(universe: dict) -> pd.DataFrame:
    all_tickers = list({t for tickers in universe.values() for t in tickers})
    print(f"Downloading {len(all_tickers)} tickers from {START} to {END}...")
    raw = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(axis=1, thresh=int(0.9 * len(raw)))
    raw = raw.ffill().dropna()
    print(f"  {raw.shape[1]} tickers available after cleaning, {raw.shape[0]} trading days\n")
    return np.log(raw)


def screen_pairs(log_prices: pd.DataFrame, universe: dict) -> pd.DataFrame:
    results = []
    available = set(log_prices.columns)

    for sector, tickers in universe.items():
        tickers = [t for t in tickers if t in available]
        if len(tickers) < 2:
            continue

        print(f"Screening {sector} ({len(tickers)} tickers, "
              f"{len(list(combinations(tickers, 2)))} pairs)...")

        for t1, t2 in combinations(tickers, 2):
            s1 = log_prices[t1]
            s2 = log_prices[t2]

            # ── Engle-Granger (both directions, take best) ──
            _, pval1, _ = coint(s1, s2)
            _, pval2, _ = coint(s2, s1)

            if pval1 <= pval2:
                pval, dep, ind = pval1, t1, t2
            else:
                pval, dep, ind = pval2, t2, t1

            if pval > PVAL_THRESH:
                continue

            # ── Spread ──
            beta   = ols_hedge_ratio(log_prices[dep], log_prices[ind])
            spread = log_prices[dep] - beta * log_prices[ind]

            # ── Half-life via OU fit ──
            ou = OUProcess()
            ou.fit(spread.values)
            if ou.kappa <= 0:
                continue
            hl = ou.half_life()
            if not (MIN_HALFLIFE <= hl <= MAX_HALFLIFE):
                continue

            # ── Hurst & ADF ──
            h        = hurst(spread)
            adf_pval = adf_pvalue(spread)

            # ── Current z-score ──
            zscore = (spread.iloc[-1] - spread.mean()) / spread.std()

            results.append({
                "sector":         sector,
                "ticker_1":       dep,
                "ticker_2":       ind,
                "beta":           round(beta, 4),
                "coint_pval":     round(pval, 4),
                "adf_pval":       round(adf_pval, 4),
                "half_life_days": round(hl, 1),
                "hurst":          round(h, 3),
                "zscore_now":     round(zscore, 3),
            })

    df = pd.DataFrame(results)
    if df.empty:
        print("\nNo cointegrated pairs found with current thresholds.")
        return df

    df = df.sort_values(["coint_pval", "half_life_days"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    log_prices = download_prices(UNIVERSE)
    results    = screen_pairs(log_prices, UNIVERSE)

    if not results.empty:
        print(f"\n{'='*70}")
        print(f"  {len(results)} cointegrated pairs found")
        print(f"{'='*70}")
        print(results.to_string(index=True))

        out = "pair_screener_results.csv"
        results.to_csv(out, index=False)
        print(f"\nResults saved to {out}")

        print(f"\nPairs per sector:")
        print(results["sector"].value_counts().to_string())

        tradeable = results[results["zscore_now"].abs() > 1.5]
        if not tradeable.empty:
            print(f"\nCurrently tradeable pairs (|z-score| > 1.5):")
            print(tradeable[["sector", "ticker_1", "ticker_2",
                              "half_life_days", "zscore_now"]].to_string(index=False))
