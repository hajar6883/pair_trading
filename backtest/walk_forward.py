import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from spread.OU_process import OUProcess
from signals.signal import generate_signals

def walk_forward_backtest(spread, train_window=252, test_window=63, window='rolling', tc=.001,
                          min_halflife=5, max_halflife=150, adf_thresh=0.05):
    n= len(spread)

    all_results = []

    
    for t in range(0,n-train_window, test_window):
        if window == 'rolling':
            train_slice = spread.iloc[t : t+train_window]
        if window == 'expanding':
            train_slice = spread.iloc[0 : t+train_window]

        test_slice = spread.iloc[t+train_window : t+train_window+test_window]

        ou_temp = OUProcess()
        ou_temp.fit(train_slice)
        # train_scores = ou_temp.zscore(train_slice)

        if ou_temp.kappa <= 0:
            continue

        hl = ou_temp.half_life()
        if hl < min_halflife or hl > max_halflife:
            continue

        from statsmodels.tsa.stattools import adfuller
        adf_pvalue = adfuller(train_slice)[1]
        if adf_pvalue > adf_thresh:
            continue

        
        test_zscores= ou_temp.zscore(test_slice)
        positions = generate_signals(test_zscores)

        delta = test_slice.diff()
        ret = positions.shift(1)*delta
        cost = positions.diff().abs()*tc

        net_pnl = ret - cost
        results = pd.DataFrame({
            'spread'   : test_slice,
            'position' : positions,
            'raw_pnl'  : ret,
            'costs'    : cost,
            'net_pnl'  : net_pnl,
        })

        all_results.append(results)
        
    res = pd.concat(all_results)
    res['equity'] = res['net_pnl'].cumsum()
    return res




        








