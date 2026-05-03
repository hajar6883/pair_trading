import numpy as np
import pandas as pd


def generate_signals(zscore, entry=2.0, exit=0.0, stop=3.0):
    """
    zscore : pd.Series of z-scores of the spread
    Returns a pd.Series of positions: +1 (long), -1 (short), 0 (flat)
    """
    n = len(zscore)
    positions = np.zeros(n)
    curr_position = 0

    for t in range(n):
        if curr_position == 0:
            if zscore.iloc[t] > entry:        # P too expensive relative to Q → short
                curr_position = -1
            elif zscore.iloc[t] < -entry:     # P too cheap relative to Q → long
                curr_position = 1

        elif curr_position == 1:
            if zscore.iloc[t] < -stop or zscore.iloc[t] >= exit:  
                curr_position = 0

        elif curr_position == -1:
            if zscore.iloc[t] > stop or zscore.iloc[t] <= exit:  
                curr_position = 0

        positions[t] = curr_position

    return pd.Series(positions, index=zscore.index, name='position')


