import numpy as np
import pandas as pd
from spread.hedge_ratio import ols_hedge_ratio


class KalmanHedge:
    """
    Kalman filter for a time-varying hedge ratio.
    State-space model:
    Observation : P_t = beta_t * Q_t + eps_t,   eps_t ~ N(0, R)
    Transition  : beta_t = beta_{t-1} + eta_t,  eta_t ~ N(0, W)

    W : State noise variance (controls how fast beta is allowed to change)
    R : Observation noise variance (controls how much weight is given to new data)
    """

    def __init__(self, W: float = 1e-4, R: float = 1e-2):
        self.W = W
        self.R = R

    def fit(self, P: pd.Series, Q: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Parameters
        ----------
        P : pd.Series
            Log prices of the dependent asset. Must share index with Q.
        Q : pd.Series
            Log prices of the independent asset. Must share index with P.

        Returns:
        spread : pd.Series        
        beta : pd.Series (both indexed like P and Q)
        """
        n = len(P)
        p = P.values
        q = Q.values

        beta_arr = np.empty(n)
        beta_init = ols_hedge_ratio(P, Q)
        beta_arr[0] = beta_init
        state_var = float(np.var(p - beta_init * q))

        for t in range(1, n):
            state_var = state_var + self.W

            innovation  = p[t] - beta_arr[t - 1] * q[t]
            S           = q[t] ** 2 * state_var + self.R
            K           = state_var * q[t] / S
            beta_arr[t] = beta_arr[t - 1] + K * innovation
            state_var   = (1 - K * q[t]) * state_var

        beta   = pd.Series(beta_arr, index=P.index, name='beta')
        spread = pd.Series(p - beta_arr * q, index=P.index, name='spread')

        return spread, beta
