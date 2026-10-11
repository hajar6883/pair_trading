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

    def __init__(self, W: float = 1e-5, R: float = 1e-2):
        self.W = W
        self.R = R

    def fit(self, P: pd.Series, Q: pd.Series, init_window: int | None = None) -> tuple[pd.Series, pd.Series]:
        """
        Parameters
        ----------
        P : pd.Series
            Log prices of the dependent asset. Must share index with Q.
        Q : pd.Series
            Log prices of the independent asset. Must share index with P.
        init_window : int, optional
            Number of leading observations used for the OLS initial beta.
            Pass the first training window so the filter never sees test data
            (default: whole sample, which leaks the future).

        Returns:
        spread : pd.Series   prior (one-step-ahead) innovation p_t - beta_{t-1} * q_t,
                             i.e. the mispricing observable *before* beta is updated with p_t
        beta : pd.Series     beta_{t-1}, the hedge ratio known at the close of t-1
                             and therefore usable for the position held over t
        (both indexed like P and Q)
        """
        n = len(P)
        p = P.values
        q = Q.values
        m = n if init_window is None else init_window

        beta_init = ols_hedge_ratio(P.iloc[:m], Q.iloc[:m])
        state_var = float(np.var(p[:m] - beta_init * q[:m]))

        beta_prior = np.empty(n)
        beta_post  = beta_init
        for t in range(n):
            state_var     = state_var + self.W
            beta_prior[t] = beta_post

            innovation  = p[t] - beta_post * q[t]
            S           = q[t] ** 2 * state_var + self.R
            K           = state_var * q[t] / S
            beta_post   = beta_post + K * innovation
            state_var   = (1 - K * q[t]) * state_var

        beta   = pd.Series(beta_prior, index=P.index, name='beta')
        spread = pd.Series(p - beta_prior * q, index=P.index, name='spread')

        return spread, beta
