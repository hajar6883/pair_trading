import numpy as np
import pandas as pd
import pytest

from spread.OU_process import OUProcess


def simulated_ou(kappa=0.05, mu=0.3, sigma=0.02, n=20_000, seed=0):
    np.random.seed(seed)
    true = OUProcess()
    true.kappa, true.mu, true.sigma = kappa, mu, sigma
    return true.simulate(n)


def test_ou_recovers_parameters():
    ou = OUProcess()
    ou.fit(simulated_ou())
    # simulate() is Euler, fit() assumes the exact AR(1) map -> small (~3%) bias on kappa
    assert ou.kappa == pytest.approx(0.05, rel=0.15)
    assert ou.mu    == pytest.approx(0.3,  abs=0.01)
    assert ou.sigma == pytest.approx(0.02, rel=0.10)


def test_ou_fit_same_for_series_and_array():
    # regression: X[1:] / X[:-1] on a pd.Series used to align by index -> sigma ~6x too small
    x = simulated_ou(n=2_000)
    s = pd.Series(x, index=pd.bdate_range('2020-01-01', periods=len(x)))
    a, b = OUProcess(), OUProcess()
    a.fit(x)
    b.fit(s)
    assert (a.kappa, a.mu, a.sigma) == pytest.approx((b.kappa, b.mu, b.sigma))
