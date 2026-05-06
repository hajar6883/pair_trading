import numpy as np 

def ols_hedge_ratio(P,Q):
    return np.cov(P, Q)[0,1]/np.var(Q)

def tls_hedge_ratio (P,Q):
    P_c = P.values - P.mean()
    Q_c = Q.values - P.mean()

    cov = np.cov(P_c, Q_c) 
    _, eigenvactors = np.linalg.eig(cov)
    v = eigenvactors[:,0] # minimize orthogonal distance instead of vertical residuals 
    return -v[0] / v[1] #symetrical ! 