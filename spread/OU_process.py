import numpy as np 
from math import pi 

class OUProcess:
    """
    Fits and represents a continuous-time OU process:
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    """
    def __init__(self):
        self.kappa = None
        self.mu = None
        self.sigma = None

    def _log_likelihood(self,X, params, dt ):
        """
        X : sequence of raw residuals between the pair  X_t = log(P_A,t) - β * log(P_B,t) """
        phi = np.exp(-params['kappa']*dt)
        var = params['sigma_tilde']**2
        mu = params['mu']

        Xt = X[1:]
        Xtm1 = X[:-1] # lagged sequence


        N = len(X) - 1
        return - (N/2 * np.log(2*pi*var)) - (1/(2*var)) * np.sum((Xt - phi*Xtm1 - mu*(1-phi))**2)
    
    def fit(self,X, dt=1.0):
        """OLS
        X : observed historical residuals 
        dt=1 for daily setting , change if using intraday data"""
        
        T = len(X) * dt
        Xt = X[1:]
        Xtm1 = X[:-1]
        Xt_mean = np.mean(Xt)
        Xtm1_mean = np.mean(Xtm1)

        C = np.cov(Xt, Xtm1)
        cov = C[1,0]
        Xtm1_var = C[1,1]

        beta = cov/ Xtm1_var
        alpha = Xt_mean - beta*Xtm1_mean

        kappa = - np.log(beta)/dt
        mu = alpha/(1-beta)

        var_eps = (dt/T) * np.sum((Xt - alpha - beta*Xtm1)**2)
        sigma = np.sqrt(var_eps * 2*kappa / (1 - np.exp(-2*kappa*dt)))

        self.kappa = kappa
        self.mu=mu 
        self.sigma= sigma

    def half_life(self):
        return np.log(2) / self.kappa
        

    def equilibrium_std(self):
        # std of the stationary distribution N(mu, sigma^2 / 2*kappa)
        return self.sigma / np.sqrt(2 * self.kappa)

    def zscore(self,spread):
        return (spread - self.mu) / self.equilibrium_std()
        

    def simulate(self, n_steps, x0=None, dt=1.0):
        # Euler-Maruyama discretisation — useful for sanity checking your fit
        pass






    