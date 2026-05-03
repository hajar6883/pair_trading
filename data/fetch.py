
import yfinance as yf
import numpy as np

def fetch_2tick(ticker_1, ticker_2, start_date, end_date):
    "pull daily closes for two tickers via yfinance "

    data = yf.download([ticker_1, ticker_2], start = start_date, end= end_date,interval='1d', group_by = 'column')['Close']
    return data
    

def compute_pair_spread(data):
    """pandas.series of the the pair of stock  in question """
    
    log_prices = np.log(data)
    P = log_prices.iloc[:, 0] #first tick..
    Q = log_prices.iloc[:, 1]
    beta  = np.cov(P, Q)[0,1]/np.var(Q)
    X =  P- beta*Q
    return X 

