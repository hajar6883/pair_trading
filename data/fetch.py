import yfinance as yf

def fetch_2tick(ticker_1, ticker_2, start_date, end_date):
    "pull daily closes for two tickers via yfinance"

    data = yf.download([ticker_1, ticker_2], start = start_date, end= end_date,interval='1d', group_by = 'column')['Close']
    return data


# print(fetch_2tick('AAPL', 'VOO', '2023-02-03', "2025-12-03"))
