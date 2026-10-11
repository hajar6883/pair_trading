import pandas as pd

import data.fetch as fetch


def test_fetch_keeps_ticker_order(monkeypatch):
    # yfinance returns columns sorted alphabetically, whatever order you ask for
    idx = pd.bdate_range('2024-01-01', periods=3)
    cols = pd.MultiIndex.from_product([['Close'], ['GDX', 'GLD']])
    fake = pd.DataFrame(1.0, index=idx, columns=cols)
    monkeypatch.setattr(fetch.yf, 'download', lambda *a, **k: fake)

    data = fetch.fetch_2tick('GLD', 'GDX', '2024-01-01', '2024-01-05')
    assert list(data.columns) == ['GLD', 'GDX']
