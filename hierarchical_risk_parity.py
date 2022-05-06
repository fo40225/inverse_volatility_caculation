import time
from datetime import datetime
from tkinter.tix import Tree
import yfinance as yf
import pandas as pd
from pypfopt.hierarchical_portfolio import HRPOpt
from datetime import timedelta
from datetime import date

date_format = "%Y-%m-%d"

symbols = ['SPY', 'TLT']

end_timestamp = datetime.strptime('2021-12-31', date_format).timestamp()
start_timestamp = datetime.strptime('2012-01-01', date_format).timestamp()

consider_dividends = True

df = None

for symbol in symbols:
    start_str = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    end_str = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')
    data = yf.download(tickers=symbol, start=start_str, end=end_str, auto_adjust=consider_dividends)
    data = data.rename(columns={'Close': symbol})

    if df is None:
        df = data[[symbol]]
    else:
        df = pd.concat([df, data[symbol]], axis=1)

returns = df.pct_change().dropna()

hrp = HRPOpt(returns)
raw_weights = hrp.optimize()
cleaned_weights = hrp.clean_weights()
print(cleaned_weights)
hrp.portfolio_performance(verbose=True)
