import time
from datetime import datetime
from tkinter.tix import Tree
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import timedelta
from datetime import date

date_format = "%Y-%m-%d"

symbols = ['IOO', 'BLK']

end_timestamp = int(time.time())
start_timestamp = datetime.strptime('2000-12-09', date_format).timestamp()

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

results = []
window_size = 240*10

for i in range(len(df.index)):
    if i+window_size > len(df.index):
        continue

    df_window = df[i:i+window_size]

    if df_window['IOO'][0] > df_window['IOO'][-1] and df_window['BLK'][0] > df_window['BLK'][-1]:
        continue

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df_window)
    S = risk_models.sample_cov(df_window)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    try:
        raw_weights = ef.max_sharpe()
    except:
        continue
    cleaned_weights = ef.clean_weights()
    results.append(cleaned_weights)

df_result = pd.DataFrame(results)
df_result.describe()
pass