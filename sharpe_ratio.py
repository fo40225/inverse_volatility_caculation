import time
from datetime import datetime
from tkinter.tix import Tree
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

date_format = "%Y-%m-%d"

symbols = ['VOO', 'VGLT']
# symbols = ['SPY', 'TLT']

# end_timestamp = datetime.strptime('2021-12-31', date_format).timestamp()
# start_timestamp = datetime.strptime('2012-01-01', date_format).timestamp()

# end_timestamp = int(time.time())
# start_timestamp = datetime.strptime('2002-07-30', date_format).timestamp()

end_timestamp = datetime.strptime('2020-03-31', date_format).timestamp()
start_timestamp = datetime.strptime('2015-07-01', date_format).timestamp()

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

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
