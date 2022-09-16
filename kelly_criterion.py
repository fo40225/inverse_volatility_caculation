from datetime import datetime
import time
import yfinance as yf
import os
import numpy as np
from datetime import timedelta
from datetime import date

date_format = "%Y-%m-%d"

# VFINX - Vanguard 500 Index Fund Investor Shares
# VUSTX - Vanguard Long-Term Treasury Fund Investor Shares
# symbols = ['VFINX', 'VUSTX']

# symbols = ['SPXL', 'SSO', 'VOO', 'TMF', 'UBT', 'VGLT']
# symbols = ['SPXL', 'SSO', 'VOO']
# symbols = ['VOO', 'VGLT']

# symbols = ['SPY', 'TLT']
# symbols = ['SPY', 'IEF']

# symbols = ['VFINX', 'VUSTX', 'VFITX']
# symbols = ['SPY', 'TLT', 'IEF']

# symbols = ['EWT', 'TLT']
# symbols = ['0050.TW', '00679B.TWO']

# symbols = ['VTSAX', 'VBTLX']
# symbols = ['VTI', 'BND']

# symbols = ['VT', 'BNDW']

symbols = ['IOO', 'BLK']

# ['VFINX', 'VUSTX']
# end_timestamp = int(time.time())
# start_timestamp = datetime.strptime('1986-05-19', date_format).timestamp()

# ['VFINX', 'VUSTX', 'VFITX']
# end_timestamp = int(time.time())
# start_timestamp = datetime.strptime('1991-10-28', date_format).timestamp()

# 10 years
# end_timestamp = datetime.strptime('2021-12-31', date_format).timestamp()
# start_timestamp = datetime.strptime('2012-01-01', date_format).timestamp()

# ['SPY', 'TLT']
# end_timestamp = int(time.time())
# start_timestamp = datetime.strptime('2002-07-30', date_format).timestamp()

# rate hike
# end_timestamp = datetime.strptime('2018-11-02', date_format).timestamp()
# start_timestamp = datetime.strptime('2015-01-30', date_format).timestamp()
# end_timestamp = datetime.strptime('2006-05-12', date_format).timestamp()
# start_timestamp = datetime.strptime('2004-03-23', date_format).timestamp()

# TLT VGLT Effective Duration ~ 18 yrs
# delta = timedelta(days= 365.25 * 18)
# end_timestamp = int(time.time())
# start_timestamp = datetime.strptime((date.today()-delta).isoformat(), date_format).timestamp()

# IEF BND BNDW Effective Duration ~ 8 yrs
# delta = timedelta(days= 365.25 * 8)
# end_timestamp = int(time.time())
# start_timestamp = datetime.strptime((date.today()-delta).isoformat(), date_format).timestamp()

end_timestamp = int(time.time())
start_timestamp = datetime.strptime('2000-12-09', date_format).timestamp()

consider_dividends = True

# 60 days rebalance
def kelly_criterion(symbol):
    start_str = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    end_str = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')
    data = yf.download(tickers=symbol, start=start_str, end=end_str, auto_adjust=consider_dividends)
    data.to_csv(f'{symbol}.csv')
    with open(f'{symbol}.csv') as file:
        lines = file.readlines()
    os.remove(f'{symbol}.csv')
    assert lines[0].split(',')[0] == 'Date'
    assert lines[0].split(',')[4] == 'Close'
    prices = []
    for line in lines[1:]:
        prices.append(float(line.split(',')[4]))

    prices.reverse()
    trading_days = len(prices)-60
    gain_loss_days = []
    for i in range(trading_days):
        gain_loss_days.append((prices[i]-prices[i+60])/prices[i+60])

    gain_loss_days = np.array(gain_loss_days)

    # wrong, should use continuous kelly criterion
    p = len(gain_loss_days[gain_loss_days > 0])/len(gain_loss_days)
    q = 1-p
    a = -(gain_loss_days[gain_loss_days <= 0].mean())
    b = gain_loss_days[gain_loss_days > 0].mean()
    f = (p/a)-(q/b)

    performance = prices[0] / prices[trading_days] - 1.0
    return f,performance

fractions = []
sum_inverse_fraction = 0.0
performances = []

for s in symbols:
    f,p = kelly_criterion(s)
    fractions.append(f)
    performances.append(p)

for i in range(len(symbols)):
    s = symbols[i]
    f = fractions[i]
    print(f'{s} - fraction: {float(100*f):.2f}%, ratio: {float(100*(f/np.sum(fractions))):.2f}%, performance: {performances[i]*100:.2f}%')
