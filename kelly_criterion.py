from datetime import datetime
import time
import yfinance as yf
import os
import numpy as np

date_format = "%Y-%m-%d"

# VFINX - Vanguard 500 Index Fund Investor Shares
# VUSTX - Vanguard Long-Term Treasury Fund Investor Shares
# symbols = ['VFINX', 'VUSTX']
symbols = ['SPXL', 'SSO', 'VOO', 'TMF', 'UBT', 'VGLT']
# symbols = ['SPXL', 'SSO', 'VOO']
# symbols = ['VOO', 'VGLT']


# end_timestamp = int(time.time())
# start_timestamp = datetime.strptime('1986-05-19', date_format).timestamp()

end_timestamp = datetime.strptime('2021-12-31', date_format).timestamp()
start_timestamp = datetime.strptime('2012-01-01', date_format).timestamp()

def kelly_criterion(symbol):
    start_str = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    end_str = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')
    data = yf.download(tickers=symbol, start=start_str, end=end_str)
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
    trading_days = len(prices)-1
    gain_loss_days = []
    for i in range(trading_days):
        gain_loss_days.append((prices[i]-prices[i+1])/prices[i+1])

    gain_loss_days = np.array(gain_loss_days)
    p = len(gain_loss_days[gain_loss_days > 0])/len(gain_loss_days)
    q = 1-p
    a = -(gain_loss_days[gain_loss_days <= 0].mean())
    b = gain_loss_days[gain_loss_days > 0].mean()
    f = (p/a)-(q/b)
    return f

fractions = []
sum_inverse_fraction = 0.0
for s in symbols:
    f = kelly_criterion(s)
    fractions.append(f)

for i in range(len(symbols)):
    s = symbols[i]
    f = fractions[i]
    print(f'{s} - fraction: {float(100*f):.2f}%, ratio: {float(100*(f/np.sum(fractions))):.2f}%')
