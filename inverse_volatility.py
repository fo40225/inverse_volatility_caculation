#!/usr/local/bin/python3

# Author: Zebing Lin (https://github.com/linzebing)

from datetime import datetime, date, timedelta
import math
import numpy as np
import time
import sys
import requests
import yfinance as yf
import os

if len(sys.argv) == 1:
    # symbols = ['SPXL', 'SSO', 'VOO', 'TMF', 'UBT', 'VGLT']
    symbols = ['SPXL', 'SSO', 'VOO']
    # symbols = ['TMF', 'UBT', 'VGLT']
    # symbols = ['TYD', 'UST', 'IEF']

    # symbols = ['VOO', 'VGLT']
    # symbols = ['SPY', 'TLT']
    # symbols = ['SPY', 'IEF']

    # symbols = ['00631L.TW', '0050.TW', '00680L.TW', '00679B.TWO']
    # symbols = ['00631L.TW', '0050.TW']
    # symbols = ['00680L.TW', '00679B.TWO']
    # symbols = ['0050.TW', '00679B.TWO']

    # symbols = ['VTI', 'BND']

    # symbols = ['VT', 'BNDW']
else:
    symbols = sys.argv[1].split(',')
    for i in range(len(symbols)):
        symbols[i] = symbols[i].strip().upper()

num_trading_days_per_year = 252
window_size = 0
date_format = "%Y-%m-%d"
loss_only = False
consider_dividends = False

if window_size == 0 :
    # season
    end_timestamp = datetime.strptime('2022-06-16', date_format).timestamp()
    start_timestamp = datetime.strptime('2022-03-18', date_format).timestamp()

    # ['SPXL', 'SSO', 'VOO', 'TMF', 'UBT', 'VGLT']
    # end_timestamp = int(time.time())
    # start_timestamp = datetime.strptime('2011-01-01', date_format).timestamp()

    # '00631L.TW
    # end_timestamp = int(time.time())
    # start_timestamp = datetime.strptime('2014-10-23', date_format).timestamp()
        
    # '0050.TW'
    # end_timestamp = int(time.time())
    # start_timestamp = datetime.strptime('2008-01-02', date_format).timestamp()

    # ['00680L.TW', '00679B.TWO']
    # end_timestamp = int(time.time())
    # start_timestamp = datetime.strptime('2017-01-11', date_format).timestamp()

    # 10 years
    # end_timestamp = datetime.strptime('2021-12-31', date_format).timestamp()
    # start_timestamp = datetime.strptime('2012-01-01', date_format).timestamp()

    # ['SPY', 'TLT']
    # end_timestamp = int(time.time())
    # start_timestamp = datetime.strptime('2002-07-30', date_format).timestamp()

    # end_timestamp = int(time.time())
    # start_timestamp = datetime.strptime('1980-01-01', date_format).timestamp()
else:
    end_timestamp = int(time.time())
    start_timestamp = int(end_timestamp - (1.4 * (window_size + 1) + 4) * 86400)


def get_volatility_and_performance(symbol):
    # download_url = "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history&crumb=a7pcO//zvcW".format(symbol, start_timestamp, end_timestamp)
    # lines = requests.get(download_url, cookies={'B': 'chjes25epq9b6&b=3&s=18'}).text.strip().split('\n')
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
    volatilities_in_window = []
    if window_size == 0:
        trading_days = len(prices)-1
    else:
        trading_days = window_size

    for i in range(trading_days):
        # volatilities_in_window.append(math.log(prices[i] / prices[i+1]))
        volatilities_in_window.append((prices[i]-prices[i+1])/prices[i+1])

    if loss_only:
        volatilities_in_window = np.array(volatilities_in_window)
        volatilities_in_window = volatilities_in_window[volatilities_in_window<=0]

    # most_recent_date = datetime.strptime(lines[-1].split(',')[0], date_format).date()
    # assert (date.today() - most_recent_date).days <= 4, "today is {}, most recent trading day is {}".format(date.today(), most_recent_date)

    return np.std(volatilities_in_window, ddof = 1) * np.sqrt(num_trading_days_per_year), prices[0] / prices[trading_days] - 1.0

volatilities = []
performances = []
sum_inverse_volatility = 0.0
for symbol in symbols:
    volatility, performance = get_volatility_and_performance(symbol)
    sum_inverse_volatility += 1 / volatility
    volatilities.append(volatility)
    performances.append(performance)

print ("Portfolio: {}, as of {} (window size is {} days) from {}".format(str(symbols), datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d'), window_size, datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')))
for i in range(len(symbols)):
    print ('{} allocation ratio: {:.2f}% (anualized volatility: {:.2f}%, performance: {:.2f}%)'.format(symbols[i], float(100 / (volatilities[i] * sum_inverse_volatility)), float(volatilities[i] * 100), float(performances[i] * 100)))

