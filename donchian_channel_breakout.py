import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------
# Parameters Setting
# ------------------------------
SYMBOL = "SPY"  # Stock ticker
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

DONCHIAN_WINDOW = 20   # Donchian channel period (N days)
MOMENTUM_WINDOW = 14   # Momentum period for additional filter
STOP_LOSS_RATE = 0.10  # 10% stop-loss for long positions
TAKE_PROFIT_RATE = 0.15 # 15% take profit for long positions
INITIAL_CAPITAL = 100000   # Initial capital
TRADE_SIZE = 1         # Number of shares per trade

# ------------------------------
# Download Data Function
# ------------------------------
def download_data(symbol, start, end):
    """
    Download historical data from yfinance.
    """
    df = yf.download(symbol, start=start, end=end)
    df = df[['Close']]
    df.rename(columns={'Close': 'price'}, inplace=True)
    return df

# ------------------------------
# Technical Indicator Calculation
# ------------------------------
def add_indicators(df):
    """
    Calculate Donchian Channel and momentum indicator.
    """
    # Calculate rolling highest and lowest values for Donchian Channel
    df['donchian_high'] = df['price'].rolling(window=DONCHIAN_WINDOW, min_periods=1).max()
    df['donchian_low'] = df['price'].rolling(window=DONCHIAN_WINDOW, min_periods=1).min()
    
    # Calculate momentum as price difference over MOMENTUM_WINDOW days
    df['momentum'] = df['price'] - df['price'].shift(MOMENTUM_WINDOW)
    
    # Discard initial data with NA from momentum calculation
    df = df.iloc[MOMENTUM_WINDOW:]
    return df

# ------------------------------
# Signal Generation
# ------------------------------
def generate_signals(df):
    """
    Generate trading signals for the Donchian Breakout strategy.
    Buy Signal:
      - When price breaks above the previous day's Donchian high and momentum > 0.
    Sell Signal:
      - Sell signals are generated during backtesting via stop-loss or take profit conditions.
    """
    df['signal'] = 0  # 1 for buy, -1 for sell/exit, 0 for hold

    # Avoid look-ahead bias by using previous day's channel values
    df['prev_donchian_high'] = df['donchian_high'].shift(1)
    df['prev_donchian_low'] = df['donchian_low'].shift(1)

    # Buy signal: price > previous day's highest and momentum > 0
    buy_condition = (df['price'].iloc[:,0] > df['prev_donchian_high']) & (df['momentum'] > 0)
    df.loc[buy_condition, 'signal'] = 1
    
    return df

# ------------------------------
# Backtesting Module
# ------------------------------
def backtest(df):
    """
    Backtest the Donchian Breakout strategy.
    Execution logic:
      - When receiving a buy signal and if not already long, enter a long position.
      - When in a long position, apply stop-loss or take profit exit conditions.
    """
    df = df.copy()
    df['position'] = 0  # Position holding
    df['cash'] = INITIAL_CAPITAL
    df['holdings'] = 0  # Value of position
    df['total'] = INITIAL_CAPITAL  # Total portfolio value
    df['trade'] = 0  # Trade quantity executed for the day
    df['entry_price'] = np.nan  # Record entry price for trade

    position = 0
    entry_price = np.nan

    for i in range(1, len(df)):
        price = df['price'].iloc[i].values[0]
        signal = df['signal'].iloc[i]
        trade_qty = 0

        # If already in a long position, check exit conditions
        if position > 0:
            # Check stop-loss condition
            if price < entry_price * (1 - STOP_LOSS_RATE):
                # Sell entire position due to stop-loss
                trade_qty = -position
            # Check take profit condition
            elif price > entry_price * (1 + TAKE_PROFIT_RATE):
                # Sell entire position due to take profit
                trade_qty = -position

        # If not in position and buy signal is generated, enter long position
        if position == 0 and signal == 1:
            trade_qty = TRADE_SIZE

        # Update the position
        position += trade_qty

        # Update entry price when entering a new position
        if trade_qty > 0:
            entry_price = price
        elif trade_qty < 0:
            entry_price = np.nan

        # Portfolio accounting
        prev_cash = df['cash'].iloc[i-1]
        df.at[df.index[i], 'position'] = position
        df.at[df.index[i], 'cash'] = prev_cash - trade_qty * price
        df.at[df.index[i], 'trade'] = trade_qty
        df.at[df.index[i], 'holdings'] = position * price
        df.at[df.index[i], 'total'] = df.loc[df.index[i], 'cash'].values[0] + df.loc[df.index[i], 'holdings'].values[0]
        df.at[df.index[i], 'entry_price'] = entry_price

    return df

# ------------------------------
# Plotting Function
# ------------------------------
def plot_results(df):
    """
    Plot the price chart with Donchian Channel and trading signals,
    along with the portfolio total value over time.
    """
    plt.figure(figsize=(14, 10))
    
    # Price chart and indicators plot
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(df.index, df['price'], label='Price', color='black')
    plt.plot(df.index, df['donchian_high'], label=f'Donchian High ({DONCHIAN_WINDOW})', color='blue', linestyle='--')
    plt.plot(df.index, df['donchian_low'], label=f'Donchian Low ({DONCHIAN_WINDOW})', color='red', linestyle='--')
    
    # Mark buy signals on the chart
    buys = df[df['signal'] == 1]
    plt.scatter(buys.index, buys['price'], marker='^', color='green', label='Buy Signal', s=100)
    
    plt.title(f"{SYMBOL} Price with Donchian Breakout Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    # Portfolio value plot
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(df.index, df['total'], label='Total Portfolio Value', color='purple')
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Value")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main Execution Flow
# ------------------------------
def main():
    # 1. Download historical data
    df = download_data(SYMBOL, START_DATE, END_DATE)

    # 2. Calculate technical indicators
    df = add_indicators(df)

    # 3. Generate trading signals based on breakout logic
    df = generate_signals(df)

    # 4. Backtest the strategy (includes exit via stop-loss / take profit)
    df_bt = backtest(df)

    # 5. Output final portfolio value
    final_value = df_bt['total'].iloc[-1]
    print(f"Final portfolio value: {final_value:.2f}")

    # 6. Plot the results
    plot_results(df_bt)

if __name__ == "__main__":
    main()
