import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------
# 參數設定
# ------------------------------
SYMBOL = "SPY"            # 股票代碼
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

SHORT_MA_WINDOW = 5      # 短期移動平均窗口
LONG_MA_WINDOW = 20      # 長期移動平均窗口
ADX_WINDOW = 14           # ADX 指標窗口
ADX_THRESHOLD = 15        # 趨勢強度門檻值

STOP_LOSS_RATE = 0.08     # 停損比例：8%
INITIAL_CAPITAL = 100000  # 初始本金
TRADE_SIZE = 1            # 每次交易數量

# ------------------------------
# 下載資料函式
# ------------------------------
def download_data(symbol, start, end):
    """
    Download historical data using yfinance.
    """
    df = yf.download(symbol, start=start, end=end)
    df = df[['Close']]
    df.rename(columns={'Close': 'price'}, inplace=True)
    return df

# ------------------------------
# 計算技術指標函式
# ------------------------------
def add_indicators(df):
    """
    Calculate technical indicators:
      - Short-term and long-term moving averages,
      - ADX to confirm trend strength.
    """
    # Short and Long MA
    df['ma_short'] = df['price'].rolling(window=SHORT_MA_WINDOW, min_periods=1).mean()
    df['ma_long'] = df['price'].rolling(window=LONG_MA_WINDOW, min_periods=1).mean()
    
    # Calculate True Range (TR) needed for ADX
    df['high'] = df['price']
    df['low'] = df['price']
    df['prev_close'] = df['price'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['TR'] = df[['tr1','tr2','tr3']].max(axis=1)
    
    # 計算方向性指標 (Directional Movement)
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    # Initialize +DM 與 -DM
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Smoothing TR, +DM, -DM using exponential moving average
    df['ATR'] = df['TR'].ewm(alpha=1/ADX_WINDOW, min_periods=ADX_WINDOW).mean()
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/ADX_WINDOW, min_periods=ADX_WINDOW).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/ADX_WINDOW, min_periods=ADX_WINDOW).mean() / df['ATR'])
    
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=ADX_WINDOW, min_periods=ADX_WINDOW).mean()
    
    # 清除多餘欄位
    df.drop(['high','low','prev_close','tr1','tr2','tr3','TR','up_move','down_move','+DM','-DM','+DI','-DI','DX'], axis=1, inplace=True)
    
    df = df.iloc[ADX_WINDOW*2:]
    return df

# ------------------------------
# 訊號生成函式
# ------------------------------
def generate_signals(df):
    """
    Generate trend-following signals:
      - Buy when short MA crosses above long MA,
        AND ADX is above threshold indicating a strong trend.
      - Sell signal when short MA crosses below long MA.
    """
    df['signal'] = 0  # 初始化訊號欄位，1 為買入，-1 為賣出
    
    # 判斷均線金叉及趨勢強度
    buy_condition = (
        (df['ma_short'] > df['ma_long']) &
        (df['ma_short'].shift(1) <= df['ma_long'].shift(1)) &
        (df['ADX'] >= ADX_THRESHOLD)
    )
    
    sell_condition = (
        (df['ma_short'] < df['ma_long']) &
        (df['ma_short'].shift(1) >= df['ma_long'].shift(1))
    )
    
    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1
    
    return df

# ------------------------------
# 回測模組
# ------------------------------
def backtest(df):
    """
    Backtest the trend-following strategy with stop-loss control.
    Execution:
      - Enter long position when a buy signal is generated.
      - Exit position when a sell signal occurs or stop-loss is hit.
    """
    df = df.copy()
    df['position'] = 0        # 持倉數量
    df['cash'] = INITIAL_CAPITAL
    df['holdings'] = 0        # 持有價值
    df['total'] = INITIAL_CAPITAL
    df['trade'] = 0           # 當日交易量
    df['entry_price'] = np.nan  # 進場價格記錄

    position = 0
    entry_price = np.nan

    for i in range(1, len(df)):
        price = df['price'].iloc[i].values[0]
        signal = df['signal'].iloc[i]
        trade_qty = 0

        # 如果有多單，檢查停損
        if position > 0:
            if price < entry_price * (1 - STOP_LOSS_RATE):
                trade_qty = -position
        
        # 如無持倉並收到買進訊號，進場建立多單
        if position == 0 and signal == 1:
            trade_qty = TRADE_SIZE
        
        # 如果收到賣出訊號，則平倉
        if position > 0 and signal == -1:
            trade_qty = -position
        
        # 更新持倉與進場價格
        position += trade_qty
        if trade_qty > 0:
            entry_price = price
        elif trade_qty < 0:
            entry_price = np.nan
        
        # 更新投資組合內容
        prev_cash = df['cash'].iloc[i-1]
        df.at[df.index[i], 'position'] = position
        df.at[df.index[i], 'cash'] = prev_cash - trade_qty * price
        df.at[df.index[i], 'trade'] = trade_qty
        df.at[df.index[i], 'holdings'] = position * price
        df.at[df.index[i], 'total'] = df.loc[df.index[i], 'cash'].values[0] + df.loc[df.index[i], 'holdings'].values[0]
        df.at[df.index[i], 'entry_price'] = entry_price

    return df

# ------------------------------
# 繪圖函式
# ------------------------------
def plot_results(df):
    """
    Plot price, moving averages, and portfolio value.
    """
    plt.figure(figsize=(14, 10))
    
    # 價格與均線圖
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(df.index, df['price'], label='Price', color='black')
    plt.plot(df.index, df['ma_short'], label=f'MA ({SHORT_MA_WINDOW})', color='blue')
    plt.plot(df.index, df['ma_long'], label=f'MA ({LONG_MA_WINDOW})', color='red')
    
    # 標註買進與賣出訊號
    buys = df[df['signal'] == 1]
    sells = df[df['signal'] == -1]
    plt.scatter(buys.index, buys['price'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(sells.index, sells['price'], marker='v', color='magenta', label='Sell Signal', s=100)
    
    plt.title(f"{SYMBOL} Price with Trend-Following Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    # 投資組合總價值圖
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
# 主流程
# ------------------------------
def main():
    # 1. 下載資料
    df = download_data(SYMBOL, START_DATE, END_DATE)
    
    # 2. 加入技術指標 (移動平均與 ADX)
    df = add_indicators(df)
    
    # 3. 產生交易訊號 (利用均線交叉與趨勢確認)
    df = generate_signals(df)
    
    # 4. 執行回測 (包含停損)
    df_bt = backtest(df)
    
    # 5. 輸出最終投資組合價值
    final_value = df_bt['total'].iloc[-1]
    print(f"Final portfolio value: {final_value:.2f}")
    
    # 6. 畫圖展示結果
    plot_results(df_bt)

if __name__ == "__main__":
    main()
