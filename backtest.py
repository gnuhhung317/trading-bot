import pandas as pd
import numpy as np
from binance.client import Client
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# Tạo thư mục temp nếu chưa tồn tại
if not os.path.exists('temp'):
    os.makedirs('temp')

# Cấu hình API Binance
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
client = Client(api_key, api_secret)

# Định nghĩa các tham số chiến lược
COINS = {
    'CETUSUSDT': {"leverage": 10, "quantity_precision": 0, "min_size": 1},
    # "XRPUSDT": {"leverage": 10, "quantity_precision": 1, "min_size": 0.1},
    # "ETHUSDT": {"leverage": 10, "quantity_precision": 3, "min_size": 0.001},
    # 'AAVEUSDT': {'leverage': 10, 'quantity_precision': 1, 'min_size': 0.1},
    # 'LINKUSDT': {'leverage': 10, 'quantity_precision': 2, 'min_size': 0.01},
    # 'VANAUSDT': {'leverage': 10, 'quantity_precision': 2, 'min_size': 0.01},
    # 'TAOUSDT': {'leverage': 10, 'quantity_precision': 3, 'min_size': 0.001},
    # 'TIAUSDT': {'leverage': 10, 'quantity_precision': 0, 'min_size': 1},
    # 'MKRUSDT': {'leverage': 10, 'quantity_precision': 3, 'min_size': 0.001},
    # 'LTCUSDT': {'leverage': 10, 'quantity_precision': 3, 'min_size': 0.001},
    # 'ENAUSDT': {'leverage': 10, 'quantity_precision': 0, 'min_size': 1},
    # 'NEARUSDT': {'leverage': 10, 'quantity_precision': 0, 'min_size': 1},
    # 'BNXUSDT': {'leverage': 6, 'quantity_precision': 1, 'min_size': 0.1}
}

TIMEFRAME = '5m'
HIGHER_TIMEFRAME = '1h'
RISK_PER_TRADE = 0.01
STOP_LOSS_THRESHOLD = 0.1
INITIAL_BALANCE = 10
MAX_POSITIONS = 5
TAKER_FEE = 0.0004
MAX_TOTAL_RISK = 0.05

def get_historical_data(symbol, interval, start_date, end_date):
    print(f"Tải dữ liệu cho {symbol} ({interval}) từ {start_date} đến {end_date}...")
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    klines = client.get_historical_klines(symbol, interval, start_ts, end_ts, limit=1000)
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'trades', 
                                       'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def add_signal_indicators(df):
    if df.empty:
        return df
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14'].fillna(0)
    df['ema_cross_up'] = (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))
    df['ema_cross_down'] = (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    df['high_5'] = df['high'].rolling(5).max()
    df['low_5'] = df['low'].rolling(5).min()
    df['breakout_up'] = (df['close'] > df['high_5'].shift(1)) & (df['close'].shift(1) <= df['high_5'].shift(2))
    df['breakout_down'] = (df['close'] < df['low_5'].shift(1)) & (df['close'].shift(1) >= df['low_5'].shift(2))
    df['volume_ma10'] = df['volume'].rolling(10).mean()
    df['volume_increase'] = df['volume'] > df['volume_ma10'] * 1.0
    return df

def add_trend_indicators(df):
    if df.empty:
        return df
    df['ema50'] = ta.ema(df['close'], length=50).bfill()
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14'].fillna(0)
    df['di_plus'] = adx['DMP_14'].fillna(0)
    df['di_minus'] = adx['DMN_14'].fillna(0)
    df['ema50_slope'] = (df['ema50'].diff(3) / df['ema50'].shift(3) * 100).fillna(0)
    df['uptrend'] = (df['close'] > df['ema50']) & (df['ema50_slope'] > 0.05)
    df['downtrend'] = (df['close'] < df['ema50']) & (df['ema50_slope'] < -0.05)
    return df

def check_entry_conditions(df, higher_tf_df, current_idx):
    if df.empty or higher_tf_df.empty or current_idx < 1:
        return None
    current = df.iloc[current_idx]
    higher_current = higher_tf_df[higher_tf_df.index <= df.index[current_idx]].iloc[-1]
    
    long_primary = [current['ema9'] > current['ema21'], 
                    current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']]
    long_secondary = [current['rsi14'] < 70, current['volume_increase'], current['macd'] > 0, current['adx'] > 25]
    long_condition = (all(long_primary) and all(long_secondary) and 
                      (higher_current['uptrend'] or (higher_current['adx'] > 20 and higher_current['di_plus'] > higher_current['di_minus'])))
    
    short_primary = [current['ema9'] < current['ema21'], 
                     current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']]
    short_secondary = [current['rsi14'] > 30, current['volume_increase'], current['macd'] < 0, current['adx'] > 25]
    short_condition = (all(short_primary) and all(short_secondary) and 
                       (higher_current['downtrend'] or (higher_current['adx'] > 20 and higher_current['di_minus'] > higher_current['di_plus'])))
    
    return 'LONG' if long_condition else 'SHORT' if short_condition else None

def backtest_strategy(data_dict, higher_tf_dict, initial_balance=INITIAL_BALANCE):
    trades = []
    balance = initial_balance
    positions = {symbol: [] for symbol in COINS}
    all_timestamps = sorted(set.intersection(*[set(df.index) for df in data_dict.values()]))
    
    if not all_timestamps:
        return pd.DataFrame(), initial_balance, 0, 0

    for timestamp in all_timestamps[1:]:
        total_risk = sum(pos['risk_per_r'] for sym in positions for pos in positions[sym])
        
        for symbol in COINS:
            if symbol not in data_dict or timestamp not in data_dict[symbol].index:
                continue
            
            df = data_dict[symbol]
            higher_tf_df = higher_tf_dict[symbol]
            current = df.loc[timestamp]
            current_idx = df.index.get_loc(timestamp)
            current_price = current['close']
            atr = current['atr14']
            
            for i, position in enumerate(positions[symbol][:]):
                profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
                r_multiple = profit / position['risk_per_r'] if position['risk_per_r'] != 0 else 0
                time_in_position = (timestamp - position['entry_time']).total_seconds() / 3600
                
                if position['type'] == 'LONG':
                    if r_multiple > 1 and not position['breakeven_activated']:
                        position['stop_loss'] = position['entry_price']
                        position['breakeven_activated'] = True
                    
                    if r_multiple > 1 and current['ema21'] > position['stop_loss']:
                        position['stop_loss'] = current['ema21']
                    
                    if r_multiple >= 2 and not position['first_target_hit']:
                        exit_size = position['size'] * 0.2
                        position['size'] -= exit_size
                        position['first_target_hit'] = True
                        profit_partial = (current_price - position['entry_price']) * exit_size * (1 - TAKER_FEE)
                        trade = {
                            'symbol': symbol, 'type': 'LONG (Partial 20% at 2R)', 
                            'entry_time': position['entry_time'], 'exit_time': timestamp,
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'size': exit_size, 'profit': profit_partial
                        }
                        trades.append(trade)
                        balance += profit_partial
                    
                    elif r_multiple >= 4 and position['first_target_hit'] and not position['second_target_hit']:
                        exit_size = position['size'] * 0.3
                        position['size'] -= exit_size
                        position['second_target_hit'] = True
                        profit_partial = (current_price - position['entry_price']) * exit_size * (1 - TAKER_FEE)
                        trade = {
                            'symbol': symbol, 'type': 'LONG (Partial 30% at 4R)', 
                            'entry_time': position['entry_time'], 'exit_time': timestamp,
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'size': exit_size, 'profit': profit_partial
                        }
                        trades.append(trade)
                        balance += profit_partial
                    
                    exit_conditions = [
                        current_price <= position['stop_loss'],
                        current['ema_cross_down'],
                        current['macd_cross_down'],
                        current['rsi14'] > 70,
                        not higher_tf_df[higher_tf_df.index <= timestamp].iloc[-1]['uptrend'] and r_multiple > 1,
                        r_multiple >= 10,
                        r_multiple < -1 and time_in_position > 2,
                        current['volume'] < current['volume_ma10'] * 0.5 and r_multiple > 0
                    ]
                    if any(exit_conditions):
                        exit_price = current_price if current_price > position['stop_loss'] else position['stop_loss']
                        profit_final = (exit_price - position['entry_price']) * position['size'] * (1 - TAKER_FEE)
                        trade = {
                            'symbol': symbol, 'type': 'LONG (Final)', 
                            'entry_time': position['entry_time'], 'exit_time': timestamp,
                            'entry_price': position['entry_price'], 'exit_price': exit_price,
                            'size': position['size'], 'profit': profit_final
                        }
                        trades.append(trade)
                        balance += profit_final
                        positions[symbol].pop(i)
                
                else:  # SHORT
                    if r_multiple > 1 and not position['breakeven_activated']:
                        position['stop_loss'] = position['entry_price']
                        position['breakeven_activated'] = True
                    
                    if r_multiple > 1 and current['ema21'] < position['stop_loss']:
                        position['stop_loss'] = current['ema21']
                    
                    if r_multiple >= 2 and not position['first_target_hit']:
                        exit_size = position['size'] * 0.2
                        position['size'] -= exit_size
                        position['first_target_hit'] = True
                        profit_partial = (position['entry_price'] - current_price) * exit_size * (1 - TAKER_FEE)
                        trade = {
                            'symbol': symbol, 'type': 'SHORT (Partial 20% at 2R)', 
                            'entry_time': position['entry_time'], 'exit_time': timestamp,
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'size': exit_size, 'profit': profit_partial
                        }
                        trades.append(trade)
                        balance += profit_partial
                    
                    elif r_multiple >= 4 and position['first_target_hit'] and not position['second_target_hit']:
                        exit_size = position['size'] * 0.3
                        position['size'] -= exit_size
                        position['second_target_hit'] = True
                        profit_partial = (position['entry_price'] - current_price) * exit_size * (1 - TAKER_FEE)
                        trade = {
                            'symbol': symbol, 'type': 'SHORT (Partial 30% at 4R)', 
                            'entry_time': position['entry_time'], 'exit_time': timestamp,
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'size': exit_size, 'profit': profit_partial
                        }
                        trades.append(trade)
                        balance += profit_partial
                    
                    exit_conditions = [
                        current_price >= position['stop_loss'],
                        current['ema_cross_up'],
                        current['macd_cross_up'],
                        current['rsi14'] < 30,
                        not higher_tf_df[higher_tf_df.index <= timestamp].iloc[-1]['downtrend'] and r_multiple > 1,
                        r_multiple >= 10,
                        r_multiple < -1 and time_in_position > 2,
                        current['volume'] < current['volume_ma10'] * 0.5 and r_multiple > 0
                    ]
                    if any(exit_conditions):
                        exit_price = current_price if current_price < position['stop_loss'] else position['stop_loss']
                        profit_final = (position['entry_price'] - exit_price) * position['size'] * (1 - TAKER_FEE)
                        trade = {
                            'symbol': symbol, 'type': 'SHORT (Final)', 
                            'entry_time': position['entry_time'], 'exit_time': timestamp,
                            'entry_price': position['entry_price'], 'exit_price': exit_price,
                            'size': position['size'], 'profit': profit_final
                        }
                        trades.append(trade)
                        balance += profit_final
                        positions[symbol].pop(i)

        total_positions = sum(len(positions[symbol]) for symbol in COINS)
        if balance > initial_balance * STOP_LOSS_THRESHOLD and total_positions < MAX_POSITIONS:
            for symbol in COINS:
                if symbol not in data_dict or timestamp not in data_dict[symbol].index or positions[symbol]:
                    continue
                
                df = data_dict[symbol]
                higher_tf_df = higher_tf_dict[symbol]
                current_idx = df.index.get_loc(timestamp)
                current = df.iloc[current_idx]
                entry_price = current['close']
                atr = current['atr14']
                
                signal = check_entry_conditions(df, higher_tf_df, current_idx)
                if signal and total_risk + (balance * RISK_PER_TRADE) <= initial_balance * MAX_TOTAL_RISK:
                    stop_loss = entry_price - atr * 2.5 if signal == 'LONG' else entry_price + atr * 2.5
                    risk_per_r = abs(entry_price - stop_loss)
                    risk_amount = balance * RISK_PER_TRADE
                    size = (risk_amount / risk_per_r) * COINS[symbol]["leverage"]
                    size = min(size, balance * COINS[symbol]["leverage"] / entry_price * (1 - TAKER_FEE * 2))
                    size = round(size, COINS[symbol]["quantity_precision"])
                    
                    if size >= COINS[symbol]["min_size"]:
                        position = {
                            'type': signal, 'entry_time': timestamp, 'entry_price': entry_price,
                            'stop_loss': stop_loss, 'size': size, 'risk_per_r': risk_amount,
                            'breakeven_activated': False, 'first_target_hit': False, 'second_target_hit': False
                        }
                        positions[symbol].append(position)
                        total_risk += risk_amount
        
        if balance < initial_balance * STOP_LOSS_THRESHOLD:
            for symbol in COINS:
                for position in positions[symbol][:]:
                    current_price = data_dict[symbol].loc[timestamp, 'close']
                    exit_price = current_price
                    profit = (exit_price - position['entry_price']) * position['size'] * (1 - TAKER_FEE) if position['type'] == 'LONG' else (position['entry_price'] - exit_price) * position['size'] * (1 - TAKER_FEE)
                    trade = {
                        'symbol': symbol, 'type': f"{position['type']} (Forced Exit)",
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': exit_price,
                        'size': position['size'], 'profit': profit
                    }
                    trades.append(trade)
                    balance += profit
                    positions[symbol].remove(position)
            break
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['profit_pct'] = (trades_df['profit'] / initial_balance) * 100
        trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
        final_balance = initial_balance + trades_df['profit'].sum()
        profit_percent = (final_balance / initial_balance - 1) * 100
    else:
        final_balance = initial_balance
        profit_percent = 0
    
    return trades_df, final_balance, final_balance - initial_balance, profit_percent

def analyze_results(trades_df, initial_balance, final_balance, profit_percent):
    """Phân tích kết quả backtest với thống kê chi tiết cho từng coin."""
    if trades_df.empty:
        print("Không có giao dịch nào!")
        return None
    
    # Tổng quan chiến lược
    win_trades = len(trades_df[trades_df['profit'] > 0])
    loss_trades = len(trades_df[trades_df['profit'] <= 0])
    win_rate = win_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
    total_profit = trades_df['profit'].sum()
    profit_factor = trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if loss_trades > 0 else float('inf')
    max_drawdown = (trades_df['cumulative_profit'].cummax() - trades_df['cumulative_profit']).max()
    
    print("\n===== KẾT QUẢ TỔNG QUAN BACKTEST =====")
    print(f"Tổng giao dịch: {len(trades_df)}")
    print(f"Tỷ lệ thắng: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.4f} USDT")
    print(f"Tổng lợi nhuận: {total_profit:.4f} USDT")
    print(f"Số dư ban đầu: {initial_balance} USDT")
    print(f"Số dư cuối: {final_balance:.4f} USDT")
    print(f"Lợi nhuận %: {profit_percent:.2f}%")

    # Thống kê chi tiết cho từng coin
    print("\n===== THỐNG KÊ CHI TIẾT THEO COIN =====")
    for symbol in COINS:
        coin_trades = trades_df[trades_df['symbol'] == symbol]
        if coin_trades.empty:
            print(f"{symbol}: Không có giao dịch")
            continue
        
        # Tính toán các chỉ số
        total_trades = len(coin_trades)
        win_trades = len(coin_trades[coin_trades['profit'] > 0])
        loss_trades = len(coin_trades[coin_trades['profit'] <= 0])
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
        total_profit = coin_trades['profit'].sum()
        avg_profit = coin_trades['profit'].mean()
        gross_profit = coin_trades[coin_trades['profit'] > 0]['profit'].sum() if win_trades > 0 else 0
        gross_loss = abs(coin_trades[coin_trades['profit'] < 0]['profit'].sum()) if loss_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        coin_cumulative_profit = coin_trades['profit'].cumsum()
        max_drawdown = (coin_cumulative_profit.cummax() - coin_cumulative_profit).max() if not coin_cumulative_profit.empty else 0
        
        print(f"\n{symbol}:")
        print(f"  Tổng giao dịch: {total_trades} (Thắng: {win_trades}, Thua: {loss_trades})")
        print(f"  Tỷ lệ thắng: {win_rate:.2f}%")
        print(f"  Tổng lợi nhuận: {total_profit:.4f} USDT")
        print(f"  Lợi nhuận trung bình mỗi giao dịch: {avg_profit:.4f} USDT")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.4f} USDT")

def plot_trades(data_dict, trades_df, start_date, end_date):
    if trades_df.empty:
        return
    
    profit_df = pd.DataFrame(index=sorted(set.union(*[set(df.index) for df in data_dict.values()])))
    profit_df['cumulative_profit'] = 0.0
    for _, trade in trades_df.iterrows():
        if pd.notna(trade['exit_time']):
            profit_df.loc[trade['exit_time'], 'cumulative_profit'] += trade['profit']
    profit_df['cumulative_profit'] = profit_df['cumulative_profit'].cumsum()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    for symbol, df in data_dict.items():
        ax1.plot(df.index, df['close'], label=f'{symbol} Price', alpha=0.5)
    
    for _, trade in trades_df.iterrows():
        symbol = trade['symbol']
        color = 'g' if 'LONG' in trade['type'] else 'r'
        ax1.scatter(trade['entry_time'], trade['entry_price'], marker='^' if 'LONG' in trade['type'] else 'v', color=color)
        ax1.scatter(trade['exit_time'], trade['exit_price'], marker='x', color='g' if trade['profit'] > 0 else 'r')
    
    ax1.set_title('Price Chart')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(profit_df.index, profit_df['cumulative_profit'], color='blue')
    ax2.set_title('Cumulative P&L')
    ax2.grid(True)
    
    plt.savefig(f'temp/backtest_{start_date}_{end_date}.png')
    plt.close()
    print(f"Đã lưu biểu đồ tại: temp/backtest_{start_date}_{end_date}.png")

def run_backtest(start_date, end_date):
    data_dict = {}
    higher_tf_dict = {}
    for symbol in COINS:
        df = get_historical_data(symbol, TIMEFRAME, start_date, end_date)
        higher_tf_df = get_historical_data(symbol, HIGHER_TIMEFRAME, start_date, end_date)
        if not df.empty and not higher_tf_df.empty:
            data_dict[symbol] = add_signal_indicators(df)
            higher_tf_dict[symbol] = add_trend_indicators(higher_tf_df)
    
    if not data_dict:
        print("Không có dữ liệu để backtest!")
        return
    
    trades_df, final_balance, profit, profit_percent = backtest_strategy(data_dict, higher_tf_dict)
    analyze_results(trades_df, INITIAL_BALANCE, final_balance, profit_percent)
    plot_trades(data_dict, trades_df, start_date, end_date)

if __name__ == "__main__":
    start_date = "2025-03-01"
    end_date = "2025-03-06"
    run_backtest(start_date, end_date)