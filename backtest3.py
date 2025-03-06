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
    "XRPUSDT": {"leverage": 10, "quantity_precision": 1, "min_size": 0.1},  # Giảm đòn bẩy
    "ETHUSDT": {"leverage": 10, "quantity_precision": 3, "min_size": 0.001},
    'AAVEUSDT': {'leverage': 10, 'quantity_precision': 1, 'min_size': 0.1},
    'LINKUSDT': {'leverage': 10, 'quantity_precision': 2, 'min_size': 0.01},
    'VANAUSDT': {'leverage': 10, 'quantity_precision': 2, 'min_size': 0.01},
    'TAOUSDT': {'leverage': 10, 'quantity_precision': 3, 'min_size': 0.001},
    'TIAUSDT': {'leverage': 10, 'quantity_precision': 0, 'min_size': 1},
    'MKRUSDT': {'leverage': 10, 'quantity_precision': 3, 'min_size': 0.001},
    'LTCUSDT': {'leverage': 10, 'quantity_precision': 3, 'min_size': 0.001},
    'ENAUSDT': {'leverage': 10, 'quantity_precision': 0, 'min_size': 1},
    'NEARUSDT': {'leverage': 10, 'quantity_precision': 0, 'min_size': 1},
    'BNXUSDT': {'leverage': 6, 'quantity_precision': 1, 'min_size': 0.1}
}

TIMEFRAME = '5m'
HIGHER_TIMEFRAME = '15m'
RISK_PER_TRADE = 0.02  # Rủi ro 2% mỗi giao dịch
STOP_LOSS_THRESHOLD = 0.1  # Dừng bot nếu mất 90% vốn
INITIAL_BALANCE = 10  # Số dư ban đầu
MAX_POSITIONS = 5  # Giới hạn số vị thế mở đồng thời
MAX_SIZE_FACTOR = 0.5  # Giới hạn kích thước lệnh tối đa là 50% số dư

def get_historical_data(symbol, interval, start_date, end_date):
    """Tải dữ liệu lịch sử từ Binance."""
    print(f"Bắt đầu tải dữ liệu cho {symbol}...")
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    klines = []
    current = start_timestamp
    while current < end_timestamp:
        temp_klines = client.get_historical_klines(
            symbol=symbol, interval=interval,
            start_str=current,
            end_str=min(current + 1000 * 60 * 60 * 24 * 30, end_timestamp)
        )
        if not temp_klines:
            break
        klines.extend(temp_klines)
        current = temp_klines[-1][0] + 1
        time.sleep(0.5)  # Tránh bị khóa API

    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']] = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].apply(pd.to_numeric, errors='coerce')
    df.set_index('timestamp', inplace=True)
    return df

def get_higher_timeframe_data(symbol, interval, start_date, end_date):
    """Tải dữ liệu khung thời gian cao hơn (4h)."""
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    adjusted_start = (start_date_obj - timedelta(days=60)).strftime("%Y-%m-%d")
    print(f"Tải dữ liệu khung thời gian {HIGHER_TIMEFRAME} cho {symbol}...")
    return get_historical_data(symbol, HIGHER_TIMEFRAME, adjusted_start, end_date)

def add_signal_indicators(df):
    """Thêm các chỉ báo tín hiệu (EMA, RSI, MACD, ATR, v.v.)."""
    if df.empty:
        return df
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema_cross_up'] = (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))
    df['ema_cross_down'] = (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    df['high_5'] = df['high'].rolling(5).max()
    df['low_5'] = df['low'].rolling(5).min()
    df['breakout_up'] = (df['close'] > df['high_5'].shift(1)) & (df['close'].shift(1) <= df['high_5'].shift(2))
    df['breakout_down'] = (df['close'] < df['low_5'].shift(1)) & (df['close'].shift(1) >= df['low_5'].shift(2))
    df['volume_ma10'] = df['volume'].rolling(10).mean()
    df['volume_increase'] = df['volume'] > df['volume_ma10']
    return df

def add_trend_indicators(df):
    """Thêm các chỉ báo xu hướng (EMA50, ADX, v.v.)."""
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
    """Kiểm tra điều kiện vào lệnh."""
    if df.empty or higher_tf_df.empty or current_idx < 1:
        return None
    current = df.iloc[current_idx]
    higher_current = higher_tf_df[higher_tf_df.index <= df.index[current_idx]].iloc[-1]

    long_primary = [current['ema9'] > current['ema21'], 
                    current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']]
    long_secondary = [current['rsi14'] < 80, current['volume_increase'], current['macd'] > 0]
    long_condition = (all(long_primary) and any(long_secondary) and 
                      (higher_current['uptrend'] or (higher_current['adx'] > 25 and higher_current['di_plus'] > higher_current['di_minus'])))

    short_primary = [current['ema9'] < current['ema21'], 
                     current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']]
    short_secondary = [current['rsi14'] > 20, current['volume_increase'], current['macd'] < 0]
    short_condition = (all(short_primary) and any(short_secondary) and 
                       (higher_current['downtrend'] or (higher_current['adx'] > 25 and higher_current['di_minus'] > higher_current['di_plus'])))

    return 'LONG' if long_condition else 'SHORT' if short_condition else None

def backtest_multi_coin_strategy(data_dict, higher_tf_dict, initial_balance=INITIAL_BALANCE):
    """Backtest chiến lược trên nhiều coin đồng thời."""
    trades = []
    balance = initial_balance
    positions = {symbol: [] for symbol in COINS}  # Theo dõi vị thế cho từng coin

    # Tìm tập hợp thời gian chung cho tất cả các coin
    all_timestamps = sorted(set.intersection(*[set(df.index) for df in data_dict.values()]))
    if not all_timestamps:
        return pd.DataFrame(), initial_balance, 0, 0

    for timestamp in all_timestamps[1:]:
        # Quản lý các vị thế hiện có
        for symbol in COINS:
            if symbol not in data_dict or timestamp not in data_dict[symbol].index:
                continue
            
            df = data_dict[symbol]
            higher_tf_df = higher_tf_dict[symbol]
            current = df.loc[timestamp]
            current_idx = df.index.get_loc(timestamp)
            current_price = current['close']

            for i, position in enumerate(positions[symbol][:]):
                profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
                r_multiple = profit / position['risk_per_r'] if position['risk_per_r'] != 0 else 0

                if position['type'] == 'LONG':
                    if r_multiple > 0.7 and not position['breakeven_activated']:
                        position['stop_loss'] = position['entry_price']
                        position['breakeven_activated'] = True

                    if r_multiple > 1:
                        trail_factor = min(1.5, 1 + r_multiple * 0.1)
                        new_stop = current_price - current['atr14'] * trail_factor
                        position['stop_loss'] = max(position['stop_loss'], new_stop)

                    if r_multiple >= 1.5 and not position['first_target_hit']:
                        exit_size = position['size'] * 0.3
                        position['size'] -= exit_size
                        position['first_target_hit'] = True
                        trade = {
                            'symbol': symbol,
                            'type': 'LONG (Partial 30%)',
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'profit': (current_price - position['entry_price']) * exit_size,
                            'hold_time': (timestamp - position['entry_time']).total_seconds() / 3600,
                            'r_multiple': r_multiple
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        # print(f"Partial 30% Exit - {symbol}: Profit = {trade['profit']:.4f}, Balance = {balance:.4f}")

                    elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
                        exit_size = position['size'] * 0.5
                        position['size'] -= exit_size
                        position['second_target_hit'] = True
                        trade = {
                            'symbol': symbol,
                            'type': 'LONG (Partial 50%)',
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'profit': (current_price - position['entry_price']) * exit_size,
                            'hold_time': (timestamp - position['entry_time']).total_seconds() / 3600,
                            'r_multiple': r_multiple
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        # print(f"Partial 50% Exit - {symbol}: Profit = {trade['profit']:.4f}, Balance = {balance:.4f}")

                    exit_conditions = [
                        current_price <= position['stop_loss'],
                        current['ema_cross_down'],
                        current['macd_cross_down'],
                        current['rsi14'] > 80,
                        not higher_tf_df[higher_tf_df.index <= timestamp].iloc[-1]['uptrend'] and r_multiple > 0,
                        r_multiple >= 4
                    ]
                    if any(exit_conditions):
                        exit_price = max(current_price, position['stop_loss'])  # Đảm bảo không âm
                        trade = {
                            'symbol': symbol,
                            'type': 'LONG (Final)',
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'profit': (exit_price - position['entry_price']) * position['size'],
                            'hold_time': (timestamp - position['entry_time']).total_seconds() / 3600,
                            'r_multiple': r_multiple
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        positions[symbol].pop(i)
                        # print(f"Final Exit - {symbol}: Profit = {trade['profit']:.4f}, Balance = {balance:.4f}")

                else:  # SHORT
                    if r_multiple > 0.7 and not position['breakeven_activated']:
                        position['stop_loss'] = position['entry_price']
                        position['breakeven_activated'] = True

                    if r_multiple > 1:
                        trail_factor = min(1.5, 1 + r_multiple * 0.1)
                        new_stop = current_price + current['atr14'] * trail_factor
                        position['stop_loss'] = min(position['stop_loss'], new_stop)

                    if r_multiple >= 1.5 and not position['first_target_hit']:
                        exit_size = position['size'] * 0.3
                        position['size'] -= exit_size
                        position['first_target_hit'] = True
                        trade = {
                            'symbol': symbol,
                            'type': 'SHORT (Partial 30%)',
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'profit': (position['entry_price'] - current_price) * exit_size,
                            'hold_time': (timestamp - position['entry_time']).total_seconds() / 3600,
                            'r_multiple': r_multiple
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        # print(f"Partial 30% Exit - {symbol}: Profit = {trade['profit']:.4f}, Balance = {balance:.4f}")

                    elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
                        exit_size = position['size'] * 0.5
                        position['size'] -= exit_size
                        position['second_target_hit'] = True
                        trade = {
                            'symbol': symbol,
                            'type': 'SHORT (Partial 50%)',
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'profit': (position['entry_price'] - current_price) * exit_size,
                            'hold_time': (timestamp - position['entry_time']).total_seconds() / 3600,
                            'r_multiple': r_multiple
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        # print(f"Partial 50% Exit - {symbol}: Profit = {trade['profit']:.4f}, Balance = {balance:.4f}")

                    exit_conditions = [
                        current_price >= position['stop_loss'],
                        current['ema_cross_up'],
                        current['macd_cross_up'],
                        current['rsi14'] < 20,
                        not higher_tf_df[higher_tf_df.index <= timestamp].iloc[-1]['downtrend'] and r_multiple > 0,
                        r_multiple >= 4
                    ]
                    if any(exit_conditions):
                        exit_price = min(current_price, position['stop_loss'])  # Đảm bảo không âm
                        trade = {
                            'symbol': symbol,
                            'type': 'SHORT (Final)',
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'profit': (position['entry_price'] - exit_price) * position['size'],
                            'hold_time': (timestamp - position['entry_time']).total_seconds() / 3600,
                            'r_multiple': r_multiple
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        positions[symbol].pop(i)
                        # print(f"Final Exit - {symbol}: Profit = {trade['profit']:.4f}, Balance = {balance:.4f}")

        # Kiểm tra và vào lệnh mới
        if balance > initial_balance * STOP_LOSS_THRESHOLD and sum(len(pos) for pos in positions.values()) < MAX_POSITIONS:
            for symbol in COINS:
                if symbol not in data_dict or timestamp not in data_dict[symbol].index:
                    continue
                
                df = data_dict[symbol]
                higher_tf_df = higher_tf_dict[symbol]
                current_idx = df.index.get_loc(timestamp)
                current = df.iloc[current_idx]
                entry_price = current['close']
                atr = current['atr14']
                leverage = COINS[symbol]["leverage"]

                signal = check_entry_conditions(df, higher_tf_df, current_idx)
                if signal and not positions[symbol]:  # Chỉ vào lệnh nếu chưa có vị thế cho coin này
                    if signal == 'LONG':
                        recent_low = df['low'].iloc[max(0, current_idx-5):current_idx+1].min()
                        stop_loss = recent_low - atr * 0.3 if recent_low < entry_price * 0.99 else entry_price - atr * 1.5
                    else:
                        recent_high = df['high'].iloc[max(0, current_idx-5):current_idx+1].max()
                        stop_loss = recent_high + atr * 0.3 if recent_high > entry_price * 1.01 else entry_price + atr * 1.5

                    risk_per_r = abs(entry_price - stop_loss)
                    risk_amount = balance * RISK_PER_TRADE
                    size = (risk_amount / risk_per_r) * leverage
                    size = min(size, balance * MAX_SIZE_FACTOR)  # Giới hạn size tối đa 50% số dư
                    size = min(size, balance * leverage / entry_price * 0.2)  # Giới hạn dựa trên 20% vốn với đòn bẩy
                    size = max(size, 0.01)  # Đảm bảo size không quá nhỏ
                    size = round(size, COINS[symbol]["quantity_precision"])

                    position = {
                        'type': signal,
                        'entry_time': timestamp,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'size': size,
                        'risk_per_r': risk_amount,
                        'breakeven_activated': False,
                        'first_target_hit': False,
                        'second_target_hit': False
                    }
                    positions[symbol].append(position)
                    # print(f"New Entry - {symbol}: Type = {signal}, Price = {entry_price:.4f}, Size = {size:.4f}, Balance = {balance:.4f}")

        # Dừng nếu số dư dưới ngưỡng
        if balance < initial_balance * STOP_LOSS_THRESHOLD:
            for symbol in COINS:
                for position in positions[symbol][:]:
                    current_price = data_dict[symbol].loc[timestamp, 'close']
                    if position['type'] == 'LONG':
                        profit = max(0, (current_price - position['entry_price']) * position['size'])  # Tránh lợi nhuận âm quá lớn
                    else:
                        profit = max(0, (position['entry_price'] - current_price) * position['size'])
                    trade = {
                        'symbol': symbol,
                        'type': f"{position['type']} (Forced Exit)",
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit,
                        'hold_time': (timestamp - position['entry_time']).total_seconds() / 3600,
                        'r_multiple': profit / position['risk_per_r'] if position['risk_per_r'] != 0 else 0
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    positions[symbol].remove(position)
            break

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # Tính lại profit_pct để tránh giá trị bất thường
        trades_df['profit_pct'] = (trades_df['profit'] / initial_balance).clip(lower=-100, upper=100) * 100  # Giới hạn -100% đến 100%
        trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
        trades_df['peak'] = trades_df['cumulative_profit'].cummax()
        trades_df['drawdown'] = trades_df['peak'] - trades_df['cumulative_profit']
        final_balance = initial_balance + trades_df['profit'].sum()
        profit_percent = (final_balance / initial_balance - 1) * 100
    else:
        final_balance = initial_balance
        profit_percent = 0

    return trades_df, final_balance, final_balance - initial_balance, profit_percent

def analyze_results(trades_df, initial_balance, final_balance, profit_percent):
    """Phân tích kết quả backtest và in ra báo cáo."""
    if trades_df.empty:
        print("Không có giao dịch nào!")
        return None

    win_trades = len(trades_df[trades_df['profit'] > 0])
    loss_trades = len(trades_df[trades_df['profit'] <= 0])
    win_rate = (win_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    avg_profit = trades_df['profit'].mean()
    total_profit = trades_df['profit'].sum()

    avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if win_trades > 0 else 0
    avg_loss = trades_df[trades_df['profit'] <= 0]['profit'].mean() if loss_trades > 0 else 0
    avg_win_pct = trades_df[trades_df['profit'] > 0]['profit_pct'].mean() if win_trades > 0 else 0
    avg_loss_pct = trades_df[trades_df['profit'] <= 0]['profit_pct'].mean() if loss_trades > 0 else 0
    max_win_pct = trades_df['profit_pct'].max()
    max_loss_pct = trades_df['profit_pct'].min()

    gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum() if win_trades > 0 else 0
    gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if loss_trades > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_hold_time = trades_df['hold_time'].mean() if 'hold_time' in trades_df.columns else 0

    max_drawdown = trades_df['drawdown'].max()
    max_drawdown_pct = (max_drawdown / (initial_balance + trades_df['peak'].max())) * 100 if trades_df['peak'].max() > 0 else 0

    long_trades = len(trades_df[trades_df['type'].str.contains('LONG')])
    short_trades = len(trades_df[trades_df['type'].str.contains('SHORT')])

    print("\n===== KẾT QUẢ BACKTEST =====")
    print(f"Tổng giao dịch: {len(trades_df)}")
    print(f"Thắng: {win_trades} ({win_rate:.2f}%)")
    print(f"Thua: {loss_trades}")
    print(f"Long/Short: {long_trades}/{short_trades}")
    print(f"Lợi nhuận trung bình mỗi giao dịch: {avg_profit:.4f} USDT")
    print(f"Lợi nhuận trung bình giao dịch thắng: {avg_win:.4f} USDT ({avg_win_pct:.2f}%)")
    print(f"Thua lỗ trung bình giao dịch thua: {avg_loss:.4f} USDT ({avg_loss_pct:.2f}%)")
    print(f"Lợi nhuận lớn nhất: {max_win_pct:.2f}%")
    print(f"Thua lỗ lớn nhất: {max_loss_pct:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Thời gian giữ lệnh trung bình: {avg_hold_time:.2f} giờ")
    print(f"Max Drawdown: {max_drawdown:.4f} USDT ({max_drawdown_pct:.2f}%)")
    print(f"Tổng lợi nhuận: {total_profit:.4f} USDT")
    print(f"Số dư ban đầu: {initial_balance} USDT")
    print(f"Số dư cuối: {final_balance:.4f} USDT")
    print(f"Lợi nhuận %: {profit_percent:.2f}%")

    # Phân tích theo từng coin
    print("\n===== THỐNG KÊ THEO COIN =====")
    for symbol in COINS:
        coin_trades = trades_df[trades_df['symbol'] == symbol]
        if not coin_trades.empty:
            coin_profit = coin_trades['profit'].sum()
            coin_trades_count = len(coin_trades)
            coin_win_rate = len(coin_trades[coin_trades['profit'] > 0]) / coin_trades_count * 100 if coin_trades_count > 0 else 0
            print(f"{symbol}: Giao dịch: {coin_trades_count}, Lợi nhuận: {coin_profit:.4f} USDT, Tỷ lệ thắng: {coin_win_rate:.2f}%")

    return trades_df

def plot_trades(data_dict, trades_df, start_date, end_date):
    """Vẽ biểu đồ giá và lợi nhuận tích lũy cho nhiều coin."""
    if trades_df.empty:
        return

    # Tạo DataFrame với tất cả timestamp và reset cumulative_profit
    all_timestamps = sorted(set.union(*[set(df.index) for df in data_dict.values()]))
    profit_df = pd.DataFrame(index=all_timestamps, columns=['cumulative_profit'])
    profit_df['cumulative_profit'] = 0.0

    # Cộng dồn lợi nhuận từ các giao dịch
    for _, trade in trades_df.iterrows():
        if pd.notna(trade['exit_time']):
            time_idx = profit_df.index.get_indexer([trade['exit_time']], method='nearest')[0]
            if time_idx >= 0 and time_idx < len(profit_df):
                current_profit = profit_df.loc[profit_df.index[time_idx], 'cumulative_profit']
                profit_df.loc[profit_df.index[time_idx], 'cumulative_profit'] = current_profit + trade['profit']

    profit_df['cumulative_profit'] = profit_df['cumulative_profit'].cumsum()

    # Vẽ biểu đồ
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax1 = plt.subplot(gs[0])
    for symbol, df in data_dict.items():
        ax1.plot(df.index, df['close'], label=f'{symbol} Price', linewidth=1, alpha=0.5)

    # Vẽ các điểm vào/thoát lệnh
    for _, trade in trades_df.iterrows():
        symbol = trade['symbol']
        if 'LONG' in trade['type']:
            ax1.scatter(trade['entry_time'], trade['entry_price'], marker='^', color='g', s=100, 
                        label=f'{symbol} Long Entry' if f'{symbol} Long Entry' not in ax1.get_legend_handles_labels()[1] else '')
        else:
            ax1.scatter(trade['entry_time'], trade['entry_price'], marker='v', color='r', s=100, 
                        label=f'{symbol} Short Entry' if f'{symbol} Short Entry' not in ax1.get_legend_handles_labels()[1] else '')
        
        profit_pct = trade['profit_pct']
        color = 'g' if profit_pct > 0 else 'r'
        ax1.scatter(trade['exit_time'], trade['exit_price'], marker='X', color=color, s=100, 
                    label=f'{symbol} Exit' if f'{symbol} Exit' not in ax1.get_legend_handles_labels()[1] else '')
        ax1.annotate(f'{profit_pct:.1f}%', xy=(trade['exit_time'], trade['exit_price']),
                     xytext=(10, 10), textcoords='offset points', color=color, fontsize=8)

    ax1.set_title('Price Chart - Multi Coin')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(profit_df.index, profit_df['cumulative_profit'], color='blue', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax2.fill_between(profit_df.index, profit_df['cumulative_profit'], 0, 
                     where=(profit_df['cumulative_profit'] >= 0), color='green', alpha=0.3)
    ax2.fill_between(profit_df.index, profit_df['cumulative_profit'], 0, 
                     where=(profit_df['cumulative_profit'] < 0), color='red', alpha=0.3)

    for _, trade in trades_df.iterrows():
        if pd.notna(trade['exit_time']):
            cumulative_profit = profit_df.loc[profit_df.index <= trade['exit_time'], 'cumulative_profit'].iloc[-1]
            color = 'g' if trade['profit'] > 0 else 'r'
            ax2.scatter(trade['exit_time'], cumulative_profit, color=color, s=50)

    ax2.set_title('Cumulative P&L')
    ax2.set_ylabel('Profit/Loss (USDT)')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    filename = f'temp/trades_chart_multi_coin_{start_date}_{end_date}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ giao dịch tại: {filename}")

def test_multi_coin_strategy(start_date, end_date, interval):
    """Chạy backtest cho nhiều coin đồng thời."""
    print(f"\n{'='*50}")
    print("BACKTEST CHIẾN LƯỢC ĐA COIN")
    print(f"{'='*50}")

    # Tải dữ liệu cho tất cả các coin
    data_dict = {}
    higher_tf_dict = {}
    for symbol in COINS:
        try:
            df = get_historical_data(symbol, interval, start_date, end_date)
            higher_tf_df = get_higher_timeframe_data(symbol, interval, start_date, end_date)
            if df.empty or higher_tf_df.empty:
                print(f"Không tải được dữ liệu đầy đủ cho {symbol}!")
                continue
            data_dict[symbol] = add_signal_indicators(df)
            higher_tf_dict[symbol] = add_trend_indicators(higher_tf_df)
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu cho {symbol}: {str(e)}")
            continue

    if not data_dict:
        print("Không có dữ liệu nào để backtest!")
        return

    # Chạy backtest
    try:
        trades_df, final_balance, profit, profit_percent = backtest_multi_coin_strategy(data_dict, higher_tf_dict)

        if not trades_df.empty:
            trades_df.to_csv(f'temp/trades_multi_coin_{start_date}_{end_date}.csv', index=False)
            print(f"Đã lưu giao dịch tại: temp/trades_multi_coin_{start_date}_{end_date}.csv")

            analyze_results(trades_df, INITIAL_BALANCE, final_balance, profit_percent)
            plot_trades(data_dict, trades_df, start_date, end_date)

            print("\n===== TÓM TẮT KẾT QUẢ =====")
            print(f"Tổng giao dịch: {len(trades_df)}")
            print(f"P&L: {profit:.4f} USDT")
            print(f"Lợi nhuận %: {profit_percent:.2f}%")
        else:
            print("Không có giao dịch nào được thực hiện!")
    except Exception as e:
        print(f"Lỗi khi chạy backtest: {str(e)}")
def fetch_and_filter_usdt_coins(client, min_volume=1000000, min_volatility=0.02, days=7):
    """
    Lấy và lọc các cặp USDT phù hợp với chiến lược trading.
    
    Parameters:
    - client: Binance Client instance
    - min_volume: Volume trung bình tối thiểu (USDT) trong 7 ngày
    - min_volatility: Độ biến động tối thiểu (theo ATR/close)
    - days: Số ngày để phân tích dữ liệu
    
    Returns:
    - filtered_coins: Dict chứa thông tin các coin được chọn
    """
    print("Đang kéo danh sách các cặp USDT từ Binance...")
    
    # Lấy tất cả thông tin ticker
    tickers = client.get_ticker()
    usdt_pairs = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
    
    # Tính thời gian bắt đầu
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    filtered_coins = {}
    print(f"Lọc {len(usdt_pairs)} cặp USDT dựa trên volume và volatility...")
    
    for pair in usdt_pairs:
        symbol = pair['symbol']
        try:
            # Lấy dữ liệu lịch sử 1h trong 7 ngày
            df = get_historical_data(symbol, '1h', start_str, end_str)
            
            if df.empty or len(df) < 24:  # Đảm bảo có đủ dữ liệu
                continue
                
            # Tính các chỉ số
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            avg_volume = df['quote_volume'].mean()  # Volume bằng USDT
            avg_volatility = (df['atr'] / df['close']).mean()  # Độ biến động chuẩn hóa
            price = float(pair['lastPrice'])
            
            # Lấy thông tin precision từ exchange info
            exchange_info = client.get_symbol_info(symbol)
            quantity_precision = int(1/float(exchange_info['filters'][1]['stepSize'])) #fix
            min_size = float(exchange_info['filters'][1]['minQty'])  # LOT_SIZE filter
            
            # Kiểm tra điều kiện lọc
            if (avg_volume >= min_volume and 
                avg_volatility >= min_volatility and 
                price > 0):  # Loại bỏ các coin giá quá thấp
                filtered_coins[symbol] = {
                    "leverage": 5,  # Đòn bẩy mặc định
                    "quantity_precision": quantity_precision,
                    "min_size": min_size,
                    "avg_volume": avg_volume,
                    "avg_volatility": avg_volatility,
                    "price": price
                }
                print(f"Đã chọn {symbol}: Volume = {avg_volume:,.0f} USDT, Volatility = {avg_volatility:.4f}")
                
        except Exception as e:
            print(f"Lỗi khi xử lý {symbol}: {str(e)}")
            continue
    
    # Sắp xếp theo volume và lấy top coin
    sorted_coins = dict(sorted(filtered_coins.items(), 
                             key=lambda x: x[1]['avg_volume'], 
                             reverse=True))
    
    print(f"\nĐã lọc được {len(sorted_coins)} cặp USDT phù hợp.")
    return sorted_coins

def test_filtered_coins(start_date, end_date, interval, num_coins=10):
    """Chạy backtest với các coin được lọc."""
    # Khởi tạo client
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    client = Client(api_key, api_secret)
    
    # Lấy và lọc coins
    # filtered_coins = fetch_and_filter_usdt_coins(client)
    # selected_coins = dict(list(filtered_coins.items())[:num_coins])  # Lấy top N coins
    
    # print(f"\nĐã chọn {len(selected_coins)} coin để backtest:")
    # for symbol, info in selected_coins.items():
        # print(f"- {symbol}: Volume = {info['avg_volume']:,.0f} USDT, "
            #   f"Volatility = {info['avg_volatility']:.4f}")
    
    # Cập nhật COINS trong chiến lược
    global COINS
    # COINS = selected_coins
    
    # Chạy backtest với các coin đã chọn
    test_multi_coin_strategy(start_date, end_date, interval)

# Sử dụng hàm
if __name__ == "__main__":
    start_date = "2025-01-04"  # Điều chỉnh ngày để có dữ liệu
    end_date = "2025-03-05"
    interval = Client.KLINE_INTERVAL_5MINUTE
    
    # Chạy test với top 10 coin
    test_filtered_coins(start_date, end_date, interval, num_coins=30)