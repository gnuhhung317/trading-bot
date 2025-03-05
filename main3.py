import time
from datetime import datetime
from binance.client import Client
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os
import logging
import math
import requests
# Load biến môi trường từ file .env
load_dotenv()


# Lấy API key từ biến môi trường
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")  # Thêm biến môi trường cho Telegram token
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # Thêm biến môi trường cho Telegram chat ID
client = Client(API_KEY, API_SECRET)

# Danh sách coin và tham số thử nghiệm
COINS = {
    "1000PEPEUSDT": {"leverage": 5, "quantity_precision": 0},
    # "BTCUSDT": {"leverage": 20, "quantity_precision": 3},
    "ETHUSDT": {"leverage": 15, "quantity_precision": 2},
    # "SOLUSDT": {"leverage": 7, "quantity_precision": 1},
    "XRPUSDT": {"leverage": 12, "quantity_precision": 1},
    "BOMEUSDT": {"leverage": 5, "quantity_precision": 0},
    "ADAUSDT": {"leverage": 12, "quantity_precision": 0},
    # "ALCHUSDT": {"leverage": 5, "quantity_precision": 1},
    # "BNBUSDT": {"leverage": 15, "quantity_precision": 1},
}

TIMEFRAME = '5m'
HIGHER_TIMEFRAME = '4h'
RISK_PER_TRADE = 0.02
STOP_LOSS_THRESHOLD = 0.1

# Khởi tạo logging
logging.basicConfig(filename='real.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# Lưu trữ vị thế cho từng coin
positions = {symbol: [] for symbol in COINS}  # Ví dụ: {'ETHUSDT': [], 'SOLUSDT': [], ...}
trades = []

# Lấy số dư ban đầu và thiết lập đòn bẩy
account_info = client.futures_account()
initial_balance = float(account_info['availableBalance'])
balance = initial_balance
for symbol in COINS:
    client.futures_change_leverage(symbol=symbol, leverage=COINS[symbol]["leverage"])
print(f"Số dư ban đầu: {balance}")

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"  # Hỗ trợ định dạng HTML
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        return response.json()
    except Exception as e:
        logging.error(f"Lỗi gửi tin nhắn Telegram: {e}")
        print(f"Lỗi gửi tin nhắn Telegram: {e}")
        return None
# Hàm lấy dữ liệu lịch sử
def get_historical_data(symbol, interval, limit=500):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                        'close_time', 'quote_asset_volume', 'trades',
                                        'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logging.error(f"Lỗi lấy dữ liệu {symbol}: {e}")
        send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, f"Lỗi lấy dữ liệu {symbol}: {e}")

        return pd.DataFrame()

# Hàm thêm chỉ báo tín hiệu
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

# Hàm thêm chỉ báo xu hướng
def add_trend_indicators(df):
    if df.empty:
        return df
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema50'] = df['ema50'].bfill()
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14'].fillna(0)
    df['di_plus'] = adx['DMP_14'].fillna(0)
    df['di_minus'] = adx['DMN_14'].fillna(0)
    df['ema50_slope'] = (df['ema50'].diff(3) / df['ema50'].shift(3) * 100).fillna(0)
    df['uptrend'] = (df['close'] > df['ema50']) & (df['ema50_slope'] > 0.05)
    df['downtrend'] = (df['close'] < df['ema50']) & (df['ema50_slope'] < -0.05)
    return df

# Hàm kiểm tra điều kiện vào lệnh
def check_entry_conditions(df, higher_tf_df):
    if df.empty or higher_tf_df.empty:
        return None
    current = df.iloc[-1]
    higher_current = higher_tf_df.iloc[-1]
    
    long_primary = [current['ema9'] > current['ema21'], current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']]
    long_secondary = [current['rsi14'] < 70, current['volume_increase'], current['macd'] > 0]
    long_condition = (all(long_primary) and any(long_secondary) and 
                      (higher_current['uptrend'] or (higher_current['adx'] > 25 and higher_current['di_plus'] > higher_current['di_minus'])))
    
    short_primary = [current['ema9'] < current['ema21'], current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']]
    short_secondary = [current['rsi14'] > 30, current['volume_increase'], current['macd'] < 0]
    short_condition = (all(short_primary) and any(short_secondary) and 
                       (higher_current['downtrend'] or (higher_current['adx'] > 25 and higher_current['di_minus'] > higher_current['di_plus'])))
    
    return 'LONG' if long_condition else 'SHORT' if short_condition else None

# Hàm vào lệnh
def enter_position(symbol, signal):

    global balance
    try:
        df = get_historical_data(symbol, TIMEFRAME)
        df = add_signal_indicators(df)
        if df.empty:
            return
        
        current = df.iloc[-1]
        entry_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
        atr = current['atr14']
        
        if signal == 'LONG':
            recent_low = df['low'].iloc[-5:].min()
            stop_loss = recent_low - atr * 0.3 if recent_low < entry_price * 0.99 else entry_price - atr * 1.5
        else:
            recent_high = df['high'].iloc[-5:].max()
            stop_loss = recent_high + atr * 0.3 if recent_high > entry_price * 1.01 else entry_price + atr * 1.5
        
        risk_per_r = abs(entry_price - stop_loss)
        risk_amount = balance * RISK_PER_TRADE
        size = (risk_amount / risk_per_r) * COINS[symbol]["leverage"]
        size = min(balance* COINS[symbol]["leverage"]/entry_price*0.2,size)
        # size = max(size,5.1/entry_price)
        size = round(size,COINS[symbol]["quantity_precision"])

        side = 'BUY' if signal=='LONG' else 'SELL'
        order = client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=size)
        if order and order['status'] == 'FILLED':
            position = {
                'id': order['orderId'],
                'type': signal,
                'entry_time': datetime.now(),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'size': size,
                'risk_per_r': risk_amount,
                'breakeven_activated': False,
                'first_target_hit': False,
                'second_target_hit': False
            }
            positions[symbol].append(position)
            stop_order = client.futures_create_order(
                symbol=symbol,
                side='SELL' if signal == 'LONG' else 'BUY',
                type='STOP_MARKET',
                stopPrice=position['stop_loss'],
                quantity=position['size']
            )
            logging.info(f"{symbol} - Vào lệnh {signal} tại {entry_price}, SL: {stop_loss}, Size: {size}, OrderID: {order['orderId']}")
            print(f"{symbol} - Vào lệnh {signal} tại {entry_price}, SL: {stop_loss}, Size: {size}, OrderID: {order['orderId']}")

            # Gửi tin nhắn Telegram
            message = (
                f"<b>{symbol} - {signal} Order</b>\n"
                f"Time: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Price: {entry_price:.4f}\n"
                f"Size: {size:.4f}\n"
                f"Stop Loss: {stop_loss:.4f}"
            )
            send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
    except Exception as e:
        logging.error(f"{symbol} - Lỗi vào lệnh: {e}")
        send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, f"{symbol} - Lỗi vào lệnh: {e}")


# Hàm quản lý nhiều vị thế
def manage_positions(symbol, df, higher_tf_df):
    global balance, trades
    if df.empty or higher_tf_df.empty:
        return
    
    current = df.iloc[-1]
    current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
    atr = current['atr14']
    
    for i, position in enumerate(positions[symbol][:]):
        profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
        r_multiple = profit / position['risk_per_r'] if (position['risk_per_r'] != 0 and position['risk_per_r']) else 0
        
        if position['type'] == 'LONG':
            if r_multiple > 0.7 and not position['breakeven_activated']:
                position['stop_loss'] = position['entry_price']
                position['breakeven_activated'] = True
            
            if r_multiple > 1:
                trail_factor = min(1.5, 1 + r_multiple * 0.1)
                new_stop = current_price - atr * trail_factor
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
            
            if r_multiple >= 1.5 and not position['first_target_hit']:
                exit_size = position['size'] * 0.3
                order = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=exit_size)
                if order and order['status'] == 'FILLED':
                    position['size'] -= exit_size
                    position['first_target_hit'] = True
                    trade = {
                        'type': 'LONG (Partial 30%)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                        'entry_price': position['entry_price'], 'exit_price': current_price,
                        'profit': (current_price - position['entry_price']) * exit_size
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    logging.info(f"{symbol} - Thoát 30% LONG tại {current_price}, Profit: {trade['profit']}, OrderID: {position['id']}")
            
                    # Gửi tin nhắn Telegram
                    message = (
                        f"<b>{symbol} - LONG Partial Exit (30%)</b>\n"
                        f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Entry Price: {position['entry_price']:.4f}\n"
                        f"Exit Price: {current_price:.4f}\n"
                        f"Size: {exit_size:.4f}\n"
                        f"Profit: {trade['profit']:.4f}"
                    )
                    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
                exit_size = position['size'] * 0.5
                order = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=exit_size)
                if order and order['status'] == 'FILLED':
                    position['size'] -= exit_size
                    position['second_target_hit'] = True
                    trade = {
                        'type': 'LONG (Partial 50%)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                        'entry_price': position['entry_price'], 'exit_price': current_price,
                        'profit': (current_price - position['entry_price']) * exit_size
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    logging.info(f"{symbol} - Thoát 50% LONG tại {current_price}, Profit: {trade['profit']}, OrderID: {position['id']}")
            
                    # Gửi tin nhắn Telegram
                    message = (
                        f"<b>{symbol} - LONG Partial Exit (50%)</b>\n"
                        f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Entry Price: {position['entry_price']:.4f}\n"
                        f"Exit Price: {current_price:.4f}\n"
                        f"Size: {exit_size:.4f}\n"
                        f"Profit: {trade['profit']:.4f}"
                    )
                    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)

            exit_conditions = [
                current_price <= position['stop_loss'], current['ema_cross_down'], current['macd_cross_down'],
                current['rsi14'] > 80, not higher_tf_df['uptrend'].iloc[-1] and r_multiple > 0, r_multiple >= 4
            ]
            if any(exit_conditions):
                exit_price = current_price if current_price > position['stop_loss'] else position['stop_loss']
                order = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=position['size'])
                if order and order['status'] == 'FILLED':
                    trade = {
                        'type': 'LONG (Final)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                        'entry_price': position['entry_price'], 'exit_price': exit_price,
                        'profit': (exit_price - position['entry_price']) * position['size']
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    positions[symbol].pop(i)
                    logging.info(f"{symbol} - Thoát toàn bộ LONG tại {exit_price}, Profit: {trade['profit']}, OrderID: {position['id']}")
                    print(f"{symbol} - Thoát toàn bộ LONG tại {exit_price}, Profit: {trade['profit']}, OrderID: {position['id']}")

                    # Gửi tin nhắn Telegram
                    message = (
                        f"<b>{symbol} - LONG Full Exit</b>\n"
                        f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Entry Price: {position['entry_price']:.4f}\n"
                        f"Exit Price: {exit_price:.4f}\n"
                        f"Size: {position['size']:.4f}\n"
                        f"Profit: {trade['profit']:.4f}"
                    )
                    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
        else:  # SHORT
            if r_multiple > 0.7 and not position['breakeven_activated']:
                position['stop_loss'] = position['entry_price']
                position['breakeven_activated'] = True
            
            if r_multiple > 1:
                trail_factor = min(1.5, 1 + r_multiple * 0.1)
                new_stop = current_price + atr * trail_factor
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
            
            if r_multiple >= 1.5 and not position['first_target_hit']:
                exit_size = position['size'] * 0.3
                order = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=exit_size)
                if order and order['status'] == 'FILLED':
                    position['size'] -= exit_size
                    position['first_target_hit'] = True
                    trade = {
                        'type': 'SHORT (Partial 30%)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                        'entry_price': position['entry_price'], 'exit_price': current_price,
                        'profit': (position['entry_price'] - current_price) * exit_size
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    logging.info(f"{symbol} - Thoát 30% SHORT tại {current_price}, Profit: {trade['profit']}, OrderID: {position['id']}")
            
                    # Gửi tin nhắn Telegram
                    message = (
                        f"<b>{symbol} - SHORT Partial Exit (30%)</b>\n"
                        f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Entry Price: {position['entry_price']:.4f}\n"
                        f"Exit Price: {current_price:.4f}\n"
                        f"Size: {exit_size:.4f}\n"
                        f"Profit: {trade['profit']:.4f}"
                    )
                    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
                exit_size = position['size'] * 0.5
                order = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=exit_size)
                if order and order['status'] == 'FILLED':
                    position['size'] -= exit_size
                    position['second_target_hit'] = True
                    trade = {
                        'type': 'SHORT (Partial 50%)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                        'entry_price': position['entry_price'], 'exit_price': current_price,
                        'profit': (position['entry_price'] - current_price) * exit_size
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    logging.info(f"{symbol} - Thoát 50% SHORT tại {current_price}, Profit: {trade['profit']}, OrderID: {position['id']}")
            
                     # Gửi tin nhắn Telegram
                    message = (
                        f"<b>{symbol} - SHORT Partial Exit (50%)</b>\n"
                        f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Entry Price: {position['entry_price']:.4f}\n"
                        f"Exit Price: {current_price:.4f}\n"
                        f"Size: {exit_size:.4f}\n"
                        f"Profit: {trade['profit']:.4f}"
                    )
                    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            exit_conditions = [
                current_price >= position['stop_loss'], current['ema_cross_up'], current['macd_cross_up'],
                current['rsi14'] < 20, not higher_tf_df['downtrend'].iloc[-1] and r_multiple > 0, r_multiple >= 4
            ]
            if any(exit_conditions):
                exit_price = current_price if current_price < position['stop_loss'] else position['stop_loss']
                order = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=position['size'])
                if order and order['status'] == 'FILLED':
                    trade = {
                        'type': 'SHORT (Final)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                        'entry_price': position['entry_price'], 'exit_price': exit_price,
                        'profit': (position['entry_price'] - exit_price) * position['size']
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    positions[symbol].pop(i)
                    logging.info(f"{symbol} - Thoát toàn bộ SHORT tại {exit_price}, Profit: {trade['profit']}, OrderID: {position['id']}")
                    print(f"{symbol} - Thoát toàn bộ SHORT tại {exit_price}, Profit: {trade['profit']}, OrderID: {position['id']}")

                    # Gửi tin nhắn Telegram
                    message = (
                        f"<b>{symbol} - SHORT Full Exit</b>\n"
                        f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Entry Price: {position['entry_price']:.4f}\n"
                        f"Exit Price: {exit_price:.4f}\n"
                        f"Size: {position['size']:.4f}\n"
                        f"Profit: {trade['profit']:.4f}"
                    )
                    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
# Vòng lặp giao dịch
def trading_loop():
    global balance, initial_balance
    while True:
        try:
            now = datetime.now()
            seconds_to_next_candle = (5 - (now.minute % 5)) * 60 - now.second
            if seconds_to_next_candle > 0:
                print(f"sleep {seconds_to_next_candle} seconds")
                # time.sleep(seconds_to_next_candle)
            
            account_info = client.futures_account()
            total_balance = float(account_info['totalWalletBalance'])
            unrealized_pnl = float(account_info['totalUnrealizedProfit'])
            total_equity = total_balance + unrealized_pnl
            print(f"{now} Số dư hiện tại (bao gồm vị thế): {total_equity}")
            message = (
                f"<b>Account Info</b>\n"
                f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total Balance: {total_balance:.4f}\n"
                f"Unrealized PnL: {unrealized_pnl:.4f}\n"
                f"Total Equity: {total_equity:.4f}\n"
                f"Available Balance: {balance:.4f}\n"
                f"Open Positions: {sum(len(positions[symbol]) for symbol in COINS)}"
            )
            send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)

            if total_equity < initial_balance * 0.1:
                print(f"Số dư đã giảm dưới {STOP_LOSS_THRESHOLD*100}% số dư ban đầu. Dừng bot.")
                for symbol in COINS:
                    for position in positions[symbol][:]:
                        if position['type'] == 'LONG':
                            client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=position['size'])
                        else:
                            client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=position['size'])
                        positions[symbol].remove(position)
                break
            
            # Duyệt qua từng coin để kiểm tra tín hiệu và quản lý vị thế
            for symbol in COINS:
                df = get_historical_data(symbol, TIMEFRAME)
                df = add_signal_indicators(df)
                higher_tf_df = get_historical_data(symbol, HIGHER_TIMEFRAME)
                higher_tf_df = add_trend_indicators(higher_tf_df)
                
                if positions[symbol]:
                    manage_positions(symbol, df, higher_tf_df)
                
                if balance > initial_balance * 0.1:
                    signal = check_entry_conditions(df, higher_tf_df)
                    print(f"{symbol} - {signal}")
                    if signal:
                        enter_position(symbol, signal)
            
        except Exception as e:
            logging.error(f"Lỗi trong vòng lặp: {e}")
            print(f"Lỗi: {e}")
            send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, f"Lỗi trong vòng lặp: {e}")

            time.sleep(60)
def sync_positions_from_binance():
    """
    Đồng bộ các vị thế mở từ Binance Futures về biến local `positions`.
    API client.futures_position_information() trả về danh sách tất cả các vị thế,
    sau đó ta lọc theo các symbol có trong danh sách COINS.
    """
    try:
        # Lấy danh sách tất cả vị thế từ Binance Futures
        all_positions = client.futures_position_information()
        for pos in all_positions:
            symbol = pos.get('symbol')

            # Chỉ đồng bộ các symbol đang giao dịch
            if symbol in COINS:
                amt = float(pos.get('positionAmt', 0))
                
                if abs(amt) > 0:  # Nếu có vị thế mở
                    position_type = 'LONG' if amt > 0 else 'SHORT'
                    entry_price = float(pos.get('entryPrice', 0))

                    # Định nghĩa Stop Loss (Ví dụ: 2% từ entry price)
                    if position_type == 'LONG':
                        stop_loss = entry_price * (1 - RISK_PER_TRADE)
                    else:  # SHORT
                        stop_loss = entry_price * (1 + RISK_PER_TRADE)

                    # Giả sử risk_amount là số tiền rủi ro cho mỗi giao dịch
                    risk_amount = balance*RISK_PER_TRADE  # Cố định hoặc tính theo vốn
                    risk_per_r = abs(entry_price - stop_loss) / risk_amount

                    # Tạo dictionary cho vị thế
                    position_dict = {
                        'id': None,
                        'type': position_type,
                        'entry_time': None,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'size': abs(amt),
                        'risk_per_r': risk_per_r,  # Auto-calculated
                        'breakeven_activated': False,
                        'first_target_hit': False,
                        'second_target_hit': False
                    }
                    positions[symbol].append(position_dict)

        print("Đồng bộ vị thế từ Binance thành công!")

    except Exception as e:
        logging.error(f"Lỗi khi đồng bộ vị thế: {e}")
        send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, f"Lỗi khi đồng bộ vị thế: {e}")

if __name__ == "__main__":
    # send_telegram_message(TELEGRAM_TOKEN,TELEGRAM_CHAT_ID,"hi anh em")
    sync_positions_from_binance()
    trading_loop()
