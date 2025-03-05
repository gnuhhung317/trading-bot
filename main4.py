import time
from datetime import datetime
from binance.client import Client
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os
import logging
import requests

load_dotenv()

# Cấu hình cơ bản
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
client = Client(API_KEY, API_SECRET)

COINS = {
    "ETHUSDT": {"leverage": 5, "quantity_precision": 2, "min_size": 0.001},     # Giữ nguyên, coin lớn ổn định
    "XRPUSDT": {"leverage": 5, "quantity_precision": 1, "min_size": 1},         # Giữ nguyên, biến động trung bình
    "ADAUSDT": {"leverage": 5, "quantity_precision": 0, "min_size": 1},         # Giữ nguyên, ổn định trung bình
    "SOLUSDT": {"leverage": 5, "quantity_precision": 1, "min_size": 0.01},      # Thêm, coin lớn, biến động cao
    "NEARUSDT": {"leverage": 5, "quantity_precision": 1, "min_size": 0.1},      # Thêm, layer-1, tiềm năng tăng trưởng
    "LINKUSDT": {"leverage": 5, "quantity_precision": 1, "min_size": 0.1}       # Thêm, utility coin, ổn định
}

TIMEFRAME = '5m'
HIGHER_TIMEFRAME = '4h'
RISK_PER_TRADE = 0.02
STOP_LOSS_THRESHOLD = 0.1
MAX_POSITIONS = 5
TAKER_FEE = 0.0004  # 0.04% phí taker

# Logging
logging.basicConfig(
    filename='real_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

positions = {symbol: [] for symbol in COINS}
trades = []
initial_balance = None
balance = None

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            logging.info(f"Gửi tin nhắn Telegram thành công: {message[:50]}...")
            return response.json()
        except Exception as e:
            logging.error(f"Lỗi gửi tin nhắn Telegram (lần {attempt+1}): {e}")
            time.sleep(2)
    logging.critical("Telegram không hoạt động sau 3 lần thử!")
    return None

def get_historical_data(symbol, interval, limit=1000):
    for attempt in range(3):
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                               'close_time', 'quote_asset_volume', 'trades',
                                               'taker_buy_base', 'taker_buy_quote', 'ignored'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            logging.debug(f"Lấy dữ liệu {symbol} ({interval}) thành công, {len(df)} nến")
            return df
        except Exception as e:
            logging.error(f"Lỗi lấy dữ liệu {symbol} (lần {attempt+1}): {e}")
            time.sleep(1)
    send_telegram_message(f"Lỗi lấy dữ liệu {symbol}: Không thể tải sau 3 lần thử")
    return pd.DataFrame()

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

def check_entry_conditions(df, higher_tf_df,symbol):
    if df.empty or higher_tf_df.empty:
        logging.warning("Dữ liệu rỗng, bỏ qua kiểm tra điều kiện vào lệnh")
        return None
    current = df.iloc[-1]
    higher_current = higher_tf_df.iloc[-1]
    
    long_primary = [current['ema9'] > current['ema21'], current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']]
    long_secondary = [current['rsi14'] < 80, current['volume_increase'], current['macd'] > 0]
    long_condition = (all(long_primary) and any(long_secondary) and 
                      (higher_current['uptrend'] or (higher_current['adx'] > 25 and higher_current['di_plus'] > higher_current['di_minus'])))
    
    short_primary = [current['ema9'] < current['ema21'], current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']]
    short_secondary = [current['rsi14'] > 20, current['volume_increase'], current['macd'] < 0]
    short_condition = (all(short_primary) and any(short_secondary) and 
                       (higher_current['downtrend'] or (higher_current['adx'] > 25 and higher_current['di_minus'] > higher_current['di_plus'])))
    
    signal = 'LONG' if long_condition else 'SHORT' if short_condition else None
    if signal:
        logging.info(f"Tín hiệu vào lệnh {signal} cho")
    return signal

def enter_position(symbol, signal):
    global balance
    try:
        account_info = client.futures_account()
        available_margin = float(account_info['availableBalance'])
        risk_amount = balance * RISK_PER_TRADE
        if available_margin < risk_amount:
            logging.warning(f"{symbol} - Không đủ margin khả dụng ({available_margin} < {risk_amount})")
            return
        
        df = get_historical_data(symbol, TIMEFRAME)
        df.name = symbol  # Gán tên để logging
        df = add_signal_indicators(df)
        if df.empty:
            logging.warning(f"{symbol} - Không có dữ liệu để vào lệnh")
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
        size = (risk_amount / risk_per_r) * COINS[symbol]["leverage"]
        max_size = available_margin * COINS[symbol]["leverage"] / entry_price * (1 - TAKER_FEE * 2)  # Trừ phí
        size = min(size, max_size)
        size = round(size, COINS[symbol]["quantity_precision"])
        
        if size < COINS[symbol]["min_size"]:
            logging.warning(f"{symbol} - Size ({size}) nhỏ hơn min_size ({COINS[symbol]['min_size']})")
            return
        
        side = 'BUY' if signal == 'LONG' else 'SELL'
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
            logging.info(f"{symbol} - Vào lệnh {signal}: Price={entry_price}, SL={stop_loss}, Size={size}, OrderID={order['orderId']}")
            message = (
                f"<b>{symbol} - {signal} Order</b>\n"
                f"Time: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Price: {entry_price:.4f}\n"
                f"Size: {size:.4f}\n"
                f"Stop Loss: {stop_loss:.4f}"
            )
            send_telegram_message(message)
    except Exception as e:
        logging.error(f"{symbol} - Lỗi vào lệnh: {e}")
        send_telegram_message(f"{symbol} - Lỗi vào lệnh: {e}")

def manage_positions(symbol, df, higher_tf_df):
    global balance, trades
    if df.empty or higher_tf_df.empty:
        logging.warning(f"{symbol} - Dữ liệu rỗng, bỏ qua quản lý vị thế")
        return
    
    current = df.iloc[-1]
    current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
    atr = current['atr14']
    
    for i, position in enumerate(positions[symbol][:]):
        profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
        r_multiple = profit / position['risk_per_r'] if position['risk_per_r'] != 0 else 0
        
        if position['type'] == 'LONG':
            if r_multiple > 0.7 and not position['breakeven_activated']:
                position['stop_loss'] = position['entry_price']
                position['breakeven_activated'] = True
                logging.info(f"{symbol} - Breakeven activated at {position['entry_price']}")
            
            if r_multiple > 1:
                trail_factor = min(1.5, 1 + r_multiple * 0.1)
                new_stop = current_price - atr * trail_factor
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
                    client.futures_cancel_all_open_orders(symbol=symbol)
                    client.futures_create_order(
                        symbol=symbol, side='SELL', type='STOP_MARKET',
                        stopPrice=new_stop, quantity=position['size']
                    )
                    logging.info(f"{symbol} - Trailing stop updated to {new_stop}")
            
            if r_multiple >= 1.5 and not position['first_target_hit']:
                exit_size = position['size'] * 0.3
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=exit_size)
                    if order and order['status'] == 'FILLED':
                        position['size'] -= exit_size
                        position['first_target_hit'] = True
                        trade = {
                            'type': 'LONG (Partial 30%)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (current_price - position['entry_price']) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 30% LONG tại {current_price}, Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - LONG Partial Exit (30%)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)
            elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
                exit_size = position['size'] * 0.5
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=exit_size)
                    if order and order['status'] == 'FILLED':
                        position['size'] -= exit_size
                        position['second_target_hit'] = True
                        trade = {
                            'type': 'LONG (Partial 50%)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (current_price - position['entry_price']) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 50% LONG tại {current_price}, Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - LONG Partial Exit (50%)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)

            exit_conditions = [
                current_price <= position['stop_loss'], current['ema_cross_down'], current['macd_cross_down'],
                current['rsi14'] > 80, not higher_tf_df['uptrend'].iloc[-1] and r_multiple > 0, r_multiple >= 4
            ]
            if any(exit_conditions):
                client.futures_cancel_all_open_orders(symbol=symbol)
                order = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=position['size'])
                if order and order['status'] == 'FILLED':
                    exit_price = current_price if current_price > position['stop_loss'] else position['stop_loss']
                    trade = {
                        'type': 'LONG (Final)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                        'entry_price': position['entry_price'], 'exit_price': exit_price,
                        'profit': (exit_price - position['entry_price']) * position['size'] * (1 - TAKER_FEE)
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    positions[symbol].pop(i)
                    logging.info(f"{symbol} - Thoát toàn bộ LONG tại {exit_price}, Profit: {trade['profit']}")
                    message = (
                        f"<b>{symbol} - LONG Full Exit</b>\n"
                        f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Entry Price: {position['entry_price']:.4f}\n"
                        f"Exit Price: {exit_price:.4f}\n"
                        f"Size: {position['size']:.4f}\n"
                        f"Profit: {trade['profit']:.4f}"
                    )
                    send_telegram_message(message)
        else:  # SHORT
            if r_multiple > 0.7 and not position['breakeven_activated']:
                position['stop_loss'] = position['entry_price']
                position['breakeven_activated'] = True
                logging.info(f"{symbol} - Breakeven activated at {position['entry_price']}")
            
            if r_multiple > 1:
                trail_factor = min(1.5, 1 + r_multiple * 0.1)
                new_stop = current_price + atr * trail_factor
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
                    client.futures_cancel_all_open_orders(symbol=symbol)
                    client.futures_create_order(
                        symbol=symbol, side='BUY', type='STOP_MARKET',
                        stopPrice=new_stop, quantity=position['size']
                    )
                    logging.info(f"{symbol} - Trailing stop updated to {new_stop}")
            
            if r_multiple >= 1.5 and not position['first_target_hit']:
                exit_size = position['size'] * 0.3
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=exit_size)
                    if order and order['status'] == 'FILLED':
                        position['size'] -= exit_size
                        position['first_target_hit'] = True
                        trade = {
                            'type': 'SHORT (Partial 30%)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (position['entry_price'] - current_price) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 30% SHORT tại {current_price}, Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - SHORT Partial Exit (30%)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)
            elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
                exit_size = position['size'] * 0.5
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=exit_size)
                    if order and order['status'] == 'FILLED':
                        position['size'] -= exit_size
                        position['second_target_hit'] = True
                        trade = {
                            'type': 'SHORT (Partial 50%)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (position['entry_price'] - current_price) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 50% SHORT tại {current_price}, Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - SHORT Partial Exit (50%)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)

            exit_conditions = [
                current_price >= position['stop_loss'], current['ema_cross_up'], current['macd_cross_up'],
                current['rsi14'] < 20, not higher_tf_df['downtrend'].iloc[-1] and r_multiple > 0, r_multiple >= 4
            ]
            if any(exit_conditions):
                client.futures_cancel_all_open_orders(symbol=symbol)
                order = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=position['size'])
                if order and order['status'] == 'FILLED':
                    exit_price = current_price if current_price < position['stop_loss'] else position['stop_loss']
                    trade = {
                        'type': 'SHORT (Final)', 'entry_time': position['entry_time'], 'exit_time': datetime.now(),
                        'entry_price': position['entry_price'], 'exit_price': exit_price,
                        'profit': (position['entry_price'] - exit_price) * position['size'] * (1 - TAKER_FEE)
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    positions[symbol].pop(i)
                    logging.info(f"{symbol} - Thoát toàn bộ SHORT tại {exit_price}, Profit: {trade['profit']}")
                    message = (
                        f"<b>{symbol} - SHORT Full Exit</b>\n"
                        f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Entry Price: {position['entry_price']:.4f}\n"
                        f"Exit Price: {exit_price:.4f}\n"
                        f"Size: {position['size']:.4f}\n"
                        f"Profit: {trade['profit']:.4f}"
                    )
                    send_telegram_message(message)

def sync_positions_from_binance():
    global balance, initial_balance
    try:
        # Lấy thông tin tài khoản trước để đảm bảo balance hợp lệ
        account_info = client.futures_account()
        balance = float(account_info.get('availableBalance', 0))  # Mặc định 0 nếu lỗi
        if initial_balance is None:
            initial_balance = balance
        logging.info(f"Khởi tạo balance: {balance}")

        all_positions = client.futures_position_information()
        positions.clear()
        positions.update({symbol: [] for symbol in COINS})
        
        for pos in all_positions:
            symbol = pos.get('symbol')
            if symbol in COINS:
                amt = float(pos.get('positionAmt', 0))
                if abs(amt) > 0:
                    position_type = 'LONG' if amt > 0 else 'SHORT'
                    entry_price = float(pos.get('entryPrice', 0))
                    stop_loss = entry_price * (1 - 0.02) if position_type == 'LONG' else entry_price * (1 + 0.02)
                    risk_amount = balance * RISK_PER_TRADE if balance > 0 else 0.01  # Tránh chia cho 0
                    risk_per_r = abs(entry_price - stop_loss) / risk_amount if risk_amount > 0 else 1
                    
                    position_dict = {
                        'id': None,
                        'type': position_type,
                        'entry_time': pd.to_datetime(pos.get('updateTime', 0), unit='ms'),
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'size': abs(amt),
                        'risk_per_r': risk_amount,
                        'breakeven_activated': False,
                        'first_target_hit': False,
                        'second_target_hit': False
                    }
                    positions[symbol].append(position_dict)
                    logging.info(f"Đồng bộ {symbol}: {position_type}, Size={abs(amt)}, Entry={entry_price}")
        
        logging.info(f"Đồng bộ vị thế từ Binance thành công. Balance: {balance}")
    except Exception as e:
        logging.error(f"Lỗi khi đồng bộ vị thế: {e}")
        send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, f"Lỗi khi đồng bộ vị thế: {e}")
        balance = balance if balance is not None else 0  # Đảm bảo balance không bị None

def send_periodic_report():
    if not trades:
        logging.info("Không có giao dịch nào để báo cáo")
        return
    
    trades_df = pd.DataFrame(trades)
    win_trades = len(trades_df[trades_df['profit'] > 0])
    loss_trades = len(trades_df[trades_df['profit'] <= 0])
    win_rate = win_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
    total_profit = trades_df['profit'].sum()
    gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum() if win_trades > 0 else 0
    gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if loss_trades > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    open_positions_details = "\n".join(
        [f"{symbol}: {len(positions[symbol])} positions" for symbol in COINS if positions[symbol]]
    ) or "Không có vị thế mở"
    message = (
        f"<b>Periodic Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</b>\n"
        f"Total Trades: {len(trades_df)}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Total Profit: {total_profit:.4f} USDT\n"
        f"Open Positions: {sum(len(positions[symbol]) for symbol in COINS)}\n"
        f"Balance: {balance:.4f} USDT\n"
        f"Positions Details:\n{open_positions_details}"
    )
    send_telegram_message(message)
    logging.info("Gửi báo cáo định kỳ thành công")

def trading_loop():
    global balance, initial_balance
    last_report_time = datetime.now()
    sync_positions_from_binance()  # Đồng bộ lúc khởi động
    
    while True:
        try:
            now = datetime.now()
            seconds_to_next_candle = (5 - (now.minute % 5)) * 60 - now.second
            if seconds_to_next_candle > 0:
                logging.debug(f"Đợi {seconds_to_next_candle} giây đến nến tiếp theo")
                time.sleep(seconds_to_next_candle)
            
            account_info = client.futures_account()
            balance = float(account_info['availableBalance'])
            total_equity = float(account_info['totalWalletBalance']) + float(account_info['totalUnrealizedProfit'])
            logging.info(f"Số dư hiện tại: Balance={balance}, Total Equity={total_equity}")
            
            if total_equity < initial_balance * STOP_LOSS_THRESHOLD:
                logging.critical(f"Số dư dưới {STOP_LOSS_THRESHOLD*100}% ({total_equity} < {initial_balance * STOP_LOSS_THRESHOLD}). Dừng bot!")
                send_telegram_message(f"Số dư dưới ngưỡng {STOP_LOSS_THRESHOLD*100}%. Dừng bot!")
                for symbol in COINS:
                    for position in positions[symbol][:]:
                        side = 'SELL' if position['type'] == 'LONG' else 'BUY'
                        client.futures_cancel_all_open_orders(symbol=symbol)
                        client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=position['size'])
                        positions[symbol].remove(position)
                        logging.info(f"{symbol} - Đóng vị thế {position['type']} do dừng bot")
                break
            
            total_positions = sum(len(positions[symbol]) for symbol in COINS)
            for symbol in COINS:
                df = get_historical_data(symbol, TIMEFRAME)
                df = add_signal_indicators(df)
                higher_tf_df = get_historical_data(symbol, HIGHER_TIMEFRAME)
                higher_tf_df = add_trend_indicators(higher_tf_df)
                
                if positions[symbol]:
                    manage_positions(symbol, df, higher_tf_df)
                
                if balance > initial_balance * STOP_LOSS_THRESHOLD and total_positions < MAX_POSITIONS:
                    signal = check_entry_conditions(df, higher_tf_df,symbol=symbol)
                    if signal and not positions[symbol]:
                        enter_position(symbol, signal)
            
            if (now - last_report_time).total_seconds() >= 3600:
                send_periodic_report()
                last_report_time = now
        
        except Exception as e:
            logging.error(f"Lỗi trong vòng lặp chính: {e}")
            send_telegram_message(f"Lỗi trong vòng lặp chính: {e}")
            time.sleep(60)

if __name__ == "__main__":
    # Khởi tạo leverage
    for symbol in COINS:
        try:
            client.futures_change_leverage(symbol=symbol, leverage=COINS[symbol]["leverage"])
            logging.info(f"Đặt leverage {COINS[symbol]['leverage']} cho {symbol}")
        except Exception as e:
            logging.error(f"Lỗi đặt leverage cho {symbol}: {e}")
            send_telegram_message(f"Lỗi đặt leverage cho {symbol}: {e}")
    
    logging.info("Bắt đầu bot giao dịch...")
    send_telegram_message("Bot giao dịch đã khởi động!")
    trading_loop()