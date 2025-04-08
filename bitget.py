import time
from datetime import datetime
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os
import logging
import requests
import sys
import ccxt

load_dotenv()

# Cấu hình cơ bản
API_KEY = os.getenv("TESTNET_BITGET_ACCESS_KEY", "")
API_SECRET = os.getenv("TESTNET_BITGET_PRIVATE_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Khởi tạo client Bitget với ccxt
client = ccxt.bitget({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password':'bitget1',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # Giao dịch futures
    }
})
client.set_sandbox_mode(True)
COINS = {
    'BTCUSDT': {"leverage": 10, "quantity_precision": 4, "min_size": 0.0001},
    'DOGEUSDT': {"leverage": 10, "quantity_precision": 3, "min_size": 0.001},
    "ETHUSDT": {"leverage": 10, "quantity_precision": 3, "min_size": 0.001},
}
markets = client.load_markets()

symbol = 'ETHUSDT'
client.set_leverage(
                leverage=COINS[symbol]["leverage"],
                symbol=f"{symbol}",
                params={'marginMode': 'cross',"productType":"USDT-FUTURES"}
            )
TIMEFRAME = '5m'
HIGHER_TIMEFRAME = '1h'
RISK_PER_TRADE = 0.01
STOP_LOSS_THRESHOLD = 0.1
MAX_POSITIONS = 5
TAKER_FEE = 0.0004
MAX_TOTAL_RISK = 0.05

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

def get_symbol_precision(symbol):
    try:
        markets = client.fetch_markets()
        for market in markets:
            if market['symbol'] == f"{symbol}_PERP":  # Bitget dùng _PERP cho futures
                return market['precision']['price']
        logging.warning(f"Không tìm thấy thông tin precision cho {symbol}")
        return 0
    except Exception as e:
        logging.error(f"Lỗi lấy precision cho {symbol}: {e}")
        return 0

for symbol in COINS:
    COINS[symbol]['price_precision'] = get_symbol_precision(symbol)

def round_to_precision(symbol, size, value_type='quantity'):
    if value_type == 'quantity':
        precision = COINS[symbol]["quantity_precision"]
    elif value_type == 'price':
        precision = COINS[symbol]["price_precision"]
    rounded_value = round(size, precision)
    if precision == 0:
        rounded_value = int(rounded_value)
    return rounded_value

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
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            logging.error(f"Lỗi gửi tin nhắn Telegram (lần {attempt+1}): {e} - {line_number}")
            time.sleep(2)
    logging.critical("Telegram không hoạt động sau 3 lần thử!")
    return None

def get_historical_data(symbol, interval, limit=1000):
    for attempt in range(3):
        try:
            ohlcv = client.fetch_ohlcv(f"{symbol}_PERP", timeframe=interval, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            logging.debug(f"Lấy dữ liệu {symbol} ({interval}) thành công, {len(df)} nến")
            return df
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            logging.error(f"Lỗi lấy dữ liệu {symbol} (lần {attempt+1}): {e} - {line_number}")
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
    df['volume_increase'] = df['volume'] > df['volume_ma10'] * 1
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

def check_entry_conditions(df, higher_tf_df, symbol):
    if df.empty or higher_tf_df.empty:
        logging.warning("Dữ liệu rỗng, bỏ qua kiểm tra điều kiện vào lệnh")
        return None
    current = df.iloc[-1]
    higher_current = higher_tf_df.iloc[-1]
    
    long_primary = [current['ema9'] > current['ema21'], current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']]
    long_secondary = [current['rsi14'] < 70, current['volume_increase'], current['macd'] > 0, current['adx'] > 25]
    long_condition = (all(long_primary) and all(long_secondary) and 
                      (higher_current['uptrend'] or (higher_current['adx'] > 20 and higher_current['di_plus'] > higher_current['di_minus'])))
    
    short_primary = [current['ema9'] < current['ema21'], current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']]
    short_secondary = [current['rsi14'] > 30, current['volume_increase'], current['macd'] < 0, current['adx'] > 25]
    short_condition = (all(short_primary) and all(short_secondary) and 
                       (higher_current['downtrend'] or (higher_current['adx'] > 20 and higher_current['di_minus'] > higher_current['di_plus'])))
    
    signal = 'LONG' if long_condition else 'SHORT' if short_condition else None
    if signal:
        logging.info(f"Tín hiệu vào lệnh {signal} cho {symbol}")
    return signal

def enter_position(symbol, signal):
    try:
        account_info = client.fetch_balance({'type': 'swap'})
        total_wallet_balance = float(account_info['total']['USDT'])
        available_margin = float(account_info['free']['USDT'])
        max_margin_per_coin = total_wallet_balance * 0.3
        
        risk_amount = available_margin * RISK_PER_TRADE
        if available_margin < risk_amount:
            logging.warning(f"{symbol} - Không đủ margin khả dụng ({available_margin} < {risk_amount})")
            return
        
        total_risk = sum(pos['risk_per_r'] for sym in positions for pos in positions[sym])
        if total_risk + risk_amount > balance * MAX_TOTAL_RISK:
            logging.warning(f"{symbol} - Vượt quá giới hạn tổng rủi ro 5%")
            return
        
        position_info = client.fetch_positions(symbols=[f"{symbol}_PERP"])
        current_margin_used = sum(float(pos['initialMargin']) for pos in position_info if pos['symbol'] == f"{symbol}_PERP")
        
        df = get_historical_data(symbol, TIMEFRAME)
        df = add_signal_indicators(df)
        if df.empty:
            logging.warning(f"{symbol} - Không có dữ liệu để vào lệnh")
            return
        
        current = df.iloc[-1]
        ticker = client.fetch_ticker(f"{symbol}_PERP")
        entry_price = float(ticker['last'])
        atr = current['atr14']
        
        if signal == 'LONG':
            stop_loss = entry_price - atr * 2.5
        else:
            stop_loss = entry_price + atr * 2.5
        
        risk_per_r = abs(entry_price - stop_loss)
        size = (risk_amount / risk_per_r) * COINS[symbol]["leverage"]
        max_size = available_margin * COINS[symbol]["leverage"] / entry_price * (1 - TAKER_FEE * 2)
        size = min(size, max_size)
        size = round_to_precision(symbol=symbol, size=size)
        
        if size < COINS[symbol]["min_size"]:
            logging.warning(f"{symbol} - Size ({size}) nhỏ hơn min_size ({COINS[symbol]['min_size']})")
            return
        
        new_margin = (size * entry_price) / COINS[symbol]["leverage"]
        if current_margin_used + new_margin > max_margin_per_coin:
            logging.warning(f"{symbol} - Vượt quá giới hạn margin 30% ({current_margin_used + new_margin:.2f} > {max_margin_per_coin:.2f})")
            new_margin = max_margin_per_coin - current_margin_used
            if new_margin <= 0:
                return
            size = new_margin * COINS[symbol]["leverage"] / entry_price
            size = round_to_precision(symbol=symbol, size=size)
        
        side = 'buy' if signal == 'LONG' else 'sell'
        order = client.create_order(
            symbol=f"{symbol}_PERP",
            type='market',
            side=side,
            amount=size
        )
        if order and order['status'] == 'closed':
            position = {
                'id': order['id'],
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
            
            logging.info(f"{symbol} - Vào lệnh {signal}: Price={entry_price}, SL={stop_loss}, Size={size}, OrderID={order['id']}")
            message = (
                f"<b>{symbol} - {signal} Order</b>\n"
                f"Time: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Price: {entry_price:.4f}\n"
                f"Size: {size:.4f}\n"
                f"Stop Loss: {stop_loss:.4f}"
            )
            send_telegram_message(message)

    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno
        logging.error(f"{symbol} - Lỗi khi enter position {symbol}-{signal}: {e} - {line_number}")
        send_telegram_message(f"{symbol} - Lỗi khi enter position {symbol}-{signal}: {e} - {line_number}")

def manage_positions(symbol, df, higher_tf_df):
    global balance, trades
    if df.empty or higher_tf_df.empty:
        logging.warning(f"{symbol} - Dữ liệu rỗng, bỏ qua quản lý vị thế")
        return
    
    current = df.iloc[-1]
    ticker = client.fetch_ticker(f"{symbol}_PERP")
    current_price = float(ticker['last'])
    atr = current['atr14']
    volume_ma10 = current.get('volume_ma10', 0)
    
    for i, position in enumerate(positions[symbol][:]):
        profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
        r_multiple = profit / position['risk_per_r'] if position['risk_per_r'] != 0 else 0
        time_in_position = (datetime.now() - position['entry_time']).total_seconds() / 3600
        
        if position['type'] == 'LONG':
            if r_multiple > 1 and not position.get('breakeven_activated', False):
                position['stop_loss'] = position['entry_price']
                position['stop_loss'] = round_to_precision(symbol=symbol, size=position['stop_loss'], value_type='price')
                position['breakeven_activated'] = True
                logging.info(f"{symbol} - Breakeven activated at {position['entry_price']}")
            
            if r_multiple > 1 and 'ema21' in current:
                new_stop = current['ema21']
                new_stop = round_to_precision(symbol=symbol, size=new_stop, value_type='price')
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
                    client.cancel_all_orders(f"{symbol}_PERP")
                    client.create_order(
                        symbol=f"{symbol}_PERP", side='sell', type='stop',
                        amount=position['size'], stopPrice=new_stop,
                        params={'reduceOnly': True}
                    )
                    logging.info(f"{symbol} - Trailing stop updated to {new_stop} (EMA21)")
            
            if r_multiple >= 2 and not position.get('first_target_hit', False):
                exit_size = position['size'] * 0.2
                exit_size = round_to_precision(symbol=symbol, size=exit_size)
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.create_order(
                        symbol=f"{symbol}_PERP", side='sell', type='market',
                        amount=exit_size, params={'reduceOnly': True}
                    )
                    if order and order['status'] == 'closed':
                        position['size'] -= exit_size
                        position['first_target_hit'] = True
                        trade = {
                            'type': 'LONG (Partial 20% at 2R)', 'entry_time': position['entry_time'],
                            'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (current_price - position['entry_price']) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 20% LONG tại {current_price} (2R), Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - LONG Partial Exit (20% at 2R)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)
            
            elif r_multiple >= 4 and position.get('first_target_hit', False) and not position.get('second_target_hit', False):
                exit_size = position['size'] * 0.3
                exit_size = round_to_precision(symbol=symbol, size=exit_size)
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.create_order(
                        symbol=f"{symbol}_PERP", side='sell', type='market',
                        amount=exit_size, params={'reduceOnly': True}
                    )
                    if order and order['status'] == 'closed':
                        position['size'] -= exit_size
                        position['second_target_hit'] = True
                        trade = {
                            'type': 'LONG (Partial 30% at 4R)', 'entry_time': position['entry_time'],
                            'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (current_price - position['entry_price']) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 30% LONG tại {current_price} (4R), Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - LONG Partial Exit (30% at 4R)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)
            
            elif r_multiple >= 6 and position.get('second_target_hit', False) and not position.get('third_target_hit', False):
                exit_size = position['size'] * 0.3
                exit_size = round_to_precision(symbol=symbol, size=exit_size)
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.create_order(
                        symbol=f"{symbol}_PERP", side='sell', type='market',
                        amount=exit_size, params={'reduceOnly': True}
                    )
                    if order and order['status'] == 'closed':
                        position['size'] -= exit_size
                        position['third_target_hit'] = True
                        trade = {
                            'type': 'LONG (Partial 30% at 6R)', 'entry_time': position['entry_time'],
                            'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (current_price - position['entry_price']) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 30% LONG tại {current_price} (6R), Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - LONG Partial Exit (30% at 6R)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)
            
            exit_conditions = [
                current_price <= position['stop_loss'],
                current.get('ema_cross_down', False),
                current.get('macd_cross_down', False),
                current.get('rsi14', 0) > 70,
                not higher_tf_df['uptrend'].iloc[-1] and r_multiple > 1,
                r_multiple >= 10,
                r_multiple < -1 and time_in_position > 2,
                current.get('volume', 0) < volume_ma10 * 0.5 and r_multiple > 0
            ]
            if any(exit_conditions):
                client.cancel_all_orders(f"{symbol}_PERP")
                order = client.create_order(
                    symbol=f"{symbol}_PERP", side='sell', type='market',
                    amount=position['size'], params={'reduceOnly': True}
                )
                if order and order['status'] == 'closed':
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
            if r_multiple > 1 and not position.get('breakeven_activated', False):
                position['stop_loss'] = position['entry_price']
                position['stop_loss'] = round_to_precision(symbol=symbol, size=position['stop_loss'], value_type='price')
                position['breakeven_activated'] = True
                logging.info(f"{symbol} - Breakeven activated at {position['entry_price']}")
            
            if r_multiple > 1 and 'ema21' in current:
                new_stop = current['ema21']
                new_stop = round_to_precision(symbol=symbol, size=new_stop, value_type='price')
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
                    client.cancel_all_orders(f"{symbol}_PERP")
                    client.create_order(
                        symbol=f"{symbol}_PERP", side='buy', type='stop',
                        amount=position['size'], stopPrice=new_stop,
                        params={'reduceOnly': True}
                    )
                    logging.info(f"{symbol} - Trailing stop updated to {new_stop} (EMA21)")
            
            if r_multiple >= 2 and not position.get('first_target_hit', False):
                exit_size = position['size'] * 0.2
                exit_size = round_to_precision(symbol=symbol, size=exit_size)
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.create_order(
                        symbol=f"{symbol}_PERP", side='buy', type='market',
                        amount=exit_size, params={'reduceOnly': True}
                    )
                    if order and order['status'] == 'closed':
                        position['size'] -= exit_size
                        position['first_target_hit'] = True
                        trade = {
                            'type': 'SHORT (Partial 20% at 2R)', 'entry_time': position['entry_time'],
                            'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (position['entry_price'] - current_price) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 20% SHORT tại {current_price} (2R), Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - SHORT Partial Exit (20% at 2R)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)
            
            elif r_multiple >= 4 and position.get('first_target_hit', False) and not position.get('second_target_hit', False):
                exit_size = position['size'] * 0.3
                exit_size = round_to_precision(symbol=symbol, size=exit_size)
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.create_order(
                        symbol=f"{symbol}_PERP", side='buy', type='market',
                        amount=exit_size, params={'reduceOnly': True}
                    )
                    if order and order['status'] == 'closed':
                        position['size'] -= exit_size
                        position['second_target_hit'] = True
                        trade = {
                            'type': 'SHORT (Partial 30% at 4R)', 'entry_time': position['entry_time'],
                            'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (position['entry_price'] - current_price) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 30% SHORT tại {current_price} (4R), Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - SHORT Partial Exit (30% at 4R)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)
            
            elif r_multiple >= 6 and position.get('second_target_hit', False) and not position.get('third_target_hit', False):
                exit_size = position['size'] * 0.3
                exit_size = round_to_precision(symbol=symbol, size=exit_size)
                if exit_size >= COINS[symbol]["min_size"]:
                    order = client.create_order(
                        symbol=f"{symbol}_PERP", side='buy', type='market',
                        amount=exit_size, params={'reduceOnly': True}
                    )
                    if order and order['status'] == 'closed':
                        position['size'] -= exit_size
                        position['third_target_hit'] = True
                        trade = {
                            'type': 'SHORT (Partial 30% at 6R)', 'entry_time': position['entry_time'],
                            'exit_time': datetime.now(),
                            'entry_price': position['entry_price'], 'exit_price': current_price,
                            'profit': (position['entry_price'] - current_price) * exit_size * (1 - TAKER_FEE)
                        }
                        trades.append(trade)
                        balance += trade['profit']
                        logging.info(f"{symbol} - Thoát 30% SHORT tại {current_price} (6R), Profit: {trade['profit']}")
                        message = (
                            f"<b>{symbol} - SHORT Partial Exit (30% at 6R)</b>\n"
                            f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Entry Price: {position['entry_price']:.4f}\n"
                            f"Exit Price: {current_price:.4f}\n"
                            f"Size: {exit_size:.4f}\n"
                            f"Profit: {trade['profit']:.4f}"
                        )
                        send_telegram_message(message)
            
            exit_conditions = [
                current_price >= position['stop_loss'],
                current.get('ema_cross_up', False),
                current.get('macd_cross_up', False),
                current.get('rsi14', 0) < 30,
                not higher_tf_df['downtrend'].iloc[-1] and r_multiple > 1,
                r_multiple >= 10,
                r_multiple < -1 and time_in_position > 2,
                current.get('volume', 0) < volume_ma10 * 0.5 and r_multiple > 0
            ]
            if any(exit_conditions):
                client.cancel_all_orders(f"{symbol}_PERP")
                order = client.create_order(
                    symbol=f"{symbol}_PERP", side='buy', type='market',
                    amount=position['size'], params={'reduceOnly': True}
                )
                if order and order['status'] == 'closed':
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

def sync_positions_from_bitget():
    global balance, initial_balance
    try:
        account_info = client.fetch_balance({'type': 'swap'})
        balance = float(account_info['free']['USDT'])
        if initial_balance is None:
            initial_balance = balance
        logging.info(f"Khởi tạo balance: {balance}")

        all_positions = client.fetch_positions()
        positions.clear()
        positions.update({symbol: [] for symbol in COINS})
        
        for pos in all_positions:
            symbol = pos['symbol'].replace('_PERP', '')
            if symbol in COINS:
                amt = float(pos['contracts'])
                if COINS[symbol]["quantity_precision"] == 0:
                    amt = int(amt)
                if abs(amt) > 0:
                    position_type = 'LONG' if amt > 0 else 'SHORT'
                    entry_price = float(pos['entryPrice'])
                    stop_loss = entry_price * (1 - 0.02) if position_type == 'LONG' else entry_price * (1 + 0.02)
                    risk_amount = balance * RISK_PER_TRADE if balance > 0 else 0.01
                    risk_per_r = abs(entry_price - stop_loss) / risk_amount if risk_amount > 0 else 1
                    
                    position_dict = {
                        'id': None,
                        'type': position_type,
                        'entry_time': pd.to_datetime(pos.get('timestamp', 0), unit='ms'),
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
        
        logging.info(f"Đồng bộ vị thế từ Bitget thành công. Balance: {balance}")
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno
        logging.error(f"Lỗi khi đồng bộ vị thế: {e} - {line_number}")
        send_telegram_message(f"Lỗi khi đồng bộ vị thế: {e} - {line_number}")
        balance = balance if balance is not None else 0

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
    
    while True:
        try:
            now = datetime.now()
            seconds_to_next_candle = (5 - (now.minute % 5)) * 60 - now.second
            if seconds_to_next_candle > 0:
                logging.debug(f"Đợi {seconds_to_next_candle} giây đến nến tiếp theo")
                sync_positions_from_bitget()
                time.sleep(seconds_to_next_candle)
            
            account_info = client.fetch_balance({'type': 'swap'})
            balance = float(account_info['free']['USDT'])
            total_equity = float(account_info['total']['USDT'])
            logging.info(f"Số dư hiện tại: Balance={balance}, Total Equity={total_equity}")
            
            if total_equity < initial_balance * STOP_LOSS_THRESHOLD:
                logging.critical(f"Số dư dưới {STOP_LOSS_THRESHOLD*100}% ({total_equity} < {initial_balance * STOP_LOSS_THRESHOLD}). Dừng bot!")
                send_telegram_message(f"Số dư dưới ngưỡng {STOP_LOSS_THRESHOLD*100}%. Dừng bot!")
                for symbol in COINS:
                    for position in positions[symbol][:]:
                        side = 'sell' if position['type'] == 'LONG' else 'buy'
                        client.cancel_all_orders(f"{symbol}_PERP")
                        client.create_order(
                            symbol=f"{symbol}_PERP", side=side, type='market',
                            amount=position['size'], params={'reduceOnly': True}
                        )
                        positions[symbol].remove(position)
                        logging.info(f"{symbol} - Đóng vị thế {position['type']} do dừng bot")
                break
            
            total_positions = sum(len(positions[symbol]) for symbol in COINS)
            for symbol in COINS:
                try:
                    df = get_historical_data(symbol, TIMEFRAME)
                    df = add_signal_indicators(df)
                    higher_tf_df = get_historical_data(symbol, HIGHER_TIMEFRAME)
                    higher_tf_df = add_trend_indicators(higher_tf_df)
                    
                    if positions[symbol]:
                        manage_positions(symbol, df, higher_tf_df)
                    
                    if balance > initial_balance * STOP_LOSS_THRESHOLD and total_positions < MAX_POSITIONS:
                        signal = check_entry_conditions(df, higher_tf_df, symbol=symbol)
                        if signal and not positions[symbol]:
                            enter_position(symbol, signal)
                except Exception as e:
                    exc_type, exc_obj, tb = sys.exc_info()
                    line_number = tb.tb_lineno
                    logging.error(f"Lỗi trong vòng lặp với {symbol}: {e} - {line_number}")
                    send_telegram_message(f"Lỗi trong vòng lặp với {symbol}: {e} - {line_number}")
            
            if (now - last_report_time).total_seconds() >= 3600:
                send_periodic_report()
                last_report_time = now
        
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            logging.error(f"Lỗi trong vòng lặp chính: {e} - {line_number}")
            send_telegram_message(f"Lỗi trong vòng lặp chính: {e} - {line_number}")
            time.sleep(60)

if __name__ == "__main__":
    for symbol in COINS:
        try:

            client.set_leverage(
                leverage=COINS[symbol]["leverage"],
                symbol=f"{symbol}",
                params={'marginMode': 'cross',"productType":"USDT-FUTURES"}
            )
            
            logging.info(f"Đặt leverage {COINS[symbol]['leverage']} cho {symbol}")
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            logging.error(f"Lỗi đặt leverage cho {symbol}: {e} - {line_number}")
            send_telegram_message(f"Lỗi đặt leverage cho {symbol}: {e} - {line_number}")
    
    logging.info("Bắt đầu bot giao dịch...")
    send_telegram_message("Bot giao dịch đã khởi động!")
    trading_loop()