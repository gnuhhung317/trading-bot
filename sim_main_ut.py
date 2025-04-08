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
    "ETHUSDT": {"leverage": 10, "quantity_precision": 3, "min_size": 0.001},
    'BTCUSDT': {'leverage': 6, 'quantity_precision': 1, 'min_size': 0.1}
}

TIMEFRAME = '5m'
RISK_PER_TRADE = 0.01
STOP_LOSS_THRESHOLD = 0.1
MAX_POSITIONS = 5
TAKER_FEE = 0.0004
MAX_TOTAL_RISK = 0.05

# Logging
logging.basicConfig(
    filename='simulated_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

positions = {symbol: [] for symbol in COINS}
trades = []
initial_balance = 100  # Số dư khởi tạo 100$
balance = initial_balance

def get_symbol_precision(symbol):
    try:
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s['pricePrecision']
        return 0
    except Exception as e:
        logging.error(f"Lỗi lấy precision cho {symbol}: {e}")
        return 0

for symbol in COINS:
    COINS[symbol]['price_precision'] = get_symbol_precision(symbol=symbol)

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
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        logging.info(f"Gửi tin nhắn Telegram thành công: {message[:50]}...")
        return response.json()
    except Exception as e:
        logging.error(f"Lỗi gửi tin nhắn Telegram: {e}")
        return None

def get_historical_data(symbol, interval, limit=1000):
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
        return pd.DataFrame()

def add_utbot_indicators(df, a=1, c=10, h=False, sma_period=200):
    if h:
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = ((df['open'].shift(1) + df['close'].shift(1)) / 2).fillna(method='ffill')
        ha_high = df[['high', 'open', 'close']].max(axis=1)
        ha_low = df[['low', 'open', 'close']].min(axis=1)
        src = ha_close
    else:
        src = df['close']

    xATR = ta.atr(df['high'], df['low'], df['close'], length=c)
    nLoss = a * xATR

    xATRTrailingStop = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop.iloc[i-1] if i > 0 else 0
        if src.iloc[i] > prev_stop and src.iloc[i-1] > prev_stop:
            xATRTrailingStop.iloc[i] = max(prev_stop, src.iloc[i] - nLoss.iloc[i])
        elif src.iloc[i] < prev_stop and src.iloc[i-1] < prev_stop:
            xATRTrailingStop.iloc[i] = min(prev_stop, src.iloc[i] + nLoss.iloc[i])
        else:
            xATRTrailingStop.iloc[i] = src.iloc[i] - nLoss.iloc[i] if src.iloc[i] > prev_stop else src.iloc[i] + nLoss.iloc[i]

    ema = ta.ema(src, length=1)
    above = (ema > xATRTrailingStop) & (ema.shift(1) <= xATRTrailingStop.shift(1))
    below = (ema < xATRTrailingStop) & (ema.shift(1) >= xATRTrailingStop.shift(1))
    sma = ta.sma(df['close'], length=sma_period)

    df['xATRTrailingStop'] = xATRTrailingStop
    df['nLoss'] = nLoss
    df['src'] = src
    df['ema'] = ema
    df['above'] = above
    df['below'] = below
    df['sma'] = sma
    df['buy'] = (src > xATRTrailingStop) & above & (df['close'] > sma)
    df['sell'] = (src < xATRTrailingStop) & below & (df['close'] < sma)

    return df
def check_entry_conditions(df, symbol):
    if df.empty:
        return None
    current = df.iloc[-1]
    if current['buy']:
        return 'LONG'
    elif current['sell']:
        return 'SHORT'
    return None

def enter_position(symbol, signal):
    global balance
    try:
        risk_amount = balance * RISK_PER_TRADE
        if balance < risk_amount:
            logging.warning(f"{symbol} - Không đủ số dư ({balance} < {risk_amount})")
            return
        
        total_risk = sum(pos['risk_per_r'] for sym in positions for pos in positions[sym])
        if total_risk + risk_amount > initial_balance * MAX_TOTAL_RISK:
            logging.warning(f"{symbol} - Vượt quá giới hạn tổng rủi ro 5%")
            return
        
        df = get_historical_data(symbol, TIMEFRAME)
        df = add_utbot_indicators(df, a=1, c=10, h=False, sma_period=200)
        if df.empty:
            return
        
        current = df.iloc[-1]
        entry_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
        nLoss = current['nLoss']
        
        position_size = risk_amount / nLoss * COINS[symbol]["leverage"]
        max_size = balance * COINS[symbol]["leverage"] / entry_price * (1 - TAKER_FEE * 2)
        position_size = min(position_size, max_size)
        position_size = round_to_precision(symbol=symbol, size=position_size)
        
        if position_size < COINS[symbol]["min_size"]:
            return
        
        # Khởi tạo trailing stop ban đầu
        if signal == 'LONG':
            trailing_stop = entry_price - nLoss  # Dưới entry price
            take_profit = entry_price + 2 * nLoss
        else:  # SHORT
            trailing_stop = entry_price + nLoss  # Trên entry price
            take_profit = entry_price - 2 * nLoss
        
        position = {
            'id': f"SIM_{symbol}_{datetime.now().timestamp()}",
            'type': signal,
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'trailing_stop': trailing_stop,  # Thay 'stop_loss' bằng 'trailing_stop'
            'take_profit': take_profit,
            'size': position_size,
            'risk_per_r': risk_amount,
            'prev_price': entry_price  # Lưu giá trước đó để tính trailing stop
        }
        positions[symbol].append(position)
        
        logging.info(f"{symbol} - [SIM] Vào lệnh {signal}: Price={entry_price}, Trailing Stop={trailing_stop}, TP={take_profit}, Size={position_size}")
        message = (
            f"<b>[SIM] {symbol} - {signal} Order</b>\n"
            f"Time: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Price: {entry_price:.4f}\n"
            f"Size: {position_size:.4f}\n"
            f"Trailing Stop: {trailing_stop:.4f}\n"
            f"Take Profit: {take_profit:.4f}"
        )
        send_telegram_message(message)

    except Exception as e:
        logging.error(f"{symbol} - Lỗi khi vào lệnh: {e}")

def manage_positions(symbol, df):
    global balance, trades
    if df.empty:
        return
    
    current = df.iloc[-1]
    current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
    nLoss = current['nLoss']  # Lấy nLoss từ dữ liệu hiện tại
    
    for i, position in enumerate(positions[symbol][:]):
        profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
        
        # Cập nhật trailing stop theo logic UT Bot
        if position['type'] == 'LONG':
            if current_price > position['prev_price'] and current_price > position['trailing_stop']:
                # Giá tăng: nâng trailing stop lên
                new_trailing_stop = max(position['trailing_stop'], current_price - nLoss)
                position['trailing_stop'] = new_trailing_stop
                logging.info(f"{symbol} - [SIM] Cập nhật Trailing Stop LONG: {new_trailing_stop}")
            # Kiểm tra thoát lệnh
            should_exit = current_price <= position['trailing_stop'] or current_price >= position['take_profit'] or current['sell']
            exit_price = position['trailing_stop'] if current_price <= position['trailing_stop'] else current_price
        
        else:  # SHORT
            if current_price < position['prev_price'] and current_price < position['trailing_stop']:
                # Giá giảm: hạ trailing stop xuống
                new_trailing_stop = min(position['trailing_stop'], current_price + nLoss)
                position['trailing_stop'] = new_trailing_stop
                logging.info(f"{symbol} - [SIM] Cập nhật Trailing Stop SHORT: {new_trailing_stop}")
            # Kiểm tra thoát lệnh
            should_exit = current_price >= position['trailing_stop'] or current_price <= position['take_profit'] or current['buy']
            exit_price = position['trailing_stop'] if current_price >= position['trailing_stop'] else current_price
        
        # Cập nhật prev_price cho lần tiếp theo
        position['prev_price'] = current_price
        
        # Thoát lệnh nếu cần
        if should_exit:
            trade = {
                'type': f"{position['type']} (Exit)",
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit': (exit_price - position['entry_price']) * position['size'] * (1 - TAKER_FEE) if position['type'] == 'LONG' else (position['entry_price'] - exit_price) * position['size'] * (1 - TAKER_FEE)
            }
            trades.append(trade)
            balance += trade['profit']
            positions[symbol].pop(i)
            logging.info(f"{symbol} - [SIM] Thoát {position['type']} tại {exit_price}, Profit: {trade['profit']}")
            message = (
                f"<b>[SIM] {symbol} - {position['type']} Exit</b>\n"
                f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Entry Price: {position['entry_price']:.4f}\n"
                f"Exit Price: {exit_price:.4f}\n"
                f"Size: {position['size']:.4f}\n"
                f"Profit: {trade['profit']:.4f}"
            )
            send_telegram_message(message)
            
def send_periodic_report():
    if not trades:
        return
    
    trades_df = pd.DataFrame(trades)
    win_trades = len(trades_df[trades_df['profit'] > 0])
    loss_trades = len(trades_df[trades_df['profit'] <= 0])
    win_rate = win_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
    total_profit = trades_df['profit'].sum()
    profit_factor = trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if loss_trades > 0 else float('inf')
    
    open_positions_details = "\n".join(
        [f"{symbol}: {len(positions[symbol])} positions" for symbol in COINS if positions[symbol]]
    ) or "Không có vị thế mở"
    message = (
        f"<b>[SIM] Periodic Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</b>\n"
        f"Total Trades: {len(trades_df)}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Total Profit: {total_profit:.4f} USDT\n"
        f"Open Positions: {sum(len(positions[symbol]) for symbol in COINS)}\n"
        f"Balance: {balance:.4f} USDT\n"
        f"Positions Details:\n{open_positions_details}"
    )
    send_telegram_message(message)

def trading_loop():
    global balance
    last_report_time = datetime.now()
    
    while True:
        try:
            now = datetime.now()
            # seconds_to_next_candle = 60 - now.second
            # if seconds_to_next_candle > 0:
            #     time.sleep(seconds_to_next_candle)
            time.sleep(1)
            logging.info(f"[SIM] Số dư hiện tại: Balance={balance}")
            
            if balance < initial_balance * STOP_LOSS_THRESHOLD:
                logging.critical(f"[SIM] Số dư dưới {STOP_LOSS_THRESHOLD*100}% ({balance} < {initial_balance * STOP_LOSS_THRESHOLD}). Dừng bot!")
                send_telegram_message(f"[SIM] Số dư dưới ngưỡng {STOP_LOSS_THRESHOLD*100}%. Dừng bot!")
                break
            
            total_positions = sum(len(positions[symbol]) for symbol in COINS)
            for symbol in COINS:
                df = get_historical_data(symbol, TIMEFRAME)
                df = add_utbot_indicators(df, a=1, c=10, h=False, sma_period=200)
                
                if positions[symbol]:
                    manage_positions(symbol, df)
                
                if balance > initial_balance * STOP_LOSS_THRESHOLD and total_positions < MAX_POSITIONS:
                    signal = check_entry_conditions(df, symbol)
                    if signal and not positions[symbol]:
                        enter_position(symbol, signal)
            
            if (now - last_report_time).total_seconds() >= 3600:
                send_periodic_report()
                last_report_time = now
        
        except Exception as e:
            logging.error(f"[SIM] Lỗi trong vòng lặp chính: {e}")
            time.sleep(60)

if __name__ == "__main__":
    logging.info("[SIM] Bắt đầu bot giao dịch mô phỏng...")
    send_telegram_message("[SIM] Bot giao dịch mô phỏng đã khởi động!")
    trading_loop()