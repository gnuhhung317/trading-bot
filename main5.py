import time
from datetime import datetime
from binance.client import Client
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os
import logging
import requests
import sys

# Tải biến môi trường từ file .env
load_dotenv()

# Cấu hình cơ bản
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
client = Client(API_KEY, API_SECRET)

COINS = {
    'ETHUSDT': {"leverage": 10, "quantity_precision": 3, "min_size": 0.001},
    'SOLUSDT': {"leverage": 10, "quantity_precision": 1, "min_size": 0.1},
    'BNBUSDT': {"leverage": 10, "quantity_precision": 2, "min_size": 0.01},
}

TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
RISK_PER_TRADE = 0.01  # 1% risk per trade
STOP_LOSS_THRESHOLD = 0.1
MAX_POSITIONS = 5
TAKER_FEE = 0.0004  # 0.04% taker fee
MAX_TOTAL_RISK = 0.05  # 5% max total risk

# Logging
logging.basicConfig(
    filename='momentum_trading.log',
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
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s['pricePrecision']
        logging.warning(f"Không tìm thấy thông tin precision cho {symbol}")
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
    # url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    # for attempt in range(3):
    #     try:
    #         response = requests.post(url, json=payload, timeout=5)
    #         response.raise_for_status()
    #         logging.info(f"Gửi tin nhắn Telegram thành công: {message[:50]}...")
    #         return response.json()
    #     except Exception as e:
    #         exc_type, exc_obj, tb = sys.exc_info()
    #         line_number = tb.tb_lineno
    #         logging.error(f"Lỗi gửi tin nhắn Telegram (lần {attempt+1}): {e} - {line_number}")
    #         time.sleep(2)
    # logging.critical("Telegram không hoạt động sau 3 lần thử!")
    logging.info(message)
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
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            logging.error(f"Lỗi lấy dữ liệu {symbol} (lần {attempt+1}): {e} - {line_number}")
            time.sleep(1)
    send_telegram_message(f"Lỗi lấy dữ liệu {symbol}: Không thể tải sau 3 lần thử")
    return pd.DataFrame()

def add_forex_fury_indicators(df):
    if df.empty:
        return df
    
    # Volume MA
    df['volume_ma'] = ta.sma(df['volume'], length=10)
    
    # Momentum
    df['momentum'] = df['close'].pct_change(10) * 100
    
    # Price Velocity (short-term momentum)
    df['price_velocity'] = df['close'].pct_change(3) * 100
    
    # Volume Acceleration
    df['volume_acceleration'] = df['volume'].pct_change(3) * 100
    
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=10)
    
    # SMA for trend
    df['sma200'] = ta.sma(df['close'], length=100)
    
    return df

def add_trend_indicators(df):
    if df.empty:
        return df
    
    # Higher timeframe trend
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    df['uptrend'] = (df['close'] > df['ema50']) & (df['ema50'] > df['ema200'])
    df['downtrend'] = (df['close'] < df['ema50']) & (df['ema50'] < df['ema200'])
    
    return df

def check_entry_conditions(df, higher_tf_df, symbol):
    if df.empty or higher_tf_df.empty:
        logging.warning("Dữ liệu rỗng, bỏ qua kiểm tra điều kiện vào lệnh")
        return None
    
    current = df.iloc[-1]
    current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
    
    # Dynamic momentum threshold based on ATR
    momentum_threshold = current['atr'] * 0.05
    
    # Long conditions
    long_conditions = [
        current['volume'] > current['volume_ma'] * 1.8,  # High volume
        current['volume'] > df['volume'].iloc[-2] * 1.3,  # Volume increasing
        current['momentum'] > momentum_threshold,  # Strong momentum
        current['price_velocity'] > 0,  # Price moving up
        current['volume_acceleration'] > 0,  # Volume accelerating
        current['rsi'] < 75,  # Not overbought
        current_price > current['sma200'] * 0.995  # Above long-term trend
    ]
    
    # Short conditions
    short_conditions = [
        current['volume'] > current['volume_ma'] * 1.8,  # High volume
        current['volume'] > df['volume'].iloc[-2] * 1.3,  # Volume increasing
        current['momentum'] < -momentum_threshold,  # Strong negative momentum
        current['price_velocity'] < 0,  # Price moving down
        current['volume_acceleration'] > 0,  # Volume accelerating
        current['rsi'] > 30,  # Not oversold
        current_price < current['sma200'] * 1.005  # Below long-term trend
    ]
    
    if all(long_conditions):
        return 'LONG'
    elif all(short_conditions):
        return 'SHORT'
    
    return None

def enter_position(symbol, signal):
    try:
        account_info = client.futures_account()
        total_wallet_balance = float(account_info['totalWalletBalance'])
        available_margin = float(account_info['availableBalance'])
        max_margin_per_coin = total_wallet_balance * 0.3
        
        risk_amount = available_margin * RISK_PER_TRADE
        if available_margin < risk_amount:
            logging.warning(f"{symbol} - Không đủ margin khả dụng ({available_margin} < {risk_amount})")
            return
        
        total_risk = sum(pos['risk_per_r'] for sym in positions for pos in positions[sym])
        if total_risk + risk_amount > balance * MAX_TOTAL_RISK:
            logging.warning(f"{symbol} - Vượt quá giới hạn tổng rủi ro 5%")
            return
        
        position_info = client.futures_position_information(symbol=symbol)
        current_margin_used = sum(float(pos['initialMargin']) for pos in position_info if pos['symbol'] == symbol)
        
        df = get_historical_data(symbol, TIMEFRAME)
        df = add_forex_fury_indicators(df)
        if df.empty:
            logging.warning(f"{symbol} - Không có dữ liệu để vào lệnh")
            return
        
        current = df.iloc[-1]
        entry_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
        atr = current['atr']
        
        if signal == 'LONG':
            stop_loss = entry_price - atr * 2.5
            take_profit = entry_price + atr * 5.0
        else:  # SHORT
            stop_loss = entry_price + atr * 2.5
            take_profit = entry_price - atr * 5.0
        
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
        
        side = 'BUY' if signal == 'LONG' else 'SELL'
        order = client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=size)
        
        if order and order['status'] == 'FILLED':
            # Place stop loss and take profit orders
            sl_side = 'SELL' if signal == 'LONG' else 'BUY'
            client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type='STOP_MARKET',
                stopPrice=stop_loss,
                quantity=size,
                reduceOnly=True
            )
            
            tp_side = 'SELL' if signal == 'LONG' else 'BUY'
            client.futures_create_order(
                symbol=symbol,
                side=tp_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit,
                quantity=size,
                reduceOnly=True
            )
            
            position = {
                'id': order['orderId'],
                'type': signal,
                'entry_time': datetime.now(),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': size,
                'risk_per_r': risk_amount,
                'atr': atr
            }
            positions[symbol].append(position)
            
            logging.info(f"{symbol} - Vào lệnh {signal}: Price={entry_price}, SL={stop_loss}, TP={take_profit}, Size={size}")
            message = (
                f"<b>{symbol} - {signal} Order</b>\n"
                f"Time: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Price: {entry_price:.4f}\n"
                f"Size: {size:.4f}\n"
                f"Stop Loss: {stop_loss:.4f}\n"
                f"Take Profit: {take_profit:.4f}"
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
    current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
    
    for i, position in enumerate(positions[symbol][:]):
        # Calculate new stop loss level with wider buffer
        if position['type'] == 'LONG':
            new_sl = current_price - current['atr'] * 2.5 * 1.2
            # Close if price hits stop loss or momentum fades
            if (current_price <= new_sl or 
                current['momentum'] < -current['atr'] * 0.05 or
                current['volume'] < current['volume_ma'] * 0.3 or
                (datetime.now() - position['entry_time']).total_seconds() > 40 * 15 * 60):  # 40 bars * 15min
                side = 'SELL'
                client.futures_cancel_all_open_orders(symbol=symbol)
                client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=position['size'])
                profit = (current_price - position['entry_price']) * position['size']
                trade = {
                    'type': position['type'],
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'profit': profit * (1 - TAKER_FEE)
                }
                trades.append(trade)
                balance += trade['profit']
                positions[symbol].pop(i)
                logging.info(f"{symbol} - Vị thế {position['type']} đã đóng tại {current_price}, Profit: {trade['profit']}")
                message = (
                    f"<b>{symbol} - {position['type']} Position Closed</b>\n"
                    f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Entry Price: {position['entry_price']:.4f}\n"
                    f"Exit Price: {current_price:.4f}\n"
                    f"Size: {position['size']:.4f}\n"
                    f"Profit: {trade['profit']:.4f}"
                )
                send_telegram_message(message)
        else:  # SHORT
            new_sl = current_price + current['atr'] * 2.5 * 1.2
            # Close if price hits stop loss or momentum fades
            if (current_price >= new_sl or 
                current['momentum'] > current['atr'] * 0.05 or
                current['volume'] < current['volume_ma'] * 0.3 or
                (datetime.now() - position['entry_time']).total_seconds() > 40 * 15 * 60):  # 40 bars * 15min
                side = 'BUY'
                client.futures_cancel_all_open_orders(symbol=symbol)
                client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=position['size'])
                profit = (position['entry_price'] - current_price) * position['size']
                trade = {
                    'type': position['type'],
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'profit': profit * (1 - TAKER_FEE)
                }
                trades.append(trade)
                balance += trade['profit']
                positions[symbol].pop(i)
                logging.info(f"{symbol} - Vị thế {position['type']} đã đóng tại {current_price}, Profit: {trade['profit']}")
                message = (
                    f"<b>{symbol} - {position['type']} Position Closed</b>\n"
                    f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Entry Price: {position['entry_price']:.4f}\n"
                    f"Exit Price: {current_price:.4f}\n"
                    f"Size: {position['size']:.4f}\n"
                    f"Profit: {trade['profit']:.4f}"
                )
                send_telegram_message(message)

def sync_positions_from_binance():
    global balance, initial_balance
    try:
        account_info = client.futures_account()
        balance = float(account_info.get('availableBalance', 0))
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
                if COINS[symbol]["quantity_precision"] == 0:
                    amt = int(amt)
                if abs(amt) > 0:
                    position_type = 'LONG' if amt > 0 else 'SHORT'
                    entry_price = float(pos.get('entryPrice', 0))
                    stop_loss = entry_price * (1 - 0.02) if position_type == 'LONG' else entry_price * (1 + 0.02)
                    risk_amount = balance * RISK_PER_TRADE if balance > 0 else 0.01
                    
                    position_dict = {
                        'id': None,
                        'type': position_type,
                        'entry_time': pd.to_datetime(pos.get('updateTime', 0), unit='ms'),
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'size': abs(amt),
                        'risk_per_r': risk_amount
                    }
                    positions[symbol].append(position_dict)
                    logging.info(f"Đồng bộ {symbol}: {position_type}, Size={abs(amt)}, Entry={entry_price}")
        
        logging.info(f"Đồng bộ vị thế từ Binance thành công. Balance: {balance}")
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
            seconds_to_next_candle = (15 - (now.minute % 15)) * 60 - now.second
            if seconds_to_next_candle > 0:
                logging.debug(f"Đợi {seconds_to_next_candle} giây đến nến tiếp theo")
                sync_positions_from_binance()
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
                try:
                    df = get_historical_data(symbol, TIMEFRAME)
                    df = add_forex_fury_indicators(df)
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
            client.futures_change_leverage(symbol=symbol, leverage=COINS[symbol]["leverage"])
            logging.info(f"Đặt leverage {COINS[symbol]['leverage']} cho {symbol}")
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            logging.error(f"Lỗi đặt leverage cho {symbol}: {e} - {line_number}")
            send_telegram_message(f"Lỗi đặt leverage cho {symbol}: {e} - {line_number}")
    
    logging.info("Bắt đầu bot giao dịch...")
    send_telegram_message("Bot giao dịch đã khởi động!")
    trading_loop()