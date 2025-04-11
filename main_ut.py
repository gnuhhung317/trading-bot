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
import math
load_dotenv()

# === Configuration ===
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
client = Client(API_KEY, API_SECRET)

# Trading parameters
COINS = {
    "ETHUSDT": {"leverage": 10, "quantity_precision": 3, "min_size": 0.001}
    # "BTCUSDT": {"leverage": 20, "quantity_precision": 1, "min_size": 0.1}
}
TIMEFRAME = "5m"
ATR_MULTIPLIER = 1.0
ATR_PERIOD = 10
SMA_PERIOD = 200
RISK_PER_TRADE = 0.1  # 1%
MAX_POSITIONS = 5
STOP_LOSS_THRESHOLD = 0.1  # 10%
TAKER_FEE = 0.0004  # 0.04%

# Logging setup
logging.basicConfig(
    filename="utbot_trading.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

positions = {symbol: [] for symbol in COINS}
trades = []
initial_balance = None
balance = None
future_balance = None
# === Utility Functions ===
def get_symbol_precision(symbol):
    try:
        exchange_info = client.futures_exchange_info()
        for s in exchange_info["symbols"]:
            if s["symbol"] == symbol:
                return s["pricePrecision"]
        logging.warning(f"Không tìm thấy precision cho {symbol}")
        return 0
    except Exception as e:
        logging.error(f"Lỗi lấy precision cho {symbol}: {e}")
        return 0

for symbol in COINS:
    COINS[symbol]["price_precision"] = get_symbol_precision(symbol)

def round_to_precision(symbol, value, value_type="quantity"):
    precision = COINS[symbol]["quantity_precision"] if value_type == "quantity" else COINS[symbol]["price_precision"]
    factor = 10 ** precision
    rounded_value = math.floor(value * factor) / factor
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
            logging.info(f"Gửi tin nhắn Telegram: {message[:50]}...")
            return response.json()
        except Exception as e:
            logging.error(f"Lỗi gửi Telegram (lần {attempt+1}): {e}")
            time.sleep(2)
    logging.critical("Telegram không hoạt động sau 3 lần thử!")
    return None

def get_historical_data(symbol, interval, limit=1000):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume",
                                           "close_time", "quote_asset_volume", "trades",
                                           "taker_buy_base", "taker_buy_quote", "ignored"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        logging.error(f"Lỗi lấy dữ liệu {symbol}: {e}")
        return pd.DataFrame()

# === Indicator Calculations ===
def calculate_indicators(df):
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    df["nLoss"] = ATR_MULTIPLIER * df["atr"]
    df["sma"] = ta.sma(df["close"], length=SMA_PERIOD)
    
    # Trailing Stop calculation
    xATRTrailingStop = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop.iloc[i-1]
        close = df["close"].iloc[i]
        close_prev = df["close"].iloc[i-1]
        nLoss = df["nLoss"].iloc[i]
        if close > prev_stop and close_prev > prev_stop:
            xATRTrailingStop.iloc[i] = max(prev_stop, close - nLoss)
        elif close < prev_stop and close_prev < prev_stop:
            xATRTrailingStop.iloc[i] = min(prev_stop, close + nLoss)
        else:
            xATRTrailingStop.iloc[i] = close - nLoss if close > prev_stop else close + nLoss
    df["xATRTrailingStop"] = xATRTrailingStop
    
    # Entry signals
    df["buySignal"] = (df["close"] > df["xATRTrailingStop"]) & (df["close"].shift(1) <= df["xATRTrailingStop"].shift(1)) & (df["close"] > df["sma"])
    df["sellSignal"] = (df["close"] < df["xATRTrailingStop"]) & (df["close"].shift(1) >= df["xATRTrailingStop"].shift(1)) & (df["close"] < df["sma"])
    return df

# === Trading Logic ===
def check_entry_conditions(df, symbol):
    if df.empty:
        return None
    current = df.iloc[-1]
    if current["buySignal"]:
        return "LONG"
    elif current["sellSignal"]:
        return "SHORT"
    return None

def enter_position(symbol, signal):
    global balance
    try:
        account_info = client.futures_account()
        balance = float(account_info["availableBalance"])
        risk_amount = balance * RISK_PER_TRADE
        
        if balance < risk_amount:
            logging.warning(f"{symbol} - Không đủ số dư ({balance} < {risk_amount})")
            return
        
        total_positions = sum(len(positions[sym]) for sym in COINS)
        if total_positions >= MAX_POSITIONS:
            logging.warning(f"{symbol} - Đã đạt giới hạn {MAX_POSITIONS} vị thế")
            return
        
        df = get_historical_data(symbol, TIMEFRAME)
        df = calculate_indicators(df)
        if df.empty:
            return
        
        current = df.iloc[-1]
        entry_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
        nLoss = current["nLoss"]
        
        # Đảm bảo nLoss không quá nhỏ để tránh position_size quá lớn
        min_nLoss = entry_price * 0.005  # Ít nhất 0.5% giá vào
        nLoss = max(nLoss, min_nLoss)
        
        position_size = (risk_amount / nLoss) * COINS[symbol]["leverage"]
        max_size = (balance * COINS[symbol]["leverage"] / entry_price) * (1 - TAKER_FEE * 2)
        position_size = min(position_size, max_size) * 0.9  # Giảm kích thước lệnh 10% để tránh margin không đủ
        position_size = round_to_precision(symbol, position_size)
        
        if position_size < COINS[symbol]["min_size"]:
            logging.warning(f"{symbol} - Kích thước lệnh ({position_size}) nhỏ hơn min_size")
            return
        
        # Thêm cơ chế stop loss cố định kết hợp với trailing stop
        if signal == "LONG":
            # Hard stop loss kết hợp với trailing stop
            hard_stop_loss = entry_price * (1 - 0.02)  # 2% dưới giá vào
            trailing_stop = entry_price - nLoss
            stop_loss = max(hard_stop_loss, trailing_stop)  # Chọn giá trị gần giá nhất
            take_profit = entry_price + 2 * nLoss
        else:  # SHORT
            hard_stop_loss = entry_price * (1 + 0.02)  # 2% trên giá vào  
            trailing_stop = entry_price + nLoss
            stop_loss = min(hard_stop_loss, trailing_stop)  # Chọn giá trị gần giá nhất
            take_profit = entry_price - 2 * nLoss
        
        side = "BUY" if signal == "LONG" else "SELL"
        
        # Tạo thông báo trước khi gửi lệnh
        order_message = f"{symbol} - Vào {signal}: Price={entry_price}, Size={position_size}, Trailing Stop={trailing_stop}, TP={take_profit}"
        logging.info(f"Chuẩn bị gửi lệnh: {order_message}")
        
        # Kiểm tra lại margin một lần nữa trước khi gửi lệnh
        try:
            # Gửi lệnh vào thị trường
            order = client.futures_create_order(symbol=symbol, side=side, type="MARKET", quantity=position_size)
            if order and order["status"] == "FILLED":
                position = {
                    "id": order["orderId"],
                    "type": signal,
                    "entry_time": datetime.now(),
                    "entry_price": entry_price,
                    "trailing_stop": round_to_precision(symbol, trailing_stop, "price"),
                    "hard_stop_loss": round_to_precision(symbol, hard_stop_loss, "price"),
                    "take_profit": round_to_precision(symbol, take_profit, "price"),
                    "size": position_size,
                    "risk_amount": risk_amount,
                    "breakeven_activated": False  # Thêm cờ để theo dõi trạng thái
                }
                positions[symbol].append(position)
                
                logging.info(order_message)
                message = (
                    f"<b>{symbol} - {signal} Entry</b>\n"
                    f"Time: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Price: {entry_price:.4f}\n"
                    f"Size: {position_size:.4f}\n"
                    f"Trailing Stop: {trailing_stop:.4f}\n"
                    f"Take Profit: {take_profit:.4f}"
                )
                send_telegram_message(message)
        except Exception as margin_error:
            if "APIError(code=-2019): Margin is insufficient" in str(margin_error):
                # Thử lại với kích thước vị thế nhỏ hơn
                smaller_position_size = round_to_precision(symbol, position_size * 0.7)  # Giảm xuống 70%
                if smaller_position_size >= COINS[symbol]["min_size"]:
                    logging.warning(f"{symbol} - Margin không đủ, thử lại với kích thước nhỏ hơn: {smaller_position_size}")
                    try:
                        # Gửi lệnh với kích thước nhỏ hơn
                        order = client.futures_create_order(symbol=symbol, side=side, type="MARKET", quantity=smaller_position_size)
                        if order and order["status"] == "FILLED":
                            position = {
                                "id": order["orderId"],
                                "type": signal,
                                "entry_time": datetime.now(),
                                "entry_price": entry_price,
                                "trailing_stop": round_to_precision(symbol, trailing_stop, "price"),
                                "hard_stop_loss": round_to_precision(symbol, hard_stop_loss, "price"),
                                "take_profit": round_to_precision(symbol, take_profit, "price"),
                                "size": smaller_position_size,
                                "risk_amount": risk_amount,
                                "breakeven_activated": False
                            }
                            positions[symbol].append(position)
                            
                            updated_message = f"{symbol} - Vào {signal} (kích thước giảm): Price={entry_price}, Size={smaller_position_size}, Trailing Stop={trailing_stop}, TP={take_profit}"
                            logging.info(updated_message)
                            message = (
                                f"<b>{symbol} - {signal} Entry (kích thước giảm)</b>\n"
                                f"Time: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"Price: {entry_price:.4f}\n"
                                f"Size: {smaller_position_size:.4f}\n"
                                f"Trailing Stop: {trailing_stop:.4f}\n"
                                f"Take Profit: {take_profit:.4f}"
                            )
                            send_telegram_message(message)
                    except Exception as retry_error:
                        logging.error(f"{symbol} - Lỗi khi thử lại với kích thước nhỏ hơn: {retry_error}")
                        send_telegram_message(f"{symbol} - Lỗi khi thử lại với kích thước nhỏ hơn: {retry_error}")
                else:
                    logging.error(f"{symbol} - Không thể giảm kích thước vị thế hơn nữa ({smaller_position_size} < {COINS[symbol]['min_size']})")
                    send_telegram_message(f"{symbol} - Không thể giảm kích thước vị thế hơn nữa do đã dưới ngưỡng tối thiểu")
            else:
                # Xử lý các lỗi khác
                raise margin_error
    except Exception as e:
        logging.error(f"{symbol} - Lỗi khi vào lệnh: {e}")
        send_telegram_message(f"{symbol} - Lỗi vào lệnh: {e}")

def manage_positions(symbol, df):
    global balance, trades
    if df.empty or not positions[symbol]:
        return
    
    current = df.iloc[-1]
    current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
    nLoss = current["nLoss"]
    
    for i, position in enumerate(positions[symbol][:]):
        # Tính toán phần trăm lợi nhuận - chỉ dùng để điều chỉnh trailing stop
        profit_pct = (current_price - position["entry_price"]) / position["entry_price"] * (1 if position["type"] == "LONG" else -1)
        
        # Cập nhật trailing stop linh hoạt hơn
        if position["type"] == "LONG":
            # Di chuyển trailing stop lên khi giá đã có lợi nhuận
            if current_price > position["entry_price"]:
                # Nếu đã đạt 1R lợi nhuận, di chuyển stop lên breakeven
                if profit_pct >= 0.01 and not position["breakeven_activated"]:
                    position["trailing_stop"] = position["entry_price"]
                    position["breakeven_activated"] = True
                # Nếu đã đạt 2R, sử dụng trailing stop chặt hơn (0.5 ATR)
                elif profit_pct >= 0.02:
                    new_stop = current_price - (nLoss * 0.5)
                    position["trailing_stop"] = round_to_precision(
                        symbol, max(position["trailing_stop"], new_stop), "price"
                    )
                # Trường hợp thông thường, trailing stop chuẩn (1 ATR)
                elif current_price > position["entry_price"] + nLoss:
                    new_stop = current_price - nLoss
                    position["trailing_stop"] = round_to_precision(
                        symbol, max(position["trailing_stop"], new_stop), "price"
                    )
            
            # Điều kiện thoát lệnh kết hợp cả hard stop và trailing stop
            should_exit = (
                current_price <= position["trailing_stop"] or
                current_price <= position["hard_stop_loss"] or
                current_price >= position["take_profit"] or
                current["sellSignal"]
            )
        
        else:  # SHORT
            # Di chuyển trailing stop xuống khi giá đã có lợi nhuận
            if current_price < position["entry_price"]:
                # Nếu đã đạt 1R lợi nhuận, di chuyển stop lên breakeven
                if profit_pct >= 0.01 and not position["breakeven_activated"]:
                    position["trailing_stop"] = position["entry_price"]
                    position["breakeven_activated"] = True
                # Nếu đã đạt 2R, sử dụng trailing stop chặt hơn (0.5 ATR)
                elif profit_pct >= 0.02:
                    new_stop = current_price + (nLoss * 0.5)
                    position["trailing_stop"] = round_to_precision(
                        symbol, min(position["trailing_stop"], new_stop), "price"
                    )
                # Trường hợp thông thường, trailing stop chuẩn (1 ATR)
                elif current_price < position["entry_price"] - nLoss:
                    new_stop = current_price + nLoss
                    position["trailing_stop"] = round_to_precision(
                        symbol, min(position["trailing_stop"], new_stop), "price"
                    )
            
            # Điều kiện thoát lệnh kết hợp cả hard stop và trailing stop
            should_exit = (
                current_price >= position["trailing_stop"] or
                current_price >= position["hard_stop_loss"] or
                current_price <= position["take_profit"] or
                current["buySignal"]
            )
        
        # Xác định lý do thoát lệnh nếu cần thoát
        if should_exit:
            # Xác định giá thoát và trigger
            exit_price = current_price  # Sử dụng giá hiện tại làm giá thoát
            
            if current_price <= position["trailing_stop"] and position["type"] == "LONG":
                exit_trigger = "trailing_stop"
            elif current_price >= position["trailing_stop"] and position["type"] == "SHORT":
                exit_trigger = "trailing_stop"
            elif current_price <= position["hard_stop_loss"] and position["type"] == "LONG":
                exit_trigger = "hard_stop"
            elif current_price >= position["hard_stop_loss"] and position["type"] == "SHORT":
                exit_trigger = "hard_stop"
            elif (current_price >= position["take_profit"] and position["type"] == "LONG") or \
                 (current_price <= position["take_profit"] and position["type"] == "SHORT"):
                exit_trigger = "take_profit"
            else:
                exit_trigger = "signal"
            
            # Exit Position
            side = "SELL" if position["type"] == "LONG" else "BUY"
            order = client.futures_create_order(
                symbol=symbol, side=side, type="MARKET", quantity=position["size"], reduceOnly=True
            )
            if order and order["status"] == "FILLED":
                # Tính lợi nhuận thực tế
                if position["type"] == "LONG":
                    profit = (exit_price - position["entry_price"]) * position["size"] * (1 - TAKER_FEE)
                else:  # SHORT
                    profit = (position["entry_price"] - exit_price) * position["size"] * (1 - TAKER_FEE)
                
                trade = {
                    "type": position["type"],
                    "entry_time": position["entry_time"],
                    "exit_time": datetime.now(),
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "exit_trigger": exit_trigger,
                    "profit": profit
                }
                trades.append(trade)
                balance += profit
                positions[symbol].pop(i)
                
                # Ghi log một lần với thông tin đầy đủ
                logging.info(f"{symbol} - Thoát {position['type']} do {exit_trigger}: Exit Price={exit_price}, Profit={profit}")
                
                # Gửi thông báo Telegram
                message = (
                    f"<b>{symbol} - {position['type']} Exit</b>\n"
                    f"Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Entry Price: {position['entry_price']:.4f}\n"
                    f"Exit Price: {exit_price:.4f}\n"
                    f"Exit Reason: {exit_trigger}\n"
                    f"Size: {position['size']:.4f}\n"
                    f"Profit: {profit:.4f}"
                )
                send_telegram_message(message)

def sync_balance():
    global balance, initial_balance, future_balance
    try:
        account_info = client.futures_account()
        balance = float(account_info["availableBalance"])
        future_balance = float(account_info["totalMarginBalance"])
        if initial_balance is None:
            initial_balance = balance

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
                    # Lấy dữ liệu thị trường để tính ATR
                    df = get_historical_data(symbol, TIMEFRAME)
                    df = calculate_indicators(df)
                    
                    if not df.empty:
                        current = df.iloc[-1]
                        nLoss = current["nLoss"]
                        
                        position_type = 'LONG' if amt > 0 else 'SHORT'
                        entry_price = float(pos.get('entryPrice', 0))
                        current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
                        
                        # Tính trailing stop và hard stop loss theo logic mới
                        if position_type == 'LONG':
                            hard_stop_loss = entry_price * (1 - 0.02)  # 2% dưới giá vào
                            base_trailing_stop = entry_price - nLoss
                            
                            # Kiểm tra nếu giá hiện tại có lợi nhuận để điều chỉnh trailing stop
                            profit_pct = (current_price - entry_price) / entry_price
                            if profit_pct >= 0.02:  # Nếu lợi nhuận >= 2%
                                trailing_stop = max(base_trailing_stop, current_price - (nLoss * 0.5))
                                breakeven_activated = True
                            elif profit_pct >= 0.01:  # Nếu lợi nhuận >= 1%
                                trailing_stop = max(base_trailing_stop, entry_price)
                                breakeven_activated = True
                            else:
                                trailing_stop = base_trailing_stop
                                breakeven_activated = False
                            
                            take_profit = entry_price + 2 * nLoss
                        else:  # SHORT
                            hard_stop_loss = entry_price * (1 + 0.02)  # 2% trên giá vào
                            base_trailing_stop = entry_price + nLoss
                            
                            # Kiểm tra nếu giá hiện tại có lợi nhuận để điều chỉnh trailing stop
                            profit_pct = (entry_price - current_price) / entry_price
                            if profit_pct >= 0.02:  # Nếu lợi nhuận >= 2%
                                trailing_stop = min(base_trailing_stop, current_price + (nLoss * 0.5))
                                breakeven_activated = True
                            elif profit_pct >= 0.01:  # Nếu lợi nhuận >= 1%
                                trailing_stop = min(base_trailing_stop, entry_price)
                                breakeven_activated = True
                            else:
                                trailing_stop = base_trailing_stop
                                breakeven_activated = False
                            
                            take_profit = entry_price - 2 * nLoss
                        
                        # Làm tròn giá trị theo độ chính xác của symbol
                        trailing_stop = round_to_precision(symbol, trailing_stop, "price")
                        hard_stop_loss = round_to_precision(symbol, hard_stop_loss, "price")
                        take_profit = round_to_precision(symbol, take_profit, "price")
                        
                        risk_amount = balance * RISK_PER_TRADE if balance > 0 else 0.01
                        
                        position_dict = {
                            'id': None,
                            'type': position_type,
                            'entry_time': pd.to_datetime(pos.get('updateTime', 0), unit='ms'),
                            'entry_price': entry_price,
                            'trailing_stop': trailing_stop,
                            'hard_stop_loss': hard_stop_loss,
                            'take_profit': take_profit,
                            'size': abs(amt),
                            'risk_amount': risk_amount,
                            'breakeven_activated': breakeven_activated
                        }
                        positions[symbol].append(position_dict)
                        logging.info(f"Đồng bộ {symbol}: {position_type}, Size={abs(amt)}, Entry={entry_price}, Trailing Stop={trailing_stop}")
        
        logging.info(f"Số dư hiện tại: {balance}")
    except Exception as e:
        logging.error(f"Lỗi đồng bộ số dư: {e}")
        balance = balance if balance is not None else 0

# === Main Trading Loop ===
def trading_loop():
    global balance
    last_report_time = datetime.now()
    flag = True
    while True:
        try:
            now = datetime.now()
            time.sleep(3)  
            sync_balance()

            if now.minute %5==0 and flag:
                send_telegram_message(f"Số dư hiện tại: Balance={future_balance:.4f} USDT")
                flag = False
            elif now.minute %5!=0 :
                flag = True
            if future_balance < initial_balance * STOP_LOSS_THRESHOLD:
                logging.critical(f"Số dư dưới {STOP_LOSS_THRESHOLD*100}% ({future_balance} < {initial_balance * STOP_LOSS_THRESHOLD}). Dừng bot!")
                send_telegram_message(f"Số dư dưới ngưỡng {STOP_LOSS_THRESHOLD*100}%. Dừng bot!")
                break
            
            total_positions = sum(len(positions[symbol]) for symbol in COINS)
            for symbol in COINS:
                df = get_historical_data(symbol, TIMEFRAME)
                df = calculate_indicators(df)
                
                if positions[symbol]:
                    manage_positions(symbol, df)
                
                if balance > initial_balance * STOP_LOSS_THRESHOLD and total_positions < MAX_POSITIONS:
                    signal = check_entry_conditions(df, symbol)
                    if signal and not positions[symbol]:
                        enter_position(symbol, signal)
            
            if (now - last_report_time).total_seconds() >= 3600:  # Báo cáo mỗi giờ
                message = (
                    f"<b>Hourly Report ({now.strftime('%Y-%m-%d %H:%M:%S')})</b>\n"
                    f"Balance: {balance:.4f} USDT\n"
                    f"Open Positions: {total_positions}"
                )
                send_telegram_message(message)
                last_report_time = now
        
        except Exception as e:
            logging.error(f"Lỗi trong vòng lặp chính: {e}")
            send_telegram_message(f"Lỗi trong vòng lặp: {e}")
            time.sleep(60)

# === Start Bot ===
if __name__ == "__main__":
    for symbol in COINS:
        try:
            client.futures_change_leverage(symbol=symbol, leverage=COINS[symbol]["leverage"])
            logging.info(f"Đặt đòn bẩy {COINS[symbol]['leverage']} cho {symbol}")
        except Exception as e:
            logging.error(f"Lỗi đặt đòn bẩy cho {symbol}: {e}")
    
    logging.info("Khởi động bot giao dịch UT Bot...")
    send_telegram_message("Bot giao dịch UT Bot đã khởi động!")
    trading_loop()