from datetime import datetime
import os
import time
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from indicators import add_signal_indicators, add_trend_indicators  # Giả sử bạn có file indicators.py
from dotenv import load_dotenv
# Load biến môi trường từ file .env
load_dotenv()

# Lấy API key từ biến môi trường
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(API_KEY, API_SECRET)
# Danh sách coin và tham số
COINS = {
    "ETHUSDT": {"leverage": 5, "quantity_precision": 4},
    "SOLUSDT": {"leverage": 5, "quantity_precision": 3},
    "BNBUSDT": {"leverage": 5, "quantity_precision": 3},
}
# Thiết lập tham số giao dịch
SYMBOL = "ETHUSDT"  # Cặp giao dịch Futures
TIMEFRAME = "5m"      # Khung thời gian nhỏ
HIGHER_TIMEFRAME = "1h"  # Khung thời gian lớn
LEVERAGE = 5
RISK_PER_TRADE = 0.02  # 2% rủi ro mỗi lệnh
QUANTITY_PRECISION = 4  # Số chữ số thập phân cho lượng coin

# Khởi tạo biến toàn cục
positions = {}
for symbol in COINS:
    client.futures_change_leverage(symbol=symbol, leverage=COINS[symbol]["leverage"])
futures_account = client.futures_account()
INITIAL_BALANCE = float(futures_account['totalWalletBalance'])  # Số dư USDT từ tài khoản Futures
balance = INITIAL_BALANCE
print(f"Số dư Futures ban đầu: {balance} USDT")
def process_message(msg):
    # Xử lý dữ liệu từ WebSocket
    current_price = float(msg['k']['c'])  # Giá đóng của cây nến
    print(f"Giá thời gian thực: {current_price}")
# Hàm lấy dữ liệu lịch sử từ Binance Futures
def get_historical_data(symbol, interval, limit=500):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'trades', 
                                       'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# Hàm xử lý dữ liệu và thêm chỉ báo
def prepare_data(df, higher_tf_df):
    df = add_signal_indicators(df)
    higher_tf_df = add_trend_indicators(higher_tf_df)
    return df, higher_tf_df

# Hàm kiểm tra điều kiện vào lệnh
def check_entry_conditions(df, higher_tf_df, balance):
    current = df.iloc[-1]
    higher_current = higher_tf_df.iloc[-1]

    long_primary = [
        current['ema9'] > current['ema21'],
        current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']
    ]
    long_secondary = [
        current['rsi14'] < 70,
        current['volume_increase'],
        current['macd'] > 0
    ]
    long_condition = (
        all(long_primary) and 
        any(long_secondary) and
        (higher_current['uptrend'] or (higher_current['adx'] > 25 and higher_current['di_plus'] > higher_current['di_minus']))
    )

    short_primary = [
        current['ema9'] < current['ema21'],
        current['ema_cross_down'] or current['macd_cross_down']
    ]
    short_secondary = [
        current['rsi14'] > 30,
        current['volume_increase'],
        current['macd'] < 0
    ]
    short_condition = (
        all(short_primary) and 
        any(short_secondary) and
        (higher_current['downtrend'] or (higher_current['adx'] > 25 and higher_current['di_minus'] > higher_current['di_plus']))
    )

    if long_condition and balance > 0:
        return "LONG"
    elif short_condition and balance > 0:
        return "SHORT"
    return None

# Hàm kiểm tra điều kiện thoát lệnh
def check_exit_conditions(position, current_price, df):
    current = df.iloc[-1]
    profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
    r_multiple = profit / position['risk_per_r'] if position['risk_per_r'] != 0 else 0

    if position['type'] == "LONG":
        exit_conditions = [
            current_price <= position['stop_loss'],
            current['ema_cross_down'],
            current['macd_cross_down'],
            current['rsi14'] > 80,
            r_multiple >= 4
        ]
        return any(exit_conditions)
    elif position['type'] == "SHORT":
        exit_conditions = [
            current_price >= position['stop_loss'],
            current['ema_cross_up'],
            current['macd_cross_up'],
            current['rsi14'] < 20,
            r_multiple >= 4
        ]
        return any(exit_conditions)
    return False

def place_futures_order(symbol, side, quantity):
    try:
        order = client.futures_create_order(symbol=symbol, side=side, type=FUTURE_ORDER_TYPE_MARKET, quantity=quantity)
        print(f"Đặt lệnh Futures {side} cho {symbol} thành công: {order}")
        return order
    except Exception as e:
        print(f"Lỗi khi đặt lệnh Futures cho {symbol}: {e}")
        return None

# Hàm đặt stop-loss trên Binance Futures
def place_futures_stop_loss(symbol, side, quantity, stop_price):
    try:
        order = client.futures_create_order(symbol=symbol, side=side, type=FUTURE_ORDER_TYPE_STOP_MARKET, quantity=quantity, stopPrice=str(stop_price))
        print(f"Đặt stop-loss Futures cho {symbol} thành công: {order}")
        return order
    except Exception as e:
        print(f"Lỗi khi đặt stop-loss Futures cho {symbol}: {e}")
        return None

# Logic giao dịch chính
def trading_loop():
    global positions, balance
    while True:
        try:
            now = datetime.now()
            seconds_to_next_candle = (5 - (now.minute % 5)) * 60 - now.second
            if seconds_to_next_candle > 0:
                print(f"Đợi {seconds_to_next_candle} giây đến cây nến 5 phút tiếp theo...")
                time.sleep(seconds_to_next_candle)

            futures_account = client.futures_account()
            balance = float(futures_account['totalWalletBalance'])

            for symbol in COINS:
                df = get_historical_data(symbol, TIMEFRAME)
                higher_tf_df = get_historical_data(symbol, HIGHER_TIMEFRAME)
                df, higher_tf_df = prepare_data(df, higher_tf_df)
                current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
                print(f"{symbol} - Giá hiện tại: {current_price}, Số dư: {balance} USDT")

                if symbol in positions and check_exit_conditions(positions[symbol], current_price, df):
                    side = SIDE_SELL if positions[symbol]['type'] == "LONG" else SIDE_BUY
                    place_futures_order(symbol, side, positions[symbol]['size'])
                    profit = (current_price - positions[symbol]['entry_price']) * positions[symbol]['size'] * (1 if positions[symbol]['type'] == 'LONG' else -1)
                    print(f"Thoát lệnh {positions[symbol]['type']} cho {symbol} tại {current_price}, Lợi nhuận: {profit}")
                    del positions[symbol]

                if symbol not in positions and balance > INITIAL_BALANCE * 0.1:
                    signal = check_entry_conditions(df, higher_tf_df, balance)
                    if signal:
                        atr = df['atr14'].iloc[-1]
                        entry_price = current_price
                        stop_loss = entry_price - atr * 1.5 if signal == "LONG" else entry_price + atr * 1.5
                        risk_per_r = abs(entry_price - stop_loss)
                        risk_amount = balance * RISK_PER_TRADE
                        position_size = (risk_amount / risk_per_r) * COINS[symbol]["leverage"]
                        position_size = round(position_size, COINS[symbol]["quantity_precision"])

                        if position_size * entry_price <= balance * COINS[symbol]["leverage"]:
                            side = SIDE_BUY if signal == "LONG" else SIDE_SELL
                            place_futures_order(symbol, side, position_size)
                            place_futures_stop_loss(symbol, SIDE_SELL if signal == "LONG" else SIDE_BUY, position_size, stop_loss)
                            positions[symbol] = {'type': signal, 'entry_price': entry_price, 'stop_loss': stop_loss, 'size': position_size, 'risk_per_r': risk_amount}
                            print(f"Vào lệnh {signal} cho {symbol} tại {entry_price}, SL: {stop_loss}, Size: {position_size}")
                        else:
                            print(f"Số dư không đủ để đặt lệnh cho {symbol}!")

        except Exception as e:
            print(f"Lỗi trong vòng lặp giao dịch: {e}")
            time.sleep(60)
# Chạy chương trình
if __name__ == "__main__":
    print("Bắt đầu giao dịch Futures thời gian thực...")
    trading_loop()