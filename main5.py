import time
from datetime import datetime
from binance.client import Client
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os
import logging

# Tải biến môi trường từ file .env
load_dotenv()

# Cấu hình cơ bản
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(API_KEY, API_SECRET)

TIMEFRAME = '5m'
HIGHER_TIMEFRAME = '1h'

# Logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_usdt_pairs():
    """Lấy danh sách tất cả các cặp USDT từ Binance Futures"""
    try:
        exchange_info = client.futures_exchange_info()
        usdt_pairs = {}
        for symbol_info in exchange_info['symbols']:
            symbol = symbol_info['symbol']
            if symbol.endswith('USDT') and symbol_info['contractType'] == 'PERPETUAL':
                # Lấy thông tin precision và min size
                usdt_pairs[symbol] = {
                    'leverage': 10,  # Đặt mặc định leverage là 10
                    'quantity_precision': symbol_info['quantityPrecision'],
                    'min_size': float(symbol_info['filters'][2]['minQty'])  # Lấy từ filter LOT_SIZE
                }
        logging.info(f"Tìm thấy {len(usdt_pairs)} cặp USDT trên Binance Futures")
        return usdt_pairs
    except Exception as e:
        logging.error(f"Lỗi khi lấy danh sách cặp USDT: {e}")
        return {}

def get_historical_data(symbol, interval, limit=1000):
    """Lấy dữ liệu lịch sử từ Binance"""
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
        logging.error(f"Lỗi lấy dữ liệu {symbol} ({interval}): {e}")
        return pd.DataFrame()

def add_signal_indicators(df):
    """Thêm các chỉ báo tín hiệu vào dữ liệu 5m"""
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
    df['volume_increase'] = df['volume'] > df['volume_ma10']
    return df

def add_trend_indicators(df):
    """Thêm các chỉ báo xu hướng vào dữ liệu 1h"""
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
    """Kiểm tra điều kiện tín hiệu LONG hoặc SHORT"""
    if df.empty or higher_tf_df.empty:
        logging.warning(f"{symbol} - Dữ liệu rỗng, bỏ qua kiểm tra")
        return None
    
    current = df.iloc[-1]
    higher_current = higher_tf_df.iloc[-1]
    
    # Điều kiện LONG
    long_primary = [
        current['ema9'] > current['ema21'],
        current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']
    ]
    long_secondary = [
        current['rsi14'] < 70,
        current['volume_increase'],
        current['macd'] > 0,
        current['adx'] > 25
    ]
    long_condition = (all(long_primary) and all(long_secondary) and 
                      (higher_current['uptrend'] or 
                       (higher_current['adx'] > 20 and higher_current['di_plus'] > higher_current['di_minus'])))
    
    # Điều kiện SHORT
    short_primary = [
        current['ema9'] < current['ema21'],
        current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']
    ]
    short_secondary = [
        current['rsi14'] > 30,
        current['volume_increase'],
        current['macd'] < 0,
        current['adx'] > 25
    ]
    short_condition = (all(short_primary) and all(short_secondary) and 
                       (higher_current['downtrend'] or 
                        (higher_current['adx'] > 20 and higher_current['di_minus'] > higher_current['di_plus'])))
    
    signal = 'LONG' if long_condition else 'SHORT' if short_condition else None
    if signal:
        logging.info(f"{symbol}: Tín hiệu {signal}")
    return signal

def predict_signals():
    """Dự đoán tín hiệu cho tất cả các cặp coin USDT"""
    # Lấy danh sách cặp USDT
    COINS = get_usdt_pairs()
    if not COINS:
        print("Không thể lấy danh sách cặp USDT. Kết thúc chương trình.")
        return
    
    print(f"Dự đoán tín hiệu giao dịch ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"Tổng số cặp USDT: {len(COINS)}")
    print("-" * 50)
    
    for symbol in COINS:
        try:
            # Lấy dữ liệu
            df_5m = get_historical_data(symbol, TIMEFRAME,limit=500)
            df_5m = add_signal_indicators(df_5m)
            df_1h = get_historical_data(symbol, HIGHER_TIMEFRAME,limit=200)
            df_1h = add_trend_indicators(df_1h)
            
            # Kiểm tra tín hiệu
            signal = check_entry_conditions(df_5m, df_1h, symbol)
            
            # In kết quả
            if signal:
                print(f"{symbol}: {signal}")
            else:
                print(f"{symbol}: NO SIGNAL")
                
        except Exception as e:
            logging.error(f"Lỗi khi dự đoán tín hiệu cho {symbol}: {e}")
            print(f"{symbol}: ERROR - {e}")
    
    print("-" * 50)

if __name__ == "__main__":
    logging.info("Bắt đầu dự đoán tín hiệu giao dịch...")
    predict_signals()