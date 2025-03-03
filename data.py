# data.py
import pandas as pd
from config import client, SYMBOL, TIMEFRAME, HIGHER_TIMEFRAME, TIMEFRAME_MAP
import time
from datetime import datetime

def fetch_ohlcv(symbol, timeframe, limit=100):
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=TIMEFRAME_MAP[timeframe],
        limit=limit
    )
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    df.set_index('timestamp', inplace=True)
    return df

def get_latest_data():
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
    higher_tf_df = fetch_ohlcv(SYMBOL, HIGHER_TIMEFRAME, limit=50)
    return df, higher_tf_df

def update_data(df, higher_tf_df):
    # Lấy dữ liệu mới nhất
    new_df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1)
    new_higher_tf = fetch_ohlcv(SYMBOL, HIGHER_TIMEFRAME, limit=1)
    
    df = pd.concat([df.iloc[:-1], new_df]).drop_duplicates()
    higher_tf_df = pd.concat([higher_tf_df.iloc[:-1], new_higher_tf]).drop_duplicates()
    return df, higher_tf_df