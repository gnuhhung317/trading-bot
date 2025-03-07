import time
from datetime import datetime
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from dotenv import load_dotenv
import os
import logging

load_dotenv()

# Binance API credentials
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(API_KEY, API_SECRET)

# Configuration (same as live trading)
COINS = {
    'SUIUSDT': {"leverage": 10, "quantity_precision": 1, "min_size": 0.1},
    'MOVEUSDT': {"leverage": 10, "quantity_precision": 0, "min_size": 1},
    'CETUSUSDT': {"leverage": 10, "quantity_precision": 0, "min_size": 1},
    "XRPUSDT": {"leverage": 10, "quantity_precision": 1, "min_size": 0.1},
    "ETHUSDT": {"leverage": 10, "quantity_precision": 3, "min_size": 0.001},
}

TIMEFRAME = '5m'
HIGHER_TIMEFRAME = '1h'
RISK_PER_TRADE = 0.01  # 1% risk per trade
STOP_LOSS_THRESHOLD = 0.1  # 10% max drawdown
MAX_POSITIONS = 5
TAKER_FEE = 0.0004  # 0.04% taker fee
MAX_TOTAL_RISK = 0.05  # 5% max total risk

# Logging setup
logging.basicConfig(
    filename='backtest.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Backtest parameters
INITIAL_BALANCE = 10  # Starting balance in USDT


# Data fetching and precision setup
def get_historical_data(symbol, interval, start_str, end_str):
    klines = client.futures_historical_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'trades',
                                       'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    logging.info(f"Fetched {len(df)} candles for {symbol} ({interval})")
    return df

def get_symbol_precision(symbol):
    exchange_info = client.futures_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            return s['pricePrecision']
    return 0

for symbol in COINS:
    COINS[symbol]['price_precision'] = get_symbol_precision(symbol=symbol)

def round_to_precision(symbol, size, value_type='quantity'):
    precision = COINS[symbol]["quantity_precision"] if value_type == 'quantity' else COINS[symbol]["price_precision"]
    rounded_value = round(size, precision)
    if precision == 0:
        rounded_value = int(rounded_value)
    return rounded_value

# Indicator functions (same as live trading)
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
    df['volume_increase'] = df['volume'] > df['volume_ma10'] * 1.5
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

# Entry and position management logic
def check_entry_conditions(df, higher_tf_df, symbol):
    if df.empty or higher_tf_df.empty:
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
    
    return 'LONG' if long_condition else 'SHORT' if short_condition else None

def simulate_trade(symbol, signal, entry_price, atr, balance):
    risk_amount = balance * RISK_PER_TRADE
    stop_loss = entry_price - atr * 2.5 if signal == 'LONG' else entry_price + atr * 2.5
    risk_per_r = abs(entry_price - stop_loss)
    size = (risk_amount / risk_per_r) * COINS[symbol]["leverage"]
    size = round_to_precision(symbol, size)
    if size < COINS[symbol]["min_size"]:
        return None
    return {
        'symbol': symbol,
        'type': signal,
        'entry_time': None,  # Will be set during simulation
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'size': size,
        'risk_per_r': risk_amount,
        'breakeven_activated': False,
        'first_target_hit': False,
        'second_target_hit': False,
        'third_target_hit': False
    }

# Backtesting simulation
def backtest(START_DATE ,END_DATE ):
    
    balance = INITIAL_BALANCE
    trades = []
    positions = {symbol: [] for symbol in COINS}
    
    for symbol in COINS:
        # Fetch data
        df = get_historical_data(symbol, TIMEFRAME, START_DATE, END_DATE)
        higher_tf_df = get_historical_data(symbol, HIGHER_TIMEFRAME, START_DATE, END_DATE)
        df = add_signal_indicators(df)
        higher_tf_df = add_trend_indicators(higher_tf_df)
        
        # Align higher timeframe with lower timeframe
        higher_tf_df = higher_tf_df.reindex(df.index, method='ffill')
        
        # Simulate trading
        for i in range(50, len(df)):  # Start after initial indicator warmup
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            atr = df['atr14'].iloc[i]
            
            # Manage open positions
            for pos in positions[symbol][:]:
                profit = (current_price - pos['entry_price']) * pos['size'] * (1 if pos['type'] == 'LONG' else -1)
                r_multiple = profit / pos['risk_per_r'] if pos['risk_per_r'] != 0 else 0
                time_in_position = (current_time - pos['entry_time']).total_seconds() / 3600
                
                if pos['type'] == 'LONG':
                    if r_multiple > 1 and not pos['breakeven_activated']:
                        pos['stop_loss'] = pos['entry_price']
                        pos['breakeven_activated'] = True
                    
                    if r_multiple > 1:
                        new_stop = df['ema21'].iloc[i]
                        if new_stop > pos['stop_loss']:
                            pos['stop_loss'] = new_stop
                    
                    if r_multiple >= 2 and not pos['first_target_hit']:
                        exit_size = pos['size'] * 0.2
                        exit_size = round_to_precision(symbol, exit_size)
                        if exit_size >= COINS[symbol]["min_size"]:
                            pos['size'] -= exit_size
                            pos['first_target_hit'] = True
                            profit = (current_price - pos['entry_price']) * exit_size * (1 - TAKER_FEE)
                            trades.append({'symbol': symbol, 'type': 'LONG (20% at 2R)', 'profit': profit})
                            balance += profit
                    
                    exit_conditions = [
                        current_price <= pos['stop_loss'],
                        df['ema_cross_down'].iloc[i],
                        df['macd_cross_down'].iloc[i],
                        df['rsi14'].iloc[i] > 70,
                        not higher_tf_df['uptrend'].iloc[i] and r_multiple > 1,
                        r_multiple >= 10,
                        r_multiple < -1 and time_in_position > 2
                    ]
                    if any(exit_conditions):
                        exit_price = current_price if current_price > pos['stop_loss'] else pos['stop_loss']
                        profit = (exit_price - pos['entry_price']) * pos['size'] * (1 - TAKER_FEE)
                        trades.append({'symbol': symbol, 'type': 'LONG (Final)', 'profit': profit})
                        balance += profit
                        positions[symbol].remove(pos)
                
                else:  # SHORT
                    if r_multiple > 1 and not pos['breakeven_activated']:
                        pos['stop_loss'] = pos['entry_price']
                        pos['breakeven_activated'] = True
                    
                    if r_multiple > 1:
                        new_stop = df['ema21'].iloc[i]
                        if new_stop < pos['stop_loss']:
                            pos['stop_loss'] = new_stop
                    
                    if r_multiple >= 2 and not pos['first_target_hit']:
                        exit_size = pos['size'] * 0.2
                        exit_size = round_to_precision(symbol, exit_size)
                        if exit_size >= COINS[symbol]["min_size"]:
                            pos['size'] -= exit_size
                            pos['first_target_hit'] = True
                            profit = (pos['entry_price'] - current_price) * exit_size * (1 - TAKER_FEE)
                            trades.append({'symbol': symbol, 'type': 'SHORT (20% at 2R)', 'profit': profit})
                            balance += profit
                    
                    exit_conditions = [
                        current_price >= pos['stop_loss'],
                        df['ema_cross_up'].iloc[i],
                        df['macd_cross_up'].iloc[i],
                        df['rsi14'].iloc[i] < 30,
                        not higher_tf_df['downtrend'].iloc[i] and r_multiple > 1,
                        r_multiple >= 10,
                        r_multiple < -1 and time_in_position > 2
                    ]
                    if any(exit_conditions):
                        exit_price = current_price if current_price < pos['stop_loss'] else pos['stop_loss']
                        profit = (pos['entry_price'] - exit_price) * pos['size'] * (1 - TAKER_FEE)
                        trades.append({'symbol': symbol, 'type': 'SHORT (Final)', 'profit': profit})
                        balance += profit
                        positions[symbol].remove(pos)
            
            # Check for new entries
            if balance > INITIAL_BALANCE * STOP_LOSS_THRESHOLD and sum(len(positions[s]) for s in COINS) < MAX_POSITIONS:
                signal = check_entry_conditions(df.iloc[:i+1], higher_tf_df.iloc[:i+1], symbol)
                if signal and not positions[symbol]:
                    position = simulate_trade(symbol, signal, current_price, atr, balance)
                    if position:
                        position['entry_time'] = current_time
                        positions[symbol].append(position)
                        logging.info(f"Opened {signal} position for {symbol} at {current_price}")
    
    # Calculate results
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_trades = len(trades_df[trades_df['profit'] > 0])
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    total_profit = trades_df['profit'].sum()
    profit_factor = trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if len(trades_df[trades_df['profit'] < 0]) > 0 else float('inf')
    
    logging.info(f"Backtest Results:\n"
                 f"Initial Balance: {INITIAL_BALANCE}\n"
                 f"Final Balance: {balance:.2f}\n"
                 f"Total Trades: {total_trades}\n"
                 f"Win Rate: {win_rate:.2f}%\n"
                 f"Profit Factor: {profit_factor:.2f}\n"
                 f"Total Profit: {total_profit:.2f}")
    print(f"Backtest Results:\n"
          f"Initial Balance: {INITIAL_BALANCE}\n"
          f"Final Balance: {balance:.2f}\n"
          f"Total Trades: {total_trades}\n"
          f"Win Rate: {win_rate:.2f}%\n"
          f"Profit Factor: {profit_factor:.2f}\n"
          f"Total Profit: {total_profit:.2f}")

if __name__ == "__main__":
    logging.info("Starting backtest...")
    START_DATE = "2025-02-04"
    END_DATE = "2025-03-08"    
    backtest(START_DATE=START_DATE,END_DATE=END_DATE)