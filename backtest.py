import os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
import logging

# Tải biến môi trường
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(API_KEY, API_SECRET)

# Cấu hình chung
COINS = ["SHIBUSDT", "DOGEUSDT", "PEPEUSDT", "FLOKIUSDT"]  # Danh sách meme coin
LEVERAGE = 5
TAKER_FEE = 0.0004
RISK_PER_TRADE = 0.005  # 0.5%
INITIAL_BALANCE = 10
TIMEFRAME = '1m'
DATA_DAYS = 10  # 30 ngày dữ liệu

# Logging
logging.basicConfig(
    filename='backtest_meme.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Lấy dữ liệu từ Binance
def get_historical_data(symbol, interval, days=30):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, 
                                       startTime=int(start_time.timestamp() * 1000), 
                                       endTime=int(end_time.timestamp() * 1000))
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'trades',
                                           'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0) & (df['volume'] >= 0)]
        df = df.dropna()
        logging.info(f"Lấy dữ liệu {symbol} ({interval}) thành công, {len(df)} nến")
        return df
    except Exception as e:
        logging.error(f"Lỗi lấy dữ liệu {symbol} ({interval}): {e}")
        return pd.DataFrame()

# Thêm chỉ báo
def add_indicators(df):
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['high_5'] = df['high'].rolling(5).max().shift(1)
    df['low_5'] = df['low'].rolling(5).min().shift(1)
    df['vol_ma10'] = df['volume'].rolling(10).mean().shift(1)
    df['breakout_up'] = (df['close'] > df['high_5']) & (df['volume'] > df['vol_ma10'] * 3)
    df['breakout_down'] = (df['close'] < df['low_5']) & (df['volume'] > df['vol_ma10'] * 3)
    return df

# Backtest chiến lược
def backtest_strategy(df, symbol):
    df = add_indicators(df)
    trades = []
    balance = INITIAL_BALANCE
    position = None
    equity_curve = [balance]
    
    for i, row in df.iterrows():
        current_price = row['close']
        
        # Quản lý vị thế
        if position:
            profit_pct = (current_price - position['entry_price']) / position['entry_price'] * (1 if position['type'] == 'LONG' else -1)
            profit = profit_pct * position['size'] * LEVERAGE * (1 - TAKER_FEE * 2)
            
            # Điều kiện thoát
            if position['type'] == 'LONG':
                if profit_pct >= 0.02 or profit_pct <= -0.01 or row['volume'] < row['vol_ma10']:
                    trades.append({
                        'symbol': symbol,
                        'type': 'LONG',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit,
                        'entry_time': position['entry_time'],
                        'exit_time': i
                    })
                    balance += profit
                    position = None
            else:  # SHORT
                if profit_pct >= 0.02 or profit_pct <= -0.01 or row['volume'] < row['vol_ma10']:
                    trades.append({
                        'symbol': symbol,
                        'type': 'SHORT',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit,
                        'entry_time': position['entry_time'],
                        'exit_time': i
                    })
                    balance += profit
                    position = None
        
        # Vào lệnh mới
        if not position:
            if row['breakout_up'] and row['ema5'] > row['ema13']:
                risk_amount = balance * RISK_PER_TRADE
                entry_price = current_price
                size = risk_amount / 0.01  # Rủi ro 1% giá
                position = {
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'size': size,
                    'entry_time': i
                }
            elif row['breakout_down'] and row['ema5'] < row['ema13']:
                risk_amount = balance * RISK_PER_TRADE
                entry_price = current_price
                size = risk_amount / 0.01  # Rủi ro 1% giá
                position = {
                    'type': 'SHORT',
                    'entry_price': entry_price,
                    'size': size,
                    'entry_time': i
                }
        
        equity_curve.append(balance)
    
    # Tính toán chỉ số hiệu suất
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {
            'Symbol': symbol,
            'Total Trades': 0,
            'Win Rate (%)': 0,
            'Profit Factor': 0,
            'Total Profit (USDT)': 0,
            'Max Drawdown (USDT)': 0,
            'Final Balance (USDT)': balance
        }
    
    win_trades = len(trades_df[trades_df['profit'] > 0])
    win_rate = win_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
    total_profit = trades_df['profit'].sum()
    gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
    gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) or 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    max_drawdown = max([0] + [max(equity_curve[:i+1]) - equity_curve[i] for i in range(len(equity_curve))])
    
    return {
        'Symbol': symbol,
        'Total Trades': len(trades_df),
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Total Profit (USDT)': total_profit,
        'Max Drawdown (USDT)': max_drawdown,
        'Final Balance (USDT)': balance
    }

# Chạy backtest cho danh sách coin
def run_backtest():
    all_results = []
    
    for symbol in COINS:
        logging.info(f"Bắt đầu backtest cho {symbol}")
        df = get_historical_data(symbol, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 50:
            logging.error(f"Dữ liệu cho {symbol} không đủ ({len(df) if not df.empty else 0} nến), cần ít nhất 50 nến")
            continue
        
        result = backtest_strategy(df, symbol)
        all_results.append(result)
        logging.info(f"Kết quả {symbol}: {result}")
    
    results_df = pd.DataFrame(all_results)
    print("\nKết quả Backtest:")
    print(results_df.to_string(index=False))
    logging.info("\nKết quả Backtest:")
    logging.info(results_df.to_string(index=False))

if __name__ == "__main__":
    logging.info("Bắt đầu backtest Momentum Breakout cho meme coin...")
    run_backtest()
    logging.info("Hoàn thành backtest!")