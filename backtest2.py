import pandas as pd
import numpy as np
from binance.client import Client
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

if not os.path.exists('temp'):
    os.makedirs('temp')

api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
client = Client(api_key, api_secret)

def get_historical_data(symbol, interval, start_date, end_date):
    print(f"Bắt đầu tải dữ liệu cho {symbol}...")
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    klines = []
    current = start_timestamp
    while current < end_timestamp:
        temp_klines = client.get_historical_klines(
            symbol=symbol, interval=interval,
            start_str=current,
            end_str=min(current + 1000 * 60 * 60 * 24 * 30, end_timestamp)
        )
        if not temp_klines:
            break
        klines.extend(temp_klines)
        current = temp_klines[-1][0] + 1
        time.sleep(0.5)

    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']] = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].apply(pd.to_numeric, errors='coerce')
    df.set_index('timestamp', inplace=True)
    return df

def get_higher_timeframe_data(symbol, interval, start_date, end_date):
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    adjusted_start = (start_date_obj - timedelta(days=30)).strftime("%Y-%m-%d")
    
    if interval == Client.KLINE_INTERVAL_15MINUTE or interval == Client.KLINE_INTERVAL_30MINUTE:
        higher_interval = Client.KLINE_INTERVAL_4HOUR
    else:
        higher_interval = Client.KLINE_INTERVAL_1HOUR
    
    print(f"Tải dữ liệu khung thời gian {higher_interval} cho {symbol}...")
    return get_historical_data(symbol, higher_interval, adjusted_start, end_date)

def add_trend_indicators(df):
    """Thêm các chỉ báo xu hướng trên khung thời gian lớn - với xử lý dữ liệu thiếu"""
    try:
        # Sử dụng EMA ngắn hơn để phù hợp với khung thời gian hạn chế
        df['ema50'] = ta.ema(df['close'], length=50)  # Thay EMA200 bằng EMA50 để giảm yêu cầu dữ liệu
        
        # Đảm bảo không có None trước khi tính slope - sử dụng bfill() thay vì fillna(method='bfill')
        df['ema50'] = df['ema50'].bfill()  # Backfill NaN values
        
        # Vẫn giữ ADX nhưng linh hoạt hơn - xử lý lỗi
        try:
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if 'ADX_14' in adx:
                df['adx'] = adx['ADX_14']
                df['di_plus'] = adx['DMP_14'] 
                df['di_minus'] = adx['DMN_14']
            else:
                # Nếu không có keys mong đợi, tạo cột giả với giá trị mặc định
                df['adx'] = 0
                df['di_plus'] = 0
                df['di_minus'] = 0
                print("Không tìm thấy thông tin ADX trong kết quả tính toán")
        except Exception as e:
            # Nếu có lỗi khi tính ADX, tạo cột giả
            print(f"Lỗi khi tính ADX: {str(e)}")
            df['adx'] = 0
            df['di_plus'] = 0
            df['di_minus'] = 0
        
        # Xử lý an toàn khi tính slope
        df['ema50_slope'] = pd.Series(dtype=float)  # Khởi tạo cột trống với kiểu float
        
        # Chỉ tính slope khi có đủ dữ liệu
        mask = df['ema50'].notna() & df['ema50'].shift(3).notna()
        if mask.any():
            df.loc[mask, 'ema50_slope'] = (
                df.loc[mask, 'ema50'].diff(3) / df.loc[mask, 'ema50'].shift(3) * 100
            )
        df['ema50_slope'] = df['ema50_slope'].fillna(0)  # Điền 0 cho các giá trị còn thiếu
        
        # Xu hướng tăng: Giá trên EMA50 và EMA50 có độ dốc dương
        df['uptrend'] = (df['close'] > df['ema50']) & (df['ema50_slope'] > 0.05)
        
        # Xu hướng giảm: Giá dưới EMA50 và EMA50 có độ dốc âm
        df['downtrend'] = (df['close'] < df['ema50']) & (df['ema50_slope'] < -0.05)
    
    except Exception as e:
        print(f"Lỗi khi tính toán chỉ báo xu hướng: {str(e)}")
        # Tạo các cột cần thiết với giá trị mặc định để tránh lỗi
        df['ema50'] = df['close']
        df['adx'] = 0
        df['di_plus'] = 0
        df['di_minus'] = 0
        df['ema50_slope'] = 0
        df['uptrend'] = False
        df['downtrend'] = False
    
    return df

def add_signal_indicators(df):
    """Thêm các chỉ báo tạo tín hiệu - đơn giản hóa và tối ưu"""
    # EMA cho tín hiệu cắt
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    
    # RSI giữ nguyên
    df['rsi14'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # ATR cho stop loss và take profit
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Tín hiệu cắt đơn giản hóa
    df['ema_cross_up'] = (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))
    df['ema_cross_down'] = (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))
    
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # Điểm breakout - thêm logic cho các đỉnh đáy
    df['high_5'] = df['high'].rolling(5).max()
    df['low_5'] = df['low'].rolling(5).min()
    df['breakout_up'] = (df['close'] > df['high_5'].shift(1)) & (df['close'].shift(1) <= df['high_5'].shift(2))
    df['breakout_down'] = (df['close'] < df['low_5'].shift(1)) & (df['close'].shift(1) >= df['low_5'].shift(2))
    
    # Volume xác nhận - đơn giản hóa
    df['volume_ma10'] = df['volume'].rolling(10).mean()
    df['volume_increase'] = df['volume'] > df['volume_ma10']
    
    return df

def backtest_momentum_strategy(df, higher_tf_df, initial_balance=10, leverage=20, risk_per_trade=0.02):
    """Chiến lược Momentum đã tối ưu hóa"""
    df = df.copy()
    
    # Đảm bảo tất cả columns cần thiết đều có trong DataFrame
    required_columns = ['ema9', 'ema21', 'rsi14', 'atr14', 'macd']
    for col in required_columns:
        if col not in df.columns:
            print(f"Thiếu cột {col} trong dữ liệu")
            return pd.DataFrame(), initial_balance, 0, 0
    
    df = df.dropna(subset=required_columns)
    
    # Đảm bảo higher_tf_df có cấu trúc cần thiết
    higher_tf_df = higher_tf_df.copy() if higher_tf_df is not None else pd.DataFrame(index=df.index)
    
    # Đảm bảo tất cả các cột cần thiết tồn tại trước khi sử dụng
    required_higher_tf_columns = ['ema50', 'adx', 'di_plus', 'di_minus', 'uptrend', 'downtrend']
    for col in required_higher_tf_columns:
        if col not in higher_tf_df.columns:
            if col == 'ema50':
                higher_tf_df[col] = higher_tf_df['close'] if 'close' in higher_tf_df.columns else 0
            else:
                higher_tf_df[col] = 0 if col in ['adx', 'di_plus', 'di_minus'] else False
    
    # Map dữ liệu từ khung thời gian cao với xử lý ngoại lệ
    try:
        # Đảm bảo higher_tf_trends có các kiểu dữ liệu phù hợp từ đầu
        higher_tf_trends = pd.DataFrame(index=df.index)
        higher_tf_trends['uptrend'] = False
        higher_tf_trends['downtrend'] = False
        higher_tf_trends['adx'] = 0.0  # Chuyển sang kiểu float thay vì int
        higher_tf_trends['di_plus'] = 0.0  # Chuyển sang kiểu float
        higher_tf_trends['di_minus'] = 0.0  # Chuyển sang kiểu float
        
        for i, row in df.iterrows():
            # Tìm điểm dữ liệu khung thời gian cao gần nhất
            mask = higher_tf_df.index <= i
            if mask.any() and not higher_tf_df.empty:
                latest_higher_tf = higher_tf_df[mask].iloc[-1]
                # Chỉ gán giá trị nếu chúng tồn tại trong latest_higher_tf
                for col in ['uptrend', 'downtrend', 'adx', 'di_plus', 'di_minus']:
                    if col in latest_higher_tf and pd.notna(latest_higher_tf[col]):
                        # Ép kiểu một cách rõ ràng để tránh cảnh báo
                        if col in ['adx', 'di_plus', 'di_minus']:
                            higher_tf_trends.loc[i, col] = float(latest_higher_tf[col])
                        else:
                            higher_tf_trends.loc[i, col] = bool(latest_higher_tf[col])
    except Exception as e:
        print(f"Lỗi khi map dữ liệu khung thời gian lớn: {str(e)}")
        # Tiếp tục với giá trị mặc định
    
    # Gán các giá trị đã map vào df chính
    df['higher_uptrend'] = higher_tf_trends['uptrend']
    df['higher_downtrend'] = higher_tf_trends['downtrend']
    df['higher_adx'] = higher_tf_trends['adx']
    df['higher_di_plus'] = higher_tf_trends['di_plus']
    df['higher_di_minus'] = higher_tf_trends['di_minus']
    
    trades = []
    balance = initial_balance
    position = None
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # ====== ENTRY CONDITIONS - Simplified & Safe ======
        # Primary conditions - must all be true
        long_primary = [
            current['ema9'] > current['ema21'],                 # EMA9 > EMA21 (in uptrend)
            current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']  # Signal
        ]
        
        # Secondary conditions - at least 1 must be true
        long_secondary = [
            current['rsi14'] < 70,                              # Not overbought
            current['volume_increase'],                         # Volume confirmation
            current['macd'] > 0                                 # MACD is positive
        ]
        
        # Only trade LONG if higher timeframe is in uptrend or in neutral trend with strong momentum
        # Use safe access to higher_adx with fallback values
        higher_adx_value = current['higher_adx'] if pd.notna(current['higher_adx']) else 0
        higher_di_plus = current['higher_di_plus'] if pd.notna(current['higher_di_plus']) else 0
        higher_di_minus = current['higher_di_minus'] if pd.notna(current['higher_di_minus']) else 0
        
        long_condition = (
            all(long_primary) and 
            any(long_secondary) and
            (current['higher_uptrend'] or (higher_adx_value > 25 and higher_di_plus > higher_di_minus))
        )
        
        # Short condition logic (similar safe approach)
        short_primary = [
            current['ema9'] < current['ema21'],                 # EMA9 < EMA21 (in downtrend)
            current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']  # Signal
        ]
        
        short_secondary = [
            current['rsi14'] > 30,                              # Not oversold
            current['volume_increase'],                         # Volume confirmation
            current['macd'] < 0                                 # MACD is negative
        ]
        
        short_condition = (
            all(short_primary) and 
            any(short_secondary) and
            (current['higher_downtrend'] or (higher_adx_value > 25 and higher_di_minus > higher_di_plus))
        )
        
        # ====== POSITION MANAGEMENT ======
        if position:
            current_price = current['close']
            profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
            
            # Calculate R-multiple (how many times risk is current profit/loss)
            risk_amount = position['risk_per_r']
            r_multiple = profit / risk_amount if risk_amount != 0 else 0
            
            # ====== EXIT LOGIC - LONG ======
            if position['type'] == 'LONG':
                # Break-even setup - move stop to entry when profit > 0.7R
                if r_multiple > 0.7 and not position['breakeven_activated']:
                    position['stop_loss'] = position['entry_price']  # Move stop to entry
                    position['breakeven_activated'] = True
                
                # Trailing stop logic - more dynamic
                if r_multiple > 1:  # If profit > 1R
                    # Use trailing stop based on ATR
                    trail_factor = min(1.5, 1 + r_multiple * 0.1)  # Adaptive trail factor
                    new_stop = current_price - current['atr14'] * trail_factor
                    position['stop_loss'] = max(position['stop_loss'], new_stop)
                
                # First partial exit at 1.5R
                if r_multiple >= 1.5 and not position['first_target_hit']:
                    exit_price = current_price
                    exit_size = position['size'] * 0.3  # Exit 30%
                    position['size'] -= exit_size
                    position['first_target_hit'] = True
                    
                    trade = {
                        'type': position['type'] + " (Partial 30%)",
                        'entry_time': position['entry_time'],
                        'exit_time': current.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': (exit_price - position['entry_price']) * exit_size,
                        'hold_time': (current.name - position['entry_time']).total_seconds()/3600,
                        'r_multiple': r_multiple
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    print(f"Partial exit (30%): {trade}")  # Debug print
                
                # Second partial exit at 2.5R
                elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
                    exit_price = current_price
                    exit_size = position['size'] * 0.5  # Exit 50% of remaining
                    position['size'] -= exit_size
                    position['second_target_hit'] = True
                    
                    trade = {
                        'type': position['type'] + " (Partial 50%)",
                        'entry_time': position['entry_time'],
                        'exit_time': current.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': (exit_price - position['entry_price']) * exit_size,
                        'hold_time': (current.name - position['entry_time']).total_seconds()/3600,
                        'r_multiple': r_multiple
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    print(f"Partial exit (50%): {trade}")  # Debug print
                
                # Exit conditions 
                exit_conditions = [
                    current_price <= position['stop_loss'],                     # Stop loss hit
                    current['ema_cross_down'],                                  # EMA crossover down
                    current['macd_cross_down'],                                 # MACD crossover down
                    current['rsi14'] > 80,                                      # RSI overbought
                    not current['higher_uptrend'] and r_multiple > 0,           # Lost higher timeframe trend and in profit
                    r_multiple >= 4                                             # Take full profit at 4R
                ]
                
                if any(exit_conditions):
                    exit_price = current_price if current_price > position['stop_loss'] else position['stop_loss']
                    
                    trade = {
                        'type': position['type'] + " (Final)",
                        'entry_time': position['entry_time'],
                        'exit_time': current.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': (exit_price - position['entry_price']) * position['size'],
                        'hold_time': (current.name - position['entry_time']).total_seconds()/3600,
                        'r_multiple': r_multiple
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    position = None
                    print(f"Final exit: {trade}")  # Debug print
            
            # ====== EXIT LOGIC - SHORT ======
            else:  # SHORT position
                # Similar logic for short positions but reversed
                if r_multiple > 0.7 and not position['breakeven_activated']:
                    position['stop_loss'] = position['entry_price']
                    position['breakeven_activated'] = True
                
                if r_multiple > 1:
                    trail_factor = min(1.5, 1 + r_multiple * 0.1)
                    new_stop = current_price + current['atr14'] * trail_factor
                    position['stop_loss'] = min(position['stop_loss'], new_stop) if position['stop_loss'] != position['entry_price'] else position['stop_loss']
                
                # First partial exit at 1.5R
                if r_multiple >= 1.5 and not position['first_target_hit']:
                    exit_price = current_price
                    exit_size = position['size'] * 0.3
                    position['size'] -= exit_size
                    position['first_target_hit'] = True
                    
                    trade = {
                        'type': position['type'] + " (Partial 30%)",
                        'entry_time': position['entry_time'],
                        'exit_time': current.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': (position['entry_price'] - exit_price) * exit_size,
                        'hold_time': (current.name - position['entry_time']).total_seconds()/3600,
                        'r_multiple': r_multiple
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    print(f"Partial exit (30%): {trade}")  # Debug print
                
                # Second partial exit at 2.5R
                elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
                    exit_price = current_price
                    exit_size = position['size'] * 0.5
                    position['size'] -= exit_size
                    position['second_target_hit'] = True
                    
                    trade = {
                        'type': position['type'] + " (Partial 50%)",
                        'entry_time': position['entry_time'],
                        'exit_time': current.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': (position['entry_price'] - exit_price) * exit_size,
                        'hold_time': (current.name - position['entry_time']).total_seconds()/3600,
                        'r_multiple': r_multiple
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    print(f"Partial exit (50%): {trade}")  # Debug print
                
                exit_conditions = [
                    current_price >= position['stop_loss'],
                    current['ema_cross_up'],
                    current['macd_cross_up'],
                    current['rsi14'] < 20,
                    not current['higher_downtrend'] and r_multiple > 0,
                    r_multiple >= 4
                ]
                
                if any(exit_conditions):
                    exit_price = current_price if current_price < position['stop_loss'] else position['stop_loss']
                    
                    trade = {
                        'type': position['type'] + " (Final)",
                        'entry_time': position['entry_time'],
                        'exit_time': current.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': (position['entry_price'] - exit_price) * position['size'],
                        'hold_time': (current.name - position['entry_time']).total_seconds()/3600,
                        'r_multiple': r_multiple
                    }
                    trades.append(trade)
                    balance += trade['profit']
                    position = None
                    print(f"Final exit: {trade}")  # Debug print
        
        # ====== ENTRY LOGIC ======
        if not position and balance > initial_balance*0.1 and balance > 0:
            if long_condition or short_condition:
                position_type = 'LONG' if long_condition else 'SHORT'
                entry_price = current['close']
                
                # ATR-based stop loss
                atr = current['atr14']
                
                # Smarter stop loss placement based on recent price action
                if position_type == 'LONG':
                    # Look back 5 bars for a swing low, or use ATR if no clear level
                    recent_low = df['low'].iloc[max(0, i-5):i].min()
                    if recent_low < entry_price * 0.99:  # Use swing low if it's less than 1% away
                        stop_distance = entry_price - recent_low
                        stop_loss = recent_low - atr * 0.3  # Add small buffer
                    else:
                        stop_loss = entry_price - atr * 1.5  # Default to 1.5 ATR
                else:  # SHORT
                    recent_high = df['high'].iloc[max(0, i-5):i].max()
                    if recent_high > entry_price * 1.01:  # Use swing high if it's more than 1% away
                        stop_distance = recent_high - entry_price
                        stop_loss = recent_high + atr * 0.3  # Add small buffer
                    else:
                        stop_loss = entry_price + atr * 1.5  # Default to 1.5 ATR
                
                # Risk per trade calculation
                risk_per_r = entry_price - stop_loss if position_type == 'LONG' else stop_loss - entry_price
                risk_amount = balance * risk_per_trade
                position_size = risk_amount / risk_per_r * leverage
                
                position = {
                    'type': position_type,
                    'entry_time': current.name,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'size': position_size,
                    'risk_per_r': risk_amount,  # Store the risk amount for R calculation
                    'breakeven_activated': False,
                    'first_target_hit': False,
                    'second_target_hit': False
                }
    
    # Process and return trades
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['profit_pct'] = (trades_df['profit'] / initial_balance) * 100
        trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
        
        trades_df['peak'] = trades_df['cumulative_profit'].cummax()
        trades_df['drawdown'] = trades_df['peak'] - trades_df['cumulative_profit']
        
        final_balance = initial_balance + trades_df['profit'].sum()
        profit_percent = (final_balance/initial_balance - 1)*100
    else:
        final_balance = initial_balance
        profit_percent = 0
    
    return trades_df, final_balance, final_balance - initial_balance, profit_percent

def analyze_results(trades_df, initial_balance, final_balance, profit_percent):
    if trades_df.empty:
        print("Không có giao dịch nào!")
        return None

    trades_df['profit'] = trades_df['profit'].fillna(0)
    win_trades = len(trades_df[trades_df['profit'] > 0])
    loss_trades = len(trades_df[trades_df['profit'] <= 0])
    win_rate = (win_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    avg_profit = trades_df['profit'].mean()
    total_profit = trades_df['profit'].sum()

    avg_win_pct = trades_df[trades_df['profit'] > 0]['profit_pct'].mean() if win_trades > 0 else 0
    avg_loss_pct = trades_df[trades_df['profit'] <= 0]['profit_pct'].mean() if loss_trades > 0 else 0
    max_win_pct = trades_df['profit_pct'].max() if not trades_df.empty else 0
    max_loss_pct = trades_df['profit_pct'].min() if not trades_df.empty else 0

    gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum() if win_trades > 0 else 0
    gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if loss_trades > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_hold_time = trades_df['hold_time'].mean() if 'hold_time' in trades_df.columns else 0

    trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
    trades_df['peak'] = trades_df['cumulative_profit'].cummax()
    trades_df['drawdown'] = trades_df['peak'] - trades_df['cumulative_profit']
    max_drawdown = trades_df['drawdown'].max()
    max_drawdown_pct = max_drawdown / (initial_balance + trades_df['peak'].max()) * 100 if trades_df['peak'].max() > 0 else 0

    long_trades = len(trades_df[trades_df['type'] == 'LONG'])
    short_trades = len(trades_df[trades_df['type'] == 'SHORT'])

    print("\n===== KẾT QUẢ BACKTEST MOMENTUM =====")
    print(f"Tổng giao dịch: {len(trades_df)}")
    print(f"Thắng: {win_trades} ({win_rate:.2f}%)")
    print(f"Thua: {loss_trades}")
    print(f"Long/Short: {long_trades}/{short_trades}")
    print(f"Lợi nhuận trung bình mỗi giao dịch: {avg_profit:.4f} USDT")
    print(f"Lợi nhuận trung bình giao dịch thắng: {avg_win_pct:.2f}%")
    print(f"Thua lỗ trung bình giao dịch thua: {avg_loss_pct:.2f}%")
    print(f"Lợi nhuận lớn nhất: {max_win_pct:.2f}%")
    print(f"Thua lỗ lớn nhất: {max_loss_pct:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Thời gian giữ lệnh trung bình: {avg_hold_time:.2f} giờ")
    print(f"Max Drawdown: {max_drawdown:.4f} USDT ({max_drawdown_pct:.2f}%)")
    print(f"Tổng lợi nhuận: {total_profit:.4f} USDT")
    print(f"Số dư ban đầu: {initial_balance} USDT")
    print(f"Số dư cuối: {final_balance:.4f} USDT")
    print(f"Lợi nhuận %: {profit_percent:.2f}%")

    return trades_df

def plot_trades(df, trades_df, symbol, start_date, end_date):
    """Vẽ biểu đồ giao dịch và lợi nhuận"""
    if trades_df.empty:
        return

    # Tạo DataFrame để lưu lợi nhuận theo thời gian
    profit_df = pd.DataFrame(index=df.index)
    profit_df['cumulative_profit'] = 0.0

    # Thêm profit từ các giao dịch vào thời điểm exit
    for idx, row in trades_df.iterrows():
        if pd.notna(row['exit_time']):
            time_idx = profit_df.index.get_indexer([row['exit_time']], method='nearest')[0]
            if time_idx >= 0 and time_idx < len(profit_df):
                profit_df.loc[profit_df.index[time_idx], 'cumulative_profit'] = row['profit']

    profit_df['cumulative_profit'] = profit_df['cumulative_profit'].fillna(0.0)
    profit_df['cumulative_profit'] = profit_df['cumulative_profit'].cumsum()
    
    # Vẽ biểu đồ
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    
    # Price chart với entry/exit points
    ax1 = plt.subplot(gs[0])
    
    # Vẽ đường giá
    ax1.plot(df.index, df['close'], color='blue', linewidth=1, label='Price')
    
    # Thêm entry/exit points
    for _, trade in trades_df.iterrows():
        # Entry points
        if trade['type'] == 'LONG':
            ax1.scatter(trade['entry_time'], trade['entry_price'], 
                       marker='^', color='g', s=100, label='Long Entry' if 'Long Entry' not in ax1.get_legend_handles_labels()[1] else '')
        else:
            ax1.scatter(trade['entry_time'], trade['entry_price'], 
                       marker='v', color='r', s=100, label='Short Entry' if 'Short Entry' not in ax1.get_legend_handles_labels()[1] else '')
        
        # Exit points với profit/loss annotation
        profit_pct = trade['profit_pct']
        color = 'g' if profit_pct > 0 else 'r'
        ax1.scatter(trade['exit_time'], trade['exit_price'], 
                   marker='X', color=color, s=100, label='Exit' if 'Exit' not in ax1.get_legend_handles_labels()[1] else '')
        
        # Thêm annotation cho profit/loss
        ax1.annotate(f'{profit_pct:.1f}%', 
                    xy=(trade['exit_time'], trade['exit_price']),
                    xytext=(10, 10), textcoords='offset points',
                    color=color)
    
    ax1.set_title(f'Price Chart - {symbol}')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Volume chart
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.bar(df.index, df['volume'], color='blue', alpha=0.5)
    ax2.set_title('Volume')
    ax2.grid(True, alpha=0.3)
    
    # Profit chart
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(profit_df.index, profit_df['cumulative_profit'], color='blue', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax3.fill_between(profit_df.index, profit_df['cumulative_profit'], 0, 
                     where=(profit_df['cumulative_profit'] >= 0), color='green', alpha=0.3)
    ax3.fill_between(profit_df.index, profit_df['cumulative_profit'], 0, 
                     where=(profit_df['cumulative_profit'] < 0), color='red', alpha=0.3)
    
    # Thêm các điểm profit/loss trên đường equity
    for _, trade in trades_df.iterrows():
        if pd.notna(trade['exit_time']):
            cumulative_profit = profit_df.loc[profit_df.index <= trade['exit_time'], 'cumulative_profit'].iloc[-1]
            color = 'g' if trade['profit'] > 0 else 'r'
            ax3.scatter(trade['exit_time'], cumulative_profit, color=color, s=50)
            
    ax3.set_title('Cumulative P&L')
    ax3.set_ylabel('Profit/Loss')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Định dạng chung
    plt.tight_layout()
    filename = f'temp/trades_chart_{symbol}_{start_date}_{end_date}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ giao dịch tại: {filename}")

def test_strategy(start_date, end_date, interval, top_symbols):
    results_summary = []
    for symbol in top_symbols:
        print(f"\n{'='*50}")
        print(f"BACKTEST MOMENTUM STRATEGY TẠI {symbol}")
        print(f"{'='*50}")

        try:
            # Tăng thời gian lấy dữ liệu quá khứ cho khung thời gian cao hơn
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            adjusted_start = (start_date_obj - timedelta(days=60)).strftime("%Y-%m-%d")  # Tăng lên 60 ngày
            
            # Xử lý an toàn cho dữ liệu
            try:
                df = get_historical_data(symbol, interval, start_date, end_date)
            except Exception as e:
                print(f"Lỗi khi tải dữ liệu chính: {str(e)}")
                df = pd.DataFrame()
                
            try:
                higher_tf_df = get_higher_timeframe_data(symbol, interval, adjusted_start, end_date)
            except Exception as e:
                print(f"Lỗi khi tải dữ liệu khung thời gian cao: {str(e)}")
                higher_tf_df = pd.DataFrame()
            
            if df.empty:
                print("Không tải được dữ liệu đầy đủ!")
                results_summary.append({'Symbol': symbol, 'Số giao dịch': 0, 'Tỷ lệ thắng (%)': 0, 'P&L (USDT)': 0, 'Lợi nhuận (%)': 0})
                continue

            df = add_signal_indicators(df)
            higher_tf_df = add_trend_indicators(higher_tf_df)
            
            trades_df, final_balance, profit, profit_percent = backtest_momentum_strategy(
                df, higher_tf_df, initial_balance=10, leverage=5, risk_per_trade=0.02
            )
            
            if not trades_df.empty:
                trades_df.to_csv(f'temp/momentum_trades_{symbol}_{start_date}_{end_date}.csv')
                print(f"Đã lưu giao dịch tại: temp/momentum_trades_{symbol}_{start_date}_{end_date}.csv")
                
                analyzed_trades = analyze_results(trades_df, 10, final_balance, profit_percent)
                
                win_trades = len(trades_df[trades_df['profit'] > 0])
                total_trades = len(trades_df)
                win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
                
                print("\n===== KẾT QUẢ BACKTEST MOMENTUM =====")
                print(f"Tổng giao dịch: {total_trades}")
                print(f"Thắng: {win_trades} ({win_rate:.2f}%)")
                print(f"P&L: {profit:.4f} USDT") 
                print(f"Lợi nhuận %: {profit_percent:.2f}%")
                
                plot_trades(df, trades_df, symbol, start_date, end_date)
                
                results_summary.append({
                    'Symbol': symbol,
                    'Số giao dịch': total_trades,
                    'Tỷ lệ thắng (%)': round(win_rate, 2),
                    'P&L (USDT)': round(profit, 2),
                    'Lợi nhuận (%)': round(profit_percent, 2)
                })
            else:
                print("Không có giao dịch nào được thực hiện!")
                results_summary.append({
                    'Symbol': symbol, 
                    'Số giao dịch': 0, 
                    'Tỷ lệ thắng (%)': 0, 
                    'P&L (USDT)': 0, 
                    'Lợi nhuận (%)': 0
                })
        except Exception as e:
            print(f"Lỗi khi xử lý {symbol}: {str(e)}")
            results_summary.append({'Symbol': symbol, 'Số giao dịch': 0, 'Tỷ lệ thắng (%)': 0, 'P&L (USDT)': 0, 'Lợi nhuận (%)': 0})

    if results_summary:
        results_df = pd.DataFrame(results_summary)
        print("\n" + "="*60)
        print("BẢNG THỐNG KÊ LỢI NHUẬN CHIẾN LƯỢC MOMENTUM")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60)
        results_df.to_csv(f'temp/momentum_summary_{start_date}_{end_date}.csv', index=False)
        print(f"Đã lưu bảng thống kê tại: temp/momentum_summary_{start_date}_{end_date}.csv")

if __name__ == "__main__":
    start_date = "2025-03-03"
    end_date = "2025-03-04"
    interval = Client.KLINE_INTERVAL_5MINUTE
    # symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
    symbols = [
    # "1000PEPEUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "BOMEUSDT",
    "ADAUSDT",
    ]
    test_strategy(start_date, end_date, interval, symbols)