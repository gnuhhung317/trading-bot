import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
import pandas_ta as ta
from datetime import datetime, timedelta
import time

# Kết nối API Binance (chỉ cần để lấy dữ liệu lịch sử)
api_key = 'Wvjb9scOCRwa95JI3eHvdrgr7XaPQx18mVzd0hfxuLSbiqwFHhjQYlKusMkCIIge'
api_secret = 'iIkZjKZCDlwjxJG6tPaEnU9uVWHR6Ygh2al3HNYsBKy77Ne8jZwrzGLThGwtRTfL'

client = Client(api_key, api_secret)

def get_historical_data(symbol, interval, start_date, end_date=None):
    """Lấy dữ liệu lịch sử từ Binance"""
    print("Bắt đầu tải dữ liệu...")
    # Chuyển đổi ngày thành timestamp
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    
    if end_date:
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    else:
        end_timestamp = int(datetime.now().timestamp() * 1000)
    
    # Lấy dữ liệu theo từng chunk để tránh giới hạn API
    klines = []
    current = start_timestamp
    
    while current < end_timestamp:
        temp_klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=current,
            end_str=min(current + 1000 * 60 * 60 * 24 * 90, end_timestamp) # 90 ngày một lần
        )
        if not temp_klines:
            break
        klines.extend(temp_klines)
        # Lấy timestamp cuối cùng và cộng thêm 1ms để tránh trùng lặp
        current = temp_klines[-1][0] + 1
        time.sleep(1)  # Tránh tràn request API
    
    # Thêm thông tin để gỡ lỗi
    print(f"Đã tải {len(klines)} nến dữ liệu.")
    if len(klines) == 0:
        print("CẢNH BÁO: Không có dữ liệu nào được tải về!")
        return pd.DataFrame()
    
    # Chuyển đổi dữ liệu thành DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])
    
    # Chuyển đổi kiểu dữ liệu
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    df.set_index('timestamp', inplace=True)
    print(f"Đã xử lý dữ liệu thành DataFrame kích thước {df.shape}")
    return df

def add_indicators(df):
    """Thêm các chỉ báo vào dữ liệu"""
    # Chỉ báo xu hướng
    df.ta.ema(length=20, append=True, col_names=('ema20',))
    df.ta.ema(length=50, append=True, col_names=('ema50',))
    
    # Chỉ thêm EMA200 nếu có đủ dữ liệu
    if len(df) >= 200:
        df.ta.ema(length=200, append=True, col_names=('ema200',))
    
    # Chỉ báo dao động
    df.ta.rsi(length=14, append=True, col_names=('rsi',))
    df.ta.atr(length=14, append=True, col_names=('atr',))
    
    # Thêm chỉ báo khối lượng
    df.ta.obv(append=True)
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    
    # Thêm nến tăng/giảm
    df['candle_body'] = df['close'] - df['open']
    df['bullish'] = df['candle_body'] > 0
    
    # Thêm nhận diện trạng thái thị trường - kiểm tra nếu có ema200
    if 'ema200' in df.columns:
        df['market_trend'] = np.where(
            (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200']), 
            'uptrend',
            np.where((df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200']), 
                    'downtrend', 
                    'sideways')
        )
    else:
        # Chỉ sử dụng ema20 và ema50 nếu không có ema200
        df['market_trend'] = np.where(
            df['ema20'] > df['ema50'], 
            'uptrend',
            np.where(df['ema20'] < df['ema50'], 
                    'downtrend', 
                    'sideways')
        )
    
    # Thêm chỉ báo biến động
    df['volatility'] = df['atr'] / df['close'] * 100
    
    return df

def backtest_strategy(df, initial_balance=1000, risk_per_trade=0.5):
    """Chiến lược giao dịch cải tiến với quản lý rủi ro tốt hơn"""
    # Khởi tạo các biến theo dõi
    balance = initial_balance
    coin_amount = 0
    in_position = False
    trades = []
    entry_price = 0
    risk_reward_ratio = 2.0  # Tăng tỷ lệ risk/reward
    wait_count = 0  # Đếm số nến phải đợi sau khi bán
    
    # Thêm biến phí giao dịch và trượt giá
    fee_rate = 0.001  # 0.1% phí giao dịch
    slippage = 0.0005  # 0.05% trượt giá
    
    # Thêm column volume_change để theo dõi thay đổi khối lượng
    df['volume_change'] = df['volume'].pct_change()
    
    # Lặp qua từng dòng dữ liệu
    for i in range(200, len(df)):  # Bắt đầu từ nến 200 để có đủ dữ liệu cho các MA dài
        current = df.iloc[i]
        previous = df.iloc[i-1]
        
        # Đang trong thời gian chờ sau khi bán
        if wait_count > 0:
            wait_count -= 1
            continue
        
        # ĐIỀU KIỆN MUA CẢI TIẾN
        if 'ema200' in df.columns:
            buy_condition = (
                # Điều kiện xu hướng
                (current['close'] > current['ema20']) and 
                (current['ema20'] > current['ema50']) and
                (current['ema50'] > current['ema200']) and  # Thêm xu hướng dài hạn
                (current['market_trend'] == 'uptrend') and  # Chỉ giao dịch trong xu hướng tăng
                
                # Lọc theo RSI
                (current['rsi'] > 40 and current['rsi'] < 70) and
                
                # Xác nhận khối lượng
                (current['volume'] > current['volume_ma5']) and
                
                # Điều kiện tín hiệu
                (
                    # Cắt lên mạnh mẽ
                    (previous['ema20'] <= previous['ema50'] and current['ema20'] > current['ema50']) or
                    
                    # Pullback và breakout
                    (previous['close'] < previous['ema20'] and 
                     current['close'] > current['ema20'] and
                     current['close'] > previous['high'])
                ) and
                
                # Lọc biến động
                (current['volatility'] > 0.5 and current['volatility'] < 3.0)
            )
        else:
            # Phiên bản đơn giản hơn khi không có EMA200
            buy_condition = (
                # Điều kiện xu hướng đơn giản hơn
                (current['close'] > current['ema20']) and 
                (current['ema20'] > current['ema50']) and
                (current['market_trend'] == 'uptrend') and
                
                # Lọc theo RSI
                (current['rsi'] > 40 and current['rsi'] < 70) and
                
                # Xác nhận khối lượng
                (current['volume'] > current['volume_ma5']) and
                
                # Điều kiện tín hiệu
                (
                    (previous['ema20'] <= previous['ema50'] and current['ema20'] > current['ema50']) or
                    (previous['close'] < previous['ema20'] and current['close'] > current['ema20'])
                ) and
                
                # Lọc biến động
                (current['volatility'] > 0.5 and current['volatility'] < 3.0)
            )
        
        # QUẢN LÝ LỆNH CHẶT CHẼ HƠN
        if in_position:
            # Di chuyển SL về breakeven khi đạt 50% target
            if current['close'] >= entry_price + ((take_profit - entry_price) * 0.5):
                stop_loss = max(entry_price, stop_loss)  # Breakeven hoặc cao hơn
            
            # Điều kiện bán tốt hơn
            sell_condition = (
                (current['close'] <= stop_loss) or                     # Stop loss
                (current['close'] >= take_profit) or                   # Take profit
                # Đảo chiều với khối lượng tăng
                (current['close'] < current['ema20'] and 
                 previous['close'] > previous['ema20'] and
                 current['volume'] > previous['volume'] * 1.2)  # Khối lượng tăng khi phá vỡ xu hướng
            )
        else:
            sell_condition = False
        
        # Xử lý mua
        if not in_position and buy_condition:
            price = current['close'] * (1 + slippage)  # Thêm trượt giá khi mua
            entry_price = price
            
            # Tính position size dựa trên ATR
            risk_amount = balance * risk_per_trade / 100
            atr_stop = current['atr'] * 2.0  # Tăng khoảng cách stop loss
            position_size = risk_amount / atr_stop
            
            # Đảm bảo không sử dụng quá 20% số dư cho một giao dịch
            max_position = balance * 0.2
            if position_size * price > max_position:
                position_size = max_position / price
                
            coin_amount = position_size
            
            # Tính stop loss và take profit
            stop_loss = price - atr_stop
            take_profit = price + (atr_stop * risk_reward_ratio)
            
            # Trừ phí giao dịch
            fee = position_size * price * fee_rate
            balance -= position_size * price + fee
            in_position = True
            
            # Ghi nhận giao dịch
            trades.append({
                'date': df.index[i],
                'type': 'BUY',
                'price': price,
                'amount': coin_amount,
                'balance': balance,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'fee': fee,
                'reason': 'NEW_TREND' if previous['ema20'] <= previous['ema50'] else 'PULLBACK'
            })
        
        # Xử lý bán
        elif in_position and sell_condition:
            price = current['close'] * (1 - slippage)  # Thêm trượt giá khi bán
            sell_value = coin_amount * price
            
            # Trừ phí giao dịch
            fee = sell_value * fee_rate
            balance += sell_value - fee
            
            # Xác định lý do bán
            reason = "UNKNOWN"
            if price <= stop_loss:
                reason = "STOP_LOSS"
            elif price >= take_profit:
                reason = "TAKE_PROFIT"
            elif current['close'] < current['ema20'] and previous['close'] > previous['ema20']:
                reason = "TREND_BREAK"
            
            # Ghi nhận giao dịch
            trades.append({
                'date': df.index[i],
                'type': 'SELL',
                'price': price,
                'amount': coin_amount,
                'balance': balance,
                'fee': fee,
                'reason': reason,
                'profit_pct': ((price / entry_price) - 1) * 100
            })
            
            # Reset biến
            coin_amount = 0
            in_position = False
            entry_price = 0
            
            # Đợi ít nhất 5 nến sau khi bán trước khi giao dịch tiếp
            wait_count = 5
        
    # Bán nốt coin nếu còn
    if in_position:
        price = df.iloc[-1]['close'] * (1 - slippage)
        sell_value = coin_amount * price
        fee = sell_value * fee_rate
        balance += sell_value - fee
        
        trades.append({
            'date': df.index[-1],
            'type': 'SELL',
            'price': price,
            'amount': coin_amount,
            'balance': balance,
            'fee': fee,
            'reason': 'END_OF_PERIOD',
            'profit_pct': ((price / entry_price) - 1) * 100
        })
    
    # Tạo DataFrame từ danh sách giao dịch
    trades_df = pd.DataFrame(trades)
    
    # Tính lợi nhuận
    final_balance = balance
    profit = final_balance - initial_balance
    profit_percent = (profit / initial_balance) * 100
    
    return trades_df, final_balance, profit, profit_percent

def analyze_results(df, trades_df, initial_balance, final_balance):
    """Phân tích kết quả backtest và tạo thống kê"""
    # Kiểm tra nếu không có giao dịch
    if trades_df.empty:
        return None
    
    # Tạo DataFrame các cặp giao dịch mua/bán
    trades_df['trade_num'] = 0
    trade_num = 0
    
    for i in range(len(trades_df)):
        if trades_df.iloc[i]['type'] == 'BUY':
            trade_num += 1
        trades_df.iloc[i, trades_df.columns.get_loc('trade_num')] = trade_num
    
    # Tạo DataFrame cho các cặp giao dịch
    pairs = []
    buy_trades = trades_df[trades_df['type'] == 'BUY']
    sell_trades = trades_df[trades_df['type'] == 'SELL']
    
    for _, buy in buy_trades.iterrows():
        # Tìm giao dịch bán tương ứng
        sell = sell_trades[sell_trades['trade_num'] == buy['trade_num']]
        if not sell.empty:
            sell = sell.iloc[0]
            entry_date = buy['date']
            exit_date = sell['date']
            entry_price = buy['price']
            exit_price = sell['price']
            profit_pct = ((exit_price / entry_price) - 1) * 100
            hold_days = (exit_date - entry_date).days + ((exit_date - entry_date).seconds / 86400)  # Tính theo ngày và giờ
            
            pairs.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'hold_days': hold_days,
                'is_profit': profit_pct > 0,
                'exit_reason': sell['reason'] if 'reason' in sell else 'UNKNOWN'
            })
    
    # Tạo DataFrame từ danh sách các cặp
    pairs_df = pd.DataFrame(pairs)
    
    if not pairs_df.empty:
        # Thêm thống kê
        win_trades = len(pairs_df[pairs_df['profit_pct'] > 0])
        loss_trades = len(pairs_df[pairs_df['profit_pct'] <= 0])
        win_rate = (win_trades / len(pairs_df)) * 100 if len(pairs_df) > 0 else 0
        avg_profit = pairs_df['profit_pct'].mean() if len(pairs_df) > 0 else 0
        avg_win = pairs_df[pairs_df['profit_pct'] > 0]['profit_pct'].mean() if win_trades > 0 else 0
        avg_loss = pairs_df[pairs_df['profit_pct'] <= 0]['profit_pct'].mean() if loss_trades > 0 else 0
        max_win = pairs_df['profit_pct'].max() if len(pairs_df) > 0 else 0
        max_loss = pairs_df['profit_pct'].min() if len(pairs_df) > 0 else 0
        avg_hold_days = pairs_df['hold_days'].mean() if len(pairs_df) > 0 else 0
        profit = final_balance - initial_balance
        profit_percent = (profit / initial_balance) * 100
        
        # Profit factor
        total_gains = pairs_df[pairs_df['profit_pct'] > 0]['profit_pct'].sum() if win_trades > 0 else 0
        total_losses = abs(pairs_df[pairs_df['profit_pct'] <= 0]['profit_pct'].sum()) if loss_trades > 0 else 0
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        # Phân tích lý do thoát lệnh
        exit_reasons = pairs_df['exit_reason'].value_counts()
        
        # In thống kê
        print("\n===== THỐNG KÊ BACKTEST =====")
        print(f"Tổng giao dịch: {len(pairs_df)}")
        print(f"Giao dịch thắng: {win_trades} ({win_rate:.2f}%)")
        print(f"Giao dịch thua: {loss_trades} ({100-win_rate:.2f}%)")
        print(f"Lợi nhuận trung bình mỗi giao dịch: {avg_profit:.2f}%")
        print(f"Lợi nhuận trung bình giao dịch thắng: {avg_win:.2f}%")
        print(f"Thua lỗ trung bình giao dịch thua: {avg_loss:.2f}%")
        print(f"Lợi nhuận lớn nhất: {max_win:.2f}%")
        print(f"Thua lỗ lớn nhất: {max_loss:.2f}%")
        print(f"Thời gian giữ lệnh trung bình: {avg_hold_days:.2f} ngày")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Lợi nhuận ròng: {profit:.2f} ({profit_percent:.2f}%)")
        print("\nThống kê lý do thoát lệnh:")
        for reason, count in exit_reasons.items():
            print(f"- {reason}: {count} giao dịch ({count/len(pairs_df)*100:.1f}%)")
        
        return pairs_df
    
    return None

def plot_results(df, trades_df, initial_balance=1000):
    """Vẽ biểu đồ kết quả backtest"""
    if trades_df.empty:
        return
        
    plt.figure(figsize=(15, 10))
    
    # Vẽ giá và các điểm mua/bán
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='Giá đóng cửa')
    
    buy_signals = trades_df[trades_df['type'] == 'BUY']
    sell_signals = trades_df[trades_df['type'] == 'SELL']
    
    plt.scatter(buy_signals['date'], buy_signals['price'], marker='^', color='green', s=100, label='Mua')
    plt.scatter(sell_signals['date'], sell_signals['price'], marker='v', color='red', s=100, label='Bán')
    
    plt.title('Backtest Kết Quả')
    plt.ylabel('Giá')
    plt.legend()
    
    # Vẽ số dư tài khoản
    plt.subplot(2, 1, 2)
    balance_history = [initial_balance]
    for i, trade in trades_df.iterrows():
        balance_history.append(trade['balance'])
    
    plt.plot(range(len(balance_history)), balance_history, label='Số dư')
    plt.axhline(y=initial_balance, color='r', linestyle='--', label='Số dư ban đầu')
    plt.xlabel('Số giao dịch')
    plt.ylabel('Số dư')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'temp/backtest_results_{symbol}_{start_date}_{end_date}.png')
    plt.show()

def get_higher_timeframe_trend(symbol, interval, start_date, end_date):
    """Lấy xu hướng từ khung thời gian cao hơn"""
    higher_intervals = {
        Client.KLINE_INTERVAL_1MINUTE: Client.KLINE_INTERVAL_15MINUTE,
        Client.KLINE_INTERVAL_5MINUTE: Client.KLINE_INTERVAL_1HOUR,
        Client.KLINE_INTERVAL_15MINUTE: Client.KLINE_INTERVAL_4HOUR,
        Client.KLINE_INTERVAL_1HOUR: Client.KLINE_INTERVAL_4HOUR,
        Client.KLINE_INTERVAL_4HOUR: Client.KLINE_INTERVAL_1DAY
    }
    
    higher_tf = higher_intervals.get(interval)
    if higher_tf:
        try:
            # Mở rộng phạm vi ngày để có đủ dữ liệu - thêm nhiều ngày hơn
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=200)  # Thay đổi từ 30 lên 200 ngày
            start_str = start_dt.strftime("%Y-%m-%d")
            
            higher_df = get_historical_data(symbol, higher_tf, start_str, end_date)
            if not higher_df.empty:
                print(f"Đã tải {len(higher_df)} nến cho khung thời gian cao hơn ({higher_tf})")
                higher_df = add_indicators(higher_df)
                return higher_df
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu khung thời gian cao hơn: {e}")
    return None

def test_strategy(symbol, start_date, end_date, interval):
    try:
        print(f"Lấy dữ liệu lịch sử {symbol} từ {start_date} đến {end_date}...")
        df = get_historical_data(symbol, interval, start_date, end_date)
        
        if df.empty:
            print("DỪNG: Không có dữ liệu để backtest!")
            return
            
        # Lấy dữ liệu khung thời gian cao hơn
        higher_tf_df = get_higher_timeframe_trend(symbol, interval, start_date, end_date)
        print(f"Đã tải dữ liệu khung thời gian cao hơn: {'Thành công' if higher_tf_df is not None else 'Thất bại'}")
        
        # 2. Thêm chỉ báo
        print("Thêm các chỉ báo...")
        df = add_indicators(df)
        print(f"Đã thêm chỉ báo, DataFrame kích thước {df.shape}")
        
        # Thêm thông tin từ khung thời gian cao hơn (nếu có)
        if higher_tf_df is not None:
            print("Thêm thông tin từ khung thời gian cao hơn...")
            # Có thể thêm logic hỗ trợ đa khung thời gian ở đây
            
        # In ra một vài dòng dữ liệu để kiểm tra
        print("\nMẫu dữ liệu:")
        print(df[['close', 'ema20', 'ema50', 'ema200', 'rsi', 'atr', 'market_trend']].tail())
        
        # 3. Chạy backtest
        print("\nChạy backtest...")
        initial_balance = 10  # USD
        risk_per_trade = 0.5  # Rủi ro 0.5% số dư mỗi giao dịch
        
        trades_df, final_balance, profit, profit_percent = backtest_strategy(
            df, initial_balance, risk_per_trade
        )
        
        print(f"Backtest hoàn tất. Tìm thấy {len(trades_df)} giao dịch.")
        
        # 4. Phân tích kết quả
        print("Phân tích kết quả...")
        pairs_df = analyze_results(df, trades_df, initial_balance, final_balance)
        
        if trades_df.empty:
            print("\nKhông tìm thấy giao dịch nào trong khoảng thời gian đã chọn.")
            print("Thử các giải pháp sau:")
            print("1. Tăng thời gian backtest")
            print("2. Điều chỉnh các điều kiện giao dịch cho dễ thỏa mãn hơn")
            print("3. Kiểm tra lại dữ liệu và chỉ báo")
        
        # 5. Lưu kết quả
        if not trades_df.empty:
            trades_df.to_csv(f'temp/backtest_trades_{symbol}_{start_date}_{end_date}.csv')
            if pairs_df is not None:
                pairs_df.to_csv(f'temp/backtest_pairs_{symbol}_{start_date}_{end_date}.csv')
            # plot_results(df, trades_df, initial_balance)
    
    except Exception as e:
        print(f"\nLỖI khi chạy backtest: {e}")
        import traceback
        traceback.print_exc()

# Đổi hàm main để bắt lỗi tốt hơn
if __name__ == "__main__":
    # symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "BCHUSDT", ]
    symbols = ["PNUTUSDT"]
    for symbol in symbols:
        # Thay đổi sang các ngày trong quá khứ
        start_date = "2025-03-01"
        end_date = "2025-03-02"
        interval = Client.KLINE_INTERVAL_1MINUTE  # Thay đổi khung thời gian
        test_strategy(symbol, start_date, end_date, interval)
