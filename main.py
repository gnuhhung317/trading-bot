import time
from binance.client import Client
import pandas as pd
import pandas_ta as ta
import logging
import smtplib

# Kết nối API
api_key = 'Q2WA7BWZniogxmQP94wgxQ7yUUCGhciotpcGlPLzxoATA5OLcwmhmPG9jAxzDqPK'
api_secret = 'MBtMAD2NV2lDzSYTjNRXCKFdL1eLzPZdGkA0cjOWXsvzJhPDWAjNm1k12LLmErlA'
client = Client(api_key, api_secret, testnet=True)

# Cấu hình log
logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

# Hàm gửi email
def send_email(subject, message):
    smtp_server = 'smtp.gmail.com'
    # smtp_port = 587
    # sender_email = 'your_email@gmail.com'
    # receiver_email = 'receiver_email@gmail.com'
    # password = 'your_password'
    # with smtplib.SMTP(smtp_server, smtp_port) as server:
    #     server.starttls()
    #     server.login(sender_email, password)
    #     server.sendmail(sender_email, receiver_email, f'Subject: {subject}\n\n{message}')

# Vòng lặp chính
in_position = False
while True:
    try:
        # Lấy dữ liệu
        klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        
        # Tính chỉ báo
        df.ta.ema(length=20, append=True, col_names=('ema20',))
        df.ta.ema(length=50, append=True, col_names=('ema50',))
        df.ta.rsi(length=14, append=True, col_names=('rsi',))
        df.ta.atr(length=14, append=True, col_names=('atr',))
        
        latest = df.iloc[-1]
        
        if not in_position:
            # Điều kiện mua
            if latest['close'] > latest['ema20'] and latest['ema20'] > latest['ema50'] and latest['rsi'] < 60:
                balance = client.get_asset_balance(asset='USDT')
                total_usdt = float(balance['free'])
                trade_usdt = total_usdt * 0.05
                price = latest['close']
                quantity = round(trade_usdt / price, 5)  # Round to 5 decimal places for BTC
                order = client.order_market_buy(symbol='BTCUSDT', quantity=quantity)
                stop_loss = price - (1.5 * latest['atr'])
                highest_price = price
                in_position = True
                logging.info(f"Mua {quantity} BTC tại {price}, Stop-loss: {stop_loss}")
                send_email("Mua BTC", f"Mua {quantity} BTC tại {price}")
        
        elif in_position:
            # Theo dõi trailing stop
            latest_price = float(client.get_symbol_ticker(symbol='BTCUSDT')['price'])
            if latest_price > highest_price:
                highest_price = latest_price
            trailing_stop = highest_price * 0.97  # 3% dưới mức cao nhất
            
            # Điều kiện bán
            if latest_price <= trailing_stop or latest_price <= stop_loss or latest['rsi'] > 70:
                order = client.order_market_sell(symbol='BTCUSDT', quantity=quantity)
                in_position = False
                logging.info(f"Bán {quantity} BTC tại {latest_price}")
                send_email("Bán BTC", f"Bán {quantity} BTC tại {latest_price}")
                time.sleep(1800)  # Chờ 30 phút
        
        time.sleep(900)  # Chờ 15 phút
        
    except Exception as e:
        logging.error(f"Lỗi: {e}")
        time.sleep(60)