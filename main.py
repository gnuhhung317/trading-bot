from data import get_latest_data, update_data
from strategy import check_entry_conditions, check_exit_conditions
from trade_manager import get_asset_precision, round_to_precision, place_order, manage_position, get_balance, close_position, set_leverage
from config import INITIAL_BALANCE, RISK_PER_TRADE, LEVERAGE, SANDBOX_MODE, SYMBOL
import time
import logging
import os

# Tạo thư mục logs nếu chưa tồn tại
if not os.path.exists("logs"):
    os.makedirs("logs")

# Cài đặt logging
logging.basicConfig(
    filename="logs/trading_log.txt", 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def run_bot():
    """Hàm chính chạy bot giao dịch"""
    
    # Kiểm tra API key và thiết lập đòn bẩy
    if not set_leverage():
        logging.error("Không thể thiết lập đòn bẩy - kiểm tra API key và quyền truy cập!")
        if not SANDBOX_MODE:
            logging.error("Bot dừng vì lỗi kết nối API trong chế độ LIVE!")
            return
        else:
            logging.warning("Tiếp tục trong chế độ SANDBOX với dữ liệu mô phỏng.")
    
    # Lấy dữ liệu ban đầu
    try:
        df, higher_tf_df = get_latest_data()
    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu ban đầu: {str(e)}")
        logging.error("Bot không thể khởi động do lỗi dữ liệu ban đầu!")
        return
    
    position = None
    balance = get_balance()
    
    # Nếu không lấy được balance thật, sử dụng giá trị mặc định
    if balance <= 0 and SANDBOX_MODE:
        balance = INITIAL_BALANCE
        logging.warning(f"Sử dụng balance mặc định: {INITIAL_BALANCE} trong chế độ SANDBOX")
    
    # Vòng lặp chính của bot
    while True:
        try:
            # Cập nhật dữ liệu
            df, higher_tf_df = update_data(df, higher_tf_df)
            current_price = df['close'].iloc[-1]
            
            # Lấy số dư mới nhất (hoặc sử dụng balance hiện tại trong chế độ SANDBOX)
            if not SANDBOX_MODE:
                balance = get_balance()
            
            # Kiểm tra tài khoản
            if balance <= 0:
                logging.error("Tài khoản cháy hoặc không có tiền!")
                if not SANDBOX_MODE:
                    break
                else:
                    # Trong chế độ sandbox, reset lại balance
                    balance = INITIAL_BALANCE
                    logging.warning(f"Đặt lại balance: {INITIAL_BALANCE} trong chế độ SANDBOX")
            
            # Quản lý vị thế hiện tại (nếu có)
            if position:
                position = manage_position(position, current_price, df)
                if check_exit_conditions(position, current_price, df):
                    profit = close_position(position, current_price)
                    logging.info(f"Thoát lệnh {position['type']} - Profit: {profit}")
                    balance += profit  # Cập nhật balance trong chế độ SANDBOX
                    position = None
            
            # Kiểm tra điều kiện vào lệnh mới
            if not position and balance > INITIAL_BALANCE * 0.1:
                signal = check_entry_conditions(df, higher_tf_df, balance)
                if signal:
                    atr = df['atr14'].iloc[-1]
                    if signal == "LONG":
                        recent_low = df['low'].iloc[-6:-1].min()
                        stop_loss = recent_low - atr * 0.3 if recent_low < current_price * 0.99 else current_price - atr * 1.5
                    else:
                        recent_high = df['high'].iloc[-6:-1].max()
                        stop_loss = recent_high + atr * 0.3 if recent_high > current_price * 1.01 else current_price + atr * 1.5
                    
                    risk_amount = balance * RISK_PER_TRADE
                    risk_per_r = abs(current_price - stop_loss)
                    size = risk_amount / risk_per_r * LEVERAGE
                    
                    # Lấy độ chính xác và làm tròn
                    precision = get_asset_precision(SYMBOL)
                    size = round_to_precision(size, precision)
                    
                    order = place_order(signal, size, current_price, stop_loss)
                    if order:
                        position = {
                            'type': signal,
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'size': size,
                            'risk_per_r': risk_amount,
                            'breakeven_activated': False,
                            'first_target_hit': False,
                            'second_target_hit': False
                        }
            
            # Ngủ một thời gian trước khi cập nhật tiếp
            time.sleep(60)
            
        except KeyboardInterrupt:
            logging.info("Bot đã dừng bởi người dùng.")
            break
        except Exception as e:
            logging.error(f"Lỗi trong loop: {str(e)}")
            time.sleep(60)  # Đợi 1 phút trước khi thử lại

if __name__ == "__main__":
    logging.info("Bot khởi động!")
    run_bot()