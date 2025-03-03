# trade_manager.py
from config import client, SYMBOL, LEVERAGE, RISK_PER_TRADE
import logging
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

def set_leverage(symbol=SYMBOL, leverage=LEVERAGE):
    """Thiết lập đòn bẩy cho giao dịch"""
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logger.info(f"Đã thiết lập đòn bẩy {leverage}x cho {symbol}")
        return True
    except BinanceAPIException as e:
        logger.error(f"Lỗi đặt leverage: {str(e)}")
        # Kiểm tra xem có phải lỗi API key hay không
        if "API-key" in str(e) or "-2015" in str(e):
            logger.error("API key không hợp lệ hoặc không có quyền truy cập!")
            logger.error("Vui lòng kiểm tra lại API key hoặc tạo mới API key trên Binance!")
        return False
    except Exception as e:
        logger.error(f"Lỗi không xác định khi đặt leverage: {str(e)}")
        return False

def get_asset_precision(symbol=SYMBOL):
    """Lấy độ chính xác của một tài sản trên Binance"""
    try:
        exchange_info = client.get_exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                step_size = None
                for filter in symbol_info['filters']:
                    if filter['filterType'] == 'LOT_SIZE':
                        step_size = float(filter['stepSize'])
                        break
                
                if step_size:
                    precision = 0
                    step_size_str = str(step_size)
                    if '.' in step_size_str:
                        precision = len(step_size_str.split('.')[1].rstrip('0'))
                    return precision
        
        # Nếu không tìm thấy, trả về giá trị mặc định an toàn
        logger.warning(f"Không tìm thấy thông tin độ chính xác cho {symbol}, sử dụng giá trị mặc định")
        return 3
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin độ chính xác: {str(e)}")
        return 3  # Giá trị mặc định an toàn

def round_to_precision(value, precision):
    """Làm tròn số đến độ chính xác nhất định"""
    return round(float(value), precision)

def place_order(signal, size, price, stop_loss):
    """Đặt lệnh giao dịch"""
    try:
        # Lấy độ chính xác của tài sản
        precision = get_asset_precision(SYMBOL)
        
        # Làm tròn kích thước lệnh theo đúng yêu cầu của Binance
        rounded_size = round_to_precision(size, precision)
        
        # Kiểm tra API kết nối trước khi đặt lệnh
        if float(rounded_size) <= 0:
            logger.error("Kích thước lệnh không hợp lệ!")
            return None
            
        # Logic đặt lệnh ở đây
        logger.info(f"Đặt lệnh {signal} với giá {price}, stop loss {stop_loss}, kích thước: {rounded_size}")
        
        # Trong SANDBOX_MODE, chỉ mô phỏng
        if SANDBOX_MODE:
            # Giả lập đặt lệnh thành công
            return {
                "orderId": "simulated_order_id",
                "price": price,
                "status": "NEW",
                "executedQty": str(rounded_size)
            }
        else:
            # Đặt lệnh thật trên Binance
            side = "BUY" if signal == "LONG" else "SELL"
            order = client.futures_create_order(
                symbol=SYMBOL,
                side=side,
                type="MARKET",
                quantity=rounded_size
            )
            
            # Đặt stop loss
            sl_side = "SELL" if signal == "LONG" else "BUY"
            client.futures_create_order(
                symbol=SYMBOL,
                side=sl_side,
                type="STOP_MARKET",
                stopPrice=stop_loss,
                quantity=rounded_size,
                reduceOnly=True
            )
            
            return order
            
    except BinanceAPIException as e:
        logger.error(f"Lỗi đặt lệnh: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Lỗi không xác định khi đặt lệnh: {str(e)}")
        return None

def close_position(position, current_price, partial_ratio=None):
    try:
        side = "SELL" if position['type'] == "LONG" else "BUY"
        
        # Lấy độ chính xác của tài sản
        precision = get_asset_precision(SYMBOL)
        
        # Tính toán size và làm tròn theo độ chính xác
        size = position['size'] * partial_ratio if partial_ratio else position['size']
        rounded_size = round_to_precision(size, precision)
        
        if SANDBOX_MODE:
            # Giả lập đóng vị thế trong chế độ sandbox
            profit = (current_price - position['entry_price']) * rounded_size * (1 if position['type'] == 'LONG' else -1)
            logger.info(f"[SANDBOX] Đóng lệnh {position['type']} - Size: {rounded_size}, Profit: {profit}")
            return profit
        else:
            # Đóng vị thế thật
            order = client.futures_create_order(
                symbol=SYMBOL,
                side=side,
                type="MARKET",
                quantity=rounded_size
            )
            
            profit = (current_price - position['entry_price']) * rounded_size * (1 if position['type'] == 'LONG' else -1)
            logger.info(f"Đóng lệnh {position['type']} - Size: {rounded_size}, Profit: {profit}")
            return profit
            
    except BinanceAPIException as e:
        logger.error(f"Lỗi đóng lệnh: {str(e)}")
        if "Precision" in str(e):
            logger.error(f"Lỗi độ chính xác! Cần làm tròn kích thước lệnh. Size gốc: {size}")
        return 0
    except Exception as e:
        logger.error(f"Lỗi không xác định khi đóng lệnh: {str(e)}")
        return 0

def manage_position(position, current_price, df):
    if not position:
        return None
    
    current = df.iloc[-1]
    profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
    r_multiple = profit / position['risk_per_r'] if position['risk_per_r'] != 0 else 0
    
    # Break-even
    if r_multiple > 0.7 and not position['breakeven_activated']:
        position['stop_loss'] = position['entry_price']
        position['breakeven_activated'] = True
        logging.info(f"Dời SL về entry: {position['stop_loss']}")
    
    # Trailing stop
    if r_multiple > 1:
        trail_factor = min(1.5, 1 + r_multiple * 0.1)
        new_stop = current_price - current['atr14'] * trail_factor if position['type'] == 'LONG' else current_price + current['atr14'] * trail_factor
        position['stop_loss'] = max(position['stop_loss'], new_stop) if position['type'] == 'LONG' else min(position['stop_loss'], new_stop)
        # Cập nhật SL trên sàn
        sl_side = "SELL" if position['type'] == "LONG" else "BUY"
        client.futures_create_order(
            symbol=SYMBOL,
            side=sl_side,
            type="STOP_MARKET",
            stopPrice=position['stop_loss'],
            quantity=position['size'],
            timeInForce="GTC",
            reduceOnly=True
        )
        logging.info(f"Trailing stop: {position['stop_loss']}")
    
    # Partial exit
    if r_multiple >= 1.5 and not position['first_target_hit']:
        profit = close_position(position, current_price, partial_ratio=0.3)
        position['size'] *= 0.7
        position['first_target_hit'] = True
        logging.info(f"Chốt 30% tại 1.5R - Profit: {profit}")
    elif r_multiple >= 2.5 and position['first_target_hit'] and not position['second_target_hit']:
        profit = close_position(position, current_price, partial_ratio=0.5)
        position['size'] *= 0.5
        position['second_target_hit'] = True
        logging.info(f"Chốt 50% tại 2.5R - Profit: {profit}")
    
    return position

def get_balance():
    """Lấy số dư tài khoản"""
    try:
        balance = client.futures_account_balance()
        usdt_balance = next((item['balance'] for item in balance if item['asset'] == 'USDT'), 0)
        logger.info(f"Số dư hiện tại: {usdt_balance} USDT")
        return float(usdt_balance)
    except BinanceAPIException as e:
        logger.error(f"Lỗi lấy balance: {str(e)}")
        # Kiểm tra cụ thể lỗi API
        if "API-key" in str(e) or "-2015" in str(e):
            logger.error("API key không hợp lệ! Vui lòng kiểm tra API key của bạn.")
            # Trả về giá trị mặc định thay vì gây crash
            return 0
        return 0
    except Exception as e:
        logger.error(f"Lỗi không xác định khi lấy balance: {str(e)}")
        return 0