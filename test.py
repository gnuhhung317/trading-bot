from binance.client import Client
import logging
# Logging
logging.basicConfig(
    filename='test.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
client = Client(api_key, api_secret)
ans = dict()
def get_symbol_precision(symbol):
    try:
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                return {
                    'quantity_precision': s['quantityPrecision'],
                    'price_precision': s['pricePrecision']
                }
        logging.warning(f"Không tìm thấy thông tin precision cho {symbol}")
        return {'quantity_precision': 0, 'price_precision': 0}
    except Exception as e:
        logging.error(f"Lỗi lấy precision cho {symbol}: {e}")
        return {'quantity_precision': 0, 'price_precision': 0}
symbols = [
    'AAVEUSDT', 
    'LINKUSDT', 
    'PEPEUSDT', 
    'VANAUSDT', 
    'TAOUSDT',	
    'TIAUSDT',	
    'RAYUSDT',	
    'SUIUSDT',	
    'MKRUSDT',	
    'LTCUSDT',	
    'ENAUSDT',	
    'NEARUSDT', 
    'RUNEUSDT', 
    'BNXUSDT',	
]

for symbol in symbols:
    precision_info = get_symbol_precision(symbol)
    ans[symbol] = dict()
    ans[symbol]["quantity_precision"] = precision_info["quantity_precision"]
    ans[symbol]["price_precision"] = precision_info["price_precision"]
logging.info(f"{symbols}")

for pair in symbols:
        symbol = pair
        try:
            # Lấy thông tin precision từ exchange info
            exchange_info = client.get_symbol_info(symbol)
            quantity_precision = int(1/float(exchange_info['filters'][1]['stepSize'])) #fix
            min_size = float(exchange_info['filters'][1]['minQty'])  # LOT_SIZE filter
            ans[symbol] ={"leverage": 20, "quantity_precision": quantity_precision, "min_size":min_size}
        except Exception as e:
             print(f"Exception {e} at {symbol}")
print(ans)
ans = {
    'AAVEUSDT': {'leverage': 20, 'quantity_precision': 1, 'min_size': 0.1},
    'LINKUSDT': {'leverage': 20, 'quantity_precision': 2, 'min_size': 0.01},
    # 'PEPEUSDT': {'leverage': 20, 'quantity_precision': 1, 'min_size': 1.0},
    'VANAUSDT': {'leverage': 20, 'quantity_precision': 2, 'min_size': 0.01},
    'TAOUSDT': {'leverage': 20, 'quantity_precision': 3, 'min_size': 0.001},
    'TIAUSDT': {'leverage': 20, 'quantity_precision': 0, 'min_size': 1},
    # 'SUIUSDT': {'leverage': 20, 'quantity_precision': 10, 'min_size': 0.1},
    'MKRUSDT': {'leverage': 20, 'quantity_precision': 3, 'min_size': 0.001},
    'LTCUSDT': {'leverage': 20, 'quantity_precision': 3, 'min_size': 0.001},
    'ENAUSDT': {'leverage': 20, 'quantity_precision': 0, 'min_size': 1},
    'NEARUSDT': {'leverage': 20, 'quantity_precision': 0, 'min_size': 1},
    'BNXUSDT': {'leverage': 6, 'quantity_precision': 1, 'min_size': 0.1}
}