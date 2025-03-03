# config.py
from binance.client import Client

# KHÔNG BAO GIỜ ĐƯUA API KEY VÀO MÃ NGUỒN MỞ - SỬ DỤNG BIẾN MÔI TRƯỜNG
import os 
from dotenv import load_dotenv

# Load biến môi trường từ file .env (nếu có)
load_dotenv()

# Lấy API key từ biến môi trường, hoặc sử dụng giá trị mặc định cho testnet
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Flag để xác định có sử dụng testnet hay không
USE_TESTNET = False

# Khởi tạo client
client = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)
# Thông số bot
SYMBOL = "ETHUSDT"
TIMEFRAME = "15m"
HIGHER_TIMEFRAME = "4h"
LEVERAGE = 5
RISK_PER_TRADE = 0.02
INITIAL_BALANCE = 10
LOG_FILE = "logs/trading_log.txt"

# Danh sách symbols để giao dịch
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

# Mapping timeframe cho python-binance
TIMEFRAME_MAP = {
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY
}

# Sandbox mode để kiểm thử
SANDBOX_MODE = True

# Cache cho thông tin asset precision để tránh gọi API quá nhiều
ASSET_PRECISION_CACHE = {}