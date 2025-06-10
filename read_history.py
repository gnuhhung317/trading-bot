import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from binance.client import Client

# 1. Kết nối đến Binance Futures thông qua API
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(API_KEY, API_SECRET)

# 2. Xác định khoảng thời gian lấy dữ liệu
start_date = "2025-04-14 04:12:00"
end_date = "2025-04-15 04:18:00"
start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
end_ts = int(datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)

# 3. Lấy lịch sử lệnh trên Futures của cặp ETHUSDT
orders = client.futures_get_all_orders(symbol="ETHUSDT", startTime=start_ts, endTime=end_ts)
if not orders:
    print("Không có lệnh giao dịch nào trong khoảng thời gian này.")
    exit()

df_orders = pd.DataFrame(orders)
df_orders = df_orders[df_orders['status'] == 'FILLED']

# Kiểm tra dữ liệu lệnh
print("Dữ liệu lệnh thô:", df_orders[['updateTime', 'side', 'avgPrice', 'executedQty']].head())

# Chuyển đổi thời gian và ép kiểu dữ liệu
df_orders['updateTime'] = pd.to_datetime(df_orders['updateTime'], unit='ms')
df_orders['avgPrice'] = df_orders['avgPrice'].astype(float)
df_orders['executedQty'] = df_orders['executedQty'].astype(float)

# Thêm nhiễu nhỏ vào thời gian để tránh trùng lặp trên biểu đồ
jitter_seconds = 5
df_orders['updateTime_jitter'] = df_orders['updateTime'] + pd.to_timedelta(np.random.uniform(-jitter_seconds, jitter_seconds, size=len(df_orders)), unit='s')

# 4. Lấy dữ liệu giá lịch sử (candlestick) khung 1 phút
interval = Client.KLINE_INTERVAL_1MINUTE
klines_list = []
current_ts = start_ts
while current_ts < end_ts:
    klines = client.futures_klines(symbol="ETHUSDT", interval=interval, startTime=current_ts, limit=1000)
    if not klines:
        break
    klines_list.extend(klines)
    current_ts = klines[-1][0] + 60000  # Tăng 1 phút để lấy đoạn tiếp theo

if not klines_list:
    print("Không lấy được dữ liệu giá lịch sử từ Binance Futures.")
    exit()

cols = ['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades', 
        'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore']
df_klines = pd.DataFrame(klines_list, columns=cols)

df_klines['Open_Time'] = pd.to_datetime(df_klines['Open_Time'], unit='ms')
df_klines['Close_Time'] = pd.to_datetime(df_klines['Close_Time'], unit='ms')
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df_klines[col] = df_klines[col].astype(float)

# Lọc dữ liệu trong khoảng thời gian mong muốn
df_klines = df_klines[(df_klines['Open_Time'] >= pd.to_datetime(start_date)) & 
                      (df_klines['Open_Time'] <= pd.to_datetime(end_date))]

# Tính khoảng giá để điều chỉnh biểu đồ
daily_min = df_klines['Low'].min()
daily_max = df_klines['High'].max()
y_lower = daily_min * 0.99
y_upper = daily_max * 1.01

# 5. Vẽ biểu đồ giá và các điểm giao dịch
plt.figure(figsize=(14, 8))
plt.plot(df_klines['Open_Time'], df_klines['Close'], label="Giá ETHUSDT (Futures)", color='blue')

# Phân loại lệnh mua và bán
buys = df_orders[df_orders['side'] == 'BUY']
sells = df_orders[df_orders['side'] == 'SELL']

# Kiểm tra dữ liệu mua/bán
print("Số lệnh mua:", len(buys))
print("Số lệnh bán:", len(sells))
if len(buys) > 0:
    print("Dữ liệu lệnh mua:", buys[['updateTime_jitter', 'avgPrice']].head())
if len(sells) > 0:
    print("Dữ liệu lệnh bán:", sells[['updateTime_jitter', 'avgPrice']].head())

# Vẽ điểm mua và bán
if not buys.empty:
    plt.scatter(buys['updateTime_jitter'], buys['avgPrice'], marker="^", color='green', s=100, alpha=0.8, label="Mua")
if not sells.empty:
    plt.scatter(sells['updateTime_jitter'], sells['avgPrice'], marker="v", color='red', s=100, alpha=0.8, label="Bán")

# Thêm đường dọc tại thời điểm giao dịch
for t in buys['updateTime'].unique():
    plt.axvline(x=t, color='green', alpha=0.2)
for t in sells['updateTime'].unique():
    plt.axvline(x=t, color='red', alpha=0.2)

# Đặt giới hạn trục X để hiển thị đầy đủ khoảng thời gian
plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
# Đặt giới hạn trục Y
plt.ylim(y_lower, y_upper)
# Cấu hình trục X với các mốc thời gian mỗi 6 giờ
plt.xticks(pd.date_range(start=start_date, end=end_date, freq='6H'), rotation=45)

plt.title(f"Biểu đồ giá ETHUSDT Futures (Ngày 2025-04-13 đến 2025-04-15) với giao dịch mua/bán")
plt.xlabel("Thời gian")
plt.ylabel("Giá (USDT)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ethusdt_trading_chart_updated.png')

# 6. Tính toán insight
total_buys_qty = buys['executedQty'].sum()
total_sells_qty = sells['executedQty'].sum()

avg_buy_price = (buys['avgPrice'] * buys['executedQty']).sum() / total_buys_qty if total_buys_qty > 0 else 0
avg_sell_price = (sells['avgPrice'] * sells['executedQty']).sum() / total_sells_qty if total_sells_qty > 0 else 0

matched_qty = min(total_buys_qty, total_sells_qty)
profit = (avg_sell_price - avg_buy_price) * matched_qty
profit_pct = (profit / (avg_buy_price * matched_qty) * 100) if matched_qty > 0 and avg_buy_price > 0 else 0

# 7. Tạo báo cáo
report = f"""
BÁO CÁO GIAO DỊCH FUTURES ETHUSDT NGÀY 2025-04-13 ĐẾN 2025-04-15
---------------------------------------------------------
**Lệnh mua:**
  - Số lượng lệnh: {len(buys)}
  - Tổng số ETH mua: {total_buys_qty:.4f}
  - Giá trung bình mua: {avg_buy_price:.4f} USDT

**Lệnh bán:**
  - Số lượng lệnh: {len(sells)}
  - Tổng số ETH bán: {total_sells_qty:.4f}
  - Giá trung bình bán: {avg_sell_price:.4f} USDT

**Lợi nhuận (cho khối lượng khớp lệnh {matched_qty:.4f} ETH):**
  - Lợi nhuận thực: {profit:.4f} USDT
  - Tỷ lệ lợi nhuận: {profit_pct:.2f}%
---------------------------------------------------------
"""

print(report)