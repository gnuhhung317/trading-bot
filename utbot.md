# UT Trading Bot - Tài liệu sử dụng

## Giới thiệu

UT Trading Bot là bot giao dịch tự động cho thị trường futures trên Binance, sử dụng chiến lược ATR Trailing Stop kết hợp với SMA. Bot được thiết kế để giao dịch nhiều cặp tiền tệ cùng lúc với quản lý rủi ro chặt chẽ, trailing stop thông minh và cập nhật thông tin qua Telegram.

## Các tính năng chính

- **Đa cặp giao dịch**: Hỗ trợ giao dịch nhiều cặp tiền tệ cùng lúc
- **Quản lý rủi ro**: Kiểm soát tỷ lệ rủi ro trên mỗi giao dịch
- **Trailing Stop thông minh**: Tự động điều chỉnh mức trailing stop theo lợi nhuận hiện tại
- **Breakeven tự động**: Tự động đưa stop loss về giá vào khi lợi nhuận đạt ngưỡng
- **Cảnh báo Telegram**: Thông báo chi tiết về vào lệnh, thoát lệnh và báo cáo định kỳ
- **Ghi log đầy đủ**: Lưu lại mọi hoạt động để theo dõi và phân tích

## Cài đặt

### Yêu cầu

- Python 3.7 trở lên
- Tài khoản Binance Futures
- Tài khoản Telegram (để nhận thông báo)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Tạo file .env

Tạo file `.env` trong thư mục chứa mã nguồn với nội dung:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Cấu hình

Các tham số chính cần điều chỉnh trong file `main_ut.py`:

### Cấu hình cặp giao dịch

```python
COINS = {
    "ETHUSDT": {"leverage": 10, "quantity_precision": 3, "min_size": 0.001}
    # "BTCUSDT": {"leverage": 20, "quantity_precision": 1, "min_size": 0.1}
}
```

- `leverage`: Đòn bẩy sử dụng
- `quantity_precision`: Độ chính xác của khối lượng giao dịch
- `min_size`: Kích thước giao dịch tối thiểu

### Tham số chiến lược

```python
TIMEFRAME = "5m"
ATR_MULTIPLIER = 1.0
ATR_PERIOD = 10
SMA_PERIOD = 200
RISK_PER_TRADE = 0.1  # 1%
MAX_POSITIONS = 5
STOP_LOSS_THRESHOLD = 0.1  # 10%
TAKER_FEE = 0.0004  # 0.04%
```

- `TIMEFRAME`: Khung thời gian phân tích
- `ATR_MULTIPLIER`: Hệ số nhân ATR
- `ATR_PERIOD`: Chu kỳ tính ATR
- `SMA_PERIOD`: Chu kỳ tính SMA
- `RISK_PER_TRADE`: Tỷ lệ rủi ro trên mỗi giao dịch (% tài khoản)
- `MAX_POSITIONS`: Số vị thế tối đa cùng lúc
- `STOP_LOSS_THRESHOLD`: Ngưỡng dừng bot khi số dư giảm xuống
- `TAKER_FEE`: Phí giao dịch taker

## Chiến lược giao dịch

### Tín hiệu vào lệnh

- **LONG**: Khi giá vượt trên ATR Trailing Stop và nằm trên SMA200
- **SHORT**: Khi giá giảm dưới ATR Trailing Stop và nằm dưới SMA200

### Quản lý trailing stop thông minh

UT Bot áp dụng 3 cấp độ trailing stop:

1. **Trailing stop cơ bản**: Điểm dừng cách giá một khoảng 1 ATR
2. **Breakeven stop**: Khi lợi nhuận đạt 1%, điểm dừng được đưa về giá vào
3. **Trailing stop thắt chặt**: Khi lợi nhuận đạt 2%, điểm dừng cách giá chỉ 0.5 ATR

### Thoát lệnh

Bot sẽ thoát lệnh trong các trường hợp:
- Kích hoạt trailing stop
- Đạt mục tiêu lợi nhuận (take profit) 
- Kích hoạt hard stop loss (2% từ giá vào)
- Xuất hiện tín hiệu đảo chiều

## Phân tích chi tiết chiến lược ATR Trailing Stop

### Nguyên lý hoạt động

UT Bot sử dụng chiến lược trend-following dựa trên Chỉ báo ATR (Average True Range) và SMA (Simple Moving Average). Chiến lược này bao gồm các thành phần chính:

1. **Chỉ báo ATR** - Đo lường biến động giá:
   - ATR được tính dựa trên chu kỳ `ATR_PERIOD` = 10 (mặc định)
   - ATR giúp xác định biên độ di chuyển bình thường của thị trường
   - Nhân ATR với `ATR_MULTIPLIER` = 1.0 để có khoảng cách lý tưởng cho trailing stop

2. **Đường xATRTrailingStop** - Đường trailing stop thông minh:
   - Đường này di chuyển theo giá và chỉ đi lên khi giá tăng (trong xu hướng tăng)
   - Chỉ đi xuống khi giá giảm (trong xu hướng giảm)
   - Khoảng cách giữa giá và đường này là 1 ATR (có thể điều chỉnh)

3. **SMA200** - Bộ lọc xu hướng:
   - Đóng vai trò bộ lọc chính để xác định xu hướng
   - Chỉ mở vị thế LONG khi giá trên SMA200
   - Chỉ mở vị thế SHORT khi giá dưới SMA200

### Thuật toán xATRTrailingStop

Việc tính toán xATRTrailingStop là cốt lõi của chiến lược:

```python
# Simplified algorithm
if price > previous_stop and previous_price > previous_stop:
    # Trong xu hướng tăng, trailing stop chỉ đi lên, không bao giờ đi xuống
    current_stop = max(previous_stop, price - ATR * multiplier)
elif price < previous_stop and previous_price < previous_stop:
    # Trong xu hướng giảm, trailing stop chỉ đi xuống, không bao giờ đi lên
    current_stop = min(previous_stop, price + ATR * multiplier)
else:
    # Khi có sự đảo chiều xu hướng, thiết lập lại trailing stop
    current_stop = price - ATR * multiplier if price > previous_stop else price + ATR * multiplier
```

Thuật toán này cho phép đường trailing stop:
- Di chuyển theo hướng xu hướng
- Không bao giờ đảo chiều khi xu hướng vẫn tiếp tục
- Chuyển đổi khi xu hướng đảo chiều

### Phân tích hiệu suất chiến lược

#### Ưu điểm:

1. **Tối ưu xu hướng**: Chiến lược bắt được những xu hướng mạnh và dài
2. **Quản lý rủi ro động**: ATR tự động điều chỉnh theo biến động thị trường
3. **Bảo vệ lợi nhuận**: Trailing stop thắt chặt dần khi lợi nhuận tăng
4. **Đa dạng thị trường**: Hoạt động tốt trên nhiều cặp giao dịch và khung thời gian
5. **Tự động hoàn toàn**: Không cần can thiệp thủ công

#### Nhược điểm:

1. **Whipsaw**: Có thể bị cắt lỗ nhiều lần trong thị trường sideway (đi ngang)
2. **Độ trễ**: Có độ trễ trong việc nhận diện xu hướng mới
3. **Hiệu quả không đồng đều**: Hoạt động tốt trong thị trường trending, kém hiệu quả trong thị trường sideway

### So sánh với các chiến lược khác

| Chiến lược | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| ATR Trailing Stop | Tự động điều chỉnh theo biến động thị trường | Có thể bị whipsaw trong thị trường sideway |
| Bollinger Bands | Tốt trong thị trường sideway | Kém hiệu quả trong xu hướng mạnh |
| Parabolic SAR | Đơn giản, dễ sử dụng | Nhiều tín hiệu giả trong thị trường biến động |
| Fixed Percentage | Đơn giản, rõ ràng | Không điều chỉnh theo biến động thị trường |

### Tối ưu hóa chiến lược

Để tối ưu hóa hiệu suất chiến lược, có thể điều chỉnh các tham số:

1. **ATR_PERIOD**:
   - Giảm (5-7): Nhạy hơn, phù hợp thị trường biến động nhiều
   - Tăng (14-21): Ổn định hơn, ít bị whipsaw

2. **ATR_MULTIPLIER**:
   - Giảm (0.5-0.8): Trailing stop sát hơn, bảo vệ lợi nhuận tốt hơn nhưng dễ bị dừng lỗ sớm
   - Tăng (1.5-2.0): Trailing stop rộng hơn, giảm whipsaw nhưng rủi ro cao hơn

3. **SMA_PERIOD**:
   - Giảm (50-100): Nhạy với xu hướng trung hạn
   - Tăng (200-300): Bắt xu hướng dài hạn, ổn định hơn

### Phù hợp với loại thị trường

| Loại thị trường | Hiệu quả | Điều chỉnh đề xuất |
|-----------------|----------|-------------------|
| Xu hướng mạnh | Rất tốt | Tăng ATR_MULTIPLIER để giữ vị thế lâu hơn |
| Dao động nhẹ | Tốt | Giữ nguyên tham số |
| Sideway | Kém | Tạm dừng giao dịch hoặc giảm kích thước vị thế |
| Biến động cao | Trung bình | Tăng ATR_PERIOD để giảm nhạy cảm |

## Sử dụng

Để khởi động bot:

```bash
python main_ut.py
```

Bot sẽ tự động:
1. Xác thực với Binance API
2. Thiết lập đòn bẩy cho các cặp giao dịch
3. Gửi thông báo khởi động qua Telegram
4. Bắt đầu chu trình giao dịch

## Thông báo Telegram

Bot gửi các thông báo sau qua Telegram:
- Thông báo khởi động bot
- Thông báo vào lệnh (Entry)
- Thông báo thoát lệnh (Exit) kèm lý do
- Báo cáo số dư định kỳ (mỗi 5 phút)
- Báo cáo tổng kết hàng giờ
- Thông báo lỗi và cảnh báo

## Quản lý rủi ro

1. **Rủi ro mỗi giao dịch**: Mỗi giao dịch rủi ro tối đa là RISK_PER_TRADE% tài khoản
2. **Hard stop loss**: Luôn có mức dừng lỗ cố định 2% từ giá vào
3. **Đa cấp trailing stop**: Đảm bảo bảo vệ lợi nhuận tốt nhất
4. **Dừng bot khi thua lỗ lớn**: Bot tự dừng khi số dư giảm quá STOP_LOSS_THRESHOLD%

## Phân tích hiệu suất

Bot ghi nhận chi tiết mỗi giao dịch vào mảng `trades`:
- Loại giao dịch (LONG/SHORT)
- Thời gian vào/ra lệnh
- Giá vào/ra
- Lợi nhuận
- Lý do thoát lệnh

Các log được lưu vào file `utbot_trading.log` để phân tích sau này.

## Xử lý lỗi và bảo trì

- Bot tự động thử lại khi gặp lỗi kết nối
- Telegram gửi thông báo lỗi để người dùng có thể xử lý kịp thời
- Cơ chế đồng bộ số dư và vị thế đảm bảo tính nhất quán

## Tùy chỉnh và mở rộng

Để thêm cặp giao dịch mới, bổ sung vào biến COINS:

```python
COINS = {
    "ETHUSDT": {"leverage": 10, "quantity_precision": 3, "min_size": 0.001},
    "BTCUSDT": {"leverage": 20, "quantity_precision": 1, "min_size": 0.1},
    "SOLUSDT": {"leverage": 10, "quantity_precision": 2, "min_size": 0.01}
}
```

Để thay đổi chiến lược, có thể tùy chỉnh hàm `calculate_indicators()` và `check_entry_conditions()`. 