    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            logging.info(f"Gửi tin nhắn Telegram thành công: {message[:50]}...")
            return response.json()
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            logging.error(f"Lỗi gửi tin nhắn Telegram (lần {attempt+1}): {e} - {line_number}")
            time.sleep(2)
    logging.critical("Telegram không hoạt động sau 3 lần thử!")