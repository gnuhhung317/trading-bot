
def check_entry_conditions(df, higher_tf_df, symbol):
    """Kiểm tra điều kiện tín hiệu LONG hoặc SHORT"""
    if df.empty or higher_tf_df.empty:
        logging.warning(f"{symbol} - Dữ liệu rỗng, bỏ qua kiểm tra")
        return None
    
    current = df.iloc[-1]
    higher_current = higher_tf_df.iloc[-1]
    
    # Điều kiện LONG
    long_primary = [
        current['ema9'] > current['ema21'],
        current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']
    ]
    long_secondary = [
        current['rsi14'] < 70,
        current['volume_increase'],
        current['macd'] > 0,
        current['adx'] > 25
    ]
    long_condition = (all(long_primary) and all(long_secondary) and 
                      (higher_current['uptrend'] or 
                       (higher_current['adx'] > 20 and higher_current['di_plus'] > higher_current['di_minus'])))
    
    # Điều kiện SHORT
    short_primary = [
        current['ema9'] < current['ema21'],
        current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']
    ]
    short_secondary = [
        current['rsi14'] > 30,
        current['volume_increase'],
        current['macd'] < 0,
        current['adx'] > 25
    ]
    short_condition = (all(short_primary) and all(short_secondary) and 
                       (higher_current['downtrend'] or 
                        (higher_current['adx'] > 20 and higher_current['di_minus'] > higher_current['di_plus'])))
    
    signal = 'LONG' if long_condition else 'SHORT' if short_condition else None
    if signal:
        logging.info(f"{symbol}: Tín hiệu {signal}")
    return signal