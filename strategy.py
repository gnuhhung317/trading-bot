from indicators import add_signal_indicators, add_trend_indicators

def check_entry_conditions(df, higher_tf_df, balance):
    df = add_signal_indicators(df)
    higher_tf_df = add_trend_indicators(higher_tf_df)
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    higher_current = higher_tf_df.iloc[-1]
    
    # LONG conditions
    long_primary = [
        current['ema9'] > current['ema21'],
        current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']
    ]
    long_secondary = [
        current['rsi14'] < 70,
        current['volume_increase'],
        current['macd'] > 0
    ]
    long_condition = (
        all(long_primary) and 
        any(long_secondary) and
        (higher_current['uptrend'] or (higher_current['adx'] > 25 and higher_current['di_plus'] > higher_current['di_minus']))
    )
    
    # SHORT conditions (nếu mày muốn, tao thêm tạm, mày chỉnh lại nếu cần)
    short_primary = [
        current['ema9'] < current['ema21'],
        current['ema_cross_down'] or current['macd_cross_down']
    ]
    short_secondary = [
        current['rsi14'] > 30,
        current['volume_increase'],
        current['macd'] < 0
    ]
    short_condition = (
        all(short_primary) and 
        any(short_secondary) and
        (higher_current['downtrend'] or (higher_current['adx'] > 25 and higher_current['di_minus'] > higher_current['di_plus']))
    )
    
    if long_condition and balance > 0:
        return "LONG"
    elif short_condition and balance > 0:
        return "SHORT"
    return None

def check_exit_conditions(position, current_price, df):
    current = df.iloc[-1]
    profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
    r_multiple = profit / position['risk_per_r'] if position['risk_per_r'] != 0 else 0
    
    if position['type'] == "LONG":
        exit_conditions = [
            current_price <= position['stop_loss'],
            current['ema_cross_down'],
            current['macd_cross_down'],
            current['rsi14'] > 80,
            r_multiple >= 4
        ]
        return any(exit_conditions)
    elif position['type'] == "SHORT":
        exit_conditions = [
            current_price >= position['stop_loss'],
            current['ema_cross_up'],
            current['macd_cross_up'],
            current['rsi14'] < 20,
            r_multiple >= 4
        ]
        return any(exit_conditions)
    return False