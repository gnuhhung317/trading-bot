 trades = []
    balance = initial_balance
    position = None
    entry_price = 0
    trailing_stop = 0
    max_risk_per_trade = initial_balance * 0.01  # Rủi ro tối đa 1% vốn

    for i in range(20, len(df)):
        current_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]

        # Phát hiện sóng nhỏ
        wave_forming = False
        wave_direction = None

        # Long khi breakout lên
        if (df['close'].iloc[i] > df['high_2'].iloc[i] and  # Breakout vượt mức cao 2 nến
            df['volume_surge'].iloc[i] and
            df['close'].iloc[i] > df['sma5'].iloc[i] and  # Giá trên SMA5
            df['rsi'].iloc[i] > 50):  # RSI xác nhận xu hướng tăng
            wave_forming = True
            wave_direction = 'UP'

        # Short khi breakout xuống
        elif (df['close'].iloc[i] < df['low_2'].iloc[i] and  # Breakout dưới mức thấp 2 nến
              df['volume_surge'].iloc[i] and
              df['close'].iloc[i] < df['sma5'].iloc[i] and  # Giá dưới SMA5
              df['rsi'].iloc[i] < 50):  # RSI xác nhận xu hướng giảm
            wave_forming = True
            wave_direction = 'DOWN'

        # Vào lệnh
        if position is None and wave_forming:
            if wave_direction == 'UP':
                position = 'LONG'
                entry_price = current_price
                stop_loss_distance = 0.75 * atr  # Stop loss 0.75 ATR
                trailing_stop = entry_price - stop_loss_distance
                risk_amount = stop_loss_distance
                position_size = min(max_risk_per_trade / risk_amount, initial_balance * 0.2) * leverage
                trades.append({
                    'type': position,
                    'entry_price': entry_price,
                    'trailing_stop': trailing_stop,
                    'entry_time': df.index[i],
                    'position_size': position_size
                })

            elif wave_direction == 'DOWN':
                position = 'SHORT'
                entry_price = current_price
                stop_loss_distance = 0.75 * atr
                trailing_stop = entry_price + stop_loss_distance
                risk_amount = stop_loss_distance
                position_size = min(max_risk_per_trade / risk_amount, initial_balance * 0.2) * leverage
                trades.append({
                    'type': position,
                    'entry_price': entry_price,
                    'trailing_stop': trailing_stop,
                    'entry_time': df.index[i],
                    'position_size': position_size
                })

        # Quản lý và thoát lệnh
        elif position == 'LONG':
            profit_ratio = (current_price - entry_price) / entry_price
            if profit_ratio > 0.003:  # 0.3% lợi nhuận
                trailing_stop = max(trailing_stop, current_price - 0.75 * atr)  # Trailing stop động
            else:
                trailing_stop = max(trailing_stop, entry_price - 0.75 * atr)

            if df['low'].iloc[i] <= trailing_stop:
                exit_price = trailing_stop
                profit = (exit_price - entry_price) / entry_price * trades[-1]['position_size']
                balance += profit
                trades[-1].update({
                    'exit_price': exit_price,
                    'exit_time': df.index[i],
                    'profit': profit,
                    'profit_pct': (exit_price - entry_price) / entry_price * 100 * leverage,
                    'exit_reason': 'trailing_stop',
                    'hold_time': (df.index[i] - trades[-1]['entry_time']).total_seconds() / 3600
                })
                position = None
            elif df['rsi'].iloc[i] < 40 or df['close'].iloc[i] < df['sma5'].iloc[i]:  # RSI đảo chiều hoặc giá dưới SMA5
                exit_price = current_price
                profit = (exit_price - entry_price) / entry_price * trades[-1]['position_size']
                balance += profit
                trades[-1].update({
                    'exit_price': exit_price,
                    'exit_time': df.index[i],
                    'profit': profit,
                    'profit_pct': (exit_price - entry_price) / entry_price * 100 * leverage,
                    'exit_reason': 'rsi_or_sma5',
                    'hold_time': (df.index[i] - trades[-1]['entry_time']).total_seconds() / 3600
                })
                position = None

        elif position == 'SHORT':
            profit_ratio = (entry_price - current_price) / entry_price
            if profit_ratio > 0.003:  # 0.3% lợi nhuận
                trailing_stop = min(trailing_stop, current_price + 0.75 * atr)  # Trailing stop động
            else:
                trailing_stop = min(trailing_stop, entry_price + 0.75 * atr)

            if df['high'].iloc[i] >= trailing_stop:
                exit_price = trailing_stop
                profit = (entry_price - exit_price) / entry_price * trades[-1]['position_size']
                balance += profit
                trades[-1].update({
                    'exit_price': exit_price,
                    'exit_time': df.index[i],
                    'profit': profit,
                    'profit_pct': (entry_price - exit_price) / entry_price * 100 * leverage,
                    'exit_reason': 'trailing_stop',
                    'hold_time': (df.index[i] - trades[-1]['entry_time']).total_seconds() / 3600
                })
                position = None
            elif df['rsi'].iloc[i] > 60 or df['close'].iloc[i] > df['sma5'].iloc[i]:  # RSI đảo chiều hoặc giá trên SMA5
                exit_price = current_price
                profit = (entry_price - exit_price) / entry_price * trades[-1]['position_size']
                balance += profit
                trades[-1].update({
                    'exit_price': exit_price,
                    'exit_time': df.index[i],
                    'profit': profit,
                    'profit_pct': (entry_price - exit_price) / entry_price * 100 * leverage,
                    'exit_reason': 'rsi_or_sma5',
                    'hold_time': (df.index[i] - trades[-1]['entry_time']).total_seconds() / 3600
                })
                position = None

    # Đóng vị thế cuối cùng
    if position is not None:
        exit_price = df['close'].iloc[-1]
        if position == 'LONG':
            profit = (exit_price - entry_price) / entry_price * trades[-1]['position_size']
        else:
            profit = (entry_price - exit_price) / entry_price * trades[-1]['position_size']
        balance += profit
        trades[-1].update({
            'exit_price': exit_price,
            'exit_time': df.index[-1],
            'profit': profit,
            'profit_pct': profit / trades[-1]['position_size'] * 100,
            'exit_reason': 'end_of_backtest',
            'hold_time': (df.index[-1] - trades[-1]['entry_time']).total_seconds() / 3600
        })

    trades_df = pd.DataFrame(trades)
    final_balance = balance
    profit = final_balance - initial_balance
    profit_percent = (profit / initial_balance) * 100