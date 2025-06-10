import pandas as pd
import numpy as np
from binance.client import Client
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestFramework:
    def __init__(self, api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET"):
        """Initialize the backtest framework"""
        self.client = Client(api_key, api_secret)
        if not os.path.exists('temp'):
            os.makedirs('temp')

    def get_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data for backtesting"""
        logger.info(f"Fetching data for {symbol}...")
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        klines = []
        current = start_timestamp
        while current < end_timestamp:
            temp_klines = self.client.get_historical_klines(
                symbol=symbol, interval=interval,
                start_str=current,
                end_str=min(current + 1000 * 60 * 60 * 24 * 30, end_timestamp)
            )
            if not temp_klines:
                break
            klines.extend(temp_klines)
            current = temp_klines[-1][0] + 1
            time.sleep(0.5)

        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                                         'taker_buy_quote', 'ignored'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']] = df[['open', 'high', 'low', 
                                                                            'close', 'volume', 'quote_volume']].apply(pd.to_numeric, errors='coerce')
        df.set_index('timestamp', inplace=True)
        return df

    def get_higher_timeframe_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get higher timeframe data for trend analysis"""
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        adjusted_start = (start_date_obj - timedelta(days=30)).strftime("%Y-%m-%d")
        
        if interval == Client.KLINE_INTERVAL_15MINUTE or interval == Client.KLINE_INTERVAL_30MINUTE:
            higher_interval = Client.KLINE_INTERVAL_4HOUR
        else:
            higher_interval = Client.KLINE_INTERVAL_4HOUR
        
        logger.info(f"Fetching {higher_interval} data for {symbol}...")
        return self.get_data(symbol, higher_interval, adjusted_start, end_date)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        # Trend indicators
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema50'] = df['ema50'].bfill()
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14']
        df['di_plus'] = adx['DMP_14']
        df['di_minus'] = adx['DMN_14']
        
        df['ema50_slope'] = pd.Series(dtype=float)
        mask = df['ema50'].notna() & df['ema50'].shift(3).notna()
        if mask.any():
            df.loc[mask, 'ema50_slope'] = (
                df.loc[mask, 'ema50'].diff(3) / df.loc[mask, 'ema50'].shift(3) * 100
            )
        df['ema50_slope'] = df['ema50_slope'].fillna(0)
        
        df['uptrend'] = (df['close'] > df['ema50']) & (df['ema50_slope'] > 0.05)
        df['downtrend'] = (df['close'] < df['ema50']) & (df['ema50_slope'] < -0.05)
        
        # Signal indicators
        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['rsi14'] = ta.rsi(df['close'], length=14)
        
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        df['ema_cross_up'] = (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))
        df['ema_cross_down'] = (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))
        
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        df['high_5'] = df['high'].rolling(5).max()
        df['low_5'] = df['low'].rolling(5).min()
        df['breakout_up'] = (df['close'] > df['high_5'].shift(1)) & (df['close'].shift(1) <= df['high_5'].shift(2))
        df['breakout_down'] = (df['close'] < df['low_5'].shift(1)) & (df['close'].shift(1) >= df['low_5'].shift(2))
        
        df['volume_ma10'] = df['volume'].rolling(10).mean()
        df['volume_increase'] = df['volume'] > df['volume_ma10']
        
        return df

    def run_backtest(self, symbol: str, start_date: str, end_date: str, interval: str,
                    initial_balance: float = 10, leverage: float = 20, risk_per_trade: float = 0.02) -> dict:
        """Run the backtest"""
        # Get data
        df = self.get_data(symbol, interval, start_date, end_date)
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return None

        # Get higher timeframe data
        higher_tf_df = self.get_higher_timeframe_data(symbol, interval, start_date, end_date)
        
        # Add indicators
        df = self.add_indicators(df)
        higher_tf_df = self.add_indicators(higher_tf_df)
        
        # Map higher timeframe data
        df['higher_uptrend'] = False
        df['higher_downtrend'] = False
        df['higher_adx'] = 0.0
        df['higher_di_plus'] = 0.0
        df['higher_di_minus'] = 0.0
        
        for i, row in df.iterrows():
            mask = higher_tf_df.index <= i
            if mask.any() and not higher_tf_df.empty:
                latest_higher_tf = higher_tf_df[mask].iloc[-1]
                for col in ['uptrend', 'downtrend', 'adx', 'di_plus', 'di_minus']:
                    if col in latest_higher_tf and pd.notna(latest_higher_tf[col]):
                        df.loc[i, f'higher_{col}'] = latest_higher_tf[col]
        
        # Run strategy
        trades = []
        balance = initial_balance
        position = None
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Entry conditions
            long_condition = self._check_long_condition(current)
            short_condition = self._check_short_condition(current)
            
            # Position management
            if position:
                position = self._manage_position(position, current, trades)
            else:
                if balance > initial_balance * 0.1 and balance > 0:
                    if long_condition or short_condition:
                        position = self._open_position(current, balance, risk_per_trade, leverage, 
                                                     long_condition)
        
        # Calculate results
        return self._calculate_results(trades, initial_balance)

    def _check_long_condition(self, current: pd.Series) -> bool:
        """Check long entry conditions"""
        long_primary = [
            current['ema9'] > current['ema21'],
            current['ema_cross_up'] or current['macd_cross_up'] or current['breakout_up']
        ]
        
        long_secondary = [
            current['rsi14'] < 80,
            current['volume_increase'],
            current['macd'] > 0
        ]
        
        higher_adx_value = current['higher_adx'] if pd.notna(current['higher_adx']) else 0
        higher_di_plus = current['higher_di_plus'] if pd.notna(current['higher_di_plus']) else 0
        higher_di_minus = current['higher_di_minus'] if pd.notna(current['higher_di_minus']) else 0
        
        return (all(long_primary) and 
                any(long_secondary) and
                (current['higher_uptrend'] or (higher_adx_value > 25 and higher_di_plus > higher_di_minus)))

    def _check_short_condition(self, current: pd.Series) -> bool:
        """Check short entry conditions"""
        short_primary = [
            current['ema9'] < current['ema21'],
            current['ema_cross_down'] or current['macd_cross_down'] or current['breakout_down']
        ]
        
        short_secondary = [
            current['rsi14'] > 20,
            current['volume_increase'],
            current['macd'] < 0
        ]
        
        higher_adx_value = current['higher_adx'] if pd.notna(current['higher_adx']) else 0
        higher_di_plus = current['higher_di_plus'] if pd.notna(current['higher_di_plus']) else 0
        higher_di_minus = current['higher_di_minus'] if pd.notna(current['higher_di_minus']) else 0
        
        return (all(short_primary) and 
                any(short_secondary) and
                (current['higher_downtrend'] or (higher_adx_value > 25 and higher_di_minus > higher_di_plus)))

    def _manage_position(self, position: dict, current: pd.Series, trades: list) -> dict:
        """Manage open position"""
        current_price = current['close']
        profit = (current_price - position['entry_price']) * position['size'] * (1 if position['type'] == 'LONG' else -1)
        risk_amount = position['size'] * abs(position['entry_price'] - position['stop_loss'])
        r_multiple = profit / risk_amount if risk_amount != 0 else 0
        
        # Update position state
        if r_multiple > 0.7 and not position['breakeven_activated']:
            position['stop_loss'] = position['entry_price']
            position['breakeven_activated'] = True
        
        if r_multiple > 1:
            trail_factor = min(1.5, 1 + r_multiple * 0.1)
            new_stop = (current_price - current['atr14'] * trail_factor if position['type'] == 'LONG'
                       else current_price + current['atr14'] * trail_factor)
            position['stop_loss'] = (max(position['stop_loss'], new_stop) if position['type'] == 'LONG'
                                   else min(position['stop_loss'], new_stop))
        
        # Check exit conditions
        if self._should_exit_position(position, current, r_multiple):
            position['exit_time'] = current.name
            position['exit_price'] = current_price
            position['profit'] = profit
            position['hold_time'] = (position['exit_time'] - position['entry_time']).total_seconds() / 3600
            position['r_multiple'] = r_multiple
            trades.append(position)
            return None
        
        return position

    def _should_exit_position(self, position: dict, current: pd.Series, r_multiple: float) -> bool:
        """Check if position should be closed"""
        if position['type'] == 'LONG':
            exit_conditions = [
                current['close'] <= position['stop_loss'],
                current['ema_cross_down'],
                current['macd_cross_down'],
                current['rsi14'] > 80,
                not current['higher_uptrend'] and r_multiple > 0,
                r_multiple >= 4
            ]
        else:
            exit_conditions = [
                current['close'] >= position['stop_loss'],
                current['ema_cross_up'],
                current['macd_cross_up'],
                current['rsi14'] < 20,
                not current['higher_downtrend'] and r_multiple > 0,
                r_multiple >= 4
            ]
        
        return any(exit_conditions)

    def _open_position(self, current: pd.Series, balance: float, risk_per_trade: float,
                      leverage: float, is_long: bool) -> dict:
        """Open a new position"""
        position_type = 'LONG' if is_long else 'SHORT'
        entry_price = current['close']
        
        # Calculate stop loss
        atr = current['atr14']
        if position_type == 'LONG':
            recent_low = current['low']
            if recent_low < entry_price * 0.99:
                stop_loss = recent_low - atr * 0.3
            else:
                stop_loss = entry_price - atr * 1.5
        else:
            recent_high = current['high']
            if recent_high > entry_price * 1.01:
                stop_loss = recent_high + atr * 0.3
            else:
                stop_loss = entry_price + atr * 1.5
        
        # Calculate position size
        risk_amount = balance * risk_per_trade
        risk_per_r = abs(entry_price - stop_loss)
        size = (risk_amount / risk_per_r) * leverage
        
        return {
            'type': position_type,
            'entry_time': current.name,
            'exit_time': None,
            'entry_price': entry_price,
            'exit_price': None,
            'size': size,
            'stop_loss': stop_loss,
            'breakeven_activated': False,
            'first_target_hit': False,
            'second_target_hit': False
        }

    def _calculate_results(self, trades: list, initial_balance: float) -> dict:
        """Calculate backtest results"""
        if not trades:
            return {
                'trades': trades,
                'initial_balance': initial_balance,
                'final_balance': initial_balance,
                'profit': 0,
                'profit_percent': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'max_drawdown_percent': 0,
                'avg_hold_time': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        total_profit = sum(trade['profit'] for trade in trades)
        final_balance = initial_balance + total_profit
        profit_percent = (final_balance / initial_balance - 1) * 100
        
        winning_trades = len([t for t in trades if t['profit'] > 0])
        losing_trades = len([t for t in trades if t['profit'] <= 0])
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        cumulative_profit = np.cumsum([t['profit'] for t in trades])
        peak = np.maximum.accumulate(cumulative_profit)
        drawdown = peak - cumulative_profit
        max_drawdown = np.max(drawdown)
        max_drawdown_percent = (max_drawdown / (initial_balance + peak[np.argmax(drawdown)])) * 100
        
        avg_hold_time = np.mean([t['hold_time'] for t in trades])
        
        return {
            'trades': trades,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'profit': total_profit,
            'profit_percent': profit_percent,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_percent,
            'avg_hold_time': avg_hold_time,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }

    def plot_results(self, df: pd.DataFrame, trades: list, symbol: str, 
                    start_date: str, end_date: str, save_path: str = None):
        """Plot backtest results"""
        if not trades:
            logger.warning("No trades to plot")
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # Price chart
        ax1 = plt.subplot(gs[0])
        ax1.plot(df.index, df['close'], color='blue', linewidth=1, label='Price')
        
        # Plot trades
        for trade in trades:
            if trade['type'] == 'LONG':
                ax1.scatter(trade['entry_time'], trade['entry_price'], 
                           marker='^', color='g', s=100, 
                           label='Long Entry' if 'Long Entry' not in ax1.get_legend_handles_labels()[1] else '')
            else:
                ax1.scatter(trade['entry_time'], trade['entry_price'], 
                           marker='v', color='r', s=100, 
                           label='Short Entry' if 'Short Entry' not in ax1.get_legend_handles_labels()[1] else '')
            
            if trade['exit_time'] and trade['exit_price']:
                color = 'g' if trade['profit'] > 0 else 'r'
                ax1.scatter(trade['exit_time'], trade['exit_price'], 
                           marker='X', color=color, s=100, 
                           label='Exit' if 'Exit' not in ax1.get_legend_handles_labels()[1] else '')
                ax1.annotate(f'{(trade["profit"]/trade["size"]/trade["entry_price"]*100):.1f}%', 
                           xy=(trade['exit_time'], trade['exit_price']),
                           xytext=(10, 10), textcoords='offset points',
                           color=color)
        
        ax1.set_title(f'Price Chart - {symbol}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Volume chart
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.bar(df.index, df['volume'], color='blue', alpha=0.5)
        ax2.set_title('Volume')
        ax2.grid(True, alpha=0.3)
        
        # Equity curve
        ax3 = plt.subplot(gs[2], sharex=ax1)
        cumulative_profit = np.cumsum([t['profit'] for t in trades])
        ax3.plot(df.index[:len(cumulative_profit)], cumulative_profit, color='blue', linewidth=1.5)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax3.fill_between(df.index[:len(cumulative_profit)], cumulative_profit, 0, 
                        where=(cumulative_profit >= 0), color='green', alpha=0.3)
        ax3.fill_between(df.index[:len(cumulative_profit)], cumulative_profit, 0, 
                        where=(cumulative_profit < 0), color='red', alpha=0.3)
        
        ax3.set_title('Cumulative P&L')
        ax3.set_ylabel('Profit/Loss')
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved chart to {save_path}")
        
        plt.close()

def main():
    """Main function to run the backtest"""
    # Initialize framework
    backtest = BacktestFramework()
    
    # Test parameters
    start_date = "2025-02-04"
    end_date = "2025-06-07"
    interval = Client.KLINE_INTERVAL_5MINUTE
    symbols = {
        "XRPUSDT",
        "ETHUSDT",
        'AAVEUSDT',
        'LINKUSDT',
        'MKRUSDT',
        'LTCUSDT',
        'ENAUSDT',
       
    }
    
    results_summary = []
    
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"BACKTEST MOMENTUM STRATEGY FOR {symbol}")
        logger.info(f"{'='*50}")
        
        try:
            # Run backtest
            result = backtest.run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                initial_balance=10,
                leverage=10,
                risk_per_trade=0.02
            )
            
            if result:
                # Save trades to CSV
                trades_df = pd.DataFrame(result['trades'])
                trades_df.to_csv(f'temp/momentum_trades_{symbol}_{start_date}_{end_date}.csv')
                
                # Plot results
                df = backtest.get_data(symbol, interval, start_date, end_date)
                backtest.plot_results(
                    df=df,
                    trades=result['trades'],
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    save_path=f'temp/trades_chart_{symbol}_{start_date}_{end_date}.png'
                )
                
                # Add to summary
                results_summary.append({
                    'Symbol': symbol,
                    'Số giao dịch': result['total_trades'],
                    'Tỷ lệ thắng (%)': round(result['win_rate'], 2),
                    'P&L (USDT)': round(result['profit'], 2),
                    'Lợi nhuận (%)': round(result['profit_percent'], 2)
                })
                
                # Log results
                logger.info(f"\nResults for {symbol}:")
                logger.info(f"Total trades: {result['total_trades']}")
                logger.info(f"Win rate: {result['win_rate']:.2f}%")
                logger.info(f"Profit: {result['profit']:.4f} USDT")
                logger.info(f"Profit %: {result['profit_percent']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            results_summary.append({
                'Symbol': symbol,
                'Số giao dịch': 0,
                'Tỷ lệ thắng (%)': 0,
                'P&L (USDT)': 0,
                'Lợi nhuận (%)': 0
            })
    
    # Save and display summary
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        logger.info("\n" + "="*60)
        logger.info("MOMENTUM STRATEGY BACKTEST SUMMARY")
        logger.info("="*60)
        logger.info("\n" + results_df.to_string(index=False))
        logger.info("="*60)
        results_df.to_csv(f'temp/momentum_summary_{start_date}_{end_date}.csv', index=False)

if __name__ == "__main__":
    main()
