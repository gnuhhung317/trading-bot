import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import logging
from binance import Client
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WaveRiderStrategy(Strategy):
    # Define parameters with optimization ranges
    volume_ma_length = 10     # [10, 20, 30]
    volume_threshold = 1.8    # [1.5, 2.0, 2.5] - Reduced to catch more waves
    momentum_length = 10       # [5, 10, 15]
    atr_length = 10           # [10, 14, 20]
    atr_sl_multiplier = 2.5    # [1.5, 2.0, 2.5] - Increased for wider stops
    atr_tp_multiplier = 5.0    # [2.0, 3.0, 4.0] - Increased for better R:R
    risk_per_trade = 0.01      # [0.005, 0.01, 0.02]
    min_volume_increase = 1.3  # [1.3, 1.5, 1.8] - Reduced to catch more waves
    rsi_length = 14            # [10, 14, 20]
    rsi_overbought = 75        # [65, 70, 75] - More lenient
    rsi_oversold = 30         # [25, 30, 35] - More lenient
    sma_length = 100           # [100, 200, 300] - Shorter for faster trend detection
    max_holding_period = 40    # [10, 20, 30] - Increased to let winners run

    def init(self):
        # Calculate Volume MA
        self.volume_ma = self.I(
            lambda x: pd.Series(x).rolling(window=self.volume_ma_length).mean(),
            self.data.Volume
        )

        # Calculate Momentum (rate of change)
        def momentum(close, period):
            return (pd.Series(close).pct_change(period) * 100).fillna(0)
        self.momentum = self.I(momentum, self.data.Close, self.momentum_length)

        # Calculate ATR for stop loss and take profit
        def atr(high, low, close, period):
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=period).mean().bfill()
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_length)

        # Calculate price velocity (rate of change over shorter period)
        def price_velocity(close, period=3):
            return (pd.Series(close).pct_change(period) * 100).fillna(0)
        self.price_velocity = self.I(price_velocity, self.data.Close)

        # Calculate volume acceleration
        def volume_acceleration(volume, period=3):
            return (pd.Series(volume).pct_change(period) * 100).fillna(0)
        self.volume_acceleration = self.I(volume_acceleration, self.data.Volume)

        # Calculate RSI
        def rsi(close, period):
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).fillna(50)
        self.rsi = self.I(rsi, self.data.Close, self.rsi_length)

        # Calculate 200-period SMA for trend filter
        self.sma200 = self.I(
            lambda x: pd.Series(x).rolling(window=self.sma_length).mean(),
            self.data.Close
        )

    def next(self):
        # Get current values
        current_volume = self.data.Volume[-1]
        current_volume_ma = self.volume_ma[-1]
        current_momentum = self.momentum[-1]
        current_velocity = self.price_velocity[-1]
        current_volume_acc = self.volume_acceleration[-1]
        prev_volume = self.data.Volume[-2]
        current_rsi = self.rsi[-1]
        current_price = self.data.Close[-1]
        current_sma200 = self.sma200[-1]

        # Dynamic momentum threshold based on ATR
        momentum_threshold = self.atr[-1] * 0.05  # Reduced to catch more moves

        # Calculate position size based on risk
        stop_distance = self.atr[-1] * self.atr_sl_multiplier
        position_size = (self.equity * self.risk_per_trade / stop_distance  ) if stop_distance > 0 else 1
        position_size = round(position_size) if position_size>1 else position_size

        # Update trailing stop for open positions
        if self.position:
            if self.position.is_long:
                # Calculate new stop loss level with wider buffer
                new_sl = current_price - self.atr[-1] * self.atr_sl_multiplier * 1.2
                # Close if price hits stop loss
                if current_price <= new_sl:
                    self.position.close()
                # Exit conditions: momentum fade, volume drop, or max holding period
                elif (self.momentum[-1] < -momentum_threshold or  # Only exit on strong reversal
                    current_volume < current_volume_ma * 0.3 or  # More lenient volume drop
                    len(self.data.Close) - self.trades[-1].entry_bar > self.max_holding_period):
                    self.position.close()
            else:
                # Calculate new stop loss level with wider buffer
                new_sl = current_price + self.atr[-1] * self.atr_sl_multiplier * 1.2
                # Close if price hits stop loss
                if current_price >= new_sl:
                    self.position.close()
                # Exit conditions: momentum fade, volume drop, or max holding period
                elif (self.momentum[-1] > momentum_threshold or  # Only exit on strong reversal
                    current_volume < current_volume_ma * 0.3 or  # More lenient volume drop
                    len(self.data.Close) - self.trades[-1].entry_bar > self.max_holding_period):
                    self.position.close()
        else:
            # Long entry conditions
            long_conditions = [
                current_volume > current_volume_ma * self.volume_threshold,  # High volume
                current_volume > prev_volume * self.min_volume_increase,     # Volume increasing
                current_momentum > momentum_threshold,                       # Strong momentum
                current_velocity > 0,                                        # Price moving up
                current_volume_acc > 0,                                      # Volume accelerating
                current_rsi < self.rsi_overbought,                           # Not overbought
                current_price > current_sma200 * 0.995                       # Slightly below trend is okay
            ]

            # Short entry conditions
            short_conditions = [
                current_volume > current_volume_ma * self.volume_threshold,  # High volume
                current_volume > prev_volume * self.min_volume_increase,     # Volume increasing
                current_momentum < -momentum_threshold,                      # Strong negative momentum
                current_velocity < 0,                                        # Price moving down
                current_volume_acc > 0,                                      # Volume accelerating
                current_rsi > self.rsi_oversold,                             # Not oversold
                current_price < current_sma200 * 1.005                       # Slightly above trend is okay
            ]

            # Enter positions if conditions are met
            if all(long_conditions):
                sl = current_price - stop_distance
                tp = current_price + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp, size=position_size)
                logger.info(f"LONG Entry - Price: {current_price:.2f}, Volume: {current_volume:.2f}, Momentum: {current_momentum:.2f}%")

            elif all(short_conditions):
                sl = current_price + stop_distance
                tp = current_price - (self.atr[-1] * self.atr_tp_multiplier)
                self.sell(sl=sl, tp=tp, size=position_size)
                logger.info(f"SHORT Entry - Price: {current_price:.2f}, Volume: {current_volume:.2f}, Momentum: {current_momentum:.2f}%")

def get_historical_data(client: Client, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get historical data from Binance"""
    logger.info(f"Fetching data for {symbol}...")
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    klines = []
    current = start_timestamp
    while current < end_timestamp:
        temp_klines = client.get_historical_klines(
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
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    
    # Rename columns to match backtesting library requirements
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    df.set_index('timestamp', inplace=True)
    return df

def run_backtest(symbol: str, interval: str, start_date: str, end_date: str, initial_balance: float = 10000, commission: float = 0.0001) -> dict:
    """Run backtest for a single symbol"""
    try:
        # Initialize Binance client
        client = Client()
        
        # Get historical data
        df = get_historical_data(client, symbol, interval, start_date, end_date)
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return None
            
        # Prepare data for backtesting
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Run backtest
        bt = Backtest(df, WaveRiderStrategy, cash=initial_balance, commission=commission)
        stats = bt.run()
        
        # Plot results
        bt.plot(filename=f'backtest_{symbol}_wave_rider.html')

        return {
            'symbol': symbol,
            'total_trades': stats['# Trades'],
            'win_rate': stats['Win Rate [%]'],
            'profit_factor': stats['Profit Factor'],
            'expectancy': stats['Expectancy [%]'],
            'max_drawdown': stats['Max. Drawdown [%]'],
            'sharpe_ratio': stats['Sharpe Ratio'],
            'sortino_ratio': stats['Sortino Ratio'],
            'calmar_ratio': stats['Calmar Ratio'],
            'equity_final': stats['Equity Final [$]'],
            'equity_peak': stats['Equity Peak [$]'],
            'return_pct': stats['Return [%]'],
            'buy_hold_return': stats['Buy & Hold Return [%]'],
            'strategy_params': stats['_strategy'].__dict__
        }

    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {str(e)}")
        return None

def main():
    """Main function to run backtest"""
    symbol = "SOLUSDT"
    
    logger.info(f"\n{'='*50}")
    logger.info(f"OPTIMIZING WAVE RIDER STRATEGY FOR {symbol}")
    logger.info(f"{'='*50}")

    # Initialize Binance client
    client = Client()
    
    # Get historical data
    df = get_historical_data(client, symbol, "15m", "2024-07-01", "2025-04-30")
    if df.empty:
        logger.error(f"No data available for {symbol}")
        return
        
    # Prepare data for backtesting
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Run optimization
    bt = Backtest(df, WaveRiderStrategy, cash=10000, commission=0.0001)
    
    # # Define parameter ranges for optimization
    # optimization_params = {
    #     'volume_ma_length': range(10, 31, 10),      # [10, 20, 30]
    #     'volume_threshold': [1.3, 1.5, 1.8, 2.0],
    #     'momentum_length': range(5, 16, 5),         # [5, 10, 15]
    #     'atr_length': [10, 14, 20],
    #     'atr_sl_multiplier': [2.0, 2.5, 3.0],
    #     'atr_tp_multiplier': [3.0, 4.0, 5.0],
    #     'risk_per_trade': [0.005, 0.01, 0.02],
    #     'min_volume_increase': [1.2, 1.3, 1.5],
    #     'rsi_length': [10, 14, 20],
    #     'rsi_overbought': [70, 75, 80],
    #     'rsi_oversold': [20, 25, 30],
    #     'sma_length': [50, 100, 200],
    #     'max_holding_period': [20, 30, 40]
    # }

    # # Run optimization
    # stats = bt.optimize(
    #     **optimization_params,
    #     maximize='Return [%]',
    #     method='grid'
    # )
    stats = bt.run()

    # Print optimization results
    logger.info("\nOptimization Results:")
    logger.info("=" * 50)
    logger.info(f"Best Parameters:")
    for param, value in stats._strategy.__dict__.items():
        if not param.startswith('_'):
            logger.info(f"{param}: {value}")
    
    logger.info("\nPerformance Metrics:")
    logger.info("=" * 50)
    logger.info(f"Total trades: {stats['# Trades']}")
    logger.info(f"Win rate: {stats['Win Rate [%]']:.2f}%")
    logger.info(f"Return: {stats['Return [%]']:.2f}%")
    logger.info(f"Max drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    logger.info(f"Profit Factor: {stats['Profit Factor']:.2f}")
    logger.info(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    logger.info(f"Sortino Ratio: {stats['Sortino Ratio']:.2f}")
    logger.info(f"Calmar Ratio: {stats['Calmar Ratio']:.2f}")
    logger.info(f"Final Equity: ${stats['Equity Final [$]']:.2f}")
    logger.info(f"Peak Equity: ${stats['Equity Peak [$]']:.2f}")
    logger.info(f"Buy & Hold Return: {stats['Buy & Hold Return [%]']:.2f}%")

    # Plot results
    bt.plot(filename=f'backtest_{symbol}_optimized.html')

if __name__ == "__main__":
    main()