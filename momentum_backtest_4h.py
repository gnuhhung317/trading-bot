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
)
logger = logging.getLogger(__name__)

class WaveRiderStrategy4H(Strategy):
    # Updated parameters for 4H timeframe
    volume_ma_length = 7       # [5, 7, 10] - ~1 day (7*4h)
    volume_threshold = 1.8     # [1.5, 1.8, 2.0] - Less sensitive volume spike
    momentum_length = 4        # [3, 4, 5] - ~16 hours
    atr_length = 14            # [10, 14, 20] - ~2.5 days
    atr_sl_multiplier = 2      # [1.5, 2.0, 2.5] - Increased for wider stops
    atr_tp_multiplier = 3.0    # [2.0, 3.0, 4.0] - Increased for better R:R
    risk_per_trade = 0.01      # [0.005, 0.01, 0.02]
    min_volume_increase = 1.5  # [1.3, 1.5, 1.8] - Reduced for 4H
    rsi_length = 14            # [10, 14, 20] - Standard for longer TFs
    rsi_overbought = 75        # [65, 70, 75] - More lenient
    rsi_oversold = 15          # [15, 20, 25] - More lenient
    sma_length = 50            # [50, 100, 200] - Medium-term trend
    max_holding_period = 6     # [4, 6, 8] - 6*4h = 24 hours

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

        # Calculate SMA for trend filter (changed from 200 to configurable)
        self.sma = self.I(
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
        current_sma = self.sma[-1]
        timestamp = self.data.index[-1]
        
        # Dynamic momentum threshold based on ATR
        momentum_threshold = self.atr[-1] * 0.05  # Reduced to catch more moves

        # Volatility filter for 4H timeframe
        atr_pct = self.atr[-1] / current_price
        valid_volatility = 0.01 < atr_pct < 0.05  # Filter choppy markets

        # Gap detection for 4H timeframe
        if len(self.data.Close) > 1:
            prev_close = self.data.Close[-2]
            current_open = self.data.Open[-1]
            gap_up = current_open > prev_close * 1.01
            gap_down = current_open < prev_close * 0.99
        else:
            gap_up = gap_down = False

        # Calculate position size based on risk
        stop_distance = self.atr[-1] * self.atr_sl_multiplier
        position_size = (self.equity * self.risk_per_trade / stop_distance) if stop_distance > 0 else 1
        position_size = round(position_size) if position_size > 1 else position_size

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
                    current_volume < current_volume_ma * 0.7 or  # More lenient volume drop (4H adjustment)
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
                    current_volume < current_volume_ma * 0.7 or  # More lenient volume drop (4H adjustment)
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
                current_price > current_sma * 0.995,                         # Slightly below trend is okay
                valid_volatility,                                            # Valid volatility range
                not gap_down                                                 # Avoid entering on negative gaps
            ]

            # Short entry conditions
            short_conditions = [
                current_volume > current_volume_ma * self.volume_threshold,  # High volume
                current_volume > prev_volume * self.min_volume_increase,     # Volume increasing
                current_momentum < -momentum_threshold,                      # Strong negative momentum
                current_velocity < 0,                                        # Price moving down
                current_volume_acc > 0,                                      # Volume accelerating
                current_rsi > self.rsi_oversold,                             # Not oversold
                current_price < current_sma * 1.005,                         # Slightly above trend is okay
                valid_volatility,                                            # Valid volatility range
                not gap_up                                                   # Avoid entering on positive gaps
            ]

            # Enter positions if conditions are met
            if all(long_conditions):
                sl = current_price - stop_distance
                tp = current_price + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp, size=position_size)
                logger.info(f"{timestamp} LONG Entry - Price: {current_price:.5f}, Volume: {current_volume:.2f}, Momentum: {current_momentum:.2f}%")

            elif all(short_conditions):
                sl = current_price + stop_distance
                tp = current_price - (self.atr[-1] * self.atr_tp_multiplier)
                self.sell(sl=sl, tp=tp, size=position_size)
                logger.info(f"{timestamp} SHORT Entry - Price: {current_price:.5f}, Volume: {current_volume:.2f}, Momentum: {current_momentum:.2f}%")

def get_historical_data(client: Client, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get historical data from Binance"""
    logger.info(f"Fetching 4H data for {symbol}...")
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

def run_single_backtest(symbol: str, start_date: str, end_date: str, params: dict = None) -> dict:
    """Run backtest with specific parameters"""
    try:
        # Initialize Binance client
        client = Client()
        
        # Get historical data
        df = get_historical_data(client, symbol, "4h", start_date, end_date)
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return None
            
        # Prepare data for backtesting
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Create strategy class with custom parameters
        if params:
            class CustomStrategy(WaveRiderStrategy4H):
                pass
            for key, value in params.items():
                setattr(CustomStrategy, key, value)
            strategy_class = CustomStrategy
        else:
            strategy_class = WaveRiderStrategy4H

        optimization_params = {
            'volume_ma_length': [5, 7, 10],
            'volume_threshold': [1.5, 1.8, 2.0],
            'momentum_length': [3, 4, 5],
            # 'atr_length': [10, 14, 20],
            # 'atr_sl_multiplier': [1.8, 2.0, 2.5],
            # 'atr_tp_multiplier': [2.5, 3.0, 4.0],
            # 'risk_per_trade': [0.005, 0.01, 0.02],
            # 'min_volume_increase': [1.3, 1.5, 1.8],
            # 'rsi_length': [10, 14, 20],
            # 'rsi_overbought': [70, 75, 80],
            # 'rsi_oversold': [15, 20, 25],
            # 'sma_length': [50, 100, 200],
            # 'max_holding_period': [4, 6, 8]
        }
        # Run backtest
        bt = Backtest(df, strategy_class, cash=10000, commission=0.0001)
        stats = bt.optimize(
            **optimization_params,)
        
        # Plot results
        bt.plot(filename=f'backtest_{symbol}_4h.html')

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
    """Main function to run backtest and optimization"""
    symbol = "SOLUSDT"  # Replace with your desired symbol
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZING WAVE RIDER STRATEGY FOR {symbol} - 4H TIMEFRAME")
    logger.info(f"{'='*60}")

    # Initialize Binance client
    client = Client()
    
    # Get historical data for 4H timeframe (longer period for better testing)
    df = get_historical_data(client, symbol, "4h", "2025-03-01", "2025-07-14")
    if df.empty:
        logger.error(f"No data available for {symbol}")
        return
        
    # Prepare data for backtesting
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    logger.info(f"Data loaded: {len(df)} 4H candles from {df.index[0]} to {df.index[-1]}")

    # Run optimization
    bt = Backtest(df, WaveRiderStrategy4H, cash=10000, commission=0.0001)
    
    # Define parameter ranges for optimization - Updated for 4H timeframe
    optimization_params = {
        'volume_ma_length': [5, 7, 10],               # Shorter periods for 4H
        'volume_threshold': [1.5, 1.8, 2.0],         # Less sensitive for 4H
        'momentum_length': [3, 4, 5],                 # Adjusted for 4H
        'atr_length': [10, 14, 20],                   # Standard ATR periods
        'atr_sl_multiplier': [1.8, 2.0, 2.5],        # Stop loss multipliers
        'atr_tp_multiplier': [2.5, 3.0, 4.0],        # Take profit multipliers
        'risk_per_trade': [0.005, 0.01, 0.02],       # Risk levels
        'min_volume_increase': [1.3, 1.5, 1.8],      # Volume increase thresholds
        'rsi_length': [10, 14, 20],                   # RSI periods
        'rsi_overbought': [70, 75, 80],               # Overbought levels
        'rsi_oversold': [15, 20, 25],                 # Oversold levels
        'sma_length': [50, 100, 200],                 # Trend filter periods
        'max_holding_period': [4, 6, 8]              # Max holding in 4H candles
    }

    # Uncomment to run optimization (can take a long time)
    logger.info("Running optimization...")
    stats = bt.optimize(
        **optimization_params,
        maximize='Return [%]',
        method='grid'
    )
    
    # Uncomment to run single backtest with default parameters
    # stats = bt.run()

    # Print optimization results
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*60)
    
    if hasattr(stats, '_strategy'):
        logger.info(f"Best Parameters:")
        for param, value in stats._strategy.__dict__.items():
            if not param.startswith('_'):
                logger.info(f"  {param}: {value}")
    
    logger.info(f"\nPerformance Metrics:")
    logger.info("-" * 40)
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
    bt.plot(filename=f'backtest_{symbol}_4h_optimized.html')
    logger.info(f"\nResults saved to: backtest_{symbol}_4h_optimized.html")

if __name__ == "__main__":
    main()
