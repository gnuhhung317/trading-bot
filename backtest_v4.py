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

class ForexFuryStrategy(Strategy):
    # Define parameters
    ema_fast_length = 15
    ema_slow_length = 32
    rsi_length = 12
    rsi_overbought = 75
    rsi_oversold = 25
    atr_length = 12
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 1.5
    volume_ma_length = 8
    risk_per_trade = 0.01  # 1% risk per trade

    def init(self):
        # Calculate EMAs
        self.ema_fast = self.I(
            lambda x: pd.Series(x).ewm(span=self.ema_fast_length, adjust=False).mean(),
            self.data.Close
        )
        self.ema_slow = self.I(
            lambda x: pd.Series(x).ewm(span=self.ema_slow_length, adjust=False).mean(),
            self.data.Close
        )

        # Calculate RSI
        def rsi(close, period):
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        self.rsi = self.I(rsi, self.data.Close, self.rsi_length)

        # Calculate ATR
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

        # Calculate Volume MA
        self.volume_ma = self.I(
            lambda x: pd.Series(x).rolling(window=self.volume_ma_length).mean(),
            self.data.Volume
        )

    def next(self):
                # Get current timestamp and hour (in UTC)
        current_time = pd.Timestamp(self.data.index[-1])
        current_hour = current_time.hour

        # Only trade within specified hours
        if not (8 <= current_hour < 16):
            return  # Skip trading outside allowed hours
        # Calculate position size based on risk
        stop_distance = self.atr[-1] * self.atr_sl_multiplier
        position_size = round((self.equity * self.risk_per_trade) / stop_distance) if stop_distance > 0 else 1
        # Check if we have a position
        if self.position:
            # Exit conditions (handled by SL/TP/trailing stop in entry)
            pass
        else:
            # Long entry
            if (crossover(self.ema_fast, self.ema_slow) and
                self.rsi[-1] < self.rsi_overbought and
                self.data.Volume[-1] > self.volume_ma[-1]):
                sl = self.data.Close[-1] - stop_distance
                tp = self.data.Close[-1] + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp, size=position_size)

            # Short entry
            elif (crossover(self.ema_slow, self.ema_fast) and
                  self.rsi[-1] > self.rsi_oversold and
                  self.data.Volume[-1] > self.volume_ma[-1]):
                sl = self.data.Close[-1] + stop_distance
                tp = self.data.Close[-1] - (self.atr[-1] * self.atr_tp_multiplier)
                self.sell(sl=sl, tp=tp, size=position_size)

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
            
        # Prepare data for backtesting - only select required columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Run backtest with optimization
        bt = Backtest(df, ForexFuryStrategy, cash=initial_balance, commission=commission)
        # stats = bt.optimize(
        #     # ema_fast_length=range(6, 16, 1),           # Test fast EMA periods from 6 to 15, step 1
        #     # ema_slow_length=range(16, 36, 2),          # Test slow EMA periods from 16 to 34, step 2
        #     # rsi_length=range(8, 18, 2),                # Test RSI periods from 8 to 16, step 2
        #     # rsi_overbought=range(65, 80, 5),           # Test RSI overbought levels from 65 to 75, step 5
        #     # rsi_oversold=range(20, 35, 5),             # Test RSI oversold levels from 20 to 30, step 5
        #     # atr_length=range(8, 18, 2),                # Test ATR periods from 8 to 16, step 2
        #     # volume_ma_length=range(6, 16, 2),          # Test volume MA periods from 6 to 14, step 2
        #     maximize='Return [%]',                   # Optimize to maximize win rate
        #     method='grid'                              # Use grid search for exhaustive testing
        # )
        stats = bt.run()
        
        # Plot results
        bt.plot(filename=f'backtest_{symbol}_forex_fury.html')

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
            'optimized_params': stats['_strategy'].__dict__  # Include optimized parameters
        }

    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {str(e)}")
        return None

def main():
    """Main function to run backtest"""
    # Example: Load data (replace with your data source)
    # For Forex, you might use a data provider like OANDA or load a CSV
    # Example data loading (replace with actual data)
    symbol ="ETHUSDT"
    
    logger.info(f"\n{'='*50}")
    logger.info(f"BACKTEST FOREX FURY STRATEGY FOR {symbol}")
    logger.info(f"{'='*50}")

    result = run_backtest(
        symbol=symbol,
        interval="15m",
        start_date="2024-07-01",  # Changed to past date for real data
        end_date="2025-04-30",    # Changed to past date for real data
        initial_balance=10000,
        commission=0.0001  # Typical Forex commission/spread
    )

    if result:
        # Print performance metrics
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"{'='*30}")
        logger.info(f"Total trades: {result['total_trades']}")
        logger.info(f"Win rate: {result['win_rate']:.2f}%")
        logger.info(f"Return: {result['return_pct']:.2f}%")
        logger.info(f"Max drawdown: {result['max_drawdown']:.2f}%")
        logger.info(f"Profit Factor: {result['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        logger.info(f"Sortino Ratio: {result['sortino_ratio']:.2f}")
        logger.info(f"Calmar Ratio: {result['calmar_ratio']:.2f}")
        logger.info(f"Final Equity: ${result['equity_final']:.2f}")
        logger.info(f"Peak Equity: ${result['equity_peak']:.2f}")
        logger.info(f"Buy & Hold Return: {result['buy_hold_return']:.2f}%")
        
        # Print optimized parameters
        logger.info(f"\nOptimized Parameters:")
        logger.info(f"{'='*30}")
        params = result['optimized_params']
        param_descriptions = {
            'ema_fast_length': 'Fast EMA Period',
            'ema_slow_length': 'Slow EMA Period',
            'rsi_length': 'RSI Period',
            'rsi_overbought': 'RSI Overbought Level',
            'rsi_oversold': 'RSI Oversold Level',
            'atr_length': 'ATR Period',
            'volume_ma_length': 'Volume MA Period',
            'risk_per_trade': 'Risk Per Trade (%)'
        }
        
        for param, value in params.items():
            if not param.startswith('_'):  # Skip internal parameters
                description = param_descriptions.get(param, param)
                if param == 'risk_per_trade':
                    value = value * 100  # Convert to percentage
                logger.info(f"{description}: {value:.2f}")

        # Save results to CSV
        results_df = pd.DataFrame([result])
        results_df.to_csv(f'backtest_summary_{symbol}.csv', index=False)
        logger.info(f"\nSaved results to backtest_summary_{symbol}.csv")

if __name__ == "__main__":
    main()