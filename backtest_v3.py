import pandas as pd
import numpy as np
from binance.client import Client
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MomentumStrategy(Strategy):
    """Momentum Strategy implementation using backtesting library"""
    
    # Define parameters
    ema_fast = 9
    ema_slow = 21
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    
    def init(self):
        """Initialize indicators"""
        # EMAs
        self.ema_fast = self.I(lambda x: pd.Series(x).ewm(span=self.ema_fast, adjust=False).mean(), self.data.Close)
        self.ema_slow = self.I(lambda x: pd.Series(x).ewm(span=self.ema_slow, adjust=False).mean(), self.data.Close)
        
        # RSI
        def rsi(close, period):
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)
        
        # MACD
        def macd(close, fast, slow, signal):
            exp1 = pd.Series(close).ewm(span=fast, adjust=False).mean()
            exp2 = pd.Series(close).ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            return macd_line, signal_line, macd_line - signal_line
            
        macd_result = macd(self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal)
        self.macd = self.I(lambda x: x, macd_result[0])
        self.macd_signal = self.I(lambda x: x, macd_result[1])
        self.macd_hist = self.I(lambda x: x, macd_result[2])
        
        # ATR
        def atr(high, low, close, period):
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr.bfill()
            
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Volume
        self.volume_ma = self.I(lambda x: pd.Series(x).rolling(window=10).mean(), self.data.Volume)
        
        # Higher timeframe trend
        self.ema50 = self.I(lambda x: pd.Series(x).ewm(span=50, adjust=False).mean(), self.data.Close)
        self.ema50_slope = self.I(lambda x: pd.Series(x).diff(3) / pd.Series(x).shift(3) * 100, self.ema50)
        
    def next(self):
        """Define trading logic"""
        # Check if we have a position
        if self.position:
            # Exit conditions
            if self.position.is_long:
                # Long exit conditions
                if (crossover(self.ema_slow, self.ema_fast) or  # EMA crossover down
                    crossover(self.macd_signal, self.macd) or   # MACD crossover down
                    self.rsi[-1] > 80 ):   
                    self.position.close()
            else:
                # Short exit conditions
                if (crossover(self.ema_fast, self.ema_slow) or  # EMA crossover up
                    crossover(self.macd, self.macd_signal) or   # MACD crossover up
                    self.rsi[-1] < 20 ):  
                    self.position.close()
        else:
            # Entry conditions
            # Long entry
            if (crossover(self.ema_fast, self.ema_slow) and    # EMA crossover up
                self.rsi[-1] < 80 and                          # Not overbought
                self.macd[-1] > 0 and                          # MACD positive
                self.data.Volume[-1] > self.volume_ma[-1] and  # Volume confirmation
                self.ema50_slope[-1] > 0.05):                  # Higher timeframe uptrend
                
                sl = self.data.Close[-1] - self.atr[-1] * 1.5
                self.buy(sl=sl)
            
            # Short entry
            elif (crossover(self.ema_slow, self.ema_fast) and  # EMA crossover down
                  self.rsi[-1] > 20 and                        # Not oversold
                  self.macd[-1] < 0 and                        # MACD negative
                  self.data.Volume[-1] > self.volume_ma[-1] and # Volume confirmation
                  self.ema50_slope[-1] < -0.05):               # Higher timeframe downtrend
                
                sl = self.data.Close[-1] + self.atr[-1] * 1.5
                self.sell(sl=sl)

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

def run_backtest(symbol: str, start_date: str, end_date: str, interval: str,
                initial_balance: float = 10000, commission: float = 0.001) -> dict:
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
        
        # Run backtest
        bt = Backtest(df, MomentumStrategy, cash=initial_balance, commission=commission)
        stats = bt.run()
        
        # Plot results
        bt.plot(filename=f'temp/backtest_{symbol}_{start_date}_{end_date}.html')
        
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
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
            'buy_hold_return': stats['Buy & Hold Return [%]']
        }
        
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {str(e)}")
        return None

def main():
    """Main function to run backtests"""
    # Test parameters
    start_date = "2025-05-04"
    end_date = "2025-06-07"
    interval = Client.KLINE_INTERVAL_5MINUTE
    symbols = {
        "SOLUSDT",
        "ETHUSDT",
        # 'LINKUSDT',
        # 'MKRUSDT',
        # 'LTCUSDT',
        # 'ENAUSDT'
    }
    
    results = []
    
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"BACKTEST MOMENTUM STRATEGY FOR {symbol}")
        logger.info(f"{'='*50}")
        
        result = run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            initial_balance=10000,
            commission=0.001
        )
        
        if result:
            results.append(result)
            logger.info(f"\nResults for {symbol}:")
            logger.info(f"Total trades: {result['total_trades']}")
            logger.info(f"Win rate: {result['win_rate']:.2f}%")
            logger.info(f"Return: {result['return_pct']:.2f}%")
            logger.info(f"Max drawdown: {result['max_drawdown']:.2f}%")
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'temp/backtest_summary_{start_date}_{end_date}.csv', index=False)
        logger.info(f"\nSaved results to temp/backtest_summary_{start_date}_{end_date}.csv")

if __name__ == "__main__":
    main() 