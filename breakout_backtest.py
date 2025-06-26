import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
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

def is_bullish_engulfing(open_prices, high_prices, low_prices, close_prices):
    """Detect Bullish Engulfing pattern"""
    if len(open_prices) < 2:
        return False
    prev_open = open_prices[-2]
    prev_close = close_prices[-2]
    curr_open = open_prices[-1]
    curr_close = close_prices[-1]
    
    is_prev_bearish = prev_close < prev_open
    is_curr_bullish = curr_close > curr_open
    is_engulfing = (curr_open <= prev_close) and (curr_close >= prev_open)
    
    return is_prev_bearish and is_curr_bullish and is_engulfing

def detect_resistance_breakout(high_prices, lookback=20):
    """Detect breakout above resistance (highest high in lookback period)"""
    if len(high_prices) < lookback:
        return False
    resistance = np.max(high_prices[-lookback:-1])
    return high_prices[-1] > resistance

class BreakoutEngulfingStrategy(Strategy):
    # Parameters
    rsi_period = 14
    rsi_threshold = 50  # RSI > 50 for bullish momentum
    sma_period = 50  # SMA for trend confirmation
    stop_loss_pct = 0.03  # 3% stop-loss
    take_profit_pct = 0.06  # 6% take-profit
    hold_period = 8  # Max holding period (32 hours on 4h)
    volume_ma_period = 20  # Volume MA period
    breakout_lookback = 20  # Lookback for resistance

    def init(self):
        super().init()
        self.entry_index = None
        self.entry_price = None
        
        # Calculate Bullish Engulfing
        self.bullish_engulfing = self.I(
            lambda: np.where(
                [is_bullish_engulfing(self.data.Open[i:], self.data.High[i:], 
                                      self.data.Low[i:], self.data.Close[i:]) 
                 for i in range(len(self.data.Open))],
                100, 0
            ),
            name='BullishEngulfing'
        )
        
        # Calculate Resistance Breakout
        self.breakout = self.I(
            lambda: np.where(
                [detect_resistance_breakout(self.data.High[i:], self.breakout_lookback) 
                 for i in range(len(self.data.High))],
                100, 0
            ),
            name='Breakout'
        )
        
        # Calculate RSI manually
        def calculate_rsi(prices, period=14):
            prices = np.array(prices)
            if len(prices) < period + 1:
                return np.full_like(prices, np.nan)
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            rsi = np.zeros_like(prices)
            rsi[:period] = np.nan
            
            if avg_loss == 0:
                rsi[period] = 100 if avg_gain > 0 else 50
            else:
                rs = avg_gain / avg_loss
                rsi[period] = 100 - (100 / (1 + rs))
            
            for i in range(period + 1, len(prices)):
                avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
                if avg_loss == 0:
                    rsi[i] = 100 if avg_gain > 0 else 50
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
            
            return rsi

        self.rsi = self.I(lambda: calculate_rsi(self.data.Close, self.rsi_period), name='RSI')
        
        # Calculate SMA manually
        def calculate_sma(prices, period):
            sma = np.zeros_like(prices)
            sma[:period-1] = np.nan
            for i in range(period-1, len(prices)):
                sma[i] = np.mean(prices[i-period+1:i+1])
            return sma
            
        self.sma = self.I(lambda: calculate_sma(self.data.Close, self.sma_period), name='SMA')
        
        # Calculate Volume MA
        def calculate_volume_ma(prices, period):
            vol_ma = np.zeros_like(prices)
            vol_ma[:period-1] = np.nan
            for i in range(period-1, len(prices)):
                vol_ma[i] = np.mean(prices[i-period+1:i+1])
            return vol_ma
            
        self.volume_ma = self.I(lambda: calculate_volume_ma(self.data.Volume, self.volume_ma_period), name='VolumeMA')

    def next(self):
        price = self.data.Close[-1]
        
        # Skip if any indicator is NaN
        if np.isnan(self.rsi[-1]) or np.isnan(self.sma[-1]) or np.isnan(self.volume_ma[-1]):
            return
        
        # Debug log
        logger.debug(f"Engulfing: {self.bullish_engulfing[-1]}, Breakout: {self.breakout[-1]}, RSI: {self.rsi[-1]:.2f}, Price: {price:.2f}, SMA: {self.sma[-1]:.2f}, Volume: {self.data.Volume[-1]:.2f}, VolumeMA: {self.volume_ma[-1]:.2f}")
        
        # Entry condition: Bullish Engulfing, Breakout, RSI > 50, price above SMA, volume above MA
        if (not self.position and 
            self.bullish_engulfing[-1] == 100 and 
            self.breakout[-1] == 100 and
            self.rsi[-1] > self.rsi_threshold and 
            price > self.sma[-1] and
            self.data.Volume[-1] > self.volume_ma[-1]):
            self.buy(size=0.02)  # 2% of available capital
            self.entry_index = len(self.data) - 1
            self.entry_price = price
            self.position.sl = price * (1 - self.stop_loss_pct)
            self.position.tp = price * (1 + self.take_profit_pct)
            logger.info(f"Buy signal at {price:.2f}, SL: {self.position.sl:.2f}, TP: {self.position.tp:.2f}")
        
        # Exit condition: Hold for 8 candles or hit stop-loss/take-profit
        elif self.position and (len(self.data) - 1 - self.entry_index) >= self.hold_period:
            self.sell(size=0.02)
            logger.info(f"Sell signal at {price:.2f} after {self.hold_period} candles")

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
        logger.error(f"No klines data retrieved for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                     'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                                     'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    
    # Clean data
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df[df['volume'] > 0]
    if df.empty:
        logger.error(f"Cleaned data is empty for {symbol}")
        return pd.DataFrame()
    
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    df.set_index('timestamp', inplace=True)
    logger.info(f"Data sample:\n{df.head()}")
    return df

def run_breakout_engulfing_backtest(symbol: str, interval: str, start_date: str, end_date: str, 
                                   initial_balance: float = 10000, commission: float = 0.0001) -> dict:
    """Run backtest for BreakoutEngulfingStrategy"""
    try:
        client = Client()
        df = get_historical_data(client, symbol, interval, start_date, end_date)
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return None
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        bt = Backtest(df, BreakoutEngulfingStrategy, cash=initial_balance, commission=commission)
        stats = bt.run()
        
        bt.plot(filename=f'backtest_{symbol}_breakout_engulfing.html')

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
    """Main function to run Breakout Engulfing backtest"""
    symbol = "SOLUSDT"
    logger.info(f"\n{'='*50}")
    logger.info(f"RUNNING BREAKOUT ENGULFING STRATEGY BACKTEST FOR {symbol}")
    logger.info(f"{'='*50}")
    
    result = run_breakout_engulfing_backtest(
        symbol=symbol,
        interval="4h",
        start_date="2024-01-01",
        end_date="2024-06-30"
    )
    
    if result:
        logger.info("\nBacktest Results:")
        for key, value in result.items():
            if key != 'strategy_params':
                logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()