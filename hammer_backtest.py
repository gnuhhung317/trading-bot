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

def is_hammer(open_price, high, low, close, body_threshold=0.3, lower_shadow_ratio=2.0):
    """Custom function to detect hammer pattern"""
    body = abs(close - open_price)
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low
    
    is_small_body = body <= (high - low) * body_threshold
    has_long_lower_shadow = lower_shadow >= body * lower_shadow_ratio
    has_small_upper_shadow = upper_shadow <= body * 0.3
    
    return is_small_body and has_long_lower_shadow and has_small_upper_shadow

def calculate_rsi(close, period=14):
    """Calculate RSI indicator"""
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    return pd.Series(close).rolling(window=period).mean()

class EnhancedHammerStrategy(Strategy):
    # Parameters
    rsi_period = 14
    rsi_oversold = 20
    rsi_overbought = 70
    sma_period = 100
    stop_loss_pct = 0.01  # 1% stop-loss
    take_profit_pct = 0.02  # 2% take-profit
    risk_per_trade = 0.02  # Risk 2% of equity per trade

    def init(self):
        super().init()
        self.entry_bar = None
        self.entry_price = None
        
        # Calculate indicators
        self.hammer = self.I(
            lambda: np.where(
                is_hammer(self.data.Open, self.data.High, self.data.Low, self.data.Close),
                100, 0
            ),
            name='Hammer'
        )
        
        # Calculate RSI
        self.rsi = self.I(
            lambda x: calculate_rsi(x, self.rsi_period),
            self.data.Close,
            name='RSI'
        )
        
        # Calculate SMA
        self.sma = self.I(
            lambda x: calculate_sma(x, self.sma_period),
            self.data.Close,
            name='SMA'
        )

    def calculate_position_size(self, price, stop_loss):
        """Calculate position size based on risk per trade"""
        risk_amount = self.equity * self.risk_per_trade
        stop_distance = abs(price - stop_loss)
        if stop_distance == 0:
            return 0
        position_size = risk_amount / stop_distance
        # Ensure we don't exceed available margin
        max_position = self.equity / price
        return min(position_size, max_position)

    def next(self):
        price = self.data.Close[-1]
        current_bar = len(self.data.Close) - 1
        
        # Long entry conditions
        long_conditions = (
            not self.position and 
            self.hammer[-1] == 100 and 
            self.rsi[-1] <= self.rsi_oversold and 
            price > self.sma[-1]
        )
        
        # Short entry conditions
        short_conditions = (
            not self.position and 
            self.hammer[-1] == 100 and 
            self.rsi[-1] >= self.rsi_overbought and 
            price < self.sma[-1]
        )
        
        # Entry logic
        if long_conditions:
            stop_loss = price * (1 - self.stop_loss_pct)
            position_size = self.calculate_position_size(price, stop_loss)
            
            if position_size > 0:
                self.buy(size=0.5)
                self.entry_bar = current_bar
                self.entry_price = price
                self.position.sl = stop_loss
                self.position.tp = price * (1 + self.take_profit_pct)
                
        elif short_conditions:
            stop_loss = price * (1 + self.stop_loss_pct)
            position_size = self.calculate_position_size(price, stop_loss)
            
            if position_size > 0:
                self.sell(size=0.5)
                self.entry_bar = current_bar
                self.entry_price = price
                self.position.sl = stop_loss
                self.position.tp = price * (1 - self.take_profit_pct)
        
        # Exit conditions for existing positions
        elif self.position:
            # Exit long position if RSI becomes overbought
            if self.position.is_long and self.rsi[-1] >= self.rsi_overbought:
                self.position.close()
            # Exit short position if RSI becomes oversold
            elif self.position.is_short and self.rsi[-1] <= self.rsi_oversold:
                self.position.close()

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
    
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    df.set_index('timestamp', inplace=True)
    return df

def main():
    """Main function to run Enhanced Hammer backtest"""
    symbol = "SOLUSDT"
    logger.info(f"\n{'='*50}")
    logger.info(f"OPTIMIZING ENHANCED HAMMER STRATEGY FOR {symbol}")
    logger.info(f"{'='*50}")

    # Initialize Binance client
    client = Client()
    
    # Get historical data
    df = get_historical_data(client, symbol, "15m", "2023-06-01", "2025-05-30")
    if df.empty:
        logger.error(f"No data available for {symbol}")
        return
        
    # Prepare data for backtesting
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Run optimization
    bt = Backtest(df, EnhancedHammerStrategy, cash=10000, commission=0.0001)
    
    # Define parameter ranges for optimization
    optimization_params = {
        'rsi_period': range(10, 21, 2),        # [10, 12, 14, 16, 18, 20]
        'rsi_oversold': range(20, 41, 5),      # [20, 25, 30, 35, 40]
        'rsi_overbought': range(60, 81, 5),    # [60, 65, 70, 75, 80]
        # 'sma_period': range(100, 301, 20),      # [20, 40, 60, 80, 100]
        # 'stop_loss_pct': [0.005, 0.01, 0.015, 0.02, 0.025],  # [0.5%, 1%, 1.5%, 2%, 2.5%]
        # 'take_profit_pct': [0.01, 0.015, 0.02, 0.025, 0.03]  # [1%, 1.5%, 2%, 2.5%, 3%]
    }

    # Run optimization
    stats = bt.optimize(
        **optimization_params,
        maximize='Return [%]',  # Optimize for maximum return
        method='grid'          # Use grid search method
    )

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