import pandas as pd
import numpy as np

import logging
from binance import Client
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    filename="waverider.log",
    encoding="utf-8",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define trading pairs and their parameters
COINS = [
    {
        'symbol': 'SOLUSDT',
        'leverage': 50,
        'quantity_precision': 1,
        'min_qty': 0.1,
        'max_qty': 100,
        'price_precision': 2,
    },
    {
        'symbol': 'ETHUSDT',
        'leverage': 50,
        'quantity_precision': 3,
        'min_qty': 0.01,
        'max_qty': 100,
        'price_precision': 2,
    }
]

class WaveRiderStrategy:
    def __init__(self, client, coin_params):
        self.client = client
        self.coin_params = coin_params
        self.symbol = coin_params['symbol']
        self.quantity_precision = coin_params["quantity_precision"]
        self.leverage = coin_params['leverage']
        self.price_precision = coin_params['price_precision']
        
        # Strategy parameters
        self.volume_ma_length = 10
        self.volume_threshold = 2
        self.momentum_length = 5
        self.atr_length = 10
        self.atr_sl_multiplier = 2
        self.atr_tp_multiplier = 3.0
        self.risk_per_trade = 0.01
        self.min_volume_increase = 1.9
        self.rsi_length = 20
        self.rsi_overbought = 75
        self.rsi_oversold = 15
        self.sma_length = 100
        self.max_holding_period = 20

        # Initialize indicators
        self.volume_ma = None
        self.momentum = None
        self.atr = None
        self.price_velocity = None
        self.volume_acceleration = None
        self.rsi = None
        self.sma200 = None
        
        # Set leverage
        self.set_leverage()

    def set_leverage(self):
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=self.leverage
            )
            logger.info(f"Set leverage to {self.leverage}x for {self.symbol}")
        except Exception as e:
            logger.error(f"Error setting leverage: {str(e)}")

    def calculate_indicators(self, df):
        # Volume MA
        self.volume_ma = df['volume'].rolling(window=self.volume_ma_length).mean()

        # Momentum
        self.momentum = df['close'].pct_change(self.momentum_length) * 100

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.atr = tr.rolling(window=self.atr_length).mean()

        # Price velocity
        self.price_velocity = df['close'].pct_change(3) * 100

        # Volume acceleration
        self.volume_acceleration = df['volume'].pct_change(3) * 100

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_length).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_length).mean()
        rs = gain / loss
        self.rsi = 100 - (100 / (1 + rs))

        # SMA
        self.sma200 = df['close'].rolling(window=self.sma_length).mean()

    def check_entry_conditions(self, df, index):
        current_volume = df['volume'].iloc[index]
        current_volume_ma = self.volume_ma.iloc[index]
        current_momentum = self.momentum.iloc[index]
        current_velocity = self.price_velocity.iloc[index]
        current_volume_acc = self.volume_acceleration.iloc[index]
        prev_volume = df['volume'].iloc[index-1]
        current_rsi = self.rsi.iloc[index]
        current_price = df['close'].iloc[index]
        current_sma200 = self.sma200.iloc[index]

        momentum_threshold = self.atr.iloc[index] * 0.05

        # Long conditions
        long_conditions = [
            current_volume > current_volume_ma * self.volume_threshold,
            current_volume > prev_volume * self.min_volume_increase,
            current_momentum > momentum_threshold,
            current_velocity > 0,
            current_volume_acc > 0,
            current_rsi < self.rsi_overbought,
            current_price > current_sma200 * 0.995
        ]

        # Short conditions
        short_conditions = [
            current_volume > current_volume_ma * self.volume_threshold,
            current_volume > prev_volume * self.min_volume_increase,
            current_momentum < -momentum_threshold,
            current_velocity < 0,
            current_volume_acc > 0,
            current_rsi > self.rsi_oversold,
            current_price < current_sma200 * 1.005
        ]

        return all(long_conditions), all(short_conditions)

    def calculate_position_size(self, current_price, stop_distance):
        try:
            # Get account balance
            account = self.client.futures_account_balance()
            usdt_balance = float(next((item['balance'] for item in account if item['asset'] == 'USDT'), 0))
            
            position_size = usdt_balance*0.3
            
            # Apply quantity constraints
            position_size = round(position_size, self.coin_params['quantity_precision'])
            position_size = max(min(position_size, self.coin_params['max_qty']), self.coin_params['min_qty'])
            
            position_size = position_size*self.leverage / current_price
            position_size = self.round_to_precision(position_size)
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return None

    def execute_trade(self, side, quantity, price, stop_loss, take_profit):
        stop_loss = self.round_to_precision(stop_loss,'price')
        take_profit= self.round_to_precision(take_profit,'price')
        try:
            # Place main order
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Place stop loss
            self.client.futures_create_order(
                symbol=self.symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                type='STOP_MARKET',
                stopPrice=stop_loss,
                closePosition=True
            )
            
            # Place take profit
            self.client.futures_create_order(
                symbol=self.symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit,
                closePosition=True
            )
            
            logger.info(f"{side} {quantity} {self.symbol} at {price}")
            return True
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False

    def check_exit_conditions(self, df, index, position):
        if not position:
            return False

        current_price = df['close'].iloc[index]
        current_volume = df['volume'].iloc[index]
        current_volume_ma = self.volume_ma.iloc[index]
        current_momentum = self.momentum.iloc[index]
        momentum_threshold = self.atr.iloc[index] * 0.15  # Increased from 0.1 to reduce sensitivity
        atr = self.atr.iloc[index]
        entry_price = float(position['entryPrice'])
        
        position_amt = float(position['positionAmt'])
        is_long = position_amt > 0

        # Trailing stop logic
        trailing_stop_price = self.trailing_stop(entry_price, current_price, atr)
        if is_long and current_price < trailing_stop_price:
            return True
        elif not is_long and current_price > trailing_stop_price:
            return True

        # Additional exit conditions with trend confirmation
        # Check momentum over the last 3 candles to confirm reversal
        try:
            recent_momentum = self.momentum.iloc[index-2:index+1].values
            if is_long and np.all(recent_momentum < -momentum_threshold):  # Consistent reversal
                return True
            elif not is_long and np.all(recent_momentum > momentum_threshold):
                return True
        except (IndexError, ValueError):
            if is_long and current_momentum < -momentum_threshold:
                return True
            elif not is_long and current_momentum > momentum_threshold:
                return True

        # Relaxed volume condition
        if current_volume < current_volume_ma * 0.7:  # Increased from 0.5 to 0.7
            return True

        try:
        # Check holding period as a soft limit (log warning but don't force exit)
            if position["updateTime"] and (datetime.now() - self.entry_time).total_seconds() / 300 > self.max_holding_period:
                logger.warning(f"Trade for {self.symbol} exceeded max holding period of {self.max_holding_period} candles")
                return False  # Allow trade to continue unless other conditions trigger
        except Exception as e:
            logger.error("Exit error: ",e)
        return False
    def round_to_precision(self,size, value_type='quantity'):
        if value_type == 'quantity':
            precision = self.quantity_precision
        elif value_type == 'price':
            precision = self.price_precision
        rounded_value = round(size, precision)
        if precision == 0:
            rounded_value = int(rounded_value)
        return rounded_value

def get_historical_data(client, symbol, interval, limit=100):
    try:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                         'taker_buy_quote', 'ignored'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return None

def main():
    # Initialize Binance client
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("API credentials not found")
        return

    client = Client(api_key, api_secret)
    
    # Initialize strategies for each coin
    strategies = {}
    for coin in COINS:
        strategies[coin['symbol']] = WaveRiderStrategy(client, coin)
    
    logger.info("Starting trading bot...")
    
    while True:
        try:
            for symbol, strategy in strategies.items():
                # Get latest data
                df = get_historical_data(client, symbol, "5m")
                if df is None or df.empty:
                    continue
                
                # Calculate indicators
                strategy.calculate_indicators(df)
                
                # Check for open positions
                positions = client.futures_position_information(symbol=symbol)
                current_position = next((pos for pos in positions if float(pos['positionAmt']) != 0), None)
                
                # Check exit conditions
                if current_position and strategy.check_exit_conditions(df, -1, current_position):
                    try:
                        position_amt = float(current_position['positionAmt'])
                        position_amt = strategy.round_to_precision(position_amt)
                        close_side = 'SELL' if position_amt > 0 else 'BUY'
                        client.futures_create_order(
                            symbol=symbol,
                            side=close_side,
                            type='MARKET',
                            quantity=abs(position_amt)
                        )
                        logger.info(f"Closed position for {symbol}")
                    except Exception as e:
                        logger.error(f"Error closing position: {str(e)}")
                
                # Check entry conditions
                if not current_position:
                    long_entry, short_entry = strategy.check_entry_conditions(df, -1)
                    if long_entry or short_entry:
                        current_price = df['close'].iloc[-1]
                        stop_distance = strategy.atr.iloc[-1] * strategy.atr_sl_multiplier
                        
                        # Calculate position size
                        quantity = strategy.calculate_position_size(current_price, stop_distance)
                        if quantity is None:
                            continue
                        
                        # Calculate stop loss and take profit
                        if long_entry:
                            stop_loss = current_price - stop_distance
                            take_profit = current_price + (strategy.atr.iloc[-1] * strategy.atr_tp_multiplier)
                            strategy.execute_trade('BUY', quantity, current_price, stop_loss, take_profit)
                        else:
                            stop_loss = current_price + stop_distance
                            take_profit = current_price - (strategy.atr.iloc[-1] * strategy.atr_tp_multiplier)
                            strategy.execute_trade('SELL', quantity, current_price, stop_loss, take_profit)
            
            # Wait for next candle
            time.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    main() 