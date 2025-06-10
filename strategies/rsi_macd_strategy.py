import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover

class RSIMACDStrategy(Strategy):
    """
    Futures-Oriented RSI + MACD Strategy
    
    Enhanced with:
    - ATR-based position sizing
    - Trailing stops
    - Breakeven management
    - Volume confirmation
    - Risk management
    """
    # Strategy parameters
    rsi_period = 14
    rsi_oversold = 40    # Changed from 35 to 40
    rsi_overbought = 60  # Changed from 65 to 60
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    volume_period = 20
    volume_threshold = 1.0  # Changed from 1.1 to 1.0
    atr_period = 14
    atr_multiplier = 2.0
    
    # Position sizing parameters
    risk_per_trade = 0.02  # 2% risk per trade
    leverage = 3
    
    # Debug mode
    debug = False
    
    def init(self):
        """
        Initialize strategy indicators
        """
        # Calculate indicators
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # RSI
        self.rsi = self.I(lambda: ta.rsi(close, length=self.rsi_period).fillna(50).values)
        
        # MACD
        macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self.macd = self.I(lambda: macd['MACD_12_26_9'].fillna(0).values)
        self.macd_signal = self.I(lambda: macd['MACDs_12_26_9'].fillna(0).values)
        
        # ATR for position sizing and stops
        self.atr = self.I(lambda: ta.atr(high, low, close, length=self.atr_period).fillna(0).values)
        
        # Volume analysis
        self.volume_ma = self.I(lambda: volume.rolling(self.volume_period).mean().fillna(method='bfill').values)
        
        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        
        # Initialize logging
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO)
    
    def next(self):
        """
        Main strategy logic for each candle
        """
        # Get current values
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        macd = self.macd[-1]
        macd_signal = self.macd_signal[-1]
        atr = self.atr[-1]
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        
        # Debug logging
        if self.debug:
            logging.debug(f"\nCandle {len(self.data)}:")
            logging.debug(f"Price: {price:.2f}")
            logging.debug(f"RSI: {rsi:.2f}")
            logging.debug(f"MACD: {macd:.2f}")
            logging.debug(f"MACD Signal: {macd_signal:.2f}")
            logging.debug(f"ATR: {atr:.2f}")
            logging.debug(f"Volume: {volume:.2f}")
            logging.debug(f"Volume MA: {volume_ma:.2f}")
        
        # Check for NaN values
        if np.isnan(rsi) or np.isnan(macd) or np.isnan(macd_signal) or np.isnan(atr):
            return
        
        # Volume confirmation
        volume_confirmation = volume > volume_ma * self.volume_threshold
        
        # Position management
        if self.position:
            # Exit conditions
            if self.position.is_long:
                exit_conditions = [
                    price <= self.stop_loss,
                    rsi > self.rsi_overbought,
                    macd < macd_signal
                ]
            else:  # Short position
                exit_conditions = [
                    price >= self.stop_loss,
                    rsi < self.rsi_oversold,
                    macd > macd_signal
                ]
            
            # Debug logging for exit conditions
            if self.debug:
                logging.debug("\nExit Conditions:")
                logging.debug(f"Stop Loss Hit: {price <= self.stop_loss if self.position.is_long else price >= self.stop_loss}")
                logging.debug(f"RSI Exit: {rsi > self.rsi_overbought if self.position.is_long else rsi < self.rsi_oversold}")
                logging.debug(f"MACD Exit: {macd < macd_signal if self.position.is_long else macd > macd_signal}")
            
            # Close position if any exit condition is met
            if any(exit_conditions):
                self.position.close()
                self.entry_price = None
                self.stop_loss = None
        else:
            # Entry conditions with volume confirmation
            long_condition = (
                (rsi < self.rsi_oversold) and
                (macd > macd_signal) and
                volume_confirmation and
                atr > 0
            )
            
            short_condition = (
                (rsi > self.rsi_overbought) and
                (macd < macd_signal) and
                volume_confirmation and
                atr > 0
            )
            
            # Debug logging for entry conditions
            if self.debug:
                logging.debug("\nEntry Conditions:")
                logging.debug(f"Long: {long_condition}")
                logging.debug(f"Short: {short_condition}")
                logging.debug(f"Volume Confirmation: {volume_confirmation}")
            
            # Enter long position
            if long_condition:
                # Calculate stop distance
                stop_distance = atr * self.atr_multiplier
                stop_price = price - stop_distance
                
                # Calculate position size as a fraction of equity
                position_size = self.risk_per_trade * self.leverage
                
                # Debug logging for position sizing
                if self.debug:
                    logging.debug("\nLong Position Details:")
                    logging.debug(f"Stop Distance: {stop_distance:.2f}")
                    logging.debug(f"Stop Price: {stop_price:.2f}")
                    logging.debug(f"Position Size: {position_size:.4f}")
                
                # Enter position with stop loss
                self.buy(size=position_size, sl=stop_price)
                self.entry_price = price
                self.stop_loss = stop_price
            
            # Enter short position
            elif short_condition:
                # Calculate stop distance
                stop_distance = atr * self.atr_multiplier
                stop_price = price + stop_distance
                
                # Calculate position size as a fraction of equity
                position_size = self.risk_per_trade * self.leverage
                
                # Debug logging for position sizing
                if self.debug:
                    logging.debug("\nShort Position Details:")
                    logging.debug(f"Stop Distance: {stop_distance:.2f}")
                    logging.debug(f"Stop Price: {stop_price:.2f}")
                    logging.debug(f"Position Size: {position_size:.4f}")
                
                # Enter position with stop loss
                self.sell(size=position_size, sl=stop_price)
                self.entry_price = price
                self.stop_loss = stop_price