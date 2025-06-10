"""
Bollinger Bands Strategy for Futures Trading
"""

import pandas as pd
import numpy as np
import logging
import pandas_ta as ta
from backtesting import Strategy

class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Strategy with volume confirmation and ATR-based stops
    """
    
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    volume_period = 20
    volume_threshold = 1.0
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
        
        # Bollinger Bands
        bb_data = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        self.bb_upper = self.I(lambda: bb_data.iloc[:, 0].fillna(method='bfill').values)
        self.bb_middle = self.I(lambda: bb_data.iloc[:, 1].fillna(method='bfill').values)
        self.bb_lower = self.I(lambda: bb_data.iloc[:, 2].fillna(method='bfill').values)
        
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
        bb_upper = self.bb_upper[-1]
        bb_middle = self.bb_middle[-1]
        bb_lower = self.bb_lower[-1]
        atr = self.atr[-1]
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        
        # Debug logging
        if self.debug:
            logging.debug(f"\nCandle {len(self.data)}:")
            logging.debug(f"Price: {price:.2f}")
            logging.debug(f"BB Upper: {bb_upper:.2f}")
            logging.debug(f"BB Middle: {bb_middle:.2f}")
            logging.debug(f"BB Lower: {bb_lower:.2f}")
            logging.debug(f"ATR: {atr:.2f}")
            logging.debug(f"Volume: {volume:.2f}")
            logging.debug(f"Volume MA: {volume_ma:.2f}")
        
        # Check for NaN values
        if np.isnan(bb_upper) or np.isnan(bb_middle) or np.isnan(bb_lower) or np.isnan(atr):
            return
        
        # Volume confirmation
        volume_confirmation = volume > volume_ma * self.volume_threshold
        
        # Position management
        if self.position:
            # Exit conditions
            if self.position.is_long:
                exit_conditions = [
                    price <= self.stop_loss,
                    price >= bb_upper,  # Exit when price hits upper band
                    price < bb_middle  # Exit when price breaks below middle band
                ]
            else:  # Short position
                exit_conditions = [
                    price >= self.stop_loss,
                    price <= bb_lower,  # Exit when price hits lower band
                    price > bb_middle  # Exit when price breaks above middle band
                ]
            
            # Debug logging for exit conditions
            if self.debug:
                logging.debug("\nExit Conditions:")
                logging.debug(f"Stop Loss Hit: {price <= self.stop_loss if self.position.is_long else price >= self.stop_loss}")
                logging.debug(f"BB Exit: {price >= bb_upper if self.position.is_long else price <= bb_lower}")
                logging.debug(f"Middle Band Exit: {price < bb_middle if self.position.is_long else price > bb_middle}")
            
            # Close position if any exit condition is met
            if any(exit_conditions):
                self.position.close()
                self.entry_price = None
                self.stop_loss = None
        else:
            # Entry conditions with volume confirmation
            long_condition = (
                price <= bb_lower and  # Price touches lower band
                volume_confirmation and
                atr > 0
            )
            
            short_condition = (
                price >= bb_upper and  # Price touches upper band
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
