import pandas as pd
import numpy as np
import logging
import pandas_ta as ta
from backtesting import Strategy

class FalseBreakoutStochasticStrategy(Strategy):
    """
    False Breakout with Stochastic Strategy for GBP/USD (H4)
    
    Based on: [Case Study] Giao dịch False Breakout với Stochastic
    Enhanced with:
    - Stochastic-based entry (K=5, D=3, Smooth=3)
    - False breakout detection
    - ATR-based risk management
    - 3x leverage futures trading
    
    Run backtest with:
        bt = Backtest(data, FalseBreakoutStochasticStrategy, cash=10000, margin=1/3, commission=0.001)
    """
    # Strategy parameters
    stochastic_k = 5
    stochastic_d = 3
    stochastic_smooth = 3
    oversold = 20
    overbought = 80
    swing_lookback = 20  # Lookback for swing high/low detection
    atr_period = 14
    atr_multiplier = 2.0
    risk_per_trade = 0.02  # 2% risk per trade
    leverage = 3
    contract_multiplier = 100000  # GBP/USD futures (standard lot size)
    
    # Debug mode
    debug = True
    
    def init(self):
        """Initialize strategy indicators"""
        # Convert data to pandas Series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Calculate Stochastic
        stoch = ta.stoch(high, low, close, k=self.stochastic_k, d=self.stochastic_d, smooth_k=self.stochastic_smooth)
        # Fill NaN values with 50 and ensure same length as data
        stoch_k = np.full(len(self.data), 50.0)  # Initialize with 50s
        stoch_d = np.full(len(self.data), 50.0)  # Initialize with 50s
        
        # Fill in actual values where available
        valid_stoch_k = stoch.iloc[:, 0].fillna(50).values
        valid_stoch_d = stoch.iloc[:, 1].fillna(50).values
        stoch_k[:len(valid_stoch_k)] = valid_stoch_k
        stoch_d[:len(valid_stoch_d)] = valid_stoch_d
        
        # Calculate ATR
        atr = np.full(len(self.data), 0.0)  # Initialize with 0s
        valid_atr = ta.atr(high, low, close, length=self.atr_period).fillna(0).values
        atr[:len(valid_atr)] = valid_atr
        
        # Calculate Swing High/Low
        swing_high = np.full(len(self.data), high.iloc[0])  # Initialize with first high
        swing_low = np.full(len(self.data), low.iloc[0])    # Initialize with first low
        
        valid_swing_high = high.rolling(self.swing_lookback).max().shift(1).fillna(method='bfill').values
        valid_swing_low = low.rolling(self.swing_lookback).min().shift(1).fillna(method='bfill').values
        
        swing_high[:len(valid_swing_high)] = valid_swing_high
        swing_low[:len(valid_swing_low)] = valid_swing_low
        
        # Assign indicators
        self.stoch_k = self.I(lambda: stoch_k)
        self.stoch_d = self.I(lambda: stoch_d)
        self.atr = self.I(lambda: atr)
        self.swing_high = self.I(lambda: swing_high)
        self.swing_low = self.I(lambda: swing_low)
        
        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
        # Logging
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO)
    
    def next(self):
        """Main strategy logic for each candle"""
        # Initialize condition variables
        long_condition = False
        short_condition = False
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        prev_high = self.data.High[-2] if len(self.data.High) > 1 else high
        prev_low = self.data.Low[-2] if len(self.data.Low) > 1 else low
        stoch_k = self.stoch_k[-1]
        stoch_d = self.stoch_d[-1]
        prev_stoch_k = self.stoch_k[-2] if len(self.stoch_k) > 1 else stoch_k
        prev_stoch_d = self.stoch_d[-2] if len(self.stoch_d) > 1 else stoch_d
        atr = self.atr[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        
        # Skip if NaN
        if np.isnan([stoch_k, stoch_d, atr, swing_high, swing_low]).any():
            return
        
        # Debug logging
        if self.debug:
            logging.debug(f"\nCandle {len(self.data)}:")
            logging.debug(f"Price: {price:.5f}, High: {high:.5f}, Low: {low:.5f}")
            logging.debug(f"Stoch K: {stoch_k:.2f}, Stoch D: {stoch_d:.2f}")
            logging.debug(f"Swing High: {swing_high:.5f}, Swing Low: {swing_low:.5f}")
            logging.debug(f"ATR: {atr:.5f}")
        
        # Detect Oversold/Overbought conditions (relaxed)
        oversold_condition = stoch_d < self.oversold
        overbought_condition = stoch_d > self.overbought
        
        # Detect Bullish Hook (relaxed conditions)
        bullish_hook = (
            stoch_k < stoch_d and  # K below D
            prev_stoch_k < prev_stoch_d and  # Previous K below D
            stoch_k > prev_stoch_k  # K is rising
        )
        
        # Detect Bearish Hook (relaxed conditions)
        bearish_hook = (
            stoch_k > stoch_d and  # K above D
            prev_stoch_k > prev_stoch_d and  # Previous K above D
            stoch_k < prev_stoch_k  # K is falling
        )
        
        # Detect False Breakout (simplified)
        bullish_div = low < swing_low and price > swing_low  # Price breaks support and returns above
        bearish_div = high > swing_high and price < swing_high  # Price breaks resistance and returns below
        
        # Detect trend (simplified)
        uptrend = high > prev_high  # Higher high
        downtrend = low < prev_low   # Lower low
        
        # Position management
        if self.position:
            if self.position.is_long and price >= self.take_profit:
                self.position.close()
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
            elif self.position.is_short and price <= self.take_profit:
                self.position.close()
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
        
        # Entry conditions
        else:
            # Calculate position size
            equity = self.equity
            risk_amount = equity * self.risk_per_trade
            stop_distance = atr * self.atr_multiplier
            contracts = risk_amount / (stop_distance * self.contract_multiplier)
            position_size = contracts * self.leverage
            
            # Long entry (relaxed conditions)
            long_condition = (
                oversold_condition and  # Oversold condition
                (bullish_hook or bullish_div) and  # Either hook or false breakout
                uptrend  # Confirming trend
            )
            
            if long_condition:
                entry_price = price  # Entry at current price
                stop_price = price - (atr * self.atr_multiplier)  # ATR-based stop
                take_price = entry_price + (2 * (entry_price - stop_price))  # 2:1 reward ratio
                self.buy(size=position_size, sl=stop_price)
                self.entry_price = entry_price
                self.stop_loss = stop_price
                self.take_profit = take_price
                if self.debug:
                    logging.debug(f"Long Entry: Price={entry_price:.5f}, SL={stop_price:.5f}, TP={take_price:.5f}, Size={position_size:.4f}")
            
            # Short entry (relaxed conditions)
            short_condition = (
                overbought_condition and  # Overbought condition
                (bearish_hook or bearish_div) and  # Either hook or false breakout
                downtrend  # Confirming trend
            )
            
            if short_condition:
                entry_price = price  # Entry at current price
                stop_price = price + (atr * self.atr_multiplier)  # ATR-based stop
                take_price = entry_price - (2 * (stop_price - entry_price))  # 2:1 reward ratio
                self.sell(size=position_size, sl=stop_price)
                self.entry_price = entry_price
                self.stop_loss = stop_price
                self.take_profit = take_price
                if self.debug:
                    logging.debug(f"Short Entry: Price={entry_price:.5f}, SL={stop_price:.5f}, TP={take_price:.5f}, Size={position_size:.4f}")
        
        # Debug entry conditions
        if self.debug:
            logging.debug(f"Long Condition: {long_condition}, Short Condition: {short_condition}")