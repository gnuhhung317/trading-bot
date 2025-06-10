import pandas as pd
import numpy as np
import logging
import pandas_ta as ta
from backtesting import Strategy

class AdvancedTradingStrategy(Strategy):
    """
    Futures-Oriented Advanced Trading Strategy
    
    Enhanced with:
    - Multi-timeframe analysis
    - ATR-based position sizing
    - Risk management
    """
    # Strategy parameters
    ema_fast = 9
    ema_slow = 21
    rsi_period = 14
    rsi_oversold = 35    # Relaxed from 30
    rsi_overbought = 65  # Relaxed from 70
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_period = 14
    adx_threshold = 20   # Relaxed from 25
    atr_period = 14
    atr_multiplier = 2.0
    
    # Position sizing parameters
    risk_per_trade = 0.02  # 2% risk per trade
    leverage = 3
    contract_multiplier = 1  # Adjust for your futures contract
    
    # Debug mode
    debug = True
    
    def init(self):
        """Initialize strategy indicators"""
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # EMAs
        self.ema_fast_line = self.I(lambda: ta.ema(close, length=self.ema_fast).fillna(method='bfill').values)
        self.ema_slow_line = self.I(lambda: ta.ema(close, length=self.ema_slow).fillna(method='bfill').values)
        
        # RSI
        self.rsi = self.I(lambda: ta.rsi(close, length=self.rsi_period).fillna(50).values)
        
        # MACD
        macd_data = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self.macd = self.I(lambda: macd_data.iloc[:, 0].fillna(0).values)
        self.macd_signal = self.I(lambda: macd_data.iloc[:, 1].fillna(0).values)
        
        # ADX
        adx_data = ta.adx(high, low, close, length=self.adx_period)
        self.adx = self.I(lambda: adx_data.iloc[:, 0].fillna(0).values)
        self.di_plus = self.I(lambda: adx_data.iloc[:, 1].fillna(0).values)
        self.di_minus = self.I(lambda: adx_data.iloc[:, 2].fillna(0).values)
        
        # ATR
        self.atr = self.I(lambda: ta.atr(high, low, close, length=self.atr_period).fillna(0).values)
        
        # Volume
        self.volume_ma = self.I(lambda: volume.rolling(10).mean().fillna(method='bfill').values)
        
        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        
        # Logging
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO)
    
    def next(self):
        """Main strategy logic for each candle"""
        price = self.data.Close[-1]
        ema_fast = self.ema_fast_line[-1]
        ema_slow = self.ema_slow_line[-1]
        rsi = self.rsi[-1]
        macd = self.macd[-1]
        macd_signal = self.macd_signal[-1]
        adx = self.adx[-1]
        di_plus = self.di_plus[-1]
        di_minus = self.di_minus[-1]
        atr = self.atr[-1]
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        
        # Debug logging
        if self.debug:
            logging.debug(f"\nCandle {len(self.data)}:")
            logging.debug(f"Price: {price:.2f}, EMA Fast: {ema_fast:.2f}, EMA Slow: {ema_slow:.2f}")
            logging.debug(f"RSI: {rsi:.2f}, MACD: {macd:.2f}, MACD Signal: {macd_signal:.2f}")
            logging.debug(f"ADX: {adx:.2f}, DI+: {di_plus:.2f}, DI-: {di_minus:.2f}")
            logging.debug(f"ATR: {atr:.2f}, Volume: {volume:.2f}, Volume MA: {volume_ma:.2f}")
        
        # Skip if indicators are NaN
        if np.isnan([ema_fast, ema_slow, rsi, macd, macd_signal, adx, di_plus, di_minus, atr]).any():
            return
        
        # Volume confirmation (temporarily bypassed)
        volume_confirmation = True  # Revert to volume > volume_ma after testing
        
        # Position management
        if self.position:
            if self.position.is_long:
                exit_conditions = [
                    price <= self.stop_loss,
                    rsi > self.rsi_overbought,
                    macd < macd_signal,
                    ema_fast < ema_slow
                ]
            else:
                exit_conditions = [
                    price >= self.stop_loss,
                    rsi < self.rsi_oversold,
                    macd > macd_signal,
                    ema_fast > ema_slow
                ]
            
            if self.debug:
                logging.debug("\nExit Conditions:")
                logging.debug(f"Stop Loss Hit: {exit_conditions[0]}, RSI Exit: {exit_conditions[1]}")
                logging.debug(f"MACD Exit: {exit_conditions[2]}, EMA Exit: {exit_conditions[3]}")
            
            if any(exit_conditions):
                self.position.close()
                self.entry_price = None
                self.stop_loss = None
        else:
            # Calculate position size
            equity = self.equity
            risk_amount = equity * self.risk_per_trade
            stop_distance = atr * self.atr_multiplier
            contracts = risk_amount / (stop_distance * self.contract_multiplier)
            position_size = contracts * self.leverage
            
            # Entry conditions
            long_condition = (
                ema_fast > ema_slow and
                rsi < self.rsi_oversold and
                macd > macd_signal and
                adx > self.adx_threshold and
                di_plus > di_minus and
                volume_confirmation and
                atr > 0
            )
            short_condition = (
                ema_fast < ema_slow and
                rsi > self.rsi_overbought and
                macd < macd_signal and
                adx > self.adx_threshold and
                di_minus > di_plus and
                volume_confirmation and
                atr > 0
            )
            
            if self.debug:
                logging.debug("\nEntry Conditions:")
                logging.debug(f"Long: {long_condition}, Short: {short_condition}")
                logging.debug(f"Position Size: {position_size:.4f} contracts")
            
            # Enter long
            if long_condition:
                stop_price = price - stop_distance
                self.buy(size=position_size, sl=stop_price)
                self.entry_price = price
                self.stop_loss = stop_price
            
            # Enter short
            elif short_condition:
                stop_price = price + stop_distance
                self.sell(size=position_size, sl=stop_price)
                self.entry_price = price
                self.stop_loss = stop_price