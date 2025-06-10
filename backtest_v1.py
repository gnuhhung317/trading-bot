import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

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
    rsi_oversold = 35
    rsi_overbought = 65
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_multiplier = 2.0
    atr_multiplier_trailing = 2.0
    
    # Futures-specific parameters
    position_size_pct = 0.03  # 3% of account per trade
    risk_per_trade = 0.015  # 1.5% risk per trade
    volume_threshold = 1.1
    def init(self):
        # Calculate indicators
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # RSI
        rsi_values = ta.rsi(close, length=self.rsi_period)
        self.rsi = self.I(lambda: rsi_values.fillna(50).values)
        
        # MACD
        macd_data = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        macd_line = macd_data.iloc[:, 0].fillna(0)
        macd_signal_line = macd_data.iloc[:, 1].fillna(0)
        
        self.macd = self.I(lambda: macd_line.values)
        self.macd_signal = self.I(lambda: macd_signal_line.values)
        
        # ATR for position sizing and stops
        self.atr = self.I(lambda: ta.atr(high, low, close, length=14).fillna(0).values)
        
        # Volume analysis
        self.volume_ma = self.I(lambda: volume.rolling(10).mean().fillna(method='bfill').values)
        
        # EMA for trailing stops
        self.ema_21 = self.I(lambda: ta.ema(close, length=21).fillna(method='bfill').values)
        
        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.breakeven_activated = False    
    def next(self):
        # Current values
        if len(self.rsi) == 0 or len(self.macd) == 0:
            return
            
        current_price = self.data.Close[-1]
        rsi_current = self.rsi[-1]
        macd_current = self.macd[-1]
        macd_signal_current = self.macd_signal[-1]
        atr_current = self.atr[-1]
        volume_current = self.data.Volume[-1]
        volume_ma_current = self.volume_ma[-1]
        ema_21_current = self.ema_21[-1]
        
        # Skip if indicators are NaN or invalid
        if pd.isna(rsi_current) or pd.isna(macd_current) or pd.isna(macd_signal_current):
            return
        
        # Volume confirmation
        volume_confirmation = volume_current > volume_ma_current * self.volume_threshold
        
        # Position management
        if self.position:
            self._manage_position(current_price, ema_21_current, rsi_current, 
                                macd_current, macd_signal_current)
        else:
            # Enhanced buy signal with volume confirmation
            if (rsi_current < self.rsi_oversold and 
                macd_current > macd_signal_current and 
                volume_confirmation and
                atr_current > 0):
                
                # Calculate position size based on risk
                stop_distance = atr_current * self.atr_multiplier
                position_size = min(self.risk_per_trade / (stop_distance / current_price), 
                                  self.position_size_pct)
                
                self.buy(size=position_size)
                self.entry_price = current_price
                self.stop_loss = current_price - stop_distance
                self.breakeven_activated = False
    
    def _manage_position(self, current_price, ema_21_current, rsi_current, 
                        macd_current, macd_signal_current):
        """Enhanced position management with trailing stops and breakeven"""
        if not self.entry_price:
            return
        
        # Calculate R multiple
        if self.position.is_long:
            r_multiple = (current_price - self.entry_price) / (self.entry_price - self.stop_loss)
            
            # Breakeven after 1R profit
            if r_multiple >= 1 and not self.breakeven_activated:
                self.stop_loss = self.entry_price
                self.breakeven_activated = True
            
            # Trailing stop using EMA21 after 1R profit
            if r_multiple >= 1 and ema_21_current > self.stop_loss:
                self.stop_loss = ema_21_current
            
            # Exit conditions
            exit_conditions = [
                current_price <= self.stop_loss,
                rsi_current > self.rsi_overbought,
                macd_current < macd_signal_current and r_multiple > 0.5,
                r_multiple >= 4,  # Take profit at 4R
                r_multiple < -1.5  # Emergency stop
            ]
        else:  # Short position
            r_multiple = (self.entry_price - current_price) / (self.stop_loss - self.entry_price)
            
            # Breakeven after 1R profit
            if r_multiple >= 1 and not self.breakeven_activated:
                self.stop_loss = self.entry_price
                self.breakeven_activated = True
            
            # Trailing stop using EMA21 after 1R profit
            if r_multiple >= 1 and ema_21_current < self.stop_loss:
                self.stop_loss = ema_21_current
            
            # Exit conditions
            exit_conditions = [
                current_price >= self.stop_loss,
                rsi_current < self.rsi_oversold,
                macd_current > macd_signal_current and r_multiple > 0.5,
                r_multiple >= 4,  # Take profit at 4R
                r_multiple < -1.5  # Emergency stop
            ]
        
        if any(exit_conditions):
            self.position.close()
            self.entry_price = None
            self.stop_loss = None
            self.breakeven_activated = False

class BollingerBandsStrategy(Strategy):
    """
    Futures-Oriented Bollinger Bands Strategy
    
    Enhanced with:
    - Mean reversion and breakout detection
    - ATR-based position sizing
    - Trailing stops and breakeven
    - Volume and momentum confirmation
    """
    
    bb_period = 20
    bb_std = 2
    atr_multiplier = 2.0
    atr_multiplier_trailing = 2.0
    
    # Futures-specific parameters
    position_size_pct = 0.025  # 2.5% of account per trade
    risk_per_trade = 0.015  # 1.5% risk per trade
    rsi_period = 14
    volume_threshold = 1.2
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # Bollinger Bands
        bb_data = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        self.bb_upper = self.I(lambda: bb_data.iloc[:, 0].fillna(method='ffill').values)
        self.bb_middle = self.I(lambda: bb_data.iloc[:, 1].fillna(method='ffill').values)
        self.bb_lower = self.I(lambda: bb_data.iloc[:, 2].fillna(method='ffill').values)
        
        # Additional indicators for futures trading
        self.rsi = self.I(lambda: ta.rsi(close, length=self.rsi_period).fillna(50).values)
        self.atr = self.I(lambda: ta.atr(high, low, close, length=14).fillna(0).values)
        self.volume_ma = self.I(lambda: volume.rolling(10).mean().fillna(method='bfill').values)
        self.ema_21 = self.I(lambda: ta.ema(close, length=21).fillna(method='bfill').values)
        
        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.breakeven_activated = False
    
    def next(self):
        current_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        rsi_current = self.rsi[-1]
        atr_current = self.atr[-1]
        volume_ma_current = self.volume_ma[-1]
        ema_21_current = self.ema_21[-1]
        
        # Skip if indicators are invalid
        if pd.isna(rsi_current) or pd.isna(atr_current):
            return
        
        # Volume confirmation
        volume_confirmation = current_volume > volume_ma_current * self.volume_threshold
        
        # Position management
        if self.position:
            self._manage_position(current_price, ema_21_current, rsi_current)
        else:
            # Enhanced entry conditions
            bb_squeeze = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1] < 0.1
            
            # Mean reversion entry (buy at lower band)
            mean_reversion_buy = (
                current_price <= self.bb_lower[-1] * 1.002 and
                len(self.data.Close) > 1 and
                self.data.Close[-1] > self.data.Close[-2] and  # Price bouncing up
                rsi_current < 40 and  # Oversold
                volume_confirmation
            )
            
            # Breakout entry (buy above upper band)
            breakout_buy = (
                current_price > self.bb_upper[-1] * 1.001 and
                rsi_current > 60 and  # Strong momentum
                volume_confirmation and
                not bb_squeeze  # Avoid false breakouts during squeeze
            )
            
            # Mean reversion entry (sell at upper band)
            mean_reversion_sell = (
                current_price >= self.bb_upper[-1] * 0.998 and
                len(self.data.Close) > 1 and
                self.data.Close[-1] < self.data.Close[-2] and  # Price turning down
                rsi_current > 60 and  # Overbought
                volume_confirmation
            )
            
            # Breakout entry (sell below lower band)
            breakout_sell = (
                current_price < self.bb_lower[-1] * 0.999 and
                rsi_current < 40 and  # Strong downward momentum
                volume_confirmation and
                not bb_squeeze  # Avoid false breakouts during squeeze
            )
            
            if (mean_reversion_buy or breakout_buy) and atr_current > 0:
                # Calculate position size
                stop_distance = atr_current * self.atr_multiplier
                position_size = min(self.risk_per_trade / (stop_distance / current_price), 
                                  self.position_size_pct)
                
                self.buy(size=position_size)
                self.entry_price = current_price
                self.stop_loss = current_price - stop_distance
                self.breakeven_activated = False
                
            elif (mean_reversion_sell or breakout_sell) and atr_current > 0:
                # Calculate position size
                stop_distance = atr_current * self.atr_multiplier
                position_size = min(self.risk_per_trade / (stop_distance / current_price), 
                                  self.position_size_pct)
                
                self.sell(size=position_size)
                self.entry_price = current_price
                self.stop_loss = current_price + stop_distance
                self.breakeven_activated = False
    
    def _manage_position(self, current_price, ema_21_current, rsi_current):
        """Enhanced position management for Bollinger Bands strategy"""
        if not self.entry_price:
            return
        
        # Calculate R multiple
        if self.position.is_long:
            r_multiple = (current_price - self.entry_price) / (self.entry_price - self.stop_loss)
            
            # Breakeven after 1R profit
            if r_multiple >= 1 and not self.breakeven_activated:
                self.stop_loss = self.entry_price
                self.breakeven_activated = True
            
            # Trailing stop using EMA21 after 1R profit
            if r_multiple >= 1 and ema_21_current > self.stop_loss:
                self.stop_loss = ema_21_current
            
            # Exit conditions
            exit_conditions = [
                current_price <= self.stop_loss,
                current_price >= self.bb_upper[-1] * 0.998 and r_multiple > 0.5,  # Take profit at upper band
                rsi_current > 75,  # Extreme overbought
                r_multiple >= 3,  # Take profit at 3R
                r_multiple < -1.5  # Emergency stop
            ]
        else:  # Short position
            r_multiple = (self.entry_price - current_price) / (self.stop_loss - self.entry_price)
            
            # Breakeven after 1R profit
            if r_multiple >= 1 and not self.breakeven_activated:
                self.stop_loss = self.entry_price
                self.breakeven_activated = True
            
            # Trailing stop using EMA21 after 1R profit
            if r_multiple >= 1 and ema_21_current < self.stop_loss:
                self.stop_loss = ema_21_current
            
            # Exit conditions
            exit_conditions = [
                current_price >= self.stop_loss,
                current_price <= self.bb_lower[-1] * 1.002 and r_multiple > 0.5,  # Take profit at lower band
                rsi_current < 25,  # Extreme oversold
                r_multiple >= 3,  # Take profit at 3R
                r_multiple < -1.5  # Emergency stop
            ]
        
        if any(exit_conditions):
            self.position.close()
            self.entry_price = None
            self.stop_loss = None
            self.breakeven_activated = False

class CombinedStrategy(Strategy):
    """
    Futures-Oriented Combined RSI + MACD + Bollinger Bands Strategy
    
    Enhanced with:
    - Multi-indicator confluence
    - ATR-based position sizing and stops
    - Trailing stops and breakeven
    - Volume confirmation
    - Advanced risk management
    """    
    rsi_period = 14
    rsi_oversold = 35
    rsi_overbought = 65
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    bb_period = 20
    bb_std = 2
    atr_multiplier = 2.0
    atr_multiplier_trailing = 2.0
    
    # Futures-specific parameters
    position_size_pct = 0.02  # 2% of account per trade
    risk_per_trade = 0.01  # 1% risk per trade
    volume_threshold = 1.3
    confluence_threshold = 3  # Minimum number of bullish/bearish signals
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # RSI
        rsi_values = ta.rsi(close, length=self.rsi_period)
        self.rsi = self.I(lambda: rsi_values.fillna(50).values)
        
        # MACD
        macd_data = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        macd_line = macd_data.iloc[:, 0].fillna(0)
        macd_signal_line = macd_data.iloc[:, 1].fillna(0)
        macd_histogram = macd_data.iloc[:, 2].fillna(0)
        
        self.macd = self.I(lambda: macd_line.values)
        self.macd_signal = self.I(lambda: macd_signal_line.values)
        self.macd_histogram = self.I(lambda: macd_histogram.values)
        
        # Bollinger Bands
        bb_data = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        self.bb_upper = self.I(lambda: bb_data.iloc[:, 0].fillna(method='ffill').values)
        self.bb_lower = self.I(lambda: bb_data.iloc[:, 2].fillna(method='ffill').values)
        self.bb_middle = self.I(lambda: bb_data.iloc[:, 1].fillna(method='ffill').values)
        
        # Additional indicators for futures trading
        self.atr = self.I(lambda: ta.atr(high, low, close, length=14).fillna(0).values)
        self.volume_ma = self.I(lambda: volume.rolling(10).mean().fillna(method='bfill').values)
        self.ema_21 = self.I(lambda: ta.ema(close, length=21).fillna(method='bfill').values)
        self.ema_50 = self.I(lambda: ta.ema(close, length=50).fillna(method='bfill').values)
        
        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.breakeven_activated = False
    
    def next(self):
        # Current values
        current_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        rsi_current = self.rsi[-1]
        macd_current = self.macd[-1]
        macd_signal_current = self.macd_signal[-1]
        macd_histogram_current = self.macd_histogram[-1]
        atr_current = self.atr[-1]
        volume_ma_current = self.volume_ma[-1]
        ema_21_current = self.ema_21[-1]
        ema_50_current = self.ema_50[-1]
        
        # Skip if indicators are invalid
        if any(pd.isna(val) for val in [rsi_current, macd_current, atr_current]):
            return
        
        # Volume confirmation
        volume_confirmation = current_volume > volume_ma_current * self.volume_threshold
        
        # Position management
        if self.position:
            self._manage_position(current_price, ema_21_current, rsi_current, 
                                macd_current, macd_signal_current)
        else:
            # Calculate confluence signals
            bullish_signals = self._count_bullish_signals(current_price, rsi_current, 
                                                        macd_current, macd_signal_current, 
                                                        macd_histogram_current, ema_21_current, 
                                                        ema_50_current, volume_confirmation)
            
            bearish_signals = self._count_bearish_signals(current_price, rsi_current, 
                                                        macd_current, macd_signal_current, 
                                                        macd_histogram_current, ema_21_current, 
                                                        ema_50_current, volume_confirmation)
            
            # Entry conditions with confluence
            if bullish_signals >= self.confluence_threshold and atr_current > 0:
                # Calculate position size
                stop_distance = atr_current * self.atr_multiplier
                position_size = min(self.risk_per_trade / (stop_distance / current_price), 
                                  self.position_size_pct)
                
                self.buy(size=position_size)
                self.entry_price = current_price
                self.stop_loss = current_price - stop_distance
                self.breakeven_activated = False
                
            elif bearish_signals >= self.confluence_threshold and atr_current > 0:
                # Calculate position size
                stop_distance = atr_current * self.atr_multiplier
                position_size = min(self.risk_per_trade / (stop_distance / current_price), 
                                  self.position_size_pct)
                
                self.sell(size=position_size)
                self.entry_price = current_price
                self.stop_loss = current_price + stop_distance
                self.breakeven_activated = False
    
    def _count_bullish_signals(self, price, rsi, macd, macd_signal, macd_histogram, 
                              ema_21, ema_50, volume_conf):
        """Count bullish confluence signals"""
        signals = 0
        
        # RSI oversold
        if rsi < self.rsi_oversold:
            signals += 1
        
        # MACD bullish
        if macd > macd_signal and macd_histogram > 0:
            signals += 1
        
        # Bollinger Bands - price near lower band
        if price <= self.bb_lower[-1] * 1.005:
            signals += 1
        
        # Price above EMAs (trend confirmation)
        if price > ema_21 and ema_21 > ema_50:
            signals += 1
        
        # Volume confirmation
        if volume_conf:
            signals += 1
        
        # Price bouncing from support
        if len(self.data.Close) > 2 and self.data.Close[-1] > self.data.Close[-2]:
            signals += 1
        
        return signals
    
    def _count_bearish_signals(self, price, rsi, macd, macd_signal, macd_histogram, 
                              ema_21, ema_50, volume_conf):
        """Count bearish confluence signals"""
        signals = 0
        
        # RSI overbought
        if rsi > self.rsi_overbought:
            signals += 1
        
        # MACD bearish
        if macd < macd_signal and macd_histogram < 0:
            signals += 1
        
        # Bollinger Bands - price near upper band
        if price >= self.bb_upper[-1] * 0.995:
            signals += 1
        
        # Price below EMAs (trend confirmation)
        if price < ema_21 and ema_21 < ema_50:
            signals += 1
        
        # Volume confirmation
        if volume_conf:
            signals += 1
        
        # Price rejecting from resistance
        if len(self.data.Close) > 2 and self.data.Close[-1] < self.data.Close[-2]:
            signals += 1
        
        return signals
    
    def _manage_position(self, current_price, ema_21_current, rsi_current, 
                        macd_current, macd_signal_current):
        """Enhanced position management for combined strategy"""
        if not self.entry_price:
            return
        
        # Calculate R multiple
        if self.position.is_long:
            r_multiple = (current_price - self.entry_price) / (self.entry_price - self.stop_loss)
            
            # Breakeven after 1R profit
            if r_multiple >= 1 and not self.breakeven_activated:
                self.stop_loss = self.entry_price
                self.breakeven_activated = True
            
            # Trailing stop using EMA21 after 1R profit
            if r_multiple >= 1 and ema_21_current > self.stop_loss:
                self.stop_loss = ema_21_current
            
            # Exit conditions
            exit_conditions = [
                current_price <= self.stop_loss,
                rsi_current > self.rsi_overbought and r_multiple > 0.5,
                macd_current < macd_signal_current and r_multiple > 0.5,
                current_price >= self.bb_upper[-1] * 0.998 and r_multiple > 1,
                r_multiple >= 5,  # Take profit at 5R
                r_multiple < -1.5  # Emergency stop
            ]
        else:  # Short position
            r_multiple = (self.entry_price - current_price) / (self.stop_loss - self.entry_price)
            
            # Breakeven after 1R profit
            if r_multiple >= 1 and not self.breakeven_activated:
                self.stop_loss = self.entry_price
                self.breakeven_activated = True
            
            # Trailing stop using EMA21 after 1R profit
            if r_multiple >= 1 and ema_21_current < self.stop_loss:
                self.stop_loss = ema_21_current
            
            # Exit conditions
            exit_conditions = [
                current_price >= self.stop_loss,
                rsi_current < self.rsi_oversold and r_multiple > 0.5,
                macd_current > macd_signal_current and r_multiple > 0.5,
                current_price <= self.bb_lower[-1] * 1.002 and r_multiple > 1,
                r_multiple >= 5,  # Take profit at 5R
                r_multiple < -1.5  # Emergency stop
            ]
        
        if any(exit_conditions):
            self.position.close()
            self.entry_price = None
            self.stop_loss = None
            self.breakeven_activated = False

class AdvancedTradingStrategy(Strategy):
    """
    Advanced Trading Strategy based on the provided Binance bot
    
    Entry Conditions:
    - LONG: EMA9 > EMA21, RSI < 70, MACD > 0, ADX > 25, volume increase, 
            trend confirmation from higher timeframe
    - SHORT: EMA9 < EMA21, RSI > 30, MACD < 0, ADX > 25, volume increase,
             trend confirmation from higher timeframe
             
    Exit Strategy:
    - Partial exits at 2R (20%), 4R (30%), 6R (30%)
    - Trailing stop using EMA21 after 1R profit
    - Multiple exit conditions including trend reversal
    """
    
    # Parameters
    ema_fast = 9
    ema_slow = 21
    ema_trend = 50
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_period = 14
    adx_threshold = 25
    atr_period = 14
    atr_multiplier = 2.5
    volume_ma_period = 10
    volume_threshold = 1.0
    atr_multiplier = 2.0
    atr_multiplier_trailing = 2.0
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # EMAs
        self.ema_fast_line = self.I(lambda: ta.ema(close, length=self.ema_fast).fillna(method='bfill').values)
        self.ema_slow_line = self.I(lambda: ta.ema(close, length=self.ema_slow).fillna(method='bfill').values)
        self.ema_trend_line = self.I(lambda: ta.ema(close, length=self.ema_trend).fillna(method='bfill').values)
        
        # RSI
        self.rsi = self.I(lambda: ta.rsi(close, length=self.rsi_period).fillna(50).values)
        
        # MACD
        macd_data = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self.macd = self.I(lambda: macd_data.iloc[:, 0].fillna(0).values)
        self.macd_signal_line = self.I(lambda: macd_data.iloc[:, 1].fillna(0).values)
        
        # ADX
        adx_data = ta.adx(high, low, close, length=self.adx_period)
        self.adx = self.I(lambda: adx_data['ADX_14'].fillna(0).values)
        self.di_plus = self.I(lambda: adx_data['DMP_14'].fillna(0).values)
        self.di_minus = self.I(lambda: adx_data['DMN_14'].fillna(0).values)
        
        # ATR for stop loss
        self.atr = self.I(lambda: ta.atr(high, low, close, length=self.atr_period).fillna(0).values)
        
        # Volume
        self.volume_ma = self.I(lambda: volume.rolling(self.volume_ma_period).mean().fillna(method='bfill').values)
        
        # Breakout levels
        self.high_5 = self.I(lambda: high.rolling(5).max().fillna(method='bfill').values)
        self.low_5 = self.I(lambda: low.rolling(5).min().fillna(method='bfill').values)
        
        # Initialize tracking variables
        self.entry_price = None
        self.stop_loss = None
        self.first_target_hit = False
        self.second_target_hit = False
        self.third_target_hit = False
        self.breakeven_activated = False
    
    def next(self):
        # Current values
        current_price = self.data.Close[-1]
        ema_fast_current = self.ema_fast_line[-1]
        ema_slow_current = self.ema_slow_line[-1]
        ema_trend_current = self.ema_trend_line[-1]
        rsi_current = self.rsi[-1]
        macd_current = self.macd[-1]
        macd_signal_current = self.macd_signal_line[-1]
        adx_current = self.adx[-1]
        di_plus_current = self.di_plus[-1]
        di_minus_current = self.di_minus[-1]
        atr_current = self.atr[-1]
        volume_current = self.data.Volume[-1]
        volume_ma_current = self.volume_ma[-1]
        
        # Check for NaN values
        if any(pd.isna(val) for val in [ema_fast_current, ema_slow_current, rsi_current, 
                                       macd_current, adx_current, atr_current]):
            return
        
        # Crossover signals
        ema_cross_up = (ema_fast_current > ema_slow_current and 
                       len(self.ema_fast_line) > 1 and 
                       self.ema_fast_line[-2] <= self.ema_slow_line[-2])
        
        ema_cross_down = (ema_fast_current < ema_slow_current and 
                         len(self.ema_fast_line) > 1 and 
                         self.ema_fast_line[-2] >= self.ema_slow_line[-2])
        
        macd_cross_up = (macd_current > macd_signal_current and 
                        len(self.macd) > 1 and 
                        self.macd[-2] <= self.macd_signal_line[-2])
        
        macd_cross_down = (macd_current < macd_signal_current and 
                          len(self.macd) > 1 and 
                          self.macd[-2] >= self.macd_signal_line[-2])
        
        # Breakout signals
        breakout_up = (current_price > self.high_5[-2] if len(self.high_5) > 1 else False)
        breakout_down = (current_price < self.low_5[-2] if len(self.low_5) > 1 else False)
        
        # Volume condition
        volume_increase = volume_current > volume_ma_current * self.volume_threshold
        
        # Trend conditions
        uptrend = current_price > ema_trend_current
        downtrend = current_price < ema_trend_current
        
        # Position management
        if self.position:
            self._manage_position(current_price, ema_slow_current, rsi_current, 
                                ema_cross_up, ema_cross_down, macd_cross_up, macd_cross_down,
                                uptrend, downtrend, volume_current, volume_ma_current)
        else:
            # Entry conditions
            long_condition = (
                ema_fast_current > ema_slow_current and
                (ema_cross_up or macd_cross_up or breakout_up) and
                rsi_current < self.rsi_overbought and
                volume_increase and
                macd_current > 0 and
                adx_current > self.adx_threshold and
                (uptrend or (adx_current > 20 and di_plus_current > di_minus_current))
            )
            
            short_condition = (
                ema_fast_current < ema_slow_current and
                (ema_cross_down or macd_cross_down or breakout_down) and
                rsi_current > self.rsi_oversold and
                volume_increase and
                macd_current < 0 and
                adx_current > self.adx_threshold and
                (downtrend or (adx_current > 20 and di_minus_current > di_plus_current))
            )
            
            if long_condition:
                self._enter_long(current_price, atr_current)
            elif short_condition:
                self._enter_short(current_price, atr_current)
    
    def _enter_long(self, current_price, atr_current):
        """Enter long position"""
        self.buy()
        self.entry_price = current_price
        self.stop_loss = current_price - atr_current * self.atr_multiplier
        self.first_target_hit = False
        self.second_target_hit = False
        self.third_target_hit = False
        self.breakeven_activated = False
    
    def _enter_short(self, current_price, atr_current):
        """Enter short position"""
        self.sell()
        self.entry_price = current_price
        self.stop_loss = current_price + atr_current * self.atr_multiplier
        self.first_target_hit = False
        self.second_target_hit = False
        self.third_target_hit = False
        self.breakeven_activated = False
    
    def _manage_position(self, current_price, ema_slow_current, rsi_current,
                        ema_cross_up, ema_cross_down, macd_cross_up, macd_cross_down,
                        uptrend, downtrend, volume_current, volume_ma_current):
        """Manage existing position"""
        if not self.entry_price:
            return
        
        # Calculate R multiple
        if self.position.is_long:
            r_multiple = (current_price - self.entry_price) / (self.entry_price - self.stop_loss)
        else:
            r_multiple = (self.entry_price - current_price) / (self.stop_loss - self.entry_price)
        
        # Breakeven after 1R
        if r_multiple > 1 and not self.breakeven_activated:
            self.stop_loss = self.entry_price
            self.breakeven_activated = True
        
        # Trailing stop using EMA21
        if r_multiple > 1:
            if self.position.is_long and ema_slow_current > self.stop_loss:
                self.stop_loss = ema_slow_current
            elif self.position.is_short and ema_slow_current < self.stop_loss:
                self.stop_loss = ema_slow_current
        
        # Partial exits (simulated by reducing position size in backtesting framework)
        # Note: The backtesting framework doesn't support partial exits, so we use early exits
        
        # Exit conditions
        if self.position.is_long:
            exit_conditions = [
                current_price <= self.stop_loss,
                ema_cross_down,
                macd_cross_down,
                rsi_current > self.rsi_overbought,
                not uptrend and r_multiple > 1,
                r_multiple >= 6,  # Take full profit at 6R instead of partial
                r_multiple < -1,  # Stop loss
                volume_current < volume_ma_current * 0.5 and r_multiple > 0
            ]
        else:  # Short position
            exit_conditions = [
                current_price >= self.stop_loss,
                ema_cross_up,
                macd_cross_up,
                rsi_current < self.rsi_oversold,
                not downtrend and r_multiple > 1,
                r_multiple >= 6,  # Take full profit at 6R instead of partial
                r_multiple < -1,  # Stop loss
                volume_current < volume_ma_current * 0.5 and r_multiple > 0
            ]
        
        if any(exit_conditions):
            self.position.close()
            self.entry_price = None
            self.stop_loss = None


class BinanceFuturesStrategy(Strategy):
    """
    Comprehensive Binance Futures Trading Strategy
    
    Features:
    - Multi-timeframe analysis (5m primary, 1h trend confirmation simulated)
    - Partial exits at 2R (20%), 4R (30%), 6R (30%) profit levels
    - EMA21-based trailing stops after 1R profit
    - Breakeven stop after 1R
    - Comprehensive entry/exit conditions
    - Advanced risk management with position sizing
    """
    
    # Parameters
    ema_fast = 9
    ema_slow = 21
    ema_trend = 50
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_period = 14
    adx_threshold = 25
    atr_period = 14
    atr_multiplier = 2.0
    atr_multiplier_trailing = 2.0
    volume_ma_period = 10
    volume_threshold = 1.2
    
    # Futures-specific parameters
    leverage = 1.0  # Start with 1x leverage for safety
    position_size_pct = 0.02  # 2% of account per trade
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # EMAs for trend analysis
        self.ema_fast_line = self.I(lambda: ta.ema(close, length=self.ema_fast).fillna(method='bfill').values)
        self.ema_slow_line = self.I(lambda: ta.ema(close, length=self.ema_slow).fillna(method='bfill').values)
        self.ema_trend_line = self.I(lambda: ta.ema(close, length=self.ema_trend).fillna(method='bfill').values)
        
        # RSI for momentum
        self.rsi = self.I(lambda: ta.rsi(close, length=self.rsi_period).fillna(50).values)
        
        # MACD for trend confirmation
        macd_data = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self.macd = self.I(lambda: macd_data.iloc[:, 0].fillna(0).values)
        self.macd_signal_line = self.I(lambda: macd_data.iloc[:, 1].fillna(0).values)
        self.macd_histogram = self.I(lambda: macd_data.iloc[:, 2].fillna(0).values)
        
        # ADX for trend strength
        adx_data = ta.adx(high, low, close, length=self.adx_period)
        self.adx = self.I(lambda: adx_data['ADX_14'].fillna(0).values)
        self.di_plus = self.I(lambda: adx_data['DMP_14'].fillna(0).values)
        self.di_minus = self.I(lambda: adx_data['DMN_14'].fillna(0).values)
        
        # ATR for volatility and stop loss
        self.atr = self.I(lambda: ta.atr(high, low, close, length=self.atr_period).fillna(0).values)
        
        # Volume analysis
        self.volume_ma = self.I(lambda: volume.rolling(self.volume_ma_period).mean().fillna(method='bfill').values)
        self.volume_ratio = self.I(lambda: (volume / volume.rolling(self.volume_ma_period).mean()).fillna(1).values)
        
        # Support and Resistance levels
        self.high_20 = self.I(lambda: high.rolling(20).max().fillna(method='bfill').values)
        self.low_20 = self.I(lambda: low.rolling(20).min().fillna(method='bfill').values)
        self.high_5 = self.I(lambda: high.rolling(5).max().fillna(method='bfill').values)
        self.low_5 = self.I(lambda: low.rolling(5).min().fillna(method='bfill').values)
        
        # Bollinger Bands for volatility
        bb_data = ta.bbands(close, length=20, std=2)
        self.bb_upper = self.I(lambda: bb_data.iloc[:, 0].fillna(method='ffill').values)
        self.bb_middle = self.I(lambda: bb_data.iloc[:, 1].fillna(method='ffill').values)
        self.bb_lower = self.I(lambda: bb_data.iloc[:, 2].fillna(method='ffill').values)
        
        # Initialize position tracking
        self.entry_price = None
        self.stop_loss = None
        self.take_profit_1 = None
        self.take_profit_2 = None
        self.take_profit_3 = None
        self.position_size = None
        self.remaining_size = 1.0
        self.breakeven_activated = False
        self.trailing_stop_activated = False
        self.partial_exits = {'2R': False, '4R': False, '6R': False}
        
    def next(self):
        # Current values
        current_price = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_volume = self.data.Volume[-1]
        
        # Technical indicators
        ema_fast_current = self.ema_fast_line[-1]
        ema_slow_current = self.ema_slow_line[-1]
        ema_trend_current = self.ema_trend_line[-1]
        rsi_current = self.rsi[-1]
        macd_current = self.macd[-1]
        macd_signal_current = self.macd_signal_line[-1]
        macd_histogram_current = self.macd_histogram[-1]
        adx_current = self.adx[-1]
        di_plus_current = self.di_plus[-1]
        di_minus_current = self.di_minus[-1]
        atr_current = self.atr[-1]
        volume_ratio_current = self.volume_ratio[-1]
        
        # Check for NaN values
        if any(pd.isna(val) for val in [ema_fast_current, ema_slow_current, rsi_current, 
                                       macd_current, adx_current, atr_current]):
            return
        
        # Market structure analysis
        uptrend = (ema_fast_current > ema_slow_current and 
                  current_price > ema_trend_current and
                  ema_slow_current > ema_trend_current)
        
        downtrend = (ema_fast_current < ema_slow_current and 
                    current_price < ema_trend_current and
                    ema_slow_current < ema_trend_current)
          # Crossover signals
        ema_bullish_cross = (ema_fast_current > ema_slow_current and 
                           len(self.ema_fast_line) > 1 and 
                           self.ema_fast_line[-2] <= self.ema_slow_line[-2])
        
        ema_bearish_cross = (ema_fast_current < ema_slow_current and 
                           len(self.ema_fast_line) > 1 and 
                           self.ema_fast_line[-2] >= self.ema_slow_line[-2])
        
        macd_bullish_cross = (macd_current > macd_signal_current and 
                            len(self.macd) > 1 and 
                            self.macd[-2] <= self.macd_signal_line[-2])
        
        macd_bearish_cross = (macd_current < macd_signal_current and 
                            len(self.macd) > 1 and 
                            self.macd[-2] >= self.macd_signal_line[-2])
        
        # Momentum conditions
        momentum_bullish = (rsi_current > 50 and rsi_current < self.rsi_overbought and
                          macd_current > 0 and macd_histogram_current > 0)
        
        momentum_bearish = (rsi_current < 50 and rsi_current > self.rsi_oversold and
                          macd_current < 0 and macd_histogram_current < 0)
        
        # Volume confirmation
        volume_confirmation = volume_ratio_current > self.volume_threshold
        
        # Breakout conditions
        breakout_bullish = current_price > self.high_5[-2] if len(self.high_5) > 1 else False
        breakout_bearish = current_price < self.low_5[-2] if len(self.low_5) > 1 else False
        
        # Position management
        if self.position:
            self._manage_futures_position(current_price, ema_slow_current, rsi_current, 
                                        ema_bearish_cross, ema_bullish_cross, 
                                        macd_bearish_cross, macd_bullish_cross,
                                        uptrend, downtrend, volume_ratio_current)
        else:
            # Entry conditions for LONG
            long_entry_conditions = [
                uptrend,
                (ema_bullish_cross or macd_bullish_cross or breakout_bullish),
                momentum_bullish,
                adx_current > self.adx_threshold,
                di_plus_current > di_minus_current,
                volume_confirmation,
                current_price > self.bb_middle[-1],  # Above middle Bollinger Band
                rsi_current < self.rsi_overbought,
                # Additional confluence
                current_price > self.ema_trend_line[-1],
                macd_current > macd_signal_current
            ]
            
            # Entry conditions for SHORT
            short_entry_conditions = [
                downtrend,
                (ema_bearish_cross or macd_bearish_cross or breakout_bearish),
                momentum_bearish,
                adx_current > self.adx_threshold,
                di_minus_current > di_plus_current,
                volume_confirmation,
                current_price < self.bb_middle[-1],  # Below middle Bollinger Band
                rsi_current > self.rsi_oversold,
                # Additional confluence
                current_price < self.ema_trend_line[-1],
                macd_current < macd_signal_current
            ]
            
            # Execute entries with proper risk management
            if sum(long_entry_conditions) >= 7:  # Require strong confluence
                self._enter_long_futures(current_price, atr_current)
            elif sum(short_entry_conditions) >= 7:  # Require strong confluence
                self._enter_short_futures(current_price, atr_current)
    
    def _enter_long_futures(self, current_price, atr_current):
        """Enter long position with futures-specific risk management"""
        # Calculate position size based on risk management
        risk_amount = self.position_size_pct
        
        # Set stop loss
        stop_loss_distance = atr_current * self.atr_multiplier
        stop_loss_price = current_price - stop_loss_distance
        
        # Calculate position size to risk 1% of account
        position_size = self.risk_per_trade / (stop_loss_distance / current_price)
        position_size = min(position_size, self.position_size_pct)  # Cap at max position size
        
        # Enter position
        self.buy(size=position_size)
        
        # Set position tracking
        self.entry_price = current_price
        self.stop_loss = stop_loss_price
        self.position_size = position_size
        self.remaining_size = 1.0
        
        # Set profit targets
        reward_distance = stop_loss_distance
        self.take_profit_1 = current_price + (reward_distance * 2)  # 2R
        self.take_profit_2 = current_price + (reward_distance * 4)  # 4R
        self.take_profit_3 = current_price + (reward_distance * 6)  # 6R
        
        # Reset flags
        self.breakeven_activated = False
        self.trailing_stop_activated = False
        self.partial_exits = {'2R': False, '4R': False, '6R': False}
    
    def _enter_short_futures(self, current_price, atr_current):
        """Enter short position with futures-specific risk management"""
        # Calculate position size based on risk management
        risk_amount = self.position_size_pct
        
        # Set stop loss
        stop_loss_distance = atr_current * self.atr_multiplier
        stop_loss_price = current_price + stop_loss_distance
        
        # Calculate position size to risk 1% of account
        position_size = self.risk_per_trade / (stop_loss_distance / current_price)
        position_size = min(position_size, self.position_size_pct)  # Cap at max position size
        
        # Enter position
        self.sell(size=position_size)
        
        # Set position tracking
        self.entry_price = current_price
        self.stop_loss = stop_loss_price
        self.position_size = position_size
        self.remaining_size = 1.0
        
        # Set profit targets
        reward_distance = stop_loss_distance
        self.take_profit_1 = current_price - (reward_distance * 2)  # 2R
        self.take_profit_2 = current_price - (reward_distance * 4)  # 4R
        self.take_profit_3 = current_price - (reward_distance * 6)  # 6R
        
        # Reset flags
        self.breakeven_activated = False
        self.trailing_stop_activated = False
        self.partial_exits = {'2R': False, '4R': False, '6R': False}
    
    def _manage_futures_position(self, current_price, ema_slow_current, rsi_current,
                               ema_bearish_cross, ema_bullish_cross,
                               macd_bearish_cross, macd_bullish_cross,
                               uptrend, downtrend, volume_ratio_current):
        """Advanced position management for futures trading"""
        if not self.entry_price:
            return
        
        # Calculate R multiple (risk-reward ratio)
        if self.position.is_long:
            r_multiple = (current_price - self.entry_price) / (self.entry_price - self.stop_loss)
            
            # Breakeven after 1R profit
            if r_multiple >= 1 and not self.breakeven_activated:
                self.stop_loss = self.entry_price
                self.breakeven_activated = True
            
            # Trailing stop using EMA21 after 1R profit
            if r_multiple >= 1 and not self.trailing_stop_activated:
                self.trailing_stop_activated = True
            
            if self.trailing_stop_activated and ema_slow_current > self.stop_loss:
                self.stop_loss = ema_slow_current
            
            # Partial exits (simulated with full exit at targets since backtesting framework limitation)
            if current_price >= self.take_profit_3 and not self.partial_exits['6R']:
                self.position.close()
                self._reset_position_tracking()
                return
            elif current_price >= self.take_profit_2 and not self.partial_exits['4R']:
                # In real implementation, would close 30% here
                self.partial_exits['4R'] = True
            elif current_price >= self.take_profit_1 and not self.partial_exits['2R']:
                # In real implementation, would close 20% here
                self.partial_exits['2R'] = True
            
            # Exit conditions
            exit_conditions = [
                current_price <= self.stop_loss,
                ema_bearish_cross and r_multiple > 0.5,
                macd_bearish_cross and r_multiple > 0.5,
                rsi_current > 80,
                not uptrend and r_multiple > 1,
                volume_ratio_current < 0.5 and r_multiple > 2,
                r_multiple < -1.5  # Emergency stop
            ]
            
        else:  # Short position
            r_multiple = (self.entry_price - current_price) / (self.stop_loss - self.entry_price)
            
            # Breakeven after 1R profit
            if r_multiple >= 1 and not self.breakeven_activated:
                self.stop_loss = self.entry_price
                self.breakeven_activated = True
            
            # Trailing stop using EMA21 after 1R profit
            if r_multiple >= 1 and not self.trailing_stop_activated:
                self.trailing_stop_activated = True
            
            if self.trailing_stop_activated and ema_slow_current < self.stop_loss:
                self.stop_loss = ema_slow_current
            
            # Partial exits
            if current_price <= self.take_profit_3 and not self.partial_exits['6R']:
                self.position.close()
                self._reset_position_tracking()
                return
            elif current_price <= self.take_profit_2 and not self.partial_exits['4R']:
                self.partial_exits['4R'] = True
            elif current_price <= self.take_profit_1 and not self.partial_exits['2R']:
                self.partial_exits['2R'] = True
            
            # Exit conditions
            exit_conditions = [
                current_price >= self.stop_loss,
                ema_bullish_cross and r_multiple > 0.5,
                macd_bullish_cross and r_multiple > 0.5,
                rsi_current < 20,
                not downtrend and r_multiple > 1,
                volume_ratio_current < 0.5 and r_multiple > 2,
                r_multiple < -1.5  # Emergency stop
            ]
        
        # Execute exit if any condition is met
        if any(exit_conditions):
            self.position.close()
            self._reset_position_tracking()
    
    def _reset_position_tracking(self):
        """Reset all position tracking variables"""
        self.entry_price = None
        self.stop_loss = None
        self.take_profit_1 = None
        self.take_profit_2 = None
        self.take_profit_3 = None
        self.position_size = None
        self.remaining_size = 1.0
        self.breakeven_activated = False
        self.trailing_stop_activated = False
        self.partial_exits = {'2R': False, '4R': False, '6R': False}

def get_crypto_data(symbol='BTC-USD', period=None, interval='1d',start_date=None, end_date=None):
    """
    Fetch cryptocurrency data from Yahoo Finance
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, period=period, interval=interval)
    
    # Clean the data
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    
    return data

def run_backtest(data, strategy_class, **kwargs):
    """
    Run backtest with given strategy
    """
    bt = Backtest(data, strategy_class, cash=1000000, commission=.002)  # 1M USD initial cash to handle BTC prices
    
    if kwargs:
        results = bt.optimize(**kwargs)
    else:
        results = bt.run()
    
    return bt, results

def main():
    # Get Bitcoin data
    print("Fetching Bitcoin data...")
    btc_data = get_crypto_data('BTC-USD', start_date="2025-02-01", end_date="2025-04-01", interval='4h')
    
    print(f"Data shape: {btc_data.shape}")
    print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
    
    # Test different strategies
    strategies = [
        ("RSI + MACD Strategy", RSIMACDStrategy),
        ("Bollinger Bands Strategy", BollingerBandsStrategy),
        ("Combined Strategy", CombinedStrategy),
        ("Advanced Trading Strategy", AdvancedTradingStrategy),
        ("Binance Futures Strategy", BinanceFuturesStrategy),
    ]
    results_summary = []
    
    for name, strategy_class in strategies:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")
        
        bt, results = run_backtest(btc_data, strategy_class,atr_multiplier=[1.5, 2.0, 2.5, 3.0],
                           atr_multiplier_trailing=[1.5, 2.0, 2.5, 3.0])
        
        # Store results - handle potential key variations
        results_summary.append({
            'Strategy': name,
            'Return [%]': results.get('Return [%]', 0),
            'Buy & Hold [%]': results.get('Buy & Hold Return [%]', 0),
            'Max Drawdown [%]': results.get('Max. Drawdown [%]', 0),
            'Sharpe Ratio': results.get('Sharpe Ratio', 0),
            '# Trades': results.get('# Trades', 0),
            'Win Rate [%]': results.get('Win Rate [%]', 0),
            'Avg Trade [%]': results.get('Avg. Trade [%]', 0)
        })        
        print(f"Total Return: {results.get('Return [%]', 0):.2f}%")
        print(f"Buy & Hold: {results.get('Buy & Hold Return [%]', 0):.2f}%")
        print(f"Max Drawdown: {results.get('Max. Drawdown [%]', 0):.2f}%")
        print(f"Sharpe Ratio: {results.get('Sharpe Ratio', 0):.2f}")
        print(f"Number of Trades: {results.get('# Trades', 0)}")
        print(f"Win Rate: {results.get('Win Rate [%]', 0):.2f}%")
        
        # Plot results
        # bt.plot()
    
    # Summary comparison
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results_summary)
    print(results_df.to_string(index=False, float_format='%.2f'))
    
    return results_df

if __name__ == "__main__":
    results = main()
