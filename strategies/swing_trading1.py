import pandas as pd
import numpy as np
import logging
import pandas_ta as ta
from backtesting import Strategy

class SwingBreakoutStrategy(Strategy):
    """
    Swing Trading Breakout Strategy for XAU/USD (D1)
    
    Based on: Swing Trading sau phá vỡ vùng tích lũy
    Enhanced with:
    - Breakout detection after consolidation
    - ATR-based risk management
    - 3x leverage futures trading
    - Risk:Reward 1:5
    
    Run backtest with:
        bt = Backtest(data, SwingBreakoutStrategy, cash=10000, margin=1/3, commission=0.001)
    """
    # Strategy parameters
    swing_lookback = 20  # Lookback for swing high/low detection
    consolidation_lookback = 50  # Lookback to detect consolidation (W1 equivalent)
    consolidation_threshold = 0.05  # Max price range (5%) for consolidation
    atr_period = 14
    atr_multiplier = 2.0
    risk_per_trade = 0.02  # 2% risk per trade
    leverage = 3
    rr_ratio = 5.0  # Risk:Reward 1:5
    contract_multiplier = 100  # XAU/USD futures (1 lot = 100 oz)
    
    # Debug mode
    debug = True
    
    def init(self):
        """Initialize strategy indicators"""
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Swing high/low for breakout detection
        self.swing_high = self.I(lambda: high.rolling(self.swing_lookback).max().shift(1).fillna(method='bfill').values)
        self.swing_low = self.I(lambda: low.rolling(self.swing_lookback).min().shift(1).fillna(method='bfill').values)
        
        # Consolidation detection (price range as % of close)
        self.price_range = self.I(lambda: (
            (high.rolling(self.consolidation_lookback).max() - 
             low.rolling(self.consolidation_lookback).min()) / close
        ).shift(1).fillna(method='bfill').values)
        
        # ATR for stop-loss
        self.atr = self.I(lambda: ta.atr(high, low, close, length=self.atr_period).fillna(0).values)
        
        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
        # Logging
        logging.basicConfig(level=print if self.debug else logging.INFO)
    
    def next(self):
        """Main strategy logic for each candle"""
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        prev_high = self.data.High[-2] if len(self.data.High) > 1 else high
        prev_low = self.data.Low[-2] if len(self.data.Low) > 1 else low
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        price_range = swing_high-swing_low
        atr = self.atr[-1]
        
        # Skip if NaN
        if np.isnan([price, high, low, swing_high, swing_low, price_range, atr]).any():
            return
        
        # Debug logging
        if self.debug:
            print(f"\nCandle {len(self.data)}:")
            print(f"Price: {price:.2f}, High: {high:.2f}, Low: {low:.2f}")
            print(f"Swing High: {swing_high:.2f}, Swing Low: {swing_low:.2f}")
            print(f"Price Range: {price_range:.4f}, ATR: {atr:.2f}")
        
        # Consolidation check
        consolidation = price_range <= self.consolidation_threshold
        
        # Breakout detection
        breakout_up = high > swing_high
        breakout_down = low < swing_low
        
        # Higher High / Lower Low detection
        hh = high > prev_high  # Higher High
        ll = low < prev_low  # Lower Low
        
        # Position management
        if self.position:
            if self.position.is_long and price >= self.take_profit:
                self.position.close()
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
                if self.debug:
                    print(f"Long Exit: TP hit at {price:.2f}")
            elif self.position.is_short and price <= self.take_profit:
                self.position.close()
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
                if self.debug:
                    print(f"Short Exit: TP hit at {price:.2f}")
        
        # Entry conditions
        else:
            # Calculate position size
            equity = self.equity
            risk_amount = equity * self.risk_per_trade
            stop_distance = atr * self.atr_multiplier  # Fallback stop distance
            contracts = risk_amount / (stop_distance * self.contract_multiplier)
            position_size = contracts * self.leverage
            
            # Long entry
            long_condition = (
                consolidation and
                breakout_up and
                hh
            )
            if long_condition:
                entry_price = price
                stop_price = swing_low  # Stop at recent swing low
                stop_distance = entry_price - stop_price
                take_price = entry_price + stop_distance * self.rr_ratio  # RR 1:5
                self.buy(size=position_size, sl=stop_price)
                self.entry_price = entry_price
                self.stop_loss = stop_price
                self.take_profit = take_price
                if self.debug:
                    print(f"Long Entry: Price={entry_price:.2f}, SL={stop_price:.2f}, TP={take_price:.2f}, Size={position_size:.4f}")
            
            # Short entry
            short_condition = (
                consolidation and
                breakout_down and
                ll
            )
            if short_condition:
                entry_price = price
                stop_price = swing_high  # Stop at recent swing high
                stop_distance = stop_price - entry_price
                take_price = entry_price - stop_distance * self.rr_ratio  # RR 1:5
                self.sell(size=position_size, sl=stop_price)
                self.entry_price = entry_price
                self.stop_loss = stop_price
                self.take_profit = take_price
                if self.debug:
                    print(f"Short Entry: Price={entry_price:.2f}, SL={stop_price:.2f}, TP={take_price:.2f}, Size={position_size:.4f}")
        
        # Debug entry conditions
        if self.debug:
            print(f"Long Condition: {long_condition}, Short Condition: {short_condition}")