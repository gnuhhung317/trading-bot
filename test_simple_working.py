#!/usr/bin/env python3
"""
Simple Working Strategy Test

This creates a basic strategy that will definitely generate trades
to test our framework and then improve from there.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')


class SimpleWorkingStrategy(Strategy):
    """
    Simple strategy that will generate trades for testing
    """
    # Simple parameters
    sma_fast = 10
    sma_slow = 20
    rsi_period = 14
    atr_multiplier = 2.0
    atr_multiplier_trailing = 2.0
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Simple moving averages
        self.sma_fast = self.I(lambda: close.rolling(self.sma_fast).mean())
        self.sma_slow = self.I(lambda: close.rolling(self.sma_slow).mean())
        
        # RSI for additional confirmation
        self.rsi = self.I(lambda: ta.rsi(close, length=self.rsi_period).fillna(50).values)
        
        # ATR for stop loss
        self.atr = self.I(lambda: ta.atr(high, low, close, length=14).fillna(0).values)
        
    def next(self):
        # Skip if not enough data
        if len(self.sma_fast) < 2 or len(self.sma_slow) < 2:
            return
            
        current_price = self.data.Close[-1]
        sma_fast_current = self.sma_fast[-1]
        sma_slow_current = self.sma_slow[-1]
        rsi_current = self.rsi[-1]
        atr_current = self.atr[-1]
        
        # Skip if data is invalid
        if pd.isna(sma_fast_current) or pd.isna(sma_slow_current) or pd.isna(rsi_current):
            return
        
        # Simple entry conditions
        if not self.position:
            # Buy when fast SMA crosses above slow SMA and RSI is not overbought
            if (sma_fast_current > sma_slow_current and 
                self.sma_fast[-2] <= self.sma_slow[-2] and  # Just crossed
                rsi_current < 70 and
                atr_current > 0):
                
                self.buy(size=0.5)  # Use 50% of available cash
                
            # Sell when fast SMA crosses below slow SMA and RSI is not oversold
            elif (sma_fast_current < sma_slow_current and 
                  self.sma_fast[-2] >= self.sma_slow[-2] and  # Just crossed
                  rsi_current > 30 and
                  atr_current > 0):
                
                self.sell(size=0.5)  # Use 50% of available cash
        else:
            # Simple exit conditions
            if self.position.is_long:
                # Exit long if SMA crosses down or RSI becomes very overbought
                if (sma_fast_current < sma_slow_current or rsi_current > 80):
                    self.position.close()
            elif self.position.is_short:
                # Exit short if SMA crosses up or RSI becomes very oversold
                if (sma_fast_current > sma_slow_current or rsi_current < 20):
                    self.position.close()


def get_crypto_data(symbol='BTC-USD', start_date=None, end_date=None, interval='1d'):
    """Fetch cryptocurrency data"""
    print(f"Fetching {symbol} data from {start_date} to {end_date} ({interval})...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data


def test_simple_strategy():
    """Test the simple strategy with different data periods"""
    
    test_scenarios = [
        ("2024-01-01", "2024-12-31", "1d", "Full Year 2024"),
        ("2024-06-01", "2024-12-31", "1d", "H2 2024"),
        ("2024-10-01", "2024-12-31", "4h", "Q4 2024 - 4H"),
        ("2024-11-01", "2024-12-31", "1h", "Nov-Dec 2024 - 1H"),
    ]
    
    print("="*70)
    print("SIMPLE WORKING STRATEGY TEST")
    print("="*70)
    
    results_summary = []
    
    for start_date, end_date, interval, description in test_scenarios:
        print(f"\n{'-'*50}")
        print(f"Testing: {description}")
        print(f"{'-'*50}")
        
        try:
            # Get data
            data = get_crypto_data('BTC-USD', start_date, end_date, interval)
            
            print(f"Data loaded: {data.shape[0]} bars")
            print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            
            if len(data) < 50:
                print("❌ Not enough data points")
                continue
            
            # Run backtest
            bt = Backtest(data, SimpleWorkingStrategy, cash=100000, commission=.002)
            results = bt.run()
            
            # Store results
            result_data = {
                'Scenario': description,
                'Return [%]': results.get('Return [%]', 0),
                'Buy & Hold [%]': results.get('Buy & Hold Return [%]', 0),
                'Max Drawdown [%]': results.get('Max. Drawdown [%]', 0),
                'Sharpe Ratio': results.get('Sharpe Ratio', 0),
                '# Trades': results.get('# Trades', 0),
                'Win Rate [%]': results.get('Win Rate [%]', 0)
            }
            results_summary.append(result_data)
            
            # Print results
            print(f"✅ RESULTS:")
            print(f"   Return: {results.get('Return [%]', 0):.2f}%")
            print(f"   Buy & Hold: {results.get('Buy & Hold Return [%]', 0):.2f}%")
            print(f"   Max Drawdown: {results.get('Max. Drawdown [%]', 0):.2f}%")
            print(f"   Sharpe Ratio: {results.get('Sharpe Ratio', 0):.2f}")
            print(f"   Trades: {results.get('# Trades', 0)}")
            print(f"   Win Rate: {results.get('Win Rate [%]', 0):.2f}%")
            
            if results.get('# Trades', 0) == 0:
                print("⚠️  No trades generated - strategy too restrictive")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue
    
    # Summary
    if results_summary:
        print(f"\n{'='*70}")
        print("SUMMARY OF ALL TESTS")
        print(f"{'='*70}")
        
        results_df = pd.DataFrame(results_summary)
        print(results_df.to_string(index=False, float_format='%.2f'))
        
        # Find best scenario
        if len(results_df) > 0:
            trading_scenarios = results_df[results_df['# Trades'] > 0]
            if len(trading_scenarios) > 0:
                best = trading_scenarios.loc[trading_scenarios['Return [%]'].idxmax()]
                print(f"\n🏆 Best performing scenario: {best['Scenario']}")
                print(f"   Return: {best['Return [%]']:.2f}%")
                print(f"   Trades: {best['# Trades']}")
                print(f"   Win Rate: {best['Win Rate [%]']:.2f}%")
            else:
                print("\n❌ No scenarios generated trades!")
    else:
        print("\n❌ No successful tests completed")


if __name__ == "__main__":
    test_simple_strategy()
