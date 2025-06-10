"""
Futures-Oriented Trading Strategy Backtest Runner

This module imports strategies from the strategies folder and runs backtests
with comprehensive performance analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from backtesting import Backtest
import warnings
warnings.filterwarnings('ignore')

# Import strategies from the strategies folder
from strategies import (
    RSIMACDStrategy,
    BollingerBandsStrategy,
    SwingBreakoutStrategy,
)


def get_futures_data(symbol='BTCUSDT', start_date=None, end_date=None, interval='1d'):
    """
    Fetch cryptocurrency futures data from Binance
    
    Args:
        symbol: Trading pair symbol (default: 'BTCUSDT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '4h', '1h', '15m', '5m')
    
    Returns:
        pandas.DataFrame: OHLCV data with funding rate
    """
    try:
        # Initialize Binance client
        client = Client()
        
        print(f"Fetching {symbol} futures data...")
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        # Get klines data
        klines = []
        current = start_timestamp
        while current < end_timestamp:
            temp_klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=current,
                endTime=min(current + 1000 * 60 * 60 * 24 * 30, end_timestamp)
            )
            if not temp_klines:
                break
            klines.extend(temp_klines)
            current = temp_klines[-1][0] + 1
            time.sleep(0.5)  # Avoid API rate limits

        if not klines:
            raise ValueError(f"No data found for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        # Get funding rate data
        funding_rates = []
        current = start_timestamp
        while current < end_timestamp:
            try:
                rates = client.futures_funding_rate(
                    symbol=symbol,
                    startTime=current,
                    endTime=min(current + 1000 * 60 * 60 * 24 * 30, end_timestamp)
                )
                if not rates:
                    break
                funding_rates.extend(rates)
                current = rates[-1]['fundingTime'] + 1
                time.sleep(0.5)
            except BinanceAPIException as e:
                print(f"Error fetching funding rates: {e}")
                break

        if funding_rates:
            # Convert funding rates to DataFrame
            funding_df = pd.DataFrame(funding_rates)
            funding_df['timestamp'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
            funding_df.set_index('timestamp', inplace=True)
            funding_df['fundingRate'] = funding_df['fundingRate'].astype(float)
            
            # Resample funding rates to match klines interval
            funding_df = funding_df.resample(interval).agg({
                'fundingRate': 'mean'
            }).fillna(0)
            
            # Merge funding rates with price data
            df = df.join(funding_df, how='left')
            df['fundingRate'] = df['fundingRate'].fillna(0)
        else:
            df['fundingRate'] = 0

        # Clean the data
        df = df.dropna()
        
        # Remove any rows with zero prices or volumes
        df = df[df['volume'] > 0]
        df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
        
        # Rename columns to match backtesting library requirements
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Keep only required columns for backtesting
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'fundingRate']]
        
        print(f"Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df

    except Exception as e:
        print(f"Error fetching futures data: {str(e)}")
        return None


def run_backtest(data, strategy_class, **kwargs):
    """
    Run backtest with given strategy
    
    Args:
        data: OHLCV data
        strategy_class: Strategy class to test
        **kwargs: Optimization parameters
    
    Returns:
        tuple: (Backtest object, results)
    """
    bt = Backtest(data, strategy_class, cash=1000000, commission=.002)
    
    if kwargs:
        results = bt.optimize(**kwargs)
    else:
        results = bt.run()
    
    return bt, results


def create_strategy_summary(results):
    """
    Create a formatted summary of strategy results
    
    Args:
        results: Backtest results object
    
    Returns:
        dict: Formatted results dictionary
    """
    return {
        'Return [%]': results.get('Return [%]', 0),
        'Buy & Hold [%]': results.get('Buy & Hold Return [%]', 0),
        'Max Drawdown [%]': results.get('Max. Drawdown [%]', 0),
        'Sharpe Ratio': results.get('Sharpe Ratio', 0),
        '# Trades': results.get('# Trades', 0),
        'Win Rate [%]': results.get('Win Rate [%]', 0),
        'Avg Trade [%]': results.get('Avg. Trade [%]', 0),
        'Profit Factor': results.get('Profit Factor', 0),
        'Max Trade [%]': results.get('Best Trade [%]', 0),
        'Worst Trade [%]': results.get('Worst Trade [%]', 0)
    }


def print_strategy_results(name, results):
    """
    Print formatted strategy results
    
    Args:
        name: Strategy name
        results: Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")
    print(f"Total Return:     {results.get('Return [%]', 0):>8.2f}%")
    print(f"Buy & Hold:       {results.get('Buy & Hold Return [%]', 0):>8.2f}%")
    print(f"Max Drawdown:     {results.get('Max. Drawdown [%]', 0):>8.2f}%")
    print(f"Sharpe Ratio:     {results.get('Sharpe Ratio', 0):>8.2f}")
    print(f"Number of Trades: {results.get('# Trades', 0):>8}")
    print(f"Win Rate:         {results.get('Win Rate [%]', 0):>8.2f}%")
    print(f"Avg Trade:        {results.get('Avg. Trade [%]', 0):>8.2f}%")
    print(f"Best Trade:       {results.get('Best Trade [%]', 0):>8.2f}%")
    print(f"Worst Trade:      {results.get('Worst Trade [%]', 0):>8.2f}%")


def main():
    """
    Main backtest execution function
    """
    print("="*80)
    print("FUTURES-ORIENTED TRADING STRATEGY BACKTEST")
    print("="*80)
    
    # Fetch Bitcoin futures data
    btc_data = get_futures_data(
        symbol='ETHUSDT', 
        start_date="2025-01-01", 
        end_date="2025-06-01", 
        interval='1h'
    )
    
    if btc_data is None:
        print("Failed to fetch futures data. Exiting...")
        return None
    
    # Define strategies to test with their specific optimization parameters
    strategies = [
        ("RSI + MACD Futures Strategy", RSIMACDStrategy, {
            'atr_multiplier': [1.5, 2.0, 2.5, 3.0],
            'atr_multiplier_trailing': [1.5, 2.0, 2.5, 3.0],
            'position_size_pct': [0.01, 0.02, 0.03, 0.05],
            'risk_per_trade': [0.01, 0.015, 0.02]
        }),
        ("Bollinger Bands Futures Strategy", BollingerBandsStrategy, {
            'atr_multiplier': [1.5, 2.0, 2.5, 3.0],
            'atr_multiplier_trailing': [1.5, 2.0, 2.5, 3.0],
            'position_size_pct': [0.01, 0.02, 0.03, 0.05],
            'risk_per_trade': [0.01, 0.015, 0.02]
        }),
        ("SwingBreakoutStrategy" , SwingBreakoutStrategy ,{})
    ]
    
    results_summary = []
    
    # Test each strategy
    for name, strategy_class, opt_params in strategies:
        print(f"\nTesting: {name}")
        print("-" * 50)
        
        try:
            # Run backtest with optimization
            bt = Backtest(btc_data, strategy_class, cash=1000000, commission=.002,margin=1/3)
            results = bt.run()
            
            # Create summary
            summary = create_strategy_summary(results)
            summary['Strategy'] = name
            results_summary.append(summary)
            
            # Print results
            print_strategy_results(name, results)
            
        except Exception as e:
            print(f"Error testing {name}: {str(e)}")
            # Add failed result to summary
            results_summary.append({
                'Strategy': name,
                'Return [%]': 0,
                'Buy & Hold [%]': 0,
                'Max Drawdown [%]': 0,
                'Sharpe Ratio': 0,
                '# Trades': 0,
                'Win Rate [%]': 0,
                'Avg Trade [%]': 0,
                'Profit Factor': 0,
                'Max Trade [%]': 0,
                'Worst Trade [%]': 0
            })
            continue
    
    # Create comprehensive summary
    print("\n" + "="*100)
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print("="*100)
    
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        
        # Sort by return
        results_df = results_df.sort_values('Return [%]', ascending=False)
        
        # Display formatted table
        pd.set_option('display.float_format', '{:.2f}'.format)
        print(results_df.to_string(index=False))
        
        # Find best performing strategy
        if len(results_df) > 0:
            best_strategy = results_df.iloc[0]
            print(f"\n🏆 BEST PERFORMING STRATEGY:")
            print(f"   Strategy: {best_strategy['Strategy']}")
            print(f"   Return: {best_strategy['Return [%]']:.2f}%")
            print(f"   Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
            print(f"   Max Drawdown: {best_strategy['Max Drawdown [%]']:.2f}%")
            print(f"   Win Rate: {best_strategy['Win Rate [%]']:.2f}%")
            print(f"   Number of Trades: {best_strategy['# Trades']}")
        
        return results_df
    else:
        print("No successful backtests completed.")
        return None


if __name__ == "__main__":
    results = main()
