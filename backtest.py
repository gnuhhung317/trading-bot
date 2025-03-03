import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple
import logging

# Import modules
from config import SYMBOLS, INITIAL_CAPITAL, LEVERAGE, RISK_PER_TRADE
from modules.data_loader import DataLoader
from modules.indicators import Indicators
from modules.strategy import MomentumStrategy
from modules.risk_manager import RiskManager
from modules.utils import setup_logging, plot_chart, calculate_drawdown

logger = logging.getLogger(__name__)

def backtest_strategy(symbol: str, start_date: str, end_date: str, 
                     interval: str = '5m', capital: float = INITIAL_CAPITAL, 
                     leverage: int = LEVERAGE, risk_per_trade: float = RISK_PER_TRADE,
                     plot_results: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest chiến lược trên một symbol
    
    Args:
        symbol (str): Symbol cần backtest
        start_date (str): Ngày bắt đầu (YYYY-MM-DD)
        end_date (str): Ngày kết thúc (YYYY-MM-DD)
        interval (str): Khung thời gian
        capital (float): Vốn ban đầu
        leverage (int): Đòn bẩy
        risk_per_trade (float): Mức rủi ro mỗi giao dịch
        plot_results (bool): Vẽ biểu đồ kết quả
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame giao dịch và thông tin backtest
    """
    print(f"\n{'='*60}")
    print(f"BACKTEST: {symbol} từ {start_date} đến {end_date}")
    print(f"{'='*60}")
    
    # Tải dữ liệu
    data_loader = DataLoader(sandbox=True)
    
    # Tăng thời gian lấy dữ liệu quá khứ cho khung thời gian cao hơn
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    adjusted_start = (start_date_obj - timedelta(days=60)).strftime("%Y-%m-%d")
    
    try:
        # Tải dữ liệu
        df = data_loader.get_historical_data(symbol, interval, lookback_days=(datetime.strptime(end_date, "%Y-%m-%d") - start_date_obj).days)
        higher_tf = data_loader.get_historical_data(
            symbol, 
            data_loader.get_higher_timeframe(interval), 
            lookback_days=(datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(adjusted_start, "%Y-%m-%d")).days
        )
        
        if df.empty or higher_tf.empty:
            print(f"Không đủ dữ liệu cho {symbol}")
            return pd.DataFrame(), {"error": "Không đủ dữ liệu"}
        
        # Lọc theo khoảng thời gian
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Thêm chỉ báo
        df = Indicators.add_signal_indicators(df)
        higher_tf = Indicators.add_trend_indicators(higher_tf)
        
        # Khởi tạo chiến lược và quản lý rủi ro
        strategy = MomentumStrategy()
        risk_manager = RiskManager(initial_capital=capital, risk_per_trade=risk_per_trade)
        
        # Khởi tạo biến theo dõi
        balance = capital
        positions = {}  # symbol -> position details
        trades = []
        equity_curve = [balance]
        
        # Chạy backtest
        for i in range(1, len(df)):
            current_time = df.index[i]
            
            # Kiểm tra nếu đang có vị thế, xem có nên thoát không
            for symbol, position in list(positions.items()):
                if "partial_exits" not in position:
                    position["partial_exits"] = [False, False]
                    
                # Tạo DataFrame từ dữ liệu đến hiện tại
                current_df = df.iloc[:i+1].copy()
                
                # Kiểm tra thoát lệnh
                should_exit, exit_reason, exit_price = strategy.should_exit(
                    current_df,
                    position["type"],
                    position["entry_price"],
                    position["stop_loss"],
                    position["target1"],
                    position["target2"],
                    position["partial_exits"]
                )
                
                if should_exit:
                    # Tính toán lợi nhuận
                    exit_price = exit_price or current_df.iloc[-1]["close"]
                    pnl_percent = 0
                    
                    if position["type"] == "LONG":
                        # Đảm bảo thoát lệnh không bị trượt giá quá xa
                        if exit_reason == "Stop Loss" and exit_price < position["stop_loss"] * 0.95:
                            # Nếu có trượt giá nghiêm trọng, hạn chế mức lỗ
                            exit_price = position["stop_loss"] * 0.95
                            print(f"WARNING: Stop loss slippage detected, limiting loss at {exit_price}")
                        
                        pnl_percent = (exit_price - position["entry_price"]) / position["entry_price"] * 100 * leverage
                    else:  # SHORT
                        # Đảm bảo thoát lệnh không bị trượt giá quá xa
                        if exit_reason == "Stop Loss" and exit_price > position["stop_loss"] * 1.05:
                            # Nếu có trượt giá nghiêm trọng, hạn chế mức lỗ
                            exit_price = position["stop_loss"] * 1.05
                            print(f"WARNING: Stop loss slippage detected, limiting loss at {exit_price}")
                        
                        pnl_percent = (position["entry_price"] - exit_price) / position["entry_price"] * 100 * leverage
                    
                    # Giới hạn mức lỗ tối đa
                    if pnl_percent < -100:
                        logger.warning(f"Extreme loss detected: {pnl_percent:.2f}%. Capping at -100%")
                        pnl_percent = -100
                    
                    pnl_amount = position["size"] * position["entry_price"] * pnl_percent / 100
                    
                    # Cập nhật balance
                    balance += pnl_amount
                    
                    # Lưu thông tin giao dịch
                    trade = {
                        "symbol": symbol,
                        "type": position["type"],
                        "entry_time": position["entry_time"],
                        "entry_price": position["entry_price"],
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "size": position["size"],
                        "profit": pnl_amount,
                        "profit_percent": pnl_percent,
                        "reason": exit_reason
                    }
                    trades.append(trade)
                    
                    print(f"EXIT {position['type']} {symbol} at {exit_price:.4f} | P&L: {pnl_amount:.4f} USDT ({pnl_percent:.2f}%) | Reason: {exit_reason}")
                    
                    # Đóng vị thế
                    del positions[symbol]
                
                elif exit_reason and exit_reason.startswith("Partial Exit"):
                    # Tính toán partial exit
                    exit_price = exit_price or current_df.iloc[-1]["close"]
                    close_percent = 0.5 if exit_reason == "Partial Exit Target 1" else 0.3
                    
                    # Kích thước đóng
                    close_size = position["size"] * close_percent
                    position["size"] -= close_size
                    
                    # Tính lợi nhuận
                    pnl_percent = 0
                    if position["type"] == "LONG":
                        pnl_percent = (exit_price - position["entry_price"]) / position["entry_price"] * 100 * leverage
                    else:  # SHORT
                        pnl_percent = (position["entry_price"] - exit_price) / position["entry_price"] * 100 * leverage
                    
                    pnl_amount = close_size * position["entry_price"] * pnl_percent / 100
                    
                    # Cập nhật balance và partial exits
                    balance += pnl_amount
                    if exit_reason == "Partial Exit Target 1":
                        position["partial_exits"][0] = True
                    else:
                        position["partial_exits"][1] = True
                    
                    print(f"PARTIAL EXIT {position['type']} {symbol} at {exit_price:.4f} | Size: {close_size:.4f} | P&L: {pnl_amount:.4f} USDT ({pnl_percent:.2f}%) | Reason: {exit_reason}")
            
            # Tìm cơ hội mở vị thế mới (nếu chưa có)
            if symbol not in positions:
                # Tạo DataFrame đến hiện tại
                current_df = df.iloc[:i+1].copy()
                
                # Map xu hướng cao hơn
                higher_df_subset = higher_tf[higher_tf.index <= current_time]
                
                if not higher_df_subset.empty:
                    # Phân tích
                    analysis = strategy.analyze(current_df, higher_df_subset)
                    
                    if analysis["signal"] in ["LONG", "SHORT"]:
                        entry_price = current_df.iloc[-1]["close"]
                        stop_loss = analysis["details"]["stop_loss"]
                        
                        # Tính kích thước vị thế
                        position_details = risk_manager.calculate_position_size(
                            balance, entry_price, stop_loss, leverage, symbol
                        )
                        
                        if position_details["size"] > 0:
                            # Mở vị thế mới
                            positions[symbol] = {
                                "type": analysis["signal"],
                                "entry_time": current_time,
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "target1": analysis["details"]["target1"],
                                "target2": analysis["details"]["target2"],
                                "size": position_details["size"],
                                "partial_exits": [False, False]
                            }
                            
                            print(f"ENTER {analysis['signal']} {symbol} at {entry_price:.4f} | Size: {position_details['size']:.4f} | Stop: {stop_loss:.4f} | Risk: {position_details['risk_amount']:.4f} USDT")
            
            # Ghi nhận equity curve
            equity_curve.append(balance)
        
        # Đóng các vị thế còn lại ở cuối backtest
        last_price = df.iloc[-1]["close"]
        for symbol, position in list(positions.items()):
            # Tính toán lợi nhuận
            pnl_percent = 0
            if position["type"] == "LONG":
                pnl_percent = (last_price - position["entry_price"]) / position["entry_price"] * 100 * leverage
            else:  # SHORT
                pnl_percent = (position["entry_price"] - last_price) / position["entry_price"] * 100 * leverage
            
            pnl_amount = position["size"] * position["entry_price"] * pnl_percent / 100
            
            # Cập nhật balance
            balance += pnl_amount
            
            # Lưu thông tin giao dịch
            trade = {
                "symbol": symbol,
                "type": position["type"],
                "entry_time": position["entry_time"],
                "entry_price": position["entry_price"],
                "exit_time": df.index[-1],
                "exit_price": last_price,
                "size": position["size"],
                "profit": pnl_amount,
                "profit_percent": pnl_percent,
                "reason": "End of Backtest"
            }
            trades.append(trade)
            
            print(f"END-OF-TEST EXIT {position['type']} {symbol} at {last_price:.4f} | P&L: {pnl_amount:.4f} USDT ({pnl_percent:.2f}%)")
        
        # Tạo DataFrame giao dịch
        trades_df = pd.DataFrame(trades)
        
        # Tính toán thống kê
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["profit"] > 0])
        losing_trades = len(trades_df[trades_df["profit"] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(trades_df["profit"]) if not trades_df.empty else 0
        profit_percent = (balance - capital) / capital * 100
        
        avg_profit = trades_df[trades_df["profit"] > 0]["profit"].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df["profit"] <= 0]["profit"].mean() if losing_trades > 0 else 0
        
        risk_reward = abs(avg_profit / avg_loss) if avg_loss < 0 else 0
        
        # Tính drawdown
        drawdown_info = calculate_drawdown(equity_curve)
        
        # Tạo báo cáo thống kê
        report = {
            "symbol": symbol,
            "period": f"{start_date} to {end_date}",
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "profit_percent": profit_percent,
            "max_drawdown": drawdown_info["max_drawdown"],
            "max_drawdown_percent": drawdown_info["max_drawdown_percent"],
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "risk_reward": risk_reward,
            "final_balance": balance
        }
        
        # In báo cáo
        print("\n" + "="*30 + " BACKTEST REPORT " + "="*30)
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Balance: {capital} USDT")
        print(f"Final Balance: {balance:.2f} USDT")
        print(f"Total Profit: {total_profit:.2f} USDT ({profit_percent:.2f}%)")
        print(f"Max Drawdown: {drawdown_info['max_drawdown']:.2f} USDT ({drawdown_info['max_drawdown_percent']:.2f}%)")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate*100:.2f}% ({winning_trades}/{total_trades})")
        print(f"Avg Profit: {avg_profit:.2f} USDT")
        print(f"Avg Loss: {avg_loss:.2f} USDT")
        print(f"Risk-Reward Ratio: {risk_reward:.2f}")
        print("="*75)
        
        # Vẽ biểu đồ
        if plot_results and not trades_df.empty:
            # Đảm bảo thư mục tồn tại
            if not os.path.exists("results"):
                os.makedirs("results")
                
            # Tạo DataFrame equity curve
            equity_df = pd.DataFrame(index=df.index)
            equity_df['balance'] = capital  # Giá trị ban đầu
            
            # Cập nhật equity curve tại thời điểm exit
            for trade in trades_df.to_dict('records'):
                exit_time = trade.get('exit_time')
                if exit_time and pd.notna(exit_time):
                    idx = equity_df.index.get_indexer([exit_time], method='nearest')[0]
                    if idx >= 0:
                        pnl = trade.get('profit', 0)
                        equity_df.loc[equity_df.index[idx:], 'balance'] += pnl
            
            # Vẽ biểu đồ với equity curve
            plt.figure(figsize=(12, 8))
            
            # Panel 1: Giá
            plt.subplot(3, 1, 1)
            plt.plot(df.index, df['close'], color='blue')
            plt.title(f"{symbol} Price")
            plt.grid(True)
            
            # Panel 2: Equity Curve
            plt.subplot(3, 1, 2)
            plt.plot(equity_df.index, equity_df['balance'], color='green')
            plt.axhline(y=capital, color='black', linestyle='--')
            plt.fill_between(equity_df.index, equity_df['balance'], capital,
                           where=(equity_df['balance'] >= capital), color='green', alpha=0.3)
            plt.fill_between(equity_df.index, equity_df['balance'], capital,
                           where=(equity_df['balance'] < capital), color='red', alpha=0.3)
            
            # Đảm bảo trục y hiển thị cả giá trị âm
            min_balance = min(equity_df['balance'].min(), 0)
            max_balance = max(equity_df['balance'].max(), capital * 1.1)
            plt.ylim(min_balance * 0.9, max_balance * 1.1)
            
            plt.title("Equity Curve")
            plt.grid(True)
            
            # Panel 3: Drawdown
            plt.subplot(3, 1, 3)
            rolling_max = equity_df['balance'].cummax()
            drawdown = ((equity_df['balance'] - rolling_max) / rolling_max) * 100
            plt.fill_between(equity_df.index, drawdown, 0, color='red', alpha=0.3)
            plt.title("Drawdown %")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"results/{symbol}_{start_date}_{end_date}_backtest.png", dpi=150)
            plt.close()
            
            # Lưu trades
            trades_df.to_csv(f"results/trades_{symbol}_{start_date}_{end_date}.csv")
        
        return trades_df, report
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {"error": str(e)}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--symbol', type=str, help='Symbol to backtest (e.g., BTCUSDT)')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-08-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='5m', help='Time interval (1m, 5m, 15m, 1h, etc.)')
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL, help='Initial capital')
    parser.add_argument('--leverage', type=int, default=LEVERAGE, help='Leverage')
    parser.add_argument('--risk', type=float, default=RISK_PER_TRADE, help='Risk per trade (0-1)')
    parser.add_argument('--all', action='store_true', help='Run on all predefined symbols')
    
    args = parser.parse_args()
    
    # Thiết lập logging
    setup_logging()
    
    # Tạo directory cho kết quả
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Chạy backtest
    if args.all:
        # Backtest trên tất cả symbols
        reports = []
        for symbol in SYMBOLS:
            _, report = backtest_strategy(
                symbol, args.start, args.end, args.interval,
                args.capital, args.leverage, args.risk, True
            )
            reports.append(report)
            time.sleep(1)  # Tránh rate limit
            
        # Tạo báo cáo tổng hợp
        summary_df = pd.DataFrame(reports)
        summary_df.to_csv(f"results/summary_{args.start}_{args.end}.csv", index=False)
        print("\nBacktest completed for all symbols. Summary saved to results folder.")
    else:
        # Backtest trên một symbol
        symbol = args.symbol or SYMBOLS[0]
        backtest_strategy(
            symbol, args.start, args.end, args.interval,
            args.capital, args.leverage, args.risk, True
        )
        print(f"\nBacktest completed for {symbol}. Results saved to results folder.")

if __name__ == "__main__":
    main()
