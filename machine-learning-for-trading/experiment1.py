"""
Student Name: Edward Capone  
GT User ID: ecapone3  
GT ID: 904057332  

Experiment 1:
Compares the performance of the ManualStrategy and StrategyLearner versus a Benchmark
for both in-sample and out-of-sample periods.

For the in-sample period (2008-01-01 to 2009-12-31) and
the out-of-sample period (2010-01-01 to 2011-12-31),
this script generates:
  - A chart with the normalized portfolio values for:
      * Benchmark (purple)
      * ManualStrategy (red)
      * StrategyLearner (green)
    (The chart with three lines does not include the vertical dashed trade entry markers.)
  - A chart with normalized portfolio values for Benchmark and ManualStrategy only,
    with vertical dashed lines marking trade entries (blue for LONG, black for SHORT).
  - Tables summarizing cumulative return, std dev of daily returns, and mean daily returns.
  
Transaction costs for both strategies:
  - Commission: $9.95
  - Impact: 0.005
"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import util as ut
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import marketsimcode as msc

def benchmark_trades(symbol, dates):
    """
    Creates a benchmark trades DataFrame by buying 1000 shares on the first day
    and holding the position (no further trades).
    """
    trades = pd.DataFrame(0, index=dates, columns=[symbol])
    trades.iloc[0] = 1000
    return trades

def compute_performance(portvals):
    """
    Computes cumulative return, standard deviation, and mean of daily returns.
    """
    cum_ret = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    daily_returns = portvals.pct_change().iloc[1:]
    std_daily = daily_returns.std()
    mean_daily = daily_returns.mean()
    return cum_ret, std_daily, mean_daily

def plot_manual_vs_benchmark_vs_strategy(dates, bench, manual, strategy, title):
    """
    Plots normalized portfolio values for Benchmark, ManualStrategy, and StrategyLearner,
    without overlaying vertical dashed lines for trade entries.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(dates, bench, color='purple', label="Benchmark")
    plt.plot(dates, manual, color='red', label="ManualStrategy")
    plt.plot(dates, strategy, color='green', label="StrategyLearner")
    
    # No vertical lines are added in this plot.
    
    plt.legend(loc='best')
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title(title)
    plt.grid(True)
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()

def plot_manual_vs_benchmark(dates, bench, manual, trades_manual, title):
    """
    Plots normalized portfolio values for Benchmark and ManualStrategy,
    and overlays vertical dashed lines for ManualStrategy trade entries.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(dates, bench, color='purple', label="Benchmark")
    plt.plot(dates, manual, color='red', label="ManualStrategy")
    
    # Overlay vertical dashed lines for trade entries from ManualStrategy:
    # Blue for LONG entry (trade > 0), Black for SHORT entry (trade < 0)
    for date, trade in trades_manual.iteritems():
        if trade != 0:
            if trade > 0:
                plt.axvline(x=date, color='blue', linestyle='--', alpha=0.5)
            elif trade < 0:
                plt.axvline(x=date, color='black', linestyle='--', alpha=0.5)
    
    # Custom legend entries for the vertical markers.
    import matplotlib.lines as mlines
    long_entry = mlines.Line2D([], [], color='blue', linestyle='--', label='Long Entry')
    short_entry = mlines.Line2D([], [], color='black', linestyle='--', label='Short Entry')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([long_entry, short_entry])
    labels.extend(['Long Entry', 'Short Entry'])
    plt.legend(handles, labels, loc='best')
    
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title(title)
    plt.grid(True)
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()

def experiment1():
    # Parameters
    symbol = "JPM"
    start_val = 100000
    commission = 9.95
    impact = 0.005

    # Define in-sample and out-of-sample dates
    insample_start = dt.datetime(2008, 1, 1)
    insample_end = dt.datetime(2009, 12, 31)
    outsample_start = dt.datetime(2010, 1, 1)
    outsample_end = dt.datetime(2011, 12, 31)
    
    # Obtain the aligned date indices using ut.get_data
    prices_all_in = ut.get_data([symbol], pd.date_range(insample_start, insample_end))
    insample_dates_aligned = prices_all_in[symbol].index

    prices_all_out = ut.get_data([symbol], pd.date_range(outsample_start, outsample_end))
    outsample_dates_aligned = prices_all_out[symbol].index

    # Initialize strategy instances
    ms = ManualStrategy(verbose=False, impact=impact, commission=commission)
    stl = StrategyLearner(verbose=False, impact=impact, commission=commission)
    
    # Train strategies
    ms.add_evidence(symbol=symbol, sd=insample_start, ed=insample_end, sv=start_val)
    stl.add_evidence(symbol=symbol, sd=insample_start, ed=insample_end, sv=start_val)
    
    # Generate trades DataFrames for in-sample period
    trades_manual_in = ms.testPolicy(symbol=symbol, sd=insample_start, ed=insample_end, sv=start_val)
    trades_sl_in = stl.testPolicy(symbol=symbol, sd=insample_start, ed=insample_end, sv=start_val)
    trades_bench_in = benchmark_trades(symbol, insample_dates_aligned)
    
    # Compute portfolio values for in-sample period
    portvals_manual_in = msc.compute_portvals(trades_manual_in, symbol, start_val, commission, impact)
    portvals_sl_in = msc.compute_portvals(trades_sl_in, symbol, start_val, commission, impact)
    portvals_bench_in = msc.compute_portvals(trades_bench_in, symbol, start_val, commission, impact)
    
    # Normalize portfolio values
    norm_manual_in = portvals_manual_in / portvals_manual_in.iloc[0]
    norm_sl_in = portvals_sl_in / portvals_sl_in.iloc[0]
    norm_bench_in = portvals_bench_in / portvals_bench_in.iloc[0]
    
    # Generate trades DataFrames for out-of-sample period
    trades_manual_out = ms.testPolicy(symbol=symbol, sd=outsample_start, ed=outsample_end, sv=start_val)
    trades_sl_out = stl.testPolicy(symbol=symbol, sd=outsample_start, ed=outsample_end, sv=start_val)
    trades_bench_out = benchmark_trades(symbol, outsample_dates_aligned)
    
    # Compute portfolio values for out-of-sample period
    portvals_manual_out = msc.compute_portvals(trades_manual_out, symbol, start_val, commission, impact)
    portvals_sl_out = msc.compute_portvals(trades_sl_out, symbol, start_val, commission, impact)
    portvals_bench_out = msc.compute_portvals(trades_bench_out, symbol, start_val, commission, impact)
    
    norm_manual_out = portvals_manual_out / portvals_manual_out.iloc[0]
    norm_sl_out = portvals_sl_out / portvals_sl_out.iloc[0]
    norm_bench_out = portvals_bench_out / portvals_bench_out.iloc[0]
    
    # ---- In-Sample Plots ----
    # Plot with Benchmark vs Manual vs StrategyLearner (three lines, no vertical markers)
    plot_manual_vs_benchmark_vs_strategy(
        insample_dates_aligned,
        norm_bench_in["Portfolio Value"],
        norm_manual_in["Portfolio Value"],
        norm_sl_in["Portfolio Value"],
        "In-Sample Performance (Benchmark, Manual, and StrategyLearner)"
    )
    
    # Plot with Benchmark vs Manual only (two lines with vertical markers)
    plot_manual_vs_benchmark(
        insample_dates_aligned,
        norm_bench_in["Portfolio Value"],
        norm_manual_in["Portfolio Value"],
        trades_manual_in[symbol],
        "In-Sample Performance (Benchmark vs. Manual)"
    )
    
    # ---- Out-of-Sample Plots ----
    # Plot with Benchmark vs Manual vs StrategyLearner (three lines, no vertical markers)
    plot_manual_vs_benchmark_vs_strategy(
        outsample_dates_aligned,
        norm_bench_out["Portfolio Value"],
        norm_manual_out["Portfolio Value"],
        norm_sl_out["Portfolio Value"],
        "Out-of-Sample Performance (Benchmark, Manual, and StrategyLearner)"
    )
    
    # Plot with Benchmark vs Manual only (two lines with vertical markers)
    plot_manual_vs_benchmark(
        outsample_dates_aligned,
        norm_bench_out["Portfolio Value"],
        norm_manual_out["Portfolio Value"],
        trades_manual_out[symbol],
        "Out-of-Sample Performance (Benchmark vs. Manual)"
    )
    
    # Compute and print performance metrics (tables printed in console)
    cum_ret_bench_in, std_bench_in, mean_bench_in = compute_performance(portvals_bench_in)
    cum_ret_manual_in, std_manual_in, mean_manual_in = compute_performance(portvals_manual_in)
    cum_ret_sl_in, std_sl_in, mean_sl_in = compute_performance(portvals_sl_in)
    
    cum_ret_bench_out, std_bench_out, mean_bench_out = compute_performance(portvals_bench_out)
    cum_ret_manual_out, std_manual_out, mean_manual_out = compute_performance(portvals_manual_out)
    cum_ret_sl_out, std_sl_out, mean_sl_out = compute_performance(portvals_sl_out)
    
    summary_in = pd.DataFrame({
        "Strategy": ["Benchmark", "ManualStrategy", "StrategyLearner"],
        "Cumulative Return": [cum_ret_bench_in, cum_ret_manual_in, cum_ret_sl_in],
        "Std of Daily Returns": [std_bench_in, std_manual_in, std_sl_in],
        "Mean of Daily Returns": [mean_bench_in, mean_manual_in, mean_sl_in]
    })
    
    summary_out = pd.DataFrame({
        "Strategy": ["Benchmark", "ManualStrategy", "StrategyLearner"],
        "Cumulative Return": [cum_ret_bench_out, cum_ret_manual_out, cum_ret_sl_out],
        "Std of Daily Returns": [std_bench_out, std_manual_out, std_sl_out],
        "Mean of Daily Returns": [mean_bench_out, mean_manual_out, mean_sl_out]
    })
    
    print("In-Sample Performance Summary:")
    print(summary_in)
    print("\nOut-of-Sample Performance Summary:")
    print(summary_out)

def author():
    return "ecapone3"

def study_group():
    return "ecapone3"

if __name__ == "__main__":
    experiment1()
