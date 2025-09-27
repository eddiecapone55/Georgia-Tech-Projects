"""
experiment2.py
Conducts an experiment with the StrategyLearner to examine the effect of different 
impact values on in-sample trading behavior for JPM. Uses a commission of $0.00.

For each chosen impact value, the StrategyLearner is trained and then tested 
on the in-sample period (2008-01-01 to 2009-12-31). Two metrics are recorded:
    - Cumulative Return (CR)
    - Trade Count (number of non-zero trade days)
    
Charts are generated to show how these metrics vary with impact.

Student: Edward Capone (ecapone3)
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import util as ut
from StrategyLearner import StrategyLearner
import marketsimcode as msc

def compute_performance(portvals):
    """
    Computes cumulative return, standard deviation, and mean daily returns.
    
    Assumes portvals is a DataFrame with a single column (e.g., "Portfolio Value").
    """
    # Get the name of the single column.
    col = portvals.columns[0]
    # Extract the scalar values for the first and last day.
    cum_ret = (portvals.iloc[-1][col] / portvals.iloc[0][col]) - 1
    daily_returns = portvals[col].pct_change().iloc[1:]
    std_daily = daily_returns.std()
    mean_daily = daily_returns.mean()
    return cum_ret, std_daily, mean_daily

def count_trades(trades):
    """
    Counts the number of days where the absolute trade is non-zero.
    """
    return (trades != 0).sum().iloc[0]

def experiment2():
    # Define experiment parameters
    symbol = "JPM"
    start_val = 100000
    commission = 0.0  # As per experiment requirement
    # Define a set of different impact values to test.
    impact_values = [0.0, 0.005, 0.01, 0.02]

    # In-sample period for training/testing
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)

    # Lists to store metric measurements for each impact value.
    cum_returns = []
    trade_counts = []

    # Loop over each impact value, train and test the StrategyLearner.
    for impact in impact_values:
        print("Running experiment with impact =", impact)

        # Instantiate the StrategyLearner with the current impact.
        learner = StrategyLearner(verbose=False, impact=impact, commission=commission)
        # Train on the in-sample period.
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=start_val)
        # Test on the in-sample period.
        trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=start_val)
        
        # Compute portfolio values using marketsimcode.
        portvals = msc.compute_portvals(trades, symbol, start_val, commission, impact)
        
        # Compute performance metrics.
        cum_ret, _, _ = compute_performance(portvals)
        num_trades = count_trades(trades)
        
        cum_returns.append(cum_ret)
        trade_counts.append(num_trades)
        print(f"Impact: {impact}, Cumulative Return: {cum_ret:.4f}, Trade Count: {num_trades}")

    # Create a DataFrame summarizing the results.
    summary_df = pd.DataFrame({
        "Impact": impact_values,
        "Cumulative Return": cum_returns,
        "Trade Count": trade_counts
    })
    print("\nExperiment 2 Summary:")
    print(summary_df)

    # Plot the metrics versus Impact.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot cumulative return vs. impact.
    ax1.plot(impact_values, cum_returns, marker='o', linestyle='-', color='blue')
    ax1.set_xlabel("Impact")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Cumulative Return vs. Impact")
    ax1.grid(True)
    
    # Plot trade count vs. impact.
    ax2.plot(impact_values, trade_counts, marker='o', linestyle='-', color='red')
    ax2.set_xlabel("Impact")
    ax2.set_ylabel("Trade Count")
    ax2.set_title("Trade Count vs. Impact")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("experiment2_results.png")
    plt.close()

if __name__ == "__main__":
    experiment2()
    
def author():
    return "ecapone3"

def study_group():
    return "ecapone3"
