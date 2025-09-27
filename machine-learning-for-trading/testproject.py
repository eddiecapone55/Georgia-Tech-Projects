"""
Student Name: Edward Capone
GT User ID: ecapone3
GT ID: 904057332  

testproject.py
Entry point for the Machine Learning for Trading Project.
This script executes all components needed for the final report:
  - Runs Experiment 1 to compare ManualStrategy and StrategyLearner for both in-sample
    and out-of-sample periods.
  - Runs Experiment 2 to analyze the impact parameter on the StrategyLearner.
  - Generates a QLearner state visit heatmap for diagnostic purposes.
  
Usage:
    PYTHONPATH=../:. python testproject.py
  
Student: Edward Capone (ecapone3)
"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

# Import experiment modules.
import experiment1
import experiment2

# Optional: Quick individual tests.
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import util as ut
import marketsimcode as msc

# Import QLearner for generating the state heatmap.
from QLearner import QLearner

def run_strategy_tests():
    """
    Runs a quick test of both strategies on JPM over an in-sample period,
    and plots normalized portfolio values.
    """
    symbol = "JPM"
    start_val = 100000
    commission = 9.95
    impact = 0.005
    insample_start = dt.datetime(2008, 1, 1)
    insample_end = dt.datetime(2009, 12, 31)
    
    dates = pd.date_range(insample_start, insample_end)
    
    # Test ManualStrategy.
    ms = ManualStrategy(verbose=False, impact=impact, commission=commission)
    ms.add_evidence(symbol=symbol, sd=insample_start, ed=insample_end, sv=start_val)
    trades_manual = ms.testPolicy(symbol=symbol, sd=insample_start, ed=insample_end, sv=start_val)
    portvals_manual = msc.compute_portvals(trades_manual, symbol, start_val, commission, impact)
    norm_manual = portvals_manual / portvals_manual.iloc[0]
    
    # Test StrategyLearner with candidate configuration:
    # Candidate: horizon = 7, trade_penalty = 0.00075.
    stl = StrategyLearner(verbose=False, impact=impact, commission=commission,
                          min_hold=5, trade_penalty=0.00075,
                          alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, horizon=7,
                          bb_low_quantile=0.20, bb_high_quantile=0.80,
                          rsi_low_quantile=0.25, rsi_high_quantile=0.75,
                          mom_low_quantile=0.30, mom_high_quantile=0.70)
    stl.add_evidence(symbol=symbol, sd=insample_start, ed=insample_end, sv=start_val)
    trades_sl = stl.testPolicy(symbol=symbol, sd=insample_start, ed=insample_end, sv=start_val)
    portvals_sl = msc.compute_portvals(trades_sl, symbol, start_val, commission, impact)
    norm_sl = portvals_sl / portvals_sl.iloc[0]
    
    # Benchmark: Buy 1000 shares on the first day and hold.
    trades_bench = pd.DataFrame(0, index=dates, columns=[symbol])
    trades_bench.iloc[0] = 1000
    portvals_bench = msc.compute_portvals(trades_bench, symbol, start_val, commission, impact)
    norm_bench = portvals_bench / portvals_bench.iloc[0]
    
    plt.figure(figsize=(12,8))
    plt.plot(dates, norm_bench[symbol], color='purple', label="Benchmark")
    plt.plot(dates, norm_manual[symbol], color='red', label="ManualStrategy")
    plt.plot(dates, norm_sl[symbol], color='green', label="StrategyLearner (horizon=7, penalty=0.00075)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title("Quick Test: In-Sample Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_qlearner_heatmap():
    """
    Generates a QLearner state visit heatmap for diagnostic purposes.
    This creates a QLearner instance, simulates state visits, and saves the heatmap.
    """
    ql = QLearner(num_states=243, num_actions=3, verbose=False)
    
    import random
    for _ in range(500):
        state = random.randint(0, 242)
        ql.querysetstate(state)
    
    ql.plot_state_heatmap(filename="qlearner_heatmap.png")
    print("State heatmap generated and saved as 'qlearner_heatmap.png'.")

def main():
    # Uncomment the following line to run a quick test.
    # run_strategy_tests()
    
    print("Running Experiment 1: ManualStrategy vs. StrategyLearner Comparison...")
    experiment1.experiment1()
    print("Experiment 1 finished. Check the saved charts and performance summary.")
    
    print("Running Experiment 2: Sensitivity of StrategyLearner to Impact...")
    experiment2.experiment2()
    print("Experiment 2 finished. Check the generated chart and summary metrics.")
    
    print("Generating QLearner state heatmap for diagnostic purposes...")
    generate_qlearner_heatmap()
    
    print("All tests completed.")

if __name__ == "__main__":
    main()