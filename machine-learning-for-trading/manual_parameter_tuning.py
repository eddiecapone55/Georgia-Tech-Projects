import datetime as dt
import pandas as pd
import util as ut
import marketsimcode as msc
from ManualStrategy import ManualStrategy

def tune_manual_parameters(symbol="JPM", 
                           sd=dt.datetime(2008, 1, 1), 
                           ed=dt.datetime(2009, 12, 31), 
                           sv=100000,
                           commission=9.95, 
                           impact=0.005):
    # Define grids for each parameter you want to tune.
    dead_zone_values = [0.25, 0.3, 0.35, 0.4]       # Lower than default may trigger trades more often.
    smoothing_windows = [3, 5, 7]                      # Shorter window for responsiveness.
    min_hold_values = [1, 3, 5]                        # Allow faster repositioning with lower values.
    
    results = []
    # Ensure dates align with market data.
    dates = pd.date_range(sd, ed)
    
    # Run grid search over parameter combinations.
    for dead_zone in dead_zone_values:
        for smoothing_window in smoothing_windows:
            for min_hold in min_hold_values:
                ms = ManualStrategy(
                    verbose=False, 
                    impact=impact, 
                    commission=commission, 
                    dead_zone=dead_zone, 
                    smoothing_window=smoothing_window, 
                    min_hold=min_hold
                )
                ms.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
                trades = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
                portvals = msc.compute_portvals(trades, symbol, sv, commission, impact)
                cum_ret = (portvals.iloc[-1]["Portfolio Value"] / portvals.iloc[0]["Portfolio Value"]) - 1
                results.append({
                    "dead_zone": dead_zone,
                    "smoothing_window": smoothing_window,
                    "min_hold": min_hold,
                    "cum_ret": cum_ret
                })
                print(f"dead_zone: {dead_zone}, smoothing_window: {smoothing_window}, min_hold: {min_hold}, cum_ret: {cum_ret:.4f}")
    
    # Return a DataFrame summarizing the results.
    return pd.DataFrame(results)

if __name__ == "__main__":
    results_df = tune_manual_parameters()
    print("\nTuning Results:")
    print(results_df)
