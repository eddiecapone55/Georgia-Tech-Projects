import datetime as dt
import pandas as pd
import util as ut
import marketsimcode as msc
from StrategyLearner import StrategyLearner

def tune_parameters(symbol="SINE_FAST_NOISE", 
                    sd=dt.datetime(2008, 1, 1), 
                    ed=dt.datetime(2009, 12, 31), 
                    sv=100000,
                    commission=9.95, 
                    impact=0.005):
    # Define parameter grids.
    horizons = [5, 10]  # Shorter and longer lookahead
    trade_penalties = [0.0005, 0.0010]  # Lower trade penalties
    alphas = [0.1, 0.2]    # Learning rates
    gammas = [0.8, 0.9]    # Discount factors
    rars = [0.4, 0.5]      # Exploration rates
    
    results = []
    
    for horizon in horizons:
        for penalty in trade_penalties:
            for alpha in alphas:
                for gamma in gammas:
                    for rar in rars:
                        stl = StrategyLearner(
                            verbose=False,
                            impact=impact,
                            commission=commission,
                            min_hold=5,
                            trade_penalty=penalty,
                            alpha=alpha,
                            gamma=gamma,
                            rar=rar,
                            radr=0.99,
                            dyna=0,
                            horizon=horizon,
                            bb_low_quantile=0.20,
                            bb_high_quantile=0.80,
                            rsi_low_quantile=0.25,
                            rsi_high_quantile=0.75,
                            mom_low_quantile=0.30,
                            mom_high_quantile=0.70
                        )
                        stl.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
                        trades = stl.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
                        portvals = msc.compute_portvals(trades, symbol, sv, commission, impact)
                        cum_ret = (portvals.iloc[-1]["Portfolio Value"] / portvals.iloc[0]["Portfolio Value"]) - 1
                        results.append({
                            "horizon": horizon,
                            "trade_penalty": penalty,
                            "alpha": alpha,
                            "gamma": gamma,
                            "rar": rar,
                            "in_sample_cum_ret": cum_ret
                        })
                        print(f"Horizon: {horizon}, Trade Penalty: {penalty}, Alpha: {alpha}, Gamma: {gamma}, RAR: {rar}, In-Sample Return: {cum_ret:.4f}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    tuning_results = tune_parameters()
    print(tuning_results)
