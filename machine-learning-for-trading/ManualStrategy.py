"""  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  
  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  
Atlanta, Georgia 30332  
All Rights Reserved  
  
Template code for CS 4646/7646  
  
Georgia Tech asserts copyright ownership of this template and all derivative  
works, including solutions to the projects assigned in this course. Students  
and other users of this template code are advised not to share it with others  
or to make it available on publicly viewable websites including repositories  
such as github and gitlab.  This copyright statement should not be removed  
or edited.  
  
We do grant permission to share solutions privately with non-students such  
as potential employers. However, sharing with other current or future  
students of CS 4646 is prohibited and subject to being investigated as a  
GT honor code violation.  
  
-----do not edit anything above this line---  
  
Student Name: Edward Capone  
GT User ID: ecapone3  
GT ID: 904057332  
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import util as ut  
import indicators  

class ManualStrategy(object):
    def __init__(self, verbose=False, impact=0.0, commission=0.0, dead_zone=0.30, 
                 smoothing_window=3, min_hold=1, weights=None):
        """
        Parameters:
          verbose: if True, print debugging info.
          impact: market impact factor.
          commission: commission per trade.
          dead_zone: minimum absolute smoothed combined signal required to trigger a trade.
                    (Default updated to 0.30 based on parameter tuning.)
          smoothing_window: number of days over which to average the combined signal.
                    (Default updated to 3 for increased responsiveness.)
          min_hold: minimum number of days to hold a position.
                    (Default updated to 1 to allow more agile position changes.)
          weights: optional list of weights for indicators [Bollinger, MACD, RSI, Momentum, Stochastic].
                   If None, default equal/weighted values are used.
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.dead_zone = dead_zone          
        self.smoothing_window = smoothing_window  
        self.min_hold = min_hold
        if weights is None:
            self.weights = [0.2, 0.3, 0.3, 0.1, 0.1]
        else:
            self.weights = weights

        # Adaptive thresholds computed during training; if not available, use fixed defaults.
        self.adaptive_thresholds = {
            'bb': (-0.1, 0.1),
            'rsi': (30, 70),
            'mom': (0, 0)  # default; ideally should be computed from training data.
        }

    def _compute_adaptive_thresholds(self, prices):
        bb_series = indicators.bollinger_band(prices)
        rsi_series = indicators.rsi(prices)
        mom_series = indicators.momentum(prices)
        bb_lower = bb_series.quantile(0.10)
        bb_upper = bb_series.quantile(0.90)
        rsi_lower = rsi_series.quantile(0.30)
        rsi_upper = rsi_series.quantile(0.70)
        mom_lower = mom_series.quantile(0.25)
        mom_upper = mom_series.quantile(0.75)
        self.adaptive_thresholds = {
            'bb': (bb_lower, bb_upper),
            'rsi': (rsi_lower, rsi_upper),
            'mom': (mom_lower, mom_upper)
        }
        if self.verbose:
            print("ManualStrategy adaptive thresholds computed:")
            print(self.adaptive_thresholds)

    def _discretize_state(self, bb, macd_val, rsi_val, m_val, current_position):
        """
        Discretizes indicator values and current position into a state value.
        For ManualStrategy, we use adaptive thresholds for Bollinger, RSI, and Momentum.
        MACD still uses 0 as fixed threshold.
        """
        bb_lower, bb_upper = self.adaptive_thresholds.get('bb', (-0.1, 0.1))
        if bb < bb_lower:
            s_bb = 0
        elif bb > bb_upper:
            s_bb = 2
        else:
            s_bb = 1

        s_macd = 2 if macd_val >= 0 else 0

        rsi_lower, rsi_upper = self.adaptive_thresholds.get('rsi', (30,70))
        if rsi_val < rsi_lower:
            s_rsi = 0
        elif rsi_val > rsi_upper:
            s_rsi = 2
        else:
            s_rsi = 1

        mom_lower, mom_upper = self.adaptive_thresholds.get('mom', (0,0))
        if m_val < mom_lower:
            s_mom = 0
        elif m_val > mom_upper:
            s_mom = 2
        else:
            s_mom = 1

        pos_map = {-1000: 0, 0: 1, 1000: 2}
        s_pos = pos_map[current_position]

        # Combine the five signals (here we can ignore the weight for discretization).
        # For simplicity, we combine just the first four for state creation.
        # That yields 3^4 = 81 states; then multiply by 3 for position: 243 states.
        state_ind = s_bb * (3**3) + s_macd * (3**2) + s_rsi * 3 + s_mom  # 0 .. 80
        state = state_ind * 3 + s_pos  # 0 .. 242
        return int(state)

    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        # For ManualStrategy, we typically do not learn. But we can compute adaptive thresholds.
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[symbol]
        self._compute_adaptive_thresholds(prices)
        # Optionally, you may precompute indicator signals here.
        pass

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[symbol]
        # Compute adaptive thresholds for the period (or use thresholds computed earlier)
        self._compute_adaptive_thresholds(prices)
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        
        # Compute indicator series.
        bb = indicators.bollinger_band(prices)
        macd_val = indicators.macd(prices)
        rsi_val = indicators.rsi(prices)
        mom_val = indicators.momentum(prices)
        
        # Compute individual signals with weights (optional) and combine.
        signal_bb = bb.apply(lambda x: 1 if x < -0.1 else (-1 if x > 0.1 else 0))
        signal_macd = macd_val.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        signal_rsi = rsi_val.apply(lambda x: 1 if x < 30 else (-1 if x > 70 else 0))
        signal_mom = mom_val.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        weighted_signal = (self.weights[0]*signal_bb +
                           self.weights[1]*signal_macd +
                           self.weights[2]*signal_rsi +
                           self.weights[3]*signal_mom)
        total_weight = sum(self.weights[:4])
        combined_signal = weighted_signal / total_weight
        
        smoothed_signal = combined_signal.rolling(window=self.smoothing_window, min_periods=1).mean()
        
        def decide_position(x):
            if x > self.dead_zone:
                return 1000
            elif x < -self.dead_zone:
                return -1000
            else:
                return 0
        desired_positions = smoothed_signal.apply(decide_position)
        
        current_position = 0
        days_since_trade = self.min_hold
        for i in range(len(prices)):
            day = prices.index[i]
            if days_since_trade < self.min_hold:
                desired = current_position
            else:
                desired = desired_positions.loc[day]
            trade = desired - current_position
            if trade != 0:
                trades.loc[day, symbol] = trade
                days_since_trade = 0
            else:
                days_since_trade += 1
            current_position = desired
        
        if self.verbose:
            print("Trades DataFrame:\n", trades)
        return trades

    def author(self):
        return "ecapone3"

    def study_group(self):
        return "ecapone3"

# Uncomment for a quick local test:
# if __name__ == "__main__":
#     ms = ManualStrategy(verbose=True, impact=0.005, commission=9.95)
#     trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)
#     print(trades)
