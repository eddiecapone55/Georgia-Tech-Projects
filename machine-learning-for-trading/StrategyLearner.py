
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
import util as ut
from QLearner import QLearner
import indicators

class StrategyLearner(object):
    def __init__(self, verbose=False, impact=0.0, commission=0.0, min_hold=5, 
                 trade_penalty=0.0010, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, horizon=10,
                 bb_low_quantile=0.20, bb_high_quantile=0.80, 
                 rsi_low_quantile=0.25, rsi_high_quantile=0.75,
                 mom_low_quantile=0.30, mom_high_quantile=0.70):
        """
        Initializes the StrategyLearner.

        Parameters:
          verbose: if True, print debugging info.
          impact: market impact factor.
          commission: commission per trade.
          min_hold: minimum number of days to hold a position during testPolicy.
          trade_penalty: additional cost per unit traded.
          alpha: learning rate for Q-Learner.
          gamma: discount factor.
          rar: initial random action rate.
          radr: random action decay rate.
          dyna: number of dyna updates.
          horizon: lookahead horizon (in days) for reward calculation.
          bb_low_quantile, bb_high_quantile: quantiles for Bollinger Bands.
          rsi_low_quantile, rsi_high_quantile: quantiles for RSI.
          mom_low_quantile, mom_high_quantile: quantiles for Momentum.
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.min_hold = min_hold
        self.trade_penalty = trade_penalty
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.horizon = horizon

        # Store quantile parameters.
        self.bb_low_quantile = bb_low_quantile
        self.bb_high_quantile = bb_high_quantile
        self.rsi_low_quantile = rsi_low_quantile
        self.rsi_high_quantile = rsi_high_quantile
        self.mom_low_quantile = mom_low_quantile
        self.mom_high_quantile = mom_high_quantile

        # Adaptive thresholds will be computed during training.
        self.adaptive_thresholds = {}

        # Using 4 indicators (Bollinger, MACD, RSI, Momentum) each discretized into 3 levels,
        # and 3 levels for position (-1000, 0, 1000) → total state space = 3^4 * 3 = 243.
        self.num_states = 243  
        self.num_actions = 3  # Mapping: 0 -> SHORT (-1000), 1 -> HOLD (0), 2 -> LONG (1000)

        self.learner = QLearner(num_states=self.num_states,
                                num_actions=self.num_actions,
                                alpha=self.alpha,
                                gamma=self.gamma,
                                rar=self.rar,
                                radr=self.radr,
                                dyna=self.dyna,
                                verbose=self.verbose)

    def _compute_adaptive_thresholds(self, prices, symbol="JPM"):
        """
        Compute adaptive thresholds from training data using configurable quantiles:
          - Bollinger Bands: use bb_low_quantile and bb_high_quantile.
          - RSI: use rsi_low_quantile and rsi_high_quantile.
          - Momentum: use mom_low_quantile and mom_high_quantile.
          - MACD: fixed threshold at 0.
        Stores the thresholds in self.adaptive_thresholds.
        """
        bb_series = indicators.bollinger_band(prices)
        rsi_series = indicators.rsi(prices)
        mom_series = indicators.momentum(prices)
        
        bb_lower = bb_series.quantile(self.bb_low_quantile)
        bb_upper = bb_series.quantile(self.bb_high_quantile)
        rsi_lower = rsi_series.quantile(self.rsi_low_quantile)
        rsi_upper = rsi_series.quantile(self.rsi_high_quantile)
        mom_lower = mom_series.quantile(self.mom_low_quantile)
        mom_upper = mom_series.quantile(self.mom_high_quantile)
        macd_threshold = 0
        
        self.adaptive_thresholds = {
            'bb': (bb_lower, bb_upper),
            'rsi': (rsi_lower, rsi_upper),
            'macd': macd_threshold,
            'mom': (mom_lower, mom_upper)
        }
        if self.verbose:
            print("Adaptive Thresholds computed:")
            print(self.adaptive_thresholds)

    def _discretize_state(self, bb, macd_val, rsi_val, m_val, current_position):
        """
        Discretizes indicator values and current position into a state index using adaptive thresholds.
        Mapping:
          Bollinger %B: if < lower threshold -> 0; if > upper threshold -> 2; else -> 1.
          MACD: if >= 0 -> 2; else -> 0.
          RSI: if < lower threshold -> 0; if > upper threshold -> 2; else -> 1.
          Momentum: if < lower threshold -> 0; if > upper threshold -> 2; else -> 1.
          Position: -1000 -> 0; 0 -> 1; 1000 -> 2.
        This produces 3^4 = 81 possible indicator states, and then combining with position yields 243 total states.
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

        mom_lower, mom_upper = self.adaptive_thresholds.get('mom', (-0.01, 0.01))
        if m_val < mom_lower:
            s_mom = 0
        elif m_val > mom_upper:
            s_mom = 2
        else:
            s_mom = 1

        pos_map = {-1000: 0, 0: 1, 1000: 2}
        s_pos = pos_map[current_position]

        state_ind = s_bb * (3**3) + s_macd * (3**2) + s_rsi * 3 + s_mom
        state = state_ind * 3 + s_pos
        return int(state)

    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008,1,1),
                     ed=dt.datetime(2009,12,31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[symbol]

        # Compute adaptive thresholds using the configured quantile parameters.
        self._compute_adaptive_thresholds(prices, symbol)

        bb_series = indicators.bollinger_band(prices)
        macd_series = indicators.macd(prices)
        rsi_series = indicators.rsi(prices)
        mom_series = indicators.momentum(prices)

        episodes = 100  # Increased training episodes.
        for ep in range(episodes):
            if self.verbose:
                print("Starting training episode", ep)
            current_position = 0
            for i in range(len(prices) - 1):
                day = prices.index[i]
                price_current = prices.loc[day]
                bb_val = bb_series.loc[day]
                macd_val = macd_series.loc[day]
                rsi_val = rsi_series.loc[day]
                m_val = mom_series.loc[day]
                state = self._discretize_state(bb_val, macd_val, rsi_val, m_val, current_position)

                if i == 0:
                    action = self.learner.querysetstate(state)
                desired_position = {0: -1000, 1: 0, 2: 1000}[action]
                trade = desired_position - current_position

                # Compute reward using weighted multi-day lookahead with exponential decay.
                if i + self.horizon < len(prices):
                    returns = [prices.iloc[i + j] - price_current for j in range(1, self.horizon + 1)]
                    decay_lambda = 0.7  # candidate decay parameter; you can further tune this.
                    weights = np.array([np.exp(-decay_lambda * (j - 1)) for j in range(1, self.horizon + 1)])
                    weighted_avg_return = np.dot(returns, weights) / np.sum(weights)
                    reward = weighted_avg_return * current_position
                else:
                    price_future = prices.iloc[i + 1]
                    reward = (price_future - price_current) * current_position

                if trade != 0:
                    impact_penalty = price_current * self.impact * (abs(trade) / 1000)
                    cost = self.commission + impact_penalty + self.trade_penalty * abs(trade)
                else:
                    cost = 0.0

                reward = reward - cost
                current_position = desired_position

                bb_next = bb_series.loc[prices.index[i + 1]]
                macd_next = macd_series.loc[prices.index[i + 1]]
                rsi_next = rsi_series.loc[prices.index[i + 1]]
                m_next = mom_series.loc[prices.index[i + 1]]
                next_state = self._discretize_state(bb_next, macd_next, rsi_next, m_next, current_position)

                action = self.learner.query(next_state, reward)
            if self.verbose:
                print("Episode", ep, "completed.")
        if self.verbose:
            print("Training complete. Final Q-table:")
            print(self.learner.Q)
    
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010,1,1),
                   ed=dt.datetime(2011,12,31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[symbol]

        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        current_position = 0
        days_since_trade = self.min_hold
        
        bb_series = indicators.bollinger_band(prices)
        macd_series = indicators.macd(prices)
        rsi_series = indicators.rsi(prices)
        mom_series = indicators.momentum(prices)
        
        self.learner.rar = 0.0  # Disable exploration for testing.
        
        for i in range(len(prices) - 1):
            day = prices.index[i]
            next_day = prices.index[i + 1]
            bb_val = bb_series.loc[day]
            macd_val = macd_series.loc[day]
            rsi_val = rsi_series.loc[day]
            m_val = mom_series.loc[day]
            state = self._discretize_state(bb_val, macd_val, rsi_val, m_val, current_position)
            
            if days_since_trade < self.min_hold:
                desired_position = current_position
            else:
                action = int(np.argmax(self.learner.Q[state, :]))
                desired_position = {0: -1000, 1: 0, 2: 1000}[action]
            
            trade = desired_position - current_position
            trades.loc[next_day, symbol] = trade

            if trade != 0:
                days_since_trade = 0
            else:
                days_since_trade += 1
            current_position = desired_position

        return trades

    def author(self):
        return "ecapone3"

    def study_group(self):
        return "ecapone3"

# Uncomment for a quick local test:
# if __name__ == "__main__":
#     stl = StrategyLearner(verbose=True, impact=0.005, commission=9.95, min_hold=5,
#                             trade_penalty=0.0010, alpha=0.2, gamma=0.9, rar=0.5,
#                             radr=0.99, dyna=0, horizon=10,
#                             bb_low_quantile=0.20, bb_high_quantile=0.80,
#                             rsi_low_quantile=0.25, rsi_high_quantile=0.75,
#                             mom_low_quantile=0.30, mom_high_quantile=0.70)
#     stl.add_evidence(symbol="JPM")
#     trades = stl.testPolicy(symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31))
#     print(trades)