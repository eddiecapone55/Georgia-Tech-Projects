"""
marketsimcode.py
An improved market simulator that accepts a trades DataFrame (instead of a file)
and computes the portfolio values.

API:
    compute_portvals(trades, symbol, start_val, commission, impact)

Parameters:
    trades      - A pandas DataFrame indexed by date with a single column for the symbol.
                  Each value represents the trade for that day.
    symbol      - The stock symbol (str).
    start_val   - Starting cash for the portfolio (int or float).
    commission  - Commission per trade (float), e.g., 0.0 for TOS.
    impact      - Market impact factor (float), e.g., 0.0 for TOS.

Returns:
    portvals    - A pandas DataFrame (single column) of portfolio values indexed by date.

Student Name: Edward Capone
GT User ID: ecapone3
GT ID: 904057332
"""

import pandas as pd
import numpy as np
from util import get_data  # Assumes this function is available

def compute_portvals(trades, symbol, start_val=100000, commission=0.0, impact=0.0):
    """
    Computes the portfolio values from a trades DataFrame.
    """
    # Get the date range from the trades DataFrame index.
    dates = trades.index

    # Fetch price data for the given symbol over these dates.
    df_prices = get_data([symbol], dates)
    prices = df_prices[symbol]

    # Reindex prices to ensure it matches the trades index.
    # Fill any missing values by forward-filling.
    prices = prices.reindex(dates).fillna(method='ffill')

    # Compute cumulative positions (number of shares held).
    positions = trades[symbol].cumsum()

    # Initialize a cash series. Start with the starting cash.
    cash = pd.Series(0.0, index=dates)
    
    # First day's cash adjustment for the initial trade.
    first_trade = trades.iloc[0, 0]
    if first_trade > 0:
        effective_price = prices.iloc[0] * (1 + impact)
        cash.iloc[0] = start_val - (first_trade * effective_price + (commission if first_trade != 0 else 0))
    elif first_trade < 0:
        effective_price = prices.iloc[0] * (1 - impact)
        cash.iloc[0] = start_val - (first_trade * effective_price - (commission if first_trade != 0 else 0))
    else:
        cash.iloc[0] = start_val

    # Loop through each subsequent day to update cash based on trades.
    for i in range(1, len(dates)):
        trade = trades.iloc[i, 0]
        price = prices.iloc[i]
        if trade > 0:
            effective_price = price * (1 + impact)
            trade_cost = trade * effective_price + (commission if trade != 0 else 0)
        elif trade < 0:
            effective_price = price * (1 - impact)
            trade_cost = trade * effective_price - (commission if trade != 0 else 0)
        else:
            trade_cost = 0.0
        cash.iloc[i] = cash.iloc[i-1] - trade_cost

    # Calculate daily portfolio value: value = cash + (number of shares held * current price).
    portvals = cash + positions * prices

    return pd.DataFrame(portvals, columns=["Portfolio Value"])

def author():
    """
    Returns the GT username of the student.
    """
    return "ecapone3"

def study_group():
    """
    Returns a comma-separated string of GT usernames of study group members.
    """
    return "ecapone3"

if __name__ == "__main__":
    # For local testing: create a simple trades DataFrame.
    # This is an example to verify the function works.
    import datetime as dt
    dates = pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 10))
    trades = pd.DataFrame(0, index=dates, columns=["JPM"])
    trades.iloc[0] = 1000
    trades.iloc[-1] = -1000
    portvals = compute_portvals(trades, symbol="JPM", start_val=100000, commission=0.0, impact=0.0)
    print(portvals)
