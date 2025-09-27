"""
indicators.py
Implements five technical indicators:
1. Bollinger Bands %B: (price - SMA) / (2 * stdev)
2. Momentum: (price / price[t-N]) - 1
3. MACD: EMA12 - EMA26
4. Stochastic Oscillator: (price - min(price))/(max(price)-min(price)) * 100
5. RSI: Relative Strength Index based on rolling average gains/losses

Each function returns a single real results vector.

Student Name: Edward Capone		  	   		 	 	 			  		 			     			  	 
GT User ID: ecapone3  		  	   		 	 	 			  		 			     			  	 
GT ID: 904057332  		  	   		 	 	 			  		 			     			  	 
"""  

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data  

# ---------------- Technical Indicator Functions ---------------- #

def bollinger_band(price_series, window=20):
    """
    Computes the Bollinger Bands %B indicator.
    
    bb_value[t] = (price[t] - SMA[t]) / (2 * stdev[t])
    
    Parameters:
        price_series (pd.Series): Series of prices.
        window (int): Rolling window size (default 20).
        
    Returns:
        pd.Series: Bollinger Bands %B values.
    """
    sma = price_series.rolling(window=window).mean()
    stdev = price_series.rolling(window=window).std()
    bb = (price_series - sma) / (2 * stdev)
    return bb

def momentum(price_series, window=10):
    """
    Computes the Momentum indicator.
    
    momentum[t] = (price[t] / price[t-window]) - 1
    
    Parameters:
        price_series (pd.Series): Series of prices.
        window (int): Lookback period (default 10).
        
    Returns:
        pd.Series: Momentum values.
    """
    mom = (price_series / price_series.shift(window)) - 1
    return mom

def macd(price_series):
    """
    Computes the MACD indicator as the difference between 12-day EMA and 26-day EMA.
    
    Parameters:
        price_series (pd.Series): Series of prices.
        
    Returns:
        pd.Series: MACD values.
    """
    ema12 = price_series.ewm(span=12, adjust=False).mean()
    ema26 = price_series.ewm(span=26, adjust=False).mean()
    macd_val = ema12 - ema26
    return macd_val

def stochastic(price_series, window=14):
    """
    Computes the Stochastic Oscillator.
    
    %K = ((price - lowest_low) / (highest_high - lowest_low)) * 100
    
    Parameters:
        price_series (pd.Series): Series of prices.
        window (int): Lookback period (default 14).
        
    Returns:
        pd.Series: Stochastic Oscillator values (in percentage).
    """
    rolling_min = price_series.rolling(window=window).min()
    rolling_max = price_series.rolling(window=window).max()
    stoch = (price_series - rolling_min) / (rolling_max - rolling_min)
    return stoch * 100

def rsi(price_series, window=14):
    """
    Computes the Relative Strength Index (RSI) using a simple moving average method.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = average gain / average loss
    
    Parameters:
        price_series (pd.Series): Series of prices.
        window (int): Lookback period (default 14).
        
    Returns:
        pd.Series: RSI values.
    """
    delta = price_series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    avg_gain = up.rolling(window=window, min_periods=window).mean()
    avg_loss = down.rolling(window=window, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

# ---------------- Plotting / Run Function ---------------- #

def run():
    """
    Retrieves JPM price data for the period 2008-01-01 to 2009-12-31,
    computes all five technical indicators, and generates charts for each.
    
    Charts are saved as .png files without being displayed.
    """
    # Define parameters and retrieve data
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = "JPM"
    dates = pd.date_range(sd, ed)
    df_prices = get_data([symbol], dates)
    price = df_prices[symbol]

    # Normalize price for visualization (set first value to 1.0)
    norm_price = price / price.iloc[0]

    # Compute indicators
    bb = bollinger_band(price)
    mom = momentum(price)
    macd_val = macd(price)
    stoch = stochastic(price)
    rsi_val = rsi(price)
    
    # ---------------- Chart 1: Bollinger Bands ---------------- #
    window_bb = 20
    sma = price.rolling(window=window_bb).mean()
    stdev = price.rolling(window=window_bb).std()
    upper_band = sma + 2 * stdev
    lower_band = sma - 2 * stdev

    plt.figure(figsize=(10, 6))
    plt.plot(norm_price.index, norm_price, label="Normalized Price", color='blue')

    # Normalize bands for comparability by dividing by price.iloc[0]
    plt.plot(sma.index, sma / price.iloc[0], label="Normalized SMA", color='orange')
    plt.plot(upper_band.index, upper_band / price.iloc[0], label="Normalized Upper Band", color='green')
    plt.plot(lower_band.index, lower_band / price.iloc[0], label="Normalized Lower Band", color='red')

    plt.title("Bollinger Bands for " + symbol)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.savefig("bollinger_bands.png")
    plt.close()
   
    # ---------------- Chart 2: Momentum ---------------- #
    plt.figure(figsize=(10, 6))
    plt.plot(norm_price.index, norm_price, label="Normalized Price", color='blue')
    plt.plot(mom.index, mom, label="Momentum", color='red')
    plt.title("Momentum Indicator for " + symbol)
    plt.xlabel("Date")
    plt.ylabel("Momentum")
    plt.legend()
    plt.savefig("momentum.png")
    plt.close()
    
    # ---------------- Chart 3: MACD ---------------- #
    plt.figure(figsize=(10, 6))
    plt.plot(norm_price.index, norm_price, label="Normalized Price", color='blue')
    plt.plot(macd_val.index, macd_val, label="MACD", color='purple')
    plt.title("MACD Indicator for " + symbol)
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.legend()
    plt.savefig("macd.png")
    plt.close()
    
        # ---------------- Chart 4: Stochastic Oscillator with Subplots ---------------- #
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Top subplot: Normalized Price
    ax1.plot(norm_price.index, norm_price, label="Normalized Price", color='blue')
    ax1.set_title("Normalized Price for " + symbol)
    ax1.set_ylabel("Normalized Price")
    ax1.legend(loc='best')

    # Bottom subplot: Stochastic Oscillator
    ax2.plot(stoch.index, stoch, label="Stochastic Oscillator", color='magenta')
    ax2.set_title("Stochastic Oscillator for " + symbol)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stochastic Oscillator (%)")
    ax2.legend(loc='upper right')  # Move legend to top-right

    plt.tight_layout()
    plt.savefig("stochastic.png")
    plt.close()
    
    # ---------------- Chart 5: RSI with Subplots ---------------- #
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Top subplot: Normalized Price
    ax1.plot(norm_price.index, norm_price, label="Normalized Price", color='blue')
    ax1.set_title("Normalized Price for " + symbol)
    ax1.set_ylabel("Normalized Price")
    ax1.legend(loc='best')

    # Bottom subplot: RSI
    ax2.plot(rsi_val.index, rsi_val, label="RSI", color='green')
    ax2.set_title("RSI for " + symbol)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RSI")
    ax2.legend(loc='upper right')  # Adjust the legend location if desired

    plt.tight_layout()
    plt.savefig("rsi.png")
    plt.close()

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
    # Run the indicator calculations and generate charts.
    run()