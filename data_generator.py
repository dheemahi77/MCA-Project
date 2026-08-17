import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_data(start_date, end_date, volatility=0.05, trend=0.02,
                             seasonal_amplitude=3.0, noise_level=0.3,
                             base_price=60.0):
    """
    Generate synthetic oil price data with trend, seasonality, and noise.

    Improvements over baseline:
    -  Lower default volatility and noise for more learnable patterns.
    -  Uses a mean-reverting (Ornstein-Uhlenbeck) random walk instead of a
       pure random walk so prices stay realistic over long horizons.
    -  Adds a secondary 90-day (quarterly) seasonal cycle.
    -  Skips weekends to mimic real trading calendars.

    Parameters
    ----------
    start_date : datetime
    end_date   : datetime
    volatility : float   – OU noise scale (lower = smoother)
    trend      : float   – long-run drift per day
    seasonal_amplitude : float
    noise_level : float  – micro-noise amplitude
    base_price  : float  – starting price level

    Returns
    -------
    pd.DataFrame with 'Date' and 'Price' columns
    """
    # --- Build date range (trading days only) ---
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:   # Mon-Fri only
            date_list.append(current_date)
        current_date += timedelta(days=1)

    n = len(date_list)
    t = np.arange(n, dtype=float)

    # --- Deterministic components ---
    trend_comp = base_price + t * trend

    annual_season = seasonal_amplitude * np.sin(2 * np.pi * t / 252)          # ~252 trading days/year
    quarterly_season = (seasonal_amplitude * 0.4) * np.sin(2 * np.pi * t / 63)

    # --- Stochastic component: Ornstein-Uhlenbeck process ---
    # dX = theta*(mu - X)*dt + sigma*dW
    theta = 0.02    # mean-reversion speed
    mu = 0.0        # long-run mean of deviation
    sigma = volatility
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = ou[i-1] + theta * (mu - ou[i-1]) + sigma * np.random.randn()

    # --- Micro noise ---
    noise = np.random.normal(0, noise_level, n)

    # --- Combine ---
    prices = trend_comp + annual_season + quarterly_season + ou + noise
    prices = np.maximum(prices, 1.0)   # floor at $1

    return pd.DataFrame({'Date': date_list, 'Price': prices})
