import pytest
import numpy as np
from backend.src.services.technical.indicators import technical_engine

def test_ema_calculation():
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    ema9 = technical_engine.calculate_ema(prices, 9)
    assert len(ema9) == len(prices)
    assert not np.isnan(ema9[-1])
    assert ema9[-1] < prices[-1] # Uptrend EMA lags current price

def test_rsi_calculation():
    # Steadily rising prices
    prices = np.linspace(100, 150, 30)
    rsi = technical_engine.calculate_rsi(prices, 14)
    assert rsi[-1] > 70.0 # Strongly bullish

    # Steadily falling prices
    falling = np.linspace(150, 100, 30)
    rsi_fall = technical_engine.calculate_rsi(falling, 14)
    assert rsi_fall[-1] < 30.0 # Strongly bearish

def test_atr_and_bollinger():
    highs = np.array([102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116])
    lows = np.array([98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])
    closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])

    atr = technical_engine.calculate_atr(highs, lows, closes, 14)
    assert atr > 0.0

    mid, up, low, width, pct_b = technical_engine.calculate_bollinger_bands(closes, 10)
    assert up > mid > low
    assert width > 0.0
