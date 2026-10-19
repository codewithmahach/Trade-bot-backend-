import pytest
from backend.src.services.ml.mln_network import mln_engine

def test_mln_neural_network_inference():
    tech_1m = {
        "current_price": 50000.0,
        "ema9": 50100.0,
        "ema21": 50020.0,
        "ema50": 49800.0,
        "vwap": 49950.0,
        "rsi": 62.0,
        "macd_hist": 15.0,
        "atr_pct": 0.4,
        "relative_volume": 2.2,
        "bb_pct_b": 0.75,
        "trend_bullish": True
    }
    tech_3m = {
        "trend_bullish": True,
        "trend_bearish": False
    }
    tech_5m = {
        "trend_bullish": True
    }
    order_book = {
        "imbalance": 0.72,
        "spread": 0.015
    }

    res = mln_engine.predict_deep_probabilities(tech_1m, tech_3m, tech_5m, order_book, 0.015)
    
    assert "mln_p_up" in res
    assert "mln_p_down" in res
    assert "mln_p_neutral" in res
    assert "mln_confidence" in res
    assert "mln_signal" in res
    assert res["mln_p_up"] >= 0.0 and res["mln_p_up"] <= 1.0
    assert res["mln_confidence"] > 0
    assert res["mln_signal"] in ["STRONG_BUY", "LEAN_BUY", "NEUTRAL", "AVOID_BEARISH"]

def test_mln_neutral_fallback():
    res = mln_engine.predict_deep_probabilities({}, {})
    assert res["mln_signal"] == "NEUTRAL"
    assert res["mln_confidence"] == 33.0
