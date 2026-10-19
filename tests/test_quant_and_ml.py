import pytest
from backend.src.services.quant.scoring import quant_scoring_engine
from backend.src.services.ml.probability_model import ml_model

def test_quant_scoring_weights():
    # Mock aligned bullish technical snapshot
    tech_1m = {
        "current_price": 50000.0,
        "ema9": 50100.0,
        "ema21": 49900.0,
        "ema50": 49500.0,
        "vwap": 49800.0,
        "rsi": 58.0,
        "macd_hist": 12.0,
        "stoch_k": 65.0,
        "stoch_d": 55.0,
        "relative_volume": 1.8,
        "atr_pct": 0.35,
        "bb_bandwidth": 1.2,
        "bb_pct_b": 0.7,
        "structure": {"breakout": True, "dist_to_resistance_pct": 0.8, "dist_to_support_pct": 0.4}
    }
    tech_3m = {"trend_bullish": True}
    tech_5m = {"trend_bullish": True}
    order_book = {"imbalance": 0.65}
    spread = 0.015

    score_res = quant_scoring_engine.calculate_score(tech_1m, tech_3m, tech_5m, order_book, spread)
    assert score_res["total_score"] >= 80.0
    assert score_res["category"] in ("HIGH_QUALITY", "POTENTIAL")
    assert score_res["is_tradeable_candidate"] is True

def test_ml_probability_prediction():
    tech_1m = {
        "current_price": 50000.0,
        "ema9": 50100.0,
        "ema21": 49900.0,
        "vwap": 49800.0,
        "rsi": 58.0,
        "macd_hist": 12.0,
        "atr_pct": 0.35,
        "relative_volume": 1.8,
        "bb_pct_b": 0.7
    }
    order_book = {"imbalance": 0.7}
    probs = ml_model.predict(tech_1m, {"trend_bullish": True}, order_book, 0.015)

    assert "p_up" in probs
    assert "p_down" in probs
    assert "p_no_move" in probs
    # Probabilities should sum to approximately 1.0
    total = probs["p_up"] + probs["p_down"] + probs["p_no_move"]
    assert 0.99 <= total <= 1.01
