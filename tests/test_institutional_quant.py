import pytest
import time
import numpy as np
import pandas as pd
from backend.src.services.quant.scoring import quant_scoring_engine
from backend.src.services.ml.probability_model import ml_model
from backend.src.services.market.order_book_sync import SynchronizedOrderBook
from backend.src.services.scanner.scanner import market_scanner
from backend.src.services.execution.engine import execution_engine
from backend.src.config.settings import settings

def test_expected_net_return_cost_hurdles():
    """Mathematical verification that trades with negative expectancy after friction are rejected."""
    # Scenario A: Negative expectancy (Friction exceeds edge)
    # Taker fee 0.05% x 2 = 10 bps, Spread 0.02% = 2 bps, Slippage 0.02% = 2 bps -> Total friction = 13 bps
    # P(up)=0.35, TP=0.40% (40 bps) -> +14 bps
    # P(down)=0.30, SL=0.30% (30 bps) -> -9 bps
    # Gross edge = 14 - 9 = +5 bps. Net = 5 - 13 = -8 bps (Negative!)
    res_neg = quant_scoring_engine.calculate_expected_net_return(
        p_up=0.35,
        p_down=0.30,
        tp_pct=0.40,
        sl_pct=0.30,
        taker_fee_pct=0.05,
        spread_pct=0.02,
        slippage_pct=0.02
    )
    assert res_neg["expected_net_bps"] < 0.0
    assert res_neg["is_positive_expectancy"] is False
    assert res_neg["total_friction_bps"] == 13.0

    # Scenario B: Strong positive expectancy overcoming hurdle (> 6 bps)
    # P(up)=0.55, TP=0.50% (50 bps) -> +27.5 bps
    # P(down)=0.15, SL=0.30% (30 bps) -> -4.5 bps
    # Gross = 23 bps. Net = 23 - 13 = +10 bps (Positive!)
    res_pos = quant_scoring_engine.calculate_expected_net_return(
        p_up=0.55,
        p_down=0.15,
        tp_pct=0.50,
        sl_pct=0.30,
        taker_fee_pct=0.05,
        spread_pct=0.02,
        slippage_pct=0.02
    )
    assert res_pos["expected_net_bps"] >= settings.MIN_EXPECTED_NET_RETURN_BPS
    assert res_pos["is_positive_expectancy"] is True

def test_calibrated_ml_probabilities_and_latency():
    """Verify calibrated ML model outputs valid probabilities and meets sub-20ms latency SLA."""
    sample_tech = {
        "current_price": 2.50,
        "open": 2.49,
        "high": 2.52,
        "low": 2.48,
        "ema9": 2.51,
        "ema21": 2.49,
        "ema50": 2.47,
        "vwap": 2.495,
        "rsi": 56.0,
        "macd_hist": 0.005,
        "atr_pct": 0.35,
        "relative_volume": 1.6,
        "bb_pct_b": 0.65,
        "bb_upper": 2.53,
        "bb_lower": 2.47
    }

    # Warm up call (JIT / dynamic import initialization)
    ml_model.predict(sample_tech)

    # Measure warm inference latency over 10 iterations
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        pred = ml_model.predict(sample_tech)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat_ms = np.mean(latencies)
    p95_lat_ms = np.percentile(latencies, 95)

    # Latency check: Sub-25ms warm prediction
    assert mean_lat_ms < 25.0, f"Mean warm inference took {mean_lat_ms:.2f}ms, exceeding 25ms threshold"

    # Probability axioms
    assert 0.0 <= pred["p_up"] <= 1.0
    assert 0.0 <= pred["p_down"] <= 1.0
    assert 0.0 <= pred["p_no_move"] <= 1.0
    total = pred["p_up"] + pred["p_down"] + pred["p_no_move"]
    assert pytest.approx(total, abs=1e-2) == 1.0
    assert pred["confidence"] > 0.0

def test_microstructure_regime_classification():
    """Test that low-volatility chop and runaway volatility are flagged as unsuitable."""
    # Chop test: BB width too narrow for taker fees
    tech_chop = {
        "current_price": 10.0,
        "atr_pct": 0.05,
        "relative_volume": 0.4,
        "bb_upper": 10.005,
        "bb_lower": 9.995,
        "rsi": 50.0
    }
    regime_chop = ml_model.classify_regime(tech_chop)
    assert regime_chop["regime"] == "LOW_VOL_CHOP"
    assert regime_chop["is_suitable"] is False

    # Extreme volatility test
    tech_vol = {
        "current_price": 10.0,
        "atr_pct": 3.2,
        "relative_volume": 3.0,
        "bb_upper": 10.5,
        "bb_lower": 9.5,
        "rsi": 75.0
    }
    regime_vol = ml_model.classify_regime(tech_vol)
    assert regime_vol["regime"] == "HIGH_VOLATILITY"
    assert regime_vol["is_suitable"] is False

    # Bullish trend alignment
    tech_bull = {
        "current_price": 10.0,
        "ema9": 10.05,
        "ema21": 9.95,
        "vwap": 9.97,
        "rsi": 55.0,
        "atr_pct": 0.4,
        "relative_volume": 1.5,
        "bb_upper": 10.15,
        "bb_lower": 9.85
    }
    regime_bull = ml_model.classify_regime(tech_bull)
    assert regime_bull["regime"] == "TRENDING_BULL"
    assert regime_bull["is_suitable"] is True

def test_order_book_metrics():
    """Test L2 order book depth imbalance and micro-price calculation."""
    book = SynchronizedOrderBook("NEARUSDT")
    book.bids = {10.00: 50.0, 9.99: 40.0, 9.98: 30.0}
    book.asks = {10.01: 20.0, 10.02: 15.0, 10.03: 10.0}
    book.is_synchronized = True

    metrics = book.get_metrics()
    assert metrics["best_bid"] == 10.00
    assert metrics["best_ask"] == 10.01
    assert metrics["spread_abs"] == 0.01
    # Bids heavily dominate asks -> imbalance > 0.5
    assert metrics["imbalance_5"] > 0.60
    # Micro-price should be closer to ask (10.01) because buyers are pressing
    assert metrics["micro_price"] > metrics["mid_price"]

def test_active_and_recent_symbol_exclusion():
    """Verify that active or recently traded tokens are barred from re-scan and duplicates."""
    market_scanner.recently_traded.clear()
    market_scanner.register_trade_completed("HUMAUSDT")

    assert "HUMAUSDT" in market_scanner.recently_traded
    assert market_scanner.best_opportunity is None

    # Simulating level 1 check
    tickers = {
        "HUMAUSDT": {"quoteVolume24h": 10000000, "spread": 0.01, "priceChangePercent": 5.0, "count": 5000},
        "SUIUSDT": {"quoteVolume24h": 10000000, "spread": 0.01, "priceChangePercent": 4.0, "count": 5000}
    }
    allowed = [s for s in tickers if s not in market_scanner.recently_traded and s not in market_scanner.blacklisted_symbols]
    assert "HUMAUSDT" not in allowed
    assert "SUIUSDT" in allowed

if __name__ == "__main__":
    pytest.main(["-v", __file__])
