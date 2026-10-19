from typing import Dict, Any

class QuantitativeSignalEngine:
    """Multi-factor quantitative scoring engine (0 - 100 pts)."""

    @staticmethod
    def calculate_score(
        tech_1m: Dict[str, Any],
        tech_3m: Dict[str, Any],
        tech_5m: Dict[str, Any],
        order_book_metrics: Dict[str, Any],
        spread_pct: float
    ) -> Dict[str, Any]:
        if not tech_1m:
            return {"total_score": 0.0, "category": "REJECT", "direction": "NEUTRAL", "breakdown": {}}

        price = tech_1m.get("current_price", 0.0)
        ema9 = tech_1m.get("ema9", 0.0)
        ema21 = tech_1m.get("ema21", 0.0)
        ema50 = tech_1m.get("ema50", 0.0)
        rsi = tech_1m.get("rsi", 50.0)
        macd_hist = tech_1m.get("macd_hist", 0.0)
        stoch_k = tech_1m.get("stoch_k", 50.0)
        stoch_d = tech_1m.get("stoch_d", 50.0)
        rvol = tech_1m.get("relative_volume", 1.0)
        atr_pct = tech_1m.get("atr_pct", 0.0)
        bb_width = tech_1m.get("bb_bandwidth", 0.0)
        vwap = tech_1m.get("vwap", price)
        imbalance = order_book_metrics.get("imbalance", 0.5)
        struct = tech_1m.get("structure", {})

        # Shared Quality Factors: Volume, Volatility, Spread
        common_pts = 0.0
        # Volume (Max 15)
        if rvol >= 2.0: common_pts += 15.0
        elif rvol >= 1.5: common_pts += 12.0
        elif rvol >= 1.2: common_pts += 8.0
        elif rvol >= 1.0: common_pts += 4.0

        # Volatility & Spread (Max 15)
        if 0.18 <= atr_pct <= 1.2: common_pts += 6.0
        elif 0.12 <= atr_pct <= 1.8: common_pts += 3.0
        if 0.30 <= bb_width <= 3.5: common_pts += 4.0
        if spread_pct <= 0.03: common_pts += 5.0
        elif spread_pct <= 0.06: common_pts += 3.0

        # ---------------- BULLISH / LONG EVALUATION ----------------
        bull_pts = common_pts
        # Trend (Max 15)
        if ema9 > ema21 > ema50:
            bull_pts += 12.0
            if price > ema9: bull_pts += 3.0
        elif ema9 > ema21:
            bull_pts += 8.0

        # Momentum (Max 15)
        if 48.0 <= rsi <= 68.0: bull_pts += 6.0
        elif 44.0 <= rsi < 48.0: bull_pts += 3.0
        if macd_hist > 0: bull_pts += 5.0
        if stoch_k > stoch_d and stoch_k < 82.0: bull_pts += 4.0

        # VWAP (Max 10)
        if price > vwap:
            vwap_dist = ((price - vwap) / vwap) * 100.0
            bull_pts += 10.0 if (0.04 <= vwap_dist <= 0.8) else 6.0

        # Order Book Buyer Imbalance (Max 10)
        if imbalance >= 0.65: bull_pts += 10.0
        elif imbalance >= 0.55: bull_pts += 7.0
        elif imbalance >= 0.50: bull_pts += 4.0

        # Structure & MTF (Max 20)
        if struct.get("breakout", False): bull_pts += 10.0
        elif struct.get("dist_to_support_pct", 1.0) < 0.3: bull_pts += 7.0
        if tech_3m and tech_3m.get("trend_bullish", False): bull_pts += 5.0
        if tech_5m and tech_5m.get("trend_bullish", False): bull_pts += 5.0

        # ---------------- BEARISH / SHORT EVALUATION ----------------
        bear_pts = common_pts
        # Trend (Max 15)
        if ema9 < ema21 < ema50:
            bear_pts += 12.0
            if price < ema9: bear_pts += 3.0
        elif ema9 < ema21:
            bear_pts += 8.0

        # Momentum (Max 15)
        if 32.0 <= rsi <= 52.0: bear_pts += 6.0
        elif 52.0 < rsi <= 56.0: bear_pts += 3.0
        if macd_hist < 0: bear_pts += 5.0
        if stoch_k < stoch_d and stoch_k > 18.0: bear_pts += 4.0

        # VWAP (Max 10)
        if price < vwap:
            vwap_dist = ((vwap - price) / vwap) * 100.0
            bear_pts += 10.0 if (0.04 <= vwap_dist <= 0.8) else 6.0

        # Order Book Seller Imbalance (Max 10)
        if imbalance <= 0.35: bear_pts += 10.0
        elif imbalance <= 0.45: bear_pts += 7.0
        elif imbalance <= 0.50: bear_pts += 4.0

        # Structure & MTF (Max 20)
        if struct.get("breakdown", False): bear_pts += 10.0
        elif struct.get("dist_to_resistance_pct", 1.0) < 0.3: bear_pts += 7.0
        if tech_3m and tech_3m.get("trend_bearish", False): bear_pts += 5.0
        if tech_5m and tech_5m.get("trend_bearish", False): bear_pts += 5.0

        bull_total = round(min(100.0, max(0.0, bull_pts)), 1)
        bear_total = round(min(100.0, max(0.0, bear_pts)), 1)

        if bull_total >= bear_total and bull_total >= 50.0:
            best_dir = "LONG"
            total = bull_total
        elif bear_total > bull_total and bear_total >= 50.0:
            best_dir = "SHORT"
            total = bear_total
        else:
            best_dir = "NEUTRAL"
            total = max(bull_total, bear_total)

        category = "HIGH_QUALITY" if total >= 75.0 else ("POTENTIAL" if total >= 60.0 else "REJECT")

        return {
            "total_score": total,
            "direction": best_dir,
            "bullish_score": bull_total,
            "bearish_score": bear_total,
            "category": category,
            "is_tradeable_candidate": total >= 65.0 and best_dir in ["LONG", "SHORT"],
            "breakdown": {"bullish": bull_total, "bearish": bear_total, "volume": common_pts}
        }

    @staticmethod
    def calculate_expected_net_return(
        p_up: float,
        p_down: float,
        tp_pct: float,
        sl_pct: float,
        taker_fee_pct: float = 0.05,
        spread_pct: float = 0.02,
        slippage_pct: float = 0.02
    ) -> Dict[str, Any]:
        """
        Calculates mathematical Expected Net Return (in basis points):
        E[R_net] = P(up) * TP_bps - P(down) * SL_bps - Friction_bps
        Where Friction = round-trip fees (0.10%) + half-spread + estimated execution slippage.
        """
        tp_bps = tp_pct * 100.0
        sl_bps = sl_pct * 100.0
        roundtrip_fee_bps = (2.0 * taker_fee_pct) * 100.0
        spread_bps = spread_pct * 100.0
        slippage_bps = slippage_pct * 100.0
        total_friction_bps = roundtrip_fee_bps + (spread_bps * 0.5) + slippage_bps

        expected_gross_bps = (p_up * tp_bps) - (p_down * sl_bps)
        expected_net_bps = expected_gross_bps - total_friction_bps

        return {
            "expected_net_bps": round(expected_net_bps, 2),
            "expected_gross_bps": round(expected_gross_bps, 2),
            "total_friction_bps": round(total_friction_bps, 2),
            "is_positive_expectancy": expected_net_bps > 0.0
        }

quant_scoring_engine = QuantitativeSignalEngine()
