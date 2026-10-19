from typing import Dict, Any

class MarketRegimeEngine:
    """Classifies real-time micro-structure market regime for scalping."""

    @staticmethod
    def detect_regime(tech: Dict[str, Any], spread_pct: float) -> Dict[str, Any]:
        if not tech:
            return {"regime": "UNSTABLE", "tradeable": False, "reason": "Insufficient technical data"}

        atr_pct = tech.get("atr_pct", 0.0)
        bb_width = tech.get("bb_bandwidth", 0.0)
        rsi = tech.get("rsi", 50.0)
        trend_bull = tech.get("trend_bullish", False)
        trend_bear = tech.get("trend_bearish", False)
        structure = tech.get("structure", {})
        breakout = structure.get("breakout", False)
        breakdown = structure.get("breakdown", False)
        rvol = tech.get("relative_volume", 1.0)

        # 1. Check for extreme instability or wide spread
        if spread_pct > 0.12:
            return {
                "regime": "UNSTABLE",
                "tradeable": False,
                "reason": f"Spread too wide ({spread_pct:.3f}% > 0.12%)"
            }

        # 2. Check for low volatility (dead market)
        if atr_pct < 0.05:
            return {
                "regime": "LOW_VOLATILITY",
                "tradeable": False,
                "reason": "Volatility too low for profitable scalping after fees"
            }

        # 3. Check for extreme volatile chaos
        if atr_pct > 6.0:
            return {
                "regime": "HIGH_VOLATILITY",
                "tradeable": False,
                "reason": "Extreme volatility detected; excessive risk of slippage"
            }

        # 4. Check for clean Breakout with volume
        if breakout and rvol > 1.2:
            return {
                "regime": "BREAKOUT",
                "tradeable": True,
                "preferred_side": "LONG",
                "reason": "Bullish breakout above resistance confirmed by volume"
            }

        if breakdown and rvol > 1.3:
            return {
                "regime": "BREAKDOWN",
                "tradeable": False,
                "preferred_side": "SHORT",
                "reason": "Bearish breakdown below key support"
            }

        # 5. Trending bullish regimes
        price = tech.get("current_price", 1.0)
        ema9 = tech.get("ema9", price)
        ema21 = tech.get("ema21", price)
        if (trend_bull or (price >= ema21 and ema9 >= ema21)) and rsi >= 46 and rsi <= 82:
            return {
                "regime": "TRENDING_BULLISH",
                "tradeable": True,
                "preferred_side": "LONG",
                "reason": "Aligned bullish momentum with upward price action"
            }

        if trend_bear and rsi <= 50 and tech.get("macd_hist", 0) < 0:
            return {
                "regime": "TRENDING_BEARISH",
                "tradeable": False,
                "preferred_side": "SHORT",
                "reason": "Consistent downward trend; unfavorable for long scalps"
            }

        # 6. Range bound
        if 42 <= rsi <= 58 and not trend_bull and not trend_bear:
            return {
                "regime": "RANGE",
                "tradeable": False,
                "reason": "Market in tight chop/range consolidation"
            }

        return {
            "regime": "UNSTABLE",
            "tradeable": False,
            "reason": "Conflicting technical indicators across moving averages and oscillators"
        }

regime_engine = MarketRegimeEngine()
