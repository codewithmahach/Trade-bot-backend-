import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("no_trade_engine")

class NoTradeEngine:
    """Aggressively filters and rejects subpar trading opportunities."""

    STANDARD_REASONS = {
        "LOW_LIQUIDITY": "Order book depth is too shallow for immediate fill without high slippage.",
        "HIGH_SPREAD": "Bid/Ask spread is too wide; fees and spread consume expected profit.",
        "LOW_VOLUME": "24h or relative volume is insufficient for scalp momentum.",
        "LOW_CONFIDENCE": "Signal score or model confidence is below the high-quality threshold.",
        "HIGH_VOLATILITY": "Market volatility is excessively chaotic; stops are vulnerable.",
        "CONFLICTING_SIGNALS": "Technical indicators and ML probabilities point in opposing directions.",
        "POOR_RISK_REWARD": "Risk-to-reward ratio is unfavorable (< 1.5:1).",
        "INSUFFICIENT_EXPECTED_RETURN": "Estimated net profit after fees and slippage is negative or negligible.",
        "MARKET_UNSTABLE": "Micro-structure regime is choppy, erratic, or non-directional.",
        "DAILY_LIMIT_REACHED": "Daily cumulative loss threshold reached. Capital preservation active.",
        "COOLDOWN": "System is in post-trade cooldown to prevent over-trading.",
        "INSUFFICIENT_BALANCE": "Account balance is lower than the requested trade amount.",
        "AI_DISAGREEMENT": "AI reasoning disagreed with technical and quantitative setup.",
        "ML_CONFIDENCE_LOW": "Machine learning directional probability P(up) is below required safety line.",
        "TARGET_UNREALISTIC": "Target profit requires an unrealistic short-term price move."
    }

    @classmethod
    def log_rejection(cls, symbol: str, reason_code: str, details: Optional[str] = None):
        description = cls.STANDARD_REASONS.get(reason_code, "Setup failed safety filter.")
        full_msg = f"[NO TRADE] {symbol} rejected -> {reason_code}: {details or description}"
        logger.info(full_msg)
        return {
            "status": "NO_TRADE",
            "symbol": symbol,
            "code": reason_code,
            "description": description,
            "details": details
        }

no_trade_engine = NoTradeEngine()
