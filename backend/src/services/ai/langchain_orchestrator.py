import json
import logging
import time
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from backend.src.config.settings import settings

logger = logging.getLogger("langchain_orchestrator")

class LangChainScalpOrchestrator:
    """Advanced Level LLM Scalping Orchestrator using LangChain.
    
    Orchestrates high-speed 2-minute breakout decision making,
    evaluating rapid order flow momentum, volume acceleration,
    and Take-Profit feasibility within 120 seconds.
    """

    def __init__(self):
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Elite Crypto Scalping Quantitative AI Engine.
Your SOLE MISSION is to evaluate if a cryptocurrency token can reach its Take-Profit target within 90 to 120 SECONDS.

STRICT SCALPING RULES:
1. If the micro-market is BULLISH (green candle, EMA9 > EMA21, RSI 48-68, buyers active), approve ("LONG").
2. If the micro-market is BEARISH (red candle, EMA9 < EMA21, RSI 32-52, sellers active), approve ("SHORT").
3. NEVER take counter-trend trades. If trend is choppy, ranging, or dormant, REJECT ("NO_TRADE").
4. Minimum profit target must be >= $0.25 USDT net. Capital preservation is paramount: Zero low-quality trades.
5. Output MUST be strictly valid JSON matching this schema:
{{
  "decision": "LONG" | "SHORT" | "NO_TRADE",
  "confidence": <float 0.50 to 0.99>,
  "velocity_score": <int 50 to 100>,
  "can_reach_tp_in_2min": <true | false>,
  "reason": "<clear concise 1-sentence technical reason>",
  "recommendedHoldingSeconds": <int between 90 and 120>,
  "final_trade_score": <float 50.0 to 100.0>
}}"""),
            ("human", """Analyze scalp setup for {symbol}:
- Current Price: ${price}
- 1m Momentum: {signal_1m} | 3m: {signal_3m} | 5m: {signal_5m}
- 1m RSI: {rsi:.1f} | MACD Hist: {macd_hist:.4f}
- Relative Volume (RVOL): {rvol:.2f}x average
- Orderbook Buy Imbalance: {imbalance:.2f} (0.5=neutral, >0.6=buy pressure)
- 1m Volatility (ATR%): {atr_pct:.2f}%
- 100-Point Quant Score: {quant_score:.1f}/100
- ML Directional Probability: {ml_prob:.1f}%
- Deep MLN Neural Probability: {mln_prob:.1f}%
- Bid-Ask Spread: {spread_pct:.3f}%

Can this token hit Take-Profit (+{target_move_pct:.2f}%) within 120 seconds?""")
        ])
        self._llm_cooldown_until: float = 0.0

    def _get_llm(self) -> Optional[ChatOpenAI]:
        if not settings.OPENAI_API_KEY:
            return None
        return ChatOpenAI(
            model=settings.OPENAI_MODEL_FAST,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
            max_tokens=150,
            timeout=2.0,
            max_retries=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    async def evaluate_2min_scalp(
        self,
        symbol: str,
        tech_1m: Dict[str, Any],
        tech_3m: Dict[str, Any],
        tech_5m: Dict[str, Any],
        quant_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        mln_result: Dict[str, Any],
        order_book_metrics: Dict[str, Any],
        spread_pct: float,
        target_move_pct: float = 0.8
    ) -> Dict[str, Any]:
        """Runs the LangChain pipeline to evaluate 2-minute Take-Profit velocity."""
        # Circuit breaker: if LLM previously throttled or in cooldown, run instant C-speed quant/ML engine
        if time.time() < self._llm_cooldown_until:
            return self._deterministic_fast_eval(
                symbol, tech_1m, tech_3m, quant_result, ml_result, mln_result, order_book_metrics, spread_pct
            )

        llm = self._get_llm()
        if not llm:
            return self._deterministic_fast_eval(
                symbol, tech_1m, tech_3m, quant_result, ml_result, mln_result, order_book_metrics, spread_pct
            )

        try:
            chain = self.prompt | llm | self.parser
            inputs = {
                "symbol": symbol,
                "price": tech_1m.get("current_price", 1.0),
                "signal_1m": "BULLISH" if tech_1m.get("trend_bullish") else "NEUTRAL",
                "signal_3m": "BULLISH" if tech_3m.get("trend_bullish") else "NEUTRAL",
                "signal_5m": "BULLISH" if tech_5m.get("trend_bullish") else "NEUTRAL",
                "rsi": tech_1m.get("rsi", 50.0),
                "macd_hist": tech_1m.get("macd_hist", 0.0),
                "rvol": tech_1m.get("relative_volume", 1.0),
                "imbalance": order_book_metrics.get("imbalance", 0.5),
                "atr_pct": tech_1m.get("atr_pct", 0.4),
                "quant_score": quant_result.get("total_score", 60.0),
                "ml_prob": ml_result.get("p_up", 0.5) * 100.0,
                "mln_prob": mln_result.get("mln_p_up", 0.5) * 100.0,
                "spread_pct": spread_pct,
                "target_move_pct": target_move_pct
            }

            res = await chain.ainvoke(inputs)
            res["recommendedHoldingSeconds"] = min(120, max(45, int(res.get("recommendedHoldingSeconds", 120))))
            logger.info(f"LangChain Decision for {symbol}: {res.get('decision')} | Conf: {res.get('confidence')} | Velocity: {res.get('velocity_score')}")
            return res

        except Exception as e:
            # Set 60-second cooldown on rate limits (429) or network errors to keep scanning instant
            self._llm_cooldown_until = time.time() + 60.0
            logger.info(f"LangChain LLM on 60s cooldown ({e}). Using ultra-fast sub-millisecond local ML/Quant engine.")
            return self._deterministic_fast_eval(
                symbol, tech_1m, tech_3m, quant_result, ml_result, mln_result, order_book_metrics, spread_pct
            )

    def _deterministic_fast_eval(
        self,
        symbol: str,
        tech_1m: Dict[str, Any],
        tech_3m: Dict[str, Any],
        quant_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        mln_result: Dict[str, Any],
        order_book_metrics: Dict[str, Any],
        spread_pct: float
    ) -> Dict[str, Any]:
        """Ultra-fast sub-millisecond quantitative reasoning engine with strict bidirectional zero-loss filtering."""
        quant_score = quant_result.get("total_score", 50.0)
        quant_dir = quant_result.get("direction", "NEUTRAL")
        ml_p_up = ml_result.get("p_up", 0.5)
        ml_p_down = ml_result.get("p_down", 0.5)
        mln_p_up = mln_result.get("mln_p_up", 0.5)
        rvol = tech_1m.get("relative_volume", 1.0)
        imbalance = order_book_metrics.get("imbalance", 0.5)
        rsi = tech_1m.get("rsi", 50.0)
        bb_pct_b = tech_1m.get("bb_pct_b", 0.5)
        atr_pct = tech_1m.get("atr_pct", 0.0)

        # Scalp liquidity: 1m ATR >= 0.10%, tight spread <= 0.05%, active volume >= 0.85x
        has_scalp_liquidity = atr_pct >= 0.10 and spread_pct <= 0.05 and rvol >= 0.85

        # Conditional directional probabilities (accounts for 3-class distribution where NO_MOVE is ~65%)
        up_down_sum = ml_p_up + ml_p_down
        rel_p_up = (ml_p_up / up_down_sum) if up_down_sum > 0 else 0.5
        rel_p_down = (ml_p_down / up_down_sum) if up_down_sum > 0 else 0.5

        # Bullish LONG Confluence
        is_bullish_scalp = (
            has_scalp_liquidity and
            quant_dir == "LONG" and
            quant_score >= 48.0 and
            (rel_p_up >= 0.46 or ml_p_up >= ml_p_down or mln_p_up >= 0.15) and
            42.0 <= rsi <= 74.0 and
            bb_pct_b >= 0.38 and
            imbalance >= 0.44
        )

        # Bearish SHORT Confluence
        is_bearish_scalp = (
            has_scalp_liquidity and
            quant_dir == "SHORT" and
            quant_score >= 48.0 and
            (rel_p_down >= 0.46 or ml_p_down >= ml_p_up or (1.0 - mln_p_up) >= 0.15) and
            26.0 <= rsi <= 58.0 and
            bb_pct_b <= 0.62 and
            imbalance <= 0.56
        )

        if is_bullish_scalp:
            ensemble_conf = round(((ml_p_up * 0.5) + (mln_p_up * 0.3) + (quant_score / 100.0 * 0.2)), 2)
            velocity_score = int(min(100, max(50, (rvol * 20.0) + (imbalance * 40.0) + (ensemble_conf * 30.0))))
            return {
                "decision": "LONG",
                "confidence": ensemble_conf,
                "velocity_score": velocity_score,
                "can_reach_tp_in_2min": True,
                "reason": f"BULLISH Breakout: RSI {rsi:.1f}, BB %B {bb_pct_b:.2f}, RVOL {rvol:.1f}x, ATR {atr_pct:.2f}%",
                "recommendedHoldingSeconds": 120,
                "final_trade_score": round(max(quant_score, (quant_score * 0.4) + (ensemble_conf * 60.0)), 1)
            }
        elif is_bearish_scalp:
            ensemble_conf = round(((ml_p_down * 0.5) + ((1.0 - mln_p_up) * 0.3) + (quant_score / 100.0 * 0.2)), 2)
            velocity_score = int(min(100, max(50, (rvol * 20.0) + ((1.0 - imbalance) * 40.0) + (ensemble_conf * 30.0))))
            return {
                "decision": "SHORT",
                "confidence": ensemble_conf,
                "velocity_score": velocity_score,
                "can_reach_tp_in_2min": True,
                "reason": f"BEARISH Breakdown: RSI {rsi:.1f}, BB %B {bb_pct_b:.2f}, RVOL {rvol:.1f}x, ATR {atr_pct:.2f}%",
                "recommendedHoldingSeconds": 120,
                "final_trade_score": round(max(quant_score, (quant_score * 0.4) + (ensemble_conf * 60.0)), 1)
            }
        else:
            reason = "Dormant ATR or insufficient momentum. Preserving capital." if not has_scalp_liquidity else "Confluence below 90-120s scalp velocity standard."
            return {
                "decision": "NO_TRADE",
                "confidence": 0.50,
                "velocity_score": 50,
                "can_reach_tp_in_2min": False,
                "reason": reason,
                "recommendedHoldingSeconds": 120,
                "final_trade_score": round(quant_score, 1)
            }

langchain_scalp_engine = LangChainScalpOrchestrator()
