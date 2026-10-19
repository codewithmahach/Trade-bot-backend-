import json
import logging
import httpx
from typing import Dict, Any, Optional
from backend.src.config.settings import settings

logger = logging.getLogger("ai_validator")

class AIModelRouter:
    """Intelligently routes analysis to the fastest OpenAI model (gpt-4o-mini) with sub-second latency."""

    def __init__(self):
        self.openai_key = settings.OPENAI_API_KEY
        self.fast_model = settings.OPENAI_MODEL_FAST  # "gpt-4o-mini"

    def select_model(self, quant_score: float, is_conflicting: bool) -> str:
        # User constraint: ALWAYS use OpenAI's fastest mind for millisecond execution
        return self.fast_model

class AIValidationEngine:
    def __init__(self):
        self.router = AIModelRouter()

    async def validate_candidate(
        self,
        symbol: str,
        tech_1m: Dict[str, Any],
        tech_3m: Dict[str, Any],
        tech_5m: Dict[str, Any],
        quant_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        regime_result: Dict[str, Any],
        order_book_metrics: Dict[str, Any],
        spread_pct: float
    ) -> Dict[str, Any]:
        """Validates scalping setup using AI Layer with fallback to deterministic quantitative rules."""
        quant_score = quant_result.get("total_score", 0.0)
        p_up = ml_result.get("p_up", 0.0)
        p_down = ml_result.get("p_down", 0.0)
        regime = regime_result.get("regime", "UNSTABLE")
        imbalance = order_book_metrics.get("imbalance", 0.5)
        rvol = tech_1m.get("relative_volume", 1.0)

        # Check for immediate structural conflicts
        is_conflicting = (quant_score >= 75.0 and p_down > 0.40) or (regime == "UNSTABLE")

        # Compact structured payload for LLM (Section 40)
        payload = {
            "symbol": symbol,
            "timeframes": {
                "1m": {
                    "rsi": round(tech_1m.get("rsi", 50.0), 1),
                    "macd_hist": round(tech_1m.get("macd_hist", 0.0), 4),
                    "price": tech_1m.get("current_price", 0.0),
                    "atr_pct": round(tech_1m.get("atr_pct", 0.0), 3)
                },
                "3m": {
                    "rsi": round(tech_3m.get("rsi", 50.0), 1) if tech_3m else 50.0,
                    "trend_bullish": bool(tech_3m.get("trend_bullish", False)) if tech_3m else False
                },
                "5m": {
                    "trend_bullish": bool(tech_5m.get("trend_bullish", False)) if tech_5m else False
                }
            },
            "technicalScore": quant_score,
            "mlProbabilityUp": p_up,
            "mlProbabilityDown": p_down,
            "marketRegime": regime,
            "orderBookImbalance": round(imbalance, 2),
            "spread": round(spread_pct, 4),
            "relativeVolume": round(rvol, 2),
            "expectedHoldSeconds": settings.MAX_HOLDING_SECONDS
        }

        # If OpenAI API Key is provided, perform live LLM validation
        api_key = settings.OPENAI_API_KEY
        if api_key:
            model_to_use = self.router.select_model(quant_score, is_conflicting)
            try:
                ai_res = await self._call_openai(api_key, model_to_use, payload)
                if ai_res:
                    return self._apply_ensemble(quant_result, ml_result, regime_result, ai_res)
            except Exception as e:
                logger.warning(f"OpenAI call failed ({e}), falling back to deterministic validation")

        # Deterministic Institutional Validator (Offline or No Key)
        return self._deterministic_validation(quant_result, ml_result, regime_result, spread_pct, rvol, imbalance)

    async def _call_openai(self, api_key: str, model: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        system_prompt = (
            "You are an elite quantitative crypto scalping validator for 2-3 minute Binance trades. "
            "Evaluate the candidate setup strictly. Never guarantee profit. Reject low-confidence or conflicting setups. "
            "Output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "decision": "LONG" | "SHORT" | "NO_TRADE",\n'
            '  "confidence": float (0.00 to 1.00),\n'
            '  "marketQuality": "HIGH" | "MEDIUM" | "LOW",\n'
            '  "risk": "LOW" | "MEDIUM" | "HIGH",\n'
            '  "reason": string,\n'
            '  "recommendedHoldingSeconds": integer (between 60 and 300)\n'
            "}"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 120
        }

        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            if resp.status_code == 200:
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"]
                return json.loads(raw_content)
            else:
                logger.error(f"OpenAI API error {resp.status_code}: {resp.text}")
                return None

    def _deterministic_validation(
        self,
        quant_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        regime_result: Dict[str, Any],
        spread_pct: float,
        rvol: float,
        imbalance: float
    ) -> Dict[str, Any]:
        """Mathematical rule engine matching exact scalping quality constraints."""
        quant_score = quant_result.get("total_score", 0.0)
        p_up = ml_result.get("p_up", 0.0)
        p_down = ml_result.get("p_down", 0.0)
        regime = regime_result.get("regime", "UNSTABLE")
        tradeable = regime_result.get("tradeable", False)

        # Rule 1: Market regime check
        if not tradeable or regime in ("UNSTABLE", "LOW_VOLATILITY"):
            return {
                "decision": "NO_TRADE",
                "confidence": 0.0,
                "marketQuality": "LOW",
                "risk": "HIGH",
                "reason": f"Market regime is {regime} ({regime_result.get('reason')})",
                "recommendedHoldingSeconds": 0,
                "final_trade_score": quant_score
            }

        # Rule 2: Quantitative score threshold
        if quant_score < 55.0:
            return {
                "decision": "NO_TRADE",
                "confidence": round(quant_score / 100.0, 2),
                "marketQuality": "LOW",
                "risk": "HIGH",
                "reason": f"Quantitative score insufficient ({quant_score}/100 < 55)",
                "recommendedHoldingSeconds": 0,
                "final_trade_score": quant_score
            }

        # Rule 3: ML Probability agreement (Section 41)
        if p_down > 0.40:
            return {
                "decision": "NO_TRADE",
                "confidence": round(p_up, 2),
                "marketQuality": "MEDIUM",
                "risk": "MEDIUM",
                "reason": f"ML Model probability downside risk too high (P(down)={p_down*100:.1f}%)",
                "recommendedHoldingSeconds": 0,
                "final_trade_score": quant_score
            }

        # Rule 4: Micro-structure liquidity & spread
        if spread_pct > 0.10:
            return {
                "decision": "NO_TRADE",
                "confidence": 0.5,
                "marketQuality": "MEDIUM",
                "risk": "HIGH",
                "reason": f"Order book spread ({spread_pct:.3f}%) exceeds scalping threshold",
                "recommendedHoldingSeconds": 0,
                "final_trade_score": quant_score
            }

        # If all pass:
        confidence = round((quant_score * 0.5 + p_up * 100 * 0.3 + (1.0 - spread_pct * 5) * 20) / 100.0, 2)
        confidence = min(0.96, max(0.70, confidence))
        final_trade_score = round(quant_score * 0.6 + p_up * 100 * 0.4, 1)

        return {
            "decision": "LONG",
            "confidence": confidence,
            "marketQuality": "HIGH" if final_trade_score >= 75 else "MEDIUM",
            "risk": "LOW" if final_trade_score >= 75 else "MEDIUM",
            "reason": f"Quant score {quant_score}/100 aligned with {regime} momentum",
            "recommendedHoldingSeconds": settings.MAX_HOLDING_SECONDS,
            "final_trade_score": final_trade_score
        }

    def _apply_ensemble(
        self,
        quant_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        regime_result: Dict[str, Any],
        ai_res: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combines Technical Score, ML Probability, Market Regime and AI Validation into Trade Quality Score (Section 13)."""
        tech_score = quant_result.get("total_score", 0.0)
        p_up = ml_result.get("p_up", 0.0)
        ai_decision = ai_res.get("decision", "NO_TRADE")
        ai_conf = float(ai_res.get("confidence", 0.0))

        # Disagreement check (Section 41)
        if ai_decision != "LONG" or ai_conf < 0.70:
            return {
                "decision": "NO_TRADE",
                "confidence": ai_conf,
                "marketQuality": ai_res.get("marketQuality", "LOW"),
                "risk": ai_res.get("risk", "HIGH"),
                "reason": f"AI Validation declined: {ai_res.get('reason', 'Confidence below 70%')}",
                "recommendedHoldingSeconds": 0,
                "final_trade_score": round(tech_score * 0.5, 1)
            }

        # Final Trade Quality Score (0-100)
        trade_quality_score = round(
            (tech_score * 0.40) +
            (p_up * 100.0 * 0.35) +
            (ai_conf * 100.0 * 0.25),
            1
        )

        return {
            "decision": "LONG",
            "confidence": ai_conf,
            "marketQuality": ai_res.get("marketQuality", "HIGH"),
            "risk": ai_res.get("risk", "LOW"),
            "reason": ai_res.get("reason", "Multi-model ensemble consensus approved"),
            "recommendedHoldingSeconds": ai_res.get("recommendedHoldingSeconds", settings.MAX_HOLDING_SECONDS),
            "final_trade_score": trade_quality_score
        }

ai_validator = AIValidationEngine()
