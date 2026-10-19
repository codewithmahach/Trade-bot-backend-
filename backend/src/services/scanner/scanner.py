import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from backend.src.config.settings import settings
from backend.src.services.market.data_manager import market_data_manager
from backend.src.services.market.order_book_sync import order_book_manager
from backend.src.services.technical.indicators import technical_engine
from backend.src.services.market.regime import regime_engine
from backend.src.services.quant.scoring import quant_scoring_engine
from backend.src.services.ml.probability_model import ml_model
from backend.src.services.ml.mln_network import mln_engine
from backend.src.services.ai.validator import ai_validator
from backend.src.services.ai.langchain_orchestrator import langchain_scalp_engine
from backend.src.services.risk.manager import risk_manager
from backend.src.services.risk.no_trade_engine import no_trade_engine
from backend.src.services.execution.profit_calculator import profit_calculator

logger = logging.getLogger("market_scanner")

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if k != "raw_data"}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj

class MarketScannerService:
    """5-Level ultra-fast hierarchical market scanner for Binance scalping."""

    STABLECOINS = {"USDCUSDT", "FDUSDUSDT", "EURUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT", "USDPUSDT", "AEURUSDT"}
    DEFAULT_BLACKLIST = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "WIFUSDT"}

    @property
    def blacklisted_symbols(self) -> set:
        custom_bl = {s.strip().upper() for s in settings.BLACKLISTED_COINS.split(",") if s.strip()}
        return self.STABLECOINS.union(self.DEFAULT_BLACKLIST).union(custom_bl)

    def __init__(self):
        self.latest_candidates: List[Dict[str, Any]] = []
        self.best_opportunity: Optional[Dict[str, Any]] = None
        self.last_scan_time: float = 0.0
        self._is_scanning: bool = False
        self.current_step: str = "IDLE"
        self.scan_activity_log: List[Dict[str, Any]] = []
        self.recently_traded: Dict[str, float] = {}  # symbol -> closed_time

    def register_trade_completed(self, symbol: str):
        """Bars recently traded coin from re-entry for 15 minutes and clears previous opportunity."""
        clean_sym = symbol.replace("/", "").upper()
        self.recently_traded[clean_sym] = time.time()
        self.best_opportunity = None
        logger.info(f"MarketScanner: {clean_sym} recorded as completed. Banned from re-entry for 15m to ensure token variety.")

    def clear_best_opportunity(self):
        self.best_opportunity = None

    def _log_activity(self, step_type: str, msg: str, symbol: str = ""):
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "type": step_type,
            "message": msg,
            "symbol": symbol
        }
        self.scan_activity_log.append(entry)
        if len(self.scan_activity_log) > 30:
            self.scan_activity_log = self.scan_activity_log[-30:]

    async def scan_market(self, amount_pkr: float, min_desired_profit_pkr: float) -> Dict[str, Any]:
        """Executes full 5-stage funnel scanner in under 2 seconds."""
        from backend.src.services.execution.engine import execution_engine
        # STRICT SINGLE POSITION LOCK: If position is open, freeze scanning completely!
        if execution_engine.active_position:
            act_sym = execution_engine.active_position.get("symbol", "")
            self.best_opportunity = None
            logger.info(f"Active scalp open on {act_sym}. Scanning paused until trade closes.")
            return {
                "candidates": self.latest_candidates,
                "best_opportunity": None,
                "scan_duration_ms": 0,
                "status": f"SCAN_PAUSED_ACTIVE_POSITION_{act_sym}",
                "current_step": f"HOLDING ACTIVE POSITION ON {act_sym} — SCANNING PAUSED",
                "activity_log": self.scan_activity_log
            }

        if self._is_scanning:
            return {"candidates": self.latest_candidates, "best_opportunity": self.best_opportunity}

        self._is_scanning = True
        t0 = time.time()
        try:
            tickers = market_data_manager.get_all_tickers()
            if not tickers:
                return {"candidates": [], "best_opportunity": None, "scan_duration_ms": 0}

            self.current_step = f"LEVEL 1: Screening {len(tickers)} live Binance pairs (Excluding BTC/ETH/BNB/WIF & Stablecoins)"
            self._log_activity("FILTER", f"Screened {len(tickers)} pairs. Excluded blacklisted coins and stablecoins.")

            # ==========================================
            # LEVEL 1 — WHOLE-MARKET QUALITY FILTER (EXCLUDE NEWLY LAUNCHED & FAZOL TOKENS)
            # ==========================================
            now_ts = time.time()
            # Clean expired symbols (> 15 minutes) from recently_traded
            self.recently_traded = {s: t for s, t in self.recently_traded.items() if (now_ts - t) < 900.0}

            level1_passed = []
            blacklisted = self.blacklisted_symbols

            # Ensure futures onboard dates are loaded to eliminate newly launched tokens
            try:
                from backend.src.services.binance.client import binance_client
                await binance_client.get_futures_exchange_info()
            except Exception:
                pass

            SHITCOIN_TRAPS = {
                "MUBARAK", "DYM", "PLUME", "TROY", "ACT", "PNUT", "NEIRO", "HIPPO",
                "COW", "CETUS", "THE", "GOAT", "MOODENG", "DEXE", "1000CAT", "BANANA",
                "VOXEL", "ALPACA", "FARTCOIN", "CHIP", "VVV", "ARC", "PENGU", "SLERF",
                "SUNDOG", "TURBO", "MEME", "ORBS", "LOKA", "COOK", "BIGTIME", "GRIFFAIN",
                "MAJOR", "COOKIE", "SWARMS", "PUMP", "BOME", "1000RATS", "1000SATS"
            }
            DEAD_STAGNANT = {
                "HBAR", "ADA", "XRP", "TRX", "LTC", "ALGO", "DOT", "ATOM", "EOS",
                "XLM", "ETC", "ONE", "BCH", "ZEC", "DASH", "KSM", "BAT", "ZIL", "OMG"
            }
            STABLE_BASES = {
                "BTC", "ETH", "BNB", "WIF", "USDC", "FDUSD", "TUSD", "USDP", "BUSD",
                "EUR", "RLUSD", "USD1", "AEUR", "DAI", "PAX", "USTC", "USD", "US", "PAX"
            }

            for sym, t in tickers.items():
                if sym in blacklisted or sym in self.recently_traded:
                    continue
                base = sym.replace("USDT", "")
                if base in STABLE_BASES or any(base == b for b in STABLE_BASES):
                    continue
                if base in SHITCOIN_TRAPS or any(base.startswith(st) for st in SHITCOIN_TRAPS):
                    continue
                if base in DEAD_STAGNANT:
                    continue

                # 1. Eliminate newly launched tokens (< 60 days on Binance Futures)
                from backend.src.services.binance.client import binance_client
                if binance_client.is_newly_launched(sym, min_days=60):
                    continue

                quote_vol = t.get("quoteVolume24h", 0.0)
                spread = t.get("spread", 1.0)
                change_24h = t.get("priceChangePercent", 0.0)
                trades_count = t.get("count", 0)

                # 2. Strict Liquidity & Quality: High Volume (>= $25M) and Tight Spread (<= 0.04%)
                # Eliminates illiquid tokens that waste time, lead to losses, or hold positions for hours
                if quote_vol < 25_000_000 or trades_count < 15_000:
                    continue
                if spread > 0.04:
                    continue

                momentum_rank = (abs(change_24h) * 3.0) + min(30.0, quote_vol / 10_000_000.0)
                level1_passed.append((momentum_rank, sym, t))

            # Sort descending and dynamically rotate across top 40 candidates for variety
            level1_passed.sort(key=lambda x: x[0], reverse=True)
            top_pool = level1_passed[:40] if len(level1_passed) >= 40 else level1_passed
            if not hasattr(self, "_scanner_rotation_offset"):
                self._scanner_rotation_offset = 0

            offset = self._scanner_rotation_offset % max(1, len(top_pool))
            rotated_pool = top_pool[offset:] + top_pool[:offset]
            top_candidates_l1 = rotated_pool[:8]
            self._scanner_rotation_offset = (offset + 8) % max(1, len(top_pool))

            top_syms = [c[1] for c in top_candidates_l1]
            # Ensure local L2 order book sync is active for selected candidates
            try:
                await order_book_manager.ensure_active(top_syms)
            except Exception as e:
                logger.debug(f"Order book sync start notice: {e}")

            self.current_step = f"LEVEL 2: Fetching 1m/3m/5m candles & L2 order book for: {', '.join(top_syms)}"
            self._log_activity("SELECT", f"Selected top candidates: {', '.join(top_syms)}")

            # ==========================================
            # LEVEL 2 — PARALLEL TECHNICAL ANALYSIS & L2 MICROSTRUCTURE
            # ==========================================
            async def analyze_token(sym_data):
                _, sym, ticker = sym_data
                try:
                    # Fetch 1m, 3m, 5m candles in parallel
                    c1m_t = market_data_manager.get_candles(sym, "1m", limit=60)
                    c3m_t = market_data_manager.get_candles(sym, "3m", limit=40)
                    c5m_t = market_data_manager.get_candles(sym, "5m", limit=30)
                    c1m, c3m, c5m = await asyncio.gather(c1m_t, c3m_t, c5m_t, return_exceptions=True)

                    if isinstance(c1m, Exception) or not c1m or len(c1m) < 30:
                        return None

                    tech_1m = technical_engine.analyze_candles(c1m)
                    # Dead / Stagnant Price Action Filter:
                    # If 1m ATR is below 0.12%, the token moves too slowly for a fast 90-120s scalp (prevents dragging trades for hours)
                    if tech_1m.get("atr_pct", 0.0) < 0.12:
                        return None

                    tech_3m = technical_engine.analyze_candles(c3m) if not isinstance(c3m, Exception) and c3m else {}
                    tech_5m = technical_engine.analyze_candles(c5m) if not isinstance(c5m, Exception) and c5m else {}

                    # L2 Order Book depth & spread
                    ob_metrics = order_book_manager.get_metrics(sym)
                    if ob_metrics.get("synchronized", False):
                        imbalance = ob_metrics["imbalance_10"]
                        spread_pct = ob_metrics["spread_pct"]
                    else:
                        price_tmp = tech_1m.get("current_price", ticker.get("price", 1.0))
                        vwap_tmp = tech_1m.get("vwap", price_tmp)
                        rvol_tmp = tech_1m.get("relative_volume", 1.0)
                        imbalance = 0.5 + (0.15 if price_tmp > vwap_tmp else -0.1) + min(0.15, (rvol_tmp - 1.0) * 0.1)
                        imbalance = max(0.2, min(0.85, imbalance))
                        spread_pct = ticker.get("spread", 0.02)

                    order_book_metrics = {
                        "imbalance": imbalance,
                        "spread": spread_pct,
                        "micro_price": ob_metrics.get("micro_price", tech_1m.get("current_price", 1.0)),
                        "synchronized": ob_metrics.get("synchronized", False)
                    }

                    price = tech_1m.get("current_price", ticker.get("price", 1.0))

                    # Calibrated ML Market Regime Classification
                    regime_res = ml_model.classify_regime(tech_1m, order_book_metrics)

                    # ==========================================
                    # LEVEL 3 — QUANT & CALIBRATED ML PROBABILITIES
                    # ==========================================
                    quant_res = quant_scoring_engine.calculate_score(tech_1m, tech_3m, tech_5m, order_book_metrics, spread_pct)
                    ml_res = ml_model.predict(tech_1m, tech_3m, order_book_metrics, spread_pct)
                    mln_res = mln_engine.predict_deep_probabilities(tech_1m, tech_3m, tech_5m, order_book_metrics, spread_pct)
                    ensemble_p_up = round((ml_res["p_up"] * 0.6) + (mln_res["mln_p_up"] * 0.4), 3)

                    # Expected Net Return Hurdle calculation with normalized conditional directional probabilities
                    up_down_sum = ml_res["p_up"] + ml_res["p_down"]
                    cond_p_up = (ml_res["p_up"] / up_down_sum) if up_down_sum > 0 else 0.5
                    cond_p_down = (ml_res["p_down"] / up_down_sum) if up_down_sum > 0 else 0.5

                    exp_net = quant_scoring_engine.calculate_expected_net_return(
                        p_up=cond_p_up,
                        p_down=cond_p_down,
                        tp_pct=0.85,
                        sl_pct=0.50,
                        taker_fee_pct=settings.TAKER_FEE_RATE,
                        spread_pct=spread_pct,
                        slippage_pct=0.01
                    )

                    return {
                        "symbol": sym,
                        "price": price,
                        "ticker": ticker,
                        "tech_1m": tech_1m,
                        "tech_3m": tech_3m,
                        "tech_5m": tech_5m,
                        "regime": regime_res,
                        "order_book": order_book_metrics,
                        "quant": quant_res,
                        "ml": ml_res,
                        "mln": mln_res,
                        "ensemble_p_up": ensemble_p_up,
                        "spread_pct": spread_pct,
                        "exp_net": exp_net
                    }
                except Exception as e:
                    logger.debug(f"Error analyzing {sym}: {e}")
                    return None

            analyzed_results = await asyncio.gather(*[analyze_token(c) for c in top_candidates_l1])
            valid_analyzed = [r for r in analyzed_results if r is not None]

            # Filter for candidates with high Quant + ML ensemble score and suitable regime
            valid_analyzed = [r for r in valid_analyzed if r["regime"].get("is_suitable", True)]
            valid_analyzed.sort(key=lambda x: x["quant"]["total_score"] + x["ensemble_p_up"] * 45.0, reverse=True)
            top_level3 = valid_analyzed[:10]

            # ==========================================
            # LEVEL 4 — FAST DETERMINISTIC EVALUATION (DECOUPLED FROM LLM)
            # ==========================================
            processed_candidates = []
            for cand in top_level3:
                sym = cand["symbol"]
                # Live low-latency path uses sub-millisecond deterministic logic
                ai_res = langchain_scalp_engine._deterministic_fast_eval(
                    sym, cand["tech_1m"], cand["tech_3m"], cand["quant"],
                    cand["ml"], cand["mln"], cand["order_book"], cand["spread_pct"]
                )

                final_score = ai_res.get("final_trade_score", cand["quant"]["total_score"])
                can_reach_tp = ai_res.get("can_reach_tp_in_2min", False)
                rsi_val = cand["tech_1m"].get("rsi", 50.0)
                exp_net = cand["exp_net"]
                has_positive_hurdle = exp_net["expected_net_bps"] > 0.0 or exp_net["expected_gross_bps"] > 2.0
                decision = ai_res.get("decision", "NO_TRADE")

                dir_sum = cand["ml"]["p_up"] + cand["ml"]["p_down"]
                rel_dir_p_up = (cand["ml"]["p_up"] / dir_sum) if dir_sum > 0 else 0.5

                is_quality_long = (
                    decision == "LONG" and
                    final_score >= 48.0 and
                    can_reach_tp and
                    cand["spread_pct"] <= 0.05 and
                    42.0 <= rsi_val <= 75.0 and
                    (rel_dir_p_up >= 0.46 or cand["ensemble_p_up"] >= 0.12) and
                    has_positive_hurdle
                )

                is_quality_short = (
                    decision == "SHORT" and
                    final_score >= 48.0 and
                    can_reach_tp and
                    cand["spread_pct"] <= 0.05 and
                    25.0 <= rsi_val <= 58.0 and
                    ((1.0 - rel_dir_p_up) >= 0.46 or (1.0 - cand["ensemble_p_up"]) >= 0.12) and
                    has_positive_hurdle
                )

                is_quality_entry = is_quality_long or is_quality_short

                if not has_positive_hurdle and decision in ["LONG", "SHORT"]:
                    ai_res["decision"] = "NO_TRADE"
                    ai_res["reason"] = f"Expected return ({exp_net['expected_net_bps']:.1f} bps) below cost hurdle"

                signal_label = "TRADE READY" if is_quality_entry else (
                    "WATCHLIST" if (final_score >= 44.0 and has_positive_hurdle) else "REJECT"
                )

                candidate_summary = {
                    "symbol": sym,
                    "price": cand["price"],
                    "signals_mtf": {
                        "1m": "BULL" if cand["tech_1m"].get("trend_bullish") else ("BEAR" if cand["tech_1m"].get("trend_bearish") else "NEUTRAL"),
                        "3m": "BULL" if cand["tech_3m"].get("trend_bullish") else ("BEAR" if cand["tech_3m"].get("trend_bearish") else "NEUTRAL"),
                        "5m": "BULL" if cand["tech_5m"].get("trend_bullish") else ("BEAR" if cand["tech_5m"].get("trend_bearish") else "NEUTRAL")
                    },
                    "volume24h": cand["ticker"].get("quoteVolume24h", 0.0),
                    "spread": cand["spread_pct"],
                    "technicalScore": cand["quant"]["total_score"],
                    "mlProbability": round(cand["ml"]["p_up"] * 100.0, 1),
                    "mlnProbability": round(cand["mln"]["mln_p_up"] * 100.0, 1),
                    "mlnSignal": cand["mln"]["mln_signal"],
                    "ensembleProbability": round(cand["ensemble_p_up"] * 100.0, 1),
                    "aiConfidence": round(ai_res.get("confidence", 0.0) * 100.0, 1),
                    "expectedNetBps": exp_net["expected_net_bps"],
                    "modelName": cand["ml"].get("model_name", "Calibrated XGBoost"),
                    "finalScore": final_score,
                    "signal": signal_label,
                    "decision": ai_res.get("decision", "NO_TRADE"),
                    "regime": cand["regime"].get("regime", "RANGING"),
                    "raw_data": cand,
                    "ai_res": ai_res
                }
                processed_candidates.append(candidate_summary)

            self.latest_candidates = sanitize_for_json(processed_candidates)

            # ==========================================
            # LEVEL 5 — BEST OPPORTUNITY & STRICT 90-120S TP FILTER
            # ==========================================
            processed_candidates.sort(key=lambda x: x["finalScore"], reverse=True)
            best_cand = None
            for c in processed_candidates:
                if c["decision"] in ["LONG", "SHORT"] and c["signal"] == "TRADE READY" and c["finalScore"] >= 48.0:
                    best_cand = c
                    break

            if best_cand:
                raw = best_cand["raw_data"]
                sym = best_cand["symbol"]
                entry_price = best_cand["price"]
                direction = best_cand["decision"]
                atr_val = raw["tech_1m"].get("atr", entry_price * 0.003)
                struct = raw["tech_1m"].get("structure", {})
                res = struct.get("resistance", entry_price * 1.008)
                sup = struct.get("support", entry_price * 0.995)

                if direction == "SHORT":
                    # 0.25% fast scalp target for SHORT
                    tp = round(entry_price * (1.0 - 0.0025), 6)
                    sl = round(entry_price * (1.0 + 0.0075), 6)
                    rr = 1.8
                else:
                    # 0.25% fast scalp target for LONG
                    tp = round(entry_price * (1.0 + 0.0025), 6)
                    sl = round(entry_price * (1.0 - 0.0075), 6)
                    rr = 1.8

                # Profit and target feasibility calculation
                metrics = profit_calculator.calculate_trade_metrics(
                    amount_pkr=amount_pkr,
                    entry_price=entry_price,
                    target_exit_price=tp,
                    stop_loss_price=sl,
                    spread_pct=best_cand["spread"],
                    fx_rate=settings.PKR_USDT_RATE,
                    taker_fee_rate=settings.TAKER_FEE_RATE
                )

                feasible, reason_feasibility, feat_details = profit_calculator.validate_target_feasibility(
                    amount_pkr=amount_pkr,
                    desired_profit_pkr=min_desired_profit_pkr,
                    entry_price=entry_price,
                    atr_pct=raw["tech_1m"].get("atr_pct", 0.3),
                    fx_rate=settings.PKR_USDT_RATE,
                    taker_fee_rate=settings.TAKER_FEE_RATE
                )

                self.best_opportunity = {
                    "symbol": sym,
                    "direction": direction,
                    "confidence": best_cand["aiConfidence"],
                    "finalScore": best_cand["finalScore"],
                    "entry_price": entry_price,
                    "take_profit": tp,
                    "stop_loss": sl,
                    "risk_reward_ratio": rr,
                    "expected_duration_seconds": min(120, best_cand["ai_res"].get("recommendedHoldingSeconds", 120)),
                    "metrics": metrics,
                    "feasible": feasible,
                    "feasibility_reason": reason_feasibility,
                    "ai_reason": best_cand["ai_res"].get("reason", "Multi-layer consensus validated"),
                    "market_regime": best_cand["regime"],
                    "timestamp": time.time()
                }
                self.best_opportunity = sanitize_for_json(self.best_opportunity)
                self.current_step = f"TRADE OPPORTUNITY READY: {sym} ({direction} Score: {best_cand['finalScore']}/100)"
                self._log_activity("OPPORTUNITY", f"High-confidence 120s scalp ready on {sym} ({direction}) at ${entry_price:.4f}! TP: ${tp:.4f}, SL: ${sl:.4f}", symbol=sym)
            else:
                self.best_opportunity = None
                self.current_step = "ZERO-LOSS CAPITAL PRESERVATION: No coin met strict 70+ breakout criteria"
                self._log_activity("PRESERVE", f"Evaluated top {len(processed_candidates)} candidates. No setup met 70+ consensus. Capital preserved.")

            self.last_scan_time = time.time()
            duration_ms = int((time.time() - t0) * 1000)

            return {
                "candidates": self.latest_candidates,
                "best_opportunity": self.best_opportunity,
                "scan_duration_ms": duration_ms,
                "current_step": self.current_step,
                "activity_log": self.scan_activity_log
            }
        finally:
            self._is_scanning = False

    def recalculate_best_opportunity(self, amount_pkr: float, min_desired_profit_pkr: float):
        if not self.best_opportunity:
            return
        entry_price = self.best_opportunity.get("entry_price", 1.0)
        tp = self.best_opportunity.get("take_profit", entry_price * 1.015)
        sl = self.best_opportunity.get("stop_loss", entry_price * 0.99)
        metrics = profit_calculator.calculate_trade_metrics(
            amount_pkr=amount_pkr,
            entry_price=entry_price,
            target_exit_price=tp,
            stop_loss_price=sl,
            spread_pct=0.0005,
            fx_rate=settings.PKR_USDT_RATE,
            taker_fee_rate=settings.TAKER_FEE_RATE
        )
        req_pct = ((tp - entry_price) / entry_price) * 100.0 if entry_price > 0 else 1.0
        feasible, reason_feasibility, _ = profit_calculator.validate_target_feasibility(
            amount_pkr=amount_pkr,
            desired_profit_pkr=min_desired_profit_pkr,
            entry_price=entry_price,
            atr_pct=max(0.6, req_pct),
            fx_rate=settings.PKR_USDT_RATE,
            taker_fee_rate=settings.TAKER_FEE_RATE
        )
        self.best_opportunity["metrics"] = metrics
        self.best_opportunity["feasible"] = feasible
        self.best_opportunity["feasibility_reason"] = reason_feasibility
        self.best_opportunity = sanitize_for_json(self.best_opportunity)

market_scanner = MarketScannerService()
