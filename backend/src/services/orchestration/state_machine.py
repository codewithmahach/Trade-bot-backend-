import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from backend.src.config.settings import settings
from backend.src.services.scanner.scanner import market_scanner
from backend.src.services.execution.engine import execution_engine
from backend.src.services.risk.manager import risk_manager

logger = logging.getLogger("state_machine")

class OrchestrationEngine:
    """State machine orchestrating the complete end-to-end crypto scalping lifecycle."""

    STATES = [
        "IDLE",
        "SCANNING",
        "CANDIDATE_DETECTED",
        "FAST_ANALYSIS",
        "TECHNICAL_ANALYSIS",
        "ML_ANALYSIS",
        "ORDER_FLOW_ANALYSIS",
        "MARKET_REGIME",
        "AI_VALIDATION",
        "RISK_VALIDATION",
        "TRADE_APPROVED",
        "NO_TRADE",
        "EXECUTION",
        "POSITION_OPEN",
        "POSITION_MONITORING",
        "EXIT_DECISION",
        "AUTO_CLOSE",
        "PNL_CALCULATION",
        "TRADE_COMPLETE"
    ]

    def __init__(self):
        self.current_state: str = "IDLE"
        self.state_history: list = []
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self.last_pipeline_run: float = 0.0
        self.latest_pipeline_result: Dict[str, Any] = {}

    def set_state(self, new_state: str, details: Optional[str] = None):
        self.current_state = new_state
        entry = {
            "timestamp": time.time(),
            "state": new_state,
            "details": details or ""
        }
        self.state_history.append(entry)
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        logger.info(f"STATE: [{new_state}] {details or ''}")

    async def start(self):
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._auto_trading_loop())
        logger.info("Orchestration State Machine started.")

    async def stop(self):
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()

    async def run_pipeline_once(self, amount_pkr: float, min_profit_pkr: float, auto_enter: bool = False) -> Dict[str, Any]:
        """Runs one full execution of the 15-state decision machine."""
        self.set_state("SCANNING", "Initiating multi-layer market scanner")
        t0 = time.time()

        scan_res = await market_scanner.scan_market(amount_pkr=amount_pkr, min_desired_profit_pkr=min_profit_pkr)
        candidates = scan_res.get("candidates", [])
        best_opp = scan_res.get("best_opportunity")

        if not best_opp:
            self.set_state("NO_TRADE", "No candidate met minimum scalping quality criteria (score >= 58)")
            self.latest_pipeline_result = {
                "decision": "NO_TRADE",
                "reason": "Market conditions do not meet strict scalping threshold. Capital preserved.",
                "best_opportunity": None,
                "candidates_scanned": len(candidates)
            }
            return self.latest_pipeline_result

        sym = best_opp["symbol"]
        self.set_state("CANDIDATE_DETECTED", f"Identified top candidate: {sym}")
        self.set_state("TECHNICAL_ANALYSIS", f"MTF momentum confirmed for {sym}")
        self.set_state("ML_ANALYSIS", f"Probabilities: {best_opp.get('confidence')}% directional confidence")
        self.set_state("MARKET_REGIME", f"Regime classified as {best_opp.get('market_regime')}")
        self.set_state("AI_VALIDATION", f"AI Decision: {best_opp.get('direction')} ({best_opp.get('ai_reason')})")

        # Feasibility check
        if not best_opp.get("feasible", False):
            self.set_state("NO_TRADE", f"Target profit infeasible: {best_opp.get('feasibility_reason')}")
            self.latest_pipeline_result = {
                "decision": "NO_TRADE",
                "reason": best_opp.get("feasibility_reason"),
                "best_opportunity": best_opp,
                "candidates_scanned": len(candidates)
            }
            return self.latest_pipeline_result

        # Risk Engine Validation
        self.set_state("RISK_VALIDATION", f"Checking exposure, cooldown, and limits for {sym}")
        approved, risk_reason = risk_manager.validate_pre_entry_risk(
            symbol=sym,
            spread_pct=best_opp["metrics"].get("spread", 0.02),
            active_positions_count=1 if execution_engine.active_position else 0,
            rr_ratio=best_opp.get("risk_reward_ratio", 1.5),
            balance_usdt=execution_engine.virtual_balance_usdt if settings.PAPER_TRADING else 10000.0,
            trade_amount_usdt=best_opp["metrics"].get("amount_usdt", 10.0)
        )

        if not approved:
            self.set_state("NO_TRADE", f"Risk engine rejected: {risk_reason}")
            self.latest_pipeline_result = {
                "decision": "NO_TRADE",
                "reason": risk_reason,
                "best_opportunity": best_opp,
                "candidates_scanned": len(candidates)
            }
            return self.latest_pipeline_result

        # Trade Approved
        self.set_state("TRADE_APPROVED", f"Approved entry for {sym} | Score: {best_opp.get('finalScore')}/100")
        market_scanner._log_activity("APPROVED", f"AI & Risk approved {sym} ({best_opp.get('direction', 'LONG')}) | Score: {best_opp.get('finalScore')}/100", sym)

        # Auto Enter if enabled
        if auto_enter and not execution_engine.active_position:
            self.set_state("EXECUTION", f"Auto-trading enabled: Executing entry order for {sym}")
            exec_res = await execution_engine.execute_trade(best_opp, amount_pkr)
            if exec_res.get("status") == "SUCCESS":
                pos = exec_res.get("position", {})
                mode = pos.get("mode", "LIVE_FUTURES")
                direction = best_opp.get("direction", "LONG")
                self.set_state("POSITION_OPEN", f"Position active: {sym} {direction}")
                if "LIVE" in mode or "FUTURES" in mode:
                    market_scanner._log_activity("EXECUTION", f"REAL ORDER PLACED ON BINANCE: {sym} {direction} 20x", sym)
                else:
                    market_scanner._log_activity("EXECUTION", f"Paper Order simulated: {sym} {direction}", sym)
            else:
                err_msg = exec_res.get('message', 'Order rejected')
                self.set_state("NO_TRADE", f"Execution error: {err_msg}")
                market_scanner._log_activity("EXEC_ERROR", f"Binance rejected {sym}: {err_msg}", sym)

        self.last_pipeline_run = time.time()
        self.latest_pipeline_result = {
            "decision": "TRADE_APPROVED",
            "best_opportunity": best_opp,
            "candidates_scanned": len(candidates),
            "duration_ms": int((time.time() - t0) * 1000)
        }
        return self.latest_pipeline_result

    async def _auto_trading_loop(self):
        """Background loop continuously scanning when AUTO_TRADING is turned ON."""
        while self._running:
            try:
                if settings.AUTO_TRADING and not risk_manager.is_paused_by_risk and not risk_manager.is_emergency_stopped:
                    if not execution_engine.active_position:
                        await self.run_pipeline_once(
                            amount_pkr=settings.USER_AMOUNT_PKR,
                            min_profit_pkr=settings.MIN_DESIRED_PROFIT_PKR,
                            auto_enter=True
                        )
                elif execution_engine.active_position:
                    self.set_state("POSITION_MONITORING", f"Monitoring active {execution_engine.active_position['symbol']} scalp")

                await asyncio.sleep(3.0) # Scan interval: every 3 seconds for speed
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-trading orchestration loop: {e}")
                await asyncio.sleep(4.0)

orchestrator = OrchestrationEngine()
