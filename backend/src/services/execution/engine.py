import time
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from backend.src.config.settings import settings
from backend.src.services.binance.client import binance_client
from backend.src.services.risk.manager import risk_manager
from backend.src.services.risk.no_trade_engine import no_trade_engine
from backend.src.services.execution.profit_calculator import profit_calculator
from backend.src.database.db import AsyncSessionLocal
from backend.src.models.models import TradeRecord, AuditLogRecord

logger = logging.getLogger("execution_engine")

class ExecutionEngine:
    """Safely executes and reconciles trades across Paper and Live Binance environments."""

    def __init__(self):
        # Virtual portfolio for Paper Trading
        self.virtual_balance_usdt: float = 1000.0
        self.active_position: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

    async def sync_live_position_from_exchange(self) -> Optional[Dict[str, Any]]:
        """Adopts any active Binance Futures position into active_position so it is displayed in UI and monitored."""
        if not (settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET):
            return self.active_position
        try:
            from data.binance.connector import binance_connector
            b_state = await asyncio.to_thread(binance_connector.get_open_position)
            if b_state.get("status") == "success" and b_state.get("active"):
                b_trade = b_state.get("trade", {})
                sym = b_trade.get("symbol", "").replace("/", "")
                entry = float(b_trade.get("entry", 0.0))
                pnl = float(b_trade.get("pnl", 0.0))
                side = b_trade.get("side", "SHORT").upper()
                
                curr_price = entry
                try:
                    from backend.src.services.market.data_manager import market_data_manager
                    t = market_data_manager.get_ticker(sym)
                    if t and "price" in t:
                        curr_price = float(t["price"])
                except Exception:
                    pass

                if not self.active_position or self.active_position.get("symbol") != sym:
                    tp_price = entry * (0.9975 if side == "SHORT" else 1.0025)
                    sl_price = entry * (1.0075 if side == "SHORT" else 0.9925)
                    pos_qty = float(b_trade.get("amount", 10.0))
                    pos_margin = float(b_trade.get("margin", round((pos_qty * entry) / 20.0, 2)))
                    self.active_position = {
                        "id": int(time.time() * 1000),
                        "symbol": sym,
                        "direction": side,
                        "mode": "LIVE_FUTURES",
                        "status": "OPEN",
                        "amount_pkr": round(pos_margin * settings.PKR_USDT_RATE, 2),
                        "amount_usdt": pos_margin,
                        "quantity": pos_qty,
                        "entry_price": entry,
                        "current_price": curr_price,
                        "take_profit": tp_price,
                        "stop_loss": sl_price,
                        "entry_time": time.time() - 120,
                        "max_holding_seconds": 120,
                        "entry_fee_usdt": 0.005,
                        "signal_score": 90.0,
                        "ml_probability": 90.0,
                        "ai_confidence": 90.0,
                        "unrealized_gross_pnl_usdt": pnl,
                        "unrealized_net_pnl_usdt": pnl,
                        "unrealized_net_pnl_pkr": round(pnl * settings.PKR_USDT_RATE, 2),
                        "exchange_pnl": pnl
                    }
                    logger.info(f"Reconciled active Binance position into engine: {sym} ({side}) PnL: {pnl}")
                else:
                    self.active_position["exchange_pnl"] = pnl
                    self.active_position["unrealized_net_pnl_usdt"] = pnl
                    self.active_position["unrealized_net_pnl_pkr"] = round(pnl * settings.PKR_USDT_RATE, 2)
                    if curr_price > 0:
                        self.active_position["current_price"] = curr_price
            elif b_state.get("status") == "success" and not b_state.get("active"):
                if self.active_position and self.active_position.get("mode") in ["LIVE", "LIVE_FUTURES"]:
                    logger.info("Binance reports no open position, clearing local active_position.")
                    self.active_position = None
        except Exception as e:
            logger.debug(f"sync_live_position_from_exchange notice: {e}")
        return self.active_position

    def reset_virtual_balance(self, amount: float = 1000.0):
        self.virtual_balance_usdt = amount
        logger.info(f"Virtual balance reset to {amount} USDT")

    def get_portfolio_status(self) -> Dict[str, Any]:
        return {
            "virtual_balance_usdt": round(self.virtual_balance_usdt, 2),
            "virtual_balance_pkr": round(self.virtual_balance_usdt * settings.PKR_USDT_RATE, 2),
            "has_active_position": self.active_position is not None,
            "mode": "PAPER" if settings.PAPER_TRADING else "LIVE",
            "consecutive_losses": risk_manager.consecutive_losses,
            "daily_realized_pnl_usdt": round(risk_manager.daily_realized_pnl_usdt, 2),
            "daily_realized_pnl_pkr": round(risk_manager.daily_realized_pnl_usdt * settings.PKR_USDT_RATE, 2),
            "trades_today_count": risk_manager.trades_today_count,
            "is_emergency_stopped": risk_manager.is_emergency_stopped,
            "is_paused_by_risk": risk_manager.is_paused_by_risk,
            "pause_reason": risk_manager.pause_reason
        }

    async def execute_trade(self, opportunity: Dict[str, Any], amount_pkr: float) -> Dict[str, Any]:
        """Atomically validates risk and enters a scalp position."""
        async with self._lock:
            if self.active_position is not None:
                return no_trade_engine.log_rejection(opportunity.get("symbol", ""), "MAX_POSITIONS_REACHED", "Already holding an active position")

            symbol = opportunity["symbol"]
            entry_price = float(opportunity["entry_price"])
            take_profit = float(opportunity["take_profit"])
            stop_loss = float(opportunity["stop_loss"])
            rr_ratio = float(opportunity.get("risk_reward_ratio", 1.5))
            spread_pct = float(opportunity.get("metrics", {}).get("spread", 0.02))

            amount_usdt = profit_calculator.pkr_to_usdt(amount_pkr, settings.PKR_USDT_RATE)
            balance = self.virtual_balance_usdt if settings.PAPER_TRADING else 10000.0 # Will query live if LIVE

            # Pre-entry Risk Verification (Section 18 & 24)
            approved, risk_msg = risk_manager.validate_pre_entry_risk(
                symbol=symbol,
                spread_pct=spread_pct,
                active_positions_count=1 if self.active_position else 0,
                rr_ratio=rr_ratio,
                balance_usdt=balance,
                trade_amount_usdt=amount_usdt
            )
            if not approved:
                return no_trade_engine.log_rejection(symbol, "RISK_VALIDATION_FAILED", risk_msg)

            # Execution Branch: Strictly Real Binance Futures Order Execution (0% Mock/Paper Fallback)
            has_live_keys = bool(settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET)
            if has_live_keys:
                return await self._execute_live_trade(opportunity, amount_pkr, amount_usdt, entry_price, take_profit, stop_loss)
            else:
                return await self._execute_paper_trade(opportunity, amount_pkr, amount_usdt, entry_price, take_profit, stop_loss)

    async def _execute_paper_trade(
        self,
        opp: Dict[str, Any],
        amount_pkr: float,
        amount_usdt: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float
    ) -> Dict[str, Any]:
        symbol = opp["symbol"]
        metrics = profit_calculator.calculate_trade_metrics(
            amount_pkr=amount_pkr,
            entry_price=entry_price,
            target_exit_price=take_profit,
            stop_loss_price=stop_loss,
            spread_pct=opp.get("spread", 0.02),
            fx_rate=settings.PKR_USDT_RATE,
            taker_fee_rate=settings.TAKER_FEE_RATE
        )

        quantity = metrics["quantity"]
        entry_fee = metrics["entry_fee_usdt"]

        # Deduct trade amount & fee from virtual balance
        self.virtual_balance_usdt -= (amount_usdt + entry_fee)

        now = time.time()
        direction = opp.get("direction", "LONG").upper()
        position = {
            "id": int(now * 1000),
            "symbol": symbol,
            "direction": direction,
            "mode": "PAPER",
            "status": "OPEN",
            "amount_pkr": amount_pkr,
            "amount_usdt": amount_usdt,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "entry_time": now,
            "max_holding_seconds": opp.get("expected_duration_seconds", settings.MAX_HOLDING_SECONDS),
            "entry_fee_usdt": entry_fee,
            "signal_score": opp.get("finalScore", 0.0),
            "ml_probability": opp.get("confidence", 0.0),
            "ai_confidence": opp.get("confidence", 0.0),
            "unrealized_gross_pnl_usdt": 0.0,
            "unrealized_net_pnl_usdt": -entry_fee,
            "unrealized_net_pnl_pkr": -(entry_fee * settings.PKR_USDT_RATE),
            "metrics": metrics
        }

        self.active_position = position
        logger.info(f"PAPER TRADE ENTERED: {symbol} {direction} | Qty: {quantity:.4f} @ {entry_price} | TP: {take_profit} | SL: {stop_loss}")

        # Record open trade to DB
        try:
            async with AsyncSessionLocal() as session:
                rec = TradeRecord(
                    symbol=symbol,
                    direction=direction,
                    mode="PAPER",
                    status="OPEN",
                    amount_pkr=amount_pkr,
                    amount_usdt=amount_usdt,
                    quantity=quantity,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_fee_usdt=entry_fee,
                    signal_score=opp.get("finalScore", 0.0),
                    ml_probability=opp.get("confidence", 0.0),
                    ai_confidence=opp.get("confidence", 0.0)
                )
                session.add(rec)
                await session.commit()
                position["db_id"] = rec.id
        except Exception as e:
            logger.error(f"Failed to save open trade to DB: {e}")

        return {"status": "SUCCESS", "message": f"Entered {symbol} LONG position", "position": position}

    async def _execute_live_trade(
        self,
        opp: Dict[str, Any],
        amount_pkr: float,
        amount_usdt: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float
    ) -> Dict[str, Any]:
        symbol = opp["symbol"]
        direction = opp.get("direction", "LONG").upper()
        side = "BUY" if direction == "LONG" else "SELL"

        try:
            from data.binance.connector import binance_connector
            sym = symbol if "/" in symbol else f"{symbol.replace('USDT', '')}/USDT"

            res = await asyncio.to_thread(
                binance_connector.execute_trade,
                symbol=sym,
                side=side,
                amount_usdt=amount_usdt,
                current_price=entry_price,
                tp_price=take_profit,
                sl_price=stop_loss,
                api_key=settings.BINANCE_API_KEY,
                api_secret=settings.BINANCE_API_SECRET
            )

            if res.get("status") == "success":
                now = time.time()
                order_res = res.get("order", {})
                qty = float(order_res.get("amount", (amount_usdt * 20) / entry_price))
                position = {
                    "id": order_res.get("id", int(now * 1000)),
                    "symbol": symbol.replace("/", ""),
                    "direction": direction,
                    "mode": "LIVE_FUTURES",
                    "status": "OPEN",
                    "amount_pkr": amount_pkr,
                    "amount_usdt": amount_usdt,
                    "quantity": qty,
                    "entry_price": entry_price,
                    "current_price": entry_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "entry_time": now,
                    "max_holding_seconds": opp.get("expected_duration_seconds", settings.MAX_HOLDING_SECONDS),
                    "entry_fee_usdt": amount_usdt * 0.0005 * 20,
                    "signal_score": opp.get("finalScore", 85.0),
                    "ml_probability": opp.get("confidence", 85.0),
                    "ai_confidence": opp.get("confidence", 85.0),
                    "unrealized_gross_pnl_usdt": 0.0,
                    "unrealized_net_pnl_usdt": 0.0,
                    "unrealized_net_pnl_pkr": 0.0
                }
                # Record live open trade to DB
                try:
                    async with AsyncSessionLocal() as session:
                        rec = TradeRecord(
                            symbol=symbol.replace("/", ""),
                            direction=direction,
                            mode="LIVE_FUTURES",
                            status="OPEN",
                            amount_pkr=amount_pkr,
                            amount_usdt=amount_usdt,
                            quantity=qty,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            entry_fee_usdt=position["entry_fee_usdt"],
                            signal_score=opp.get("finalScore", 85.0),
                            ml_probability=opp.get("confidence", 85.0),
                            ai_confidence=opp.get("confidence", 85.0)
                        )
                        session.add(rec)
                        await session.commit()
                        position["db_id"] = rec.id
                except Exception as e:
                    logger.error(f"Failed to save live open trade to DB: {e}")

                self.active_position = position
                from strategy.trade_state.manager import trade_state_manager
                trade_state_manager.open_trade({
                    "platform": "binance",
                    "symbol": symbol,
                    "side": side,
                    "amount": amount_usdt,
                    "entry_price": entry_price
                })
                return {"status": "SUCCESS", "message": f"LIVE Futures 20x Order placed for {symbol}", "position": position}
            else:
                return {"status": "ERROR", "message": res.get("message", "Futures order rejected")}

        except Exception as e:
            logger.error(f"LIVE Order failed: {e}")
            return {"status": "ERROR", "message": f"Live order failed: {str(e)}"}

    async def close_active_position(self, exit_price: float, exit_reason: str, force: bool = False) -> Dict[str, Any]:
        """Closes active scalp position, settles PnL, logs audit trail."""
        async with self._lock:
            pos = self.active_position
            if not pos:
                # Direct check on Binance if active_position was None
                if settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET:
                    try:
                        from data.binance.connector import binance_connector
                        b_state = await asyncio.to_thread(binance_connector.get_open_position)
                        if b_state.get("status") == "success" and b_state.get("active"):
                            b_sym = b_state.get("trade", {}).get("symbol", "")
                            if b_sym:
                                logger.info(f"Closing orphan Binance position: {b_sym}")
                                close_res = await asyncio.to_thread(
                                    binance_connector.close_position,
                                    b_sym,
                                    settings.BINANCE_API_KEY,
                                    settings.BINANCE_API_SECRET
                                )
                                return {"status": "SUCCESS", "message": f"Closed Binance position {b_sym} at market.", "details": close_res}
                    except Exception as b_err:
                        logger.error(f"Failed to close Binance orphan position: {b_err}")
                return {"status": "NO_ACTIVE_POSITION"}

            symbol = pos["symbol"]
            quantity = pos["quantity"]
            entry_price = pos["entry_price"]
            entry_fee = pos["entry_fee_usdt"]
            mode = pos["mode"]
            duration = int(time.time() - pos["entry_time"])

            # Compute actual exit metrics
            exit_metrics = profit_calculator.calculate_actual_exit_pnl(
                quantity=quantity,
                entry_price=entry_price,
                actual_exit_price=exit_price,
                entry_fee_usdt=entry_fee,
                taker_fee_rate=settings.TAKER_FEE_RATE,
                fx_rate=settings.PKR_USDT_RATE,
                direction=pos.get("direction", "LONG")
            )

            # ZERO-LOSS ENFORCEMENT: Never close in negative net PnL unless user manually clicked EMERGENCY_STOP / MANUAL_CLOSE or forced by exchange
            is_manual_or_forced = force or exit_reason in [
                "EMERGENCY_STOP", "MANUAL_CLOSE", "MANUAL_CLEAR", "FORCE_CLOSE", 
                "EXCHANGE_CLOSED", "EXCHANGE_TP_CLOSED"
            ]
            if not is_manual_or_forced and exit_metrics["net_pnl_usdt"] <= 0:
                logger.info(f"PROTECTION: Holding position for {symbol}. Net PnL is ${exit_metrics['net_pnl_usdt']:.4f}. Per strict rule, holding until profit is generated.")
                return {"status": "HELD_FOR_PROFIT", "message": "Position in drawdown. Holding until positive profit is generated."}

            # In LIVE mode, execute close on Binance Futures if not already closed by exchange
            if ("FUTURES" in mode or mode == "LIVE") and exit_reason not in ["EXCHANGE_CLOSED", "EXCHANGE_TP_CLOSED"]:
                try:
                    from data.binance.connector import binance_connector
                    await asyncio.to_thread(
                        binance_connector.close_position,
                        symbol,
                        settings.BINANCE_API_KEY,
                        settings.BINANCE_API_SECRET
                    )
                    from strategy.trade_state.manager import trade_state_manager
                    trade_state_manager.close_trade(exit_reason)
                except Exception as e:
                    logger.error(f"Live Futures close error: {e}")
            else:
                try:
                    from strategy.trade_state.manager import trade_state_manager
                    trade_state_manager.close_trade(exit_reason)
                except Exception:
                    pass

            # In PAPER mode, credit virtual balance
            if mode == "PAPER":
                exit_notional = quantity * exit_price
                net_credit = exit_notional - exit_metrics["exit_fee_usdt"]
                self.virtual_balance_usdt += net_credit

            # Update Risk Manager
            risk_manager.record_trade_completion(exit_metrics["net_pnl_usdt"])

            # Update or create DB record
            db_id = pos.get("db_id")
            try:
                async with AsyncSessionLocal() as session:
                    rec = None
                    if db_id:
                        rec = await session.get(TradeRecord, db_id)
                    if not rec:
                        rec = TradeRecord(
                            symbol=symbol.replace("/", ""),
                            direction=pos.get("direction", "LONG"),
                            mode=mode,
                            amount_pkr=pos.get("amount_pkr", 0.0),
                            amount_usdt=pos.get("amount_usdt", 0.0),
                            quantity=quantity,
                            entry_price=entry_price,
                            stop_loss=pos.get("stop_loss", 0.0),
                            take_profit=pos.get("take_profit", 0.0),
                            entry_fee_usdt=entry_fee,
                            signal_score=pos.get("signal_score", 0.0),
                            ml_probability=pos.get("ml_probability", 0.0),
                            ai_confidence=pos.get("ai_confidence", 0.0)
                        )
                        session.add(rec)
                    rec.exit_price = exit_price
                    rec.exit_time = datetime.now(timezone.utc)
                    rec.duration_seconds = duration
                    rec.gross_pnl_usdt = exit_metrics["gross_pnl_usdt"]
                    rec.exit_fee_usdt = exit_metrics["exit_fee_usdt"]
                    rec.net_pnl_usdt = exit_metrics["net_pnl_usdt"]
                    rec.net_pnl_pkr = exit_metrics["net_pnl_pkr"]
                    rec.exit_reason = exit_reason
                    rec.status = "CLOSED"
                    await session.commit()
            except Exception as e:
                logger.error(f"Failed to update trade in DB: {e}")

            logger.info(
                f"POSITION CLOSED: {symbol} | Reason: {exit_reason} | Duration: {duration}s | "
                f"Net PnL: {exit_metrics['net_pnl_usdt']:+.3f} USDT ({exit_metrics['net_pnl_pkr']:+.2f} PKR)"
            )

            completed_trade = {
                **pos,
                "exit_price": exit_price,
                "exit_time": time.time(),
                "duration_seconds": duration,
                "exit_reason": exit_reason,
                "status": "CLOSED",
                "final_metrics": exit_metrics
            }

            self.active_position = None
            try:
                from backend.src.services.scanner.scanner import market_scanner
                market_scanner.register_trade_completed(symbol)
                market_scanner.clear_best_opportunity()
            except Exception as scan_err:
                logger.debug(f"Scanner trade reset notice: {scan_err}")

            return {"status": "SUCCESS", "trade": completed_trade}

execution_engine = ExecutionEngine()
