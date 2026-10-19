import asyncio
import time
import logging
from typing import Dict, Any, Optional
from backend.src.config.settings import settings
from backend.src.services.market.data_manager import market_data_manager
from backend.src.services.execution.engine import execution_engine
from backend.src.services.risk.manager import risk_manager

logger = logging.getLogger("position_monitor")

class PositionMonitorService:
    """Continuously monitors open positions, tracks live PnL, and enforces strict auto-close policies."""

    def __init__(self):
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Position Monitor Service started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self):
        while self._running:
            try:
                pos = execution_engine.active_position
                if not pos and not settings.PAPER_TRADING and (time.time() - getattr(self, "_last_empty_sync", 0)) > 3.0:
                    self._last_empty_sync = time.time()
                    pos = await execution_engine.sync_live_position_from_exchange()

                if pos:
                    symbol = pos["symbol"]
                    ticker = market_data_manager.get_ticker(symbol)
                    current_price = ticker.get("price", pos["current_price"]) if ticker else pos["current_price"]
                    pos["current_price"] = current_price

                    entry_price = pos["entry_price"]
                    take_profit = pos["take_profit"]
                    stop_loss = pos["stop_loss"]
                    quantity = pos["quantity"]
                    entry_fee = pos["entry_fee_usdt"]
                    now = time.time()

                    # Exchange Sync: If trade is in LIVE mode, verify if it was already filled/closed on Binance
                    if pos.get("mode") in ["LIVE_FUTURES", "LIVE"] and (now - pos.get("_last_binance_sync", 0)) > 2.0:
                        pos["_last_binance_sync"] = now
                        try:
                            from data.binance.connector import binance_connector
                            b_state = binance_connector.get_open_position()
                            if b_state.get("status") == "success" and not b_state.get("active"):
                                logger.info(f"EXCHANGE SYNC: Position {symbol} is NO LONGER active on Binance (Take Profit filled). Finalizing close.")
                                await execution_engine.close_active_position(
                                    current_price, 
                                    "EXCHANGE_TP_CLOSED", 
                                    force=True
                                )
                                continue
                            elif b_state.get("status") == "success" and b_state.get("active"):
                                b_trade = b_state.get("trade", {})
                                if "pnl" in b_trade:
                                    pos["exchange_pnl"] = b_trade["pnl"]
                        except Exception as sync_err:
                            logger.debug(f"Binance sync check notice: {sync_err}")

                    elapsed = int(now - pos["entry_time"])
                    max_allowed = pos.get("max_holding_seconds", settings.MAX_HOLDING_SECONDS)
                    pos["elapsed_seconds"] = elapsed
                    pos["remaining_seconds"] = max(0, max_allowed - elapsed)

                    direction = pos.get("direction", "LONG").upper()
                    is_short = direction == "SHORT"

                    # Update live unrealized PnL (Direction aware)
                    if is_short:
                        gross_pnl_usdt = (entry_price - current_price) * quantity
                        move_pct = ((entry_price - current_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
                    else:
                        gross_pnl_usdt = (current_price - entry_price) * quantity
                        move_pct = ((current_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0

                    est_exit_fee = (quantity * current_price) * settings.TAKER_FEE_RATE
                    net_pnl_usdt = gross_pnl_usdt - entry_fee - est_exit_fee
                    pos["unrealized_gross_pnl_usdt"] = round(gross_pnl_usdt, 4)
                    pos["unrealized_net_pnl_usdt"] = round(net_pnl_usdt, 4)
                    pos["unrealized_net_pnl_pkr"] = round(net_pnl_usdt * settings.PKR_USDT_RATE, 2)

                    # Progress percentage towards TP vs SL
                    if is_short:
                        total_range = abs(stop_loss - take_profit) if abs(stop_loss - take_profit) > 0 else 1.0
                        pos["progress_pct"] = round(min(100.0, max(0.0, ((stop_loss - current_price) / total_range) * 100.0)), 1)
                    else:
                        total_range = abs(take_profit - stop_loss) if abs(take_profit - stop_loss) > 0 else 1.0
                        pos["progress_pct"] = round(min(100.0, max(0.0, ((current_price - stop_loss) / total_range) * 100.0)), 1)

                    # CAPITAL PRESERVATION 1: ULTRA-FAST BREAKEVEN STOP LOCK (at +0.12% move = +2.4% ROE)
                    # Once in profit, move SL to breakeven immediately to guarantee zero-loss!
                    if is_short:
                        breakeven_price = entry_price * (1.0 - (settings.TAKER_FEE_RATE * 2.1))
                        if move_pct >= 0.12 and (stop_loss > breakeven_price or not pos.get("is_breakeven_locked")):
                            logger.info(f"CAPITAL PRESERVATION: Moving Stop Loss to Breakeven ({breakeven_price:.6f}) for {symbol} SHORT")
                            pos["stop_loss"] = breakeven_price
                            stop_loss = breakeven_price
                            pos["is_breakeven_locked"] = True
                    else:
                        breakeven_price = entry_price * (1.0 + (settings.TAKER_FEE_RATE * 2.1))
                        if move_pct >= 0.12 and (stop_loss < breakeven_price or not pos.get("is_breakeven_locked")):
                            logger.info(f"CAPITAL PRESERVATION: Moving Stop Loss to Breakeven ({breakeven_price:.6f}) for {symbol} LONG")
                            pos["stop_loss"] = breakeven_price
                            stop_loss = breakeven_price
                            pos["is_breakeven_locked"] = True

                    # CAPITAL PRESERVATION 2: DYNAMIC HIGH-WATERMARK TRAILING PROFIT RATCHET & USER PROFIT RANGE
                    # User Rule: Min profit $0.15 floor, target $0.25 range, let runners go higher
                    peak_profit = pos.get("peak_profit_usdt", 0.0)
                    if net_pnl_usdt > peak_profit:
                        pos["peak_profit_usdt"] = net_pnl_usdt
                        peak_profit = net_pnl_usdt

                    # Minimum profit floor reached ($0.15)
                    if net_pnl_usdt >= settings.MIN_PROFIT_USDT_FLOOR:
                        pos["min_profit_reached"] = True

                    # Dynamic Trailing Profit Stop: Ratchet stop loss to lock in gains
                    if not is_short:
                        # Once profit hits $0.25+, lock at least $0.20+ so it runs higher safely
                        if net_pnl_usdt >= settings.TARGET_PROFIT_USDT_MAX and stop_loss < (entry_price * 1.008):
                            pos["stop_loss"] = entry_price * 1.008
                            stop_loss = pos["stop_loss"]
                        # Once profit hits $0.15+, lock at least entry + profit floor
                        elif net_pnl_usdt >= settings.MIN_PROFIT_USDT_FLOOR and stop_loss < (entry_price * 1.003):
                            pos["stop_loss"] = entry_price * 1.003
                            stop_loss = pos["stop_loss"]
                        elif move_pct >= 0.20 and stop_loss < (entry_price * 1.001):
                            pos["stop_loss"] = entry_price * 1.001
                            stop_loss = pos["stop_loss"]
                    else:
                        if net_pnl_usdt >= settings.TARGET_PROFIT_USDT_MAX and stop_loss > (entry_price * 0.992):
                            pos["stop_loss"] = entry_price * 0.992
                            stop_loss = pos["stop_loss"]
                        elif net_pnl_usdt >= settings.MIN_PROFIT_USDT_FLOOR and stop_loss > (entry_price * 0.997):
                            pos["stop_loss"] = entry_price * 0.997
                            stop_loss = pos["stop_loss"]
                        elif move_pct >= 0.20 and stop_loss > (entry_price * 0.999):
                            pos["stop_loss"] = entry_price * 0.999
                            stop_loss = pos["stop_loss"]

                    # AUTO-CLOSE EVALUATION — USER PROFIT RANGE: $0.15 MIN FLOOR, $0.25 TARGET, CAN RUN HIGHER
                    hit_tp = (current_price <= take_profit) if is_short else (current_price >= take_profit)
                    hit_be = (current_price >= stop_loss) if is_short else (current_price <= stop_loss)

                    if risk_manager.is_emergency_stopped:
                        logger.warning(f"AUTO-CLOSE: Manual Emergency stop active for {symbol}")
                        await execution_engine.close_active_position(current_price, "EMERGENCY_STOP")

                    # 1. RUNNER PROFIT LOCK (Above $0.25 Target):
                    # If profit went beyond $0.25 and peaks, but starts pulling back by $0.03, lock in maximum gain!
                    elif peak_profit >= settings.TARGET_PROFIT_USDT_MAX and (peak_profit - net_pnl_usdt) >= 0.03:
                        logger.info(f"AUTO-CLOSE: Trailing profit secured above $0.25 target! Peak was ${peak_profit:.3f}, Closed at +${net_pnl_usdt:.4f} USDT ({symbol} {direction})")
                        await execution_engine.close_active_position(current_price, "TRAILING_PROFIT_RUNNER_SECURED")

                    # 2. EXTENDED RUNNER PROFIT TARGET ($0.35+ USDT):
                    elif net_pnl_usdt >= 0.35:
                        logger.info(f"AUTO-CLOSE: Extended runner target reached (+${net_pnl_usdt:.4f} USDT)! ({symbol} {direction})")
                        await execution_engine.close_active_position(current_price, "EXTENDED_RUNNER_0.35_HIT")

                    # 3. PRIMARY TARGET RANGE HIT ($0.25 USDT Target):
                    # If net profit reaches $0.25 target and hits technical TP or holds for 60s+ in range
                    elif (hit_tp or elapsed >= 60) and net_pnl_usdt >= settings.TARGET_PROFIT_USDT_MAX:
                        logger.info(f"AUTO-CLOSE: Target Profit $0.25 reached in target range! (+${net_pnl_usdt:.4f} USDT) ({symbol} {direction})")
                        await execution_engine.close_active_position(current_price, "TARGET_PROFIT_0.25_HIT")

                    # 4. MINIMUM PROFIT RANGE EXIT ($0.15 - $0.24 USDT Floor):
                    # User rule: "0.15$ se opr profit de agr itna bhe profit aye to thk hy mgr 0.15$ minimum profit range ho"
                    elif net_pnl_usdt >= settings.MIN_PROFIT_USDT_FLOOR:
                        # If scalp duration reached 90s, bank the acceptable profit!
                        if elapsed >= 90:
                            logger.info(f"AUTO-CLOSE: 90s scalp reached in acceptable profit range (+${net_pnl_usdt:.4f} USDT >= $0.15). Securing profit! ({symbol} {direction})")
                            await execution_engine.close_active_position(current_price, "MIN_PROFIT_RANGE_TIME_EXIT")
                        # If price touched peak >= $0.18 and pulled back to $0.15 floor, lock in before dropping below $0.15
                        elif peak_profit >= 0.18 and net_pnl_usdt <= 0.155:
                            logger.info(f"AUTO-CLOSE: Profit floor protection hit (+${net_pnl_usdt:.4f} USDT). Securing minimum $0.15 profit! ({symbol} {direction})")
                            await execution_engine.close_active_position(current_price, "PROFIT_FLOOR_PROTECTION_HIT")

                    # 5. Breakeven Profit Stop Exit:
                    elif pos.get("is_breakeven_locked") and hit_be and net_pnl_usdt >= settings.MIN_PROFIT_USDT_FLOOR:
                        logger.info(f"AUTO-CLOSE: Breakeven Profit lock hit at +${net_pnl_usdt:.4f}! ({symbol} {direction})")
                        await execution_engine.close_active_position(current_price, "BREAKEVEN_PROFIT_EXIT")

                    # 6. STRICT CAPITAL PRESERVATION & ZERO LOSS:
                    # Do NOT close if net_pnl_usdt < 0.15 (let trade reach the $0.15+ profit range)
                    # Never close in negative PnL unless user presses manual Emergency Stop!
                    else:
                        if net_pnl_usdt < settings.MIN_PROFIT_USDT_FLOOR:
                            pos["recovery_mode"] = True
                            pos["status_note"] = "ACCUMULATING_TO_MIN_PROFIT_0.15" if net_pnl_usdt > 0 else "HOLDING_FOR_PROFIT"
                            if elapsed % 30 == 0:
                                logger.info(f"HOLDING: {symbol} Net PnL is ${net_pnl_usdt:.4f}. Waiting for $0.15 - $0.25+ profit range per user rule.")
                        else:
                            pos["recovery_mode"] = False
                            pos["status_note"] = "IN_PROFIT_RANGE"

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(1.0)

position_monitor = PositionMonitorService()
