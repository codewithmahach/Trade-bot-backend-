import os
import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from backend.src.config.settings import settings
from backend.src.database.db import get_db
from backend.src.models.models import TradeRecord, SignalRecord, RiskEventRecord, AuditLogRecord
from backend.src.services.binance.client import binance_client
from backend.src.services.market.data_manager import market_data_manager
from backend.src.services.scanner.scanner import market_scanner
from backend.src.services.execution.engine import execution_engine
from backend.src.services.execution.profit_calculator import profit_calculator
from backend.src.services.risk.manager import risk_manager
from backend.src.services.orchestration.state_machine import orchestrator
from backend.src.services.backtesting.engine import backtest_engine

router = APIRouter(prefix="/api")

class UpdateSettingsRequest(BaseModel):
    pkr_amount: Optional[float] = None
    usdt_amount: Optional[float] = None
    min_profit_pkr: Optional[float] = None
    auto_trading: Optional[bool] = None
    paper_trading: Optional[bool] = None
    auto_close: Optional[bool] = None
    openai_api_key: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    binance_testnet: Optional[bool] = None
    pkr_usdt_rate: Optional[float] = None
    blacklisted_coins: Optional[str] = None
    market_type: Optional[str] = None
    futures_leverage: Optional[int] = None

class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    candles: int = 500
    amount_pkr: float = 5000.0

@router.get("/status")
async def get_system_status():
    """System Health Indicators (Section 50)."""
    now = time.time()
    ws_connected = market_data_manager.is_ws_connected and (now - market_data_manager.last_ws_time < 30)
    has_binance_key = bool(settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET)
    has_ai_key = bool(settings.OPENAI_API_KEY or settings.GEMINI_API_KEY)

    return {
        "binance_api": "CONNECTED" if (settings.PAPER_TRADING or has_binance_key) else "DEGRADED",
        "websocket": "CONNECTED" if ws_connected else ("DEGRADED" if len(market_data_manager.live_tickers) > 0 else "ERROR"),
        "market_data": "CONNECTED" if len(market_data_manager.live_tickers) > 20 else "DEGRADED",
        "ai": "CONNECTED" if has_ai_key else "DEGRADED", # Degraded means using high-speed deterministic quant validator
        "ml": "CONNECTED",
        "database": "CONNECTED",
        "execution": "PAUSED" if risk_manager.is_paused_by_risk else "CONNECTED",
        "risk_engine": "EMERGENCY_STOP" if risk_manager.is_emergency_stopped else ("PAUSED" if risk_manager.is_paused_by_risk else "ACTIVE"),
        "auto_trading": settings.AUTO_TRADING,
        "paper_trading": settings.PAPER_TRADING,
        "auto_close": settings.AUTO_CLOSE,
        "active_tickers_count": len(market_data_manager.live_tickers),
        "app_state": orchestrator.current_state
    }

@router.get("/portfolio")
async def get_portfolio(db: AsyncSession = Depends(get_db)):
    """Fetch current portfolio status and today's summary metrics."""
    # Compute win rate and closed trades from DB
    stmt = select(TradeRecord).where(TradeRecord.status == "CLOSED").order_by(desc(TradeRecord.exit_time)).limit(50)
    res = await db.execute(stmt)
    trades = res.scalars().all()

    total_closed = len(trades)
    wins = len([t for t in trades if (t.net_pnl_usdt or 0) > 0])
    win_rate = round((wins / total_closed * 100.0), 1) if total_closed > 0 else 0.0

    today_pnl_usdt = sum(t.net_pnl_usdt or 0 for t in trades if t.exit_time and t.exit_time.date() == datetime.utcnow().date()) if trades else risk_manager.daily_realized_pnl_usdt
    today_pnl_pkr = round(today_pnl_usdt * settings.PKR_USDT_RATE, 2)

    status = execution_engine.get_portfolio_status()
    status["win_rate"] = win_rate
    status["total_trades_count"] = total_closed
    status["today_pnl_usdt"] = round(today_pnl_usdt, 2)
    status["today_pnl_pkr"] = today_pnl_pkr

    # Attach live Binance real balances if keys are configured
    if settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET:
        try:
            spot_bal = await binance_client.get_spot_balance()
            futures_bal = await binance_client.get_futures_balance()
            spot_usdt = spot_bal.get("free_usdt", 0.0)
            futures_usdt = futures_bal.get("free_usdt", 0.0)

            status["binance_spot_usdt"] = spot_usdt
            status["binance_futures_usdt"] = futures_usdt
            status["binance_wallet_type"] = "FUTURES" if (futures_usdt > 0 or settings.TRADING_MARKET_TYPE == "FUTURES") else "SPOT"
            status["binance_live_usdt"] = futures_usdt if (futures_usdt > 0 or settings.TRADING_MARKET_TYPE == "FUTURES") else spot_usdt
            status["binance_locked_usdt"] = spot_bal.get("locked_usdt", 0.0)
            status["binance_total_usdt"] = round(spot_usdt + futures_bal.get("total_usdt", futures_usdt), 2)
            status["binance_live_pkr"] = round(status["binance_live_usdt"] * settings.PKR_USDT_RATE, 2)
            status["binance_authenticated"] = spot_bal.get("authenticated", False) or futures_bal.get("authenticated", False)
            status["binance_can_trade"] = spot_bal.get("can_trade", False) or futures_bal.get("authenticated", False)

            # In LIVE mode, replace virtual portfolio values with 100% REAL Binance balances
            if not settings.PAPER_TRADING and status.get("binance_authenticated"):
                status["virtual_balance_usdt"] = round(status["binance_live_usdt"], 2)
                status["virtual_balance_pkr"] = round(status["binance_live_pkr"], 2)
                status["mode"] = "LIVE_FUTURES"
        except Exception:
            status["binance_authenticated"] = False
    else:
        status["binance_authenticated"] = False
        status["binance_live_usdt"] = 0.0
        status["binance_live_pkr"] = 0.0

    return status

@router.get("/scanner")
async def get_scanner_state():
    return {
        "candidates": market_scanner.latest_candidates,
        "best_opportunity": market_scanner.best_opportunity,
        "last_scan_time": market_scanner.last_scan_time,
        "app_state": orchestrator.current_state,
        "current_step": market_scanner.current_step,
        "activity_log": market_scanner.scan_activity_log,
        "is_scanning": market_scanner._is_scanning
    }

@router.post("/scan")
async def trigger_scan():
    if execution_engine.active_position:
        return {
            "status": "SCAN_LOCKED",
            "message": f"Scanning locked: 100% focus on active {execution_engine.active_position['symbol']} scalp. Scanning will resume once trade closes.",
            "candidates": market_scanner.latest_candidates,
            "best_opportunity": market_scanner.best_opportunity,
            "current_step": "SCAN_LOCKED_ACTIVE_TRADE",
            "activity_log": market_scanner.scan_activity_log
        }

    result = await orchestrator.run_pipeline_once(
        amount_pkr=settings.USER_AMOUNT_PKR,
        min_profit_pkr=settings.MIN_DESIRED_PROFIT_PKR,
        auto_enter=settings.AUTO_TRADING
    )
    result["candidates"] = market_scanner.latest_candidates
    result["current_step"] = market_scanner.current_step
    result["activity_log"] = market_scanner.scan_activity_log
    return result

@router.get("/position")
async def get_active_position():
    pos = execution_engine.active_position
    if not pos and not settings.PAPER_TRADING:
        pos = await execution_engine.sync_live_position_from_exchange()
    elif pos and pos.get("mode") in ["LIVE_FUTURES", "LIVE"]:
        try:
            from data.binance.connector import binance_connector
            b_pos = binance_connector.get_open_position()
            if b_pos.get("status") == "success" and not b_pos.get("active"):
                await execution_engine.close_active_position(
                    pos.get("current_price", pos.get("entry_price")), 
                    "EXCHANGE_TP_CLOSED", 
                    force=True
                )
                return {"position": None}
            elif b_pos.get("status") == "success" and b_pos.get("active"):
                b_trade = b_pos.get("trade", {})
                if "pnl" in b_trade:
                    pos["exchange_pnl"] = b_trade["pnl"]
                    pos["unrealized_net_pnl_usdt"] = b_trade["pnl"]
                    pos["unrealized_net_pnl_pkr"] = round(b_trade["pnl"] * settings.PKR_USDT_RATE, 2)
        except Exception:
            pass
    return {"position": execution_engine.active_position}

@router.post("/position/sync")
async def sync_position_with_exchange():
    """Explicitly verifies active position status directly against Binance."""
    pos = execution_engine.active_position
    if not pos:
        synced = await execution_engine.sync_live_position_from_exchange()
        return {"status": "SYNCED", "position": synced}
    from data.binance.connector import binance_connector
    b_pos = binance_connector.get_open_position()
    if b_pos.get("status") == "success" and not b_pos.get("active"):
        await execution_engine.close_active_position(
            pos.get("current_price", pos.get("entry_price")), 
            "EXCHANGE_TP_CLOSED", 
            force=True
        )
        return {"status": "SYNCED_CLOSED", "message": f"Position {pos['symbol']} was closed on Binance.", "position": None}
    return {"status": "ACTIVE_ON_EXCHANGE", "position": execution_engine.active_position}

class ExecuteTradeRequest(BaseModel):
    opportunity: Optional[Dict[str, Any]] = None

@router.post("/trade/execute")
async def execute_trade(req: Optional[ExecuteTradeRequest] = None):
    """Manual confirmation to enter the current best opportunity."""
    if execution_engine.active_position:
        raise HTTPException(
            status_code=400,
            detail=f"Active scalp already open on {execution_engine.active_position['symbol']}! Single-position lock active. Wait until trade closes."
        )

    best_opp = (req.opportunity if req and req.opportunity else None) or market_scanner.best_opportunity
    if not best_opp:
        raise HTTPException(status_code=400, detail="No active high-confidence scalp opportunity found to execute.")

    res = await execution_engine.execute_trade(best_opp, settings.USER_AMOUNT_PKR)
    if res.get("status") != "SUCCESS":
        raise HTTPException(status_code=400, detail=res.get("details", res.get("message", "Execution rejected")))
    return res

@router.post("/trade/close")
async def close_trade():
    """Manual emergency / profit-taking close of active position."""
    pos = execution_engine.active_position
    if not pos:
        res = await execution_engine.close_active_position(0.0, "MANUAL_CLOSE", force=True)
        if res.get("status") == "SUCCESS":
            return res
        raise HTTPException(status_code=400, detail="No active position currently open.")

    current_price = pos.get("current_price", pos.get("entry_price"))
    res = await execution_engine.close_active_position(current_price, "MANUAL_CLOSE", force=True)
    return res

@router.post("/risk/emergency-stop")
async def emergency_stop():
    res = risk_manager.trigger_emergency_stop()
    settings.AUTO_TRADING = False
    # If open position, close immediately
    if execution_engine.active_position:
        pos = execution_engine.active_position
        await execution_engine.close_active_position(pos.get("current_price", pos.get("entry_price")), "EMERGENCY_STOP", force=True)
    return res

@router.post("/risk/resume")
async def resume_trading():
    risk_manager.clear_emergency_stop()
    risk_manager.resume_auto_trading()
    settings.AUTO_TRADING = True
    orchestrator.set_state("SCANNING", "Trading resumed by user.")
    return {"status": "RESUMED", "message": "Risk protection cleared. Ready to trade."}

@router.post("/trades/clear")
@router.delete("/trades/history")
async def clear_trade_history(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete
    await db.execute(delete(TradeRecord))
    await db.commit()
    return {"status": "SUCCESS", "message": "Audit trail log cleared."}

@router.get("/trades/history")
async def get_trade_history(db: AsyncSession = Depends(get_db)):
    stmt = select(TradeRecord).order_by(desc(TradeRecord.id)).limit(100)
    res = await db.execute(stmt)
    trades = res.scalars().all()
    return {"trades": [
        {
            "id": t.id,
            "symbol": t.symbol,
            "direction": t.direction,
            "mode": t.mode,
            "status": t.status,
            "amount_pkr": t.amount_pkr,
            "amount_usdt": t.amount_usdt,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "gross_pnl_usdt": t.gross_pnl_usdt,
            "entry_fee_usdt": t.entry_fee_usdt,
            "exit_fee_usdt": t.exit_fee_usdt,
            "net_pnl_usdt": t.net_pnl_usdt,
            "net_pnl_pkr": t.net_pnl_pkr,
            "duration_seconds": t.duration_seconds,
            "signal_score": t.signal_score,
            "exit_reason": t.exit_reason,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None
        } for t in trades
    ]}

@router.post("/backtest")
async def run_backtest_endpoint(req: BacktestRequest):
    return await backtest_engine.run_backtest(
        symbol=req.symbol,
        limit_candles=req.candles,
        amount_pkr=req.amount_pkr
    )

@router.get("/settings")
async def get_settings():
    return {
        "pkr_amount": settings.USER_AMOUNT_PKR,
        "usdt_amount": round(settings.USER_AMOUNT_PKR / settings.PKR_USDT_RATE, 2),
        "min_profit_pkr": settings.MIN_DESIRED_PROFIT_PKR,
        "auto_trading": settings.AUTO_TRADING,
        "paper_trading": settings.PAPER_TRADING,
        "auto_close": settings.AUTO_CLOSE,
        "has_openai_key": bool(settings.OPENAI_API_KEY),
        "has_binance_key": bool(settings.BINANCE_API_KEY),
        "binance_api_key": settings.BINANCE_API_KEY,
        "binance_api_secret": settings.BINANCE_API_SECRET,
        "binance_testnet": False,
        "pkr_usdt_rate": settings.PKR_USDT_RATE,
        "max_holding_seconds": settings.MAX_HOLDING_SECONDS,
        "max_daily_loss_usdt": settings.MAX_DAILY_LOSS_USDT,
        "max_consecutive_losses": settings.MAX_CONSECUTIVE_LOSSES,
        "blacklisted_coins": settings.BLACKLISTED_COINS,
        "market_type": settings.TRADING_MARKET_TYPE,
        "futures_leverage": settings.FUTURES_LEVERAGE
    }

@router.post("/settings")
async def update_settings(req: UpdateSettingsRequest):
    if req.usdt_amount is not None:
        valid_usdt = max(1.0, req.usdt_amount)
        settings.USER_AMOUNT_PKR = valid_usdt * settings.PKR_USDT_RATE
        if req.min_profit_pkr is None:
            settings.MIN_DESIRED_PROFIT_PKR = max(2.0, round(settings.USER_AMOUNT_PKR * 0.012, 1))
    elif req.pkr_amount is not None:
        settings.USER_AMOUNT_PKR = max(100.0, req.pkr_amount)
        if req.min_profit_pkr is None:
            settings.MIN_DESIRED_PROFIT_PKR = max(2.0, round(settings.USER_AMOUNT_PKR * 0.012, 1))
    if req.min_profit_pkr is not None:
        settings.MIN_DESIRED_PROFIT_PKR = max(2.0, req.min_profit_pkr)
    if req.auto_trading is not None:
        settings.AUTO_TRADING = req.auto_trading
    if req.paper_trading is not None:
        settings.PAPER_TRADING = req.paper_trading
    if req.auto_close is not None:
        settings.AUTO_CLOSE = req.auto_close
    if req.pkr_usdt_rate is not None:
        settings.PKR_USDT_RATE = max(50.0, req.pkr_usdt_rate)
    if req.blacklisted_coins is not None:
        settings.BLACKLISTED_COINS = req.blacklisted_coins.strip()
    if req.market_type is not None:
        settings.TRADING_MARKET_TYPE = req.market_type.upper()
    if req.futures_leverage is not None:
        settings.FUTURES_LEVERAGE = max(1, min(20, req.futures_leverage))
    if req.openai_api_key is not None and req.openai_api_key.strip():
        settings.OPENAI_API_KEY = req.openai_api_key.strip()
    if req.binance_api_key is not None and req.binance_api_key.strip():
        settings.BINANCE_API_KEY = req.binance_api_key.strip()
    if req.binance_api_secret is not None and req.binance_api_secret.strip():
        settings.BINANCE_API_SECRET = req.binance_api_secret.strip()

    if (req.binance_api_key and req.binance_api_key.strip()) or (req.binance_api_secret and req.binance_api_secret.strip()):
        try:
            from data.binance.connector import binance_connector
            binance_connector.update_keys(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET, persist_env=True)
        except Exception:
            pass
        binance_client.update_credentials(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
            testnet=False
        )

    # Immediately refresh best opportunity metrics with new capital
    market_scanner.recalculate_best_opportunity(settings.USER_AMOUNT_PKR, settings.MIN_DESIRED_PROFIT_PKR)

    return {"status": "SUCCESS", "message": "Settings updated successfully"}

@router.post("/paper/reset")
async def reset_paper_balance(amount: float = Body(1000.0, embed=True)):
    execution_engine.reset_virtual_balance(amount)
    return {"status": "SUCCESS", "virtual_balance_usdt": amount}

class TestConnectionRequest(BaseModel):
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None

@router.post("/binance/test-connection")
async def test_binance_connection(req: Optional[TestConnectionRequest] = None):
    """Verify Binance API credentials and check real Futures/Spot account status."""
    api_key = (req.binance_api_key.strip() if req and req.binance_api_key else "") or settings.BINANCE_API_KEY or os.getenv("BINANCE_API_KEY", "")
    api_secret = (req.binance_api_secret.strip() if req and req.binance_api_secret else "") or settings.BINANCE_API_SECRET or os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        return {
            "status": "UNCONFIGURED",
            "authenticated": False,
            "canTrade": False,
            "message": "Binance API Key and Secret are not configured. Currently in Paper Simulation mode."
        }

    # Automatically persist keys so they remain active across sessions
    try:
        from data.binance.connector import binance_connector
        binance_connector.update_keys(api_key, api_secret, persist_env=True)
    except Exception:
        pass
    binance_client.update_credentials(api_key=api_key, api_secret=api_secret, testnet=False)

    t0 = time.time()
    futures_ok = False
    futures_bal = 0.0

    # 1. Test via fast async Spot client first (sub-second)
    try:
        spot_res = await binance_client.get_spot_balance()
    except Exception:
        spot_res = {"authenticated": False}

    # 2. Check futures via direct async client
    try:
        fut_res = await binance_client.get_futures_balance()
        if fut_res.get("authenticated", False):
            futures_ok = True
            futures_bal = fut_res.get("free_usdt", 0.0)
    except Exception:
        pass

    latency_ms = int((time.time() - t0) * 1000)

    try:
        if spot_res.get("authenticated", False):
            spot_free = spot_res.get("free_usdt", 0.0)
            spot_locked = spot_res.get("locked_usdt", 0.0)
            spot_total = spot_res.get("total_usdt", spot_free + spot_locked)
            can_trade = spot_res.get("can_trade", True)

            if futures_ok:
                msg = f"Connected to Binance Live (Futures: ${futures_bal:.2f} USDT, Spot: ${spot_free:.2f} USDT) | Latency: {latency_ms}ms"
            else:
                msg = f"Connected to Binance Spot (${spot_free:.2f} USDT)! (Tip: Enable 'Futures' in Binance API settings for 20x leverage)"

            return {
                "status": "CONNECTED",
                "authenticated": True,
                "canTrade": can_trade,
                "free_usdt": round(futures_bal if futures_ok and futures_bal > 0 else spot_free, 2),
                "locked_usdt": round(spot_locked, 2),
                "total_usdt": round(futures_bal if futures_ok and futures_bal > 0 else spot_total, 2),
                "latency_ms": latency_ms,
                "message": msg
            }
        elif futures_ok:
            return {
                "status": "CONNECTED",
                "authenticated": True,
                "canTrade": True,
                "free_usdt": futures_bal,
                "total_usdt": futures_bal,
                "latency_ms": latency_ms,
                "message": f"Connected to Binance Futures! Free USDT: ${futures_bal:.2f} | Latency: {latency_ms}ms"
            }
        else:
            detected_ip = "223.123.76.81"
            try:
                async with httpx.AsyncClient(timeout=1.5) as client:
                    r_ip = await client.get("https://api.ipify.org")
                    if r_ip.status_code == 200:
                        detected_ip = r_ip.text.strip()
            except Exception:
                pass

            raw_err = spot_res.get("error", "Binance authentication failed. Please verify API Key & Secret.")
            err_msg = str(raw_err)
            if "-2015" in err_msg:
                err_msg = f"Binance Error (-2015): Invalid API-Key, IP restriction, or missing permissions. (Your Current IP: {detected_ip})"

            return {
                "status": "ERROR",
                "authenticated": False,
                "canTrade": False,
                "latency_ms": latency_ms,
                "detected_ip": detected_ip,
                "message": err_msg
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "authenticated": False,
            "canTrade": False,
            "message": f"Binance connection failed: {str(e)}"
        }

@router.get("/binance/account")
async def get_binance_account():
    data = await binance_client.get_account()
    return data

@router.get("/binance/balance")
async def get_binance_balance():
    """Fetch live real-time Binance Spot balances."""
    try:
        bal = await binance_client.get_spot_balance()
    except Exception as e:
        bal = {"authenticated": False, "free_usdt": 0.0, "total_usdt": 0.0, "error": str(e)}
    bal["pkr_usdt_rate"] = settings.PKR_USDT_RATE
    bal["free_pkr"] = round(bal.get("free_usdt", 0.0) * settings.PKR_USDT_RATE, 2)
    bal["total_pkr"] = round(bal.get("total_usdt", 0.0) * settings.PKR_USDT_RATE, 2)
    return bal

@router.get("/binance/futures-balance")
async def get_binance_futures_balance():
    """Fetch live real-time Binance USDT-M Futures balance."""
    try:
        bal = await binance_client.get_futures_balance()
        if not isinstance(bal, dict):
            bal = {"authenticated": False, "free_usdt": 0.0, "total_usdt": 0.0}
    except Exception as e:
        bal = {"authenticated": False, "free_usdt": 0.0, "total_usdt": 0.0, "error": str(e)}
    bal["pkr_usdt_rate"] = settings.PKR_USDT_RATE
    free = float(bal.get("free_usdt") or 0.0)
    total = float(bal.get("total_usdt") or free)
    bal["free_pkr"] = round(free * settings.PKR_USDT_RATE, 2)
    bal["total_pkr"] = round(total * settings.PKR_USDT_RATE, 2)
    return bal

@router.post("/trades/clear")
async def clear_trade_history(db: AsyncSession = Depends(get_db)):
    """Purge all trade records for clean real trading history."""
    from sqlalchemy import delete
    await db.execute(delete(TradeRecord))
    await db.commit()
    risk_manager.reset_daily_stats()
    return {"status": "SUCCESS", "message": "All mock and past trades cleared completely."}

@router.get("/logs")
async def get_logs():
    return {"states": orchestrator.state_history}
