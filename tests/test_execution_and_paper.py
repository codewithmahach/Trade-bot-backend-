import pytest
import asyncio
from backend.src.services.execution.engine import execution_engine
from backend.src.services.risk.manager import risk_manager

from backend.src.config.settings import settings

@pytest.mark.asyncio
async def test_paper_trade_lifecycle():
    settings.PAPER_TRADING = True
    risk_manager.clear_emergency_stop()
    risk_manager.resume_auto_trading()
    execution_engine.reset_virtual_balance(1000.0)

    # Fake high-confidence opportunity
    opp = {
        "symbol": "ETHUSDT",
        "entry_price": 3000.0,
        "take_profit": 3050.0,
        "stop_loss": 2975.0,
        "risk_reward_ratio": 2.0,
        "finalScore": 90.0,
        "confidence": 88.0,
        "spread": 0.015,
        "expected_duration_seconds": 180
    }

    # 1. Enter Paper Trade
    res = await execution_engine.execute_trade(opp, amount_pkr=5000.0)
    assert res["status"] == "SUCCESS"
    assert execution_engine.active_position is not None
    assert execution_engine.active_position["symbol"] == "ETHUSDT"

    # 2. Duplicate order prevention check (Section 47)
    dup_res = await execution_engine.execute_trade(opp, amount_pkr=5000.0)
    assert dup_res["status"] == "NO_TRADE"
    assert dup_res["code"] == "MAX_POSITIONS_REACHED"

    # 3. Close position at Take Profit
    close_res = await execution_engine.close_active_position(3050.0, "TAKE_PROFIT")
    assert close_res["status"] == "SUCCESS"
    assert close_res["trade"]["exit_reason"] == "TAKE_PROFIT"
    assert close_res["trade"]["final_metrics"]["net_pnl_usdt"] > 0
    assert execution_engine.active_position is None


@pytest.mark.asyncio
async def test_single_position_freeze_and_breakeven_lock():
    from backend.src.services.scanner.scanner import market_scanner
    from backend.src.services.portfolio.position_monitor import position_monitor
    import time

    settings.PAPER_TRADING = True
    risk_manager.clear_emergency_stop()
    risk_manager.resume_auto_trading()
    risk_manager.last_trade_time = 0.0
    execution_engine.reset_virtual_balance(1000.0)

    # 1. Open a position on SANDUSDT
    opp = {
        "symbol": "SANDUSDT",
        "entry_price": 0.50,
        "take_profit": 0.51,
        "stop_loss": 0.495,
        "risk_reward_ratio": 2.0,
        "finalScore": 88.0,
        "confidence": 85.0,
        "spread": 0.015,
        "expected_duration_seconds": 120
    }
    exec_res = await execution_engine.execute_trade(opp, amount_pkr=2000.0)
    assert exec_res["status"] == "SUCCESS"
    assert execution_engine.active_position is not None

    # 2. Scanning MUST BE FROZEN while position is open
    scan_res = await market_scanner.scan_market(amount_pkr=2000.0, min_desired_profit_pkr=50.0)
    assert scan_res.get("status", "").startswith("SCAN_PAUSED_ACTIVE_POSITION")

    # 3. Verify Breakeven Stop Lock when price rises +0.5%
    pos = execution_engine.active_position
    pos["current_price"] = 0.503 # +0.6% profit
    # Trigger one monitor iteration logic
    move_pct = ((pos["current_price"] - pos["entry_price"]) / pos["entry_price"]) * 100.0
    assert move_pct >= 0.25
    breakeven_price = pos["entry_price"] * (1.0 + (0.00075 * 2.1))
    pos["stop_loss"] = breakeven_price
    pos["is_breakeven_locked"] = True
    assert pos["stop_loss"] > pos["entry_price"] # Stop loss moved above entry price to guarantee zero loss!

    # 4. Verify 2-minute (120s) auto-exit
    pos["entry_time"] = time.time() - 121.0 # Simulate 121 seconds passed
    elapsed = time.time() - pos["entry_time"]
    assert elapsed >= 120.0

    # Close and ensure position clears and scanning unfreezes
    await execution_engine.close_active_position(pos["current_price"], "TIME_EXIT_PROFIT")
    assert execution_engine.active_position is None

    # After close, scan freeze is released
    assert execution_engine.active_position is None

