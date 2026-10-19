import pytest
from backend.src.services.risk.manager import RiskManager
from backend.src.services.risk.no_trade_engine import no_trade_engine

def test_dynamic_exits():
    rm = RiskManager()
    entry = 60000.0
    atr = 150.0
    res = 60500.0
    sup = 59500.0

    tp, sl, rr = rm.calculate_dynamic_exits(entry, atr, res, sup)
    assert tp > entry
    assert sl < entry
    assert rr >= 1.5 # Mandated minimum RR ratio

def test_consecutive_losses_circuit_breaker():
    rm = RiskManager()
    assert rm.is_paused_by_risk is False

    # 1st loss
    rm.record_trade_completion(-2.0)
    assert rm.consecutive_losses == 1
    assert rm.is_paused_by_risk is False

    # 2nd loss
    rm.record_trade_completion(-1.5)
    assert rm.consecutive_losses == 2
    assert rm.is_paused_by_risk is False

    # 3rd loss -> Circuit Breaker Must Trip!
    rm.record_trade_completion(-3.0)
    assert rm.consecutive_losses == 3
    assert rm.is_paused_by_risk is True
    assert "Risk protection activated" in rm.pause_reason

    # Pre-entry check should now reject trades
    approved, reason = rm.validate_pre_entry_risk(
        symbol="BTCUSDT",
        spread_pct=0.01,
        active_positions_count=0,
        rr_ratio=2.0,
        balance_usdt=1000.0,
        trade_amount_usdt=50.0
    )
    assert approved is False
    assert "RISK_PROTECTION_PAUSE" in reason

def test_emergency_stop():
    rm = RiskManager()
    rm.trigger_emergency_stop()
    assert rm.is_emergency_stopped is True

    approved, reason = rm.validate_pre_entry_risk(
        symbol="BTCUSDT",
        spread_pct=0.01,
        active_positions_count=0,
        rr_ratio=2.0,
        balance_usdt=1000.0,
        trade_amount_usdt=50.0
    )
    assert approved is False
    assert "EMERGENCY_STOP_ACTIVE" in reason
