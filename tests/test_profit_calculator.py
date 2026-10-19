import pytest
from backend.src.services.execution.profit_calculator import profit_calculator

def test_pkr_usdt_conversion():
    usdt = profit_calculator.pkr_to_usdt(5000, 278.50)
    assert round(usdt, 2) == 17.95

    pkr = profit_calculator.usdt_to_pkr(17.95, 278.50)
    assert round(pkr, 0) == 4999.0

def test_trade_metrics_calculation():
    # Trade: Entry 100, TP 102 (2% gain), SL 99 (1% risk)
    metrics = profit_calculator.calculate_trade_metrics(
        amount_pkr=5000.0,
        entry_price=100.0,
        target_exit_price=102.0,
        stop_loss_price=99.0,
        spread_pct=0.02,
        fx_rate=278.50,
        taker_fee_rate=0.00075
    )

    assert metrics["gross_profit_usdt"] > 0
    assert metrics["entry_fee_usdt"] > 0
    assert metrics["exit_fee_usdt"] > 0
    assert metrics["slippage_usdt"] > 0
    assert metrics["net_profit_usdt"] < metrics["gross_profit_usdt"] # Fees deducted
    assert metrics["net_profit_usdt"] > 0
    assert metrics["risk_reward_ratio"] == 2.0 # 2 / 1 = 2.0

def test_unrealistic_target_rejection():
    # User asks for PKR 2,000 profit on PKR 5,000 capital (requires >40% move in 2-3 mins!)
    feasible, reason, details = profit_calculator.validate_target_feasibility(
        amount_pkr=5000.0,
        desired_profit_pkr=2000.0,
        entry_price=100.0,
        atr_pct=0.3,
        fx_rate=278.50
    )

    assert feasible is False
    assert "Current market conditions do not provide a realistic opportunity" in reason
