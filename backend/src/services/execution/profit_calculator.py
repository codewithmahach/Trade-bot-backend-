from typing import Dict, Any, Tuple
from backend.src.config.settings import settings

class ProfitCalculator:
    """Centralized high-precision fee, slippage, and net PnL calculator."""

    @staticmethod
    def pkr_to_usdt(pkr_amount: float, rate: float = 278.50) -> float:
        if rate <= 0:
            rate = 278.50
        return round(pkr_amount / rate, 4)

    @staticmethod
    def usdt_to_pkr(usdt_amount: float, rate: float = 278.50) -> float:
        return round(usdt_amount * rate, 2)

    @classmethod
    def calculate_trade_metrics(
        cls,
        amount_pkr: float,
        entry_price: float,
        target_exit_price: float,
        stop_loss_price: float,
        spread_pct: float = 0.02,
        fx_rate: float = 278.50,
        taker_fee_rate: float = 0.00075
    ) -> Dict[str, Any]:
        """Calculates precise Entry Fee, Exit Fee, Slippage, Net Profit and Max Loss in USDT and PKR."""
        amount_usdt = cls.pkr_to_usdt(amount_pkr, fx_rate)
        if entry_price <= 0:
            return {}

        quantity = amount_usdt / entry_price

        # Fees
        entry_fee_usdt = amount_usdt * taker_fee_rate
        target_exit_notional = quantity * target_exit_price
        exit_fee_usdt = target_exit_notional * taker_fee_rate

        # Estimated Slippage (based on spread and execution size)
        # Typically half the spread plus minor market impact
        slippage_rate = max(0.0003, (spread_pct / 100.0) * 0.5)
        slippage_usdt = amount_usdt * slippage_rate

        # Gross PnL for target exit
        gross_profit_usdt = (target_exit_price - entry_price) * quantity
        # Net Profit
        net_profit_usdt = gross_profit_usdt - entry_fee_usdt - exit_fee_usdt - slippage_usdt
        net_profit_pkr = cls.usdt_to_pkr(net_profit_usdt, fx_rate)

        # Maximum loss calculation if stop loss hits
        stop_loss_exit_notional = quantity * stop_loss_price
        stop_loss_fee_usdt = stop_loss_exit_notional * taker_fee_rate
        gross_loss_usdt = (entry_price - stop_loss_price) * quantity # positive number for magnitude of loss
        max_loss_usdt = gross_loss_usdt + entry_fee_usdt + stop_loss_fee_usdt + slippage_usdt
        max_loss_pkr = cls.usdt_to_pkr(max_loss_usdt, fx_rate)

        # Percentage moves
        required_tp_pct = ((target_exit_price - entry_price) / entry_price) * 100.0
        sl_pct = ((entry_price - stop_loss_price) / entry_price) * 100.0
        risk_reward_ratio = (target_exit_price - entry_price) / (entry_price - stop_loss_price) if (entry_price - stop_loss_price) > 0 else 0.0

        return {
            "amount_pkr": round(amount_pkr, 2),
            "amount_usdt": round(amount_usdt, 4),
            "quantity": quantity,
            "entry_price": entry_price,
            "target_exit_price": target_exit_price,
            "stop_loss_price": stop_loss_price,
            "entry_fee_usdt": round(entry_fee_usdt, 5),
            "exit_fee_usdt": round(exit_fee_usdt, 5),
            "slippage_usdt": round(slippage_usdt, 5),
            "total_friction_usdt": round(entry_fee_usdt + exit_fee_usdt + slippage_usdt, 5),
            "gross_profit_usdt": round(gross_profit_usdt, 4),
            "net_profit_usdt": round(net_profit_usdt, 4),
            "net_profit_pkr": round(net_profit_pkr, 2),
            "max_loss_usdt": round(max_loss_usdt, 4),
            "max_loss_pkr": round(max_loss_pkr, 2),
            "required_move_pct": round(required_tp_pct, 3),
            "stop_loss_pct": round(sl_pct, 3),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "fx_rate": fx_rate
        }

    @classmethod
    def validate_target_feasibility(
        cls,
        amount_pkr: float,
        desired_profit_pkr: float,
        entry_price: float,
        atr_pct: float,
        fx_rate: float = 278.50,
        taker_fee_rate: float = 0.00075
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validates if user requested profit is realistically achievable in a 2-3 minute scalp (Section 4)."""
        amount_usdt = cls.pkr_to_usdt(amount_pkr, fx_rate)
        desired_profit_usdt = cls.pkr_to_usdt(desired_profit_pkr, fx_rate)

        if amount_usdt <= 0:
            return False, "Invalid trade amount", {}

        # Support Futures leverage: 20x leverage amplifies margin so required price move is realistic
        is_futures = getattr(settings, "TRADING_MARKET_TYPE", "FUTURES") == "FUTURES"
        leverage = float(getattr(settings, "FUTURES_LEVERAGE", 20)) if is_futures else 1.0
        effective_notional = amount_usdt * max(1.0, leverage)

        # Two round-trip taker fees (approx 0.15%) + minimal slippage (0.05%) = 0.20% friction
        friction_rate = (taker_fee_rate * 2.0) + 0.0005
        friction_usdt = effective_notional * friction_rate

        total_needed_usdt = desired_profit_usdt + friction_usdt
        required_pct_move = (total_needed_usdt / effective_notional) * 100.0

        # Normal short-term 2-3 minute scalp move capacity is bounded by ATR (typically 0.2% - 1.5%)
        max_realistic_scalp_move_pct = max(0.95, min(2.20, atr_pct * 3.0))

        details = {
            "amount_pkr": amount_pkr,
            "desired_profit_pkr": desired_profit_pkr,
            "amount_usdt": round(amount_usdt, 2),
            "effective_notional_usdt": round(effective_notional, 2),
            "leverage": leverage,
            "desired_profit_usdt": round(desired_profit_usdt, 2),
            "required_pct_move": round(required_pct_move, 2),
            "max_realistic_scalp_move_pct": round(max_realistic_scalp_move_pct, 2),
            "friction_usdt": round(friction_usdt, 4)
        }

        if required_pct_move > max_realistic_scalp_move_pct:
            reason = (
                f"Current market conditions do not provide a realistic opportunity for the requested target. "
                f"Requested PKR {desired_profit_pkr:.0f} on PKR {amount_pkr:.0f} requires a {required_pct_move:.2f}% short-term move (at {leverage:.0f}x leverage), "
                f"exceeding the realistic 2-3 minute market capacity ({max_realistic_scalp_move_pct:.2f}%)."
            )
            return False, reason, details

        return True, "Target feasible", details

    @classmethod
    def calculate_actual_exit_pnl(
        cls,
        quantity: float,
        entry_price: float,
        actual_exit_price: float,
        entry_fee_usdt: float,
        taker_fee_rate: float = 0.00075,
        fx_rate: float = 278.50,
        direction: str = "LONG"
    ) -> Dict[str, Any]:
        """Calculates final realized PnL upon trade closure for both LONG and SHORT."""
        exit_notional = quantity * actual_exit_price
        exit_fee_usdt = exit_notional * taker_fee_rate
        if direction.upper() == "SHORT":
            gross_pnl_usdt = (entry_price - actual_exit_price) * quantity
        else:
            gross_pnl_usdt = (actual_exit_price - entry_price) * quantity
        net_pnl_usdt = gross_pnl_usdt - entry_fee_usdt - exit_fee_usdt
        net_pnl_pkr = cls.usdt_to_pkr(net_pnl_usdt, fx_rate)

        return {
            "gross_pnl_usdt": round(gross_pnl_usdt, 4),
            "entry_fee_usdt": round(entry_fee_usdt, 5),
            "exit_fee_usdt": round(exit_fee_usdt, 5),
            "net_pnl_usdt": round(net_pnl_usdt, 4),
            "net_pnl_pkr": round(net_pnl_pkr, 2)
        }

profit_calculator = ProfitCalculator()
