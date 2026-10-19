import time
import logging
from typing import Dict, Any, Tuple, Optional
from backend.src.config.settings import settings

logger = logging.getLogger("risk_manager")

class RiskManager:
    """Institutional-grade risk management and capital preservation engine."""

    def __init__(self):
        self.consecutive_losses: int = 0
        self.daily_realized_pnl_usdt: float = 0.0
        self.trades_today_count: int = 0
        self.last_trade_time: float = 0.0
        self.is_emergency_stopped: bool = False
        self.is_paused_by_risk: bool = False
        self.pause_reason: str = ""

    def reset_daily_stats(self):
        self.daily_realized_pnl_usdt = 0.0
        self.trades_today_count = 0
        self.consecutive_losses = 0
        self.is_paused_by_risk = False
        self.pause_reason = ""

    def record_trade_completion(self, net_pnl_usdt: float):
        """Updates loss streaks and daily PnL after a trade concludes."""
        self.trades_today_count += 1
        self.daily_realized_pnl_usdt += net_pnl_usdt
        self.last_trade_time = time.time()

        if net_pnl_usdt < 0:
            self.consecutive_losses += 1
            logger.warning(f"Trade closed in loss: {net_pnl_usdt:.2f} USDT. Streak: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            logger.info(f"Trade closed in profit: +{net_pnl_usdt:.2f} USDT. Loss streak reset.")

        # Capital Preservation check (Section 25)
        if self.consecutive_losses >= settings.MAX_CONSECUTIVE_LOSSES:
            self.is_paused_by_risk = True
            self.pause_reason = f"Risk protection activated. Auto trading paused after {self.consecutive_losses} consecutive losses."
            logger.critical(self.pause_reason)

        elif self.daily_realized_pnl_usdt <= -abs(settings.MAX_DAILY_LOSS_USDT):
            self.is_paused_by_risk = True
            self.pause_reason = f"Risk protection activated. Daily loss limit reached ({self.daily_realized_pnl_usdt:.2f} USDT)."
            logger.critical(self.pause_reason)

    def resume_auto_trading(self):
        """User manual action required to unpause after risk protection trigger."""
        self.is_paused_by_risk = False
        self.pause_reason = ""
        self.consecutive_losses = 0
        logger.info("Auto trading manually resumed by user.")

    def trigger_emergency_stop(self) -> Dict[str, Any]:
        """Immediate kill-switch: stops entries, halts bot."""
        self.is_emergency_stopped = True
        self.is_paused_by_risk = True
        self.pause_reason = "EMERGENCY STOP activated by operator."
        logger.critical("EMERGENCY STOP TRIGGERED.")
        return {
            "status": "STOPPED",
            "message": "Emergency stop initiated. New entries blocked. Auto trading disabled."
        }

    def clear_emergency_stop(self):
        self.is_emergency_stopped = False
        self.is_paused_by_risk = False
        self.pause_reason = ""
        logger.info("Emergency stop cleared.")

    def calculate_dynamic_exits(
        self,
        entry_price: float,
        atr_value: float,
        resistance: float,
        support: float
    ) -> Tuple[float, float, float]:
        """Calculates dynamic ATR and market-structure based Stop-Loss and Take-Profit.
        Returns: (take_profit, stop_loss, risk_reward_ratio)
        """
        if entry_price <= 0:
            return 0.0, 0.0, 0.0

        # Bound ATR between 0.15% and 1.5% of price
        # For ultra-tight 2-minute scalping and zero-loss prevention:
        # SL strictly capped at 1.0x ATR or max 0.50% of price, TP at 2.0x ATR (min 1.6 R:R)
        effective_atr = max(entry_price * 0.002, min(atr_value, entry_price * 0.010))
        max_sl_distance = entry_price * 0.0050  # Hard cap: max 0.50% loss distance

        raw_sl = entry_price - min(max_sl_distance, 1.0 * effective_atr)
        raw_tp = entry_price + (2.0 * effective_atr)

        # Respect support/resistance if nearby
        if support > 0 and (entry_price - support) < max_sl_distance:
            stop_loss = max(raw_sl, support * 0.9995)
        else:
            stop_loss = raw_sl

        if resistance > entry_price and (resistance - entry_price) > (1.2 * effective_atr):
            take_profit = min(raw_tp, resistance * 0.9995)
        else:
            take_profit = raw_tp

        # Ensure minimal risk/reward ratio >= 1.5
        sl_distance = entry_price - stop_loss
        tp_distance = take_profit - entry_price
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0

        if rr_ratio < 1.5:
            take_profit = entry_price + (sl_distance * 1.6)
            rr_ratio = 1.6

        return round(take_profit, 6), round(stop_loss, 6), round(rr_ratio, 2)

    def validate_pre_entry_risk(
        self,
        symbol: str,
        spread_pct: float,
        active_positions_count: int,
        rr_ratio: float,
        balance_usdt: float,
        trade_amount_usdt: float
    ) -> Tuple[bool, str]:
        """Validates all safety and risk constraints before any order can be entered."""
        if self.is_emergency_stopped:
            return False, "EMERGENCY_STOP_ACTIVE: Trading is disabled"

        if self.is_paused_by_risk:
            return False, f"RISK_PROTECTION_PAUSE: {self.pause_reason}"

        if active_positions_count >= settings.MAX_OPEN_POSITIONS:
            return False, f"MAX_POSITIONS_REACHED: Currently holding {active_positions_count} open scalp(s)"

        # Cooldown between scalps
        now = time.time()
        if (now - self.last_trade_time) < settings.COOLDOWN_SECONDS:
            rem = int(settings.COOLDOWN_SECONDS - (now - self.last_trade_time))
            return False, f"COOLDOWN_ACTIVE: {rem}s remaining before next scalp"

        # Spread constraint
        if spread_pct > settings.MAX_SPREAD_PERCENT:
            return False, f"HIGH_SPREAD: Current spread {spread_pct:.3f}% exceeds max {settings.MAX_SPREAD_PERCENT}%"

        # Risk reward constraint
        if rr_ratio < settings.MIN_RR_RATIO:
            return False, f"POOR_RISK_REWARD: Setup R:R {rr_ratio:.2f} is below minimum {settings.MIN_RR_RATIO}"

        # Capital balance check
        if balance_usdt < 0.40:
            return False, f"INSUFFICIENT_BALANCE: Available Futures balance is ${balance_usdt:.2f} USDT. Minimum required is $0.40 USDT."

        return True, "APPROVED"

risk_manager = RiskManager()
