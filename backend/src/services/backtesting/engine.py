import numpy as np
import logging
from typing import Dict, Any, List
from backend.src.config.settings import settings
from backend.src.services.binance.client import binance_client
from backend.src.services.technical.indicators import technical_engine
from backend.src.services.quant.scoring import quant_scoring_engine
from backend.src.services.execution.profit_calculator import profit_calculator

logger = logging.getLogger("backtest_engine")

class BacktestingEngine:
    """Historical simulator for 1m/3m/5m crypto scalping strategies."""

    async def run_backtest(
        self,
        symbol: str = "BTCUSDT",
        limit_candles: int = 500,
        amount_pkr: float = 5000.0,
        taker_fee_rate: float = 0.00075,
        slippage_bps: float = 4.0
    ) -> Dict[str, Any]:
        try:
            klines = await binance_client.get_klines(symbol, interval="1m", limit=limit_candles)
            if not klines or len(klines) < 60:
                return {"error": "Insufficient historical candle data"}

            trades = []
            active_trade = None
            equity_curve = [profit_calculator.pkr_to_usdt(amount_pkr, settings.PKR_USDT_RATE)]
            initial_equity = equity_curve[0]

            amount_usdt = initial_equity

            for i in range(50, len(klines)):
                sub_klines = klines[:i]
                current_candle = klines[i]
                # [open_time, open, high, low, close, volume, ...]
                open_price = float(current_candle[1])
                high_price = float(current_candle[2])
                low_price = float(current_candle[3])
                close_price = float(current_candle[4])
                candle_time = int(current_candle[0])

                # 1. Manage active trade
                if active_trade:
                    active_trade["elapsed_minutes"] += 1
                    exit_price = None
                    exit_reason = None

                    # Check TP
                    if high_price >= active_trade["take_profit"]:
                        exit_price = active_trade["take_profit"]
                        exit_reason = "TAKE_PROFIT"
                    # Check SL
                    elif low_price <= active_trade["stop_loss"]:
                        exit_price = active_trade["stop_loss"]
                        exit_reason = "STOP_LOSS"
                    # Time exit (3 minutes scalp duration)
                    elif active_trade["elapsed_minutes"] >= 3:
                        exit_price = close_price
                        exit_reason = "TIME_EXIT"

                    if exit_price:
                        quantity = active_trade["quantity"]
                        entry_price = active_trade["entry_price"]
                        gross_pnl = (exit_price - entry_price) * quantity
                        entry_fee = quantity * entry_price * taker_fee_rate
                        exit_fee = quantity * exit_price * taker_fee_rate
                        slippage = quantity * entry_price * (slippage_bps / 10000.0)
                        net_pnl = gross_pnl - entry_fee - exit_fee - slippage

                        trade_record = {
                            "symbol": symbol,
                            "entry_time": active_trade["entry_time"],
                            "exit_time": candle_time,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "gross_pnl": round(gross_pnl, 4),
                            "net_pnl": round(net_pnl, 4),
                            "exit_reason": exit_reason,
                            "duration_minutes": active_trade["elapsed_minutes"],
                            "is_win": net_pnl > 0
                        }
                        trades.append(trade_record)
                        amount_usdt += net_pnl
                        equity_curve.append(round(amount_usdt, 2))
                        active_trade = None

                # 2. Check for new entry if no active trade
                elif not active_trade:
                    tech = technical_engine.analyze_candles(sub_klines)
                    if not tech:
                        continue

                    dummy_ob = {"imbalance": 0.65, "spread": 0.02}
                    quant = quant_scoring_engine.calculate_score(tech, {}, {}, dummy_ob, 0.02)
                    score = quant.get("total_score", 0.0)

                    # Scalping trigger condition: Score >= 80, positive MACD, clean ATR
                    if score >= 80.0 and tech.get("trend_bullish"):
                        atr_val = tech.get("atr", close_price * 0.003)
                        sl = close_price - (1.5 * atr_val)
                        tp = close_price + (2.5 * atr_val)
                        quantity = amount_usdt / close_price

                        active_trade = {
                            "symbol": symbol,
                            "entry_time": candle_time,
                            "entry_price": close_price,
                            "take_profit": tp,
                            "stop_loss": sl,
                            "quantity": quantity,
                            "elapsed_minutes": 0
                        }

            # Calculate analytics
            total_trades = len(trades)
            if total_trades == 0:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "net_profit_usdt": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown_pct": 0.0,
                    "trades": []
                }

            winning_trades = [t for t in trades if t["is_win"]]
            losing_trades = [t for t in trades if not t["is_win"]]
            win_rate = round((len(winning_trades) / total_trades) * 100.0, 1)
            loss_rate = round(100.0 - win_rate, 1)

            total_gain = sum(t["net_pnl"] for t in winning_trades)
            total_loss = abs(sum(t["net_pnl"] for t in losing_trades))
            profit_factor = round(total_gain / total_loss, 2) if total_loss > 0 else (99.0 if total_gain > 0 else 1.0)
            net_profit_usdt = round(amount_usdt - initial_equity, 2)
            net_profit_pkr = round(net_profit_usdt * settings.PKR_USDT_RATE, 2)

            # Max Drawdown
            eq_arr = np.array(equity_curve)
            peak = np.maximum.accumulate(eq_arr)
            drawdowns = (peak - eq_arr) / peak * 100.0
            max_drawdown_pct = round(float(np.max(drawdowns)), 2)

            avg_trade = round(net_profit_usdt / total_trades, 3)
            avg_duration = round(sum(t["duration_minutes"] for t in trades) / total_trades, 1)
            expectancy = round((len(winning_trades) / total_trades * (total_gain / len(winning_trades) if winning_trades else 0)) -
                               (len(losing_trades) / total_trades * (total_loss / len(losing_trades) if losing_trades else 0)), 3)

            return {
                "symbol": symbol,
                "candle_count": limit_candles,
                "total_trades": total_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "net_profit_usdt": net_profit_usdt,
                "net_profit_pkr": net_profit_pkr,
                "profit_factor": profit_factor,
                "max_drawdown_pct": max_drawdown_pct,
                "avg_trade_usdt": avg_trade,
                "avg_duration_minutes": avg_duration,
                "expectancy": expectancy,
                "equity_curve": equity_curve,
                "trades": trades[-20:] # Return last 20 trades
            }

        except Exception as e:
            logger.error(f"Backtesting error: {e}")
            return {"error": str(e)}

backtest_engine = BacktestingEngine()
