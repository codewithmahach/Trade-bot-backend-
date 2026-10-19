import pandas as pd
from typing import Dict, Any, List

class MetricsCalculator:
    @staticmethod
    def calculate_performance(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates core performance metrics from a list of completed trades.
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0
            }

        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        
        gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        total_pnl = gross_profit - gross_loss
        
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        # Simple max drawdown calculation based on cumulative PNL
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        
        for trade in trades:
            cumulative_pnl += trade.get('pnl', 0)
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "max_drawdown": round(max_drawdown, 2),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades)
        }
