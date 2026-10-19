from fastapi import APIRouter

router = APIRouter()

@router.get("/summary")
async def get_dashboard_summary():
    """
    Returns a high-level summary of system performance and current status.
    """
    return {
        "system_status": "online",
        "active_pairs": ["XAUUSD", "BTC", "ETH", "BNB"],
        "paper_trading_performance": {
            "win_rate": 0.65,
            "profit_factor": 1.5,
            "total_trades": 120
        },
        "current_market_bias": {
            "XAUUSD": "neutral",
            "BTC": "bullish"
        }
    }
