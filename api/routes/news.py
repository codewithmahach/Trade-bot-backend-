from fastapi import APIRouter

router = APIRouter()

from news_engine.scheduled_events.manager import news_engine

@router.get("/status")
async def get_news_status():
    """
    Checks if there's a high-impact news blackout currently active.
    """
    is_blackout = news_engine.check_blackout()
    return {
        "blackout_active": is_blackout,
        "events": news_engine.scheduled_events,
        "message": "Blackout Active: NO NEW TRADE" if is_blackout else "Market conditions stable"
    }

@router.get("/sentiment")
async def get_market_sentiment(asset: str = "BTC"):
    """
    Returns AI-analyzed news sentiment for an asset.
    """
    return {
        "asset": asset,
        "sentiment": "neutral",
        "recent_events": [
            {"title": "ETF Inflows Stabilize", "impact": "low"}
        ]
    }
