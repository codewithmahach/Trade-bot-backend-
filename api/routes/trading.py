from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class TradeSetup(BaseModel):
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    setup_quality: float

from strategy.trade_state.manager import trade_state_manager

class LiveTradeRequest(BaseModel):
    platform: str
    symbol: str
    bias: str # 'bullish' or 'bearish'
    amount: float
    current_price: float
    take_profit: float = None
    stop_loss: float = None
    binance_api_key: str = None
    binance_api_secret: str = None

@router.post("/execute-live-trade")
async def execute_live_trade(req: LiveTradeRequest):
    """
    Executes a real live market order on the selected platform.
    """
    if not trade_state_manager.check_can_trade():
        return {"status": "error", "message": "Another position is currently active."}
        
    side = "BUY" if req.bias.lower() == "bullish" else "SELL"
    
    if req.platform.lower() == "binance":
        from data.binance.connector import binance_connector
        # Formatting symbol for CCXT (e.g., BTC/USDT)
        sym = req.symbol if "/" in req.symbol else f"{req.symbol.replace('USDT', '')}/USDT"
        
        result = binance_connector.execute_trade(
            symbol=sym, 
            side=side, 
            amount_usdt=req.amount, 
            current_price=req.current_price,
            tp_price=req.take_profit,
            sl_price=req.stop_loss,
            api_key=req.binance_api_key,
            api_secret=req.binance_api_secret
        )
    else:
        from data.mt5.connector import mt5_connector
        result = mt5_connector.execute_trade(req.symbol, side, req.amount)
        
    if result.get("status") == "success":
        # Register the trade state
        trade_state_manager.open_trade({
            "platform": req.platform,
            "symbol": req.symbol,
            "side": side,
            "amount": req.amount,
            "entry_price": req.current_price
        })
        return {"status": "success", "message": "Live order executed!", "details": result}
        
    return {"status": "error", "message": result.get("message", "Execution failed.")}

@router.get("/state")
async def get_trade_state():
    """
    Returns the current ONE-TRADE-AT-A-TIME state.
    """
    return trade_state_manager.get_state()

@router.post("/close")
async def close_trade(reason: str = "target_reached"):
    """
    Closes the active trade.
    """
    success = trade_state_manager.close_trade(reason)
    return {"status": "success" if success else "failed", "active_position": trade_state_manager.active_position}

class CloseLivePositionRequest(BaseModel):
    platform: str
    symbol: str
    binance_api_key: str = None
    binance_api_secret: str = None

@router.post("/close-live-position")
async def close_live_position(req: CloseLivePositionRequest):
    """Closes an active live position immediately on the exchange at market price."""
    if req.platform.lower() == "binance":
        from data.binance.connector import binance_connector
        res = binance_connector.close_position(req.symbol, req.binance_api_key, req.binance_api_secret)
        if res.get("status") == "success":
            trade_state_manager.close_trade("live_market_exit")
        return res
    return {"status": "error", "message": "Not implemented"}

class SyncRequest(BaseModel):
    platform: str
    binance_api_key: str = None
    binance_api_secret: str = None

@router.post("/sync")
async def sync_trade_state(req: SyncRequest):
    """Syncs the bot's trade state directly from Binance Futures."""
    if req.platform.lower() == "binance":
        from data.binance.connector import binance_connector
        result = binance_connector.get_open_position(req.binance_api_key, req.binance_api_secret)
        if result.get("status") == "success":
            if result.get("active"):
                # Ensure local state manager blocks new trades
                trade_state_manager.active_position = True
                trade_state_manager.current_trade = result.get("trade")
            else:
                trade_state_manager.active_position = False
                trade_state_manager.current_trade = None
        return result
    return {"status": "error", "message": "Not implemented"}

class BalanceRequest(BaseModel):
    platform: str
    binance_api_key: str = None
    binance_api_secret: str = None

@router.post("/balance")
async def get_balance(req: BalanceRequest):
    """Fetches the current balance for the given API keys."""
    if req.platform.lower() == "binance":
        from data.binance.connector import binance_connector
        return binance_connector.get_balance(req.binance_api_key, req.binance_api_secret)
    return {"status": "error", "message": "Balance not implemented for MT5"}

