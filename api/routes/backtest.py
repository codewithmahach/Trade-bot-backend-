from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from backtesting.engine.backtester import backtest_engine

router = APIRouter()

class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    days_back: int

@router.post("/run")
async def run_backtest(req: BacktestRequest):
    """
    Executes a backtest over the specified historical period and returns the performance metrics.
    """
    # In reality, fetch historical data from data.mt5.connector or data.binance.connector
    # For now, pass an empty dataframe to simulate the run
    df = pd.DataFrame() 
    
    strategy_rules = {
        "strict_htf_alignment": True,
        "require_liquidity_sweep": True
    }
    
    results = backtest_engine.run_simulation(df, strategy_rules)
    
    return {
        "symbol": req.symbol,
        "period": f"Last {req.days_back} days",
        "results": results
    }
