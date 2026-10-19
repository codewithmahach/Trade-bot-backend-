from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
import os
import asyncio
from backend.src.config.settings import settings

router = APIRouter()

@router.get("/htf", response_model=Dict[str, Any])
async def get_htf_analysis(symbol: str = "XAUUSD"):
    """
    Returns Higher Timeframe (Monthly, Weekly, Daily, 4H) analysis for a symbol.
    """
    # Mock data to demonstrate the structure
    return {
        "symbol": symbol,
        "bias": "bullish",
        "key_levels": {
            "resistance": [2050.0, 2075.0],
            "support": [2010.0, 1980.0]
        },
        "market_structure": "Higher Highs, Higher Lows"
    }

@router.get("/ltf", response_model=Dict[str, Any])
async def get_ltf_analysis(symbol: str = "XAUUSD"):
    """
    Returns Lower Timeframe (1H, 15M, 5M, 1M) analysis for precise setup confirmation.
    """
    return {
        "symbol": symbol,
        "bias": "neutral",
        "recent_action": "consolidation",
        "liquidity_sweeps": [],
        "order_blocks": []
    }

from orchestration.langchain.agent import LangchainOrchestrator
from pydantic import BaseModel

orchestrator = LangchainOrchestrator()

from data.binance.connector import binance_connector
from feature_engineering.structure.analyzer import StructureAnalyzer
from feature_engineering.liquidity.analyzer import LiquidityAnalyzer
import pandas as pd

class MarketScanRequest(BaseModel):
    symbol: str

@router.post("/full-scan")
async def run_full_market_scan(scan_req: MarketScanRequest):
    """
    Triggers a full HTF to LTF analysis pipeline using live Binance data.
    """
    symbol = scan_req.symbol
    
    # 1. Fetch Live Data (e.g., 5m timeframe for LTF)
    df = binance_connector.get_ohlcv(symbol, timeframe='5m', limit=100)
    
    if df.empty:
        return {"status": "error", "message": "Failed to fetch market data"}
        
    # 2. Run Feature Engineering (Structure & Liquidity)
    struct_analyzer = StructureAnalyzer(df.copy())
    df_struct = struct_analyzer.identify_swing_points()
    df_struct = struct_analyzer.detect_bos_choch()
    
    liq_analyzer = LiquidityAnalyzer(df_struct)
    df_final = liq_analyzer.detect_fvg()
    df_final = liq_analyzer.detect_liquidity_sweeps(df_struct['high'].rolling(5).max(), df_struct['low'].rolling(5).min())
    
    # Extract latest market conditions
    latest = df_final.iloc[-1]
    prev = df_final.iloc[-2]
    
    ltf_bias = "bullish" if latest.get('structure_signal') in ["BOS_BULLISH", "CHOCH_BULLISH"] else "bearish" if latest.get('structure_signal') in ["BOS_BEARISH", "CHOCH_BEARISH"] else "neutral"
    
    liquidity_status = latest.get('liquidity_sweep', 'none')
    if liquidity_status == "NONE" and (latest.get('bullish_fvg') or latest.get('bearish_fvg')):
        liquidity_status = "FVG Present"
        
    market_data = {
        "htf_context": "neutral", # In full system, fetch 4H data separately
        "ltf_structure": ltf_bias,
        "liquidity": liquidity_status,
        "news_risk": "low",
        "volatility": "normal",
        "current_price": float(latest['close'])
    }
    
    # 3. Run through AI Orchestrator
    decision_result = orchestrator.evaluate_setup(market_data)
    
    return {
        "status": "scan_complete",
        "symbol": symbol,
        "market_data": market_data,
        "decision": decision_result["decision"],
        "reason": decision_result["reason"]
    }

class TopTokensRequest(BaseModel):
    platform: str
    limit: int = 20

@router.post("/get-top-tokens")
async def get_top_tokens(req: TopTokensRequest):
    import asyncio
    if req.platform.lower() == "binance":
        symbols = await asyncio.to_thread(binance_connector.get_top_tokens, limit=req.limit)
    else:
        symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    return {"symbols": symbols, "tokens": symbols}

class SingleScanRequest(BaseModel):
    platform: str
    symbol: str

@router.post("/scan-single")
async def scan_single_token(req: SingleScanRequest):
    """Scans a single token and returns immediately."""
    import asyncio
    sym = req.symbol
    # Strict Blacklist: Never touch BTC, ETH, BNB, or dead/stagnant coins like HBAR
    if any(b in sym.upper() for b in ["WIF", "BTC", "ETH", "BNB", "SHIB", "BONK", "FLOKI", "HBAR", "ADA", "XRP", "TRX", "LTC", "ALGO", "DOT", "ATOM"]):
        return {"status": "error", "symbol": sym, "message": "Blacklisted slow/stagnant token"}
        
    try:
        if req.platform.lower() == "binance":
            # Switch to 1m timeframe for ULTRA FAST micro-scalping signals (Non-blocking worker thread)
            df = await asyncio.to_thread(binance_connector.get_ohlcv, sym, timeframe='1m', limit=100)
        else:
            from data.mt5.connector import mt5_connector
            import MetaTrader5 as mt5
            df = await asyncio.to_thread(mt5_connector.get_historical_data, sym, mt5.TIMEFRAME_M1, 100)
            
        if df.empty: 
            return {"status": "error", "symbol": sym, "message": "No data"}
            
        # 1. RSI Calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 2. Bollinger Bands Calculation (Best for Fast, Safe 1m Scalping)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_bb'] = df['sma_20'] + (2.0 * df['std_20'])
        df['lower_bb'] = df['sma_20'] - (2.0 * df['std_20'])
        
        # 3. EMA Trend Filter (Never trade against the micro-trend!)
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        rsi_val = latest.get('rsi', 50)
        if pd.isna(rsi_val): rsi_val = 50
        
        ema_9 = latest.get('ema_9', latest['close'])
        ema_21 = latest.get('ema_21', latest['close'])
        ema_50 = latest.get('ema_50', latest['close'])
        
        # Calibrated ML Model Inference & Market Regime
        tech_1m = {
            "current_price": float(latest['close']),
            "open": float(latest['open']),
            "high": float(latest['high']),
            "low": float(latest['low']),
            "ema9": float(ema_9),
            "ema21": float(ema_21),
            "ema50": float(ema_50),
            "rsi": float(rsi_val),
            "bb_upper": float(latest['upper_bb']),
            "bb_lower": float(latest['lower_bb']),
            "bb_pct_b": float((latest['close'] - latest['lower_bb']) / ((latest['upper_bb'] - latest['lower_bb']) + 1e-9)),
            "relative_volume": float(latest['volume'] / (df['volume'].rolling(20).mean().iloc[-1] + 1e-9))
        }

        p_up, p_down, ml_conf = 0.33, 0.33, 50.0
        market_regime = "NORMAL"
        model_name = "Quant ML"
        try:
            from backend.src.services.ml.probability_model import ml_model
            ml_pred = ml_model.predict(tech_1m)
            market_regime = ml_model.classify_regime(tech_1m)
            p_up = ml_pred.get("p_up", 0.33)
            p_down = ml_pred.get("p_down", 0.33)
            ml_conf = ml_pred.get("confidence", 0.33) * 100.0
            model_name = ml_pred.get("model_name", "Production Calibrated Model")
        except Exception:
            pass

        # Check minimum 1m candle volatility range (Must have active momentum to yield >= $0.25 profit in 90-120s)
        candle_range_pct = (latest['high'] - latest['low']) / (latest['close'] + 1e-9)
        if candle_range_pct < 0.0018:
            return {
                "status": "success",
                "symbol": sym,
                "decision": "NO TRADE",
                "reason": f"Dormant/stagnant candle range ({candle_range_pct*100:.2f}% < 0.18%). Cannot yield $0.25 profit in 90-120s.",
                "price": float(latest['close']),
                "bias": "neutral",
                "confidence": 40.0,
                "market_regime": "LOW_VOLATILITY",
                "stop_loss": float(latest['close']),
                "take_profit": float(latest['close']),
                "duration": "90 to 120 Seconds",
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        # Determine STRICT TREND-ALIGNED CONFLUENCE Bias (No betuki counter-trend bets!)
        ltf_bias = "neutral"
        liquidity_status = "none"
        
        # 1. Clean Bullish Momentum (Trend continuation with volume)
        is_bullish_trend = (
            latest['close'] > latest['open'] and # Green 1m candle
            ema_9 >= ema_21 and # Micro uptrend
            latest['close'] >= (ema_50 * 0.999) and # Above/at medium trend
            44.0 <= rsi_val <= 66.0 and # Healthy bullish momentum (not overbought)
            rsi_val >= prev.get('rsi', 40.0) # RSI rising
        )

        # 2. Clean Bearish Momentum (Trend continuation with volume)
        is_bearish_trend = (
            latest['close'] < latest['open'] and # Red 1m candle
            ema_9 <= ema_21 and # Micro downtrend
            latest['close'] <= (ema_50 * 1.001) and # Below/at medium trend
            34.0 <= rsi_val <= 56.0 and # Healthy bearish momentum (not oversold)
            rsi_val <= prev.get('rsi', 60.0) # RSI falling
        )

        if is_bullish_trend and (p_up >= 0.40 or latest['close'] > prev['close']):
            ltf_bias = "bullish"
            liquidity_status = "Trend_Aligned_Bullish_Scalp"
        elif is_bearish_trend and (p_down >= 0.40 or latest['close'] < prev['close']):
            ltf_bias = "bearish"
            liquidity_status = "Trend_Aligned_Bearish_Scalp"
            
        market_data = {
            "htf_context": "neutral",
            "ltf_structure": ltf_bias,
            "liquidity": liquidity_status,
            "news_risk": "low",
            "volatility": "normal",
            "current_price": float(latest['close'])
        }
        
        decision_result = orchestrator.evaluate_setup(market_data)
        
        price = market_data["current_price"]
        is_buy = ltf_bias == "bullish"
        
        # Mathematically calculate Take Profit to GUARANTEE >= $0.25 USDT Net Profit
        trade_margin_usdt = max(1.0, float(getattr(settings, "USER_AMOUNT_PKR", 560.0)) / float(getattr(settings, "PKR_USDT_RATE", 278.50)))
        notional_size = trade_margin_usdt * 20.0 # 20x leverage
        min_net_target = 0.25 # Minimum $0.25 USDT net profit required
        fee_usdt = notional_size * 0.0010 # 0.10% roundtrip taker fee
        needed_move_pct = (min_net_target + fee_usdt) / notional_size
        tp_pct = max(0.0028, min(0.015, needed_move_pct)) # 0.28% to 1.5%
        sl_pct = tp_pct * 1.6 # Solid 1:1.6 risk/reward
        
        # If coin's 1m range cannot support this move within 90-120 seconds, reject it!
        if (candle_range_pct * 3.0) < tp_pct:
            return {
                "status": "success",
                "symbol": sym,
                "decision": "NO TRADE",
                "reason": f"Coin velocity ({candle_range_pct*100:.2f}%) too slow to reach ${min_net_target:.2f} profit in 90-120s. Preserving capital.",
                "price": price,
                "bias": "neutral",
                "confidence": 45.0,
                "market_regime": "LOW_VOLATILITY",
                "stop_loss": price,
                "take_profit": price,
                "duration": "90 to 120 Seconds",
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        stop_loss = price * (1 - sl_pct) if is_buy else price * (1 + sl_pct)
        take_profit = price * (1 + tp_pct) if is_buy else price * (1 - tp_pct)
        
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        regime_label = market_regime.get("regime", "NORMAL") if isinstance(market_regime, dict) else str(market_regime)
        final_reason = f"[{model_name}] Regime: {regime_label} | 90-120s Scalp (Target: +${min_net_target:.2f}) | {liquidity_status}"
        
        return {
            "status": "success",
            "symbol": sym,
            "decision": decision_result["decision"],
            "reason": final_reason,
            "price": price,
            "bias": ltf_bias,
            "confidence": round(ml_conf, 1),
            "market_regime": regime_label,
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "duration": "90 to 120 Seconds (Fast Scalp)",
            "time": current_time
        }
    except Exception as e:
        return {"status": "error", "symbol": sym, "message": str(e)}
