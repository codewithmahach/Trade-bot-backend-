import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "AI Crypto Scalping Trading Bot"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # Binance API Settings (Backend only - never expose secret to client)
    BINANCE_API_KEY: str = Field(default="", description="Binance API Key")
    BINANCE_API_SECRET: str = Field(default="", description="Binance API Secret")
    BINANCE_BASE_URL: str = "https://api.binance.com"
    BINANCE_FUTURES_BASE_URL: str = "https://fapi.binance.com"
    BINANCE_WS_URL: str = "wss://fstream.binance.com/ws"
    BINANCE_FUTURES_WS_URL: str = "wss://fstream.binance.com/ws"
    
    # Market Type & Futures Leverage
    TRADING_MARKET_TYPE: str = Field(default="FUTURES", description="AUTO, SPOT, or FUTURES")
    FUTURES_LEVERAGE: int = Field(default=20, description="Futures default scalping leverage (capped at 20x)")
    MAX_LEVERAGE_CAP: int = Field(default=20, description="Strict hard cap on leverage across all models")
    MAX_NOTIONAL_USDT: float = Field(default=50.0, description="Maximum single-trade position notional in USDT")
    
    # AI Model Settings
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
    OPENAI_MODEL_FAST: str = "gpt-4o-mini"
    OPENAI_MODEL_STRONG: str = "gpt-4o"
    GEMINI_API_KEY: str = Field(default="", description="Gemini API Key (optional fallback)")
    SHADOW_MODE: bool = Field(default=False, description="Shadow mode: evaluate ML models out-of-sample without placing real orders")
    
    # Currency and FX conversion
    PKR_USDT_RATE: float = Field(default=278.50, description="Exchange rate: 1 USDT in PKR")
    USER_AMOUNT_PKR: float = Field(default=5000.0, description="Current user trade amount in PKR")
    MIN_DESIRED_PROFIT_PKR: float = Field(default=42.0, description="User requested minimum profit in PKR (approx $0.15 USDT)")
    MIN_PROFIT_USDT_FLOOR: float = Field(default=0.15, description="Strict minimum net profit in USDT before exiting")
    TARGET_PROFIT_USDT_MAX: float = Field(default=0.25, description="Primary target profit in USDT (can run higher if momentum holds)")
    
    # Trading Modes and Controls
    PAPER_TRADING: bool = Field(default=False, description="Default Live Binance Futures mode")
    AUTO_TRADING: bool = Field(default=True, description="Auto-trading loop enabled")
    AUTO_CLOSE: bool = Field(default=True, description="Enforce auto-close at TP, SL or time expiration")
    
    # Risk and Scalping Parameters
    MAX_HOLDING_SECONDS: int = Field(default=120, description="Target scalp duration in seconds (strict 2 mins)")
    MAX_HOLDING_SECONDS_LIMIT: int = Field(default=135, description="Hard maximum duration (2 mins 15s)")
    BLACKLISTED_COINS: str = Field(default="BTCUSDT,ETHUSDT,BNBUSDT,WIFUSDT", description="Comma-separated tokens to ignore")
    MAX_CONSECUTIVE_LOSSES: int = Field(default=3, description="Stop after 3 consecutive losses for circuit breaker protection")
    MAX_DAILY_LOSS_USDT: float = Field(default=10.0, description="Max daily loss allowed in USDT")
    MAX_OPEN_POSITIONS: int = Field(default=1, description="Max simultaneous open scalp trades")
    COOLDOWN_SECONDS: int = Field(default=30, description="Cooldown between trades")
    MAX_SLIPPAGE_BPS: float = Field(default=8.0, description="Max acceptable slippage in basis points (0.08%)")
    MAX_SPREAD_PERCENT: float = Field(default=0.08, description="Max acceptable spread percentage")
    MIN_RR_RATIO: float = Field(default=1.5, description="Minimum Risk/Reward ratio")
    MIN_EXPECTED_NET_RETURN_BPS: float = Field(default=6.0, description="Minimum net expected return in basis points after fees (0.06%)")
    
    # USDⓈ-M Futures Fee Rates (Binance standard: 0.05% taker, 0.02% maker)
    TAKER_FEE_RATE: float = Field(default=0.00050, description="Futures taker fee fraction (0.050%)")
    MAKER_FEE_RATE: float = Field(default=0.00020, description="Futures maker fee fraction (0.020%)")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./trading_bot.db", description="Database connection URL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()
