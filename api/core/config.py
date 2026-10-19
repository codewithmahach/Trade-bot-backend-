import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Trading System"
    API_V1_STR: str = "/api/v1"
    
    # Configure these based on your environment
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "https://mine-ai-trading-bot.vercel.app"]
    
    # Add other configuration like DB URIs, API keys here
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    MT5_SERVER: str = os.getenv("MT5_SERVER", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore" # Important: ignores extra keys in .env

settings = Settings()
