import asyncio
import json
import time
import logging
import websockets
from typing import Dict, Any, List, Optional
from backend.src.config.settings import settings
from backend.src.services.binance.client import binance_client

logger = logging.getLogger("market_data_manager")

class MarketDataManager:
    def __init__(self):
        self.live_tickers: Dict[str, Dict[str, Any]] = {}
        self.live_order_books: Dict[str, Dict[str, Any]] = {}
        self.kline_cache: Dict[str, Dict[str, Any]] = {} # symbol -> {interval: (timestamp, klines)}
        self.is_ws_connected: bool = False
        self.last_ws_time: float = 0
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._running:
            return
        self._running = True
        # Pre-seed tickers with fast REST call
        try:
            await self._seed_rest_tickers()
        except Exception as e:
            logger.warning(f"Initial REST seed warning: {e}")

        # Start WebSocket loop in background
        self._task = asyncio.create_task(self._ws_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _seed_rest_tickers(self):
        tickers = await binance_client.get_24h_tickers()
        for t in tickers:
            symbol = t.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            last_price = float(t.get("lastPrice", 0.0))
            bid_price = float(t.get("bidPrice", last_price))
            ask_price = float(t.get("askPrice", last_price))
            spread = max(0.0, (ask_price - bid_price) / max(bid_price, 0.00000001)) * 100.0

            self.live_tickers[symbol] = {
                "symbol": symbol,
                "price": last_price,
                "bid": bid_price,
                "ask": ask_price,
                "spread": spread,
                "volume24h": float(t.get("volume", 0.0)),
                "quoteVolume24h": float(t.get("quoteVolume", 0.0)),
                "priceChangePercent": float(t.get("priceChangePercent", 0.0)),
                "high24h": float(t.get("highPrice", last_price)),
                "low24h": float(t.get("lowPrice", last_price)),
                "count": int(t.get("count", 0)),
                "timestamp": time.time()
            }
        logger.info(f"Loaded {len(self.live_tickers)} USDT tickers into in-memory state")

    async def _ws_loop(self):
        ws_url = "wss://stream.binance.com:9443/ws/!ticker@arr"
        backoff = 2
        while self._running:
            try:
                logger.info("Connecting to Binance WebSocket Ticker Stream...")
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self.is_ws_connected = True
                    backoff = 2
                    logger.info("Binance WebSocket Ticker Stream CONNECTED.")
                    while self._running:
                        msg = await ws.recv()
                        self.last_ws_time = time.time()
                        data = json.loads(msg)
                        if isinstance(data, list):
                            for item in data:
                                sym = item.get("s", "")
                                if not sym.endswith("USDT"):
                                    continue
                                last_price = float(item.get("c", 0.0))
                                bid_price = float(item.get("b", last_price))
                                ask_price = float(item.get("a", last_price))
                                spread = max(0.0, (ask_price - bid_price) / max(bid_price, 0.00000001)) * 100.0

                                self.live_tickers[sym] = {
                                    "symbol": sym,
                                    "price": last_price,
                                    "bid": bid_price,
                                    "ask": ask_price,
                                    "spread": spread,
                                    "volume24h": float(item.get("v", 0.0)),
                                    "quoteVolume24h": float(item.get("q", 0.0)),
                                    "priceChangePercent": float(item.get("P", 0.0)),
                                    "high24h": float(item.get("h", last_price)),
                                    "low24h": float(item.get("l", last_price)),
                                    "count": int(item.get("n", 0)),
                                    "timestamp": self.last_ws_time
                                }
            except (websockets.ConnectionClosed, Exception) as e:
                self.is_ws_connected = False
                logger.warning(f"Binance WebSocket disconnected ({e}). Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 30)

    async def get_candles(self, symbol: str, interval: str = "1m", limit: int = 100, max_age: float = 3.0) -> List[List[Any]]:
        """Get cached candles or fetch fresh from Binance REST."""
        now = time.time()
        sym_cache = self.kline_cache.setdefault(symbol, {})
        cached = sym_cache.get(interval)
        if cached and (now - cached[0] < max_age):
            return cached[1]

        try:
            klines = await binance_client.get_klines(symbol, interval=interval, limit=limit)
            sym_cache[interval] = (now, klines)
            return klines
        except Exception as e:
            if cached:
                return cached[1]
            raise e

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.live_tickers.get(symbol)

    def get_all_tickers(self) -> Dict[str, Dict[str, Any]]:
        return self.live_tickers

market_data_manager = MarketDataManager()
