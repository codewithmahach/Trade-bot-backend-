import asyncio
import json
import time
import logging
import httpx
import websockets
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("order_book_sync")

class SynchronizedOrderBook:
    """Institutional L2 Local Order Book with strict Binance sequence validation.
    
    Adheres strictly to Binance USDⓈ-M Futures depth synchronization specifications:
    1. Buffer WebSocket depth events (<symbol>@depth@100ms).
    2. Retrieve REST depth snapshot with lastUpdateId.
    3. Drop buffered events where u < lastUpdateId.
    4. First event must cover lastUpdateId + 1: U <= lastUpdateId + 1 and u >= lastUpdateId + 1.
    5. Maintain continuous updates where U == prev_u + 1.
    6. Drop zero-quantity levels and maintain sorted bids (descending) and asks (ascending).
    """

    def __init__(self, symbol: str):
        self.symbol: str = symbol.upper().replace("/", "")
        self.bids: Dict[float, float] = {}  # price -> qty
        self.asks: Dict[float, float] = {}  # price -> qty
        self.last_update_id: int = 0
        self.prev_u: int = 0
        self.is_synchronized: bool = False
        self.sync_time: float = 0.0
        self.last_event_time: float = 0.0
        self._buffer: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_sync_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _fetch_snapshot(self) -> Optional[Dict[str, Any]]:
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit=500"
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(f"Failed to fetch depth snapshot for {self.symbol}: {resp.status_code}")
        except Exception as e:
            logger.debug(f"Snapshot fetch retry notice for {self.symbol}: {e}")
        return None

    async def _apply_snapshot(self, snapshot: Dict[str, Any]):
        self.last_update_id = int(snapshot.get("lastUpdateId", 0))
        self.bids = {float(p): float(q) for p, q in snapshot.get("bids", []) if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in snapshot.get("asks", []) if float(q) > 0}
        self.prev_u = self.last_update_id
        self.sync_time = time.time()
        logger.info(f"L2 Snapshot applied for {self.symbol}. lastUpdateId: {self.last_update_id}, Bids: {len(self.bids)}, Asks: {len(self.asks)}")

    def _apply_depth_event(self, event: Dict[str, Any]) -> bool:
        U = int(event.get("U", 0))
        u = int(event.get("u", 0))

        # Check sequence continuity
        if self.is_synchronized:
            pu = int(event.get("pu", 0))
            if pu != 0 and pu != self.prev_u and abs(pu - self.prev_u) > 1:
                logger.debug(f"Sequence gap for {self.symbol}: pu={pu} != prev_u={self.prev_u}")
                return False

        # Apply bid updates
        for p_str, q_str in event.get("b", []):
            p = float(p_str)
            q = float(q_str)
            if q == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q

        # Apply ask updates
        for p_str, q_str in event.get("a", []):
            p = float(p_str)
            q = float(q_str)
            if q == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q

        self.prev_u = u
        self.last_event_time = time.time()
        return True

    async def _run_sync_loop(self):
        stream_name = f"{self.symbol.lower()}@depth@100ms"
        ws_url = f"wss://fstream.binance.com/ws/{stream_name}"
        backoff = 1.0

        while self._running:
            try:
                self.is_synchronized = False
                self._buffer.clear()

                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    logger.info(f"Connected to USD-M Depth WebSocket for {self.symbol}.")
                    backoff = 1.0

                    # 1. Buffer events until snapshot is obtained
                    snapshot = None
                    fetch_snapshot_task = asyncio.create_task(self._fetch_snapshot())

                    while self._running:
                        msg = await ws.recv()
                        event = json.loads(msg)
                        self._buffer.append(event)

                        if fetch_snapshot_task.done() and snapshot is None:
                            snapshot = fetch_snapshot_task.result()
                            if not snapshot:
                                await asyncio.sleep(0.5)
                                fetch_snapshot_task = asyncio.create_task(self._fetch_snapshot())
                                continue

                            async with self._lock:
                                await self._apply_snapshot(snapshot)

                                # Process buffered events
                                processed_first = False
                                for ev in self._buffer:
                                    u = int(ev.get("u", 0))
                                    U = int(ev.get("U", 0))
                                    if u < self.last_update_id:
                                        continue
                                    if not processed_first:
                                        if (U <= self.last_update_id + 1 and u >= self.last_update_id) or (u >= self.last_update_id):
                                            self._apply_depth_event(ev)
                                            processed_first = True
                                            self.is_synchronized = True
                                    else:
                                        if not self._apply_depth_event(ev):
                                            self.is_synchronized = False
                                            break

                                self._buffer.clear()

                        elif self.is_synchronized:
                            async with self._lock:
                                if not self._apply_depth_event(event):
                                    logger.debug(f"Sequence broke on live event for {self.symbol}. Reconnecting...")
                                    break

            except (websockets.ConnectionClosed, Exception) as e:
                self.is_synchronized = False
                logger.warning(f"Order book WS error for {self.symbol}: {e}. Retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates instantaneous microstructure order book depth features."""
        if not self.bids or not self.asks:
            return {
                "synchronized": False,
                "best_bid": 0.0,
                "best_ask": 0.0,
                "mid_price": 0.0,
                "spread_abs": 0.0,
                "spread_pct": 0.05,
                "micro_price": 0.0,
                "imbalance_5": 0.5,
                "imbalance_10": 0.5,
                "imbalance_20": 0.5,
                "bid_depth_10_usdt": 0.0,
                "ask_depth_10_usdt": 0.0,
                "total_depth_ratio": 1.0
            }

        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])

        best_bid, best_bid_qty = sorted_bids[0]
        best_ask, best_ask_qty = sorted_asks[0]

        mid_price = (best_bid + best_ask) / 2.0
        spread_abs = max(0.0, best_ask - best_bid)
        spread_pct = (spread_abs / max(best_bid, 1e-8)) * 100.0

        # Volume-weighted Micro-price
        total_top_qty = best_bid_qty + best_ask_qty
        if total_top_qty > 0:
            micro_price = (best_bid * best_ask_qty + best_ask * best_bid_qty) / total_top_qty
        else:
            micro_price = mid_price

        def calc_depth_imbalance(k: int) -> float:
            bid_k = sum(q for _, q in sorted_bids[:k])
            ask_k = sum(q for _, q in sorted_asks[:k])
            tot = bid_k + ask_k
            if tot == 0:
                return 0.5
            return (bid_k / tot) # 0.0 (all ask) to 1.0 (all bid)

        bid_depth_10_usdt = sum(p * q for p, q in sorted_bids[:10])
        ask_depth_10_usdt = sum(p * q for p, q in sorted_asks[:10])
        tot_notional = bid_depth_10_usdt + ask_depth_10_usdt
        depth_ratio = (bid_depth_10_usdt / max(ask_depth_10_usdt, 1e-8)) if tot_notional > 0 else 1.0

        return {
            "synchronized": self.is_synchronized,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": round(mid_price, 6),
            "spread_abs": round(spread_abs, 6),
            "spread_pct": round(spread_pct, 4),
            "micro_price": round(micro_price, 6),
            "imbalance_5": round(calc_depth_imbalance(5), 4),
            "imbalance_10": round(calc_depth_imbalance(10), 4),
            "imbalance_20": round(calc_depth_imbalance(20), 4),
            "bid_depth_10_usdt": round(bid_depth_10_usdt, 2),
            "ask_depth_10_usdt": round(ask_depth_10_usdt, 2),
            "total_depth_ratio": round(depth_ratio, 3)
        }

class OrderBookSyncManager:
    """Manages active local order book synchronization for high-priority scalping pairs."""

    def __init__(self):
        self._books: Dict[str, SynchronizedOrderBook] = {}

    def get_or_create(self, symbol: str) -> SynchronizedOrderBook:
        clean = symbol.upper().replace("/", "")
        if clean not in self._books:
            book = SynchronizedOrderBook(clean)
            self._books[clean] = book
        return self._books[clean]

    async def ensure_active(self, symbols: List[str]):
        """Starts background local book sync for specified tokens."""
        for sym in symbols:
            book = self.get_or_create(sym)
            if not book._running:
                await book.start()

    def get_metrics(self, symbol: str) -> Dict[str, Any]:
        clean = symbol.upper().replace("/", "")
        book = self._books.get(clean)
        if not book:
            return {
                "synchronized": False,
                "best_bid": 0.0,
                "best_ask": 0.0,
                "mid_price": 0.0,
                "spread_abs": 0.0,
                "spread_pct": 0.04,
                "micro_price": 0.0,
                "imbalance_5": 0.5,
                "imbalance_10": 0.5,
                "imbalance_20": 0.5,
                "bid_depth_10_usdt": 0.0,
                "ask_depth_10_usdt": 0.0,
                "total_depth_ratio": 1.0
            }
        return book.get_metrics()

    async def stop_all(self):
        for book in self._books.values():
            await book.stop()
        self._books.clear()

order_book_manager = OrderBookSyncManager()
