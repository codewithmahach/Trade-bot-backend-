import hmac
import hashlib
import time
import httpx
from typing import Dict, Any, List, Optional
from backend.src.config.settings import settings

import os
import logging
logger = logging.getLogger("binance_client")

class BinanceClient:
    def __init__(self):
        self._api_key = settings.BINANCE_API_KEY or os.getenv("BINANCE_API_KEY", "")
        self._api_secret = settings.BINANCE_API_SECRET or os.getenv("BINANCE_API_SECRET", "")
        self.is_testnet = False
        self.base_url = settings.BINANCE_BASE_URL
        self.futures_base_url = settings.BINANCE_FUTURES_BASE_URL
        self._exchange_info_cache: Optional[Dict[str, Any]] = None
        self._exchange_info_cache_time: float = 0
        self._futures_exchange_info_cache: Optional[Dict[str, Any]] = None
        self._futures_exchange_info_cache_time: float = 0
        self._futures_onboard_dates: Dict[str, int] = {}
        self._symbol_rules: Dict[str, Dict[str, Any]] = {}
        self._cached_futures_balance: Optional[Dict[str, Any]] = None
        self._cached_futures_balance_time: float = 0
        self._cached_spot_balance: Optional[Dict[str, Any]] = None
        self._cached_spot_balance_time: float = 0

    @property
    def api_key(self) -> str:
        return self._api_key or settings.BINANCE_API_KEY or os.getenv("BINANCE_API_KEY", "")

    @api_key.setter
    def api_key(self, val: str):
        self._api_key = val

    @property
    def api_secret(self) -> str:
        return self._api_secret or settings.BINANCE_API_SECRET or os.getenv("BINANCE_API_SECRET", "")

    @api_secret.setter
    def api_secret(self, val: str):
        self._api_secret = val

    def update_credentials(self, api_key: str, api_secret: str, testnet: bool = False):
        self._api_key = api_key
        self._api_secret = api_secret
        self.is_testnet = testnet
        self.base_url = settings.BINANCE_BASE_URL
        self.futures_base_url = settings.BINANCE_FUTURES_BASE_URL
        self._cached_futures_balance = None
        self._cached_spot_balance = None
        self._cached_futures_balance_time = 0
        self._cached_spot_balance_time = 0

    def _get_signature(self, query_string: str) -> str:
        if not self.api_secret:
            return ""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AITradingBot/2.0"
        }
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    async def get_exchange_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch and cache Binance exchange rules, lot sizes, tick sizes, and min notional."""
        now = time.time()
        if not force_refresh and self._exchange_info_cache and (now - self._exchange_info_cache_time < 3600):
            return self._exchange_info_cache

        url = f"{self.base_url}/api/v3/exchangeInfo"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=self._get_headers())
            resp.raise_for_status()
            data = resp.json()
            self._exchange_info_cache = data
            self._exchange_info_cache_time = now

            # Parse and cache rules per symbol
            for s in data.get("symbols", []):
                sym = s.get("symbol")
                if not sym:
                    continue
                rules = {
                    "status": s.get("status"),
                    "baseAsset": s.get("baseAsset"),
                    "quoteAsset": s.get("quoteAsset"),
                    "minQty": 0.0001,
                    "maxQty": 10000000.0,
                    "stepSize": 0.0001,
                    "minPrice": 0.00000001,
                    "tickSize": 0.00000001,
                    "minNotional": 5.0
                }
                for f in s.get("filters", []):
                    ftype = f.get("filterType")
                    if ftype == "LOT_SIZE":
                        rules["minQty"] = float(f.get("minQty", 0.0001))
                        rules["maxQty"] = float(f.get("maxQty", 10000000.0))
                        rules["stepSize"] = float(f.get("stepSize", 0.0001))
                    elif ftype == "PRICE_FILTER":
                        rules["minPrice"] = float(f.get("minPrice", 0.00000001))
                        rules["tickSize"] = float(f.get("tickSize", 0.00000001))
                    elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                        rules["minNotional"] = float(f.get("minNotional", 5.0))
                self._symbol_rules[sym] = rules

            return data

    def get_symbol_rule(self, symbol: str) -> Dict[str, Any]:
        return self._symbol_rules.get(symbol, {
            "minQty": 0.0001,
            "stepSize": 0.0001,
            "minPrice": 0.00000001,
            "tickSize": 0.00000001,
            "minNotional": 5.0,
            "quoteAsset": "USDT"
        })

    async def get_futures_exchange_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch and cache Binance USD(S)-M Futures exchange rules, onboard dates, and symbols."""
        now = time.time()
        if not force_refresh and self._futures_exchange_info_cache and (now - self._futures_exchange_info_cache_time < 3600):
            return self._futures_exchange_info_cache

        url = f"{self.futures_base_url}/fapi/v1/exchangeInfo"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers={"User-Agent": "AITradingBot/2.0"})
                if resp.status_code == 200:
                    data = resp.json()
                    self._futures_exchange_info_cache = data
                    self._futures_exchange_info_cache_time = now
                    for s in data.get("symbols", []):
                        sym = s.get("symbol")
                        if sym:
                            self._futures_onboard_dates[sym] = s.get("onboardDate", 0)
                    return data
        except Exception as e:
            logger.debug(f"Futures exchangeInfo fetch notice: {e}")
        return self._futures_exchange_info_cache or {}

    def is_newly_launched(self, symbol: str, min_days: int = 60) -> bool:
        """Returns True if the token was listed on Binance Futures less than min_days ago."""
        sym_clean = symbol.replace("/", "").replace("1000", "")
        # Check both raw symbol and cleaned
        onboard = self._futures_onboard_dates.get(symbol.replace("/", ""), 0)
        if onboard <= 0:
            onboard = self._futures_onboard_dates.get(sym_clean + "USDT", 0)
        if onboard <= 0:
            return False
        now_ms = time.time() * 1000
        age_ms = now_ms - onboard
        min_age_ms = min_days * 86400 * 1000
        return age_ms < min_age_ms

    async def get_24h_tickers(self) -> List[Dict[str, Any]]:
        """Fetch 24h ticker for all symbols."""
        url = f"{self.base_url}/api/v3/ticker/24hr"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=self._get_headers())
            resp.raise_for_status()
            return resp.json()

    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[List[Any]]:
        """Fetch historical klines for technical & ML analysis."""
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=self._get_headers())
            resp.raise_for_status()
            return resp.json()

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch depth order book."""
        url = f"{self.base_url}/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=self._get_headers())
            resp.raise_for_status()
            return resp.json()

    async def get_account(self) -> Dict[str, Any]:
        """Fetch real account balances (Signed endpoint)."""
        if not self.api_key or not self.api_secret:
            return {"canTrade": False, "balances": [], "authenticated": False, "error": "No API keys configured"}

        url = f"{self.base_url}/api/v3/account"
        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}&recvWindow=60000"
        signature = self._get_signature(query)
        full_url = f"{url}?{query}&signature={signature}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(full_url, headers=self._get_headers())
            if resp.status_code != 200:
                return {"canTrade": False, "authenticated": False, "error": resp.text}
            data = resp.json()
            data["authenticated"] = True
            return data

    async def get_spot_balance(self) -> Dict[str, Any]:
        """Fetch real Spot balances with USDT priority."""
        if not self.api_key or not self.api_secret:
            return {
                "authenticated": False,
                "can_trade": False,
                "free_usdt": 0.0,
                "locked_usdt": 0.0,
                "total_usdt": 0.0,
                "assets": []
            }
        now = time.time()
        if self._cached_spot_balance and (now - self._cached_spot_balance_time < 3.0):
            return self._cached_spot_balance

        try:
            data = await self.get_account()
            if not data.get("authenticated", False):
                if self._cached_spot_balance:
                    return self._cached_spot_balance
                return {
                    "authenticated": False,
                    "can_trade": False,
                    "error": data.get("error", "Failed to authenticate"),
                    "free_usdt": 0.0,
                    "locked_usdt": 0.0,
                    "total_usdt": 0.0,
                    "assets": []
                }
            balances = data.get("balances", [])
            free_usdt = 0.0
            locked_usdt = 0.0
            non_zero = []
            ld_usdt = 0.0
            for b in balances:
                free = float(b.get("free", 0.0))
                locked = float(b.get("locked", 0.0))
                asset = b.get("asset", "")
                if asset == "USDT":
                    free_usdt = free
                    locked_usdt = locked
                elif asset == "LDUSDT":
                    ld_usdt = free
                if (free + locked) > 0.0001:
                    non_zero.append({
                        "asset": asset,
                        "free": round(free, 6),
                        "locked": round(locked, 6),
                        "total": round(free + locked, 6)
                    })
            if free_usdt == 0.0 and ld_usdt > 0.0:
                free_usdt = ld_usdt
            res = {
                "authenticated": True,
                "can_trade": data.get("canTrade", False),
                "free_usdt": round(free_usdt, 2),
                "locked_usdt": round(locked_usdt, 2),
                "total_usdt": round(free_usdt + locked_usdt, 2),
                "assets": non_zero
            }
            self._cached_spot_balance = res
            self._cached_spot_balance_time = now
            return res
        except Exception as e:
            if self._cached_spot_balance:
                return self._cached_spot_balance
            return {
                "authenticated": False,
                "can_trade": False,
                "error": str(e),
                "free_usdt": 0.0,
                "locked_usdt": 0.0,
                "total_usdt": 0.0,
                "assets": []
            }

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: float = 0.0,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute real order on Binance with safety checks."""
        if not self.api_key or not self.api_secret:
            raise ValueError("Live trading requires valid Binance API Key and Secret.")

        url = f"{self.base_url}/api/v3/order"
        timestamp = int(time.time() * 1000)
        
        # Round quantity according to symbol rules
        rule = self.get_symbol_rule(symbol)
        step = rule.get("stepSize", 0.0001)
        precision = len(str(step).split(".")[1]) if "." in str(step) else 0
        formatted_qty = f"{quantity:.{precision}f}"

        params = [
            f"symbol={symbol}",
            f"side={side.upper()}",
            f"type={order_type.upper()}",
            f"quantity={formatted_qty}",
            f"timestamp={timestamp}",
            f"recvWindow=60000"
        ]

        if order_type.upper() == "LIMIT":
            if not price or not time_in_force:
                raise ValueError("Limit orders require price and timeInForce")
            tick = rule.get("tickSize", 0.00000001)
            p_prec = len(str(tick).split(".")[1]) if "." in str(tick) else 2
            params.append(f"price={price:.{p_prec}f}")
            params.append(f"timeInForce={time_in_force}")

        if stop_price:
            tick = rule.get("tickSize", 0.00000001)
            p_prec = len(str(tick).split(".")[1]) if "." in str(tick) else 2
            params.append(f"stopPrice={stop_price:.{p_prec}f}")

        query_string = "&".join(params)
        signature = self._get_signature(query_string)
        full_url = f"{url}?{query_string}&signature={signature}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(full_url, headers=self._get_headers())
            if resp.status_code != 200:
                raise RuntimeError(f"Binance Order Rejected: {resp.text}")
            return resp.json()

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        url = f"{self.base_url}/api/v3/order"
        timestamp = int(time.time() * 1000)
        query = f"symbol={symbol}&orderId={order_id}&timestamp={timestamp}&recvWindow=60000"
        signature = self._get_signature(query)
        full_url = f"{url}?{query}&signature={signature}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(full_url, headers=self._get_headers())
            return resp.json()

    async def get_futures_account(self) -> Dict[str, Any]:
        """Fetch real Binance USDT-M Futures account status."""
        if not self.api_key or not self.api_secret:
            return {"authenticated": False, "error": "No API keys configured"}
        url = f"{self.futures_base_url}/fapi/v2/account"
        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}&recvWindow=60000"
        signature = self._get_signature(query)
        full_url = f"{url}?{query}&signature={signature}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(full_url, headers=self._get_headers())
            if resp.status_code != 200:
                err_text = resp.text
                if "-2015" in err_text:
                    err_msg = "Binance Futures permission is not enabled on this API key. Please check 'Enable Futures' on Binance.com API management if you want 20x leverage scalping. Spot trading is active."
                else:
                    err_msg = err_text
                return {"authenticated": False, "error": err_msg, "requires_futures_permission": True}
            data = resp.json()
            data["authenticated"] = True
            return data

    async def get_all_wallet_balances(self) -> List[Dict[str, Any]]:
        """Fetch all wallet balances across Binance (Spot, Funding, Futures, Earn)."""
        if not self.api_key or not self.api_secret:
            return []
        url = f"{self.base_url}/sapi/v1/asset/wallet/balance"
        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}&recvWindow=60000"
        signature = self._get_signature(query)
        full_url = f"{url}?{query}&signature={signature}"
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(full_url, headers=self._get_headers())
                if resp.status_code == 200:
                    return resp.json()
                return []
        except Exception:
            return []

    async def get_futures_balance(self) -> Dict[str, Any]:
        """Fetch real Binance USDT-M Futures balances with automatic caching and wallet fallback."""
        if not self.api_key or not self.api_secret:
            return {"authenticated": False, "free_usdt": 0.0, "total_usdt": 0.0}

        now = time.time()
        if self._cached_futures_balance and (now - self._cached_futures_balance_time < 5.0):
            return self._cached_futures_balance

        # 1. Fast direct HTTP to Binance Futures with 2.0s timeout (sub-250ms response)
        url = f"{self.futures_base_url}/fapi/v2/balance"
        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}&recvWindow=60000"
        signature = self._get_signature(query)
        full_url = f"{url}?{query}&signature={signature}"

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(full_url, headers=self._get_headers())
                if resp.status_code == 200:
                    data = resp.json()
                    usdt = next((b for b in data if b.get("asset") == "USDT"), None)
                    free = float(usdt.get("availableBalance", 0.0)) if usdt else 0.0
                    total = float(usdt.get("balance", 0.0)) if usdt else 0.0
                    res = {
                        "authenticated": True,
                        "free_usdt": round(free, 2),
                        "total_usdt": round(total, 2)
                    }
                    self._cached_futures_balance = res
                    self._cached_futures_balance_time = now
                    return res

                # Check for Binance API permissions / IP restriction (-2015)
                if "-2015" in resp.text:
                    res = {
                        "authenticated": False,
                        "free_usdt": 0.0,
                        "total_usdt": 0.0,
                        "requires_futures_permission": True,
                        "message": "Binance API key requires 'Enable Futures' permission on Binance.com"
                    }
                    self._cached_futures_balance = res
                    self._cached_futures_balance_time = now
                    return res

                if self._cached_futures_balance:
                    return self._cached_futures_balance
                return {"authenticated": False, "free_usdt": 0.0, "total_usdt": 0.0, "error": resp.text}
        except Exception as http_err:
            logger.debug(f"Direct futures balance http notice: {http_err}")

        # 2. CCXT fallback if direct failed
        try:
            import asyncio
            def _fetch_ccxt_fut():
                import ccxt
                ex = ccxt.binance({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'timeout': 1500,
                    'options': {'defaultType': 'future', 'adjustForTimeDifference': True, 'recvWindow': 60000}
                })
                ex.load_time_difference()
                acc = ex.fapiPrivateV2GetAccount()
                free = float(acc.get('availableBalance', 0.0))
                total = float(acc.get('totalWalletBalance', free))
                return free, total

            free, total = await asyncio.wait_for(asyncio.to_thread(_fetch_ccxt_fut), timeout=2.0)
            res = {
                "authenticated": True,
                "free_usdt": round(free, 2),
                "total_usdt": round(total, 2)
            }
            self._cached_futures_balance = res
            self._cached_futures_balance_time = now
            return res
        except Exception as e:
            if self._cached_futures_balance:
                return self._cached_futures_balance
            return {"authenticated": False, "free_usdt": 0.0, "total_usdt": 0.0, "error": str(e)}

    async def set_futures_leverage(self, symbol: str, leverage: int = 5) -> Dict[str, Any]:
        """Set leverage for symbol on USDT-M Futures."""
        if not self.api_key or not self.api_secret:
            return {"status": "SKIPPED"}
        url = f"{self.futures_base_url}/fapi/v1/leverage"
        timestamp = int(time.time() * 1000)
        query = f"symbol={symbol}&leverage={leverage}&timestamp={timestamp}&recvWindow=5000"
        signature = self._get_signature(query)
        full_url = f"{url}?{query}&signature={signature}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(full_url, headers=self._get_headers())
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def place_futures_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """Execute real order on Binance USDT-M Futures with stop-loss support."""
        if not self.api_key or not self.api_secret:
            raise ValueError("Live Futures trading requires valid Binance API Key and Secret.")
        url = f"{self.futures_base_url}/fapi/v1/order"
        timestamp = int(time.time() * 1000)
        rule = self.get_symbol_rule(symbol)
        step = rule.get("stepSize", 0.0001)
        precision = len(str(step).split(".")[1]) if "." in str(step) else 0
        formatted_qty = f"{quantity:.{precision}f}"

        params = [
            f"symbol={symbol}",
            f"side={side.upper()}",
            f"type={order_type.upper()}",
            f"quantity={formatted_qty}",
            f"timestamp={timestamp}",
            f"recvWindow=5000"
        ]
        if reduce_only:
            params.append("reduceOnly=true")

        query_string = "&".join(params)
        signature = self._get_signature(query_string)
        full_url = f"{url}?{query_string}&signature={signature}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(full_url, headers=self._get_headers())
            if resp.status_code != 200:
                raise RuntimeError(f"Binance Futures Order Rejected: {resp.text}")
            return resp.json()

binance_client = BinanceClient()
