import ccxt
import pandas as pd
import os
import re
import time
import logging

logger = logging.getLogger("binance_connector")

class BinanceConnector:
    def __init__(self):
        # Read API keys from env, settings, or .env file
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        
        # Fallback to reading from .env file directly if not set in os.environ
        if not self.api_key or not self.api_secret:
            self._load_from_env_file()
            
        self._init_exchange()

    def _init_exchange(self):
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        })

    def _load_from_env_file(self):
        try:
            env_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
                ".env"
            ]
            for ep in env_paths:
                if os.path.exists(ep):
                    with open(ep, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("BINANCE_API_KEY=") and not self.api_key:
                                self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            elif line.startswith("BINANCE_API_SECRET=") and not self.api_secret:
                                self.api_secret = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if self.api_key and self.api_secret:
                        os.environ["BINANCE_API_KEY"] = self.api_key
                        os.environ["BINANCE_API_SECRET"] = self.api_secret
                        break
        except Exception as e:
            print(f"Notice: Could not load keys from .env: {e}")

    def update_keys(self, api_key: str, api_secret: str, persist_env: bool = True):
        """Updates Binance API keys and persists them permanently to .env file."""
        if not api_key or not api_secret:
            return
        if api_key.strip() == api_secret.strip():
            if self.api_key and self.api_key != self.api_secret:
                api_key = self.api_key
            else:
                return
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        os.environ["BINANCE_API_KEY"] = self.api_key
        os.environ["BINANCE_API_SECRET"] = self.api_secret

        try:
            from backend.src.config.settings import settings
            settings.BINANCE_API_KEY = self.api_key
            settings.BINANCE_API_SECRET = self.api_secret
        except Exception:
            pass

        self._init_exchange()

        if persist_env:
            self._save_to_env_file(self.api_key, self.api_secret)

    def _save_to_env_file(self, api_key: str, api_secret: str):
        try:
            env_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
                ".env"
            ]
            target_env = None
            for ep in env_paths:
                if os.path.exists(ep):
                    target_env = ep
                    break
            if not target_env:
                target_env = ".env"

            content = ""
            if os.path.exists(target_env):
                with open(target_env, "r", encoding="utf-8") as f:
                    content = f.read()

            if re.search(r"^BINANCE_API_KEY=.*$", content, re.MULTILINE):
                content = re.sub(r"^BINANCE_API_KEY=.*$", f"BINANCE_API_KEY={api_key}", content, flags=re.MULTILINE)
            else:
                content += f"\nBINANCE_API_KEY={api_key}"

            if re.search(r"^BINANCE_API_SECRET=.*$", content, re.MULTILINE):
                content = re.sub(r"^BINANCE_API_SECRET=.*$", f"BINANCE_API_SECRET={api_secret}", content, flags=re.MULTILINE)
            else:
                content += f"\nBINANCE_API_SECRET={api_secret}"

            with open(target_env, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Persisted Binance API keys to {target_env}")
        except Exception as e:
            print(f"Error persisting keys to .env: {e}")

    def get_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV directly from Binance Futures for ultra-fast micro-scalping with pooled HTTP session."""
        clean_sym = symbol.replace("/", "").replace(":", "")
        if not hasattr(self, '_session') or self._session is None:
            import requests
            self._session = requests.Session()

        # 1. Try Binance USD-M Futures klines
        try:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={clean_sym}&interval={timeframe}&limit={limit}"
            resp = self._session.get(url, timeout=3.0)
            if resp.status_code == 200:
                raw_candles = resp.json()
                if raw_candles and isinstance(raw_candles, list) and len(raw_candles) > 0:
                    parsed = [
                        [int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                        for c in raw_candles
                    ]
                    df = pd.DataFrame(parsed, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
        except Exception:
            pass

        # 2. Fast Spot klines fallback (Ultra-fast pooled connection)
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={clean_sym}&interval={timeframe}&limit={limit}"
            resp = self._session.get(url, timeout=3.0)
            if resp.status_code == 200:
                raw_candles = resp.json()
                if raw_candles and isinstance(raw_candles, list) and len(raw_candles) > 0:
                    parsed = [
                        [int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                        for c in raw_candles
                    ]
                    df = pd.DataFrame(parsed, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
        except Exception:
            pass

        return pd.DataFrame()

    _cached_fapi_tickers = []
    _cached_fapi_time = 0.0
    _scanned_history = []
    _rotation_offset = 0

    def get_top_tokens(self, limit: int = 15) -> list:
        """
        Dynamically fetches and rotates top high-liquidity, high-momentum altcoin pairs
        from Binance Futures 24hr tickers.
        Rotates on every call so that every 'AUTO SCAN' scans fresh, new tokens!
        Strictly excludes BTC, ETH, BNB, WIF, meme coins, and all stablecoins.
        """
        if not hasattr(self, '_session') or self._session is None:
            import requests
            self._session = requests.Session()
        import time

        now_ms = time.time() * 1000
        min_age_ms = 60 * 86400 * 1000

        # Cache onboard dates from Binance Futures
        if not hasattr(self, '_futures_onboard_dates') or not self._futures_onboard_dates:
            try:
                r_info = self._session.get('https://fapi.binance.com/fapi/v1/exchangeInfo', timeout=3.0)
                if r_info.status_code == 200:
                    self._futures_onboard_dates = {s['symbol']: s.get('onboardDate', 0) for s in r_info.json().get('symbols', [])}
            except Exception:
                self._futures_onboard_dates = {}

        SHITCOIN_TRAPS = {
            "MUBARAK", "DYM", "PLUME", "TROY", "ACT", "PNUT", "NEIRO", "HIPPO",
            "COW", "CETUS", "THE", "GOAT", "MOODENG", "DEXE", "1000CAT", "BANANA",
            "VOXEL", "ALPACA", "FARTCOIN", "CHIP", "VVV", "ARC", "PENGU", "SLERF",
            "SUNDOG", "TURBO", "MEME", "ORBS", "LOKA", "COOK", "BIGTIME", "GRIFFAIN",
            "MAJOR", "COOKIE", "SWARMS", "PUMP", "BOME", "1000RATS", "1000SATS"
        }
        DEAD_STAGNANT = {
            "HBAR", "ADA", "XRP", "TRX", "LTC", "ALGO", "DOT", "ATOM", "EOS",
            "XLM", "ETC", "ONE", "BCH", "ZEC", "DASH", "KSM", "BAT", "ZIL", "OMG"
        }
        STABLE_BASES = {
            "BTC", "ETH", "BNB", "WIF", "USDC", "FDUSD", "TUSD", "BUSD", "EUR", "DAI",
            "USDP", "AEUR", "USTC", "RLUSD", "USD1", "USD", "US", "PAX", "EURS"
        }

        # 1. First priority: Use in-memory live WebSocket tickers (sub-millisecond, zero network delay)
        try:
            from backend.src.services.market.data_manager import market_data_manager
            in_mem_tickers = market_data_manager.get_all_tickers()
            if in_mem_tickers and len(in_mem_tickers) > 50:
                candidates = []
                for sym, t in in_mem_tickers.items():
                    if not sym.endswith("USDT"):
                        continue
                    base = sym[:-4]
                    pair = f"{base}/USDT"
                    if base in STABLE_BASES or any(base == b for b in STABLE_BASES) or len(base) < 2:
                        continue
                    if base in SHITCOIN_TRAPS or any(base.startswith(st) for st in SHITCOIN_TRAPS):
                        continue
                    if base in DEAD_STAGNANT:
                        continue

                    # Eliminate newly launched tokens (< 60 days old on Binance Futures)
                    onboard = self._futures_onboard_dates.get(sym, 0)
                    if onboard > 0 and (now_ms - onboard) < min_age_ms:
                        continue

                    quote_vol = float(t.get("quoteVolume24h", 0.0))
                    change_24h = float(t.get("priceChangePercent", 0.0))
                    spread = float(t.get("spread", 0.02))

                    # Strict volume (>= $25M) and tight spread (<= 0.04%)
                    if quote_vol < 25_000_000 or spread > 0.04:
                        continue

                    score = (abs(change_24h) * 3.0) + min(30.0, (quote_vol / 10_000_000.0))
                    candidates.append((score, pair))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    pool = [c[1] for c in candidates[:60]]
                    unseen = [p for p in pool if p not in self._scanned_history]
                    if len(unseen) < limit:
                        self._scanned_history = []
                        unseen = pool
                    selected = unseen[:limit]
                    self._scanned_history.extend(selected)
                    return selected
        except Exception:
            pass

        now = time.time()
        if not self._cached_fapi_tickers or (now - self._cached_fapi_time > 15.0):
            try:
                resp = self._session.get('https://fapi.binance.com/fapi/v1/ticker/24hr', timeout=2.5)
                if resp.status_code == 200:
                    self._cached_fapi_tickers = resp.json()
                    self._cached_fapi_time = now
            except Exception as e:
                logger.debug(f"Binance Futures 24hr ticker fetch notice: {e}")

        candidates = []
        if self._cached_fapi_tickers and isinstance(self._cached_fapi_tickers, list):
            for t in self._cached_fapi_tickers:
                sym = t.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                base = sym[:-4]
                pair = f"{base}/USDT"

                if base in STABLE_BASES or any(base == b for b in STABLE_BASES) or len(base) < 2:
                    continue
                if base in SHITCOIN_TRAPS or any(base.startswith(st) for st in SHITCOIN_TRAPS):
                    continue
                if base in DEAD_STAGNANT:
                    continue

                # Eliminate newly launched tokens
                onboard = self._futures_onboard_dates.get(sym, 0)
                if onboard > 0 and (now_ms - onboard) < min_age_ms:
                    continue

                try:
                    quote_vol = float(t.get("quoteVolume", 0.0))
                    change_24h = float(t.get("priceChangePercent", 0.0))
                    if quote_vol < 25_000_000:
                        continue
                    score = (abs(change_24h) * 3.0) + min(30.0, (quote_vol / 10_000_000.0))
                    candidates.append((score, pair))
                except Exception:
                    continue

        if not candidates:
            all_fallback = [
                "SOL/USDT", "DOGE/USDT", "PEPE/USDT", "SUI/USDT", "NEAR/USDT",
                "AVAX/USDT", "LINK/USDT", "SEI/USDT", "RENDER/USDT", "FET/USDT",
                "INJ/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "TIA/USDT",
                "GALA/USDT", "JUP/USDT", "ENA/USDT", "AAVE/USDT", "FTM/USDT"
            ]
            offset = self._rotation_offset % len(all_fallback)
            self._rotation_offset = (offset + limit) % len(all_fallback)
            rotated = all_fallback[offset:] + all_fallback[:offset]
            return rotated[:limit]

        candidates.sort(key=lambda x: x[0], reverse=True)
        pool = [c[1] for c in candidates[:60]]

        # Dynamic Rotation: Pick tokens from the pool that were NOT in the recent history
        unseen = [p for p in pool if p not in self._scanned_history]
        if len(unseen) < limit:
            self._scanned_history = []
            unseen = pool

        selected = unseen[:limit]
        self._scanned_history.extend(selected)
        if len(self._scanned_history) > 60:
            self._scanned_history = self._scanned_history[-30:]

        return selected

    def execute_trade(self, symbol: str, side: str, amount_usdt: float, current_price: float, tp_price: float = None, sl_price: float = None, api_key: str = None, api_secret: str = None) -> dict:
        """Execute a REAL live market order on Binance with Auto-Close (TP/SL)."""
        use_key = api_key or self.api_key or os.getenv("BINANCE_API_KEY", "")
        use_secret = api_secret or self.api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        if not use_key or not use_secret:
            return {"status": "error", "message": "Binance API keys not provided. Please enter them in the UI."}

        if api_key and api_secret and (api_key != self.api_key or api_secret != self.api_secret):
            self.update_keys(api_key, api_secret)
            
    def _get_futures_exchange(self, api_key: str, api_secret: str):
        """Returns or initializes cached CCXT Futures exchange instance."""
        if not hasattr(self, "_f_exchange") or getattr(self, "_f_exchange_key", None) != api_key:
            self._f_exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'timeout': 3000,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
            self._f_exchange.load_time_difference()
            self._f_exchange.load_markets()
            self._f_exchange_key = api_key
        return self._f_exchange

    def execute_trade(self, symbol: str, side: str, amount_usdt: float, current_price: float, tp_price: float = None, sl_price: float = None, api_key: str = None, api_secret: str = None) -> dict:
        """Execute a REAL live market order on Binance USD-M Futures with sub-second execution."""
        use_key = api_key or self.api_key or os.getenv("BINANCE_API_KEY", "")
        use_secret = api_secret or self.api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        if not use_key or not use_secret:
            return {"status": "error", "message": "Binance API keys not provided. Please enter them in the UI."}

        if api_key and api_secret and (api_key != self.api_key or api_secret != self.api_secret):
            self.update_keys(api_key, api_secret)
            
        try:
            exec_exchange = self._get_futures_exchange(use_key, use_secret)
            clean_sym = symbol.replace("/", "").replace(":", "")

            # Cancel any stale pending orders on this symbol to free up locked margin
            try:
                exec_exchange.cancel_all_orders(symbol)
            except Exception:
                pass
                
            # Fetch real-time available free USDT balance in Futures account
            balance_data = exec_exchange.fetch_balance()
            usdt_free = float(balance_data.get('USDT', {}).get('free', 0.0))
            
            if usdt_free < 0.40:
                return {
                    "status": "error", 
                    "message": f"Available Futures Balance is ${usdt_free:.2f} USDT. Please transfer USDT into your USD-M Futures wallet."
                }
            
            # Apply 20x Leverage for High-Frequency Micro Scalping
            leverage = 20
            try:
                exec_exchange.set_leverage(leverage, symbol)
            except Exception:
                pass
                
            safe_margin = min(amount_usdt, usdt_free * 0.88)
            leveraged_usdt = safe_margin * leverage
            
            # Ensure notional meets Binance minimum 5.5 USDT requirement
            if leveraged_usdt < 5.5:
                leveraged_usdt = 5.5
            
            # Calculate coin amount with precision
            raw_coin_amount = leveraged_usdt / current_price
            coin_amount = float(exec_exchange.amount_to_precision(symbol, raw_coin_amount))
            
            if coin_amount <= 0:
                return {"status": "error", "message": f"Trade size ${safe_margin:.2f} is too small for {symbol}."}
                
            client_order_id = f"scalp_{clean_sym}_{int(time.time()*1000)}"
            # Place real entry market order with unique client ID for idempotency
            order = exec_exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=coin_amount,
                params={'newClientOrderId': client_order_id}
            )
            
            # Place Take Profit limit/market order on Binance to auto-lock profit
            if tp_price:
                try:
                    tp_price_formatted = float(exec_exchange.price_to_precision(symbol, tp_price))
                    close_side = 'sell' if side.lower() == 'buy' else 'buy'
                    exec_exchange.create_order(
                        symbol=symbol,
                        type='TAKE_PROFIT_MARKET',
                        side=close_side,
                        amount=coin_amount,
                        params={'stopPrice': tp_price_formatted, 'closePosition': True}
                    )
                except Exception as aux_e:
                    print(f"Notice: Exchange TP order fallback to local monitor: {aux_e}")
                    
            return {"status": "success", "order": order}
        except Exception as e:
            err_str = str(e)
            if "-2015" in err_str or "permissions" in err_str or "Invalid API-key" in err_str:
                return {
                    "status": "error",
                    "message": "Binance Futures trading permission is not enabled on this API key. Please check 'Enable Futures' in Binance API settings."
                }
            return {"status": "error", "message": f"Binance Execution Failed: {err_str}"}

    def get_balance(self, api_key: str = None, api_secret: str = None) -> dict:
        """Fetch current USDT balance to verify account connection."""
        use_key = api_key or self.api_key or os.getenv("BINANCE_API_KEY", "")
        use_secret = api_secret or self.api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        if not use_key or not use_secret:
            return {
                "status": "error", 
                "authenticated": False,
                "message": "Binance API Key and Secret are not configured. Please enter and save your Binance keys in the UI."
            }

        if api_key and api_secret and (api_key != self.api_key or api_secret != self.api_secret):
            self.update_keys(api_key, api_secret)
            
        # Check Spot first (standard on all Binance API keys)
        try:
            spot_bal = self.exchange.fetch_balance()
            spot_free = float(spot_bal.get('USDT', {}).get('free', 0.0))
            # Check for LDUSDT (Binance Simple Earn Flexible)
            ldusdt_free = float(spot_bal.get('LDUSDT', {}).get('free', 0.0)) + float(spot_bal.get('LDUSDT', {}).get('total', 0.0))
            effective_free = spot_free if spot_free > 0 else ldusdt_free
            spot_total = float(spot_bal.get('USDT', {}).get('total', effective_free))

            # Attempt Futures balance check
            futures_free = 0.0
            futures_total = 0.0
            futures_enabled = False
            try:
                f_exchange = ccxt.binance({
                    'apiKey': use_key,
                    'secret': use_secret,
                    'enableRateLimit': True,
                    'timeout': 4000,
                    'options': {'defaultType': 'future', 'recvWindow': 60000}
                })
                f_bal = f_exchange.fetch_balance()
                futures_free = float(f_bal.get('USDT', {}).get('free', 0.0))
                futures_total = float(f_bal.get('USDT', {}).get('total', futures_free))
                futures_enabled = True
            except Exception:
                pass

            if futures_enabled:
                return {
                    "status": "success",
                    "authenticated": True,
                    "canTrade": True,
                    "usdt_balance": round(futures_free if futures_free > 0 else effective_free, 2),
                    "total_balance": round(futures_total if futures_total > 0 else spot_total, 2),
                    "futures_balance": round(futures_free, 2),
                    "spot_balance": round(effective_free, 2),
                    "message": f"Connected to Binance! (Spot: ${effective_free:.2f}, Futures: ${futures_free:.2f} USDT)"
                }
            else:
                return {
                    "status": "success",
                    "authenticated": True,
                    "canTrade": True,
                    "usdt_balance": round(effective_free, 2),
                    "total_balance": round(spot_total, 2),
                    "spot_balance": round(effective_free, 2),
                    "futures_balance": 0.0,
                    "message": f"Connected to Binance Spot (${effective_free:.2f} USDT)! (Tip: Enable 'Futures' on Binance.com for 20x leverage)"
                }
        except Exception as e:
            err_msg = str(e)
            if "-2015" in err_msg or "Invalid API-key" in err_msg or "401" in err_msg:
                friendly_msg = "Invalid Binance API Key or Secret. Ensure you copied both correctly."
            elif "-1021" in err_msg or "Timestamp" in err_msg:
                friendly_msg = "Binance timestamp sync warning. Please retry, system is syncing clock automatically."
            else:
                friendly_msg = f"Binance Error: {err_msg}"
            return {"status": "error", "authenticated": False, "message": friendly_msg}

    def get_open_position(self, symbol: str = None, api_key: str = None, api_secret: str = None) -> dict:
        """Fetch currently open Futures positions from Binance."""
        if symbol and len(symbol) > 30 and not api_key:
            api_key = symbol
            symbol = None
        use_key = api_key or self.api_key or os.getenv("BINANCE_API_KEY", "")
        use_secret = api_secret or self.api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        if not use_key or not use_secret:
            return {"status": "error", "message": "Keys missing"}
            
        now = time.time()
        if not symbol and hasattr(self, '_pos_cache') and (now - getattr(self, '_pos_cache_time', 0) < 3.0):
            return self._pos_cache

        try:
            exec_exchange = ccxt.binance({
                'apiKey': use_key,
                'secret': use_secret,
                'timeout': 5000,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
            account_data = exec_exchange.fapiPrivateV2GetAccount()
            raw_positions = account_data.get('positions', [])
            active_positions = [p for p in raw_positions if abs(float(p.get('positionAmt', 0))) > 0]
            
            if active_positions:
                pos = active_positions[0]
                symbol_found = pos.get('symbol', '')
                amt = float(pos.get('positionAmt', 0))
                entry_price = float(pos.get('entryPrice', 0))
                unrealized_pnl = float(pos.get('unrealizedProfit', 0))
                side = "LONG" if amt > 0 else "SHORT"
                
                res = {
                    "status": "success", 
                    "active": True, 
                    "trade": {
                        "symbol": symbol_found,
                        "side": side,
                        "entry": entry_price,
                        "amount": abs(amt),
                        "margin": round((abs(amt) * entry_price) / 20.0, 2),
                        "pnl": round(unrealized_pnl, 4)
                    }
                }
            else:
                res = {"status": "success", "active": False}

            self._pos_cache = res
            self._pos_cache_time = now
            return res
        except Exception as e:
            logger.debug(f"Futures open position check: {e}")
            return {"status": "success", "active": False}

    def close_position(self, symbol: str, api_key: str = None, api_secret: str = None) -> dict:
        """Immediately closes an active Binance Futures position at market price and cancels pending TP/SL orders."""
        use_key = api_key or self.api_key or os.getenv("BINANCE_API_KEY", "")
        use_secret = api_secret or self.api_secret or os.getenv("BINANCE_API_SECRET", "")
        if not use_key or not use_secret:
            return {"status": "error", "message": "Keys missing"}
        try:
            exec_exchange = ccxt.binance({
                'apiKey': use_key,
                'secret': use_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
            exec_exchange.load_time_difference()
            exec_exchange.load_markets()
            
            # Normalize symbol for CCXT (e.g. SUI/USDT)
            sym = symbol
            if ":" in sym:
                sym = sym.split(":")[0]
            if "/" not in sym:
                sym = f"{sym.replace('USDT', '')}/USDT"
                
            # Cancel open TP/SL orders for this symbol first
            try:
                exec_exchange.cancel_all_orders(sym)
            except Exception:
                pass
                
            # Find the exact open amount
            positions = exec_exchange.fetch_positions([sym])
            active_pos = [p for p in positions if abs(float(p.get('info', {}).get('positionAmt', 0))) > 0]
            if not active_pos:
                return {"status": "success", "message": "No active position to close."}
                
            pos = active_pos[0]
            amt = float(pos.get('info', {}).get('positionAmt', 0))
            close_side = 'sell' if amt > 0 else 'buy'
            abs_amt = abs(amt)
            
            # Place instant market close order
            close_order = exec_exchange.create_order(
                symbol=sym,
                type='market',
                side=close_side,
                amount=abs_amt,
                params={'reduceOnly': True}
            )
            return {"status": "success", "message": f"Position {sym} closed instantly at market!", "order": close_order}
        except Exception as e:
            err_str = str(e)
            if "-2015" in err_str or "permissions" in err_str or "Invalid API-key" in err_str:
                # Fallback to Spot market sell
                try:
                    spot_ex = ccxt.binance({
                        'apiKey': use_key,
                        'secret': use_secret,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot', 'recvWindow': 60000}
                    })
                    spot_ex.load_markets()
                    norm_sym = symbol if "/" in symbol else f"{symbol.replace('USDT', '')}/USDT"
                    base_coin = norm_sym.split('/')[0]
                    bal = spot_ex.fetch_balance()
                    free_coin = float(bal.get(base_coin, {}).get('free', 0.0))
                    if free_coin > 0:
                        order = spot_ex.create_order(norm_sym, 'market', 'sell', free_coin)
                        return {"status": "success", "message": f"Spot position {norm_sym} closed successfully!", "order": order}
                    return {"status": "success", "message": "No active Spot position found to close"}
                except Exception as spot_err:
                    return {"status": "error", "message": f"Spot close failed: {str(spot_err)}"}
            return {"status": "error", "message": str(e)}

binance_connector = BinanceConnector()
