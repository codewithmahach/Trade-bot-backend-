import os
import time
import httpx
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class DatasetBuilder:
    """Builds reproducible, chronologically segmented training and evaluation sets
    from Binance USDⓈ-M Futures market data.
    """

    FEATURE_NAMES = [
        "return_1m",
        "return_3m",
        "ema_9_21_spread",
        "ema_21_50_spread",
        "vwap_dist",
        "rsi_norm",
        "macd_hist_norm",
        "atr_pct",
        "bb_pct_b",
        "bb_width",
        "rvol",
        "high_low_ratio"
    ]

    @staticmethod
    def fetch_futures_klines(symbol: str = "NEARUSDT", limit: int = 1500) -> pd.DataFrame:
        """Fetches historical 1m klines from Binance USDⓈ-M Futures."""
        clean_sym = symbol.upper().replace("/", "")
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={clean_sym}&interval=1m&limit={limit}"
        try:
            resp = httpx.get(url, timeout=10.0)
            if resp.status_code == 200:
                raw = resp.json()
                df = pd.DataFrame(raw, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base_vol",
                    "taker_buy_quote_vol", "ignore"
                ])
                for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_vol"]:
                    df[col] = df[col].astype(float)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                return df
        except Exception as e:
            print(f"Error fetching futures klines for {symbol}: {e}")
        return pd.DataFrame()

    @classmethod
    def extract_features_and_labels(
        cls,
        df: pd.DataFrame,
        tp_pct: float = 0.35, # +0.35% favorable price move
        sl_pct: float = 0.30, # -0.30% adverse price move
        horizon_bars: int = 3   # 3 bars = 3 minutes
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if df.empty or len(df) < 60:
            return pd.DataFrame(), pd.Series()

        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]

        # 1. Micro-returns
        df["return_1m"] = c.pct_change(1) * 100.0
        df["return_3m"] = c.pct_change(3) * 100.0

        # 2. Moving averages
        df["ema_9"] = c.ewm(span=9, adjust=False).mean()
        df["ema_21"] = c.ewm(span=21, adjust=False).mean()
        df["ema_50"] = c.ewm(span=50, adjust=False).mean()
        df["ema_9_21_spread"] = ((df["ema_9"] - df["ema_21"]) / c) * 100.0
        df["ema_21_50_spread"] = ((df["ema_21"] - df["ema_50"]) / c) * 100.0

        # 3. Rolling VWAP proxy (20 bars)
        typical_p = (h + l + c) / 3.0
        cum_pv = (typical_p * v).rolling(20).sum()
        cum_v = v.rolling(20).sum()
        vwap = cum_pv / (cum_v + 1e-9)
        df["vwap_dist"] = ((c - vwap) / vwap) * 100.0

        # 4. RSI (14)
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df["rsi_norm"] = (rsi - 50.0) / 25.0

        # 5. MACD (12, 26, 9)
        fast_ema = c.ewm(span=12, adjust=False).mean()
        slow_ema = c.ewm(span=26, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist_norm"] = ((macd_line - signal_line) / c) * 1000.0

        # 6. ATR (14)
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df["atr_pct"] = (atr / c) * 100.0

        # 7. Bollinger Bands (20, 2.0)
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        upper_bb = sma20 + (2.0 * std20)
        lower_bb = sma20 - (2.0 * std20)
        df["bb_pct_b"] = (c - lower_bb) / ((upper_bb - lower_bb) + 1e-9)
        df["bb_width"] = ((upper_bb - lower_bb) / (sma20 + 1e-9)) * 100.0

        # 8. RVOL (20)
        vol_mean = v.rolling(20).mean()
        df["rvol"] = v / (vol_mean + 1e-9)
        df["high_low_ratio"] = (h - l) / (c + 1e-9) * 100.0

        # Triple-Barrier Labeling over horizon
        # 0: DOWN (adverse tick hits -sl_pct first)
        # 1: CHOP (neither barrier reached in horizon)
        # 2: UP (favorable tick hits +tp_pct first)
        labels = []
        for i in range(len(df)):
            if i + horizon_bars >= len(df):
                labels.append(np.nan)
                continue
            entry_p = c.iloc[i]
            future_window = df.iloc[i+1 : i+1+horizon_bars]

            max_future_h = future_window["high"].max()
            min_future_l = future_window["low"].min()

            up_move = ((max_future_h - entry_p) / entry_p) * 100.0
            down_move = ((min_future_l - entry_p) / entry_p) * 100.0

            if up_move >= tp_pct and down_move > -sl_pct:
                labels.append(2) # Clean UP
            elif down_move <= -sl_pct and up_move < tp_pct:
                labels.append(0) # DOWN / adverse stop
            elif up_move >= tp_pct and down_move <= -sl_pct:
                # Intrabar clash: assign based on candle closes
                if future_window["close"].iloc[-1] >= entry_p:
                    labels.append(2)
                else:
                    labels.append(0)
            else:
                labels.append(1) # CHOP

        df["target"] = labels

        # Drop NaN warmup rows
        valid = df.dropna(subset=cls.FEATURE_NAMES + ["target"]).copy()
        valid["target"] = valid["target"].astype(int)

        X = valid[cls.FEATURE_NAMES]
        y = valid["target"]

        return X, y
