import numpy as np
from typing import Dict, Any, List, Tuple

class TechnicalAnalysisEngine:
    """High-speed vectorized technical indicator engine for crypto scalping."""

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        if len(prices) < period:
            return np.full_like(prices, np.nan)
        alpha = 2.0 / (period + 1.0)
        ema = np.empty_like(prices, dtype=float)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def calculate_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        typical_price = (highs + lows + closes) / 3.0
        cum_vol_price = np.cumsum(typical_price * volumes)
        cum_vol = np.cumsum(volumes)
        # Avoid division by zero
        cum_vol[cum_vol == 0] = 1e-9
        return cum_vol_price / cum_vol

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        if len(prices) <= period:
            return np.full_like(prices, 50.0)
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi = np.zeros(len(prices))
        rsi[:period] = 50.0

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            rsi[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fast_ema = TechnicalAnalysisEngine.calculate_ema(prices, fast)
        slow_ema = TechnicalAnalysisEngine.calculate_ema(prices, slow)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalAnalysisEngine.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_stoch_rsi(rsi_values: np.ndarray, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[float, float]:
        if len(rsi_values) < period + smooth_k + smooth_d:
            return 50.0, 50.0
        recent_rsi = rsi_values[-period:]
        min_rsi = np.min(recent_rsi)
        max_rsi = np.max(recent_rsi)
        if max_rsi - min_rsi == 0:
            stoch = 50.0
        else:
            stoch = ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100.0
        # Fast smoothing
        k = max(0.0, min(100.0, stoch))
        d = max(0.0, min(100.0, (stoch + 50.0) / 2.0))
        return k, d

    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < 2:
            return 0.0
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-period:]))

    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float, float, float]:
        if len(prices) < period:
            mid = float(prices[-1])
            return mid, mid * 1.01, mid * 0.99, 2.0, 0.5
        window = prices[-period:]
        middle = float(np.mean(window))
        std = float(np.std(window))
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        bandwidth = ((upper - lower) / middle) * 100.0 if middle > 0 else 0.0
        percent_b = (prices[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        return middle, upper, lower, bandwidth, percent_b

    @staticmethod
    def calculate_volume_metrics(volumes: np.ndarray, period: int = 20) -> Tuple[float, float, bool]:
        if len(volumes) < period:
            return float(volumes[-1]), 1.0, False
        # Exclude currently forming candle from baseline SMA
        vol_sma = float(np.mean(volumes[-period-1:-1])) if len(volumes) > period else float(np.mean(volumes[-period:]))
        # Benchmark volume: use last completed candle (full 60s) or current candle, whichever is stronger
        completed_vol = float(volumes[-2]) if len(volumes) >= 2 else float(volumes[-1])
        current_vol = float(volumes[-1])
        bench_vol = max(completed_vol, current_vol)
        rvol = bench_vol / vol_sma if vol_sma > 0 else 1.0
        spike = rvol > 1.8
        return vol_sma, rvol, spike

    @staticmethod
    def calculate_market_structure(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 20) -> Dict[str, Any]:
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        res = float(np.max(recent_highs))
        sup = float(np.min(recent_lows))
        curr = float(closes[-1])
        prev = float(closes[-2]) if len(closes) > 1 else curr

        breakout = curr > res and prev <= res
        breakdown = curr < sup and prev >= sup
        dist_to_res = ((res - curr) / curr) * 100.0 if curr > 0 else 0.0
        dist_to_sup = ((curr - sup) / curr) * 100.0 if curr > 0 else 0.0

        return {
            "resistance": res,
            "support": sup,
            "recent_high": res,
            "recent_low": sup,
            "breakout": breakout,
            "breakdown": breakdown,
            "dist_to_resistance_pct": dist_to_res,
            "dist_to_support_pct": dist_to_sup
        }

    def analyze_candles(self, klines: List[List[Any]]) -> Dict[str, Any]:
        """Runs complete technical analysis suite on candle history."""
        if not klines or len(klines) < 30:
            return {}

        # Binance kline format: [open_time, open, high, low, close, volume, ...]
        opens = np.array([float(k[1]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        closes = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])

        current_price = float(closes[-1])
        ema9 = self.calculate_ema(closes, 9)
        ema21 = self.calculate_ema(closes, 21)
        ema50 = self.calculate_ema(closes, 50)
        vwap = self.calculate_vwap(highs, lows, closes, volumes)
        rsi = self.calculate_rsi(closes, 14)
        macd_line, macd_sig, macd_hist = self.calculate_macd(closes)
        stoch_k, stoch_d = self.calculate_stoch_rsi(rsi)
        atr_val = self.calculate_atr(highs, lows, closes, 14)
        bb_mid, bb_up, bb_low, bb_width, bb_pct_b = self.calculate_bollinger_bands(closes, 20)
        vol_sma, rvol, vol_spike = self.calculate_volume_metrics(volumes, 20)
        structure = self.calculate_market_structure(highs, lows, closes, 20)

        # Trend alignment
        trend_bullish = ema9[-1] > ema21[-1] > ema50[-1] and current_price > vwap[-1]
        trend_bearish = ema9[-1] < ema21[-1] < ema50[-1] and current_price < vwap[-1]

        return {
            "current_price": current_price,
            "ema9": float(ema9[-1]),
            "ema21": float(ema21[-1]),
            "ema50": float(ema50[-1]),
            "vwap": float(vwap[-1]),
            "rsi": float(rsi[-1]),
            "macd": float(macd_line[-1]),
            "macd_signal": float(macd_sig[-1]),
            "macd_hist": float(macd_hist[-1]),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "atr": atr_val,
            "atr_pct": (atr_val / current_price) * 100.0 if current_price > 0 else 0.0,
            "bb_middle": bb_mid,
            "bb_upper": bb_up,
            "bb_lower": bb_low,
            "bb_bandwidth": bb_width,
            "bb_pct_b": bb_pct_b,
            "volume_sma": vol_sma,
            "relative_volume": rvol,
            "volume_spike": vol_spike,
            "structure": structure,
            "trend_bullish": trend_bullish,
            "trend_bearish": trend_bearish
        }

technical_engine = TechnicalAnalysisEngine()
