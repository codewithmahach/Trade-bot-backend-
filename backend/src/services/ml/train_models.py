import os
import json
import time
import logging
import urllib.request
import numpy as np
import joblib
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", 
    "SUIUSDT", "AVAXUSDT", "NEARUSDT", "LINKUSDT", 
    "ADAUSDT", "XRPUSDT"
]

def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> List[List[Any]]:
    """Fetch 1m klines from Binance Futures REST API."""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except Exception as e:
        logger.warning(f"Error fetching klines for {symbol}: {e}")
        return []

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    ema = np.empty_like(prices, dtype=float)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

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

def calculate_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    typical = (highs + lows + closes) / 3.0
    cum_vol_price = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    cum_vol[cum_vol == 0] = 1e-9
    return cum_vol_price / cum_vol

def extract_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and asymmetric 120s scalp labels from real Binance klines."""
    x_10_list = []
    x_12_list = []
    y_list = []

    for sym in SYMBOLS:
        klines = fetch_binance_klines(sym, interval="1m", limit=1000)
        if len(klines) < 100:
            continue
        
        opens = np.array([float(k[1]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        closes = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])

        ema9 = calculate_ema(closes, 9)
        ema21 = calculate_ema(closes, 21)
        ema50 = calculate_ema(closes, 50)
        vwap = calculate_vwap(highs, lows, closes, volumes)
        rsi = calculate_rsi(closes, 14)

        # MACD (12, 26, 9)
        fast_ema = calculate_ema(closes, 12)
        slow_ema = calculate_ema(closes, 26)
        macd_line = fast_ema - slow_ema
        signal_line = calculate_ema(macd_line, 9)
        macd_hist = macd_line - signal_line

        # ATR 14
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_arr = np.zeros(len(closes))
        for idx in range(14, len(closes)):
            atr_arr[idx] = np.mean(tr[idx - 14:idx])

        # Rolling volume mean
        vol_sma20 = np.zeros(len(volumes))
        for idx in range(20, len(volumes)):
            vol_sma20[idx] = np.mean(volumes[idx - 20:idx])
        vol_sma20[vol_sma20 == 0] = 1.0

        # Bollinger %B
        bb_b = np.full(len(closes), 0.5)
        for idx in range(20, len(closes)):
            window = closes[idx - 20:idx]
            mid = np.mean(window)
            std = np.std(window)
            upper = mid + (2.0 * std)
            lower = mid - (2.0 * std)
            if upper - lower > 0:
                bb_b[idx] = (closes[idx] - lower) / (upper - lower)

        # We look ahead up to 3 candles (120 - 180 seconds scalp)
        # Minimum period for indicator warmth: 50
        for i in range(50, len(closes) - 4):
            price = closes[i]
            if price <= 0:
                continue

            # Lookahead outcomes
            future_high = np.max(highs[i + 1: i + 4])
            future_low = np.min(lows[i + 1: i + 4])

            up_return_pct = ((future_high - price) / price) * 100.0
            down_return_pct = ((price - future_low) / price) * 100.0

            # Target threshold for 20x scalping (User Rule: +0.25% profit move = +5.0% ROE):
            target_tp = 0.25
            stop_sl = 0.60

            # Class 2: BULLISH PROFIT (Hits +0.25% TP without falling into SL)
            if up_return_pct >= target_tp and down_return_pct < stop_sl:
                label = 2
            # Class 0: BEARISH PROFIT (Hits downside target without rising into SL)
            elif down_return_pct >= target_tp and up_return_pct < stop_sl:
                label = 0
            # Class 1: CHOPPY / FLAT
            else:
                label = 1

            # Telemetry Features
            f_ema_spread = ((ema9[i] - ema21[i]) / price) * 100.0
            f_trend_spread = ((ema21[i] - ema50[i]) / price) * 100.0
            f_vwap_dist = ((price - vwap[i]) / price) * 100.0
            f_rsi_norm = (rsi[i] - 50.0) / 25.0
            f_macd_norm = (macd_hist[i] / price) * 1000.0
            atr_pct = (atr_arr[i] / price) * 100.0
            f_atr = min(atr_pct, 3.5)
            rvol = volumes[i] / vol_sma20[i]
            f_rvol = min(rvol, 6.0)

            # Order book imbalance proxy: candle delta + volume surge
            candle_delta = (closes[i] - opens[i]) / (highs[i] - lows[i] + 1e-9)
            f_imbalance = np.clip(candle_delta * (rvol / 2.0), -1.0, 1.0)
            f_spread = 0.02
            f_bb_pct_b = float(bb_b[i])

            # MTF proxies
            mtf_3m_bull = 1.0 if (closes[i] > closes[i - 3] and ema9[i] > ema21[i]) else (-1.0 if closes[i] < closes[i - 3] else 0.0)
            mtf_5m_bull = 1.0 if (closes[i] > closes[i - 5] and ema21[i] > ema50[i]) else 0.0

            # 10-feature vector for HistGradientBoosting
            feat_10 = np.array([
                f_ema_spread,
                f_vwap_dist,
                f_rsi_norm,
                f_macd_norm,
                f_atr,
                f_rvol,
                f_imbalance,
                f_spread,
                f_bb_pct_b,
                1.0 if mtf_3m_bull > 0 else 0.0
            ], dtype=float)

            # 12-feature vector for MultiLayerNeuralNetwork
            feat_12 = np.array([
                f_ema_spread,
                f_trend_spread,
                f_vwap_dist,
                f_rsi_norm,
                f_macd_norm,
                f_atr,
                f_rvol,
                f_imbalance,
                f_spread,
                f_bb_pct_b,
                mtf_3m_bull,
                mtf_5m_bull
            ], dtype=float)

            x_10_list.append(feat_10)
            x_12_list.append(feat_12)
            y_list.append(label)

        logger.info(f"Processed {sym}: {len(klines)} klines")
        time.sleep(0.2) # Rate limit courtesy

    return np.array(x_10_list), np.array(x_12_list), np.array(y_list)

def train_and_save():
    logger.info("Starting historical market data extraction from Binance Futures...")
    X_10, X_12, y = extract_dataset()
    logger.info(f"Extracted {len(y)} samples. Class distribution: {np.bincount(y)}")

    # 1. Train HistGradientBoostingClassifier (Ultra-fast probability model)
    logger.info("Training HistGradientBoostingClassifier with balanced weights...")
    hgb = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42
    )
    hgb.fit(X_10, y)

    # 2. Train MultiLayerNeuralNetwork (MLP 64-32-16)
    logger.info("Scaling features and training Deep Neural Network (MLN)...")
    scaler = StandardScaler()
    X_12_scaled = scaler.fit_transform(X_12)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.005,
        batch_size=64,
        learning_rate_init=0.005,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12
    )
    mlp.fit(X_12_scaled, y)

    # Save models
    hgb_path = os.path.join(MODELS_DIR, "hist_gradient_boosting_scalp.joblib")
    mlp_path = os.path.join(MODELS_DIR, "mlp_neural_network_scalp.joblib")
    scaler_path = os.path.join(MODELS_DIR, "ml_scaler.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(hgb, hgb_path)
    joblib.dump(mlp, mlp_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "training_timestamp": time.time(),
        "total_samples": int(len(y)),
        "class_distribution": {
            "down_breakdowns_0": int(np.sum(y == 0)),
            "choppy_neutral_1": int(np.sum(y == 1)),
            "bullish_breakouts_2": int(np.sum(y == 2))
        },
        "symbols_used": SYMBOLS,
        "hgb_score": float(hgb.score(X_10, y)),
        "mlp_score": float(mlp.score(X_12_scaled, y))
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Models successfully trained and saved to {MODELS_DIR}")
    logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")

if __name__ == "__main__":
    train_and_save()
