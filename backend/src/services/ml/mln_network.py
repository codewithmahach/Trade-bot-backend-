import os
import joblib
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("mln_network")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")
TRAINED_MLP_PATH = os.path.join(MODELS_DIR, "mlp_neural_network_scalp.joblib")
TRAINED_SCALER_PATH = os.path.join(MODELS_DIR, "ml_scaler.joblib")

class MultiLayerNeuralNetwork:
    """Multi-Layer Neural Network (MLN / MLP) for Deep Scalping Pattern Recognition.
    
    Architecture:
      Input Layer: 12 normalized market telemetry features
      Hidden Layer 1: 64 Neurons (ReLU activation, Batch-normalized feature space)
      Hidden Layer 2: 32 Neurons (ReLU activation)
      Hidden Layer 3: 16 Neurons (ReLU activation)
      Output Layer: 3 Neurons (Softmax probabilities for DOWN, NEUTRAL, UP)
    """

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=32,
            learning_rate_init=0.01,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self._initialize_and_train_network()

    def _extract_mln_features(
        self,
        tech_1m: Dict[str, Any],
        tech_3m: Dict[str, Any],
        tech_5m: Dict[str, Any],
        order_book_metrics: Dict[str, Any],
        spread_pct: float
    ) -> np.ndarray:
        """Constructs 12-dimensional continuous feature tensor."""
        if not tech_1m:
            return np.zeros((1, 12))

        price = tech_1m.get("current_price", 1.0)
        ema9 = tech_1m.get("ema9", price)
        ema21 = tech_1m.get("ema21", price)
        ema50 = tech_1m.get("ema50", price)
        vwap = tech_1m.get("vwap", price)
        rsi = tech_1m.get("rsi", 50.0)
        macd_hist = tech_1m.get("macd_hist", 0.0)
        atr_pct = tech_1m.get("atr_pct", 0.2)
        rvol = tech_1m.get("relative_volume", 1.0)
        imbalance = order_book_metrics.get("imbalance", 0.5)

        # 1. EMA 9/21 ribbon spread %
        f_ema_spread = ((ema9 - ema21) / price) * 100.0
        # 2. EMA 21/50 trend ribbon %
        f_trend_spread = ((ema21 - ema50) / price) * 100.0
        # 3. Distance from VWAP %
        f_vwap_dist = ((price - vwap) / price) * 100.0
        # 4. Normalized RSI (centered around 50)
        f_rsi = (rsi - 50.0) / 25.0
        # 5. MACD histogram relative to price
        f_macd = (macd_hist / price) * 1000.0
        # 6. ATR volatility %
        f_atr = min(atr_pct, 4.0)
        # 7. Relative Volume multiplier
        f_rvol = min(rvol, 6.0)
        # 8. Order book depth imbalance (-1 to +1)
        f_imbalance = (imbalance - 0.5) * 2.0
        # 9. Bid-ask spread %
        f_spread = min(spread_pct, 0.5)
        # 10. Bollinger %B (position inside bands)
        f_bb_pct_b = tech_1m.get("bb_pct_b", 0.5)
        # 11. MTF 3m trend agreement (1.0 = Bullish, -1.0 = Bearish, 0.0 = Neutral)
        f_mtf_3m = 1.0 if (tech_3m and tech_3m.get("trend_bullish")) else (-1.0 if (tech_3m and tech_3m.get("trend_bearish")) else 0.0)
        # 12. MTF 5m structural regime (1.0 = Bullish, 0.0 = Neutral)
        f_mtf_5m = 1.0 if (tech_5m and tech_5m.get("trend_bullish")) else 0.0

        vec = np.array([
            f_ema_spread,
            f_trend_spread,
            f_vwap_dist,
            f_rsi,
            f_macd,
            f_atr,
            f_rvol,
            f_imbalance,
            f_spread,
            f_bb_pct_b,
            f_mtf_3m,
            f_mtf_5m
        ]).reshape(1, -1)

        return vec

    def _initialize_and_train_network(self):
        """Loads pre-trained Deep Neural Network (MLN) on real market klines if available, else initializes baseline."""
        if os.path.exists(TRAINED_MLP_PATH) and os.path.exists(TRAINED_SCALER_PATH):
            try:
                self.model = joblib.load(TRAINED_MLP_PATH)
                self.scaler = joblib.load(TRAINED_SCALER_PATH)
                self.is_trained = True
                logger.info(f"Loaded trained Deep Neural Network (MLN 64-32-16) and Scaler from {MODELS_DIR}")
                return
            except Exception as e:
                logger.warning(f"Failed to load trained MLN models from disk: {e}. Falling back to baseline.")

        np.random.seed(42)
        n_samples = 1200
        
        # 12-dimensional synthetic market feature space
        X_train = np.random.randn(n_samples, 12)
        
        # Ground-truth multi-layer non-linear scoring
        linear_combo = (
            X_train[:, 0] * 1.8 +   # EMA ribbon
            X_train[:, 1] * 1.2 +   # Trend ribbon
            X_train[:, 2] * 1.4 +   # VWAP push
            X_train[:, 3] * 1.1 +   # RSI momentum
            X_train[:, 6] * 1.5 +   # RVOL volume surge
            X_train[:, 7] * 2.0 +   # Orderbook buy wall
            X_train[:, 10] * 1.3    # MTF 3m confluence
        )
        
        # Add non-linear interaction terms (e.g. Volume surge * Orderbook imbalance)
        non_linear = linear_combo + (X_train[:, 6] * X_train[:, 7] * 0.5)
        
        y_train = np.ones(n_samples, dtype=int) # Default Class 1: NEUTRAL / NO_MOVE
        y_train[non_linear > 1.5] = 2   # Class 2: BULLISH BREAKOUT (UP)
        y_train[non_linear < -1.5] = 0  # Class 0: BEARISH BREAKDOWN (DOWN)

        # Scale features
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        
        # Train MLP Multi-Layer Network
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        logger.info("Multi-Layer Neural Network (MLN 64-32-16) initialized and trained with baseline distribution.")

    def predict_deep_probabilities(
        self,
        tech_1m: Dict[str, Any],
        tech_3m: Dict[str, Any],
        tech_5m: Dict[str, Any] = None,
        order_book_metrics: Dict[str, Any] = None,
        spread_pct: float = 0.01
    ) -> Dict[str, Any]:
        """Runs forward inference through the 3 hidden layers and returns softmax probabilities."""
        if not self.is_trained or not tech_1m:
            return {
                "mln_p_up": 0.33,
                "mln_p_down": 0.33,
                "mln_p_neutral": 0.34,
                "mln_confidence": 33.0,
                "mln_signal": "NEUTRAL"
            }

        order_book_metrics = order_book_metrics or {}
        raw_features = self._extract_mln_features(tech_1m, tech_3m, tech_5m, order_book_metrics, spread_pct)
        
        try:
            scaled_features = self.scaler.transform(raw_features)
            probs = self.model.predict_proba(scaled_features)[0]
            
            classes = self.model.classes_
            prob_dict = {cls: probs[idx] for idx, cls in enumerate(classes)}
            
            p_down = float(prob_dict.get(0, 0.1))
            p_neutral = float(prob_dict.get(1, 0.2))
            p_up = float(prob_dict.get(2, 0.7))
            
            total = p_down + p_neutral + p_up
            if total > 0:
                p_down = round(p_down / total, 3)
                p_neutral = round(p_neutral / total, 3)
                p_up = round(p_up / total, 3)

            confidence = round(p_up * 100.0, 1)
            
            if p_up >= 0.70:
                signal = "STRONG_BUY"
            elif p_up >= 0.55:
                signal = "LEAN_BUY"
            elif p_down >= 0.60:
                signal = "AVOID_BEARISH"
            else:
                signal = "NEUTRAL"

            return {
                "mln_p_up": p_up,
                "mln_p_down": p_down,
                "mln_p_neutral": p_neutral,
                "mln_confidence": confidence,
                "mln_signal": signal
            }
        except Exception as e:
            logger.warning(f"MLN inference error: {e}")
            return {
                "mln_p_up": 0.33,
                "mln_p_down": 0.33,
                "mln_p_neutral": 0.34,
                "mln_confidence": 33.0,
                "mln_signal": "NEUTRAL"
            }

mln_engine = MultiLayerNeuralNetwork()
