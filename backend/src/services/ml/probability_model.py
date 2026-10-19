import os
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger("ml_probability_model")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")

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

class MLProbabilityModel:
    """Supervised machine learning model predicting calibrated short-term price movement probabilities.
    Outputs: P(up), P(down), P(no_move), and Market Regime.
    """

    def __init__(self):
        self.model = None
        self.model_name = "None"
        self.is_fitted = False
        self._load_production_model()

    def _load_production_model(self):
        """Attempts to load trained calibrated models in priority order."""
        candidate_files = [
            ("production_scalp_model.joblib", "Production Calibrated Model"),
            ("calibrated_lgbm.joblib", "Calibrated LightGBM"),
            ("calibrated_xgboost.joblib", "Calibrated XGBoost"),
            ("hist_gradient_boosting_scalp.joblib", "HistGradientBoosting")
        ]

        for fname, name in candidate_files:
            fpath = os.path.join(MODELS_DIR, fname)
            if os.path.exists(fpath):
                try:
                    self.model = joblib.load(fpath)
                    self.model_name = name
                    self.is_fitted = True
                    logger.info(f"Loaded trained ML model: {name} from {fpath}")
                    return
                except Exception as e:
                    logger.warning(f"Failed loading {fname}: {e}")

        # Fallback if no model file exists
        logger.warning("No pre-trained model file found. Initializing fast HistGradientBoosting fallback.")
        self.model = HistGradientBoostingClassifier(
            max_iter=50,
            learning_rate=0.08,
            max_leaf_nodes=15,
            random_state=42
        )
        self.model_name = "Baseline HGB"
        X_dummy = np.zeros((10, len(FEATURE_NAMES)))
        y_dummy = np.array([0, 1, 2, 1, 0, 2, 1, 1, 0, 2])
        self.model.fit(X_dummy, y_dummy)
        self.is_fitted = True

    def _extract_features(
        self,
        tech_1m: Dict[str, Any],
        tech_3m: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """Extract standardized 11-dimensional feature vector matching dataset_builder."""
        price = float(tech_1m.get("current_price") or 1.0)
        ema9 = float(tech_1m.get("ema9") or price)
        ema21 = float(tech_1m.get("ema21") or price)
        ema50 = float(tech_1m.get("ema50") or ema21)
        vwap = float(tech_1m.get("vwap") or price)
        rsi = float(tech_1m.get("rsi") or 50.0)
        macd_hist = float(tech_1m.get("macd_hist") or 0.0)
        atr_pct = float(tech_1m.get("atr_pct") or 0.2)
        rvol = float(tech_1m.get("relative_volume") or 1.0)
        bb_pct_b = float(tech_1m.get("bb_pct_b") or 0.5)
        bb_upper = float(tech_1m.get("bb_upper") or (price * 1.002))
        bb_lower = float(tech_1m.get("bb_lower") or (price * 0.998))
        bb_width = ((bb_upper - bb_lower) / (price + 1e-9)) * 100.0

        # return_1m and return_3m
        p_1m_open = float(tech_1m.get("open") or price)
        ret_1m = ((price - p_1m_open) / (p_1m_open + 1e-9)) * 100.0

        ret_3m = ret_1m
        if tech_3m:
            p_3m_open = float(tech_3m.get("open") or price)
            ret_3m = ((price - p_3m_open) / (p_3m_open + 1e-9)) * 100.0

        # High/low ratio
        high = float(tech_1m.get("high") or price)
        low = float(tech_1m.get("low") or price)
        high_low_ratio = ((high - low) / (price + 1e-9)) * 100.0

        row = {
            "return_1m": ret_1m,
            "return_3m": ret_3m,
            "ema_9_21_spread": ((ema9 - ema21) / (price + 1e-9)) * 100.0,
            "ema_21_50_spread": ((ema21 - ema50) / (price + 1e-9)) * 100.0,
            "vwap_dist": ((price - vwap) / (vwap + 1e-9)) * 100.0,
            "rsi_norm": (rsi - 50.0) / 25.0,
            "macd_hist_norm": (macd_hist / (price + 1e-9)) * 1000.0,
            "atr_pct": atr_pct,
            "bb_pct_b": bb_pct_b,
            "bb_width": bb_width,
            "rvol": rvol,
            "high_low_ratio": high_low_ratio
        }

        return pd.DataFrame([row], columns=FEATURE_NAMES)

    def classify_regime(
        self,
        tech_1m: Dict[str, Any],
        order_book_metrics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Microstructure regime classification to filter low-probability environments."""
        if not tech_1m:
            return {"regime": "UNKNOWN", "is_suitable": False, "reason": "No technical data"}

        atr_pct = float(tech_1m.get("atr_pct") or 0.2)
        rvol = float(tech_1m.get("relative_volume") or 1.0)
        rsi = float(tech_1m.get("rsi") or 50.0)
        price = float(tech_1m.get("current_price") or 1.0)
        bb_upper = float(tech_1m.get("bb_upper") or (price * 1.002))
        bb_lower = float(tech_1m.get("bb_lower") or (price * 0.998))
        bb_width = ((bb_upper - bb_lower) / (price + 1e-9)) * 100.0
        ema9 = float(tech_1m.get("ema9") or price)
        ema21 = float(tech_1m.get("ema21") or price)
        vwap = float(tech_1m.get("vwap") or price)

        # 1. Low-Volatility Chop: Taker fees will overwhelm any micro price movement
        if bb_width < 0.12 and rvol < 0.7:
            return {
                "regime": "LOW_VOL_CHOP",
                "is_suitable": False,
                "reason": f"BB Width ({bb_width:.2f}%) too narrow; fees exceed potential move"
            }

        # 2. Extreme Volatility / Flash Spike: High slippage and liquidation risk
        if atr_pct > 2.5 or bb_width > 4.0:
            return {
                "regime": "HIGH_VOLATILITY",
                "is_suitable": False,
                "reason": f"ATR ({atr_pct:.2f}%) or BB Width ({bb_width:.2f}%) indicates uncontrolled volatility"
            }

        # 3. Trending Bullish
        if ema9 > ema21 and price > vwap and rsi >= 48:
            return {
                "regime": "TRENDING_BULL",
                "is_suitable": True,
                "recommended_direction": "LONG",
                "reason": "Strong upward momentum above VWAP and EMAs (Optimal LONG Scalp)"
            }

        # 4. Trending Bearish (Full Short Scalping Confluence)
        if ema9 < ema21 and price < vwap and rsi <= 52:
            return {
                "regime": "TRENDING_BEAR",
                "is_suitable": True,
                "recommended_direction": "SHORT",
                "reason": "Strong downward momentum below VWAP and EMAs (Optimal SHORT Scalp)"
            }

        # 5. Ranging / Mean Reversion
        return {
            "regime": "RANGING",
            "is_suitable": False,  # Strict Rule: No choppy/ranging trades in fast scalper
            "recommended_direction": "NEUTRAL",
            "reason": "Range bound / choppy. Rejecting to preserve capital and prevent prolonged holding."
        }

    def predict(
        self,
        tech_1m: Dict[str, Any],
        tech_3m: Dict[str, Any] = None,
        order_book_metrics: Dict[str, Any] = None,
        spread_pct: float = 0.02
    ) -> Dict[str, Any]:
        """Returns calibrated probabilistic forecast: P(up), P(down), P(no_move), and confidence."""
        if not self.is_fitted or not tech_1m:
            return {
                "p_up": 0.33,
                "p_down": 0.33,
                "p_no_move": 0.34,
                "confidence": 0.33,
                "model_name": self.model_name
            }

        features_df = self._extract_features(tech_1m, tech_3m)
        try:
            probs = self.model.predict_proba(features_df)[0]
            classes = list(self.model.classes_)
            prob_map = {cls: float(probs[idx]) for idx, cls in enumerate(classes)}

            # Class 0: DOWN, Class 1: NO-MOVE / CHOP, Class 2: UP
            p_down = prob_map.get(0, 0.15)
            p_no_move = prob_map.get(1, 0.70)
            p_up = prob_map.get(2, 0.15)

            # Micro-adjustment if order-book imbalance is strongly directional
            if order_book_metrics:
                imbalance = order_book_metrics.get("imbalance", 0.5)
                # Bids heavily dominating book (+0.2 delta shifts P(up) slightly)
                delta = (imbalance - 0.5) * 0.08
                p_up = max(0.01, min(0.90, p_up + delta))
                p_down = max(0.01, min(0.90, p_down - delta))

            # Normalize
            total = p_down + p_no_move + p_up
            if total > 0:
                p_down = round(p_down / total, 4)
                p_no_move = round(p_no_move / total, 4)
                p_up = round(p_up / total, 4)

            confidence = round(max(p_up, p_down, p_no_move), 4)

            return {
                "p_up": p_up,
                "p_down": p_down,
                "p_no_move": p_no_move,
                "confidence": confidence,
                "model_name": self.model_name
            }
        except Exception as e:
            logger.warning(f"ML prediction error: {e}")
            return {
                "p_up": 0.33,
                "p_down": 0.33,
                "p_no_move": 0.34,
                "confidence": 0.33,
                "model_name": self.model_name
            }

ml_model = MLProbabilityModel()
probability_model = ml_model
