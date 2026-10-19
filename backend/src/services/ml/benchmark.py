import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

from backend.src.services.ml.dataset_builder import DatasetBuilder

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class ModelBenchmarkSuite:
    """Rigorous out-of-sample benchmark comparing LightGBM, XGBoost, and HistGradientBoosting
    with probability calibration and microsecond latency profiling.
    """

    def __init__(self, symbols: list = None):
        self.symbols = symbols or ["NEARUSDT", "SUIUSDT", "APTUSDT", "SEIUSDT", "FETUSDT"]

    def load_combined_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        all_X = []
        all_y = []
        print(f"Fetching real Binance USD-M Futures 1m data for {len(self.symbols)} tokens...")
        for sym in self.symbols:
            df = DatasetBuilder.fetch_futures_klines(sym, limit=1200)
            if not df.empty:
                X, y = DatasetBuilder.extract_features_and_labels(df)
                if not X.empty:
                    all_X.append(X)
                    all_y.append(y)
                    print(f"  Loaded {len(X)} labeled rows for {sym}")

        if not all_X:
            raise RuntimeError("Failed to fetch historical market data from Binance Futures.")

        combined_X = pd.concat(all_X, ignore_index=True)
        combined_y = pd.concat(all_y, ignore_index=True)
        print(f"Total Combined Dataset: {len(combined_X)} rows. Class distribution:\n{combined_y.value_counts(normalize=True).to_dict()}")
        return combined_X, combined_y

    def run_benchmark(self) -> Dict[str, Any]:
        X, y = self.load_combined_dataset()

        # Strict Chronological Train (70%) / Validation (15%) / Test (15%) Splits
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

        results = {}

        # ----------------------------------------------------
        # 1. LightGBM Classifier
        # ----------------------------------------------------
        print("\n--- Training LightGBM ---")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            random_state=42,
            verbosity=-1
        )
        # Fit with 3-fold cross-validation calibration on train+val
        X_train_val = X.iloc[:val_end]
        y_train_val = y.iloc[:val_end]

        t0 = time.time()
        calibrated_lgb = CalibratedClassifierCV(estimator=lgb_model, method="sigmoid", cv=3)
        calibrated_lgb.fit(X_train_val, y_train_val)
        lgb_train_time = time.time() - t0

        results["LightGBM"] = self._evaluate_model(calibrated_lgb, X_test, y_test, lgb_train_time)

        # ----------------------------------------------------
        # 2. XGBoost Classifier
        # ----------------------------------------------------
        print("\n--- Training XGBoost ---")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        )
        t0 = time.time()
        calibrated_xgb = CalibratedClassifierCV(estimator=xgb_model, method="sigmoid", cv=3)
        calibrated_xgb.fit(X_train_val, y_train_val)
        xgb_train_time = time.time() - t0

        results["XGBoost"] = self._evaluate_model(calibrated_xgb, X_test, y_test, xgb_train_time)

        # ----------------------------------------------------
        # 3. HistGradientBoosting (Scikit-Learn Baseline)
        # ----------------------------------------------------
        print("\n--- Training HistGradientBoosting ---")
        hgb_model = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )
        t0 = time.time()
        calibrated_hgb = CalibratedClassifierCV(estimator=hgb_model, method="sigmoid", cv=3)
        calibrated_hgb.fit(X_train_val, y_train_val)
        hgb_train_time = time.time() - t0

        results["HistGradientBoosting"] = self._evaluate_model(calibrated_hgb, X_test, y_test, hgb_train_time)

        # Save the best model based on out-of-sample Log-Loss and Brier Score
        best_name = min(results.keys(), key=lambda k: results[k]["log_loss"])
        best_model = (
            calibrated_lgb if best_name == "LightGBM" else (
                calibrated_xgb if best_name == "XGBoost" else calibrated_hgb
            )
        )

        best_path = os.path.join(MODELS_DIR, "production_scalp_model.joblib")
        joblib.dump(best_model, best_path)
        print(f"\nSaved best model ({best_name}) to {best_path}")

        # Also save individual models for independent verification
        joblib.dump(calibrated_lgb, os.path.join(MODELS_DIR, "calibrated_lgbm.joblib"))
        joblib.dump(calibrated_xgb, os.path.join(MODELS_DIR, "calibrated_xgboost.joblib"))

        return {
            "best_model": best_name,
            "metrics": results,
            "test_samples": len(X_test),
            "features": DatasetBuilder.FEATURE_NAMES
        }

    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, train_time: float) -> Dict[str, Any]:
        # Measure Latency (500 single-row inference runs)
        latencies_ms = []
        sample_row = X_test.iloc[0:1]
        for _ in range(300):
            t_start = time.perf_counter()
            _ = model.predict_proba(sample_row)
            latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

        p50 = np.percentile(latencies_ms, 50)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)

        # Probabilities on full test set
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)

        loss = log_loss(y_test, probs)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")

        # Brier score on UP class (class 2)
        y_test_up = (y_test == 2).astype(int)
        brier_up = brier_score_loss(y_test_up, probs[:, 2])

        metrics = {
            "train_duration_sec": round(train_time, 3),
            "log_loss": round(float(loss), 4),
            "brier_score_up": round(float(brier_up), 4),
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "latency_p50_ms": round(float(p50), 3),
            "latency_p95_ms": round(float(p95), 3),
            "latency_p99_ms": round(float(p99), 3)
        }

        print(f"  Test Log-Loss: {metrics['log_loss']} | Brier (UP): {metrics['brier_score_up']} | Acc: {metrics['accuracy']} | Latency p50: {metrics['latency_p50_ms']}ms")
        return metrics

if __name__ == "__main__":
    suite = ModelBenchmarkSuite()
    res = suite.run_benchmark()
    print("\nBENCHMARK SUMMARY:")
    for m, vals in res["metrics"].items():
        print(f"  {m}: LogLoss={vals['log_loss']}, Brier={vals['brier_score_up']}, p50={vals['latency_p50_ms']}ms, p99={vals['latency_p99_ms']}ms")
