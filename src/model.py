import numpy as np
import pandas as pd
import joblib
import io
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score

from src.features import FEATURE_COLS


class NiftyPredictor:
    """
    Ensemble of:
      - XGBRegressor  → predicts exact next-candle close price
      - XGBClassifier → predicts direction (UP/DOWN) + confidence
    """

    def __init__(self):
        self.regressor = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
        )
        self.classifier = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
        )
        self.scaler = RobustScaler()
        self.is_trained = False
        self.metrics: dict = {}
        self.feature_importance: pd.Series = pd.Series(dtype=float)

    # ──────────────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train both models using walk-forward cross-validation.
        Returns performance metrics.
        """
        clean = df[FEATURE_COLS + ["target_price", "target_direction", "close"]].dropna()
        if len(clean) < 60:
            raise ValueError(f"Not enough data to train (need ≥60 rows, got {len(clean)})")

        X = clean[FEATURE_COLS].values
        y_price = clean["target_price"].values
        y_dir   = clean["target_direction"].values
        prices  = clean["close"].values

        X_scaled = self.scaler.fit_transform(X)

        # Walk-forward CV (3 splits)
        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores, acc_scores, dir_acc_scores = [], [], []

        for train_idx, val_idx in tscv.split(X_scaled):
            Xtr, Xval = X_scaled[train_idx], X_scaled[val_idx]
            ytr_p, yval_p = y_price[train_idx], y_price[val_idx]
            ytr_d, yval_d = y_dir[train_idx],   y_dir[val_idx]
            p_val = prices[val_idx]

            self.regressor.fit(Xtr, ytr_p)
            self.classifier.fit(Xtr, ytr_d)

            pred_p = self.regressor.predict(Xval)
            pred_d = self.classifier.predict(Xval)

            mape_scores.append(mean_absolute_percentage_error(yval_p, pred_p) * 100)
            acc_scores.append(accuracy_score(yval_d, pred_d) * 100)

            # Directional accuracy from regression output too
            reg_dir = (pred_p > p_val).astype(int)
            dir_acc_scores.append(accuracy_score(yval_d, reg_dir) * 100)

        # Final fit on all data
        self.regressor.fit(X_scaled, y_price)
        self.classifier.fit(X_scaled, y_dir)
        self.is_trained = True

        self.metrics = {
            "mape":        round(float(np.mean(mape_scores)), 3),
            "dir_acc":     round(float(np.mean(acc_scores)), 1),
            "reg_dir_acc": round(float(np.mean(dir_acc_scores)), 1),
            "n_samples":   len(clean),
        }

        # Feature importance (from classifier)
        self.feature_importance = pd.Series(
            self.classifier.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)

        return self.metrics

    # ──────────────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> dict | None:
        """
        Predict next candle close price and direction for the latest row.
        Returns dict with prediction details, or None if model not ready.
        """
        if not self.is_trained:
            return None

        clean = df[FEATURE_COLS].dropna()
        if clean.empty:
            return None

        latest = clean.iloc[[-1]]
        X_scaled = self.scaler.transform(latest.values)

        pred_price  = float(self.regressor.predict(X_scaled)[0])
        prob_up     = float(self.classifier.predict_proba(X_scaled)[0][1])
        direction   = "UP" if prob_up >= 0.5 else "DOWN"
        confidence  = max(prob_up, 1 - prob_up)

        current_price = float(df["close"].dropna().iloc[-1])
        price_change  = pred_price - current_price
        pct_change    = (price_change / current_price) * 100

        # Predicted range using ATR
        atr = df["atr14"].dropna().iloc[-1] if "atr14" in df.columns else abs(price_change) * 2
        pred_low  = round(float(pred_price - 0.5 * atr), 2)
        pred_high = round(float(pred_price + 0.5 * atr), 2)

        return {
            "current_price": round(current_price, 2),
            "predicted_price": round(pred_price, 2),
            "predicted_low":   pred_low,
            "predicted_high":  pred_high,
            "price_change":    round(price_change, 2),
            "pct_change":      round(pct_change, 2),
            "direction":       direction,
            "confidence":      round(confidence * 100, 1),
            "prob_up":         round(prob_up * 100, 1),
        }

    # ──────────────────────────────────────────────────────────────────────
    def save(self) -> bytes:
        """Serialize model to bytes (for caching in session state)."""
        buf = io.BytesIO()
        joblib.dump(self, buf)
        return buf.getvalue()

    @classmethod
    def load(cls, data: bytes) -> "NiftyPredictor":
        buf = io.BytesIO(data)
        return joblib.load(buf)
