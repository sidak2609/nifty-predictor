import numpy as np
import pandas as pd
import joblib
import io
import warnings
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from src.features import FEATURE_COLS

warnings.filterwarnings("ignore")

N_ENSEMBLE = 5   # number of models in ensemble


class NiftyPredictor:
    """
    Improved ensemble model:
    - Predicts RETURN (%) not absolute price → removes upward bias
    - Ensemble of N XGBoost regressors (different seeds) → more stable predictions
    - Calibrated classifier → honest confidence scores
    - Empirical error quantiles from validation → realistic price range
    - Bias correction using validation residuals
    """

    def __init__(self):
        # Ensemble of regressors (predict next-candle return %)
        self.regressors = [
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.75,
                colsample_bytree=0.7,
                min_child_weight=8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=seed,
                tree_method="hist",
            )
            for seed in range(N_ENSEMBLE)
        ]

        # Base classifier (direction) — will be calibrated
        self._base_clf = XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.75,
            colsample_bytree=0.7,
            min_child_weight=8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self.classifier = None   # set after calibration

        self.scaler = RobustScaler()
        self.is_trained = False
        self.metrics: dict = {}
        self.feature_importance: pd.Series = pd.Series(dtype=float)

        # Bias and error quantiles learned from validation
        self._bias: float = 0.0
        self._q10: float = -0.002   # 10th pct of return errors
        self._q90: float = +0.002   # 90th pct of return errors

    # ──────────────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        required = FEATURE_COLS + ["target_return", "target_direction", "close"]
        clean = df[required].dropna()
        if len(clean) < 100:
            raise ValueError(f"Not enough data to train (need ≥100 rows, got {len(clean)})")

        X      = clean[FEATURE_COLS].values
        y_ret  = clean["target_return"].values
        y_dir  = clean["target_direction"].values
        prices = clean["close"].values

        X_scaled = self.scaler.fit_transform(X)

        # ── Walk-forward CV for metrics + bias/range estimation ────────────
        tscv = TimeSeriesSplit(n_splits=4)
        all_ret_errors, dir_accs = [], []

        for train_idx, val_idx in tscv.split(X_scaled):
            Xtr, Xval   = X_scaled[train_idx], X_scaled[val_idx]
            ytr_r, yval_r = y_ret[train_idx],  y_ret[val_idx]
            ytr_d, yval_d = y_dir[train_idx],  y_dir[val_idx]
            p_val         = prices[val_idx]

            # Ensemble regressor predictions (averaged)
            preds_r = []
            for reg in self.regressors:
                reg.fit(Xtr, ytr_r)
                preds_r.append(reg.predict(Xval))
            avg_pred_r = np.mean(preds_r, axis=0)

            # Directional accuracy from regressor
            dir_from_reg = (avg_pred_r > 0).astype(int)
            dir_accs.append(accuracy_score(yval_d, dir_from_reg) * 100)

            # Collect return prediction errors for quantile estimation
            ret_errors = avg_pred_r - yval_r
            all_ret_errors.extend(ret_errors.tolist())

        # Bias = mean of all validation return errors
        all_ret_errors = np.array(all_ret_errors)
        self._bias = float(np.mean(all_ret_errors))
        self._q10  = float(np.percentile(all_ret_errors, 10))
        self._q90  = float(np.percentile(all_ret_errors, 90))

        # ── Final fit on ALL data ──────────────────────────────────────────
        for reg in self.regressors:
            reg.fit(X_scaled, y_ret)

        # Calibrated classifier using cross-val (isotonic for reliable probs)
        self.classifier = CalibratedClassifierCV(
            self._base_clf, method="isotonic", cv=TimeSeriesSplit(n_splits=3)
        )
        self.classifier.fit(X_scaled, y_dir)

        self.is_trained = True

        # MAPE on full training set (indicative, not validation)
        preds_train = np.mean(
            [reg.predict(X_scaled) for reg in self.regressors], axis=0
        ) - self._bias
        pred_prices  = prices * (1 + preds_train)
        actual_prices = prices * (1 + y_ret)
        mape = float(np.mean(np.abs(pred_prices - actual_prices) / actual_prices) * 100)

        self.metrics = {
            "mape":      round(mape, 3),
            "dir_acc":   round(float(np.mean(dir_accs)), 1),
            "n_samples": len(clean),
            "bias_corr": round(self._bias * 100, 4),
        }

        # Feature importance (average across ensemble)
        avg_imp = np.mean([r.feature_importances_ for r in self.regressors], axis=0)
        self.feature_importance = pd.Series(avg_imp, index=FEATURE_COLS).sort_values(ascending=False)

        return self.metrics

    # ──────────────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> dict | None:
        if not self.is_trained:
            return None

        clean = df[FEATURE_COLS].dropna()
        if clean.empty:
            return None

        latest   = clean.iloc[[-1]]
        X_scaled = self.scaler.transform(latest.values)

        # ── Ensemble return prediction with bias correction ────────────────
        raw_return = float(np.mean([r.predict(X_scaled)[0] for r in self.regressors]))
        corrected_return = raw_return - self._bias   # remove learned bias

        current_price = float(df["close"].dropna().iloc[-1])
        pred_price    = round(current_price * (1 + corrected_return), 2)
        price_change  = pred_price - current_price
        pct_change    = corrected_return * 100

        # ── Empirical price range from validation error distribution ──────
        # q10/q90 are return errors → convert to price range
        pred_low  = round(current_price * (1 + corrected_return - abs(self._q90)), 2)
        pred_high = round(current_price * (1 + corrected_return + abs(self._q90)), 2)

        # ── Calibrated classifier confidence ──────────────────────────────
        prob_up    = float(self.classifier.predict_proba(X_scaled)[0][1])
        direction  = "UP" if prob_up >= 0.5 else "DOWN"
        confidence = max(prob_up, 1 - prob_up)

        # Blend: if regressor and classifier disagree, reduce confidence
        reg_dir = corrected_return > 0
        clf_dir = prob_up >= 0.5
        if reg_dir != clf_dir:
            confidence = min(confidence, 0.58)   # cap when models disagree

        return {
            "current_price":   round(current_price, 2),
            "predicted_price": pred_price,
            "predicted_low":   pred_low,
            "predicted_high":  pred_high,
            "price_change":    round(price_change, 2),
            "pct_change":      round(pct_change, 4),
            "direction":       direction,
            "confidence":      round(confidence * 100, 1),
            "prob_up":         round(prob_up * 100, 1),
            "reg_return_pct":  round(corrected_return * 100, 4),
        }

    # ──────────────────────────────────────────────────────────────────────
    def save(self) -> bytes:
        buf = io.BytesIO()
        joblib.dump(self, buf)
        return buf.getvalue()

    @classmethod
    def load(cls, data: bytes) -> "NiftyPredictor":
        buf = io.BytesIO(data)
        return joblib.load(buf)
