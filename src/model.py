import numpy as np
import pandas as pd
import joblib
import io
import warnings
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

from src.features import FEATURE_COLS, MLP_LAG_COLS

warnings.filterwarnings("ignore")

# ── LightGBM (optional — graceful fallback if not available) ──────────────
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

N_ENSEMBLE = 3          # XGBoost models per horizon
ROLLING_DAYS = 30       # train on most recent N days only


def _rolling_window(df: pd.DataFrame, days: int = ROLLING_DAYS) -> pd.DataFrame:
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


class NiftyPredictor:
    """
    Full ensemble:
      • 3× XGBoost regressor  — 10-min horizon
      • 3× XGBoost regressor  — 30-min horizon
      • 1× LightGBM regressor — 10-min (if available)
      • 1× MLP (sequence lag) — 10-min blend
      • 3× session sub-models — open rush / midday / close rush
      • 1× Calibrated XGBoost classifier — direction + confidence
    Training: rolling 30-day window, 4-fold TimeSeriesSplit CV
    """

    def __init__(self):
        _xgb_params = dict(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            subsample=0.75, colsample_bytree=0.7, min_child_weight=8,
            reg_alpha=0.5, reg_lambda=2.0, tree_method="hist",
        )

        # 10-min horizon ensemble
        self.regressors = [
            XGBRegressor(**_xgb_params, random_state=s) for s in range(N_ENSEMBLE)
        ]
        # 30-min horizon ensemble
        self.regressors_3 = [
            XGBRegressor(**{**_xgb_params, "random_state": s + 10}) for s in range(N_ENSEMBLE)
        ]

        # LightGBM (10-min)
        self.lgbm_regressor = None
        if _LGBM_AVAILABLE:
            try:
                self.lgbm_regressor = lgb.LGBMRegressor(
                    n_estimators=400, learning_rate=0.03, max_depth=4,
                    subsample=0.75, colsample_bytree=0.7,
                    min_child_weight=8, reg_alpha=0.5, reg_lambda=2.0,
                    random_state=99, n_jobs=1, verbose=-1,
                )
            except Exception:
                self.lgbm_regressor = None

        # MLP sequence model
        self.mlp_regressor = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=200, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
        )
        self.mlp_scaler  = RobustScaler()
        self.mlp_trained = False

        # Time-of-day sub-models
        _session_params = dict(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, tree_method="hist",
        )
        self.session_models = {
            s: XGBRegressor(**_session_params)
            for s in ["open_rush", "midday", "close_rush"]
        }
        self.session_scalers  = {s: RobustScaler() for s in self.session_models}
        self.session_trained: set = set()

        # Direction classifier
        self._base_clf = XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            subsample=0.75, colsample_bytree=0.7, min_child_weight=8,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42,
            tree_method="hist", eval_metric="logloss",
        )
        self.classifier = None

        self.scaler   = RobustScaler()
        self.is_trained = False
        self.metrics: dict = {}
        self.feature_importance: pd.Series = pd.Series(dtype=float)

        self._bias: float = 0.0
        self._bias_3: float = 0.0
        self._q10: float = -0.002
        self._q90: float = +0.002

    # ── Internal helpers ───────────────────────────────────────────────────
    def _train_mlp(self, df: pd.DataFrame):
        available = [c for c in MLP_LAG_COLS if c in df.columns]
        if len(available) < 10:
            return
        clean = df[available + ["target_return"]].dropna()
        if len(clean) < 100:
            return
        X = self.mlp_scaler.fit_transform(clean[available].values)
        y = clean["target_return"].values
        try:
            self.mlp_regressor.fit(X, y)
            self.mlp_trained = True
        except Exception:
            self.mlp_trained = False

    def _train_session_models(self, df: pd.DataFrame):
        if "session" not in df.columns:
            return
        clean = df[FEATURE_COLS + ["target_return", "session"]].dropna()
        for session_name, model in self.session_models.items():
            subset = clean[clean["session"] == session_name]
            if len(subset) < 50:
                continue
            X = self.session_scalers[session_name].fit_transform(
                subset[FEATURE_COLS].values
            )
            try:
                model.fit(X, subset["target_return"].values)
                self.session_trained.add(session_name)
            except Exception:
                pass

    # ── Train ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        # Rolling window: most recent 30 days
        df = _rolling_window(df, days=ROLLING_DAYS)

        required = FEATURE_COLS + ["target_return", "target_return_3", "target_direction", "close"]
        clean = df[required].dropna()
        if len(clean) < 100:
            raise ValueError(f"Not enough data (need ≥100 rows, got {len(clean)})")

        X      = clean[FEATURE_COLS].values
        y_ret  = clean["target_return"].values
        y_ret3 = clean["target_return_3"].values
        y_dir  = clean["target_direction"].values
        prices = clean["close"].values

        X_scaled = self.scaler.fit_transform(X)

        # ── Walk-forward CV ────────────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=4)
        all_errors, all_errors_3, dir_accs = [], [], []

        for train_idx, val_idx in tscv.split(X_scaled):
            Xtr, Xval   = X_scaled[train_idx], X_scaled[val_idx]
            ytr_r,  yval_r  = y_ret[train_idx],  y_ret[val_idx]
            ytr_r3, yval_r3 = y_ret3[train_idx], y_ret3[val_idx]
            ytr_d,  yval_d  = y_dir[train_idx],  y_dir[val_idx]

            preds_r, preds_r3 = [], []
            for reg, reg3 in zip(self.regressors, self.regressors_3):
                reg.fit(Xtr, ytr_r);   preds_r.append(reg.predict(Xval))
                reg3.fit(Xtr, ytr_r3); preds_r3.append(reg3.predict(Xval))

            if self.lgbm_regressor is not None:
                try:
                    self.lgbm_regressor.fit(Xtr, ytr_r)
                    preds_r.append(self.lgbm_regressor.predict(Xval))
                except Exception:
                    pass

            avg_r  = np.mean(preds_r,  axis=0)
            avg_r3 = np.mean(preds_r3, axis=0)

            dir_accs.append(accuracy_score(yval_d, (avg_r > 0).astype(int)) * 100)
            all_errors.extend((avg_r - yval_r).tolist())
            all_errors_3.extend((avg_r3 - yval_r3).tolist())

        # Bias + error quantiles from CV
        all_errors   = np.array(all_errors)
        all_errors_3 = np.array(all_errors_3)
        self._bias   = float(np.mean(all_errors))
        self._bias_3 = float(np.mean(all_errors_3))
        self._q10    = float(np.percentile(all_errors, 10))
        self._q90    = float(np.percentile(all_errors, 90))

        # ── Final fit ──────────────────────────────────────────────────────
        for reg, reg3 in zip(self.regressors, self.regressors_3):
            reg.fit(X_scaled, y_ret)
            reg3.fit(X_scaled, y_ret3)

        if self.lgbm_regressor is not None:
            try:
                self.lgbm_regressor.fit(X_scaled, y_ret)
            except Exception:
                self.lgbm_regressor = None

        self.classifier = CalibratedClassifierCV(
            self._base_clf, method="isotonic", cv=TimeSeriesSplit(n_splits=3)
        )
        self.classifier.fit(X_scaled, y_dir)

        self._train_mlp(df)
        self._train_session_models(df)

        self.is_trained = True

        # MAPE on training set (indicative)
        preds_train = np.mean(
            [r.predict(X_scaled) for r in self.regressors], axis=0
        ) - self._bias
        pred_prices   = prices * (1 + preds_train)
        actual_prices = prices * (1 + y_ret)
        mape = float(np.mean(np.abs(pred_prices - actual_prices) / actual_prices) * 100)

        # Feature importance (average XGB ensemble)
        avg_imp = np.mean([r.feature_importances_ for r in self.regressors], axis=0)
        self.feature_importance = pd.Series(avg_imp, index=FEATURE_COLS).sort_values(ascending=False)

        self.metrics = {
            "mape":            round(mape, 3),
            "dir_acc":         round(float(np.mean(dir_accs)), 1),
            "n_samples":       len(clean),
            "bias_corr":       round(self._bias * 100, 4),
            "lgbm":            self.lgbm_regressor is not None,
            "mlp":             self.mlp_trained,
            "sessions_trained": sorted(self.session_trained),
        }
        return self.metrics

    # ── Predict 10-min ────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> dict | None:
        if not self.is_trained:
            return None
        clean = df[FEATURE_COLS].dropna()
        if clean.empty:
            return None

        latest   = clean.iloc[[-1]]
        X_scaled = self.scaler.transform(latest.values)

        # Ensemble: XGBoost + LightGBM
        raw_preds = [r.predict(X_scaled)[0] for r in self.regressors]
        if self.lgbm_regressor is not None:
            try:
                raw_preds.append(float(self.lgbm_regressor.predict(X_scaled)[0]))
            except Exception:
                pass
        corrected_return = float(np.mean(raw_preds)) - self._bias

        # Blend MLP (30% weight)
        if self.mlp_trained:
            try:
                avail_lag = [c for c in MLP_LAG_COLS if c in df.columns]
                lat_lag   = df[avail_lag].dropna().iloc[[-1]]
                X_mlp     = self.mlp_scaler.transform(lat_lag.values)
                mlp_ret   = float(self.mlp_regressor.predict(X_mlp)[0])
                corrected_return = 0.70 * corrected_return + 0.30 * mlp_ret
            except Exception:
                pass

        # Blend session model (20% weight)
        if "session" in df.columns:
            cur_session = df["session"].iloc[-1]
            if cur_session in self.session_trained:
                try:
                    X_s    = self.session_scalers[cur_session].transform(latest.values)
                    s_ret  = float(self.session_models[cur_session].predict(X_s)[0])
                    corrected_return = 0.80 * corrected_return + 0.20 * s_ret
                except Exception:
                    pass

        current_price = float(df["close"].dropna().iloc[-1])
        pred_price    = round(current_price * (1 + corrected_return), 2)

        # Calibrated classifier
        prob_up    = float(self.classifier.predict_proba(X_scaled)[0][1])
        direction  = "UP" if prob_up >= 0.5 else "DOWN"
        confidence = max(prob_up, 1 - prob_up)

        # Cap confidence when regressor and classifier disagree
        if (corrected_return > 0) != (prob_up >= 0.5):
            confidence = min(confidence, 0.58)

        pred_low  = round(float(current_price * (1 + corrected_return - abs(self._q90))), 2)
        pred_high = round(float(current_price * (1 + corrected_return + abs(self._q90))), 2)

        return {
            "current_price":   round(current_price, 2),
            "predicted_price": pred_price,
            "predicted_low":   pred_low,
            "predicted_high":  pred_high,
            "price_change":    round(pred_price - current_price, 2),
            "pct_change":      round(corrected_return * 100, 4),
            "direction":       direction,
            "confidence":      round(confidence * 100, 1),
            "prob_up":         round(prob_up * 100, 1),
        }

    # ── Predict 30-min ────────────────────────────────────────────────────
    def predict_30min(self, df: pd.DataFrame) -> dict | None:
        if not self.is_trained:
            return None
        clean = df[FEATURE_COLS].dropna()
        if clean.empty:
            return None
        latest   = clean.iloc[[-1]]
        X_scaled = self.scaler.transform(latest.values)

        raw_return_3  = float(np.mean([r.predict(X_scaled)[0] for r in self.regressors_3]))
        corrected_3   = raw_return_3 - self._bias_3
        current_price = float(df["close"].dropna().iloc[-1])
        pred_price_3  = round(current_price * (1 + corrected_3), 2)

        return {
            "predicted_price_30min": pred_price_3,
            "pct_change_30min":      round(corrected_3 * 100, 4),
            "direction_30min":       "UP" if corrected_3 > 0 else "DOWN",
            "price_change_30min":    round(pred_price_3 - current_price, 2),
        }

    # ── Serialization ─────────────────────────────────────────────────────
    def save(self) -> bytes:
        buf = io.BytesIO()
        joblib.dump(self, buf)
        return buf.getvalue()

    @classmethod
    def load(cls, data: bytes) -> "NiftyPredictor":
        buf = io.BytesIO(data)
        return joblib.load(buf)
