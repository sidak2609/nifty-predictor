import numpy as np
import pandas as pd
import joblib
import io
import warnings
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score

from src.features import FEATURE_COLS, MLP_LAG_COLS

warnings.filterwarnings("ignore")

# ── LightGBM (optional) ──────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

ROLLING_DAYS = 55       # training window
MAX_FEATURES = 25       # keep top N features after MI selection
PURGE_GAP = 6           # bars between train/val in CV (avoids target leakage)
N_CV_SPLITS = 4
CONF_THRESHOLD = 0.52   # below this → HOLD


def _rolling_window(df: pd.DataFrame, days: int = ROLLING_DAYS) -> pd.DataFrame:
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


def _drop_constant_features(X: np.ndarray, feature_names: list) -> tuple[np.ndarray, list]:
    """Drop features that are constant (std=0)."""
    stds = X.std(axis=0)
    mask = stds > 1e-8
    return X[:, mask], [f for f, m in zip(feature_names, mask) if m]


def _select_features_mi(X, y, feature_names, k=MAX_FEATURES):
    """Select top-k features by mutual information with the target."""
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
    selected = mi_series.head(k).index.tolist()
    return selected, mi_series


def _purged_ts_split(n, n_splits=N_CV_SPLITS, purge=PURGE_GAP):
    """Time series split with purge gap to prevent target leakage."""
    fold_size = n // (n_splits + 1)
    for i in range(n_splits):
        train_end  = fold_size * (i + 1)
        test_start = train_end + purge
        test_end   = min(test_start + fold_size, n)
        if test_start >= n:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def _sample_weights(n, decay=0.997):
    """Exponential recency weights: newest=1.0, oldest~0.3."""
    w = np.array([decay ** (n - 1 - i) for i in range(n)])
    return w / w.mean()


class NiftyPredictor:
    """
    Stacked ensemble with MI feature selection, conformal intervals, and abstention.

    Base: XGB-shallow, XGB-deep, LightGBM-DART, Ridge
    Meta: Ridge (heavily regularized)
    Direction: Calibrated XGBoost ternary classifier (UP/FLAT/DOWN)
    Intervals: Split conformal prediction (90% coverage)
    """

    def __init__(self):
        # ── Base models (10-min smoothed target) ──────────────────────────
        self.base_models = {
            "xgb_shallow": XGBRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=2,
                subsample=0.6, colsample_bytree=0.5, min_child_weight=20,
                reg_alpha=1.0, reg_lambda=5.0, gamma=0.1,
                random_state=42, tree_method="hist",
            ),
            "xgb_deep": XGBRegressor(
                n_estimators=200, learning_rate=0.03, max_depth=4,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
                reg_alpha=0.5, reg_lambda=3.0, gamma=0.05,
                random_state=99, tree_method="hist",
            ),
        }

        # LightGBM with DART boosting
        self.lgbm_model = None
        if _LGBM_AVAILABLE:
            try:
                self.lgbm_model = lgb.LGBMRegressor(
                    n_estimators=200, learning_rate=0.03, max_depth=3,
                    subsample=0.7, colsample_bytree=0.6,
                    min_child_weight=15, reg_alpha=0.5, reg_lambda=3.0,
                    boosting_type="dart", drop_rate=0.1,
                    random_state=77, n_jobs=1, verbose=-1,
                )
            except Exception:
                self.lgbm_model = None

        # Ridge (linear baseline — often hard to beat at high noise)
        self.ridge_model = Ridge(alpha=100.0)

        # ── 30-min models ─────────────────────────────────────────────────
        self.base_models_30 = {
            "xgb_30_shallow": XGBRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=2,
                subsample=0.6, colsample_bytree=0.5, min_child_weight=20,
                reg_alpha=1.0, reg_lambda=5.0,
                random_state=42, tree_method="hist",
            ),
            "xgb_30_deep": XGBRegressor(
                n_estimators=150, learning_rate=0.03, max_depth=3,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=15,
                reg_alpha=0.5, reg_lambda=3.0,
                random_state=99, tree_method="hist",
            ),
        }

        # ── Meta-learners (stacking) ─────────────────────────────────────
        self.meta_model    = Ridge(alpha=10.0)
        self.meta_model_30 = Ridge(alpha=10.0)

        # ── Ternary direction classifier ──────────────────────────────────
        self._base_clf = XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=15,
            reg_alpha=1.0, reg_lambda=3.0, gamma=0.1,
            random_state=42, tree_method="hist",
            eval_metric="mlogloss",
        )
        self.classifier = None

        # ── Scaler ────────────────────────────────────────────────────────
        self.scaler = RobustScaler()

        # ── State ─────────────────────────────────────────────────────────
        self.is_trained         = False
        self.selected_features  = FEATURE_COLS
        self._active_features   = FEATURE_COLS
        self._meta_model_names  = None
        self._meta_model_names_30 = None
        self._bias              = 0.0
        self._bias_30           = 0.0
        self._conformal_width   = 0.002
        self._conformal_width_30 = 0.005
        self._ensemble_weights  = {}
        self.metrics: dict      = {}
        self.feature_importance = pd.Series(dtype=float)
        self.mi_scores          = pd.Series(dtype=float)

    # ── Train ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        df = _rolling_window(df, days=ROLLING_DAYS)

        required = (FEATURE_COLS +
                    ["target_return_smooth", "target_return_3",
                     "target_direction_3class", "close"])
        clean = df[required].dropna()
        if len(clean) < 200:
            raise ValueError(f"Not enough data (need >=200 rows, got {len(clean)})")

        # Drop constant features
        X_raw, active_features = _drop_constant_features(
            clean[FEATURE_COLS].values, FEATURE_COLS
        )
        self._active_features = active_features

        y_smooth = clean["target_return_smooth"].values
        y_30     = clean["target_return_3"].values
        y_dir    = clean["target_direction_3class"].values
        prices   = clean["close"].values

        # Feature selection via mutual information
        selected, mi_scores = _select_features_mi(
            X_raw, y_smooth, active_features, k=MAX_FEATURES
        )
        self.selected_features = selected
        self.mi_scores = mi_scores

        sel_idx  = [active_features.index(f) for f in selected]
        X        = X_raw[:, sel_idx]
        X_scaled = self.scaler.fit_transform(X)

        n       = len(X_scaled)
        weights = _sample_weights(n)

        # ── Purged walk-forward CV ────────────────────────────────────────
        all_model_names = list(self.base_models.keys()) + ["lgbm", "ridge"]
        oof_preds    = {name: np.full(n, np.nan) for name in all_model_names}
        oof_preds_30 = {name: np.full(n, np.nan) for name in self.base_models_30}

        all_errors, all_errors_30, dir_accs = [], [], []
        last_val_idx = None

        for train_idx, val_idx in _purged_ts_split(n):
            Xtr, Xval     = X_scaled[train_idx], X_scaled[val_idx]
            ytr_s, yval_s = y_smooth[train_idx], y_smooth[val_idx]
            ytr_30, yval_30 = y_30[train_idx], y_30[val_idx]
            yval_d = y_dir[val_idx]
            w_tr   = weights[train_idx]

            # Base models — 10-min
            val_preds = []
            for name, model in self.base_models.items():
                model.fit(Xtr, ytr_s, sample_weight=w_tr)
                p = model.predict(Xval)
                oof_preds[name][val_idx] = p
                val_preds.append(p)

            # LightGBM
            if self.lgbm_model is not None:
                try:
                    self.lgbm_model.fit(Xtr, ytr_s, sample_weight=w_tr)
                    p = self.lgbm_model.predict(Xval)
                    oof_preds["lgbm"][val_idx] = p
                    val_preds.append(p)
                except Exception:
                    pass

            # Ridge
            self.ridge_model.fit(Xtr, ytr_s, sample_weight=w_tr)
            p = self.ridge_model.predict(Xval)
            oof_preds["ridge"][val_idx] = p
            val_preds.append(p)

            # 30-min models
            val_preds_30 = []
            for name, model in self.base_models_30.items():
                model.fit(Xtr, ytr_30, sample_weight=w_tr)
                p = model.predict(Xval)
                oof_preds_30[name][val_idx] = p
                val_preds_30.append(p)

            # Fold metrics
            avg_r   = np.mean(val_preds, axis=0)
            avg_r30 = np.mean(val_preds_30, axis=0)
            pred_dir = np.where(avg_r > CONF_THRESHOLD * 0.001, 2,
                                np.where(avg_r < -CONF_THRESHOLD * 0.001, 0, 1))
            dir_accs.append(accuracy_score(yval_d, pred_dir) * 100)
            all_errors.extend((avg_r - yval_s).tolist())
            all_errors_30.extend((avg_r30 - yval_30).tolist())
            last_val_idx = val_idx

        # Bias
        all_errors    = np.array(all_errors)
        all_errors_30 = np.array(all_errors_30)
        self._bias    = float(np.mean(all_errors))
        self._bias_30 = float(np.mean(all_errors_30))

        # ── Train meta-learner on OOF predictions ─────────────────────────
        valid_names = [k for k in all_model_names
                       if not np.all(np.isnan(oof_preds[k]))]
        oof_mask = np.ones(n, dtype=bool)
        for name in valid_names:
            oof_mask &= ~np.isnan(oof_preds[name])

        if oof_mask.sum() > 50:
            oof_matrix = np.column_stack([oof_preds[name][oof_mask]
                                          for name in valid_names])
            self.meta_model.fit(oof_matrix, y_smooth[oof_mask])
            self._meta_model_names = valid_names
        else:
            self._meta_model_names = None

        # 30-min meta
        valid_30 = [k for k in self.base_models_30
                    if not np.all(np.isnan(oof_preds_30[k]))]
        oof_mask_30 = np.ones(n, dtype=bool)
        for name in valid_30:
            oof_mask_30 &= ~np.isnan(oof_preds_30[name])
        if oof_mask_30.sum() > 50:
            oof_matrix_30 = np.column_stack([oof_preds_30[name][oof_mask_30]
                                             for name in valid_30])
            self.meta_model_30.fit(oof_matrix_30, y_30[oof_mask_30])
            self._meta_model_names_30 = valid_30
        else:
            self._meta_model_names_30 = None

        # ── Conformal prediction intervals ────────────────────────────────
        if last_val_idx is not None and len(last_val_idx) > 10:
            cal_preds = []
            for name in valid_names:
                vals = oof_preds[name][last_val_idx]
                if not np.all(np.isnan(vals)):
                    cal_preds.append(vals)
            if cal_preds:
                avg_cal = np.nanmean(cal_preds, axis=0)
                cal_resid = np.abs(avg_cal - y_smooth[last_val_idx])
                alpha = 0.10
                q_level = min(
                    np.ceil((len(cal_resid) + 1) * (1 - alpha)) / len(cal_resid),
                    1.0
                )
                self._conformal_width = float(np.quantile(cal_resid, q_level))

            cal_preds_30 = []
            for name in valid_30:
                vals = oof_preds_30[name][last_val_idx]
                if not np.all(np.isnan(vals)):
                    cal_preds_30.append(vals)
            if cal_preds_30:
                avg_cal_30 = np.nanmean(cal_preds_30, axis=0)
                cal_resid_30 = np.abs(avg_cal_30 - y_30[last_val_idx])
                self._conformal_width_30 = float(np.quantile(cal_resid_30, q_level))

        # ── Adaptive ensemble weights (inverse MAE on last fold) ──────────
        if last_val_idx is not None:
            inv_errors = {}
            for name in valid_names:
                vals = oof_preds[name][last_val_idx]
                if np.all(np.isnan(vals)):
                    continue
                mae = np.nanmean(np.abs(vals - y_smooth[last_val_idx]))
                inv_errors[name] = 1.0 / (mae + 1e-10)
            total_w = sum(inv_errors.values())
            if total_w > 0:
                self._ensemble_weights = {k: v / total_w for k, v in inv_errors.items()}

        # ── Final fit on all data ─────────────────────────────────────────
        for name, model in self.base_models.items():
            model.fit(X_scaled, y_smooth, sample_weight=weights)

        if self.lgbm_model is not None:
            try:
                self.lgbm_model.fit(X_scaled, y_smooth, sample_weight=weights)
            except Exception:
                self.lgbm_model = None

        self.ridge_model.fit(X_scaled, y_smooth, sample_weight=weights)

        for name, model in self.base_models_30.items():
            model.fit(X_scaled, y_30, sample_weight=weights)

        # Ternary classifier
        try:
            self.classifier = CalibratedClassifierCV(
                self._base_clf, method="sigmoid", cv=3
            )
            self.classifier.fit(X_scaled, y_dir)
        except Exception:
            self.classifier = None

        self.is_trained = True

        # ── Metrics ───────────────────────────────────────────────────────
        all_preds = [m.predict(X_scaled) for m in self.base_models.values()]
        if self.lgbm_model is not None:
            all_preds.append(self.lgbm_model.predict(X_scaled))
        all_preds.append(self.ridge_model.predict(X_scaled))
        avg_pred = np.mean(all_preds, axis=0) - self._bias

        pred_prices   = prices * (1 + avg_pred)
        actual_prices = prices * (1 + y_smooth)
        mape = float(np.mean(np.abs(pred_prices - actual_prices) / actual_prices) * 100)

        try:
            imp = self.base_models["xgb_deep"].feature_importances_
            self.feature_importance = pd.Series(
                imp, index=self.selected_features
            ).sort_values(ascending=False)
        except Exception:
            self.feature_importance = pd.Series(dtype=float)

        dropped = len(FEATURE_COLS) - len(active_features)
        self.metrics = {
            "mape":              round(mape, 3),
            "dir_acc":           round(float(np.mean(dir_accs)), 1),
            "n_samples":         len(clean),
            "bias_corr":         round(self._bias * 100, 4),
            "conformal_width":   round(self._conformal_width * 100, 4),
            "lgbm":              self.lgbm_model is not None,
            "stacking":          self._meta_model_names is not None,
            "selected_features": len(self.selected_features),
            "active_features":   len(active_features),
            "dropped_features":  dropped,
        }
        return self.metrics

    # ── Predict 10-min ────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> dict | None:
        if not self.is_trained:
            return None

        available = [f for f in self.selected_features if f in df.columns]
        if len(available) != len(self.selected_features):
            return None
        clean = df[self.selected_features].dropna()
        if clean.empty:
            return None

        latest   = clean.iloc[[-1]]
        X_scaled = self.scaler.transform(latest.values)

        # Base model predictions
        base_preds = {}
        for name, model in self.base_models.items():
            base_preds[name] = float(model.predict(X_scaled)[0])
        if self.lgbm_model is not None:
            try:
                base_preds["lgbm"] = float(self.lgbm_model.predict(X_scaled)[0])
            except Exception:
                pass
        base_preds["ridge"] = float(self.ridge_model.predict(X_scaled)[0])

        # Meta-learner or weighted average
        if (self._meta_model_names and
                all(n in base_preds for n in self._meta_model_names)):
            meta_in = np.array([[base_preds[n] for n in self._meta_model_names]])
            corrected_return = float(self.meta_model.predict(meta_in)[0]) - self._bias
        elif self._ensemble_weights:
            total_w = sum(self._ensemble_weights.get(n, 0) for n in base_preds)
            if total_w > 0:
                corrected_return = (
                    sum(base_preds[n] * self._ensemble_weights.get(n, 0)
                        for n in base_preds) / total_w
                ) - self._bias
            else:
                corrected_return = float(np.mean(list(base_preds.values()))) - self._bias
        else:
            corrected_return = float(np.mean(list(base_preds.values()))) - self._bias

        current_price = float(df["close"].dropna().iloc[-1])
        pred_price    = round(current_price * (1 + corrected_return), 2)

        # Conformal intervals
        pred_low  = round(current_price * (1 + corrected_return - self._conformal_width), 2)
        pred_high = round(current_price * (1 + corrected_return + self._conformal_width), 2)

        # Direction
        direction = "UP" if corrected_return > 0 else "DOWN"

        # Ensemble agreement
        n_up   = sum(1 for v in base_preds.values() if v > 0)
        n_down = len(base_preds) - n_up
        agreement = max(n_up, n_down) / len(base_preds)

        # Magnitude signal
        magnitude = min(abs(corrected_return) / max(self._conformal_width, 1e-6), 1.0)

        # Classifier
        prob_up   = 0.5
        prob_flat = 0.0
        if self.classifier is not None:
            try:
                proba = self.classifier.predict_proba(X_scaled)[0]
                classes = list(self.classifier.classes_)
                prob_down = proba[classes.index(0)] if 0 in classes else 0.0
                prob_flat = proba[classes.index(1)] if 1 in classes else 0.0
                prob_up   = proba[classes.index(2)] if 2 in classes else 0.5
            except Exception:
                prob_up, prob_flat = 0.5, 0.0

        # Composite confidence
        clf_conf   = max(prob_up, 1 - prob_up - prob_flat)
        confidence = (0.35 * clf_conf + 0.35 * agreement +
                      0.20 * magnitude + 0.10 * 0.5)

        # Cap confidence when regressor and classifier disagree
        if self.classifier is not None and (corrected_return > 0) != (prob_up > 0.5):
            confidence = min(confidence, 0.55)

        # Abstention
        if confidence < CONF_THRESHOLD or prob_flat > 0.45:
            direction = "HOLD"

        return {
            "current_price":      round(current_price, 2),
            "predicted_price":    pred_price,
            "predicted_low":      pred_low,
            "predicted_high":     pred_high,
            "price_change":       round(pred_price - current_price, 2),
            "pct_change":         round(corrected_return * 100, 4),
            "direction":          direction,
            "confidence":         round(confidence * 100, 1),
            "prob_up":            round(prob_up * 100, 1),
            "prob_flat":          round(prob_flat * 100, 1),
            "ensemble_agreement": round(agreement * 100, 1),
        }

    # ── Predict 30-min ────────────────────────────────────────────────────
    def predict_30min(self, df: pd.DataFrame) -> dict | None:
        if not self.is_trained:
            return None
        available = [f for f in self.selected_features if f in df.columns]
        if len(available) != len(self.selected_features):
            return None
        clean = df[self.selected_features].dropna()
        if clean.empty:
            return None

        latest   = clean.iloc[[-1]]
        X_scaled = self.scaler.transform(latest.values)

        base_preds_30 = {}
        for name, model in self.base_models_30.items():
            base_preds_30[name] = float(model.predict(X_scaled)[0])

        if (self._meta_model_names_30 and
                all(n in base_preds_30 for n in self._meta_model_names_30)):
            meta_in = np.array([[base_preds_30[n]
                                 for n in self._meta_model_names_30]])
            corrected_30 = float(self.meta_model_30.predict(meta_in)[0]) - self._bias_30
        else:
            corrected_30 = float(np.mean(list(base_preds_30.values()))) - self._bias_30

        current_price = float(df["close"].dropna().iloc[-1])
        pred_price_30 = round(current_price * (1 + corrected_30), 2)

        return {
            "predicted_price_30min": pred_price_30,
            "pct_change_30min":      round(corrected_30 * 100, 4),
            "direction_30min":       "UP" if corrected_30 > 0 else "DOWN",
            "price_change_30min":    round(pred_price_30 - current_price, 2),
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
