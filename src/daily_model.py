"""
NiftyDailyPredictor — multi-horizon daily prediction model for Nifty 50.
Predicts 5/10/20/30-day returns using a stacked ensemble of XGBoost,
LightGBM, Ridge, and MLPRegressor with conformal prediction intervals.
"""

import numpy as np
import pandas as pd
import joblib
import io
import warnings
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, r2_score

from src.daily_features import (
    DAILY_FEATURE_COLS,
    TARGET_COLS,
    fetch_and_prepare_daily,
)
from src.sentiment import (
    fetch_news_sentiment,
    fetch_reddit_sentiment,
    fetch_market_breadth,
)

warnings.filterwarnings("ignore")

# ── LightGBM (optional) ──────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS = [5, 10, 20, 30]
MAX_FEATURES = 30
PURGE_GAP = 30          # days gap between train/val (target looks 30d forward)
RECENCY_DECAY = 0.999
N_CV_SPLITS = 4
MLP_LOOKBACK = 20       # days of returns to feed MLP as flattened sequence


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_weights(n: int, decay: float = RECENCY_DECAY) -> np.ndarray:
    """Exponential recency weights: newest=1.0."""
    w = np.array([decay ** (n - 1 - i) for i in range(n)])
    return w / w.mean()


def _purged_ts_split(n: int, n_splits: int = N_CV_SPLITS,
                     purge: int = PURGE_GAP):
    """Time series split with purge gap to prevent target leakage."""
    fold_size = n // (n_splits + 1)
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end + purge
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def _select_features_mi(X: np.ndarray, y: np.ndarray,
                         feature_names: list,
                         k: int = MAX_FEATURES) -> tuple:
    """Select top-k features by mutual information with target."""
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
    selected = mi_series.head(k).index.tolist()
    return selected, mi_series


def _drop_constant_features(X: np.ndarray,
                              feature_names: list) -> tuple:
    """Drop features with zero variance."""
    stds = np.nanstd(X, axis=0)
    mask = stds > 1e-8
    return X[:, mask], [f for f, m in zip(feature_names, mask) if m]


def _build_mlp_sequence_features(df: pd.DataFrame,
                                  lookback: int = MLP_LOOKBACK) -> np.ndarray:
    """
    Build flattened sequence features for MLP: last `lookback` days of
    daily returns as a flat vector per row.
    """
    ret = df["returns_1d"].values if "returns_1d" in df.columns else df["close"].pct_change(1).values
    n = len(ret)
    seq = np.full((n, lookback), np.nan)
    for i in range(lookback, n):
        seq[i] = ret[i - lookback:i]
    return seq


# ── Model class ───────────────────────────────────────────────────────────────

class NiftyDailyPredictor:
    """
    Stacked ensemble for multi-horizon daily Nifty 50 prediction.

    Per horizon (5d, 10d, 20d, 30d):
      - XGBoost regressor (moderate regularization)
      - LightGBM regressor (if available)
      - Ridge regression (linear baseline)
      - MLPRegressor (sequence proxy via flattened 20-day returns)
      - Ridge meta-learner (stacking)

    Features: top-30 selected via mutual_info_regression.
    Training: 4 years train, 1 year validation, purged TimeSeriesSplit.
    Conformal prediction intervals per horizon.
    """

    def __init__(self):
        self.models = {}          # {horizon: {model_name: model}}
        self.meta_models = {}     # {horizon: Ridge}
        self.scalers = {}         # {horizon: RobustScaler}
        self.mlp_scalers = {}     # {horizon: RobustScaler for MLP}
        self.selected_features = {}  # {horizon: list of feature names}
        self.active_features = DAILY_FEATURE_COLS.copy()

        self._bias = {}           # {horizon: float}
        self._conformal_width = {}  # {horizon: float}
        self._meta_model_names = {}  # {horizon: list of model names}

        self.is_trained = False
        self.metrics = {}
        self.mi_scores = {}
        self.feature_importance = {}
        self._df = None           # cached DataFrame for predict()

    def _build_base_models(self, horizon: int) -> dict:
        """Create base model instances for a given horizon."""
        models = {
            "xgb": XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=4,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
                reg_alpha=0.5, reg_lambda=3.0, gamma=0.05,
                random_state=42 + horizon, tree_method="hist",
            ),
            "ridge": Ridge(alpha=100.0),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42 + horizon,
                learning_rate="adaptive",
                learning_rate_init=0.001,
            ),
        }

        if _LGBM_AVAILABLE:
            try:
                models["lgbm"] = lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.03, max_depth=4,
                    subsample=0.7, colsample_bytree=0.6,
                    min_child_weight=10, reg_alpha=0.5, reg_lambda=3.0,
                    random_state=77 + horizon, n_jobs=1, verbose=-1,
                )
            except Exception:
                pass

        return models

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Train multi-horizon models.
        If df is None, fetches and prepares daily data automatically.
        Returns dict of metrics per horizon.
        """
        if df is None:
            df = fetch_and_prepare_daily()

        self._df = df.copy()
        all_metrics = {}

        for horizon in HORIZONS:
            print(f"\n[daily_model] Training {horizon}d horizon...")
            metrics = self._train_horizon(df, horizon)
            all_metrics[f"{horizon}d"] = metrics
            print(f"[daily_model] {horizon}d -> MAE={metrics['mae']:.4f}, "
                  f"R2={metrics['r2']:.4f}, DirAcc={metrics['dir_acc']:.1f}%")

        self.is_trained = True
        self.metrics = all_metrics
        return all_metrics

    def _train_horizon(self, df: pd.DataFrame, horizon: int) -> dict:
        """Train all models for a single horizon."""
        target_col = f"target_{horizon}d"

        # Prepare feature matrix
        feat_cols = [c for c in DAILY_FEATURE_COLS if c in df.columns]
        required = feat_cols + [target_col, "close"]
        available = [c for c in required if c in df.columns]
        missing = set(required) - set(available)
        if missing:
            # Add missing columns as 0
            for col in missing:
                if col not in [target_col, "close"]:
                    df[col] = 0.0

        # Sentiment: fill NaN with 0 for training (model learns without them)
        sentiment_cols = ["news_sentiment", "reddit_sentiment",
                          "breadth_pct_above_ema50"]
        for col in sentiment_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        feat_cols = [c for c in DAILY_FEATURE_COLS if c in df.columns]

        # Drop rows where target or any critical feature is NaN
        clean = df[feat_cols + [target_col, "close"]].dropna(
            subset=[target_col]
        ).copy()

        # Fill remaining NaN in features with 0
        clean[feat_cols] = clean[feat_cols].fillna(0.0)

        if len(clean) < 200:
            raise ValueError(
                f"Not enough data for {horizon}d (need >= 300, got {len(clean)})"
            )

        # Drop constant features
        X_raw = clean[feat_cols].values
        X_raw, active_feats = _drop_constant_features(X_raw, feat_cols)
        self.active_features = active_feats

        y = clean[target_col].values
        prices = clean["close"].values

        # Feature selection via mutual information
        selected, mi_scores = _select_features_mi(
            X_raw, y, active_feats, k=MAX_FEATURES
        )
        self.selected_features[horizon] = selected
        self.mi_scores[horizon] = mi_scores

        sel_idx = [active_feats.index(f) for f in selected]
        X = X_raw[:, sel_idx]

        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[horizon] = scaler

        # Build MLP sequence features
        mlp_seq = _build_mlp_sequence_features(clean)
        # Mask: rows where sequence is fully available
        mlp_valid = ~np.isnan(mlp_seq).any(axis=1)
        mlp_scaler = RobustScaler()
        mlp_seq_clean = mlp_seq.copy()
        mlp_seq_clean[~mlp_valid] = 0.0
        mlp_seq_scaled = mlp_scaler.fit_transform(mlp_seq_clean)
        self.mlp_scalers[horizon] = mlp_scaler

        n = len(X_scaled)
        weights = _sample_weights(n)

        # Build base models
        base_models = self._build_base_models(horizon)
        model_names = list(base_models.keys())

        # ── Purged walk-forward CV ────────────────────────────────────────
        oof_preds = {name: np.full(n, np.nan) for name in model_names}
        all_errors = []
        dir_accs = []
        last_val_idx = None

        for train_idx, val_idx in _purged_ts_split(n, purge=PURGE_GAP):
            Xtr, Xval = X_scaled[train_idx], X_scaled[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            w_tr = weights[train_idx]

            val_preds_list = []

            for name, model in base_models.items():
                try:
                    if name == "mlp":
                        # MLP uses sequence features
                        Xtr_mlp = mlp_seq_scaled[train_idx]
                        Xval_mlp = mlp_seq_scaled[val_idx]
                        # Only train on rows with valid sequences
                        mlp_tr_valid = mlp_valid[train_idx]
                        if mlp_tr_valid.sum() < 50:
                            continue
                        model.fit(Xtr_mlp[mlp_tr_valid],
                                  ytr[mlp_tr_valid])
                        p = model.predict(Xval_mlp)
                    elif name in ("xgb",):
                        model.fit(Xtr, ytr, sample_weight=w_tr)
                        p = model.predict(Xval)
                    elif name == "lgbm":
                        model.fit(Xtr, ytr, sample_weight=w_tr)
                        p = model.predict(Xval)
                    else:
                        # Ridge etc.
                        model.fit(Xtr, ytr, sample_weight=w_tr)
                        p = model.predict(Xval)

                    oof_preds[name][val_idx] = p
                    val_preds_list.append(p)
                except Exception:
                    continue

            if val_preds_list:
                avg_pred = np.mean(val_preds_list, axis=0)
                all_errors.extend((avg_pred - yval).tolist())
                pred_dir = (avg_pred > 0).astype(int)
                true_dir = (yval > 0).astype(int)
                dir_accs.append(
                    float(np.mean(pred_dir == true_dir) * 100)
                )
                last_val_idx = val_idx

        # Bias correction
        all_errors = np.array(all_errors) if all_errors else np.array([0.0])
        bias = float(np.mean(all_errors))
        self._bias[horizon] = bias

        # ── Train meta-learner on OOF predictions ─────────────────────────
        valid_names = [k for k in model_names
                       if not np.all(np.isnan(oof_preds[k]))]
        oof_mask = np.ones(n, dtype=bool)
        for name in valid_names:
            oof_mask &= ~np.isnan(oof_preds[name])

        meta_model = Ridge(alpha=10.0)
        if oof_mask.sum() > 50:
            oof_matrix = np.column_stack(
                [oof_preds[name][oof_mask] for name in valid_names]
            )
            meta_model.fit(oof_matrix, y[oof_mask])
            self._meta_model_names[horizon] = valid_names
        else:
            self._meta_model_names[horizon] = None

        self.meta_models[horizon] = meta_model

        # ── Conformal prediction intervals ────────────────────────────────
        conformal_width = 0.05  # default 5%
        if last_val_idx is not None and len(last_val_idx) > 10:
            cal_preds = []
            for name in valid_names:
                vals = oof_preds[name][last_val_idx]
                if not np.all(np.isnan(vals)):
                    cal_preds.append(vals)
            if cal_preds:
                avg_cal = np.nanmean(cal_preds, axis=0)
                cal_resid = np.abs(avg_cal - y[last_val_idx])
                alpha = 0.10
                q_level = min(
                    np.ceil((len(cal_resid) + 1) * (1 - alpha))
                    / len(cal_resid),
                    1.0,
                )
                conformal_width = float(np.quantile(cal_resid, q_level))

        self._conformal_width[horizon] = conformal_width

        # ── Final fit on all data ─────────────────────────────────────────
        for name, model in base_models.items():
            try:
                if name == "mlp":
                    valid_mask = mlp_valid
                    if valid_mask.sum() >= 50:
                        model.fit(mlp_seq_scaled[valid_mask],
                                  y[valid_mask])
                elif name in ("xgb", "lgbm"):
                    model.fit(X_scaled, y, sample_weight=weights)
                else:
                    model.fit(X_scaled, y, sample_weight=weights)
            except Exception:
                continue

        self.models[horizon] = base_models

        # ── Feature importance ────────────────────────────────────────────
        try:
            imp = base_models["xgb"].feature_importances_
            self.feature_importance[horizon] = pd.Series(
                imp, index=selected
            ).sort_values(ascending=False)
        except Exception:
            self.feature_importance[horizon] = pd.Series(dtype=float)

        # ── Metrics ───────────────────────────────────────────────────────
        # Use last fold validation metrics
        val_mae = float(np.mean(np.abs(all_errors)))
        avg_dir_acc = float(np.mean(dir_accs)) if dir_accs else 50.0

        # In-sample R2 for sanity check
        all_preds_is = []
        for name, model in base_models.items():
            try:
                if name == "mlp":
                    all_preds_is.append(model.predict(mlp_seq_scaled))
                else:
                    all_preds_is.append(model.predict(X_scaled))
            except Exception:
                continue

        if all_preds_is:
            avg_is = np.mean(all_preds_is, axis=0) - bias
            r2 = r2_score(y, avg_is)
        else:
            r2 = 0.0

        return {
            "mae": round(val_mae, 6),
            "r2": round(r2, 4),
            "dir_acc": round(avg_dir_acc, 1),
            "bias": round(bias * 100, 4),
            "conformal_width_pct": round(conformal_width * 100, 2),
            "n_samples": n,
            "n_features": len(selected),
            "stacking": self._meta_model_names.get(horizon) is not None,
            "lgbm": "lgbm" in base_models,
        }

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame = None) -> dict | None:
        """
        Predict all 4 horizons. Returns:
        {
            "current_price": float,
            "5d":  {"price": ..., "pct_change": ..., "direction": ...,
                    "range_low": ..., "range_high": ..., "confidence": ...},
            "10d": {...},
            "20d": {...},
            "30d": {...},
        }
        """
        if not self.is_trained:
            return None

        if df is None:
            df = self._df
        if df is None:
            return None

        df = df.copy()

        # Fill sentiment for prediction (live values)
        sentiment_cols = ["news_sentiment", "reddit_sentiment",
                          "breadth_pct_above_ema50"]
        try:
            news = fetch_news_sentiment()
            df.loc[df.index[-1], "news_sentiment"] = news.get(
                "news_sentiment", 0.0
            )
        except Exception:
            pass
        try:
            reddit = fetch_reddit_sentiment()
            df.loc[df.index[-1], "reddit_sentiment"] = reddit.get(
                "reddit_sentiment", 0.0
            )
        except Exception:
            pass
        try:
            breadth = fetch_market_breadth()
            df.loc[df.index[-1], "breadth_pct_above_ema50"] = breadth.get(
                "breadth_pct_above_ema50", 0.5
            )
        except Exception:
            pass

        # Fill NaN in sentiment with 0 (neutral)
        for col in sentiment_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        current_price = float(df["close"].dropna().iloc[-1])
        result = {"current_price": round(current_price, 2)}

        for horizon in HORIZONS:
            pred = self._predict_horizon(df, horizon, current_price)
            result[f"{horizon}d"] = pred

        return result

    def _predict_horizon(self, df: pd.DataFrame, horizon: int,
                          current_price: float) -> dict:
        """Predict a single horizon."""
        if horizon not in self.models:
            return self._fallback_prediction(current_price)

        selected = self.selected_features.get(horizon, [])
        if not selected:
            return self._fallback_prediction(current_price)

        # Ensure all selected features exist
        for col in selected:
            if col not in df.columns:
                df[col] = 0.0

        clean = df[selected].fillna(0.0)
        if clean.empty:
            return self._fallback_prediction(current_price)

        latest = clean.iloc[[-1]]
        scaler = self.scalers.get(horizon)
        if scaler is None:
            return self._fallback_prediction(current_price)

        X_scaled = scaler.transform(latest.values)

        # MLP sequence features
        mlp_seq = _build_mlp_sequence_features(df)
        mlp_scaler = self.mlp_scalers.get(horizon)
        if mlp_scaler is not None:
            last_seq = mlp_seq[[-1]]
            last_seq = np.nan_to_num(last_seq, nan=0.0)
            mlp_scaled = mlp_scaler.transform(last_seq)
        else:
            mlp_scaled = np.zeros((1, MLP_LOOKBACK))

        # Base model predictions
        base_preds = {}
        base_models = self.models[horizon]
        for name, model in base_models.items():
            try:
                if name == "mlp":
                    base_preds[name] = float(model.predict(mlp_scaled)[0])
                else:
                    base_preds[name] = float(model.predict(X_scaled)[0])
            except Exception:
                continue

        if not base_preds:
            return self._fallback_prediction(current_price)

        # Meta-learner or simple average
        meta_names = self._meta_model_names.get(horizon)
        meta_model = self.meta_models.get(horizon)
        bias = self._bias.get(horizon, 0.0)

        if (meta_names and meta_model and
                all(n in base_preds for n in meta_names)):
            meta_in = np.array([[base_preds[n] for n in meta_names]])
            pred_return = float(meta_model.predict(meta_in)[0]) - bias
        else:
            pred_return = float(np.mean(list(base_preds.values()))) - bias

        # Predicted price
        pred_price = round(current_price * (1 + pred_return), 2)

        # Conformal interval
        cw = self._conformal_width.get(horizon, 0.05)
        range_low = round(current_price * (1 + pred_return - cw), 2)
        range_high = round(current_price * (1 + pred_return + cw), 2)

        # Direction + confidence
        direction = "UP" if pred_return > 0 else "DOWN"

        # Ensemble agreement
        n_up = sum(1 for v in base_preds.values() if v > 0)
        agreement = max(n_up, len(base_preds) - n_up) / len(base_preds)

        # Magnitude signal
        magnitude = min(
            abs(pred_return) / max(cw, 1e-6), 1.0
        )

        confidence = round(
            (0.5 * agreement + 0.3 * magnitude + 0.2 * 0.5) * 100, 1
        )

        return {
            "price": pred_price,
            "pct_change": round(pred_return * 100, 4),
            "direction": direction,
            "range_low": range_low,
            "range_high": range_high,
            "confidence": confidence,
        }

    @staticmethod
    def _fallback_prediction(current_price: float) -> dict:
        return {
            "price": round(current_price, 2),
            "pct_change": 0.0,
            "direction": "HOLD",
            "range_low": round(current_price * 0.95, 2),
            "range_high": round(current_price * 1.05, 2),
            "confidence": 0.0,
        }

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str = "daily_model.pkl") -> None:
        """Save model to disk."""
        # Don't save the cached DataFrame (too large)
        df_backup = self._df
        self._df = None
        joblib.dump(self, path)
        self._df = df_backup
        print(f"[daily_model] Saved to {path}")

    @classmethod
    def load(cls, path: str = "daily_model.pkl") -> "NiftyDailyPredictor":
        """Load model from disk."""
        model = joblib.load(path)
        print(f"[daily_model] Loaded from {path}")
        return model

    def save_bytes(self) -> bytes:
        """Serialize to bytes."""
        df_backup = self._df
        self._df = None
        buf = io.BytesIO()
        joblib.dump(self, buf)
        self._df = df_backup
        return buf.getvalue()

    @classmethod
    def load_bytes(cls, data: bytes) -> "NiftyDailyPredictor":
        """Deserialize from bytes."""
        buf = io.BytesIO(data)
        return joblib.load(buf)
