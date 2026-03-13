import pandas as pd
import numpy as np
import ta
import warnings

warnings.filterwarnings("ignore")


def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-reset VWAP."""
    df = df.copy()
    df["date"] = df.index.date
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_tpv"] = df.groupby("date").apply(
        lambda g: (g["tp"] * g["volume"]).cumsum()
    ).values
    df["cum_vol"] = df.groupby("date")["volume"].cumsum().values
    df["vwap"] = df["cum_tpv"] / df["cum_vol"].replace(0, np.nan)
    df.drop(columns=["date", "tp", "cum_tpv", "cum_vol"], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Price-derived features ─────────────────────────────────────────────
    df["returns_1"]  = df["close"].pct_change(1)
    df["returns_2"]  = df["close"].pct_change(2)
    df["returns_3"]  = df["close"].pct_change(3)
    df["returns_5"]  = df["close"].pct_change(5)
    df["returns_10"] = df["close"].pct_change(10)
    df["hl_pct"]     = (df["high"] - df["low"]) / df["close"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
    df["body_size"]    = abs(df["close"] - df["open"]) / df["close"]

    # ── Volume features ────────────────────────────────────────────────────
    df["vol_sma20"]   = df["volume"].rolling(20).mean()
    df["vol_ratio"]   = df["volume"] / df["vol_sma20"].replace(0, np.nan)
    df["vol_returns"] = df["volume"].pct_change(1)

    # ── Moving averages ────────────────────────────────────────────────────
    df["ema9"]  = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["sma20"] = df["close"].rolling(20).mean()

    df["price_vs_ema9"]  = (df["close"] - df["ema9"])  / df["ema9"]
    df["price_vs_ema21"] = (df["close"] - df["ema21"]) / df["ema21"]
    df["price_vs_ema50"] = (df["close"] - df["ema50"]) / df["ema50"]
    df["ema9_vs_ema21"]  = (df["ema9"] - df["ema21"])  / df["ema21"]
    df["ema9_cross_ema21"] = (df["ema9"] > df["ema21"]).astype(int)

    # ── RSI ────────────────────────────────────────────────────────────────
    df["rsi14"] = ta.momentum.rsi(df["close"], window=14)
    df["rsi7"]  = ta.momentum.rsi(df["close"], window=7)
    df["rsi_diff"] = df["rsi14"].diff(1)

    # ── MACD ───────────────────────────────────────────────────────────────
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_cross"] = (df["macd"] > df["macd_sig"]).astype(int)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_pct_b"]  = bb.bollinger_pband()   # 0-1: position within bands
    df["bb_width"]  = bb.bollinger_wband()   # band width

    # ── ATR ────────────────────────────────────────────────────────────────
    df["atr14"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )
    df["atr_pct"] = df["atr14"] / df["close"]   # normalised

    # ── Stochastic ─────────────────────────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]

    # ── OBV ────────────────────────────────────────────────────────────────
    df["obv"]      = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["obv_sma20"] = df["obv"].rolling(20).mean()
    df["obv_ratio"] = df["obv"] / df["obv_sma20"].replace(0, np.nan)

    # ── VWAP ───────────────────────────────────────────────────────────────
    df = _add_vwap(df)
    df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

    # ── Supertrend (simplified) ────────────────────────────────────────────
    # Uses ATR multiplier 3 on period 10
    _hl2 = (df["high"] + df["low"]) / 2
    _atr10 = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=10)
    df["supertrend_upper"] = _hl2 + 3 * _atr10
    df["supertrend_lower"] = _hl2 - 3 * _atr10
    df["above_supertrend"] = (df["close"] > df["supertrend_lower"]).astype(int)

    # ── Previous-day high / low / close ────────────────────────────────────
    daily = df[["high", "low", "close"]].resample("1D").agg(
        {"high": "max", "low": "min", "close": "last"}
    ).rename(columns={"high": "d_high", "low": "d_low", "close": "d_close"})
    daily = daily[daily["d_close"].notna()]
    prev_day = daily.shift(1)

    # Merge by normalised date
    df_date = df.index.normalize().tz_localize(None) if df.index.tzinfo else df.index.normalize()
    prev_day.index = prev_day.index.tz_localize(None) if prev_day.index.tzinfo else prev_day.index

    df["prev_d_high"]  = df_date.map(prev_day["d_high"])
    df["prev_d_low"]   = df_date.map(prev_day["d_low"])
    df["prev_d_close"] = df_date.map(prev_day["d_close"])

    df["dist_prev_high"]  = (df["close"] - df["prev_d_high"])  / df["prev_d_high"]
    df["dist_prev_low"]   = (df["close"] - df["prev_d_low"])   / df["prev_d_low"]
    df["dist_prev_close"] = (df["close"] - df["prev_d_close"]) / df["prev_d_close"]

    # ── Time features (cyclical encoding) ─────────────────────────────────
    total_minutes = (15 * 60 + 30) - (9 * 60 + 15)   # 375 min session
    elapsed = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 15)
    df["session_progress"] = np.clip(elapsed / total_minutes, 0, 1)

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 5)

    df["is_open_rush"]  = (
        (df.index.hour == 9)  | ((df.index.hour == 10) & (df.index.minute == 0))
    ).astype(int)
    df["is_close_rush"] = (df.index.hour == 15).astype(int)
    df["is_lunch"]      = (
        (df.index.hour == 12) & (df.index.minute >= 30) |
        (df.index.hour == 13) & (df.index.minute <= 30)
    ).astype(int)

    # ── Rolling price statistics ───────────────────────────────────────────
    df["rolling_std10"]  = df["close"].rolling(10).std() / df["close"]
    df["rolling_max10"]  = df["close"].rolling(10).max()
    df["rolling_min10"]  = df["close"].rolling(10).min()
    df["range10_pct"]    = (df["rolling_max10"] - df["rolling_min10"]) / df["rolling_min10"]

    # ── Target ────────────────────────────────────────────────────────────
    df["target_price"]     = df["close"].shift(-1)
    df["target_direction"] = (df["target_price"] > df["close"]).astype(int)
    df["target_return"]    = (df["target_price"] - df["close"]) / df["close"]

    return df


FEATURE_COLS = [
    "returns_1", "returns_2", "returns_3", "returns_5", "returns_10",
    "hl_pct", "upper_shadow", "lower_shadow", "body_size",
    "vol_ratio", "vol_returns",
    "price_vs_ema9", "price_vs_ema21", "price_vs_ema50", "ema9_vs_ema21", "ema9_cross_ema21",
    "rsi14", "rsi7", "rsi_diff",
    "macd", "macd_sig", "macd_hist", "macd_cross",
    "bb_pct_b", "bb_width",
    "atr_pct",
    "stoch_k", "stoch_d", "stoch_diff",
    "obv_ratio",
    "price_vs_vwap",
    "above_supertrend",
    "dist_prev_high", "dist_prev_low", "dist_prev_close",
    "session_progress",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_open_rush", "is_close_rush", "is_lunch",
    "rolling_std10", "range10_pct",
]
