import pandas as pd
import numpy as np
import ta
import warnings

warnings.filterwarnings("ignore")

MLP_BASE_FEATURES = ["returns_1", "rsi14", "macd_hist", "vol_ratio", "price_vs_vwap"]
MLP_N_LAGS = 10
MLP_LAG_COLS = [f"{f}_lag{l}" for f in MLP_BASE_FEATURES for l in range(1, MLP_N_LAGS + 1)]


def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
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


def _session_label(hour: int, minute: int) -> str:
    if hour == 9 or (hour == 10 and minute == 0):
        return "open_rush"
    if hour == 15:
        return "close_rush"
    return "midday"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Price-derived ──────────────────────────────────────────────────────
    df["returns_1"]  = df["close"].pct_change(1)
    df["returns_2"]  = df["close"].pct_change(2)
    df["returns_3"]  = df["close"].pct_change(3)
    df["returns_5"]  = df["close"].pct_change(5)
    df["returns_10"] = df["close"].pct_change(10)
    df["accel_1"]    = df["returns_1"] - df["returns_2"]
    df["accel_3"]    = df["returns_3"] - df["returns_5"]
    df["hl_pct"]     = (df["high"] - df["low"]) / df["close"]
    df["body_size"]  = abs(df["close"] - df["open"]) / df["close"]

    # ── Volume ────────────────────────────────────────────────────────────
    has_volume = df["volume"].sum() > 0
    df["vol_sma20"]   = df["volume"].rolling(20).mean() if has_volume else 0.0
    df["vol_ratio"]   = (df["volume"] / df["vol_sma20"].replace(0, np.nan)).fillna(1.0) if has_volume else 1.0
    df["vol_returns"] = df["volume"].pct_change(1).fillna(0.0) if has_volume else 0.0

    # ── Moving averages ───────────────────────────────────────────────────
    df["ema9"]  = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["price_vs_ema9"]    = (df["close"] - df["ema9"])  / df["ema9"]
    df["price_vs_ema21"]   = (df["close"] - df["ema21"]) / df["ema21"]
    df["price_vs_ema50"]   = (df["close"] - df["ema50"]) / df["ema50"]
    df["ema9_vs_ema21"]    = (df["ema9"] - df["ema21"])  / df["ema21"]
    df["ema9_cross_ema21"] = (df["ema9"] > df["ema21"]).astype(int)

    # ── RSI ───────────────────────────────────────────────────────────────
    df["rsi14"]     = ta.momentum.rsi(df["close"], window=14)
    df["rsi7"]      = ta.momentum.rsi(df["close"], window=7)
    df["rsi_diverge"] = df["rsi14"] - df["rsi7"]

    # ── MACD ──────────────────────────────────────────────────────────────
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_pct_b"] = bb.bollinger_pband()

    # ── ATR ───────────────────────────────────────────────────────────────
    df["atr14"]   = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["atr_pct"] = df["atr14"] / df["close"]

    # ── ADX ───────────────────────────────────────────────────────────────
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"]      = adx.adx()
    df["adx_pos"]  = adx.adx_pos()
    df["adx_neg"]  = adx.adx_neg()
    df["di_diff"]  = df["adx_pos"] - df["adx_neg"]
    df["trending"] = (df["adx"] > 25).astype(int)

    # ── Stochastic ────────────────────────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── OBV ───────────────────────────────────────────────────────────────
    if has_volume:
        df["obv"]       = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["obv_sma20"] = df["obv"].rolling(20).mean()
        df["obv_ratio"] = (df["obv"] / df["obv_sma20"].replace(0, np.nan)).fillna(1.0)
    else:
        df["obv_ratio"] = 1.0

    # ── VWAP ──────────────────────────────────────────────────────────────
    if has_volume:
        df = _add_vwap(df)
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
    else:
        df["price_vs_vwap"] = 0.0

    # ── Rolling stats ─────────────────────────────────────────────────────
    df["rolling_std10"]  = df["close"].rolling(10).std() / df["close"]
    df["range10_pct"]    = (
        (df["close"].rolling(10).max() - df["close"].rolling(10).min())
        / df["close"].rolling(10).min()
    )
    df["ret_autocorr5"] = (
        df["returns_1"].rolling(10)
        .apply(lambda x: x.autocorr(lag=1) if x.std() > 0 else 0, raw=False)
        .fillna(0)
    )

    # ── Previous-day levels ───────────────────────────────────────────────
    daily = df[["high", "low", "close"]].resample("1D").agg(
        {"high": "max", "low": "min", "close": "last"}
    ).rename(columns={"high": "d_high", "low": "d_low", "close": "d_close"})
    daily = daily[daily["d_close"].notna()]
    prev_day = daily.shift(1)
    df_date = df.index.normalize().tz_localize(None) if df.index.tzinfo else df.index.normalize()
    prev_day.index = prev_day.index.tz_localize(None) if prev_day.index.tzinfo else prev_day.index
    df["prev_d_high"]  = df_date.map(prev_day["d_high"])
    df["prev_d_low"]   = df_date.map(prev_day["d_low"])
    df["prev_d_close"] = df_date.map(prev_day["d_close"])
    df["dist_prev_high"]  = (df["close"] - df["prev_d_high"])  / df["prev_d_high"]
    df["dist_prev_low"]   = (df["close"] - df["prev_d_low"])   / df["prev_d_low"]
    df["dist_prev_close"] = (df["close"] - df["prev_d_close"]) / df["prev_d_close"]

    # ── Time features ─────────────────────────────────────────────────────
    total_min = (15 * 60 + 30) - (9 * 60 + 15)
    elapsed   = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 15)
    df["session_progress"] = np.clip(elapsed / total_min, 0, 1)
    df["hour_sin"]         = np.sin(2 * np.pi * df.index.hour / 24)
    df["is_open_rush"]     = (
        (df.index.hour == 9) | ((df.index.hour == 10) & (df.index.minute == 0))
    ).astype(int)
    df["is_close_rush"] = (df.index.hour == 15).astype(int)
    df["is_lunch"]      = (
        ((df.index.hour == 12) & (df.index.minute >= 30)) |
        ((df.index.hour == 13) & (df.index.minute <= 30))
    ).astype(int)

    # ── Session label (for time-of-day models) ────────────────────────────
    df["session"] = [
        _session_label(h, m) for h, m in zip(df.index.hour, df.index.minute)
    ]

    # ── Targets ───────────────────────────────────────────────────────────
    df["target_return"]    = df["close"].pct_change(1).shift(-1)
    df["target_price"]     = df["close"].shift(-1)
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    df["target_return_3"]  = df["close"].pct_change(3).shift(-3)   # 30-min ahead
    df["target_price_3"]   = df["close"].shift(-3)

    return df


def engineer_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features for MLP: top 5 features × 10 lags = 50 columns."""
    df = df.copy()
    for feat in MLP_BASE_FEATURES:
        if feat in df.columns:
            for lag in range(1, MLP_N_LAGS + 1):
                df[f"{feat}_lag{lag}"] = df[feat].shift(lag)
    return df


FEATURE_COLS = [
    # Price momentum
    "returns_1", "returns_2", "returns_3", "returns_5", "returns_10",
    "accel_1", "accel_3", "hl_pct", "body_size",
    # Volume
    "vol_ratio", "vol_returns",
    # Moving averages
    "price_vs_ema9", "price_vs_ema21", "price_vs_ema50",
    "ema9_vs_ema21", "ema9_cross_ema21",
    # Oscillators
    "rsi14", "rsi7", "rsi_diverge",
    "macd", "macd_sig", "macd_hist", "bb_pct_b",
    # Volatility / Regime
    "atr_pct", "adx", "di_diff", "trending",
    # Stochastic
    "stoch_k", "stoch_d",
    # OBV + VWAP
    "obv_ratio", "price_vs_vwap",
    # Rolling stats
    "rolling_std10", "range10_pct", "ret_autocorr5",
    # Previous-day levels
    "dist_prev_high", "dist_prev_low", "dist_prev_close",
    # Time
    "session_progress", "hour_sin",
    "is_open_rush", "is_close_rush", "is_lunch",
    # Global market context
    "sp500_change", "nasdaq_change", "dow_change",
    "india_vix", "india_vix_change",
    "vix_level", "vix_change",
    "nikkei_change", "hangseng_change",
    "usdinr", "usdinr_change",
    "gold_change", "crude_change",
    "news_sentiment", "news_count",
    # Institutional flows (FII/DII)
    "fii_net", "dii_net", "fii_dii_ratio",
    # Options market (PCR)
    "pcr", "pcr_signal",
    # Social sentiment
    "reddit_sentiment", "reddit_count",
    # Market breadth
    "breadth_pct_above_ema50",
]
