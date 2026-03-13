"""
Daily-resolution feature engineering for Nifty 50 multi-horizon prediction.
Fetches 5 years of daily OHLCV and computes momentum, mean-reversion,
volatility-regime, volume, trend-strength, seasonality, and global-market
features.  Targets are direct multi-step (5/10/20/30 day).
"""

import pandas as pd
import numpy as np
import ta
import yfinance as yf
import warnings
from datetime import datetime
import calendar

from src.sentiment import (
    GLOBAL_TICKERS,
    fetch_news_sentiment,
    fetch_reddit_sentiment,
    fetch_market_breadth,
)

warnings.filterwarnings("ignore")

NIFTY_SYMBOL = "^NSEI"

# ── Feature column list (exported) ────────────────────────────────────────────
DAILY_FEATURE_COLS = [
    # Momentum (multi-horizon)
    "returns_1d", "returns_5d", "returns_10d", "returns_20d",
    "returns_60d", "returns_120d", "momentum_ratio_5_20",
    # Moving averages
    "price_vs_sma50", "price_vs_sma200", "sma50_vs_sma200", "golden_cross",
    # Mean reversion
    "rsi14_daily", "rsi7_daily", "bb_pct_b_daily",
    "distance_from_52w_high", "distance_from_52w_low",
    # Volatility regime
    "realized_vol_10d", "realized_vol_30d", "realized_vol_60d",
    "vol_ratio_10_30", "vol_ratio_10_60", "garman_klass_vol",
    # Volume
    "volume_sma20_ratio", "volume_trend_10d",
    # Trend strength
    "adx_daily", "di_diff_daily", "efficiency_ratio_20d",
    # MACD daily
    "macd_daily", "macd_signal_daily", "macd_hist_daily",
    # Seasonality
    "month_sin", "month_cos", "is_march", "is_oct_nov",
    "day_of_week", "is_expiry_week",
    # Global market context (historical daily)
    "sp500_ret_5d", "sp500_ret_20d",
    "vix_level", "vix_change_5d",
    "india_vix_level", "india_vix_change_5d",
    "crude_ret_5d", "gold_ret_5d",
    "usdinr_level", "usdinr_ret_5d",
    "dxy_change",
    # Sentiment (point-in-time; NaN during training, filled at prediction)
    "news_sentiment", "reddit_sentiment", "breadth_pct_above_ema50",
]

# Target columns
TARGET_COLS = [
    "target_5d", "target_10d", "target_20d", "target_30d",
    "target_dir_5d", "target_dir_10d", "target_dir_20d", "target_dir_30d",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _garman_klass_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Garman-Klass volatility estimator using OHLC data.
    More efficient than close-to-close volatility.
    """
    log_hl = (np.log(df["high"] / df["low"])) ** 2
    log_co = (np.log(df["close"] / df["open"])) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window).mean().apply(np.sqrt)


def _is_expiry_week(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Determine if a date falls in the expiry week (week containing
    the last Thursday of the month — NSE options expiry).
    """
    result = []
    for dt in dates:
        year, month = dt.year, dt.month
        # Find last Thursday of this month
        last_day = calendar.monthrange(year, month)[1]
        last_date = datetime(year, month, last_day)
        # weekday(): Monday=0 .. Sunday=6; Thursday=3
        offset = (last_date.weekday() - 3) % 7
        last_thursday = last_day - offset
        # Expiry week: Monday through Friday of that week
        expiry_monday = last_thursday - (3)  # Thursday - 3 = Monday
        expiry_friday = last_thursday + 1     # Thursday + 1 = Friday
        result.append(1 if expiry_monday <= dt.day <= expiry_friday else 0)
    return pd.Series(result, index=dates)


def _kaufman_efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Kaufman efficiency ratio: |direction| / volatility.
    1 = perfectly trending, 0 = perfectly choppy.
    """
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (direction / volatility.replace(0, np.nan)).fillna(0.5)


# ── Global market historical features ────────────────────────────────────────

def _fetch_global_daily(period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical daily data for global tickers and compute
    multi-day returns and level features.
    """
    # Tickers we care about for daily model
    ticker_map = {
        "sp500":     "^GSPC",
        "vix":       "^VIX",
        "india_vix": "^INDIAVIX",
        "crude":     "CL=F",
        "gold":      "GC=F",
        "usdinr":    "INR=X",
        "dxy":       "DX-Y.NYB",
    }

    symbols = list(ticker_map.values())
    try:
        raw = yf.download(
            symbols, period=period, interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=raw.index)
    lvl0 = raw.columns.get_level_values(0)

    for name, sym in ticker_map.items():
        try:
            if sym not in lvl0:
                continue
            close = raw[sym]["Close"]

            if name == "sp500":
                result["sp500_ret_5d"] = close.pct_change(5)
                result["sp500_ret_20d"] = close.pct_change(20)
            elif name == "vix":
                result["vix_level"] = close
                result["vix_change_5d"] = close.pct_change(5)
            elif name == "india_vix":
                result["india_vix_level"] = close
                result["india_vix_change_5d"] = close.pct_change(5)
            elif name == "crude":
                result["crude_ret_5d"] = close.pct_change(5)
            elif name == "gold":
                result["gold_ret_5d"] = close.pct_change(5)
            elif name == "usdinr":
                result["usdinr_level"] = close
                result["usdinr_ret_5d"] = close.pct_change(5)
            elif name == "dxy":
                result["dxy_change"] = close.pct_change(5)
        except Exception:
            pass

    # Normalize index
    if result.index.tzinfo is not None:
        result.index = result.index.tz_localize(None)
    result.index = result.index.normalize()

    # Shift by 1 day to avoid look-ahead bias
    result = result.shift(1)
    return result


# ── Core feature engineering ──────────────────────────────────────────────────

def engineer_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all daily features from OHLCV DataFrame.
    Input: DataFrame with columns [open, high, low, close, volume].
    Output: DataFrame with all DAILY_FEATURE_COLS + TARGET_COLS added.
    """
    df = df.copy()
    close = df["close"]

    # ── Momentum (multi-horizon) ──────────────────────────────────────────
    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_10d"] = close.pct_change(10)
    df["returns_20d"] = close.pct_change(20)
    df["returns_60d"] = close.pct_change(60)
    df["returns_120d"] = close.pct_change(120)
    df["momentum_ratio_5_20"] = (
        df["returns_5d"] / df["returns_20d"].replace(0, np.nan)
    ).fillna(0.0)

    # ── Moving averages ───────────────────────────────────────────────────
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    df["price_vs_sma50"] = (close - sma50) / sma50
    df["price_vs_sma200"] = (close - sma200) / sma200
    df["sma50_vs_sma200"] = (sma50 - sma200) / sma200
    df["golden_cross"] = (sma50 > sma200).astype(int)

    # ── Mean reversion ────────────────────────────────────────────────────
    df["rsi14_daily"] = ta.momentum.rsi(close, window=14)
    df["rsi7_daily"] = ta.momentum.rsi(close, window=7)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct_b_daily"] = bb.bollinger_pband()

    high_52w = close.rolling(252, min_periods=200).max()
    low_52w = close.rolling(252, min_periods=200).min()
    df["distance_from_52w_high"] = (close - high_52w) / high_52w
    df["distance_from_52w_low"] = (close - low_52w) / low_52w

    # ── Volatility regime ─────────────────────────────────────────────────
    daily_ret = close.pct_change(1)
    df["realized_vol_10d"] = daily_ret.rolling(10).std() * np.sqrt(252)
    df["realized_vol_30d"] = daily_ret.rolling(30).std() * np.sqrt(252)
    df["realized_vol_60d"] = daily_ret.rolling(60).std() * np.sqrt(252)
    df["vol_ratio_10_30"] = (
        df["realized_vol_10d"] / df["realized_vol_30d"].replace(0, np.nan)
    ).fillna(1.0)
    df["vol_ratio_10_60"] = (
        df["realized_vol_10d"] / df["realized_vol_60d"].replace(0, np.nan)
    ).fillna(1.0)
    df["garman_klass_vol"] = _garman_klass_vol(df, window=20)

    # ── Volume ────────────────────────────────────────────────────────────
    vol = df["volume"]
    vol_sma20 = vol.rolling(20).mean()
    df["volume_sma20_ratio"] = (vol / vol_sma20.replace(0, np.nan)).fillna(1.0)

    # Volume trend: slope of volume over 10 days (normalized)
    def _vol_slope(x):
        if len(x) < 2 or x.std() == 0:
            return 0.0
        t = np.arange(len(x))
        slope = np.polyfit(t, x, 1)[0]
        return slope / (x.mean() + 1e-10)

    df["volume_trend_10d"] = vol.rolling(10).apply(_vol_slope, raw=True).fillna(0.0)

    # ── Trend strength ────────────────────────────────────────────────────
    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], close, window=14)
    df["adx_daily"] = adx_ind.adx()
    df["di_diff_daily"] = adx_ind.adx_pos() - adx_ind.adx_neg()
    df["efficiency_ratio_20d"] = _kaufman_efficiency_ratio(close, 20)

    # ── MACD daily ────────────────────────────────────────────────────────
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_daily"] = macd_ind.macd()
    df["macd_signal_daily"] = macd_ind.macd_signal()
    df["macd_hist_daily"] = macd_ind.macd_diff()

    # ── Seasonality ───────────────────────────────────────────────────────
    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_march"] = (month == 3).astype(int)
    df["is_oct_nov"] = ((month == 10) | (month == 11)).astype(int)
    df["day_of_week"] = df.index.dayofweek  # 0=Monday, 4=Friday
    df["is_expiry_week"] = _is_expiry_week(df.index)

    # ── Sentiment (point-in-time: NaN for historical rows) ────────────────
    df["news_sentiment"] = np.nan
    df["reddit_sentiment"] = np.nan
    df["breadth_pct_above_ema50"] = np.nan

    # ── Targets (direct multi-step, NOT recursive) ────────────────────────
    df["target_5d"] = close.pct_change(5).shift(-5)
    df["target_10d"] = close.pct_change(10).shift(-10)
    df["target_20d"] = close.pct_change(20).shift(-20)
    df["target_30d"] = close.pct_change(30).shift(-30)
    df["target_dir_5d"] = (df["target_5d"] > 0).astype(int)
    df["target_dir_10d"] = (df["target_10d"] > 0).astype(int)
    df["target_dir_20d"] = (df["target_20d"] > 0).astype(int)
    df["target_dir_30d"] = (df["target_30d"] > 0).astype(int)

    return df


# ── Master fetch + prepare ────────────────────────────────────────────────────

def fetch_and_prepare_daily(symbol: str = NIFTY_SYMBOL,
                            period: str = "2y") -> pd.DataFrame:
    """
    End-to-end pipeline:
    1. Fetch 5 years of daily OHLCV for Nifty 50.
    2. Fetch global market historical data.
    3. Engineer all features.
    4. Merge global context.
    5. Fill live sentiment into the last row.
    Returns a fully-featured DataFrame ready for training/prediction.
    """
    # 1. Fetch Nifty OHLCV
    print(f"[daily_features] Fetching {period} daily OHLCV for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", auto_adjust=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch daily OHLCV for {symbol}: {e}")

    if df.empty:
        raise RuntimeError(f"No daily data returned for {symbol}")

    # Normalize index
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df.index = df.index.normalize()
    if df.index.tzinfo is not None:
        df.index = df.index.tz_localize(None)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna(subset=["close"])
    print(f"[daily_features] Got {len(df)} daily bars")

    # 2. Engineer features
    print("[daily_features] Engineering features...")
    df = engineer_daily_features(df)

    # 3. Fetch and merge global market context
    print("[daily_features] Fetching global market data...")
    global_df = _fetch_global_daily(period=period)

    if not global_df.empty:
        # Normalize global index
        if global_df.index.tzinfo is not None:
            global_df.index = global_df.index.tz_localize(None)
        global_df.index = global_df.index.normalize()

        # Merge by date
        for col in global_df.columns:
            if col in df.columns:
                # Already created placeholder; overwrite
                df[col] = np.nan
            mapping = dict(zip(global_df.index, global_df[col]))
            df[col] = df.index.map(mapping)
            df[col] = df[col].ffill()

    # Fill missing global columns with defaults
    _global_defaults = {
        "sp500_ret_5d": 0.0, "sp500_ret_20d": 0.0,
        "vix_level": 20.0, "vix_change_5d": 0.0,
        "india_vix_level": 15.0, "india_vix_change_5d": 0.0,
        "crude_ret_5d": 0.0, "gold_ret_5d": 0.0,
        "usdinr_level": 84.0, "usdinr_ret_5d": 0.0,
        "dxy_change": 0.0,
    }
    for col, default in _global_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # 4. Fill live sentiment into the last row only
    print("[daily_features] Fetching live sentiment...")
    try:
        news = fetch_news_sentiment()
        df.loc[df.index[-1], "news_sentiment"] = news.get("news_sentiment", 0.0)
    except Exception:
        pass

    try:
        reddit = fetch_reddit_sentiment()
        df.loc[df.index[-1], "reddit_sentiment"] = reddit.get("reddit_sentiment", 0.0)
    except Exception:
        pass

    try:
        breadth = fetch_market_breadth()
        df.loc[df.index[-1], "breadth_pct_above_ema50"] = breadth.get(
            "breadth_pct_above_ema50", 0.5
        )
    except Exception:
        pass

    print(f"[daily_features] Final dataset: {len(df)} rows, "
          f"{len(DAILY_FEATURE_COLS)} features")
    return df
