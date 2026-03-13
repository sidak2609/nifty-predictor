import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")


def fetch_ohlcv(symbol: str, days: int = 55) -> pd.DataFrame:
    """
    Fetch 5-min OHLCV from yfinance and resample to 10-min bars.
    Returns DataFrame indexed by IST datetime.
    """
    end = datetime.now(IST)
    start = end - timedelta(days=days)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="5m",
            auto_adjust=True,
        )
    except Exception as e:
        raise RuntimeError(f"yfinance fetch failed for {symbol}: {e}")

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    # Normalize timezone to IST
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)

    df.index.name = "datetime"
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]

    # Filter to NSE market hours only (9:15 – 15:30 IST)
    df = df.between_time("09:15", "15:30")

    # Resample to 10-min bars
    df = (
        df.resample("10min", closed="left", label="left")
        .agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        })
        .dropna()
    )

    # Drop candles with zero volume (pre-market artefacts)
    df = df[df["volume"] > 0]

    return df


def get_current_price(symbol: str) -> float | None:
    """Return latest close price."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        return round(float(info.last_price), 2)
    except Exception:
        return None


def is_market_open() -> bool:
    """Check if NSE is currently open."""
    now = datetime.now(IST)
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_market_status() -> dict:
    now = datetime.now(IST)
    open_flag = is_market_open()
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if open_flag:
        remaining = market_close - now
        hours, rem = divmod(int(remaining.total_seconds()), 3600)
        minutes = rem // 60
        status_text = f"Open  •  closes in {hours}h {minutes}m"
    else:
        if now < market_open and now.weekday() < 5:
            diff = market_open - now
            minutes = int(diff.total_seconds() // 60)
            status_text = f"Closed  •  opens in {minutes}m"
        else:
            status_text = "Closed  •  opens Monday 9:15 AM IST" if now.weekday() >= 4 else "Closed  •  opens tomorrow 9:15 AM IST"

    return {"is_open": open_flag, "text": status_text, "time": now.strftime("%H:%M:%S IST")}
