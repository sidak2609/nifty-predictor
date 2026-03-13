"""
External signal module — adds global market context and news sentiment.
All sources are 100% free, no API keys required.
"""

import requests
import xml.etree.ElementTree as ET
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re
import warnings

warnings.filterwarnings("ignore")

IST = pytz.timezone("Asia/Kolkata")

# ── Sentiment keyword dictionaries (financial domain) ──────────────────────
_POS = {
    "surge", "rally", "gain", "gains", "rise", "rises", "bull", "bullish",
    "positive", "strong", "strength", "buy", "boost", "jump", "soar",
    "recover", "recovery", "growth", "profit", "beat", "exceed", "record",
    "peak", "outperform", "upgrade", "upside", "optimism", "optimistic",
    "high", "higher", "uptrend", "breakout", "support", "advance",
}
_NEG = {
    "fall", "falls", "drop", "drops", "crash", "decline", "declines",
    "bear", "bearish", "negative", "weak", "weakness", "sell", "loss",
    "losses", "plunge", "slump", "tumble", "concern", "worry", "risk",
    "fear", "underperform", "downgrade", "downside", "pressure", "tension",
    "low", "lower", "downtrend", "breakdown", "resistance", "retreat",
    "inflation", "recession", "slowdown", "crisis",
}

# ── Global market tickers ──────────────────────────────────────────────────
GLOBAL_TICKERS = {
    "sp500":      "^GSPC",      # S&P 500
    "nasdaq":     "^IXIC",      # Nasdaq
    "dow":        "^DJI",       # Dow Jones
    "vix":        "^VIX",       # CBOE VIX (global fear)
    "india_vix":  "^INDIAVIX",  # India VIX
    "nikkei":     "^N225",      # Japan (Asian proxy)
    "hangseng":   "^HSI",       # Hong Kong
    "usdinr":     "INR=X",      # USD/INR (inverse: higher = INR weaker)
    "gold":       "GC=F",       # Gold futures
    "crude":      "CL=F",       # Crude oil (WTI)
    "sgx_nifty":  "^NSEI",      # Gift/SGX Nifty proxy (spot)
}


def _safe_pct_change(series: pd.Series) -> float:
    """Return latest 1-day % change, or 0.0 on error."""
    try:
        s = series.dropna()
        if len(s) < 2:
            return 0.0
        return float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100)
    except Exception:
        return 0.0


def _safe_last(series: pd.Series) -> float:
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return 0.0


def fetch_global_markets() -> dict:
    """
    Fetch latest daily close for global market proxies.
    Returns a dict of feature_name → value.
    """
    features = {}
    try:
        symbols = list(GLOBAL_TICKERS.values())
        raw = yf.download(
            symbols,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        for name, sym in GLOBAL_TICKERS.items():
            try:
                if len(symbols) == 1:
                    close = raw["Close"]
                else:
                    close = raw[sym]["Close"] if sym in raw.columns.get_level_values(0) else pd.Series()

                if name == "india_vix":
                    features["india_vix"]        = _safe_last(close)
                    features["india_vix_change"]  = _safe_pct_change(close)
                elif name == "vix":
                    features["vix_level"]  = _safe_last(close)
                    features["vix_change"] = _safe_pct_change(close)
                elif name == "usdinr":
                    features["usdinr"]        = _safe_last(close)
                    features["usdinr_change"] = _safe_pct_change(close)
                else:
                    features[f"{name}_change"] = _safe_pct_change(close)
            except Exception:
                features[f"{name}_change"] = 0.0

    except Exception:
        # Fallback: all zeros
        for name in GLOBAL_TICKERS:
            features[f"{name}_change"] = 0.0
        features.update({"india_vix": 15.0, "india_vix_change": 0.0,
                         "vix_level": 20.0, "vix_change": 0.0,
                         "usdinr": 84.0, "usdinr_change": 0.0})

    return features


def fetch_news_sentiment(query: str = "Nifty India stock market") -> dict:
    """
    Fetch Google News RSS and score sentiment using keyword matching.
    Returns sentiment_score (-1 to +1) and article_count.
    """
    try:
        encoded = query.replace(" ", "+")
        url = (
            f"https://news.google.com/rss/search"
            f"?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)

        titles, descriptions = [], []
        for item in root.findall(".//item")[:20]:   # latest 20 articles
            t = item.findtext("title", "") or ""
            d = item.findtext("description", "") or ""
            titles.append(t.lower())
            descriptions.append(d.lower())

        if not titles:
            return {"news_sentiment": 0.0, "news_count": 0,
                    "news_pos": 0, "news_neg": 0}

        pos_count = neg_count = 0
        for text in titles + descriptions:
            words = set(re.findall(r"[a-z]+", text))
            pos_count += len(words & _POS)
            neg_count += len(words & _NEG)

        total = pos_count + neg_count
        score = (pos_count - neg_count) / total if total > 0 else 0.0

        return {
            "news_sentiment": round(score, 4),
            "news_count":     len(titles),
            "news_pos":       pos_count,
            "news_neg":       neg_count,
        }

    except Exception:
        return {"news_sentiment": 0.0, "news_count": 0,
                "news_pos": 0, "news_neg": 0}


def get_all_external_features(symbol: str) -> dict:
    """
    Combine global market + news sentiment into one dict.
    Used to enrich the 10-min feature dataframe.
    """
    # Strip exchange suffix for news query
    name = symbol.replace(".NS", "").replace("^NSEI", "Nifty 50").replace("^", "")
    query = f"{name} India stock market NSE"

    markets = fetch_global_markets()
    news    = fetch_news_sentiment(query)

    return {**markets, **news}


def merge_into_df(df: pd.DataFrame, ext: dict) -> pd.DataFrame:
    """
    Broadcast external (daily) features into the 10-min OHLCV dataframe.
    Every row gets the same external feature values (they're daily signals).
    """
    df = df.copy()
    for k, v in ext.items():
        df[k] = float(v) if v is not None else 0.0
    return df


# Feature names added by this module
SENTIMENT_FEATURE_COLS = [
    "sp500_change", "nasdaq_change", "dow_change",
    "india_vix", "india_vix_change",
    "vix_level", "vix_change",
    "nikkei_change", "hangseng_change",
    "usdinr", "usdinr_change",
    "gold_change", "crude_change",
    "news_sentiment", "news_count",
]
