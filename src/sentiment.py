"""
External signal module — global market context, institutional flows,
options market, social sentiment, and market breadth.
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
    "sp500":     "^GSPC",
    "nasdaq":    "^IXIC",
    "dow":       "^DJI",
    "vix":       "^VIX",
    "india_vix": "^INDIAVIX",
    "nikkei":    "^N225",
    "hangseng":  "^HSI",
    "usdinr":    "INR=X",
    "gold":      "GC=F",
    "crude":     "CL=F",
    "sgx_nifty": "^NSEI",
}

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


# ── Helpers ────────────────────────────────────────────────────────────────
def _safe_pct_change(series: pd.Series) -> float:
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


def _get_nse_session() -> requests.Session:
    """Create a requests.Session with NSE cookies."""
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return session


# ── Global markets ─────────────────────────────────────────────────────────
def fetch_global_markets() -> dict:
    features = {}
    try:
        symbols = list(GLOBAL_TICKERS.values())
        raw = yf.download(
            symbols, period="5d", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        for name, sym in GLOBAL_TICKERS.items():
            try:
                close = raw[sym]["Close"] if sym in raw.columns.get_level_values(0) else pd.Series()
                if name == "india_vix":
                    features["india_vix"]       = _safe_last(close)
                    features["india_vix_change"] = _safe_pct_change(close)
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
        for name in GLOBAL_TICKERS:
            features[f"{name}_change"] = 0.0
        features.update({
            "india_vix": 15.0, "india_vix_change": 0.0,
            "vix_level": 20.0, "vix_change": 0.0,
            "usdinr": 84.0,    "usdinr_change": 0.0,
        })
    return features


# ── FII / DII institutional flows ─────────────────────────────────────────
def fetch_fii_dii(session: requests.Session) -> dict:
    fallback = {"fii_net": 0.0, "dii_net": 0.0, "fii_dii_ratio": 0.0}
    try:
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        resp = session.get(url, timeout=8)
        data = resp.json()
        fii_net = dii_net = 0.0
        for row in data:
            cat = row.get("category", "").upper()
            net = float(str(row.get("netValue", "0")).replace(",", ""))
            if "FII" in cat or "FPI" in cat:
                fii_net = net
            elif "DII" in cat:
                dii_net = net
        ratio = fii_net / dii_net if dii_net != 0 else 0.0
        return {
            "fii_net":       round(fii_net, 2),
            "dii_net":       round(dii_net, 2),
            "fii_dii_ratio": round(ratio, 4),
        }
    except Exception:
        return fallback


# ── PCR — Put-Call Ratio ───────────────────────────────────────────────────
def fetch_pcr(session: requests.Session) -> dict:
    fallback = {"pcr": 1.0, "pcr_signal": 0.0}
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        resp = session.get(url, timeout=10)
        data = resp.json()
        ce_oi = data["filtered"]["CE"]["totOI"]
        pe_oi = data["filtered"]["PE"]["totOI"]
        pcr = pe_oi / ce_oi if ce_oi > 0 else 1.0
        # PCR > 1.2 = too many puts → contrarian bullish; < 0.8 = too many calls → contrarian bearish
        signal = -1.0 if pcr > 1.2 else (1.0 if pcr < 0.8 else 0.0)
        return {"pcr": round(pcr, 4), "pcr_signal": signal}
    except Exception:
        return fallback


# ── Reddit sentiment ───────────────────────────────────────────────────────
def fetch_reddit_sentiment() -> dict:
    fallback = {"reddit_sentiment": 0.0, "reddit_count": 0}
    headers = {"User-Agent": "NiftyPredictor/1.0 (educational project)"}
    all_texts = []
    try:
        for sub in ["IndiaInvestments", "IndianStockMarket"]:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit=25"
            resp = requests.get(url, headers=headers, timeout=8)
            posts = resp.json()["data"]["children"]
            for p in posts:
                title    = p["data"].get("title", "").lower()
                selftext = p["data"].get("selftext", "").lower()
                all_texts.append(title + " " + selftext)
        if not all_texts:
            return fallback
        pos = neg = 0
        for text in all_texts:
            words = set(re.findall(r"[a-z]+", text))
            pos += len(words & _POS)
            neg += len(words & _NEG)
        total = pos + neg
        score = (pos - neg) / total if total > 0 else 0.0
        return {
            "reddit_sentiment": round(score, 4),
            "reddit_count":     len(all_texts),
        }
    except Exception:
        return fallback


# ── Market breadth ─────────────────────────────────────────────────────────
def fetch_market_breadth() -> dict:
    fallback = {"breadth_pct_above_ema50": 0.5}
    try:
        from src.constants import NIFTY50_SYMBOLS
        symbols = [s for s in NIFTY50_SYMBOLS.keys() if not s.startswith("^")]
        raw = yf.download(
            symbols, period="90d", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        above = counted = 0
        for sym in symbols:
            try:
                close = raw[sym]["Close"].dropna() if len(symbols) > 1 else raw["Close"].dropna()
                if len(close) < 50:
                    continue
                ema50 = close.ewm(span=50, adjust=False).mean()
                if close.iloc[-1] > ema50.iloc[-1]:
                    above += 1
                counted += 1
            except Exception:
                continue
        if counted == 0:
            return fallback
        return {"breadth_pct_above_ema50": round(above / counted, 4)}
    except Exception:
        return fallback


# ── Google News sentiment ──────────────────────────────────────────────────
def fetch_news_sentiment(query: str = "Nifty India stock market") -> dict:
    try:
        encoded = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
        titles, descriptions = [], []
        for item in root.findall(".//item")[:20]:
            titles.append((item.findtext("title", "") or "").lower())
            descriptions.append((item.findtext("description", "") or "").lower())
        if not titles:
            return {"news_sentiment": 0.0, "news_count": 0, "news_pos": 0, "news_neg": 0}
        pos = neg = 0
        for text in titles + descriptions:
            words = set(re.findall(r"[a-z]+", text))
            pos += len(words & _POS)
            neg += len(words & _NEG)
        total = pos + neg
        score = (pos - neg) / total if total > 0 else 0.0
        return {"news_sentiment": round(score, 4), "news_count": len(titles),
                "news_pos": pos, "news_neg": neg}
    except Exception:
        return {"news_sentiment": 0.0, "news_count": 0, "news_pos": 0, "news_neg": 0}


# ── Master fetch ───────────────────────────────────────────────────────────
def get_all_external_features(symbol: str) -> dict:
    name  = symbol.replace(".NS", "").replace("^NSEI", "Nifty 50").replace("^", "")
    query = f"{name} India stock market NSE"

    nse_session = _get_nse_session()   # one session for both NSE endpoints

    markets = fetch_global_markets()
    news    = fetch_news_sentiment(query)
    fii_dii = fetch_fii_dii(nse_session)
    pcr     = fetch_pcr(nse_session)
    reddit  = fetch_reddit_sentiment()
    breadth = fetch_market_breadth()

    return {**markets, **news, **fii_dii, **pcr, **reddit, **breadth}


def merge_into_df(df: pd.DataFrame, ext: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in ext.items():
        df[k] = float(v) if v is not None else 0.0
    return df


# ── Historical global market data (for training) ───────────────────────────
GLOBAL_MKT_FEATURE_COLS = [
    "sp500_change", "nasdaq_change", "dow_change",
    "india_vix", "india_vix_change",
    "vix_level", "vix_change",
    "nikkei_change", "hangseng_change",
    "usdinr", "usdinr_change",
    "gold_change", "crude_change",
]

_FALLBACK_GLOBAL = {
    "sp500_change": 0.0, "nasdaq_change": 0.0, "dow_change": 0.0,
    "india_vix": 15.0,   "india_vix_change": 0.0,
    "vix_level": 20.0,   "vix_change": 0.0,
    "nikkei_change": 0.0, "hangseng_change": 0.0,
    "usdinr": 84.0,       "usdinr_change": 0.0,
    "gold_change": 0.0,   "crude_change": 0.0,
}


def fetch_global_markets_historical(days: int = 70) -> pd.DataFrame:
    """
    Return a date-indexed DataFrame of daily global market features.
    Each row is one trading day with pct-change and level values.
    Used for training so the model sees day-to-day variation (not a constant broadcast).
    """
    try:
        symbols = list(GLOBAL_TICKERS.values())
        raw = yf.download(
            symbols, period=f"{days + 10}d", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        if raw.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=raw.index)
        lvl0 = raw.columns.get_level_values(0)

        for name, sym in GLOBAL_TICKERS.items():
            if name == "sgx_nifty":
                continue  # skip — same as ^NSEI, redundant
            try:
                if sym not in lvl0:
                    continue
                close = raw[sym]["Close"].dropna()
                pct   = close.pct_change() * 100

                if name == "india_vix":
                    result["india_vix"]        = close
                    result["india_vix_change"] = pct
                elif name == "vix":
                    result["vix_level"] = close
                    result["vix_change"] = pct
                elif name == "usdinr":
                    result["usdinr"]        = close
                    result["usdinr_change"] = pct
                else:
                    result[f"{name}_change"] = pct
            except Exception:
                pass

        result = result.dropna(how="all")
        # Shift by 1 day: use previous day's values for each training row
        # (avoids look-ahead bias; Indian market reacts to prior-day global moves)
        result = result.shift(1)
        return result.iloc[days * -1:]   # keep only the requested window

    except Exception:
        return pd.DataFrame()


def merge_global_historical(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Join daily global market data into an intraday DataFrame by date.
    Each intraday row gets the previous trading day's global market values.
    """
    if hist.empty:
        # Apply fallback constants so FEATURE_COLS columns exist
        for col, val in _FALLBACK_GLOBAL.items():
            df[col] = val
        return df

    df = df.copy()

    # Normalise index timezones
    df_dates = df.index.normalize()
    if df.index.tzinfo is not None:
        df_dates = df_dates.tz_localize(None)

    hist_idx = hist.index.tz_localize(None) if hist.index.tzinfo is not None else hist.index

    for col in hist.columns:
        mapping = dict(zip(hist_idx, hist[col]))
        df[col] = df_dates.map(mapping)

    # Forward-fill missing dates (weekends, holidays)
    for col in hist.columns:
        df[col] = df[col].ffill()

    # Fill any remaining NaN with fallback
    for col, val in _FALLBACK_GLOBAL.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val

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
    "fii_net", "dii_net", "fii_dii_ratio",
    "pcr", "pcr_signal",
    "reddit_sentiment", "reddit_count",
    "breadth_pct_above_ema50",
]
