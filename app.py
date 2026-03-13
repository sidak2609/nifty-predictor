import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pytz

from src.constants import NIFTY50_SYMBOLS, NIFTY_INDEX, MIN_CONFIDENCE, HIGH_CONFIDENCE
from src.data_fetcher import fetch_ohlcv, get_market_status
from src.features import engineer_features, engineer_lag_features
from src.model import NiftyPredictor
from src.sentiment import get_all_external_features, merge_into_df

IST = pytz.timezone("Asia/Kolkata")

st.set_page_config(
    page_title="Nifty50 Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .pred-up   { color: #00E676; font-size: 2rem; font-weight: 700; }
    .pred-down { color: #FF5252; font-size: 2rem; font-weight: 700; }
    .conf-high { color: #00E676; }
    .conf-med  { color: #FFD740; }
    .conf-low  { color: #FF5252; }
    .tag-up    { background:#00E676; color:#000; padding:3px 10px; border-radius:12px; font-weight:700; }
    .tag-down  { background:#FF5252; color:#fff; padding:3px 10px; border-radius:12px; font-weight:700; }
    .section-title { font-size:1.1rem; font-weight:600; color:#AAB4C8; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


# ── Session-state helpers ─────────────────────────────────────────────────
def _get_predictor(symbol):
    return st.session_state.get(f"model_{symbol}")

def _set_predictor(symbol, model):
    st.session_state[f"model_{symbol}"] = model


# ── Data pipeline ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_data(symbol: str) -> pd.DataFrame:
    raw = fetch_ohlcv(symbol, days=55)
    df  = engineer_features(raw)
    df  = engineer_lag_features(df)
    return df

@st.cache_data(ttl=600, show_spinner=False)
def load_external(symbol: str) -> dict:
    return get_all_external_features(symbol)

def train_and_predict(symbol: str, df: pd.DataFrame):
    model    = _get_predictor(symbol)
    now      = time.time()
    last_key = f"last_train_{symbol}"
    should_retrain = model is None or (now - st.session_state.get(last_key, 0)) > 3600

    if should_retrain:
        model = NiftyPredictor()
        model.train(df)
        _set_predictor(symbol, model)
        st.session_state[last_key] = now

    prediction   = model.predict(df)
    prediction30 = model.predict_30min(df)
    return model.metrics, prediction, prediction30, model


# ── Chart ─────────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, prediction, symbol: str):
    display = df.tail(50).copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
    )

    fig.add_trace(go.Candlestick(
        x=display.index,
        open=display["open"], high=display["high"],
        low=display["low"],   close=display["close"],
        name="Price",
        increasing_line_color="#00E676", decreasing_line_color="#FF5252",
        increasing_fillcolor="#00E676",  decreasing_fillcolor="#FF5252",
    ), row=1, col=1)

    for col, color, label in [("ema9", "#40C4FF", "EMA9"), ("ema21", "#FFD740", "EMA21")]:
        if col in display.columns:
            fig.add_trace(go.Scatter(
                x=display.index, y=display[col],
                line=dict(color=color, width=1.2), name=label, opacity=0.8,
            ), row=1, col=1)

    if "vwap" in display.columns:
        fig.add_trace(go.Scatter(
            x=display.index, y=display["vwap"],
            line=dict(color="#CE93D8", width=1.2, dash="dot"),
            name="VWAP", opacity=0.8,
        ), row=1, col=1)

    if prediction:
        last_time = display.index[-1]
        pred_time = last_time + pd.tseries.frequencies.to_offset("10min")
        color = "#00E676" if prediction["direction"] == "UP" else "#FF5252"
        fig.add_trace(go.Scatter(
            x=[pred_time], y=[prediction["predicted_price"]],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol="diamond"),
            text=[f"₹{prediction['predicted_price']:,.2f}"],
            textposition="top center", textfont=dict(color=color, size=11),
            name="10-min Pred",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[pred_time, pred_time],
            y=[prediction["predicted_low"], prediction["predicted_high"]],
            mode="lines", line=dict(color=color, width=3),
            showlegend=False, opacity=0.5,
        ), row=1, col=1)

    colors = ["#00E676" if c >= o else "#FF5252"
              for c, o in zip(display["close"], display["open"])]
    fig.add_trace(go.Bar(
        x=display.index, y=display["volume"],
        marker_color=colors, name="Volume", opacity=0.7,
    ), row=2, col=1)

    if "rsi14" in display.columns:
        fig.add_trace(go.Scatter(
            x=display.index, y=display["rsi14"],
            line=dict(color="#40C4FF", width=1.5), name="RSI(14)",
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00E676", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=560, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=30, b=10, l=10, r=10),
    )
    fig.update_xaxes(gridcolor="#2D3139")
    fig.update_yaxes(gridcolor="#2D3139")
    return fig


# ── All-stocks scan ───────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def scan_all_stocks() -> list[dict]:
    shared_ext = get_all_external_features("^NSEI")
    results = []
    for sym, name in NIFTY50_SYMBOLS.items():
        try:
            df_raw = load_data(sym)
            df     = merge_into_df(df_raw, shared_ext)
            model  = NiftyPredictor()
            model.train(df)
            pred = model.predict(df)
            if pred and pred["confidence"] >= MIN_CONFIDENCE * 100:
                results.append({
                    "Symbol":     sym.replace(".NS", "").replace("^", ""),
                    "Name":       name,
                    "Price (₹)":  pred["current_price"],
                    "Target (₹)": pred["predicted_price"],
                    "Change":     f"{pred['pct_change']:+.4f}%",
                    "Direction":  pred["direction"],
                    "Confidence": f"{pred['confidence']:.1f}%",
                })
        except Exception:
            continue
    results.sort(key=lambda x: float(x["Confidence"].rstrip("%")), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
market = get_market_status()
if market["is_open"]:
    st_autorefresh(interval=600_000, key="live_refresh")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Nifty50 Predictor")
    status_color = "#00E676" if market["is_open"] else "#FF5252"
    st.markdown(
        f"<div style='color:{status_color}; font-weight:600;'>● {market['text']}</div>"
        f"<div style='color:#666; font-size:0.8rem;'>{market['time']}</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    symbol_options = list(NIFTY50_SYMBOLS.keys())
    display_names  = [f"{s.replace('.NS','').replace('^','')} — {n}" for s, n in NIFTY50_SYMBOLS.items()]
    selected_idx   = st.selectbox("Select Stock", range(len(symbol_options)),
                                  format_func=lambda i: display_names[i], index=0)
    selected_symbol = symbol_options[selected_idx]
    selected_name   = NIFTY50_SYMBOLS[selected_symbol]

    st.divider()
    st.markdown("**Model Config**")
    st.caption("• Training: last 30 days (rolling)")
    st.caption("• Ensemble: XGBoost × 3 + LightGBM + MLP")
    st.caption("• 3 session sub-models (time-of-day)")
    st.caption("• Horizons: 10-min & 30-min")
    st.caption(f"• Min confidence: {int(MIN_CONFIDENCE*100)}%")

    st.divider()
    show_scan = st.checkbox("Show all Nifty50 signals", value=False)
    if st.button("Force Retrain"):
        st.session_state[f"last_train_{selected_symbol}"] = 0
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────
st.title(f"📈  {selected_name}  ({selected_symbol.replace('.NS','').replace('^','')})")

# ── Load data + sentiment + train ─────────────────────────────────────────
with st.spinner("Loading data, sentiment & training model…"):
    try:
        df_raw      = load_data(selected_symbol)
        ext_features = load_external(selected_symbol)
        df          = merge_into_df(df_raw, ext_features)
        metrics, prediction, prediction30, model = train_and_predict(selected_symbol, df)
        data_ok = True
    except Exception as e:
        st.error(f"Failed to load data for {selected_symbol}: {e}")
        st.info("Yahoo Finance may temporarily not have data for this symbol. Try another stock or click Force Retrain.")
        data_ok = False
        ext_features = {}

if not data_ok:
    st.stop()

# ── Top metric row ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Current Price", f"₹{prediction['current_price']:,.2f}" if prediction else "—")
with c2:
    if prediction:
        st.metric("Predicted (10 min)", f"₹{prediction['predicted_price']:,.2f}",
                  f"{prediction['pct_change']:+.4f}%")
with c3:
    if prediction30:
        st.metric("Predicted (30 min)", f"₹{prediction30['predicted_price_30min']:,.2f}",
                  f"{prediction30['pct_change_30min']:+.4f}%")
with c4:
    if prediction:
        conf = prediction["confidence"]
        cls  = "conf-high" if conf >= 75 else ("conf-med" if conf >= 60 else "conf-low")
        st.markdown(f"<p class='section-title'>Confidence</p>"
                    f"<p class='{cls}' style='font-size:1.8rem;font-weight:700'>{conf:.1f}%</p>",
                    unsafe_allow_html=True)
with c5:
    if prediction:
        tag = "tag-up" if prediction["direction"] == "UP" else "tag-down"
        arr = "▲" if prediction["direction"] == "UP" else "▼"
        st.markdown(f"<p class='section-title'>Direction</p>"
                    f"<p><span class='{tag}'>{arr} {prediction['direction']}</span></p>",
                    unsafe_allow_html=True)
with c6:
    if prediction:
        st.markdown(
            f"<p class='section-title'>Price Range</p>"
            f"<p style='font-size:0.95rem'>₹{prediction['predicted_low']:,.2f} — ₹{prediction['predicted_high']:,.2f}</p>",
            unsafe_allow_html=True)

st.divider()

# ── Global Market Context ─────────────────────────────────────────────────
if ext_features:
    st.markdown("### 🌍 Market Context & Sentiment")

    def _arrow(v):
        if v > 0.1:  return f"▲ +{v:.2f}%"
        if v < -0.1: return f"▼ {v:.2f}%"
        return f"→ {v:.2f}%"

    def _col(v):
        if v > 0.1:  return "#00E676"
        if v < -0.1: return "#FF5252"
        return "#FFD740"

    g = st.columns(8)
    for col, label, key in zip(g, [
        "S&P 500","Nasdaq","Nikkei","Hang Seng","Gold","Crude Oil","USD/INR","India VIX"
    ], [
        "sp500_change","nasdaq_change","nikkei_change","hangseng_change",
        "gold_change","crude_change","usdinr_change","india_vix_change"
    ]):
        v = ext_features.get(key, 0)
        col.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:0.75rem;color:#AAB4C8'>{label}</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{_col(v)}'>{_arrow(v)}</div>"
            f"</div>", unsafe_allow_html=True)

    news_score = ext_features.get("news_sentiment", 0)
    news_cnt   = int(ext_features.get("news_count", 0))
    news_pos   = int(ext_features.get("news_pos", 0))
    news_neg   = int(ext_features.get("news_neg", 0))
    sent_label = "Bullish" if news_score > 0.1 else ("Bearish" if news_score < -0.1 else "Neutral")
    sent_color = "#00E676" if news_score > 0.1 else ("#FF5252" if news_score < -0.1 else "#FFD740")
    india_vix  = ext_features.get("india_vix", 0)
    st.markdown(
        f"<div style='margin-top:8px;font-size:0.9rem'>"
        f"📰 News: <span style='color:{sent_color};font-weight:700'>{sent_label} ({news_score:+.2f})</span>"
        f" | {news_cnt} articles | 👍 {news_pos} | 👎 {news_neg}"
        f" | India VIX: <b>{india_vix:.1f}</b>"
        f"</div>", unsafe_allow_html=True)

# ── Institutional Flows & Market Structure ────────────────────────────────
if ext_features:
    with st.expander("🏦 Institutional Flows & Market Structure", expanded=True):
        i1, i2, i3, i4, i5 = st.columns(5)

        fii_net  = ext_features.get("fii_net", 0)
        dii_net  = ext_features.get("dii_net", 0)
        pcr      = ext_features.get("pcr", 1.0)
        breadth  = ext_features.get("breadth_pct_above_ema50", 0.5) * 100
        reddit   = ext_features.get("reddit_sentiment", 0.0)
        fii_rat  = ext_features.get("fii_dii_ratio", 0.0)

        i1.metric("FII Net (₹ Cr)", f"{fii_net:+,.0f}",
                  "Buying" if fii_net > 0 else "Selling")
        i2.metric("DII Net (₹ Cr)", f"{dii_net:+,.0f}",
                  "Buying" if dii_net > 0 else "Selling")

        pcr_label = "Bearish" if pcr > 1.2 else ("Bullish" if pcr < 0.8 else "Neutral")
        i3.metric("Put-Call Ratio", f"{pcr:.2f}", pcr_label)

        i4.metric("Breadth (>EMA50)", f"{breadth:.1f}%",
                  "Strong" if breadth > 60 else ("Weak" if breadth < 40 else "Neutral"))

        reddit_label = "Bullish" if reddit > 0.1 else ("Bearish" if reddit < -0.1 else "Neutral")
        reddit_col   = "#00E676" if reddit > 0.1 else ("#FF5252" if reddit < -0.1 else "#FFD740")
        i5.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:0.75rem;color:#AAB4C8'>Reddit Sentiment</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{reddit_col}'>"
            f"{reddit_label} ({reddit:+.2f})</div></div>", unsafe_allow_html=True)

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────
chart = build_chart(df_raw, prediction, selected_symbol)
st.plotly_chart(chart, use_container_width=True)

# ── Confidence warning ────────────────────────────────────────────────────
if prediction:
    conf = prediction["confidence"]
    if conf >= HIGH_CONFIDENCE * 100:
        st.success(f"High confidence ({conf:.1f}%) — models strongly agree on direction.")
    elif conf >= MIN_CONFIDENCE * 100:
        st.warning(f"Moderate confidence ({conf:.1f}%) — treat as indicative.")
    else:
        st.error(f"Low confidence ({conf:.1f}%) — avoid trading on this signal.")

# ── Model performance ─────────────────────────────────────────────────────
st.markdown("### Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Directional Accuracy", f"{metrics.get('dir_acc', 0):.1f}%")
m2.metric("Price MAPE",           f"{metrics.get('mape', 0):.3f}%")
m3.metric("Training Samples",     f"{metrics.get('n_samples', 0):,}")
m4.metric("Bias Correction",      f"{metrics.get('bias_corr', 0):.4f}%")

# Session model status
sc1, sc2, sc3, sc4 = st.columns(4)
trained = metrics.get("sessions_trained", [])
sc1.metric("LightGBM",       "✓ Active" if metrics.get("lgbm") else "✗ Unavailable")
sc2.metric("MLP Sequence",   "✓ Active" if metrics.get("mlp")  else "✗ Unavailable")
sc3.metric("Session Models", f"{len(trained)}/3 trained")
sc4.metric("Sessions",       ", ".join(trained) if trained else "none")

st.caption(
    "Walk-forward CV on last 30 days | XGBoost ensemble + LightGBM + MLP + session sub-models | "
    "Features: 62 technical + 23 external signals"
)

# ── Feature importance ────────────────────────────────────────────────────
if model and not model.feature_importance.empty:
    with st.expander("Top 20 most important features"):
        top20 = model.feature_importance.head(20)
        fig_imp = go.Figure(go.Bar(
            x=top20.values[::-1], y=top20.index[::-1],
            orientation="h", marker_color="#40C4FF",
        ))
        fig_imp.update_layout(
            height=420, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"), margin=dict(t=10, b=10, l=180, r=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────────────────
with st.expander("Raw data (last 30 candles)"):
    base_cols = ["open", "high", "low", "close", "volume", "rsi14", "macd", "vwap"]
    extra_cols = [c for c in ["session", "adx", "pcr", "fii_net", "breadth_pct_above_ema50"]
                  if c in df.columns]
    display_df = df[base_cols + extra_cols].tail(30).copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display_df.style.format(
        {c: "{:.2f}" for c in display_df.select_dtypes(float).columns}
    ), use_container_width=True)

# ── All-stocks scan ───────────────────────────────────────────────────────
if show_scan:
    st.divider()
    st.markdown("### All Nifty50 Signals")
    with st.spinner("Scanning all 50 stocks…"):
        scan_results = scan_all_stocks()
    if scan_results:
        scan_df = pd.DataFrame(scan_results)
        def _dir_color(val):
            return "color: #00E676; font-weight:bold" if val == "UP" else "color: #FF5252; font-weight:bold"
        st.dataframe(scan_df.style.map(_dir_color, subset=["Direction"]),
                     use_container_width=True, hide_index=True)
        up   = sum(1 for r in scan_results if r["Direction"] == "UP")
        down = len(scan_results) - up
        st.caption(f"▲ {up} bullish  |  ▼ {down} bearish  |  Total: {len(scan_results)}")
    else:
        st.info("No high-confidence signals right now.")

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Educational purposes only. Not financial advice. "
    "Past accuracy does not guarantee future results."
)
st.caption(f"Last updated: {market['time']}")
