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
from src.features import engineer_features
from src.model import NiftyPredictor

IST = pytz.timezone("Asia/Kolkata")

st.set_page_config(
    page_title="Nifty50 Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS tweaks ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1A1D23;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #2D3139;
    }
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


# ── Session-state helpers ────────────────────────────────────────────────────
def _get_predictor(symbol: str) -> NiftyPredictor:
    key = f"model_{symbol}"
    if key not in st.session_state:
        st.session_state[key] = None
    return st.session_state[key]


def _set_predictor(symbol: str, model: NiftyPredictor):
    st.session_state[f"model_{symbol}"] = model


# ── Data + model pipeline ────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_data(symbol: str) -> pd.DataFrame:
    """Fetch and engineer features. Cached for 10 minutes."""
    raw = fetch_ohlcv(symbol, days=55)
    return engineer_features(raw)


def train_and_predict(symbol: str, df: pd.DataFrame) -> tuple[dict, dict]:
    """Train model (or use cached) and return metrics + prediction."""
    model = _get_predictor(symbol)

    retrain_key = f"last_train_{symbol}"
    now = time.time()
    last_train = st.session_state.get(retrain_key, 0)
    should_retrain = model is None or (now - last_train) > 3600   # retrain hourly

    if should_retrain:
        model = NiftyPredictor()
        metrics = model.train(df)
        _set_predictor(symbol, model)
        st.session_state[retrain_key] = now
    else:
        metrics = model.metrics

    prediction = model.predict(df)
    return metrics, prediction


# ── Chart ────────────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, prediction: dict | None, symbol: str) -> go.Figure:
    display = df.tail(50).copy()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("", "Volume", "RSI"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=display.index,
        open=display["open"], high=display["high"],
        low=display["low"],   close=display["close"],
        name="Price",
        increasing_line_color="#00E676",
        decreasing_line_color="#FF5252",
        increasing_fillcolor="#00E676",
        decreasing_fillcolor="#FF5252",
    ), row=1, col=1)

    # EMA lines
    for col, color, label in [("ema9", "#40C4FF", "EMA9"), ("ema21", "#FFD740", "EMA21")]:
        if col in display.columns:
            fig.add_trace(go.Scatter(
                x=display.index, y=display[col],
                line=dict(color=color, width=1.2),
                name=label, opacity=0.8,
            ), row=1, col=1)

    # VWAP
    if "vwap" in display.columns:
        fig.add_trace(go.Scatter(
            x=display.index, y=display["vwap"],
            line=dict(color="#CE93D8", width=1.2, dash="dot"),
            name="VWAP", opacity=0.8,
        ), row=1, col=1)

    # Prediction marker
    if prediction:
        last_time = display.index[-1]
        freq = pd.tseries.frequencies.to_offset("10min")
        pred_time = last_time + freq

        color = "#00E676" if prediction["direction"] == "UP" else "#FF5252"
        fig.add_trace(go.Scatter(
            x=[pred_time],
            y=[prediction["predicted_price"]],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol="diamond"),
            text=[f"₹{prediction['predicted_price']:,.2f}"],
            textposition="top center",
            textfont=dict(color=color, size=11),
            name="Prediction",
        ), row=1, col=1)

        # Prediction range band
        fig.add_trace(go.Scatter(
            x=[pred_time, pred_time],
            y=[prediction["predicted_low"], prediction["predicted_high"]],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
            opacity=0.5,
        ), row=1, col=1)

    # Volume bars
    colors = ["#00E676" if c >= o else "#FF5252"
              for c, o in zip(display["close"], display["open"])]
    fig.add_trace(go.Bar(
        x=display.index, y=display["volume"],
        marker_color=colors, name="Volume", opacity=0.7,
    ), row=2, col=1)

    # RSI
    if "rsi14" in display.columns:
        fig.add_trace(go.Scatter(
            x=display.index, y=display["rsi14"],
            line=dict(color="#40C4FF", width=1.5),
            name="RSI(14)",
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00E676", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=580,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=30, b=10, l=10, r=10),
    )
    fig.update_xaxes(gridcolor="#2D3139", showgrid=True)
    fig.update_yaxes(gridcolor="#2D3139", showgrid=True)

    return fig


# ── All-stocks scan ──────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def scan_all_stocks() -> list[dict]:
    results = []
    for sym, name in NIFTY50_SYMBOLS.items():
        try:
            df = load_data(sym)
            model = NiftyPredictor()
            model.train(df)
            pred = model.predict(df)
            if pred and pred["confidence"] >= MIN_CONFIDENCE * 100:
                results.append({
                    "Symbol":     sym.replace(".NS", "").replace("^", ""),
                    "Name":       name,
                    "Price (₹)":  pred["current_price"],
                    "Target (₹)": pred["predicted_price"],
                    "Change":     f"{pred['pct_change']:+.2f}%",
                    "Direction":  pred["direction"],
                    "Confidence": f"{pred['confidence']:.1f}%",
                })
        except Exception:
            continue
    results.sort(key=lambda x: float(x["Confidence"].rstrip("%")), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

# Auto-refresh every 10 minutes (600 000 ms) during market hours
market = get_market_status()
if market["is_open"]:
    st_autorefresh(interval=600_000, key="live_refresh")

# ── Sidebar ──────────────────────────────────────────────────────────────────
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
                                  format_func=lambda i: display_names[i], index=0)   # Nifty50 Index default
    selected_symbol = symbol_options[selected_idx]
    selected_name   = NIFTY50_SYMBOLS[selected_symbol]

    st.divider()
    st.markdown("**Prediction Info**")
    st.caption("• Horizon: next 10-min candle")
    st.caption("• Model: XGBoost Ensemble")
    st.caption(f"• Min confidence shown: {int(MIN_CONFIDENCE*100)}%")
    st.caption("• Retrained: every hour")

    st.divider()
    show_scan = st.checkbox("Show all Nifty50 signals", value=False)
    if st.button("Force Retrain"):
        st.session_state[f"last_train_{selected_symbol}"] = 0
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"📈  {selected_name}  ({selected_symbol.replace('.NS','').replace('^','')})")

# ── Load data & train ─────────────────────────────────────────────────────────
with st.spinner(f"Loading data and training model for {selected_symbol}…"):
    try:
        df = load_data(selected_symbol)
        metrics, prediction = train_and_predict(selected_symbol, df)
        data_ok = True
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        data_ok = False

if not data_ok:
    st.stop()

# ── Metric row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Current Price", f"₹{prediction['current_price']:,.2f}" if prediction else "—")

with col2:
    if prediction:
        delta_str = f"{prediction['pct_change']:+.2f}%"
        st.metric("Predicted (next 10 min)", f"₹{prediction['predicted_price']:,.2f}", delta_str)
    else:
        st.metric("Predicted (next 10 min)", "—")

with col3:
    if prediction:
        conf = prediction["confidence"]
        conf_class = "conf-high" if conf >= 75 else ("conf-med" if conf >= 60 else "conf-low")
        st.markdown(f"<p class='section-title'>Confidence</p>"
                    f"<p class='{conf_class}' style='font-size:1.8rem; font-weight:700;'>{conf:.1f}%</p>",
                    unsafe_allow_html=True)
    else:
        st.metric("Confidence", "—")

with col4:
    if prediction:
        tag_class = "tag-up" if prediction["direction"] == "UP" else "tag-down"
        arrow = "▲" if prediction["direction"] == "UP" else "▼"
        st.markdown(f"<p class='section-title'>Direction</p>"
                    f"<p><span class='{tag_class}'>{arrow} {prediction['direction']}</span></p>",
                    unsafe_allow_html=True)

with col5:
    if prediction:
        st.markdown(
            f"<p class='section-title'>Predicted Range</p>"
            f"<p style='font-size:1rem;'>₹{prediction['predicted_low']:,.2f} — ₹{prediction['predicted_high']:,.2f}</p>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────────
chart = build_chart(df, prediction, selected_symbol)
st.plotly_chart(chart, use_container_width=True)

# ── Confidence warning ────────────────────────────────────────────────────────
if prediction:
    conf = prediction["confidence"]
    if conf >= HIGH_CONFIDENCE * 100:
        st.success(f"High confidence signal ({conf:.1f}%) — model strongly agrees on direction.")
    elif conf >= MIN_CONFIDENCE * 100:
        st.warning(f"Moderate confidence ({conf:.1f}%) — treat as indicative, not definitive.")
    else:
        st.error(f"Low confidence ({conf:.1f}%) — avoid trading on this signal.")

# ── Model performance ─────────────────────────────────────────────────────────
st.markdown("### Model Performance (Walk-forward CV)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Directional Accuracy", f"{metrics.get('dir_acc', 0):.1f}%")
m2.metric("Price MAPE",           f"{metrics.get('mape', 0):.2f}%")
m3.metric("Regression Dir Acc",   f"{metrics.get('reg_dir_acc', 0):.1f}%")
m4.metric("Training Samples",     f"{metrics.get('n_samples', 0):,}")

st.caption(
    "Dir Acc = % of candles where predicted UP/DOWN matched actual. "
    "MAPE = mean absolute % error of the predicted price vs actual price. "
    "Walk-forward CV uses 3 time-series splits on 55 days of 10-min data."
)

# ── Feature importance ────────────────────────────────────────────────────────
model = _get_predictor(selected_symbol)
if model and not model.feature_importance.empty:
    with st.expander("Top 15 most important features"):
        top15 = model.feature_importance.head(15)
        fig_imp = go.Figure(go.Bar(
            x=top15.values[::-1],
            y=top15.index[::-1],
            orientation="h",
            marker_color="#40C4FF",
        ))
        fig_imp.update_layout(
            height=350,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            margin=dict(t=10, b=10, l=150, r=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("Raw data (last 30 candles)"):
    display_df = df[["open","high","low","close","volume","rsi14","macd","vwap"]].tail(30).copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display_df.style.format("{:.2f}"), use_container_width=True)

# ── All-stocks scan ───────────────────────────────────────────────────────────
if show_scan:
    st.divider()
    st.markdown("### All Nifty50 Signals (confidence ≥ 60%)")
    st.caption("Scanning all 50 stocks — this may take 1–2 minutes…")
    with st.spinner("Scanning all Nifty50 stocks…"):
        scan_results = scan_all_stocks()

    if scan_results:
        scan_df = pd.DataFrame(scan_results)
        def color_direction(val):
            if val == "UP":
                return "color: #00E676; font-weight: bold"
            return "color: #FF5252; font-weight: bold"

        st.dataframe(
            scan_df.style.map(color_direction, subset=["Direction"]),
            use_container_width=True,
            hide_index=True,
        )
        up_count   = sum(1 for r in scan_results if r["Direction"] == "UP")
        down_count = len(scan_results) - up_count
        st.caption(f"▲ {up_count} bullish  |  ▼ {down_count} bearish  |  Total signals: {len(scan_results)}")
    else:
        st.info("No high-confidence signals found right now.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️  This app is for educational purposes only. Not financial advice. "
    "Past prediction accuracy does not guarantee future results. "
    "Always do your own research before investing."
)
st.caption(f"Last updated: {market['time']}")
