import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz

from src.constants import NIFTY50_SYMBOLS, NIFTY_INDEX, MIN_CONFIDENCE, HIGH_CONFIDENCE
from src.data_fetcher import fetch_ohlcv, fetch_daily_ohlcv, get_market_status
from src.features import engineer_features, engineer_lag_features, compute_daily_context, merge_daily_context
from src.model import NiftyPredictor
from src.daily_model import NiftyDailyPredictor
from src.sentiment import (
    get_all_external_features,
    merge_into_df,
    fetch_global_markets_historical,
    merge_global_historical,
)

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
    .tag-hold  { background:#FFD740; color:#000; padding:3px 10px; border-radius:12px; font-weight:700; }
    .section-title { font-size:1.1rem; font-weight:600; color:#AAB4C8; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


# ── Session-state helpers ─────────────────────────────────────────────────
def _get_predictor(symbol):
    return st.session_state.get(f"model_{symbol}")

def _set_predictor(symbol, model):
    st.session_state[f"model_{symbol}"] = model


# ── Data pipeline (intraday) ─────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_global_hist() -> pd.DataFrame:
    return fetch_global_markets_historical(days=70)

@st.cache_data(ttl=600, show_spinner=False)
def load_daily_ctx(symbol: str) -> pd.DataFrame:
    daily = fetch_daily_ohlcv(symbol, period="1y")
    if daily.empty:
        return pd.DataFrame()
    return compute_daily_context(daily)

@st.cache_data(ttl=600, show_spinner=False)
def load_data(symbol: str) -> pd.DataFrame:
    raw       = fetch_ohlcv(symbol, days=58)
    df        = engineer_features(raw)
    df        = engineer_lag_features(df)
    hist_ext  = load_global_hist()
    df        = merge_global_historical(df, hist_ext)
    daily_ctx = load_daily_ctx(symbol)
    df        = merge_daily_context(df, daily_ctx)
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


# ── Daily (30-day) model ──────────────────────────────────────────────────
@st.cache_resource(ttl=3600, show_spinner=False)
def get_daily_model():
    dm = NiftyDailyPredictor()
    metrics = dm.train()
    return dm, metrics


# ── Intraday chart ────────────────────────────────────────────────────────
def build_chart(df, prediction, symbol):
    display = df.tail(50).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(
        x=display.index, open=display["open"], high=display["high"],
        low=display["low"], close=display["close"], name="Price",
        increasing_line_color="#00E676", decreasing_line_color="#FF5252",
        increasing_fillcolor="#00E676", decreasing_fillcolor="#FF5252",
    ), row=1, col=1)
    for col, color, label in [("ema9", "#40C4FF", "EMA9"), ("ema21", "#FFD740", "EMA21")]:
        if col in display.columns:
            fig.add_trace(go.Scatter(x=display.index, y=display[col],
                line=dict(color=color, width=1.2), name=label, opacity=0.8), row=1, col=1)
    if "vwap" in display.columns:
        fig.add_trace(go.Scatter(x=display.index, y=display["vwap"],
            line=dict(color="#CE93D8", width=1.2, dash="dot"), name="VWAP", opacity=0.8), row=1, col=1)
    if prediction:
        last_time = display.index[-1]
        pred_time = last_time + pd.tseries.frequencies.to_offset("10min")
        d = prediction["direction"]
        color = "#00E676" if d == "UP" else ("#FF5252" if d == "DOWN" else "#FFD740")
        fig.add_trace(go.Scatter(x=[pred_time], y=[prediction["predicted_price"]],
            mode="markers+text", marker=dict(size=14, color=color, symbol="diamond"),
            text=[f"₹{prediction['predicted_price']:,.2f}"],
            textposition="top center", textfont=dict(color=color, size=11), name="Pred"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[pred_time, pred_time],
            y=[prediction["predicted_low"], prediction["predicted_high"]],
            mode="lines", line=dict(color=color, width=3), showlegend=False, opacity=0.5), row=1, col=1)
    colors = ["#00E676" if c >= o else "#FF5252" for c, o in zip(display["close"], display["open"])]
    fig.add_trace(go.Bar(x=display.index, y=display["volume"],
        marker_color=colors, name="Volume", opacity=0.7), row=2, col=1)
    if "rsi14" in display.columns:
        fig.add_trace(go.Scatter(x=display.index, y=display["rsi14"],
            line=dict(color="#40C4FF", width=1.5), name="RSI(14)"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00E676", opacity=0.5, row=3, col=1)
    fig.update_layout(height=560, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0), margin=dict(t=30, b=10, l=10, r=10))
    fig.update_xaxes(gridcolor="#2D3139")
    fig.update_yaxes(gridcolor="#2D3139")
    return fig


# ── All-stocks scan ───────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def scan_all_stocks():
    results = []
    for sym, name in NIFTY50_SYMBOLS.items():
        try:
            df_stock = load_data(sym)
            mdl = NiftyPredictor()
            mdl.train(df_stock)
            pred = mdl.predict(df_stock)
            if pred and pred["confidence"] >= MIN_CONFIDENCE * 100:
                results.append({"Symbol": sym.replace(".NS","").replace("^",""),
                    "Name": name, "Price": pred["current_price"],
                    "Target": pred["predicted_price"],
                    "Change": f"{pred['pct_change']:+.4f}%",
                    "Direction": pred["direction"],
                    "Confidence": f"{pred['confidence']:.1f}%"})
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
        unsafe_allow_html=True)
    st.divider()

    symbol_options = list(NIFTY50_SYMBOLS.keys())
    display_names  = [f"{s.replace('.NS','').replace('^','')} — {n}" for s, n in NIFTY50_SYMBOLS.items()]
    selected_idx   = st.selectbox("Select Stock", range(len(symbol_options)),
                                  format_func=lambda i: display_names[i], index=0)
    selected_symbol = symbol_options[selected_idx]
    selected_name   = NIFTY50_SYMBOLS[selected_symbol]

    st.divider()
    st.markdown("**Intraday Model**")
    st.caption("Stacked: XGB + LightGBM-DART + Ridge")
    st.caption("MI selection (top 25) | Conformal 90%")
    st.caption("Ternary: UP / HOLD / DOWN")
    st.markdown("**Daily Model (30-day)**")
    st.caption("5yr training | 4 horizons: 5/10/20/30d")
    st.caption("Stacked + sentiment + global macro")

    st.divider()
    show_scan = st.checkbox("Show all Nifty50 signals", value=False)
    if st.button("Force Retrain"):
        st.session_state[f"last_train_{selected_symbol}"] = 0
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────
st.title(f"📈  {selected_name}  ({selected_symbol.replace('.NS','').replace('^','')})")

tab_intraday, tab_daily = st.tabs(["Intraday (10-min)", "30-Day Forecast"])

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1: INTRADAY
# ═══════════════════════════════════════════════════════════════════════════
with tab_intraday:
    with st.spinner("Loading data & training intraday model..."):
        try:
            df_raw       = load_data(selected_symbol)
            ext_features = load_external(selected_symbol)
            metrics, prediction, prediction30, model = train_and_predict(selected_symbol, df_raw)
            data_ok = True
        except Exception as e:
            st.error(f"Failed: {e}")
            data_ok = False
            ext_features = {}

    if not data_ok:
        st.stop()

    # ── Metrics row ───────────────────────────────────────────────────────
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
            cls = "conf-high" if conf >= 75 else ("conf-med" if conf >= 60 else "conf-low")
            st.markdown(f"<p class='section-title'>Confidence</p>"
                        f"<p class='{cls}' style='font-size:1.8rem;font-weight:700'>{conf:.1f}%</p>",
                        unsafe_allow_html=True)
    with c5:
        if prediction:
            d = prediction["direction"]
            tag = "tag-up" if d == "UP" else ("tag-down" if d == "DOWN" else "tag-hold")
            arr = "▲" if d == "UP" else ("▼" if d == "DOWN" else "→")
            st.markdown(f"<p class='section-title'>Direction</p>"
                        f"<p><span class='{tag}'>{arr} {d}</span></p>", unsafe_allow_html=True)
    with c6:
        if prediction:
            st.markdown(f"<p class='section-title'>Price Range</p>"
                f"<p style='font-size:0.95rem'>₹{prediction['predicted_low']:,.2f} — "
                f"₹{prediction['predicted_high']:,.2f}</p>", unsafe_allow_html=True)

    st.divider()

    # ── Global Market Context ─────────────────────────────────────────────
    if ext_features:
        st.markdown("### 🌍 Market Context & Sentiment")
        def _arrow(v):
            if v > 0.1: return f"▲ +{v:.2f}%"
            if v < -0.1: return f"▼ {v:.2f}%"
            return f"→ {v:.2f}%"
        def _col(v):
            if v > 0.1: return "#00E676"
            if v < -0.1: return "#FF5252"
            return "#FFD740"
        g = st.columns(8)
        for col, label, key in zip(g,
            ["S&P 500","Nasdaq","Nikkei","Hang Seng","Gold","Crude Oil","USD/INR","India VIX"],
            ["sp500_change","nasdaq_change","nikkei_change","hangseng_change",
             "gold_change","crude_change","usdinr_change","india_vix_change"]):
            v = ext_features.get(key, 0)
            col.markdown(f"<div style='text-align:center'>"
                f"<div style='font-size:0.75rem;color:#AAB4C8'>{label}</div>"
                f"<div style='font-size:1rem;font-weight:700;color:{_col(v)}'>{_arrow(v)}</div>"
                f"</div>", unsafe_allow_html=True)

        news_score = ext_features.get("news_sentiment", 0)
        news_cnt = int(ext_features.get("news_count", 0))
        sent_label = "Bullish" if news_score > 0.1 else ("Bearish" if news_score < -0.1 else "Neutral")
        sent_color = "#00E676" if news_score > 0.1 else ("#FF5252" if news_score < -0.1 else "#FFD740")
        india_vix = ext_features.get("india_vix", 0)
        st.markdown(f"<div style='margin-top:8px;font-size:0.9rem'>"
            f"📰 News: <span style='color:{sent_color};font-weight:700'>{sent_label} ({news_score:+.2f})</span>"
            f" | {news_cnt} articles | India VIX: <b>{india_vix:.1f}</b></div>", unsafe_allow_html=True)

    # ── Institutional Flows ───────────────────────────────────────────────
    if ext_features:
        with st.expander("🏦 Institutional Flows & Market Structure", expanded=True):
            i1, i2, i3, i4, i5 = st.columns(5)
            fii_net = ext_features.get("fii_net", 0)
            dii_net = ext_features.get("dii_net", 0)
            pcr = ext_features.get("pcr", 1.0)
            breadth = ext_features.get("breadth_pct_above_ema50", 0.5) * 100
            reddit = ext_features.get("reddit_sentiment", 0.0)
            i1.metric("FII Net (Cr)", f"{fii_net:+,.0f}", "Buying" if fii_net > 0 else "Selling")
            i2.metric("DII Net (Cr)", f"{dii_net:+,.0f}", "Buying" if dii_net > 0 else "Selling")
            pcr_label = "Bearish" if pcr > 1.2 else ("Bullish" if pcr < 0.8 else "Neutral")
            i3.metric("Put-Call Ratio", f"{pcr:.2f}", pcr_label)
            i4.metric("Breadth (>EMA50)", f"{breadth:.1f}%",
                      "Strong" if breadth > 60 else ("Weak" if breadth < 40 else "Neutral"))
            reddit_label = "Bullish" if reddit > 0.1 else ("Bearish" if reddit < -0.1 else "Neutral")
            reddit_col = "#00E676" if reddit > 0.1 else ("#FF5252" if reddit < -0.1 else "#FFD740")
            i5.markdown(f"<div style='text-align:center'>"
                f"<div style='font-size:0.75rem;color:#AAB4C8'>Reddit Sentiment</div>"
                f"<div style='font-size:1rem;font-weight:700;color:{reddit_col}'>"
                f"{reddit_label} ({reddit:+.2f})</div></div>", unsafe_allow_html=True)

    st.divider()

    # ── Chart ─────────────────────────────────────────────────────────────
    chart = build_chart(df_raw, prediction, selected_symbol)
    st.plotly_chart(chart, use_container_width=True)

    # ── Confidence warning ────────────────────────────────────────────────
    if prediction:
        conf = prediction["confidence"]
        if conf >= HIGH_CONFIDENCE * 100:
            st.success(f"High confidence ({conf:.1f}%) — models strongly agree.")
        elif conf >= MIN_CONFIDENCE * 100:
            st.warning(f"Moderate confidence ({conf:.1f}%) — treat as indicative.")
        else:
            st.error(f"Low confidence ({conf:.1f}%) — avoid trading on this signal.")

    # ── Model performance ─────────────────────────────────────────────────
    st.markdown("### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Directional Accuracy", f"{metrics.get('dir_acc', 0):.1f}%")
    m2.metric("Price MAPE", f"{metrics.get('mape', 0):.3f}%")
    m3.metric("Training Samples", f"{metrics.get('n_samples', 0):,}")
    m4.metric("Bias Correction", f"{metrics.get('bias_corr', 0):.4f}%")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("LightGBM-DART", "Active" if metrics.get("lgbm") else "N/A")
    sc2.metric("Stacking", "Active" if metrics.get("stacking") else "N/A")
    sc3.metric("Selected Features", f"{metrics.get('selected_features', 0)}/{metrics.get('active_features', 0)}")
    sc4.metric("Conformal Width", f"{metrics.get('conformal_width', 0):.4f}%")

    # ── Feature importance ────────────────────────────────────────────────
    if model and not model.feature_importance.empty:
        with st.expander("Top 20 most important features"):
            top20 = model.feature_importance.head(20)
            fig_imp = go.Figure(go.Bar(x=top20.values[::-1], y=top20.index[::-1],
                orientation="h", marker_color="#40C4FF"))
            fig_imp.update_layout(height=420, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#FAFAFA"), margin=dict(t=10, b=10, l=180, r=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    # ── All-stocks scan ───────────────────────────────────────────────────
    if show_scan:
        st.divider()
        st.markdown("### All Nifty50 Signals")
        with st.spinner("Scanning all 50 stocks..."):
            scan_results = scan_all_stocks()
        if scan_results:
            scan_df = pd.DataFrame(scan_results)
            def _dir_color(val):
                if val == "UP": return "color: #00E676; font-weight:bold"
                if val == "DOWN": return "color: #FF5252; font-weight:bold"
                return "color: #FFD740; font-weight:bold"
            st.dataframe(scan_df.style.map(_dir_color, subset=["Direction"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No high-confidence signals right now.")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2: 30-DAY FORECAST
# ═══════════════════════════════════════════════════════════════════════════
with tab_daily:
    with st.spinner("Training 30-day forecast model..."):
        try:
            dm, dm_metrics = get_daily_model()
            dm_pred = dm.predict()
            daily_ok = True
        except Exception as e:
            st.error(f"Daily model failed: {e}")
            st.caption(f"Details: {e}")
            daily_ok = False

    if daily_ok:
        cur_price = dm_pred["current_price"]

        # ── Headline predictions ──────────────────────────────────────────
        st.markdown("### Price Forecast")
        h1, h2, h3, h4 = st.columns(4)
        for col, horizon, label in [
            (h1, "5d", "5 Days"), (h2, "10d", "10 Days"),
            (h3, "20d", "20 Days"), (h4, "30d", "30 Days"),
        ]:
            p = dm_pred[horizon]
            d = p["direction"]
            color = "#00E676" if d == "UP" else "#FF5252"
            with col:
                st.metric(label, f"₹{p['price']:,.2f}", f"{p['pct_change']:+.2f}%")
                st.markdown(f"<div style='font-size:0.8rem;color:{color};font-weight:700'>"
                    f"{'▲' if d=='UP' else '▼'} {d} ({p['confidence']:.0f}% conf)</div>",
                    unsafe_allow_html=True)
                st.caption(f"Range: ₹{p['range_low']:,.0f} — ₹{p['range_high']:,.0f}")

        st.divider()

        # ── Forecast chart with confidence bands ──────────────────────────
        st.markdown("### Forecast Visualization")

        try:
            import yfinance as yf
            hist_daily = yf.Ticker("^NSEI").history(period="6mo", interval="1d", auto_adjust=True)
            hist_close = hist_daily["Close"].dropna().tail(90)
        except Exception:
            hist_close = pd.Series(dtype=float)

        fig_fc = go.Figure()

        if not hist_close.empty:
            fig_fc.add_trace(go.Scatter(
                x=hist_close.index, y=hist_close.values,
                line=dict(color="#40C4FF", width=2), name="Historical",
            ))

        today = datetime.now()
        fc_dates = [today + timedelta(days=d) for d in [5, 10, 20, 30]]
        fc_prices = [dm_pred[h]["price"] for h in ["5d", "10d", "20d", "30d"]]
        fc_lows   = [dm_pred[h]["range_low"] for h in ["5d", "10d", "20d", "30d"]]
        fc_highs  = [dm_pred[h]["range_high"] for h in ["5d", "10d", "20d", "30d"]]

        all_dates  = [today] + fc_dates
        all_prices = [cur_price] + fc_prices
        all_lows   = [cur_price] + fc_lows
        all_highs  = [cur_price] + fc_highs

        fig_fc.add_trace(go.Scatter(
            x=all_dates + all_dates[::-1],
            y=all_highs + all_lows[::-1],
            fill="toself", fillcolor="rgba(64, 196, 255, 0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="90% Confidence",
        ))

        fc_color = "#00E676" if fc_prices[-1] > cur_price else "#FF5252"
        fig_fc.add_trace(go.Scatter(
            x=all_dates, y=all_prices,
            line=dict(color=fc_color, width=3, dash="dash"), name="Forecast",
            mode="lines+markers", marker=dict(size=10),
        ))

        fig_fc.add_trace(go.Scatter(
            x=[today], y=[cur_price], mode="markers+text",
            marker=dict(size=14, color="#FFD740", symbol="star"),
            text=[f"₹{cur_price:,.0f}"], textposition="top center",
            textfont=dict(color="#FFD740", size=12), name="Current",
        ))

        for i, (d, p, h) in enumerate(zip(fc_dates, fc_prices, ["5d","10d","20d","30d"])):
            direction = dm_pred[h]["direction"]
            color = "#00E676" if direction == "UP" else "#FF5252"
            fig_fc.add_annotation(x=d, y=p, text=f"₹{p:,.0f}<br>{h}",
                showarrow=True, arrowhead=2, arrowcolor=color,
                font=dict(color=color, size=11), bgcolor="#0E1117", bordercolor=color)

        fig_fc.update_layout(
            height=450, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"), margin=dict(t=30, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.02, x=0),
            yaxis_title="Nifty 50",
        )
        fig_fc.update_xaxes(gridcolor="#2D3139")
        fig_fc.update_yaxes(gridcolor="#2D3139")
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Detailed table ────────────────────────────────────────────────
        st.markdown("### Detailed Forecast")
        rows = []
        for h, label in [("5d","5 Days"),("10d","10 Days"),("20d","20 Days"),("30d","30 Days")]:
            p = dm_pred[h]
            rows.append({
                "Horizon": label,
                "Price": f"₹{p['price']:,.2f}",
                "Change": f"{p['pct_change']:+.2f}%",
                "Direction": p["direction"],
                "Low (P5)": f"₹{p['range_low']:,.0f}",
                "High (P95)": f"₹{p['range_high']:,.0f}",
                "Confidence": f"{p['confidence']:.0f}%",
            })
        fc_df = pd.DataFrame(rows)
        def _fc_dir_color(val):
            if val == "UP": return "color: #00E676; font-weight:bold"
            return "color: #FF5252; font-weight:bold"
        st.dataframe(fc_df.style.map(_fc_dir_color, subset=["Direction"]),
                     use_container_width=True, hide_index=True)

        # ── Model metrics ─────────────────────────────────────────────────
        st.markdown("### Daily Model Performance")
        dm1, dm2, dm3, dm4 = st.columns(4)
        for col, h, label in [(dm1,"5d","5-Day"),(dm2,"10d","10-Day"),(dm3,"20d","20-Day"),(dm4,"30d","30-Day")]:
            m = dm_metrics.get(h, {})
            with col:
                st.metric(f"{label} MAE", f"{m.get('mae', 0)*100:.2f}%")
                st.caption(f"R²: {m.get('r2', 0):.3f}")
                st.caption(f"Dir acc: {m.get('dir_acc', 0):.1f}%")
                st.caption(f"Samples: {m.get('n_samples', 0):,}")

        # ── Sentiment used ────────────────────────────────────────────────
        st.markdown("### Sentiment Inputs")
        s1, s2, s3 = st.columns(3)
        try:
            from src.sentiment import fetch_news_sentiment, fetch_reddit_sentiment, fetch_market_breadth
            news = fetch_news_sentiment("Nifty India stock market")
            reddit = fetch_reddit_sentiment()
            breadth = fetch_market_breadth()
            ns = news.get("news_sentiment", 0)
            rs = reddit.get("reddit_sentiment", 0)
            br = breadth.get("breadth_pct_above_ema50", 0.5) * 100

            nc = "#00E676" if ns > 0.1 else ("#FF5252" if ns < -0.1 else "#FFD740")
            rc = "#00E676" if rs > 0.1 else ("#FF5252" if rs < -0.1 else "#FFD740")
            bc = "#00E676" if br > 60 else ("#FF5252" if br < 40 else "#FFD740")

            s1.markdown(f"<div style='text-align:center'>"
                f"<div style='color:#AAB4C8'>📰 News Sentiment</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:{nc}'>{ns:+.2f}</div>"
                f"<div style='color:#666'>{news.get('news_count',0)} articles</div></div>",
                unsafe_allow_html=True)
            s2.markdown(f"<div style='text-align:center'>"
                f"<div style='color:#AAB4C8'>💬 Reddit Sentiment</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:{rc}'>{rs:+.2f}</div>"
                f"<div style='color:#666'>{reddit.get('reddit_count',0)} posts</div></div>",
                unsafe_allow_html=True)
            s3.markdown(f"<div style='text-align:center'>"
                f"<div style='color:#AAB4C8'>📊 Market Breadth</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:{bc}'>{br:.0f}%</div>"
                f"<div style='color:#666'>Nifty50 above EMA50</div></div>",
                unsafe_allow_html=True)
        except Exception:
            st.caption("Sentiment data unavailable")


# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Educational purposes only. Not financial advice. "
    "Past accuracy does not guarantee future results."
)
st.caption(f"Last updated: {market['time']}")
