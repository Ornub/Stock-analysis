"""
dashboard.py — NSE Swing Trading Live Dashboard (Streamlit + Plotly)

Run:
    streamlit run dashboard.py

Three panels:
  1. Market Regime Bar  — Nifty close, EMA50/200, VIX, trend regime, pivot levels
  2. Signal Scan Table  — full Nifty-50 scored with BUY/WATCH/HOLD/SELL + grade
  3. Stock Chart        — candlestick, EMA 9/20/50/200, BUY▲/SELL▼ markers,
                          stop-loss & target bands, Volume, RSI-14, MACD histogram

Click any row in the scan table to load that stock's chart.

⚠️ Educational only — not financial advice.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── project imports ──────────────────────────────────────────────────────────
from swing_v2 import (
    NIFTY_50, FEATURE_COLS, MODEL_PATH,
    compute_features, _fetch_market_regime, classify_regime, rank_candidates,
    ADX_MIN, BUY_PROBA, BUY_PROBA_STRONG_NEWS, MIN_CONFIRMATIONS,
    REQUIRE_EMA200, REQUIRE_EMA50, REQUIRE_EMA20, REQUIRE_ADX,
    REQUIRE_MACD, REQUIRE_BREAKOUT, REQUIRE_VOLUME, VOLUME_THRESHOLD,
    GRADE_A_MIN, GRADE_B_MIN, GRADE_C_MIN,
    ATR_STOP_MULT, ATR_TARGET_MULT, ATR_TRAIL_MULT,
    TOP_N_CANDIDATES,
)
from stock_fetcher import fetch_historical_data


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="NSE Swing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 0rem; }
  .metric-label  { font-size: 0.75rem !important; }
  .metric-value  { font-size: 1.1rem !important; font-weight: 700 !important; }
  .stDataFrame   { font-size: 0.82rem; }
  div[data-testid="stHorizontalBlock"] > div { padding: 0 4px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Colours
# =============================================================================

C = {
    "bull":    "#26a69a", "bear":    "#ef5350",
    "ema9":    "#ff9800", "ema20":   "#ffeb3b",
    "ema50":   "#29b6f6", "ema200":  "#ab47bc",
    "vol_up":  "#26a69a", "vol_dn":  "#ef5350",
    "rsi_ln":  "#ff9800", "rsi_ob":  "#ef5350", "rsi_os": "#26a69a",
    "macd_h+": "#26a69a", "macd_h-": "#ef5350",
    "macd_ln": "#ff9800", "sig_ln":  "#29b6f6",
    "stop":    "#ef5350", "target":  "#26a69a",
    "pivot":   "#9e9e9e",
    "buy_mk":  "#00e676", "sell_mk": "#ff1744",
    "bg":      "#131722", "grid":    "#1e2130",
}

SIGNAL_COLORS = {
    "BUY":   "#00e676",
    "WATCH": "#ffeb3b",
    "HOLD":  "#78909c",
    "SELL":  "#ff1744",
}


# =============================================================================
# Data helpers (all cached)
# =============================================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    blob = joblib.load(MODEL_PATH)
    return blob["model"], blob


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_data(symbol: str, lookback_days: int) -> pd.DataFrame | None:
    today     = date.today()
    from_date = today - timedelta(days=lookback_days + 120)   # extra for indicators
    df = fetch_historical_data(symbol, from_date, today)
    if df is None or len(df) < 60:
        return None
    return compute_features(df)


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_data():
    return _fetch_market_regime(5)


@st.cache_data(ttl=900, show_spinner=False)
def run_scan() -> pd.DataFrame:
    """Score all NIFTY_50 stocks with the v3 model and 7-gate confirmation."""
    model, blob = load_model()
    if model is None:
        return pd.DataFrame()

    model_features = blob.get("features", FEATURE_COLS)
    today     = date.today()
    from_date = today - timedelta(days=420)

    market = get_market_data()
    trend_regime, vol_regime = classify_regime(market)

    # Collect latest feature rows for all stocks
    symbol_data: dict[str, pd.DataFrame] = {}
    for sym in NIFTY_50:
        df = fetch_historical_data(sym, from_date, today)
        if df is None or len(df) < 252:
            continue
        df = compute_features(df).dropna(subset=[c for c in model_features if c in compute_features(df).columns])
        if not df.empty:
            symbol_data[sym] = df

    ranked    = rank_candidates(symbol_data, market, top_n=TOP_N_CANDIDATES)
    ranked_set = set(ranked)
    regime_blocked = trend_regime == "BEAR"

    rows = []
    for sym, df in symbol_data.items():
        latest = df.iloc[-1]

        feat_vals = [float(latest.get(c, 0.0) or 0.0) for c in model_features]
        proba     = model.predict_proba(np.array(feat_vals).reshape(1, -1))[0]
        buy_p, sell_p = float(proba[1]), float(proba[0])

        def gv(col, fb=0.0):
            return float(latest.get(col, fb) or fb)

        confirmations = {
            "ema200":   (not REQUIRE_EMA200)   or gv("feat_ema200_ratio") > 0,
            "ema50":    (not REQUIRE_EMA50)    or gv("feat_ema50_ratio")  > 0,
            "ema20":    (not REQUIRE_EMA20)    or gv("feat_ema20_ratio")  > 0,
            "adx":      (not REQUIRE_ADX)      or gv("feat_adx")          >= ADX_MIN,
            "macd":     (not REQUIRE_MACD)     or gv("feat_macd_hist")    >= 0,
            "breakout": (not REQUIRE_BREAKOUT) or gv("feat_dist_20d_high") > 0,
            "volume":   (not REQUIRE_VOLUME)   or gv("feat_volume_ratio") >= VOLUME_THRESHOLD,
        }
        n_pass = sum(confirmations.values())
        failed = [k for k, v in confirmations.items() if not v]

        grade = ("A" if n_pass >= GRADE_A_MIN else
                 "B" if n_pass >= GRADE_B_MIN else
                 "C" if n_pass >= GRADE_C_MIN else "D")

        ema200_ok = confirmations["ema200"]
        rank_ok   = sym in ranked_set
        gate_pass = ema200_ok and rank_ok and not regime_blocked and n_pass >= MIN_CONFIRMATIONS

        if buy_p >= BUY_PROBA and gate_pass:
            signal = "BUY"
        elif buy_p >= BUY_PROBA:
            signal = "WATCH"
        elif sell_p >= BUY_PROBA:
            signal = "SELL"
        else:
            signal = "HOLD"

        atr    = float(latest.get("ATR", 0) or 0)
        price  = float(latest["Close"])
        stop   = round(price - ATR_STOP_MULT  * atr, 1) if signal == "BUY" else None
        target = round(price + ATR_TARGET_MULT * atr, 1) if signal == "BUY" else None

        rows.append({
            "Symbol":  sym,
            "Price":   round(price, 1),
            "Signal":  signal,
            "Grade":   grade if signal in ("BUY", "WATCH") else "",
            "BUY%":    round(buy_p * 100, 1),
            "Confs":   f"{n_pass}/7",
            "Rank":    ranked.index(sym) + 1 if sym in ranked_set else 99,
            "Stop":    stop,
            "Target":  target,
            "Blockers": ",".join(failed[:3]) if signal == "WATCH" else "",
        })

    df_out = pd.DataFrame(rows)
    sig_ord = {"BUY": 0, "WATCH": 1, "HOLD": 2, "SELL": 3}
    df_out["_s"] = df_out["Signal"].map(sig_ord).fillna(4)
    df_out = df_out.sort_values(["_s", "BUY%"], ascending=[True, False]).drop(columns=["_s"])
    return df_out.reset_index(drop=True)


# =============================================================================
# Pivot calculator
# =============================================================================

def pivot_levels(high, low, close) -> dict:
    p  = (high + low + close) / 3
    r1 = 2 * p - low
    r2 = p + (high - low)
    r3 = high + 2 * (p - low)
    s1 = 2 * p - high
    s2 = p - (high - low)
    s3 = low - 2 * (high - p)
    return dict(P=p, R1=r1, R2=r2, R3=r3, S1=s1, S2=s2, S3=s3)


# =============================================================================
# Plotly chart
# =============================================================================

def make_chart(df: pd.DataFrame, symbol: str, lookback: int,
               show_ema9: bool, show_ema20: bool,
               show_ema50: bool, show_ema200: bool,
               model, model_features: list[str]) -> go.Figure:

    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime")
    df = df.tail(lookback)                         # trim to display window
    df = df.reset_index(drop=True)

    c   = df["Close"]
    h   = df["High"]
    lo  = df["Low"]
    vol = df["Volume"]
    dt  = df["DateTime"]

    # Score every bar (simplified: model only, no news for historical)
    feat_matrix = np.column_stack(
        [df.get(col, pd.Series(0, index=df.index)).fillna(0).values for col in model_features]
    )
    df["buy_prob"] = model.predict_proba(feat_matrix)[:, 1]

    # Historical signal markers — EMA200 + probability threshold only
    buy_mask  = (df["buy_prob"] >= BUY_PROBA) & (df.get("feat_ema200_ratio", pd.Series(0, index=df.index)).fillna(0) > 0)
    sell_mask = (df["buy_prob"] <= (1 - BUY_PROBA)) & (df.get("feat_ema200_ratio", pd.Series(0, index=df.index)).fillna(0) < 0)

    # Latest signal for stop/target annotation
    last       = df.iloc[-1]
    last_atr   = float(last.get("ATR", 0) or 0)
    last_price = float(last["Close"])
    last_buy_p = float(last["buy_prob"])
    last_conf  = sum([
        float(last.get("feat_ema200_ratio", 0) or 0) > 0,
        float(last.get("feat_ema50_ratio",  0) or 0) > 0,
        float(last.get("feat_ema20_ratio",  0) or 0) > 0,
        float(last.get("feat_adx", 0) or 0) >= ADX_MIN,
        float(last.get("feat_macd_hist", 0) or 0) >= 0,
        float(last.get("feat_dist_20d_high", -1) or -1) > 0,
        float(last.get("feat_volume_ratio", 0) or 0) >= VOLUME_THRESHOLD,
    ])

    show_buy_levels  = last_buy_p >= BUY_PROBA  and last_conf >= MIN_CONFIRMATIONS
    show_sell_levels = last_buy_p <= (1 - BUY_PROBA)

    # ── Subplots ─────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        vertical_spacing=0.02,
        subplot_titles=("", "Volume", "RSI-14", "MACD"),
    )

    # ── Row 1: Candlestick ───────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dt, open=df["Open"], high=h, low=lo, close=c,
        increasing_line_color=C["bull"], decreasing_line_color=C["bear"],
        increasing_fillcolor=C["bull"], decreasing_fillcolor=C["bear"],
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    # EMA overlays
    ema9_s   = c.ewm(span=9,   adjust=False).mean()
    ema20_s  = c.ewm(span=20,  adjust=False).mean()
    ema50_s  = c.ewm(span=50,  adjust=False).mean()
    ema200_s = c.ewm(span=200, adjust=False).mean()

    if show_ema9:
        fig.add_trace(go.Scatter(x=dt, y=ema9_s,   name="EMA9",
                                 line=dict(color=C["ema9"],   width=1)), row=1, col=1)
    if show_ema20:
        fig.add_trace(go.Scatter(x=dt, y=ema20_s,  name="EMA20",
                                 line=dict(color=C["ema20"],  width=1)), row=1, col=1)
    if show_ema50:
        fig.add_trace(go.Scatter(x=dt, y=ema50_s,  name="EMA50",
                                 line=dict(color=C["ema50"],  width=1.5)), row=1, col=1)
    if show_ema200:
        fig.add_trace(go.Scatter(x=dt, y=ema200_s, name="EMA200",
                                 line=dict(color=C["ema200"], width=2)), row=1, col=1)

    # BUY signal markers
    buy_df = df[buy_mask]
    if not buy_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_df["DateTime"], y=buy_df["Low"] * 0.992,
            mode="markers", name="BUY signal",
            marker=dict(symbol="triangle-up", size=12,
                        color=C["buy_mk"], line=dict(color="white", width=1)),
            hovertemplate="<b>BUY</b> %{x}<br>Prob: %{customdata:.1%}",
            customdata=buy_df["buy_prob"].values,
        ), row=1, col=1)

    # SELL signal markers
    sell_df = df[sell_mask]
    if not sell_df.empty:
        fig.add_trace(go.Scatter(
            x=sell_df["DateTime"], y=sell_df["High"] * 1.008,
            mode="markers", name="SELL signal",
            marker=dict(symbol="triangle-down", size=12,
                        color=C["sell_mk"], line=dict(color="white", width=1)),
        ), row=1, col=1)

    # Stop-loss + Target bands for current signal
    x_start = dt.iloc[max(-20, -len(dt))]
    x_end   = dt.iloc[-1] + timedelta(days=7)

    if show_buy_levels and last_atr > 0:
        stop_lvl   = last_price - ATR_STOP_MULT  * last_atr
        target_lvl = last_price + ATR_TARGET_MULT * last_atr
        trail_lvl  = last_price - ATR_TRAIL_MULT  * last_atr

        # Shaded stop zone
        fig.add_hrect(y0=stop_lvl * 0.998, y1=stop_lvl * 1.002,
                      fillcolor="rgba(239,83,80,0.15)", line_width=0,
                      annotation_text=f"STOP ₹{stop_lvl:,.1f}",
                      annotation_font_color=C["stop"],
                      annotation_position="right", row=1, col=1)
        # Shaded target zone
        fig.add_hrect(y0=target_lvl * 0.998, y1=target_lvl * 1.002,
                      fillcolor="rgba(38,166,154,0.15)", line_width=0,
                      annotation_text=f"TARGET ₹{target_lvl:,.1f}",
                      annotation_font_color=C["target"],
                      annotation_position="right", row=1, col=1)
        # Trailing stop line
        fig.add_hline(y=trail_lvl, line_dash="dot",
                      line_color="rgba(239,83,80,0.5)", line_width=1,
                      annotation_text=f"Trail ₹{trail_lvl:,.1f}",
                      annotation_font_color="rgba(239,83,80,0.7)",
                      row=1, col=1)

    # Pivot levels (computed from last 2 bars)
    if len(df) >= 2:
        ph = float(h.iloc[-2]); pl = float(lo.iloc[-2]); pc = float(c.iloc[-2])
        pv = pivot_levels(ph, pl, pc)
        pivot_style = [
            ("R2", pv["R2"], "rgba(239,83,80,0.6)",  "dot"),
            ("R1", pv["R1"], "rgba(239,83,80,0.9)",  "dash"),
            ("P",  pv["P"],  "rgba(158,158,158,0.9)","solid"),
            ("S1", pv["S1"], "rgba(38,166,154,0.9)", "dash"),
            ("S2", pv["S2"], "rgba(38,166,154,0.6)", "dot"),
        ]
        for lbl, lvl, col, dash in pivot_style:
            fig.add_hline(y=lvl, line_dash=dash, line_color=col, line_width=1,
                          annotation_text=f"{lbl} {lvl:,.0f}",
                          annotation_font_color=col,
                          annotation_position="left", row=1, col=1)

    # ── Row 2: Volume ────────────────────────────────────────────────────────
    vol_colors = [C["bull"] if cl >= op else C["bear"]
                  for cl, op in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=dt, y=vol, name="Volume",
        marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)
    vol_ma = vol.rolling(20).mean()
    fig.add_trace(go.Scatter(x=dt, y=vol_ma, name="Vol MA20",
                             line=dict(color=C["ema20"], width=1),
                             showlegend=False), row=2, col=1)

    # ── Row 3: RSI ───────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).where(delta < 0, 0.0).ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    rsi   = 100 - 100 / (1 + gain / loss)
    rsi_color = [C["rsi_ob"] if v >= 70 else C["rsi_os"] if v <= 30 else C["rsi_ln"]
                 for v in rsi.fillna(50)]

    fig.add_trace(go.Scatter(x=dt, y=rsi, name="RSI",
                             line=dict(color=C["rsi_ln"], width=1.5),
                             showlegend=False), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)",
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.08)",
                  line_width=0, row=3, col=1)
    for lvl, lbl in [(70, "OB 70"), (30, "OS 30")]:
        fig.add_hline(y=lvl, line_dash="dot",
                      line_color="rgba(255,255,255,0.2)", line_width=1, row=3, col=1)

    # ── Row 4: MACD ──────────────────────────────────────────────────────────
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd_l = ema12 - ema26
    sig_l  = macd_l.ewm(span=9, adjust=False).mean()
    hist   = macd_l - sig_l
    hist_colors = [C["macd_h+"] if v >= 0 else C["macd_h-"] for v in hist.fillna(0)]

    fig.add_trace(go.Bar(x=dt, y=hist, name="MACD Hist",
                         marker_color=hist_colors, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=dt, y=macd_l, name="MACD",
                             line=dict(color=C["macd_ln"], width=1.5),
                             showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=dt, y=sig_l, name="Signal",
                             line=dict(color=C["sig_ln"], width=1, dash="dot"),
                             showlegend=False), row=4, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1, row=4, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        margin=dict(l=10, r=120, t=30, b=10),
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        font=dict(family="monospace", size=11),
        title=dict(
            text=f"<b>{symbol}</b>  ₹{last_price:,.1f}  "
                 f"<span style='color:{'#00e676' if show_buy_levels else '#78909c'}'>"
                 f"BUY% {last_buy_p*100:.1f}  Confs {last_conf}/7</span>",
            x=0.01, font=dict(size=14),
        ),
    )
    fig.update_xaxes(gridcolor=C["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=C["grid"], showgrid=True)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",    row=2, col=1)
    fig.update_yaxes(title_text="RSI",       row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",      row=4, col=1)

    return fig


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    selected_symbol = st.selectbox(
        "📌 Stock", NIFTY_50, index=NIFTY_50.index("BAJFINANCE"),
        help="Select a stock to view its chart"
    )

    lookback = st.selectbox(
        "📅 Chart Lookback",
        [30, 60, 90, 180, 365],
        index=2,
        format_func=lambda x: f"{x} days",
    )

    st.markdown("#### EMA Lines")
    show_ema9   = st.checkbox("EMA 9",   value=True)
    show_ema20  = st.checkbox("EMA 20",  value=True)
    show_ema50  = st.checkbox("EMA 50",  value=True)
    show_ema200 = st.checkbox("EMA 200", value=True)

    st.markdown("---")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("⚠️ Educational only\nNot financial advice.")


# =============================================================================
# Load model
# =============================================================================

model, blob = load_model()
if model is None:
    st.error('No trained model found. Run `python swing_v2.py train` first.')
    st.stop()

model_features = blob.get("features", FEATURE_COLS)

# =============================================================================
# Header — title + quick status
# =============================================================================

st.markdown("## 📈 NSE Swing Trading Dashboard")

# =============================================================================
# Panel 1 — Market Regime Bar
# =============================================================================

with st.spinner("Loading market data…"):
    market = get_market_data()

trend_regime, vol_regime = classify_regime(market)

# Pull latest market row
latest_mkt = {}
if market is not None and not market.empty:
    latest_mkt = market.sort_values("DateTime").iloc[-1].to_dict()

nifty_close  = latest_mkt.get("nifty_close",  0)
nifty_50dma  = latest_mkt.get("nifty_50dma",  0)
nifty_200dma = latest_mkt.get("nifty_200dma", 0)
vix_raw      = latest_mkt.get("feat_vix_level", 0.15) * 100
mkt_ret_1d   = latest_mkt.get("feat_market_return_1d", 0) * 100

# Regime colour
regime_color = {"BULL": "🟢", "SIDEWAYS": "🟡", "BEAR": "🔴"}.get(trend_regime, "⚪")
vol_color    = {"LOW": "🟢", "NORMAL": "🟢", "HIGH": "🟡", "EXTREME": "🔴"}.get(vol_regime, "⚪")

# Pivot levels from yesterday
piv = {}
if market is not None and len(market) >= 2:
    import yfinance as yf
    try:
        ndf = yf.download("^NSEI", period="5d", auto_adjust=True, progress=False)
        if isinstance(ndf.columns, pd.MultiIndex):
            ndf.columns = ndf.columns.get_level_values(0)
        if len(ndf) >= 2:
            ph = float(ndf["High"].iloc[-2])
            pl = float(ndf["Low"].iloc[-2])
            pc = float(ndf["Close"].iloc[-2])
            piv = pivot_levels(ph, pl, pc)
    except Exception:
        pass

st.markdown("### 🌐 Market Regime")
mcols = st.columns(10)

mcols[0].metric("Nifty", f"{nifty_close:,.0f}", f"{mkt_ret_1d:+.2f}%")
mcols[1].metric("EMA50",  f"{nifty_50dma:,.0f}")
mcols[2].metric("EMA200", f"{nifty_200dma:,.0f}")
mcols[3].metric("VIX",    f"{vix_raw:.1f}")
mcols[4].metric("Trend",  f"{regime_color} {trend_regime}")
mcols[5].metric("Vol",    f"{vol_color} {vol_regime}")

if piv:
    mcols[6].metric("R2", f"{piv['R2']:,.0f}")
    mcols[7].metric("R1", f"{piv['R1']:,.0f}")
    mcols[8].metric("Pivot", f"{piv['P']:,.0f}")
    mcols[9].metric("S1/S2", f"{piv['S1']:,.0f} / {piv['S2']:,.0f}")

# Regime warning banners
if trend_regime == "BEAR":
    st.warning("⚠️ **BEAR Regime** — New long entries blocked by regime filter. Showing scores for monitoring.")
elif vol_regime in ("HIGH", "EXTREME"):
    st.warning(f"⚠️ **VIX {vol_regime}** ({vix_raw:.1f}) — Reduce position sizes; elevated volatility.")

st.divider()

# =============================================================================
# Panel 2 + 3 — Signal Table (left) + Chart (right)
# =============================================================================

left_col, right_col = st.columns([1, 2.8], gap="medium")

# ── LEFT: Signal scan table ──────────────────────────────────────────────────

with left_col:
    st.markdown("### 🔍 Signal Scan — Nifty 50")

    with st.spinner("Running model on all 45 stocks…"):
        scan_df = run_scan()

    if scan_df.empty:
        st.info("No scan data — model may need retraining.")
    else:
        def row_color(signal):
            return {
                "BUY":   "background-color: #003d1f; color: #00e676",
                "WATCH": "background-color: #3d3200; color: #ffeb3b",
                "SELL":  "background-color: #3d0000; color: #ff1744",
                "HOLD":  "",
            }.get(signal, "")

        styled = scan_df[["Symbol", "Price", "Signal", "Grade", "BUY%", "Confs", "Rank"]].style
        styled = styled.apply(
            lambda row: [row_color(row["Signal"])] * len(row), axis=1
        )
        styled = styled.format({"BUY%": "{:.1f}", "Price": "{:,.1f}"})

        # Clickable table — Streamlit 1.35+ supports on_select
        try:
            event = st.dataframe(
                styled,
                use_container_width=True,
                height=500,
                on_select="rerun",
                selection_mode="single-row",
                key="scan_table",
            )
            if event.selection and event.selection.rows:
                clicked_sym = scan_df.iloc[event.selection.rows[0]]["Symbol"]
                selected_symbol = clicked_sym   # override sidebar choice
        except Exception:
            # Older Streamlit — display without click
            st.dataframe(styled, use_container_width=True, height=500)

        # Summary counts
        sig_counts = scan_df["Signal"].value_counts()
        cnt_cols = st.columns(4)
        for i, (sig, emoji) in enumerate([("BUY","🟢"),("WATCH","🟡"),("HOLD","⚪"),("SELL","🔴")]):
            cnt_cols[i].metric(f"{emoji} {sig}", sig_counts.get(sig, 0))

        # Download button
        st.download_button(
            "⬇ Download Signals CSV",
            scan_df.to_csv(index=False),
            file_name=f"nse_signals_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── RIGHT: Stock chart ───────────────────────────────────────────────────────

with right_col:
    st.markdown(f"### 📊 Chart — {selected_symbol}")

    with st.spinner(f"Loading {selected_symbol} data…"):
        stock_df = get_stock_data(selected_symbol, lookback + 200)

    if stock_df is None or stock_df.empty:
        st.error(f"No data for {selected_symbol}.")
    else:
        # Quick stats row above chart
        last_row   = stock_df.iloc[-1]
        prev_row   = stock_df.iloc[-2]
        last_price = float(last_row["Close"])
        day_chg    = (last_price / float(prev_row["Close"]) - 1) * 100
        week_chg   = (last_price / float(stock_df.iloc[-6]["Close"]) - 1) * 100 if len(stock_df) >= 6 else 0
        high52     = float(stock_df["High"].tail(252).max())
        low52      = float(stock_df["Low"].tail(252).min())
        atr_val    = float(last_row.get("ATR", 0) or 0)
        rsi_val    = float(last_row.get("feat_rsi", 50) or 50)
        adx_val    = float(last_row.get("feat_adx", 0) or 0) * 100

        sc = st.columns(7)
        sc[0].metric("Price",    f"₹{last_price:,.1f}", f"{day_chg:+.2f}%")
        sc[1].metric("1W",       f"{week_chg:+.2f}%")
        sc[2].metric("52W High", f"₹{high52:,.0f}")
        sc[3].metric("52W Low",  f"₹{low52:,.0f}")
        sc[4].metric("ATR-14",   f"₹{atr_val:,.0f}")
        sc[5].metric("RSI-14",   f"{rsi_val:.1f}")
        sc[6].metric("ADX-14",   f"{adx_val:.1f}")

        fig = make_chart(
            stock_df, selected_symbol, lookback,
            show_ema9, show_ema20, show_ema50, show_ema200,
            model, model_features,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Signal detail for this stock
        if not scan_df.empty:
            stock_row = scan_df[scan_df["Symbol"] == selected_symbol]
            if not stock_row.empty:
                sr = stock_row.iloc[0]
                sig = sr["Signal"]
                color = SIGNAL_COLORS.get(sig, "#78909c")

                st.markdown(f"""
<div style='background:#1e2130;padding:12px 16px;border-radius:8px;
     border-left:4px solid {color};margin-top:4px'>
  <b style='color:{color};font-size:1.1em'>{sig} {sr.get("Grade","")}</b>
  &nbsp;&nbsp;BUY% <b>{sr["BUY%"]}</b>
  &nbsp;&nbsp;Confirmations <b>{sr["Confs"]}</b>
  &nbsp;&nbsp;Rank <b>#{sr["Rank"]}</b>
  {f"&nbsp;&nbsp;Stop <b>₹{sr['Stop']:,.1f}</b>" if pd.notna(sr.get("Stop")) else ""}
  {f"&nbsp;&nbsp;Target <b>₹{sr['Target']:,.1f}</b>" if pd.notna(sr.get("Target")) else ""}
  {f"<br><span style='color:#ffeb3b;font-size:0.85em'>Blockers: {sr['Blockers']}</span>" if sr.get("Blockers") else ""}
</div>
""", unsafe_allow_html=True)

st.divider()
st.caption(
    f"Model trained {blob.get('trained_at','?')} · {blob.get('n_train',0):,} samples · "
    f"{len(model_features)} features · val acc {blob.get('val_acc',0)*100:.1f}% · "
    "⚠️ Educational only — not financial advice"
)
