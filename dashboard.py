"""
dashboard.py — NSE Swing Trading Live Dashboard v2  (Streamlit + Plotly)

Tabs:
  1. 📊 Dashboard   — Market regime, signal scan, interactive chart
  2. 💼 Portfolio   — Monthly allocation with ATR-based sizing
  3. 🗺  Sectors     — Sector heatmap & rotation signals
  4. 📰 Deep Dive   — Extended stock analysis with news & fundamentals

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
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from swing_v2 import (
    NIFTY_50, FEATURE_COLS, MODEL_PATH,
    compute_features, _fetch_market_regime, classify_regime, rank_candidates,
    ADX_MIN, BUY_PROBA, MIN_CONFIRMATIONS,
    REQUIRE_EMA200, REQUIRE_EMA50, REQUIRE_EMA20, REQUIRE_ADX,
    REQUIRE_MACD, REQUIRE_BREAKOUT, REQUIRE_VOLUME, VOLUME_THRESHOLD,
    GRADE_A_MIN, GRADE_B_MIN, GRADE_C_MIN,
    ATR_STOP_MULT, ATR_TARGET_MULT, ATR_TRAIL_MULT,
    TOP_N_CANDIDATES, SECTOR_MAP,
)
from stock_fetcher import fetch_historical_data


# ─────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Swing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 0.6rem; padding-bottom: 0rem; }
  .metric-label  { font-size: 0.72rem !important; color: #90a4ae !important; }
  .metric-value  { font-size: 1.05rem !important; font-weight: 700 !important; }
  .stDataFrame   { font-size: 0.82rem; }
  div[data-testid="stHorizontalBlock"] > div { padding: 0 3px; }
  .signal-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;
  }
  .badge-BUY   { background: #003d1f; color: #00e676; border: 1px solid #00e676; }
  .badge-WATCH { background: #3d3200; color: #ffeb3b; border: 1px solid #ffeb3b; }
  .badge-SELL  { background: #3d0000; color: #ff1744; border: 1px solid #ff1744; }
  .badge-HOLD  { background: #1e2130; color: #78909c; border: 1px solid #455a64; }
  .grade-A { color: #ffd700; font-weight: 900; }
  .grade-B { color: #c0c0c0; font-weight: 700; }
  .grade-C { color: #cd7f32; font-weight: 600; }
  .regime-bull { background: linear-gradient(90deg,#003d1f,#131722); border-left: 4px solid #00e676; }
  .regime-bear { background: linear-gradient(90deg,#3d0000,#131722); border-left: 4px solid #ff1744; }
  .regime-side { background: linear-gradient(90deg,#3d3200,#131722); border-left: 4px solid #ffeb3b; }
  .info-card {
    background: #1e2130; border-radius: 10px; padding: 14px 18px;
    border: 1px solid #2a2f45; margin: 4px 0;
  }
  div[data-baseweb="tab-list"] button { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "bull": "#26a69a", "bear": "#ef5350",
    "ema9": "#ff9800", "ema20": "#ffeb3b",
    "ema50": "#29b6f6", "ema200": "#ab47bc",
    "vol_up": "#26a69a", "vol_dn": "#ef5350",
    "rsi_ln": "#ff9800",
    "macd_h+": "#26a69a", "macd_h-": "#ef5350",
    "macd_ln": "#ff9800", "sig_ln": "#29b6f6",
    "stop": "#ef5350", "target": "#26a69a",
    "bb": "#4fc3f7",
    "supertrend_bull": "#26a69a", "supertrend_bear": "#ef5350",
    "buy_mk": "#00e676", "sell_mk": "#ff1744",
    "bg": "#131722", "grid": "#1e2130",
}

SIGNAL_COLORS = {
    "BUY": "#00e676", "WATCH": "#ffeb3b",
    "HOLD": "#78909c", "SELL": "#ff1744",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    blob = joblib.load(MODEL_PATH)
    return blob["model"], blob


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_data(symbol: str, lookback_days: int) -> pd.DataFrame | None:
    today     = date.today()
    from_date = today - timedelta(days=lookback_days + 150)
    df = fetch_historical_data(symbol, from_date, today)
    if df is None or len(df) < 60:
        return None
    return compute_features(df)


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_data():
    return _fetch_market_regime(5)


@st.cache_data(ttl=900, show_spinner=False)
def run_scan() -> pd.DataFrame:
    """Score all NIFTY_50 stocks with v3 model + 7-gate gates."""
    model, blob = load_model()
    if model is None:
        return pd.DataFrame()

    model_features = blob.get("features", FEATURE_COLS)
    today     = date.today()
    from_date = today - timedelta(days=420)

    market = get_market_data()
    trend_regime, vol_regime = classify_regime(market)

    symbol_data: dict[str, pd.DataFrame] = {}
    for sym in NIFTY_50:
        df = fetch_historical_data(sym, from_date, today)
        if df is None or len(df) < 200:
            continue
        df_f = compute_features(df)
        valid_cols = [c for c in model_features if c in df_f.columns]
        df_f = df_f.dropna(subset=valid_cols)
        if not df_f.empty:
            symbol_data[sym] = df_f

    ranked     = rank_candidates(symbol_data, market, top_n=TOP_N_CANDIDATES)
    ranked_set = set(ranked)
    regime_blocked = trend_regime == "BEAR"

    rows = []
    for sym, df in symbol_data.items():
        latest = df.iloc[-1]
        prev   = df.iloc[-2] if len(df) >= 2 else latest

        feat_vals = [float(latest.get(c, 0.0) or 0.0) for c in model_features]
        proba     = model.predict_proba(np.array(feat_vals).reshape(1, -1))[0]
        buy_p, sell_p = float(proba[1]), float(proba[0])

        def gv(col, fb=0.0):
            return float(latest.get(col, fb) or fb)

        confirmations = {
            "EMA200":   (not REQUIRE_EMA200)   or gv("feat_ema200_ratio") > 0,
            "EMA50":    (not REQUIRE_EMA50)    or gv("feat_ema50_ratio")  > 0,
            "EMA20":    (not REQUIRE_EMA20)    or gv("feat_ema20_ratio")  > 0,
            "ADX":      (not REQUIRE_ADX)      or gv("feat_adx")          >= ADX_MIN,
            "MACD":     (not REQUIRE_MACD)     or gv("feat_macd_hist")    >= 0,
            "Breakout": (not REQUIRE_BREAKOUT) or gv("feat_dist_20d_high") > 0,
            "Volume":   (not REQUIRE_VOLUME)   or gv("feat_volume_ratio") >= VOLUME_THRESHOLD,
        }
        n_pass = sum(confirmations.values())
        passed  = [k for k, v in confirmations.items() if v]
        failed  = [k for k, v in confirmations.items() if not v]

        grade = ("A" if n_pass >= GRADE_A_MIN else
                 "B" if n_pass >= GRADE_B_MIN else
                 "C" if n_pass >= GRADE_C_MIN else "D")

        ema200_ok = confirmations["EMA200"]
        gate_pass = ema200_ok and sym in ranked_set and not regime_blocked and n_pass >= MIN_CONFIRMATIONS

        if buy_p >= BUY_PROBA and gate_pass:
            signal = "BUY"
        elif buy_p >= BUY_PROBA:
            signal = "WATCH"
        elif sell_p >= BUY_PROBA:
            signal = "SELL"
        else:
            signal = "HOLD"

        price  = float(latest["Close"])
        prev_p = float(prev["Close"])
        chg    = (price / prev_p - 1) * 100 if prev_p else 0
        atr    = float(latest.get("ATR", 0) or 0)
        rsi    = float(latest.get("feat_rsi", 50) or 50)
        adx    = float(latest.get("feat_adx", 0) or 0) * 100
        vol_r  = float(latest.get("feat_volume_ratio", 1) or 1)

        stop   = round(price - ATR_STOP_MULT  * atr, 1) if signal == "BUY" else None
        target = round(price + ATR_TARGET_MULT * atr, 1) if signal == "BUY" else None

        rows.append({
            "Symbol":   sym,
            "Sector":   SECTOR_MAP.get(sym, "Other"),
            "Price":    round(price, 1),
            "1D%":      round(chg, 2),
            "Signal":   signal,
            "Grade":    grade if signal in ("BUY", "WATCH") else "",
            "BUY%":     round(buy_p * 100, 1),
            "Confs":    f"{n_pass}/7",
            "RSI":      round(rsi, 1),
            "ADX":      round(adx, 1),
            "VolRatio": round(vol_r, 2),
            "Rank":     ranked.index(sym) + 1 if sym in ranked_set else 99,
            "Stop":     stop,
            "Target":   target,
            "Passed":   ", ".join(passed),
            "Blockers": ", ".join(failed[:3]) if signal == "WATCH" else "",
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    sig_ord = {"BUY": 0, "WATCH": 1, "HOLD": 2, "SELL": 3}
    df_out["_s"] = df_out["Signal"].map(sig_ord).fillna(4)
    return df_out.sort_values(["_s", "BUY%"], ascending=[True, False]).drop(columns=["_s"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def calc_bollinger(close: pd.Series, window=20, num_std=2.0):
    mid  = close.rolling(window).mean()
    std  = close.rolling(window).std()
    return mid - num_std * std, mid, mid + num_std * std


def calc_supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    """ATR-based Supertrend indicator."""
    hl2  = (df["High"] + df["Low"]) / 2
    atr  = df["High"].combine(df["Low"], max).sub(df["Low"]).ewm(com=period - 1, adjust=False).mean()
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    trend   = pd.Series(1, index=df.index)
    final_u = upper.copy()
    final_l = lower.copy()

    for i in range(1, len(df)):
        final_u.iloc[i] = upper.iloc[i] if upper.iloc[i] < final_u.iloc[i - 1] or df["Close"].iloc[i - 1] > final_u.iloc[i - 1] else final_u.iloc[i - 1]
        final_l.iloc[i] = lower.iloc[i] if lower.iloc[i] > final_l.iloc[i - 1] or df["Close"].iloc[i - 1] < final_l.iloc[i - 1] else final_l.iloc[i - 1]
        if trend.iloc[i - 1] == -1 and df["Close"].iloc[i] > final_u.iloc[i]:
            trend.iloc[i] = 1
        elif trend.iloc[i - 1] == 1 and df["Close"].iloc[i] < final_l.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    line = pd.Series(np.where(trend == 1, final_l, final_u), index=df.index)
    return trend, line


def pivot_levels(high, low, close) -> dict:
    p  = (high + low + close) / 3
    return dict(
        P=p, R1=2*p-low, R2=p+(high-low), R3=high+2*(p-low),
        S1=2*p-high, S2=p-(high-low), S3=low-2*(high-p),
    )


def find_support_resistance(df: pd.DataFrame, n_levels=3):
    """Simple pivot-based S/R from recent highs and lows."""
    highs = df["High"].rolling(5, center=True).max()
    lows  = df["Low"].rolling(5, center=True).min()
    res = sorted(highs.dropna().nlargest(n_levels).tolist(), reverse=True)
    sup = sorted(lows.dropna().nsmallest(n_levels).tolist())
    return res, sup


# ─────────────────────────────────────────────────────────────────────────────
# Main chart builder
# ─────────────────────────────────────────────────────────────────────────────

def make_chart(
    df: pd.DataFrame, symbol: str, lookback: int,
    show_ema9: bool, show_ema20: bool, show_ema50: bool, show_ema200: bool,
    show_bb: bool, show_supertrend: bool, show_pivots: bool, show_sr: bool,
    model, model_features: list[str],
) -> go.Figure:

    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").tail(lookback).reset_index(drop=True)

    c   = df["Close"]
    h   = df["High"]
    lo  = df["Low"]
    vol = df["Volume"]
    dt  = df["DateTime"]

    feat_matrix  = np.column_stack(
        [df.get(col, pd.Series(0, index=df.index)).fillna(0).values for col in model_features]
    )
    df["buy_prob"] = model.predict_proba(feat_matrix)[:, 1]

    buy_mask  = (df["buy_prob"] >= BUY_PROBA) & (df.get("feat_ema200_ratio", pd.Series(0, index=df.index)).fillna(0) > 0)
    sell_mask = (df["buy_prob"] <= (1 - BUY_PROBA)) & (df.get("feat_ema200_ratio", pd.Series(0, index=df.index)).fillna(0) < 0)

    last       = df.iloc[-1]
    last_atr   = float(last.get("ATR", 0) or 0)
    last_price = float(last["Close"])
    last_buy_p = float(last["buy_prob"])
    last_conf  = sum([
        float(last.get("feat_ema200_ratio", 0) or 0) > 0,
        float(last.get("feat_ema50_ratio",  0) or 0) > 0,
        float(last.get("feat_ema20_ratio",  0) or 0) > 0,
        float(last.get("feat_adx",          0) or 0) >= ADX_MIN,
        float(last.get("feat_macd_hist",    0) or 0) >= 0,
        float(last.get("feat_dist_20d_high",-1) or -1) > 0,
        float(last.get("feat_volume_ratio", 0) or 0) >= VOLUME_THRESHOLD,
    ])
    show_buy_levels  = last_buy_p >= BUY_PROBA and last_conf >= MIN_CONFIRMATIONS
    show_sell_levels = last_buy_p <= (1 - BUY_PROBA)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        vertical_spacing=0.015,
        subplot_titles=("", "Volume", "RSI-14", "MACD"),
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dt, open=df["Open"], high=h, low=lo, close=c,
        increasing_line_color=C["bull"], decreasing_line_color=C["bear"],
        increasing_fillcolor=C["bull"],  decreasing_fillcolor=C["bear"],
        name="OHLC", showlegend=False, whiskerwidth=0.6,
    ), row=1, col=1)

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    if show_bb:
        bb_lo, bb_mid, bb_hi = calc_bollinger(c)
        fig.add_trace(go.Scatter(x=dt, y=bb_hi,  name="BB Upper",
                                 line=dict(color=C["bb"], width=1, dash="dot"),
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=dt, y=bb_mid, name="BB Mid",
                                 line=dict(color=C["bb"], width=1),
                                 showlegend=False, opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dt, y=bb_lo, name="BB",
            line=dict(color=C["bb"], width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(79,195,247,0.05)",
        ), row=1, col=1)

    # ── EMAs ─────────────────────────────────────────────────────────────────
    ema_defs = [
        (9,   C["ema9"],   1.0, show_ema9,   "EMA9"),
        (20,  C["ema20"],  1.0, show_ema20,  "EMA20"),
        (50,  C["ema50"],  1.5, show_ema50,  "EMA50"),
        (200, C["ema200"], 2.0, show_ema200, "EMA200"),
    ]
    for span, color, width, show, label in ema_defs:
        ema_s = c.ewm(span=span, adjust=False).mean()
        if show:
            fig.add_trace(go.Scatter(x=dt, y=ema_s, name=label,
                                     line=dict(color=color, width=width)), row=1, col=1)

    # ── Supertrend ───────────────────────────────────────────────────────────
    if show_supertrend and len(df) > 20:
        st_trend, st_line = calc_supertrend(df)
        bull_idx = st_trend == 1
        bear_idx = st_trend == -1
        if bull_idx.any():
            fig.add_trace(go.Scatter(
                x=dt[bull_idx], y=st_line[bull_idx], name="Supertrend↑",
                mode="lines", line=dict(color=C["supertrend_bull"], width=2),
            ), row=1, col=1)
        if bear_idx.any():
            fig.add_trace(go.Scatter(
                x=dt[bear_idx], y=st_line[bear_idx], name="Supertrend↓",
                mode="lines", line=dict(color=C["supertrend_bear"], width=2),
            ), row=1, col=1)

    # ── Support / Resistance ─────────────────────────────────────────────────
    if show_sr and len(df) > 30:
        res_levels, sup_levels = find_support_resistance(df.tail(60))
        for lvl in res_levels[:2]:
            fig.add_hline(y=lvl, line_dash="dot",
                          line_color="rgba(239,83,80,0.4)", line_width=1,
                          annotation_text=f"R {lvl:,.0f}",
                          annotation_font_color="rgba(239,83,80,0.7)",
                          annotation_position="right", row=1, col=1)
        for lvl in sup_levels[:2]:
            fig.add_hline(y=lvl, line_dash="dot",
                          line_color="rgba(38,166,154,0.4)", line_width=1,
                          annotation_text=f"S {lvl:,.0f}",
                          annotation_font_color="rgba(38,166,154,0.7)",
                          annotation_position="right", row=1, col=1)

    # ── Buy / Sell signal markers ────────────────────────────────────────────
    buy_df = df[buy_mask]
    if not buy_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_df["DateTime"], y=buy_df["Low"] * 0.991,
            mode="markers", name="BUY ▲",
            marker=dict(symbol="triangle-up", size=13,
                        color=C["buy_mk"], line=dict(color="white", width=1)),
            hovertemplate="<b>BUY</b> %{x|%d %b}<br>Prob: %{customdata:.1%}<extra></extra>",
            customdata=buy_df["buy_prob"].values,
        ), row=1, col=1)
    sell_df = df[sell_mask]
    if not sell_df.empty:
        fig.add_trace(go.Scatter(
            x=sell_df["DateTime"], y=sell_df["High"] * 1.009,
            mode="markers", name="SELL ▼",
            marker=dict(symbol="triangle-down", size=13,
                        color=C["sell_mk"], line=dict(color="white", width=1)),
            hovertemplate="<b>SELL</b> %{x|%d %b}<extra></extra>",
        ), row=1, col=1)

    # ── Stop / Target / Trailing bands ───────────────────────────────────────
    if show_buy_levels and last_atr > 0:
        stop_lvl   = last_price - ATR_STOP_MULT  * last_atr
        target_lvl = last_price + ATR_TARGET_MULT * last_atr
        trail_lvl  = last_price - ATR_TRAIL_MULT  * last_atr
        fig.add_hrect(y0=stop_lvl * 0.998, y1=stop_lvl * 1.002,
                      fillcolor="rgba(239,83,80,0.18)", line_width=0,
                      annotation_text=f"STOP ₹{stop_lvl:,.0f}",
                      annotation_font_color=C["stop"],
                      annotation_position="right", row=1, col=1)
        fig.add_hrect(y0=target_lvl * 0.998, y1=target_lvl * 1.002,
                      fillcolor="rgba(38,166,154,0.18)", line_width=0,
                      annotation_text=f"TARGET ₹{target_lvl:,.0f}",
                      annotation_font_color=C["target"],
                      annotation_position="right", row=1, col=1)
        fig.add_hline(y=trail_lvl, line_dash="dot",
                      line_color="rgba(239,83,80,0.5)", line_width=1.2,
                      annotation_text=f"Trail ₹{trail_lvl:,.0f}",
                      annotation_font_color="rgba(239,83,80,0.7)",
                      row=1, col=1)

    # ── Pivot levels ─────────────────────────────────────────────────────────
    if show_pivots and len(df) >= 2:
        ph = float(h.iloc[-2]); pl = float(lo.iloc[-2]); pc = float(c.iloc[-2])
        pv = pivot_levels(ph, pl, pc)
        for lbl, lvl, col, dash in [
            ("R2", pv["R2"], "rgba(239,83,80,0.55)", "dot"),
            ("R1", pv["R1"], "rgba(239,83,80,0.85)", "dash"),
            ("P",  pv["P"],  "rgba(158,158,158,0.9)", "solid"),
            ("S1", pv["S1"], "rgba(38,166,154,0.85)", "dash"),
            ("S2", pv["S2"], "rgba(38,166,154,0.55)", "dot"),
        ]:
            fig.add_hline(y=lvl, line_dash=dash, line_color=col, line_width=1,
                          annotation_text=f"{lbl} {lvl:,.0f}",
                          annotation_font_color=col,
                          annotation_position="left", row=1, col=1)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_colors = [C["bull"] if cl >= op else C["bear"]
                  for cl, op in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=dt, y=vol, name="Volume",
                         marker_color=vol_colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=dt, y=vol.rolling(20).mean(), name="Vol MA20",
                             line=dict(color=C["ema20"], width=1),
                             showlegend=False), row=2, col=1)

    # ── RSI ──────────────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).where(delta < 0, 0.0).ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    rsi   = 100 - 100 / (1 + gain / loss)
    fig.add_trace(go.Scatter(x=dt, y=rsi, name="RSI",
                             line=dict(color=C["rsi_ln"], width=1.5),
                             showlegend=False), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.07)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.07)", line_width=0, row=3, col=1)
    for lvl in [70, 50, 30]:
        fig.add_hline(y=lvl, line_dash="dot",
                      line_color="rgba(255,255,255,0.15)", line_width=1, row=3, col=1)

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd_l = ema12 - ema26
    sig_l  = macd_l.ewm(span=9, adjust=False).mean()
    hist   = macd_l - sig_l
    hist_colors = [C["macd_h+"] if v >= 0 else C["macd_h-"] for v in hist.fillna(0)]
    fig.add_trace(go.Bar(x=dt, y=hist, name="Hist",
                         marker_color=hist_colors, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=dt, y=macd_l, name="MACD",
                             line=dict(color=C["macd_ln"], width=1.5),
                             showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=dt, y=sig_l, name="Signal",
                             line=dict(color=C["sig_ln"], width=1, dash="dot"),
                             showlegend=False), row=4, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.18)", line_width=1, row=4, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    signal_color = "#00e676" if show_buy_levels else ("#ff1744" if show_sell_levels else "#78909c")
    signal_label = "BUY" if show_buy_levels else ("SELL" if show_sell_levels else "HOLD")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        margin=dict(l=10, r=130, t=40, b=10),
        height=720,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="'Courier New', monospace", size=11),
        title=dict(
            text=(f"<b>{symbol}</b>  ₹{last_price:,.1f}"
                  f"  <span style='color:{signal_color}'>{signal_label}</span>"
                  f"  <span style='font-size:12px'>BUY% {last_buy_p*100:.1f} | Confs {last_conf}/7</span>"),
            x=0.01, font=dict(size=14),
        ),
    )
    fig.update_xaxes(gridcolor=C["grid"], showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], showgrid=True, zeroline=False)
    fig.update_yaxes(title_text="₹ Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume",  row=2, col=1)
    fig.update_yaxes(title_text="RSI",     row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",    row=4, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    selected_symbol = st.selectbox(
        "📌 Stock", NIFTY_50,
        index=NIFTY_50.index("RELIANCE") if "RELIANCE" in NIFTY_50 else 0,
    )
    lookback = st.selectbox(
        "📅 Lookback", [30, 60, 90, 180, 365], index=2,
        format_func=lambda x: f"{x} days",
    )

    st.markdown("#### Chart Overlays")
    col_a, col_b = st.columns(2)
    with col_a:
        show_ema9   = st.checkbox("EMA 9",    value=True)
        show_ema20  = st.checkbox("EMA 20",   value=True)
        show_ema50  = st.checkbox("EMA 50",   value=True)
        show_ema200 = st.checkbox("EMA 200",  value=True)
    with col_b:
        show_bb          = st.checkbox("Bollinger",   value=False)
        show_supertrend  = st.checkbox("Supertrend",  value=False)
        show_pivots      = st.checkbox("Pivots",      value=True)
        show_sr          = st.checkbox("S/R Levels",  value=False)

    st.markdown("---")
    sig_filter = st.multiselect(
        "🎯 Filter signals", ["BUY", "WATCH", "HOLD", "SELL"],
        default=["BUY", "WATCH"],
    )

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("⚠️ Educational only — not financial advice.")


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

model, blob = load_model()
if model is None:
    st.error("No trained model found. Run `python swing_v2.py train` first.")
    st.stop()
model_features = blob.get("features", FEATURE_COLS)

# ─────────────────────────────────────────────────────────────────────────────
# Load market data (shared across tabs)
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading market data…"):
    market = get_market_data()

trend_regime, vol_regime = classify_regime(market)
latest_mkt = market.sort_values("DateTime").iloc[-1].to_dict() if market is not None and not market.empty else {}

nifty_close  = latest_mkt.get("nifty_close",  0)
nifty_50dma  = latest_mkt.get("nifty_50dma",  0)
nifty_200dma = latest_mkt.get("nifty_200dma", 0)
vix_raw      = latest_mkt.get("feat_vix_level", 0.15) * 100
mkt_ret_1d   = latest_mkt.get("feat_market_return_1d", 0) * 100

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

regime_css = {"BULL": "regime-bull", "BEAR": "regime-bear"}.get(trend_regime, "regime-side")
regime_emoji = {"BULL": "🟢 BULL", "BEAR": "🔴 BEAR", "SIDEWAYS": "🟡 SIDEWAYS"}.get(trend_regime, "⚪")
vol_emoji    = {"LOW": "🟢", "NORMAL": "🟢", "HIGH": "🟠", "EXTREME": "🔴"}.get(vol_regime, "⚪")

st.markdown(f"""
<div class='info-card {regime_css}' style='padding:10px 20px;margin-bottom:6px;'>
  <span style='font-size:1.3em;font-weight:800;'>📈 NSE Swing Dashboard</span>
  &emsp;
  <span style='font-size:1.05em'><b>Nifty</b> {nifty_close:,.0f}
  &nbsp;<span style='color:{"#00e676" if mkt_ret_1d>=0 else "#ff1744"}'>{mkt_ret_1d:+.2f}%</span></span>
  &emsp;
  <b>Regime:</b> {regime_emoji}
  &emsp;
  <b>VIX:</b> {vol_emoji} {vix_raw:.1f}
  &emsp;
  <b>EMA50:</b> {nifty_50dma:,.0f} &nbsp;|&nbsp; <b>EMA200:</b> {nifty_200dma:,.0f}
  &emsp;
  <span style='font-size:0.8em;color:#90a4ae'>{date.today().strftime("%d %b %Y")}</span>
</div>
""", unsafe_allow_html=True)

if trend_regime == "BEAR":
    st.warning("⚠️ **BEAR Regime** — New long entries blocked. Showing scores for monitoring only.")
elif vol_regime in ("HIGH", "EXTREME"):
    st.warning(f"⚠️ **VIX {vol_regime}** ({vix_raw:.1f}) — Reduce position sizes.")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💼 Portfolio", "🗺 Sectors", "🔬 Deep Dive"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    with st.spinner("Running signal scan on Nifty 50…"):
        scan_df = run_scan()

    left_col, right_col = st.columns([1.1, 2.7], gap="medium")

    # ── Signal scan table ────────────────────────────────────────────────────
    with left_col:
        st.markdown("#### 🔍 Signal Scan")

        if scan_df.empty:
            st.info("No scan data — model may need retraining.")
        else:
            display_df = scan_df if not sig_filter else scan_df[scan_df["Signal"].isin(sig_filter)]

            sig_counts = scan_df["Signal"].value_counts()
            b1, b2, b3, b4 = st.columns(4)
            for col, sig, emoji in [(b1,"BUY","🟢"),(b2,"WATCH","🟡"),(b3,"HOLD","⚪"),(b4,"SELL","🔴")]:
                col.metric(f"{emoji} {sig}", sig_counts.get(sig, 0))

            show_cols = ["Symbol", "Price", "1D%", "Signal", "Grade", "BUY%", "Confs", "RSI"]

            def _style_row(row):
                styles = []
                for col in row.index:
                    sig = row.get("Signal", "HOLD")
                    if col == "Signal":
                        bg = {"BUY":"#003d1f","WATCH":"#3d3200","SELL":"#3d0000"}.get(sig,"")
                        fg = {"BUY":"#00e676","WATCH":"#ffeb3b","SELL":"#ff1744"}.get(sig,"#78909c")
                        styles.append(f"background:{bg};color:{fg};font-weight:700")
                    elif col == "1D%" and pd.notna(row.get("1D%")):
                        fg = "#00e676" if row["1D%"] >= 0 else "#ef5350"
                        styles.append(f"color:{fg}")
                    elif col == "Grade":
                        fg = {"A":"#ffd700","B":"#c0c0c0","C":"#cd7f32"}.get(row.get("Grade",""),"#78909c")
                        styles.append(f"color:{fg};font-weight:700")
                    else:
                        styles.append("")
                return styles

            styled = (
                display_df[show_cols].style
                .apply(_style_row, axis=1)
                .format({
                    "Price": "₹{:,.1f}",
                    "1D%":   "{:+.2f}%",
                    "BUY%":  "{:.1f}",
                })
            )

            try:
                event = st.dataframe(
                    styled, use_container_width=True, height=460,
                    on_select="rerun", selection_mode="single-row",
                    key="scan_table",
                )
                if event.selection and event.selection.rows:
                    idx = event.selection.rows[0]
                    selected_symbol = display_df.iloc[idx]["Symbol"]
            except Exception:
                st.dataframe(styled, use_container_width=True, height=460)

            st.download_button(
                "⬇ Download CSV",
                scan_df.to_csv(index=False),
                file_name=f"nse_signals_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Chart panel ──────────────────────────────────────────────────────────
    with right_col:
        st.markdown(f"#### 📊 {selected_symbol}")

        with st.spinner(f"Loading {selected_symbol}…"):
            stock_df = get_stock_data(selected_symbol, lookback + 250)

        if stock_df is None or stock_df.empty:
            st.error(f"No data available for {selected_symbol}.")
        else:
            last_row   = stock_df.iloc[-1]
            prev_row   = stock_df.iloc[-2] if len(stock_df) >= 2 else last_row
            last_price = float(last_row["Close"])
            day_chg    = (last_price / float(prev_row["Close"]) - 1) * 100
            week_chg   = (last_price / float(stock_df.iloc[-6]["Close"]) - 1) * 100 if len(stock_df) >= 6 else 0
            month_chg  = (last_price / float(stock_df.iloc[-22]["Close"]) - 1) * 100 if len(stock_df) >= 22 else 0
            high52     = float(stock_df["High"].tail(252).max())
            low52      = float(stock_df["Low"].tail(252).min())
            pct_from_hi = (last_price / high52 - 1) * 100
            atr_val    = float(last_row.get("ATR", 0) or 0)
            rsi_val    = float(last_row.get("feat_rsi", 50) or 50)
            adx_val    = float(last_row.get("feat_adx", 0) or 0) * 100
            vol_ratio  = float(last_row.get("feat_volume_ratio", 1) or 1)

            sc = st.columns(8)
            sc[0].metric("Price",     f"₹{last_price:,.1f}", f"{day_chg:+.2f}%")
            sc[1].metric("1W",        f"{week_chg:+.2f}%")
            sc[2].metric("1M",        f"{month_chg:+.2f}%")
            sc[3].metric("52W High",  f"₹{high52:,.0f}", f"{pct_from_hi:.1f}%")
            sc[4].metric("ATR",       f"₹{atr_val:,.0f}")
            sc[5].metric("RSI-14",    f"{rsi_val:.1f}")
            sc[6].metric("ADX-14",    f"{adx_val:.1f}")
            sc[7].metric("Vol×Avg",   f"{vol_ratio:.2f}×")

            fig = make_chart(
                stock_df, selected_symbol, lookback,
                show_ema9, show_ema20, show_ema50, show_ema200,
                show_bb, show_supertrend, show_pivots, show_sr,
                model, model_features,
            )
            st.plotly_chart(fig, use_container_width=True)

            if not scan_df.empty:
                sr_row = scan_df[scan_df["Symbol"] == selected_symbol]
                if not sr_row.empty:
                    sr = sr_row.iloc[0]
                    sig = sr["Signal"]
                    color = SIGNAL_COLORS.get(sig, "#78909c")
                    grade_html = f" <span class='grade-{sr.get(\"Grade\",\"\")}'>{sr.get('Grade','')}</span>" if sr.get("Grade") else ""
                    stop_s  = f"Stop <b>₹{sr['Stop']:,.1f}</b> &nbsp;" if pd.notna(sr.get("Stop"))   else ""
                    tgt_s   = f"Target <b>₹{sr['Target']:,.1f}</b>" if pd.notna(sr.get("Target")) else ""
                    block_s = f"<br><small style='color:#ffeb3b'>⚠ Blocked: {sr['Blockers']}</small>" if sr.get("Blockers") else ""
                    pass_s  = f"<br><small style='color:#90a4ae'>✅ {sr['Passed']}</small>" if sr.get("Passed") else ""
                    st.markdown(f"""
<div class='info-card' style='border-left:4px solid {color}'>
  <b style='color:{color};font-size:1.15em'>{sig}</b>{grade_html}
  &ensp;BUY% <b>{sr['BUY%']}</b>
  &ensp;Confs <b>{sr['Confs']}</b>
  &ensp;Rank <b>#{sr['Rank']}</b>
  &ensp;{stop_s}{tgt_s}
  {pass_s}{block_s}
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO BUILDER
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 💼 Monthly Portfolio Builder")

    pc1, pc2, pc3 = st.columns([1, 1, 2])
    capital = pc1.number_input("Capital (₹)", min_value=10000, max_value=10000000,
                                value=100000, step=10000, format="%d")
    max_pos = pc2.number_input("Max positions", min_value=1, max_value=10, value=5)
    pc3.markdown(
        "<div style='padding:8px 0;color:#90a4ae;font-size:0.85rem'>"
        "Grades A=25–30%, B=20–25%, C=10–15% of capital. "
        "Cash reserve ≥10%. Max risk per trade 2% of capital."
        "</div>", unsafe_allow_html=True
    )

    if st.button("🏗 Build Portfolio", use_container_width=False, type="primary"):
        with st.spinner("Running signal scan and building portfolio…"):
            if scan_df.empty:
                scan_df = run_scan()

        buys = scan_df[scan_df["Signal"] == "BUY"].copy()

        if buys.empty:
            st.warning("No BUY signals available. Check WATCH list or run again later.")
            watch = scan_df[scan_df["Signal"] == "WATCH"].head(5)
            if not watch.empty:
                st.markdown("**WATCH list (close to BUY):**")
                st.dataframe(watch[["Symbol","Price","BUY%","Confs","Blockers"]],
                             use_container_width=True, hide_index=True)
        else:
            grade_alloc = {"A": 0.275, "B": 0.225, "C": 0.125}
            buys = buys.head(int(max_pos))

            portfolio_rows = []
            total_invested = 0
            total_risk     = 0

            for _, row in buys.iterrows():
                grade   = row.get("Grade", "C") or "C"
                alloc   = capital * grade_alloc.get(grade, 0.125)
                alloc   = min(alloc, capital * 0.30)
                price   = float(row["Price"])
                stop    = float(row["Stop"]) if pd.notna(row.get("Stop")) else price * 0.95
                target  = float(row["Target"]) if pd.notna(row.get("Target")) else price * 1.10
                risk_pp = price - stop
                if risk_pp <= 0:
                    continue

                max_shares_by_risk = int((capital * 0.02) / risk_pp)
                shares_by_alloc    = int(alloc / price)
                shares             = min(max_shares_by_risk, shares_by_alloc)
                if shares <= 0:
                    shares = 1

                invested   = shares * price
                risk_trade = shares * risk_pp
                potential  = shares * (target - price)
                rr         = round(potential / risk_trade, 2) if risk_trade else 0

                total_invested += invested
                total_risk     += risk_trade

                portfolio_rows.append({
                    "#":          len(portfolio_rows) + 1,
                    "Stock":      row["Symbol"],
                    "Sector":     row.get("Sector", ""),
                    "Grade":      grade,
                    "Entry ₹":    f"₹{price:,.1f}",
                    "Shares":     shares,
                    "Invested ₹": f"₹{invested:,.0f}",
                    "Stop ₹":     f"₹{stop:,.1f}",
                    "Target ₹":   f"₹{target:,.1f}",
                    "Max Risk ₹": f"₹{risk_trade:,.0f}",
                    "R:R":        f"1:{rr}",
                    "BUY%":       row["BUY%"],
                })

            if not portfolio_rows:
                st.error("Could not size any positions. Check stop-loss levels.")
            else:
                port_df = pd.DataFrame(portfolio_rows)

                st.markdown(f"""
<div class='info-card' style='border-left:4px solid #00e676'>
  <b>Capital:</b> ₹{capital:,.0f} &ensp;
  <b>Deployed:</b> ₹{total_invested:,.0f} ({total_invested/capital*100:.1f}%) &ensp;
  <b>Cash reserve:</b> ₹{capital-total_invested:,.0f} &ensp;
  <b>Total risk:</b> ₹{total_risk:,.0f} ({total_risk/capital*100:.1f}%) &ensp;
  <b>Expected R:R</b> 1:2 (ATR-based)
</div>""", unsafe_allow_html=True)

                def _grade_style(val):
                    return {"A": "color:#ffd700;font-weight:900",
                            "B": "color:#c0c0c0;font-weight:700",
                            "C": "color:#cd7f32;font-weight:600"}.get(val, "")

                styled_port = port_df.style.applymap(_grade_style, subset=["Grade"])
                st.dataframe(styled_port, use_container_width=True, hide_index=True)

                st.markdown("""
<div class='info-card'>
<b>Exit rules:</b>
&nbsp; ✅ Hit target → close full position &nbsp;
❌ Hit stop → close, accept loss &nbsp;
⏰ 10 trading days elapsed → exit if no trigger
</div>""", unsafe_allow_html=True)

                alloc_vals  = [int(r["Invested ₹"].replace("₹","").replace(",","")) for r in portfolio_rows]
                alloc_names = [r["Stock"] for r in portfolio_rows]
                cash_val    = capital - sum(alloc_vals)
                if cash_val > 0:
                    alloc_names.append("Cash")
                    alloc_vals.append(int(cash_val))

                pie_fig = go.Figure(go.Pie(
                    labels=alloc_names, values=alloc_vals,
                    hole=0.45, textinfo="label+percent",
                    marker=dict(colors=px.colors.qualitative.Set2),
                ))
                pie_fig.update_layout(
                    template="plotly_dark", paper_bgcolor=C["bg"],
                    height=340, margin=dict(l=10, r=10, t=30, b=10),
                    title="Portfolio Allocation", showlegend=False,
                )
                st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("""
<div class='info-card'>
<b>How to use:</b> Set your capital, click Build Portfolio. The system picks top-graded BUY signals,
sizes each position so max loss ≤ 2% of capital, and shows entry/stop/target levels.
Re-run every Monday morning before market open for fresh signals.
<br><br>⚠️ <i>Educational only — not financial advice.</i>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SECTOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🗺 Sector Analysis")

    with st.spinner("Analysing sectors…"):
        if scan_df.empty:
            scan_df = run_scan()

    if scan_df.empty:
        st.info("No scan data available.")
    else:
        sector_df = scan_df.copy()
        sector_stats = (
            sector_df.groupby("Sector")
            .agg(
                Stocks=("Symbol", "count"),
                BUY=("Signal", lambda x: (x == "BUY").sum()),
                WATCH=("Signal", lambda x: (x == "WATCH").sum()),
                SELL=("Signal", lambda x: (x == "SELL").sum()),
                Avg_BUY_pct=("BUY%", "mean"),
                Avg_1D=("1D%", "mean"),
            )
            .reset_index()
        )
        sector_stats["BUY_rate%"] = (sector_stats["BUY"] / sector_stats["Stocks"] * 100).round(1)
        sector_stats = sector_stats.sort_values("BUY_rate%", ascending=False)

        sc_l, sc_r = st.columns([1.3, 1.7])

        with sc_l:
            st.markdown("#### Sector Scorecard")
            def _sect_style(row):
                styles = []
                for col in row.index:
                    if col == "BUY":
                        styles.append("color:#00e676;font-weight:700" if row["BUY"] > 0 else "")
                    elif col == "SELL":
                        styles.append("color:#ff1744;font-weight:700" if row["SELL"] > 0 else "")
                    elif col == "BUY_rate%":
                        pct = row["BUY_rate%"]
                        styles.append(f"color:{'#00e676' if pct>30 else '#ffeb3b' if pct>0 else '#78909c'};font-weight:700")
                    elif col == "Avg_1D":
                        styles.append(f"color:{'#00e676' if row['Avg_1D']>=0 else '#ef5350'}")
                    else:
                        styles.append("")
                return styles

            styled_s = (
                sector_stats.rename(columns={"Avg_BUY_pct": "AvgBUY%", "Avg_1D": "Avg1D%"})
                .style.apply(_sect_style, axis=1)
                .format({"AvgBUY%": "{:.1f}", "Avg1D%": "{:+.2f}%", "BUY_rate%": "{:.1f}%"})
            )
            st.dataframe(styled_s, use_container_width=True, height=420, hide_index=True)

        with sc_r:
            st.markdown("#### BUY Signal Distribution by Sector")
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                y=sector_stats["Sector"], x=sector_stats["BUY"],
                name="BUY", orientation="h", marker_color="#00e676",
                text=sector_stats["BUY"], textposition="outside",
            ))
            bar_fig.add_trace(go.Bar(
                y=sector_stats["Sector"], x=sector_stats["WATCH"],
                name="WATCH", orientation="h", marker_color="#ffeb3b",
                text=sector_stats["WATCH"], textposition="outside",
            ))
            bar_fig.update_layout(
                template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                height=420, barmode="stack",
                margin=dict(l=10, r=60, t=20, b=10),
                legend=dict(orientation="h", y=1.05),
                xaxis_title="Number of stocks",
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("#### Stock-level breakdown by sector")
        selected_sector = st.selectbox(
            "Select sector", ["All"] + sorted(sector_df["Sector"].unique().tolist())
        )
        filtered = sector_df if selected_sector == "All" else sector_df[sector_df["Sector"] == selected_sector]
        show_sect_cols = ["Symbol", "Sector", "Price", "1D%", "Signal", "Grade", "BUY%", "Confs", "RSI", "VolRatio"]
        st.dataframe(
            filtered[show_sect_cols].reset_index(drop=True),
            use_container_width=True, height=300, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — DEEP DIVE
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown(f"### 🔬 Deep Dive — {selected_symbol}")

    dd_symbol = st.selectbox("Stock", NIFTY_50,
                              index=NIFTY_50.index(selected_symbol) if selected_symbol in NIFTY_50 else 0,
                              key="dd_sym")

    with st.spinner(f"Loading {dd_symbol} data…"):
        dd_df = get_stock_data(dd_symbol, 365 + 200)

    if dd_df is None or dd_df.empty:
        st.error(f"No data for {dd_symbol}.")
    else:
        dd_last  = dd_df.iloc[-1]
        dd_price = float(dd_last["Close"])
        dd_atr   = float(dd_last.get("ATR", 0) or 0)

        st.markdown("#### Return Profile")
        rc = st.columns(6)
        for i, (label, days) in enumerate([("1D",1),("1W",5),("1M",22),("3M",66),("6M",132),("1Y",252)]):
            if len(dd_df) > days:
                chg = (dd_price / float(dd_df.iloc[-(days+1)]["Close"]) - 1) * 100
                rc[i].metric(label, f"{chg:+.2f}%")

        st.markdown("#### 7-Gate Confirmation Detail")
        def gv(col, fb=0.0):
            return float(dd_last.get(col, fb) or fb)

        gates = {
            "EMA200 (price > 200-day EMA)": gv("feat_ema200_ratio") > 0,
            "EMA50  (price > 50-day EMA)":  gv("feat_ema50_ratio")  > 0,
            "EMA20  (price > 20-day EMA)":  gv("feat_ema20_ratio")  > 0,
            f"ADX >= {ADX_MIN*100:.0f}":     gv("feat_adx")         >= ADX_MIN,
            "MACD Histogram >= 0":           gv("feat_macd_hist")   >= 0,
            "Price near 20-day high":        gv("feat_dist_20d_high") > 0,
            f"Volume >= {VOLUME_THRESHOLD}x avg": gv("feat_volume_ratio") >= VOLUME_THRESHOLD,
        }
        g_cols = st.columns(7)
        for i, (label, passed) in enumerate(gates.items()):
            icon  = "✅" if passed else "❌"
            color = "#00e676" if passed else "#ef5350"
            g_cols[i].markdown(
                f"<div class='info-card' style='text-align:center;padding:8px'>"
                f"<div style='font-size:1.5rem'>{icon}</div>"
                f"<div style='font-size:0.68rem;color:{color}'>{label}</div>"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("#### Model Feature Values (top 16)")
        feat_vals = {}
        for col in model_features[:16]:
            v = float(dd_last.get(col, 0) or 0)
            feat_vals[col.replace("feat_", "")] = round(v, 4)

        feat_df = pd.DataFrame([feat_vals])
        feat_fig = go.Figure(go.Heatmap(
            z=feat_df.values,
            x=list(feat_vals.keys()),
            y=[""],
            colorscale="RdYlGn",
            showscale=True,
            text=[[f"{v:.3f}" for v in feat_df.values[0]]],
            texttemplate="%{text}",
        ))
        feat_fig.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"],
            height=120, margin=dict(l=10, r=10, t=20, b=40),
        )
        st.plotly_chart(feat_fig, use_container_width=True)

        st.markdown("#### Rolling BUY Probability (last 90 days)")
        dd_trim = dd_df.tail(90 + 50).copy()
        feat_matrix = np.column_stack(
            [dd_trim.get(col, pd.Series(0, index=dd_trim.index)).fillna(0).values
             for col in model_features]
        )
        probs = model.predict_proba(feat_matrix)[:, 1]
        dd_trim["prob"] = probs
        dd_trim = dd_trim.tail(90)
        dd_trim["DateTime"] = pd.to_datetime(dd_trim["DateTime"])

        prob_fig = go.Figure()
        prob_fig.add_trace(go.Scatter(
            x=dd_trim["DateTime"], y=dd_trim["prob"] * 100,
            fill="tozeroy",
            fillcolor="rgba(38,166,154,0.15)",
            line=dict(color="#26a69a", width=1.5),
            name="BUY prob %",
        ))
        prob_fig.add_hline(y=BUY_PROBA * 100, line_dash="dot",
                           line_color="#00e676", line_width=1.5,
                           annotation_text=f"BUY threshold {BUY_PROBA*100:.0f}%",
                           annotation_font_color="#00e676")
        prob_fig.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=220, margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="BUY%", yaxis_range=[0, 100],
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        st.markdown("#### Trade Risk Calculator")
        rk1, rk2, rk3 = st.columns(3)
        trade_capital  = rk1.number_input("Your capital (₹)", value=100000, step=10000, min_value=10000, key="rk_cap")
        risk_pct       = rk2.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.25, key="rk_pct")
        custom_entry   = rk3.number_input("Entry price (₹)", value=float(round(dd_price, 1)), key="rk_entry")

        stop_price   = custom_entry - 1.5 * dd_atr
        target_price = custom_entry + 3.0 * dd_atr
        risk_amount  = trade_capital * risk_pct / 100
        risk_per_share = custom_entry - stop_price
        shares_calc  = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        invest_total = shares_calc * custom_entry
        potential    = shares_calc * (target_price - custom_entry)

        ra, rb, rc2, rd, re = st.columns(5)
        ra.metric("Shares",        f"{shares_calc}")
        rb.metric("Invested",      f"₹{invest_total:,.0f}")
        rc2.metric("Stop",         f"₹{stop_price:,.1f}")
        rd.metric("Target",        f"₹{target_price:,.1f}")
        re.metric("Potential P&L", f"₹{potential:,.0f}", "R:R 1:2")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Model: v{blob.get('version','2')} · trained {blob.get('trained_at','?')} · "
    f"{blob.get('n_train',0):,} samples · {len(model_features)} features · "
    f"val acc {blob.get('val_acc',0)*100:.1f}% · "
    f"Updated every 30 min  ·  ⚠️ Educational only — not financial advice"
)
