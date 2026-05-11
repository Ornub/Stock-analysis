"""
dashboard.py — NSE Swing Trading Dashboard v3  (Streamlit + Plotly)
Institutional-grade light theme with news panel, signal intelligence, and quick filters.
Educational only — not financial advice.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import xml.etree.ElementTree as ET
from datetime import date, timedelta, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
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
    VIX_VERY_HIGH,
)
from stock_fetcher import fetch_historical_data


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Swing Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Base / background */
  .stApp { background-color: #f5f6fa; }
  .block-container { padding-top: 2.5rem; padding-bottom: 0.5rem; max-width: 1400px; }

  /* Sidebar */
  [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3,
  [data-testid="stSidebar"] .stMarkdown h4 { color: #1e293b; }

  /* Typography */
  h1,h2,h3,h4 { color: #1e293b !important; font-weight: 700 !important; }
  .stMarkdown p { color: #334155; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px !important;
  }
  [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; }
  [data-testid="stMetricValue"] { color: #1e293b !important; font-size: 1.05rem !important; font-weight: 700 !important; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

  /* Tables */
  .stDataFrame { font-size: 0.82rem; border-radius: 8px; overflow: hidden; }

  /* Tab labels */
  div[data-baseweb="tab-list"] button { font-size: 0.9rem; font-weight: 600; color: #475569; }
  div[data-baseweb="tab-list"] button[aria-selected="true"] { color: #1d4ed8 !important; }
  div[data-baseweb="tab-highlight"] { background-color: #1d4ed8 !important; }

  /* Signal badges */
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    font-weight: 700; font-size: 0.78rem; letter-spacing: 0.4px;
  }
  .badge-BUY   { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
  .badge-WATCH { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
  .badge-SELL  { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
  .badge-HOLD  { background: #f1f5f9; color: #64748b; border: 1px solid #cbd5e1; }

  /* Cards */
  .card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 20px; margin: 6px 0;
  }
  .card-accent-green  { border-left: 4px solid #16a34a; }
  .card-accent-yellow { border-left: 4px solid #ca8a04; }
  .card-accent-red    { border-left: 4px solid #dc2626; }
  .card-accent-blue   { border-left: 4px solid #1d4ed8; }
  .card-accent-gray   { border-left: 4px solid #94a3b8; }

  /* Regime banner */
  .regime-banner {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 12px 22px; margin-bottom: 10px;
    display: flex; align-items: center; gap: 24px;
  }

  /* Top signal card */
  .top-signal-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 14px 18px;
  }

  /* News item */
  .news-item {
    padding: 9px 0; border-bottom: 1px solid #f1f5f9;
    line-height: 1.45;
  }
  .news-item:last-child { border-bottom: none; }
  .news-source {
    display: inline-block; font-size: 0.7rem; font-weight: 700;
    padding: 1px 7px; border-radius: 99px; margin-right: 6px;
  }
  .src-ET   { background: #dbeafe; color: #1e40af; }
  .src-MC   { background: #fce7f3; color: #9d174d; }
  .src-Reuters { background: #fef3c7; color: #92400e; }
  .src-BS   { background: #ede9fe; color: #5b21b6; }

  /* Confidence bar */
  .conf-bar-wrap { background:#f1f5f9; border-radius:99px; height:6px; margin-top:4px; }
  .conf-bar-fill { border-radius:99px; height:6px; }

  /* Intelligence pills */
  .intel-pill {
    display: inline-block; padding: 3px 10px; border-radius: 99px;
    font-size: 0.72rem; font-weight: 600; margin: 2px 2px;
  }
  .pill-green  { background: #dcfce7; color: #15803d; }
  .pill-red    { background: #fee2e2; color: #b91c1c; }
  .pill-yellow { background: #fef9c3; color: #a16207; }
  .pill-blue   { background: #dbeafe; color: #1e40af; }
  .pill-gray   { background: #f1f5f9; color: #475569; }

  /* Grade colors */
  .grade-A { color: #b45309; font-weight: 800; }
  .grade-B { color: #4b5563; font-weight: 700; }
  .grade-C { color: #6b7280; font-weight: 600; }

  /* Watchlist chips */
  .wl-chip {
    display: inline-block; background: #f8fafc; border: 1px solid #cbd5e1;
    border-radius: 8px; padding: 3px 9px; font-size: 0.78rem;
    color: #334155; font-weight: 600; margin: 2px;
  }
  div[data-testid="stHorizontalBlock"] > div { padding: 0 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (light theme)
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "bull": "#16a34a", "bear": "#dc2626",
    "ema9": "#f97316", "ema20": "#eab308",
    "ema50": "#0ea5e9", "ema200": "#8b5cf6",
    "vol_up": "#16a34a", "vol_dn": "#dc2626",
    "rsi_ln": "#f97316",
    "macd_h+": "#16a34a", "macd_h-": "#dc2626",
    "macd_ln": "#f97316", "sig_ln": "#0ea5e9",
    "stop": "#dc2626", "target": "#16a34a",
    "bb": "#0ea5e9",
    "supertrend_bull": "#16a34a", "supertrend_bear": "#dc2626",
    "buy_mk": "#16a34a", "sell_mk": "#dc2626",
    "bg": "#ffffff", "grid": "#f1f5f9",
    "text": "#1e293b",
}

SIGNAL_COLORS  = {"BUY": "#16a34a", "WATCH": "#ca8a04", "HOLD": "#64748b", "SELL": "#dc2626"}
SIGNAL_BG      = {"BUY": "#dcfce7", "WATCH": "#fef9c3", "HOLD": "#f1f5f9", "SELL": "#fee2e2"}

NEWS_FEEDS = [
    {
        "name": "ET Markets",
        "cls":  "ET",
        "url":  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    },
    {
        "name": "Moneycontrol",
        "cls":  "MC",
        "url":  "https://www.moneycontrol.com/rss/marketreports.xml",
    },
    {
        "name": "Business Standard",
        "cls":  "BS",
        "url":  "https://www.business-standard.com/rss/markets-106.rss",
    },
    {
        "name": "Reuters India",
        "cls":  "Reuters",
        "url":  "https://feeds.reuters.com/reuters/INbusinessNews",
    },
]


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
    try:
        return _fetch_market_regime(5)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def run_scan() -> pd.DataFrame:
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

    _mkt_last = market.sort_values("DateTime").iloc[-1] if market is not None and not market.empty else {}
    latest_mkt_vix = float(_mkt_last.get("feat_vix_level", 0.15) if hasattr(_mkt_last, "get") else 0.15) * 100

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

        ema200_ok    = confirmations["EMA200"]
        gate_pass    = ema200_ok and sym in ranked_set and not regime_blocked and n_pass >= MIN_CONFIRMATIONS

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
            "RSI_warn":  signal == "BUY" and rsi > 70,
            "ATR_warn":  signal == "BUY" and (atr / price > 0.025 if price > 0 else False),
            "WeakBUY":   signal == "BUY" and grade == "C" and rsi > 65 and vol_r < 1.3,
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    # VIX extreme override — engine spec: halt new longs when VIX > VIX_VERY_HIGH
    vix_now = latest_mkt_vix  # captured before loop
    if vix_now > VIX_VERY_HIGH:
        mask_buy   = df_out["Signal"] == "BUY"
        mask_watch = df_out["Signal"] == "WATCH"
        df_out.loc[mask_buy,   "Signal"]   = "WATCH"
        df_out.loc[mask_buy,   "Blockers"] = df_out.loc[mask_buy, "Blockers"].apply(
            lambda b: f"VIX Extreme ({vix_now:.1f})" + (f", {b}" if b else ""))
        df_out.loc[mask_watch, "Signal"]   = "HOLD"
        df_out.loc[mask_watch, "Blockers"] = df_out.loc[mask_watch, "Blockers"].apply(
            lambda b: f"VIX Extreme ({vix_now:.1f})" + (f", {b}" if b else ""))

    df_out["ScanTime"] = datetime.now().strftime("%H:%M")
    sig_ord = {"BUY": 0, "WATCH": 1, "HOLD": 2, "SELL": 3}
    df_out["_s"] = df_out["Signal"].map(sig_ord).fillna(4)
    return df_out.sort_values(["_s", "BUY%"], ascending=[True, False]).drop(columns=["_s"]).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(max_per_feed: int = 3) -> list[dict]:
    """Fetch headlines from RSS feeds, sorted newest-first, max 7 days old."""
    import html
    articles = []
    cutoff = datetime.now() - timedelta(days=7)

    _pub_fmts = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S",
        "%a, %d %b %Y %H:%M",
    ]

    for feed in NEWS_FEEDS:
        try:
            resp = requests.get(feed["url"], timeout=6,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            count = 0
            for item in items:
                if count >= max_per_feed:
                    break
                title = html.unescape((item.findtext("title") or "").strip())
                link  = (item.findtext("link") or "").strip()
                pub   = (item.findtext("pubDate") or "").strip()
                if not title or not link:
                    continue

                # parse publish date
                pub_dt = None
                for fmt in _pub_fmts:
                    try:
                        pub_dt = datetime.strptime(pub[:31].strip(), fmt)
                        pub_dt = pub_dt.replace(tzinfo=None)
                        break
                    except Exception:
                        continue

                # skip articles older than 7 days
                if pub_dt and pub_dt < cutoff:
                    continue

                pub_fmt = pub_dt.strftime("%d %b %H:%M") if pub_dt else "recent"
                articles.append({
                    "title":  title[:120],
                    "link":   link,
                    "source": feed["name"],
                    "cls":    feed["cls"],
                    "pub":    pub_fmt,
                    "pub_dt": pub_dt or datetime.min,
                })
                count += 1
        except Exception:
            continue

    # sort all articles newest-first
    articles.sort(key=lambda a: a["pub_dt"], reverse=True)
    # remove internal sort key before returning
    for a in articles:
        a.pop("pub_dt", None)
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# Intelligence helpers
# ─────────────────────────────────────────────────────────────────────────────

def explain_signal(signal: str, passed: str, blockers: str, buy_pct: float, grade: str) -> str:
    parts = passed.split(", ") if passed else []
    blk   = blockers.split(", ") if blockers else []

    trend_ok  = "EMA200" in parts and "EMA50" in parts
    momentum  = "ADX" in parts and "MACD" in parts
    breakout  = "Breakout" in parts
    vol_ok    = "Volume" in parts

    all_gates = ["EMA200", "EMA50", "EMA20", "ADX", "MACD", "Breakout", "Volume"]
    failed_gates = [g for g in all_gates if g not in parts]

    if signal == "BUY":
        base = "Uptrend confirmed"
        if trend_ok:
            base += " — price above EMA200 & EMA50"
        if momentum:
            base += ", strong momentum (ADX + MACD)"
        if breakout:
            base += ", near 20-day high"
        if vol_ok:
            base += " with volume expansion"
        base += f". Confidence {buy_pct:.0f}%, Grade {grade}."
        if grade != "A" and failed_gates:
            base += f" Failed gates: {', '.join(failed_gates)}."
        return base
    elif signal == "WATCH":
        base = f"Model score {buy_pct:.0f}% is bullish"
        if blk:
            base += f", but {len(blk)} gate(s) failed: {', '.join(blk)}"
        base += ". Observe only — do not prepare entry until conditions improve."
        return base
    elif signal == "SELL":
        return "Bearish model score. Price structure weakening — avoid new longs."
    else:
        return "No clear directional edge. Holding pattern — monitor for breakout."


def classify_stock_state(latest) -> dict:
    def gv(col, fb=0.0):
        return float(latest.get(col, fb) or fb)

    ema200_ratio = gv("feat_ema200_ratio")
    ema50_ratio  = gv("feat_ema50_ratio")
    ema20_ratio  = gv("feat_ema20_ratio")
    adx          = gv("feat_adx") * 100
    macd_hist    = gv("feat_macd_hist")
    dist_20d     = gv("feat_dist_20d_high")
    vol_ratio    = gv("feat_volume_ratio")
    rsi          = gv("feat_rsi")
    atr_pct      = gv("feat_atr_pct", 0.015)

    # Trend
    if ema200_ratio > 0 and ema50_ratio > 0 and ema20_ratio > 0:
        trend = ("Uptrend", "pill-green")
    elif ema200_ratio < 0 and ema50_ratio < 0:
        trend = ("Downtrend", "pill-red")
    elif ema200_ratio > 0:
        trend = ("Weak uptrend", "pill-yellow")
    else:
        trend = ("Consolidation", "pill-gray")

    # Momentum
    if adx >= 25 and macd_hist > 0:
        momentum = ("Rising", "pill-green")
    elif adx >= 20 and macd_hist >= 0:
        momentum = ("Neutral+", "pill-yellow")
    elif macd_hist < 0 and adx >= 20:
        momentum = ("Fading", "pill-yellow")
    else:
        momentum = ("Weak", "pill-gray")

    # Volatility
    if atr_pct > 0.025:
        volatility = ("High", "pill-red")
    elif atr_pct > 0.015:
        volatility = ("Normal", "pill-yellow")
    else:
        volatility = ("Low", "pill-green")

    # State — breakout only valid with genuine volume expansion (>=1.5x avg)
    if dist_20d > 0 and vol_ratio >= 1.5:
        state = ("Breakout", "pill-green")
    elif dist_20d > 0 and vol_ratio >= VOLUME_THRESHOLD:
        state = ("Thin breakout", "pill-yellow")
    elif rsi > 65 and dist_20d > 0:
        state = ("Overbought", "pill-yellow")
    elif rsi < 35:
        state = ("Oversold", "pill-blue")
    elif abs(dist_20d) < 0.005 and adx < 20:
        state = ("Consolidating", "pill-gray")
    else:
        state = ("Pullback", "pill-yellow")

    return {
        "trend": trend,
        "momentum": momentum,
        "volatility": volatility,
        "state": state,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def calc_bollinger(close: pd.Series, window=20, num_std=2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid - num_std * std, mid, mid + num_std * std


def calc_supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
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
    p = (high + low + close) / 3
    return dict(P=p, R1=2*p-low, R2=p+(high-low), R3=high+2*(p-low),
                S1=2*p-high, S2=p-(high-low), S3=low-2*(high-p))


def find_support_resistance(df: pd.DataFrame, n_levels=3):
    highs = df["High"].rolling(5, center=True).max()
    lows  = df["Low"].rolling(5, center=True).min()
    res = sorted(highs.dropna().nlargest(n_levels).tolist(), reverse=True)
    sup = sorted(lows.dropna().nsmallest(n_levels).tolist())
    return res, sup


# ─────────────────────────────────────────────────────────────────────────────
# Chart builder
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
        row_heights=[0.58, 0.14, 0.14, 0.14],
        vertical_spacing=0.018,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dt, open=df["Open"], high=h, low=lo, close=c,
        increasing_line_color=C["bull"], decreasing_line_color=C["bear"],
        increasing_fillcolor="#bbf7d0",  decreasing_fillcolor="#fecaca",
        name="OHLC", showlegend=False, whiskerwidth=0.4,
        line=dict(width=1),
    ), row=1, col=1)

    # Bollinger Bands
    if show_bb:
        bb_lo, bb_mid, bb_hi = calc_bollinger(c)
        fig.add_trace(go.Scatter(x=dt, y=bb_hi,  name="BB Upper",
                                 line=dict(color=C["bb"], width=1, dash="dot"),
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=dt, y=bb_mid, name="BB Mid",
                                 line=dict(color=C["bb"], width=1),
                                 showlegend=False, opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=dt, y=bb_lo, name="BB",
                                 line=dict(color=C["bb"], width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(14,165,233,0.06)"), row=1, col=1)

    # EMAs
    ema_defs = [
        (9,   C["ema9"],   1.0, show_ema9,   "EMA9"),
        (20,  C["ema20"],  1.0, show_ema20,  "EMA20"),
        (50,  C["ema50"],  1.5, show_ema50,  "EMA50"),
        (200, C["ema200"], 2.0, show_ema200, "EMA200"),
    ]
    for span, color, width, show, label in ema_defs:
        if show:
            ema_s = c.ewm(span=span, adjust=False).mean()
            fig.add_trace(go.Scatter(x=dt, y=ema_s, name=label,
                                     line=dict(color=color, width=width)), row=1, col=1)

    # Supertrend
    if show_supertrend and len(df) > 20:
        st_trend, st_line = calc_supertrend(df)
        bull_idx = st_trend == 1
        bear_idx = st_trend == -1
        if bull_idx.any():
            fig.add_trace(go.Scatter(x=dt[bull_idx], y=st_line[bull_idx], name="ST↑",
                                     mode="lines", line=dict(color=C["supertrend_bull"], width=2)), row=1, col=1)
        if bear_idx.any():
            fig.add_trace(go.Scatter(x=dt[bear_idx], y=st_line[bear_idx], name="ST↓",
                                     mode="lines", line=dict(color=C["supertrend_bear"], width=2)), row=1, col=1)

    # S/R
    if show_sr and len(df) > 30:
        res_levels, sup_levels = find_support_resistance(df.tail(60))
        for lvl in res_levels[:2]:
            fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(220,38,38,0.45)", line_width=1,
                          annotation_text=f"R {lvl:,.0f}", annotation_font_color="rgba(220,38,38,0.8)",
                          annotation_position="right", row=1, col=1)
        for lvl in sup_levels[:2]:
            fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(22,163,74,0.45)", line_width=1,
                          annotation_text=f"S {lvl:,.0f}", annotation_font_color="rgba(22,163,74,0.8)",
                          annotation_position="right", row=1, col=1)

    # Buy/Sell markers
    buy_df = df[buy_mask]
    if not buy_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_df["DateTime"], y=buy_df["Low"] * 0.991,
            mode="markers", name="BUY ▲",
            marker=dict(symbol="triangle-up", size=12,
                        color=C["buy_mk"], line=dict(color="white", width=1)),
            hovertemplate="<b>BUY</b> %{x|%d %b}<br>Prob: %{customdata:.1%}<extra></extra>",
            customdata=buy_df["buy_prob"].values,
        ), row=1, col=1)
    sell_df = df[sell_mask]
    if not sell_df.empty:
        fig.add_trace(go.Scatter(
            x=sell_df["DateTime"], y=sell_df["High"] * 1.009,
            mode="markers", name="SELL ▼",
            marker=dict(symbol="triangle-down", size=12,
                        color=C["sell_mk"], line=dict(color="white", width=1)),
            hovertemplate="<b>SELL</b> %{x|%d %b}<extra></extra>",
        ), row=1, col=1)

    # Stop/Target bands
    if show_buy_levels and last_atr > 0:
        stop_lvl   = last_price - ATR_STOP_MULT  * last_atr
        target_lvl = last_price + ATR_TARGET_MULT * last_atr
        trail_lvl  = last_price - ATR_TRAIL_MULT  * last_atr
        fig.add_hrect(y0=stop_lvl * 0.998, y1=stop_lvl * 1.002,
                      fillcolor="rgba(220,38,38,0.12)", line_width=0,
                      annotation_text=f"STOP ₹{stop_lvl:,.0f}",
                      annotation_font_color=C["stop"], annotation_position="right", row=1, col=1)
        fig.add_hrect(y0=target_lvl * 0.998, y1=target_lvl * 1.002,
                      fillcolor="rgba(22,163,74,0.12)", line_width=0,
                      annotation_text=f"TARGET ₹{target_lvl:,.0f}",
                      annotation_font_color=C["target"], annotation_position="right", row=1, col=1)
        fig.add_hline(y=trail_lvl, line_dash="dot", line_color="rgba(220,38,38,0.45)", line_width=1.2,
                      annotation_text=f"Trail ₹{trail_lvl:,.0f}",
                      annotation_font_color="rgba(220,38,38,0.65)", row=1, col=1)

    # Pivot levels
    if show_pivots and len(df) >= 2:
        ph = float(h.iloc[-2]); pl = float(lo.iloc[-2]); pc = float(c.iloc[-2])
        pv = pivot_levels(ph, pl, pc)
        for lbl, lvl, col, dash in [
            ("R2", pv["R2"], "rgba(220,38,38,0.40)", "dot"),
            ("R1", pv["R1"], "rgba(220,38,38,0.70)", "dash"),
            ("P",  pv["P"],  "rgba(100,116,139,0.75)", "solid"),
            ("S1", pv["S1"], "rgba(22,163,74,0.70)", "dash"),
            ("S2", pv["S2"], "rgba(22,163,74,0.40)", "dot"),
        ]:
            fig.add_hline(y=lvl, line_dash=dash, line_color=col, line_width=1,
                          annotation_text=f"{lbl} {lvl:,.0f}",
                          annotation_font_size=9, annotation_font_color=col,
                          annotation_position="right", row=1, col=1)

    # Volume
    vol_colors = [C["bull"] if cl >= op else C["bear"]
                  for cl, op in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=dt, y=vol, name="Volume",
                         marker_color=vol_colors, opacity=0.55,
                         showlegend=False, marker_line_width=0), row=2, col=1)
    fig.add_trace(go.Scatter(x=dt, y=vol.rolling(20).mean(), name="Vol MA20",
                             line=dict(color="#0ea5e9", width=1.2), showlegend=False), row=2, col=1)

    # RSI
    delta = c.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).where(delta < 0, 0.0).ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    rsi   = 100 - 100 / (1 + gain / loss)
    fig.add_trace(go.Scatter(x=dt, y=rsi, name="RSI",
                             line=dict(color=C["rsi_ln"], width=1.5), showlegend=False), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.06)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(22,163,74,0.06)", line_width=0, row=3, col=1)
    for lvl in [70, 50, 30]:
        fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(100,116,139,0.3)", line_width=1, row=3, col=1)

    # MACD
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd_l = ema12 - ema26
    sig_l  = macd_l.ewm(span=9, adjust=False).mean()
    hist   = macd_l - sig_l
    hist_colors = [C["macd_h+"] if v >= 0 else C["macd_h-"] for v in hist.fillna(0)]
    fig.add_trace(go.Bar(x=dt, y=hist, name="Hist", marker_color=hist_colors,
                         opacity=0.6, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=dt, y=macd_l, name="MACD",
                             line=dict(color=C["macd_ln"], width=1.5), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=dt, y=sig_l, name="Signal",
                             line=dict(color=C["sig_ln"], width=1, dash="dot"), showlegend=False), row=4, col=1)
    fig.add_hline(y=0, line_color="rgba(100,116,139,0.25)", line_width=1, row=4, col=1)

    # Current-value labels on right edge for RSI and MACD
    cur_rsi  = float(rsi.iloc[-1])  if not rsi.isna().all()   else 50.0
    cur_macd = float(macd_l.iloc[-1]) if not macd_l.isna().all() else 0.0
    rsi_col  = C["bear"] if cur_rsi >= 70 else (C["bull"] if cur_rsi <= 30 else C["rsi_ln"])
    macd_col = C["macd_h+"] if cur_macd >= 0 else C["macd_h-"]

    for ann_row, ann_y, ann_text, ann_col, ann_ref in [
        (3, cur_rsi,  f"<b>{cur_rsi:.1f}</b>",  rsi_col,  "y3"),
        (4, cur_macd, f"<b>{cur_macd:.2f}</b>", macd_col, "y4"),
    ]:
        fig.add_annotation(
            x=1, y=ann_y, xref="paper", yref=ann_ref,
            text=ann_text, showarrow=False,
            font=dict(size=9, color=ann_col),
            xanchor="left", align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=ann_col, borderwidth=1, borderpad=2,
        )

    # Small panel labels (top-left of each sub-panel)
    for ann_row, ann_ref, ann_text in [
        (2, "y2", "VOL"),
        (3, "y3", "RSI 14"),
        (4, "y4", "MACD"),
    ]:
        fig.add_annotation(
            x=0, y=1, xref="paper", yref=f"{ann_ref} domain",
            text=f"<span style='font-size:8px'>{ann_text}</span>",
            showarrow=False,
            font=dict(size=8, color="#94a3b8"),
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0)",
        )

    # Layout
    signal_color = C["bull"] if show_buy_levels else (C["bear"] if show_sell_levels else "#64748b")
    signal_label = "BUY" if show_buy_levels else ("SELL" if show_sell_levels else "HOLD")
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
        margin=dict(l=10, r=100, t=44, b=10),
        height=720,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#e2e8f0", borderwidth=1),
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color="#334155"),
        title=dict(
            text=(f"<b style='color:#1e293b'>{symbol}</b>"
                  f"  <span style='color:#334155'>₹{last_price:,.1f}</span>"
                  f"  <span style='color:{signal_color};font-weight:700'> {signal_label}</span>"
                  f"  <span style='color:#94a3b8;font-size:10px'>  BUY% {last_buy_p*100:.0f} · {last_conf}/7 gates</span>"),
            x=0.01, font=dict(size=14),
        ),
        hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                        font=dict(size=11, color="#1e293b")),
    )
    fig.update_xaxes(
        gridcolor=C["grid"], showgrid=True, zeroline=False,
        linecolor="#e2e8f0", tickfont=dict(color="#94a3b8", size=10),
        showticklabels=False,  # hide on all rows except bottom
    )
    fig.update_xaxes(showticklabels=True, tickfont=dict(color="#94a3b8", size=10), row=4, col=1)
    fig.update_yaxes(
        gridcolor=C["grid"], showgrid=True, zeroline=False,
        linecolor="#e2e8f0", tickfont=dict(color="#94a3b8", size=10),
        ticklen=3,
    )
    fig.update_yaxes(title_text="", row=1, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="", row=2, col=1, tickformat=".2s")
    fig.update_yaxes(title_text="", row=3, col=1, range=[0, 100], dtick=20)
    fig.update_yaxes(title_text="", row=4, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["RELIANCE", "INFY", "TCS", "HDFCBANK"]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## NSE Swing Intelligence")
    st.markdown("---")

    # Stock selector
    selected_symbol = st.selectbox(
        "Stock", NIFTY_50,
        index=NIFTY_50.index("RELIANCE") if "RELIANCE" in NIFTY_50 else 0,
        label_visibility="collapsed",
    )
    lookback = st.selectbox("Lookback", [30, 60, 90, 180, 365], index=2,
                             format_func=lambda x: f"{x} days")

    st.markdown("---")
    st.markdown("#### Chart Overlays")
    col_a, col_b = st.columns(2)
    with col_a:
        show_ema9   = st.checkbox("EMA 9",   value=True)
        show_ema20  = st.checkbox("EMA 20",  value=True)
        show_ema50  = st.checkbox("EMA 50",  value=True)
        show_ema200 = st.checkbox("EMA 200", value=True)
    with col_b:
        show_bb         = st.checkbox("Bollinger", value=False)
        show_supertrend = st.checkbox("Supertrend",value=False)
        show_pivots     = st.checkbox("Pivots",    value=True)
        show_sr         = st.checkbox("S/R",       value=False)

    st.markdown("---")
    st.markdown("#### Quick Filters")
    sig_filter   = st.multiselect("Signal", ["BUY","WATCH","HOLD","SELL"], default=["BUY","WATCH"])
    all_sectors  = sorted(set(SECTOR_MAP.values()))
    sect_filter  = st.multiselect("Sector", all_sectors, default=[])
    min_conf     = st.slider("Min BUY%", 0, 100, 0, 5)
    rsi_range    = st.slider("RSI range", 0, 100, (0, 100), 5)

    st.markdown("---")
    st.markdown("#### Watchlist")
    wl_items = "  ".join([f"<span class='wl-chip'>{s}</span>" for s in st.session_state.watchlist])
    st.markdown(wl_items, unsafe_allow_html=True)
    wl_add = st.selectbox("Add stock", ["—"] + [s for s in NIFTY_50 if s not in st.session_state.watchlist],
                           label_visibility="collapsed")
    wc1, wc2 = st.columns(2)
    if wc1.button("+ Add", use_container_width=True) and wl_add != "—":
        st.session_state.watchlist.append(wl_add)
        st.rerun()
    wl_rem = st.selectbox("Remove", ["—"] + st.session_state.watchlist, label_visibility="collapsed")
    if wc2.button("− Remove", use_container_width=True) and wl_rem != "—":
        st.session_state.watchlist.remove(wl_rem)
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("⚠️ Educational only — not financial advice.")


# ─────────────────────────────────────────────────────────────────────────────
# Load model & market data
# ─────────────────────────────────────────────────────────────────────────────

model, blob = load_model()
if model is None:
    st.error("No trained model found. Run `python swing_v2.py train` first.")
    st.stop()
model_features = blob.get("features", FEATURE_COLS)

with st.spinner("Loading market data…"):
    market = get_market_data()

trend_regime, vol_regime = classify_regime(market)
latest_mkt = market.sort_values("DateTime").iloc[-1].to_dict() if market is not None and not market.empty else {}

nifty_close  = latest_mkt.get("nifty_close",  0)
nifty_50dma  = latest_mkt.get("nifty_50dma",  0)
nifty_200dma = latest_mkt.get("nifty_200dma", 0)
vix_raw      = latest_mkt.get("feat_vix_level", 0.15) * 100
mkt_ret_1d   = latest_mkt.get("feat_market_return_1d", 0) * 100
mkt_data_ok  = nifty_close > 0

# Display strings — show N/A when market fetch returned zeros
_nifty_str  = f"{nifty_close:,.0f}" if mkt_data_ok else "N/A"
_dma50_str  = f"{nifty_50dma:,.0f}" if mkt_data_ok else "N/A"
_dma200_str = f"{nifty_200dma:,.0f}" if mkt_data_ok else "N/A"
_chg_str    = f"{mkt_ret_1d:+.2f}%" if mkt_data_ok else ""
_vix_str    = f"{vix_raw:.1f}" if mkt_data_ok else "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Regime banner
# ─────────────────────────────────────────────────────────────────────────────

regime_icon  = {"BULL": "🟢", "BEAR": "🔴"}.get(trend_regime, "🟡")
regime_label = {"BULL": "Bull Market", "BEAR": "Bear Market", "SIDEWAYS": "Sideways"}.get(trend_regime, trend_regime)
vix_icon     = "🔴" if vol_regime in ("HIGH","EXTREME") else ("🟡" if vol_regime == "NORMAL" else "🟢")
chg_color    = "#16a34a" if mkt_ret_1d >= 0 else "#dc2626"

_regime_asterisk = "" if mkt_data_ok else "<span style='color:#ca8a04;font-size:0.7rem'> (est.)</span>"
st.markdown(f"""
<div class='regime-banner'>
  <div style='min-width:180px'>
    <div style='font-size:1.2rem;font-weight:800;color:#1e293b;white-space:nowrap'>NSE Swing Intelligence</div>
    <div style='font-size:0.72rem;color:#64748b'>{date.today().strftime("%A, %d %B %Y")}</div>
  </div>
  <div style='display:flex;gap:20px;flex-wrap:wrap;align-items:center;justify-content:flex-end;flex:1'>
    <div style='text-align:center;min-width:70px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>NIFTY 50</div>
      <div style='font-size:1rem;font-weight:700;color:#1e293b;white-space:nowrap'>{_nifty_str} <span style='color:{chg_color};font-size:0.82rem'>{_chg_str}</span></div>
    </div>
    <div style='text-align:center;min-width:90px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>REGIME</div>
      <div style='font-size:0.88rem;font-weight:700;color:#1e293b;white-space:nowrap'>{regime_icon} {regime_label}{_regime_asterisk}</div>
    </div>
    <div style='text-align:center;min-width:55px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>VIX</div>
      <div style='font-size:0.88rem;font-weight:700;color:#1e293b'>{vix_icon} {_vix_str}</div>
    </div>
    <div style='text-align:center;min-width:60px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>EMA50</div>
      <div style='font-size:0.88rem;font-weight:600;color:#1e293b'>{_dma50_str}</div>
    </div>
    <div style='text-align:center;min-width:60px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>EMA200</div>
      <div style='font-size:0.88rem;font-weight:600;color:#1e293b'>{_dma200_str}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not mkt_data_ok:
    st.warning("⚠️ Market data unavailable (^NSEI fetch failed) — regime shown is fallback SIDEWAYS. Refresh to retry.")
elif trend_regime == "BEAR":
    st.warning("⚠️ Bear regime active — new long entries blocked. Scores shown for monitoring only.")
elif vol_regime in ("HIGH", "EXTREME"):
    st.warning(f"⚠️ VIX elevated ({vix_raw:.1f}) — reduce position sizes.")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Portfolio", "Sectors", "Deep Dive"])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    with st.spinner("Running signal scan…"):
        scan_df = run_scan()

    # ── Apply quick filters ──────────────────────────────────────────
    filtered_df = scan_df.copy() if not scan_df.empty else pd.DataFrame()
    if not filtered_df.empty:
        if sig_filter:
            filtered_df = filtered_df[filtered_df["Signal"].isin(sig_filter)]
        if sect_filter:
            filtered_df = filtered_df[filtered_df["Sector"].isin(sect_filter)]
        filtered_df = filtered_df[filtered_df["BUY%"] >= min_conf]
        filtered_df = filtered_df[
            (filtered_df["RSI"] >= rsi_range[0]) & (filtered_df["RSI"] <= rsi_range[1])
        ]

    # ── Scan age indicator ───────────────────────────────────────────
    if not scan_df.empty and "ScanTime" in scan_df.columns:
        try:
            scan_time_str = scan_df["ScanTime"].iloc[0]
            scan_dt = datetime.now().replace(
                hour=int(scan_time_str.split(":")[0]),
                minute=int(scan_time_str.split(":")[1]),
                second=0, microsecond=0,
            )
            scan_age_min = max(0, int((datetime.now() - scan_dt).total_seconds() // 60))
            age_color = "#dc2626" if scan_age_min > 10 else ("#ca8a04" if scan_age_min > 5 else "#64748b")
            st.markdown(
                f"<div style='text-align:right;font-size:0.72rem;color:{age_color};margin-bottom:4px'>"
                f"Scan: {scan_time_str}"
                f"{'  ⚠ ' + str(scan_age_min) + ' min old — prices may have moved' if scan_age_min > 10 else ''}"
                f"</div>", unsafe_allow_html=True
            )
        except Exception:
            pass

    # ── Top 3 Actionable Signals ─────────────────────────────────────
    if not scan_df.empty:
        _buys = scan_df[scan_df["Signal"] == "BUY"].copy()
        # quality filter: exclude overbought entries and thin-volume breakouts
        _buys_filtered = _buys[(_buys["RSI"] <= 72) & (_buys["VolRatio"] >= 1.3)]
        top3 = _buys_filtered.head(3) if not _buys_filtered.empty else _buys.head(3)
        _top3_is_watch = top3.empty
        if _top3_is_watch:
            top3 = scan_df[scan_df["Signal"] == "WATCH"].head(3)

        if not top3.empty:
            if _top3_is_watch:
                st.markdown("#### Best Watch Candidates")
                st.caption("No qualifying BUY signals today — showing highest-conviction WATCH setups.")
            else:
                st.markdown("#### Top BUY Signals Today")
            t_cols = st.columns(len(top3))
            for col, (_, row) in zip(t_cols, top3.iterrows()):
                sig   = row["Signal"]
                grade = row.get("Grade", "")
                color = SIGNAL_COLORS[sig]
                bg    = SIGNAL_BG[sig]
                conf_w = int(row["BUY%"])
                expl  = explain_signal(sig, row.get("Passed",""), row.get("Blockers",""),
                                        row["BUY%"], grade)
                stop_s   = f"<br><span style='color:#64748b;font-size:0.75rem'>Stop ₹{row['Stop']:,.1f}</span>" if pd.notna(row.get("Stop")) else ""
                tgt_s    = f" · Target ₹{row['Target']:,.1f}" if pd.notna(row.get("Target")) else ""
                warn_tags = ""
                if row.get("RSI_warn"):
                    warn_tags += "<span style='font-size:0.68rem;background:#fef9c3;color:#a16207;border-radius:4px;padding:1px 6px;margin-right:4px'>⚠ RSI>70</span>"
                if row.get("ATR_warn"):
                    warn_tags += "<span style='font-size:0.68rem;background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px'>⚠ Wide ATR</span>"
                col.markdown(f"""
<div class='top-signal-card' style='border-top: 3px solid {color}'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <span style='font-size:1.05rem;font-weight:800;color:#1e293b'>{row['Symbol']}</span>
    <span class='badge badge-{sig}'>{sig}{' ' + grade if grade else ''}</span>
  </div>
  <div style='color:#334155;font-size:1.0rem;font-weight:700;margin:4px 0'>₹{row['Price']:,.1f}
    <span style='font-size:0.82rem;color:{"#16a34a" if row["1D%"]>=0 else "#dc2626"}'>{row["1D%"]:+.2f}%</span>
  </div>
  <div style='font-size:0.72rem;color:#64748b'>RSI {row['RSI']:.0f} · ADX {row['ADX']:.0f} · Vol×{row['VolRatio']:.1f}</div>
  <div style='margin:6px 0'>
    <div style='font-size:0.68rem;color:#64748b;margin-bottom:2px'>Confidence {conf_w}%</div>
    <div class='conf-bar-wrap'>
      <div class='conf-bar-fill' style='width:{conf_w}%;background:{color}'></div>
    </div>
  </div>
  {stop_s}{tgt_s}
  {warn_tags}
  <div style='font-size:0.72rem;color:#64748b;margin-top:6px;line-height:1.4'>{expl}</div>
</div>""", unsafe_allow_html=True)

    # ── Signal count summary row (full width) ────────────────────────
    if not scan_df.empty:
        sig_counts = scan_df["Signal"].value_counts()
        _sig_colors = {"BUY":"#16a34a","WATCH":"#ca8a04","HOLD":"#64748b","SELL":"#dc2626"}
        _sig_bg     = {"BUY":"#dcfce7","WATCH":"#fef9c3","HOLD":"#f1f5f9","SELL":"#fee2e2"}
        _cnt_cols = st.columns([1, 1, 1, 1, 6])
        for col_m, sig in zip(_cnt_cols[:4], ["BUY","WATCH","HOLD","SELL"]):
            n = sig_counts.get(sig, 0)
            col_m.markdown(
                f"<div style='background:{_sig_bg[sig]};border:1px solid {_sig_colors[sig]}33;"
                f"border-radius:10px;padding:10px 6px;text-align:center'>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{_sig_colors[sig]}'>{n}</div>"
                f"<div style='font-size:0.72rem;font-weight:600;color:{_sig_colors[sig]}'>{sig}</div>"
                f"</div>", unsafe_allow_html=True
            )

    st.markdown("")

    # ── Watchlist chips (full width) ──────────────────────────────────
    if st.session_state.watchlist:
        wl_display = st.columns(len(st.session_state.watchlist))
        for wl_c, sym in zip(wl_display, st.session_state.watchlist):
            if not scan_df.empty:
                row_m = scan_df[scan_df["Symbol"] == sym]
                sig_m   = row_m.iloc[0]["Signal"] if not row_m.empty else "HOLD"
                price_m = row_m.iloc[0]["Price"]  if not row_m.empty else 0
                chg_m   = row_m.iloc[0]["1D%"]    if not row_m.empty else 0
                col_m   = SIGNAL_COLORS.get(sig_m,"#64748b")
                wl_c.markdown(f"""
<div style='background:#fff;border:1px solid #e2e8f0;border-top:3px solid {col_m};
     border-radius:8px;padding:7px 10px;text-align:center;cursor:pointer'>
  <div style='font-weight:700;font-size:0.82rem;color:#1e293b'>{sym}</div>
  <div style='font-size:0.78rem;color:#334155'>₹{price_m:,.0f}</div>
  <div style='font-size:0.72rem;color:{col_m}'>{sig_m}</div>
  <div style='font-size:0.7rem;color:{"#16a34a" if chg_m>=0 else "#dc2626"}'>{chg_m:+.1f}%</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"#### {selected_symbol}")

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
        sc[0].metric("Price",    f"₹{last_price:,.1f}", f"{day_chg:+.2f}%")
        sc[1].metric("1W",       f"{week_chg:+.2f}%")
        sc[2].metric("1M",       f"{month_chg:+.2f}%")
        sc[3].metric("52W High", f"₹{high52:,.0f}", f"{pct_from_hi:.1f}%")
        sc[4].metric("ATR",      f"₹{atr_val:,.0f}")
        sc[5].metric("RSI-14",   f"{rsi_val:.1f}")
        sc[6].metric("ADX-14",   f"{adx_val:.1f}")
        sc[7].metric("Vol×Avg",  f"{vol_ratio:.2f}×")

        # Stock intelligence
        intel = classify_stock_state(last_row)
        pills_html = ""
        for label, (text, cls) in [("Trend", intel["trend"]), ("Momentum", intel["momentum"]),
                                    ("Volatility", intel["volatility"]), ("State", intel["state"])]:
            pills_html += f"<span style='font-size:0.68rem;color:#64748b;margin-right:2px'>{label}:</span>"
            pills_html += f"<span class='intel-pill {cls}'>{text}</span> &nbsp;"
        st.markdown(f"<div style='margin:4px 0 8px'>{pills_html}</div>", unsafe_allow_html=True)

        # Signal card with explanation
        if not scan_df.empty:
            sr_row = scan_df[scan_df["Symbol"] == selected_symbol]
            if not sr_row.empty:
                sr     = sr_row.iloc[0]
                sig    = sr["Signal"]
                color  = SIGNAL_COLORS.get(sig, "#64748b")
                bg     = SIGNAL_BG.get(sig, "#f1f5f9")
                grade  = sr.get("Grade","")
                expl   = explain_signal(sig, sr.get("Passed",""), sr.get("Blockers",""), sr["BUY%"], grade)
                stop_s = f"Stop <b>₹{sr['Stop']:,.1f}</b> &nbsp;" if pd.notna(sr.get("Stop")) else ""
                tgt_s  = f"Target <b>₹{sr['Target']:,.1f}</b>" if pd.notna(sr.get("Target")) else ""
                pass_s = f"<span style='font-size:0.72rem;color:#64748b'>Gates: {sr['Passed']}</span>" if sr.get("Passed") else ""
                risk_tags = ""
                if sr.get("RSI_warn"):
                    risk_tags += "<span style='font-size:0.7rem;background:#fef9c3;color:#a16207;border-radius:4px;padding:1px 7px;margin-right:4px'>⚠ RSI >70 — overbought entry</span>"
                if sr.get("ATR_warn"):
                    risk_tags += "<span style='font-size:0.7rem;background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 7px;margin-right:4px'>⚠ High ATR — stop wider than normal</span>"
                if sr.get("WeakBUY"):
                    risk_tags += "<span style='font-size:0.7rem;background:#f1f5f9;color:#64748b;border-radius:4px;padding:1px 7px'>Grade C + RSI>65 + low volume — HOLD is acceptable</span>"
                sideways_note = ""
                if trend_regime == "SIDEWAYS" and sig == "BUY":
                    sideways_note = "<div style='font-size:0.7rem;background:#fef9c3;color:#a16207;border-radius:4px;padding:3px 8px;margin-top:4px'>⚠ Sideways regime — lower signal reliability, tighten stop</div>"
                st.markdown(f"""
<div class='card' style='border-left:4px solid {color};margin-bottom:8px;background:{bg}20'>
  <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>
    <span class='badge badge-{sig}'>{sig}{' ' + grade if grade else ''}</span>
    <span style='font-size:0.85rem;color:#334155'>BUY% <b>{sr['BUY%']}</b> &nbsp; Confs <b>{sr['Confs']}</b> &nbsp; Rank <b>#{sr['Rank']}</b></span>
    <span style='font-size:0.85rem;color:#334155'>{stop_s}{tgt_s}</span>
  </div>
  <div style='font-size:0.78rem;color:#475569;margin-bottom:4px'>{expl}</div>
  {pass_s}
  {risk_tags}
  {sideways_note}
</div>""", unsafe_allow_html=True)

        fig = make_chart(
            stock_df, selected_symbol, lookback,
            show_ema9, show_ema20, show_ema50, show_ema200,
            show_bb, show_supertrend, show_pivots, show_sr,
            model, model_features,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Signal Scan table (collapsible) ──────────────────────────────
    if not scan_df.empty:
        n_total = len(scan_df)
        n_shown = len(filtered_df) if not filtered_df.empty else 0
        label   = f"Signal Scan — {n_shown} of {n_total} signals"
        with st.expander(label, expanded=False):
            def _style_row(row):
                styles = []
                for col in row.index:
                    sig = row.get("Signal","HOLD")
                    if col == "Signal":
                        bg = {"BUY":"#dcfce7","WATCH":"#fef9c3","SELL":"#fee2e2"}.get(sig,"#f1f5f9")
                        fg = {"BUY":"#15803d","WATCH":"#a16207","SELL":"#b91c1c"}.get(sig,"#64748b")
                        styles.append(f"background:{bg};color:{fg};font-weight:700;border-radius:6px")
                    elif col == "1D%" and pd.notna(row.get("1D%")):
                        styles.append(f"color:{'#16a34a' if row['1D%']>=0 else '#dc2626'}")
                    elif col == "Grade":
                        fg = {"A":"#b45309","B":"#4b5563","C":"#6b7280"}.get(row.get("Grade",""),"#94a3b8")
                        styles.append(f"color:{fg};font-weight:700")
                    else:
                        styles.append("color:#334155")
                return styles

            show_cols = ["Symbol","Price","1D%","Signal","Grade","BUY%","Confs","RSI"]
            disp = filtered_df[show_cols] if not filtered_df.empty else pd.DataFrame(columns=show_cols)
            styled = (
                disp.style
                .apply(_style_row, axis=1)
                .format({"Price": "₹{:,.1f}", "1D%": "{:+.2f}%", "BUY%": "{:.1f}"})
            )
            try:
                event = st.dataframe(
                    styled, use_container_width=True, height=400,
                    on_select="rerun", selection_mode="single-row", key="scan_table",
                )
                if event.selection and event.selection.rows and not filtered_df.empty:
                    selected_symbol = filtered_df.iloc[event.selection.rows[0]]["Symbol"]
            except Exception:
                st.dataframe(styled, use_container_width=True, height=400)

            st.download_button(
                "⬇ Export CSV", scan_df.to_csv(index=False),
                file_name=f"nse_signals_{date.today()}.csv",
                mime="text/csv",
            )

    # ── News panel ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Market News")
    with st.spinner("Fetching latest headlines…"):
        articles = fetch_news(max_per_feed=2)

    if articles:
        n_cols = st.columns(2)
        mid = max(1, len(articles) // 2)
        for col_n, chunk in zip(n_cols, [articles[:mid], articles[mid:]]):
            if not chunk:
                continue
            with col_n:
                news_html = ""
                for a in chunk:
                    news_html += f"""
<div class='news-item'>
  <span class='news-source src-{a["cls"]}'>{a["source"]}</span>
  <span style='font-size:0.7rem;color:#94a3b8'>{a["pub"]}</span><br>
  <a href='{a["link"]}' target='_blank'
     style='color:#1e293b;font-size:0.82rem;font-weight:500;text-decoration:none'>
    {a["title"]}
  </a>
</div>"""
                st.markdown(f"<div class='card'>{news_html}</div>", unsafe_allow_html=True)
    else:
        st.info("No news from the last 7 days — feeds may be unavailable or returning stale content.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO BUILDER
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Monthly Portfolio Builder")
    st.markdown("<p style='color:#64748b;font-size:0.88rem'>ATR-based position sizing · max 2% risk per trade · grade-weighted allocation</p>", unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns([1, 1, 2])
    capital = pc1.number_input("Capital (₹)", min_value=10000, max_value=10000000,
                                value=100000, step=10000, format="%d")
    max_pos = pc2.number_input("Max positions", min_value=1, max_value=10, value=5)
    pc3.markdown(
        "<div style='padding:10px 0;color:#64748b;font-size:0.83rem'>"
        "Grade A → 25–30% allocation · Grade B → 20–25% · Grade C → 10–15%<br>"
        "Cash reserve ≥ 10% of capital · Max risk per trade: 2%</div>",
        unsafe_allow_html=True
    )

    if st.button("Build Portfolio", type="primary"):
        with st.spinner("Building portfolio…"):
            if scan_df.empty:
                scan_df = run_scan()

        buys = scan_df[scan_df["Signal"] == "BUY"].copy()

        if buys.empty:
            st.warning("No BUY signals — check WATCH list or refresh.")
            watch = scan_df[scan_df["Signal"] == "WATCH"].head(5)
            if not watch.empty:
                st.markdown("**WATCH candidates (close to BUY):**")
                st.dataframe(watch[["Symbol","Price","BUY%","Confs","Blockers"]], hide_index=True)
        else:
            grade_alloc = {"A": 0.275, "B": 0.225, "C": 0.125}
            buys = buys.head(int(max_pos))
            portfolio_rows = []
            total_invested = total_risk = 0

            for _, row in buys.iterrows():
                grade  = row.get("Grade","C") or "C"
                alloc  = min(capital * grade_alloc.get(grade, 0.125), capital * 0.30)
                price  = float(row["Price"])
                stop   = float(row["Stop"])   if pd.notna(row.get("Stop"))   else price * 0.95
                target = float(row["Target"]) if pd.notna(row.get("Target")) else price * 1.10
                risk_pp = price - stop
                if risk_pp <= 0:
                    continue

                shares = min(int((capital * 0.02) / risk_pp), int(alloc / price))
                if shares <= 0:
                    shares = 1

                invested   = shares * price
                risk_trade = shares * risk_pp
                potential  = shares * (target - price)
                rr = round(potential / risk_trade, 2) if risk_trade else 0

                total_invested += invested
                total_risk     += risk_trade

                portfolio_rows.append({
                    "#": len(portfolio_rows) + 1,
                    "Stock": row["Symbol"], "Sector": row.get("Sector",""), "Grade": grade,
                    "Entry ₹": f"₹{price:,.1f}", "Shares": shares,
                    "Invested ₹": f"₹{invested:,.0f}", "Stop ₹": f"₹{stop:,.1f}",
                    "Target ₹": f"₹{target:,.1f}", "Max Risk ₹": f"₹{risk_trade:,.0f}",
                    "R:R": f"1:{rr}", "BUY%": row["BUY%"],
                })

            if not portfolio_rows:
                st.error("Could not size positions. Check stop-loss levels.")
            else:
                port_df = pd.DataFrame(portfolio_rows)
                deployed_pct   = total_invested / capital * 100
                risk_pct_total = total_risk / capital * 100

                # Sector concentration check
                from collections import Counter
                sector_counts = Counter(r["Sector"] for r in portfolio_rows if r["Sector"])
                overweight = [(sec, n) for sec, n in sector_counts.items() if n >= 2]
                if overweight:
                    warn_parts = ", ".join(f"{sec} ×{n}" for sec, n in overweight)
                    st.warning(f"⚠ Sector concentration: {warn_parts} — correlated drawdown risk if sector declines.")

                st.markdown(f"""
<div class='card card-accent-green' style='margin-bottom:12px'>
  <div style='display:flex;gap:32px;flex-wrap:wrap'>
    <div><span style='font-size:0.7rem;color:#64748b'>CAPITAL</span><br><b style='color:#1e293b'>₹{capital:,.0f}</b></div>
    <div><span style='font-size:0.7rem;color:#64748b'>DEPLOYED</span><br><b style='color:#1e293b'>₹{total_invested:,.0f} ({deployed_pct:.1f}%)</b></div>
    <div><span style='font-size:0.7rem;color:#64748b'>CASH RESERVE</span><br><b style='color:#1e293b'>₹{capital-total_invested:,.0f}</b></div>
    <div><span style='font-size:0.7rem;color:#dc2626'>TOTAL RISK</span><br><b style='color:#dc2626'>₹{total_risk:,.0f} ({risk_pct_total:.1f}%)</b></div>
  </div>
</div>""", unsafe_allow_html=True)

                styled_port = port_df.style.applymap(
                    lambda v: {"A":"color:#b45309;font-weight:800","B":"color:#4b5563;font-weight:700","C":"color:#6b7280"}.get(v,""),
                    subset=["Grade"]
                )
                st.dataframe(styled_port, use_container_width=True, hide_index=True)

                alloc_vals  = [int(r["Invested ₹"].replace("₹","").replace(",","")) for r in portfolio_rows]
                alloc_names = [r["Stock"] for r in portfolio_rows]
                cash_val    = capital - sum(alloc_vals)
                if cash_val > 0:
                    alloc_names.append("Cash"); alloc_vals.append(int(cash_val))

                pie_fig = go.Figure(go.Pie(
                    labels=alloc_names, values=alloc_vals, hole=0.48,
                    textinfo="label+percent",
                    marker=dict(colors=px.colors.qualitative.Pastel),
                ))
                pie_fig.update_layout(
                    template="plotly_white", paper_bgcolor="#ffffff",
                    height=320, margin=dict(l=10, r=10, t=30, b=10),
                    title=dict(text="Portfolio Allocation", font=dict(color="#1e293b", size=13)),
                    showlegend=False,
                )
                st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("""
<div class='card' style='margin-top:12px'>
  <b style='color:#1e293b'>Exit rules:</b>
  <span style='color:#334155;font-size:0.85rem'>
  &nbsp; ✅ Hit target → close full position &nbsp;
  ❌ Hit stop → close, accept loss &nbsp;
  ⏰ 10 trading days elapsed → exit if no trigger
  </span><br>
  <span style='color:#94a3b8;font-size:0.78rem'>⚠️ Educational only — not financial advice.</span>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — SECTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Sector Analysis")

    if scan_df.empty:
        scan_df = run_scan()

    if scan_df.empty:
        st.info("No scan data available.")
    else:
        sector_df    = scan_df.copy()
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
                        styles.append("color:#15803d;font-weight:700" if row["BUY"] > 0 else "color:#334155")
                    elif col == "SELL":
                        styles.append("color:#b91c1c;font-weight:700" if row["SELL"] > 0 else "color:#334155")
                    elif col == "BUY_rate%":
                        pct = row["BUY_rate%"]
                        styles.append(f"color:{'#15803d' if pct>30 else '#a16207' if pct>0 else '#64748b'};font-weight:700")
                    elif col == "Avg_1D":
                        styles.append(f"color:{'#16a34a' if row['Avg_1D']>=0 else '#dc2626'}")
                    else:
                        styles.append("color:#334155")
                return styles

            styled_s = (
                sector_stats.rename(columns={"Avg_BUY_pct":"AvgBUY%","Avg_1D":"Avg1D%"})
                .style.apply(_sect_style, axis=1)
                .format({"AvgBUY%":"{:.1f}","Avg1D%":"{:+.2f}%","BUY_rate%":"{:.1f}%"})
            )
            st.dataframe(styled_s, use_container_width=True, height=420, hide_index=True)

        with sc_r:
            st.markdown("#### BUY Signal Distribution")
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                y=sector_stats["Sector"], x=sector_stats["BUY"], name="BUY",
                orientation="h", marker_color="#86efac",
                marker_line_color="#16a34a", marker_line_width=1,
                text=sector_stats["BUY"], textposition="outside",
            ))
            bar_fig.add_trace(go.Bar(
                y=sector_stats["Sector"], x=sector_stats["WATCH"], name="WATCH",
                orientation="h", marker_color="#fde68a",
                marker_line_color="#ca8a04", marker_line_width=1,
                text=sector_stats["WATCH"], textposition="outside",
            ))
            bar_fig.update_layout(
                template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                height=420, barmode="stack",
                margin=dict(l=10, r=60, t=20, b=10),
                legend=dict(orientation="h", y=1.05, font=dict(color="#334155")),
                xaxis_title="Number of stocks",
                font=dict(color="#334155"),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("#### Stock-level breakdown")
        selected_sector = st.selectbox("Sector", ["All"] + sorted(sector_df["Sector"].unique().tolist()))
        filtered_sect = sector_df if selected_sector == "All" else sector_df[sector_df["Sector"] == selected_sector]
        show_sect_cols = ["Symbol","Sector","Price","1D%","Signal","Grade","BUY%","Confs","RSI","VolRatio"]
        st.dataframe(filtered_sect[show_sect_cols].reset_index(drop=True),
                     use_container_width=True, height=300, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### Deep Dive Analysis")

    dd_symbol = st.selectbox("Stock", NIFTY_50,
                              index=NIFTY_50.index(selected_symbol) if selected_symbol in NIFTY_50 else 0,
                              key="dd_sym")

    with st.spinner(f"Loading {dd_symbol}…"):
        dd_df = get_stock_data(dd_symbol, 365 + 200)

    if dd_df is None or dd_df.empty:
        st.error(f"No data for {dd_symbol}.")
    else:
        dd_last  = dd_df.iloc[-1]
        dd_price = float(dd_last["Close"])
        dd_atr   = float(dd_last.get("ATR", 0) or 0)

        # Intelligence card
        intel = classify_stock_state(dd_last)
        i_html = ""
        for label, (text, cls) in [("Trend", intel["trend"]), ("Momentum", intel["momentum"]),
                                     ("Volatility", intel["volatility"]), ("State", intel["state"])]:
            i_html += f"<div style='text-align:center'><div style='font-size:0.65rem;color:#64748b;font-weight:600;margin-bottom:2px'>{label.upper()}</div><span class='intel-pill {cls}'>{text}</span></div>"

        st.markdown(f"<div class='card' style='display:flex;gap:24px;justify-content:flex-start;padding:12px 20px'>{i_html}</div>", unsafe_allow_html=True)

        st.markdown("#### Return Profile")
        rc = st.columns(6)
        for i, (label, days) in enumerate([("1D",1),("1W",5),("1M",22),("3M",66),("6M",132),("1Y",252)]):
            if len(dd_df) > days:
                chg = (dd_price / float(dd_df.iloc[-(days+1)]["Close"]) - 1) * 100
                rc[i].metric(label, f"{chg:+.2f}%")

        st.markdown("#### 7-Gate Confirmation")

        def gv_dd(col, fb=0.0):
            return float(dd_last.get(col, fb) or fb)

        gates = {
            "EMA200\n(price > 200d)": gv_dd("feat_ema200_ratio") > 0,
            "EMA50\n(price > 50d)":   gv_dd("feat_ema50_ratio")  > 0,
            "EMA20\n(price > 20d)":   gv_dd("feat_ema20_ratio")  > 0,
            f"ADX ≥ {ADX_MIN*100:.0f}":    gv_dd("feat_adx")         >= ADX_MIN,
            "MACD hist ≥ 0":          gv_dd("feat_macd_hist")   >= 0,
            "Near 20d high":          gv_dd("feat_dist_20d_high") > 0,
            f"Vol ≥ {VOLUME_THRESHOLD}× avg": gv_dd("feat_volume_ratio") >= VOLUME_THRESHOLD,
        }
        g_cols = st.columns(7)
        for i, (label, passed) in enumerate(gates.items()):
            icon  = "✅" if passed else "❌"
            bg    = "#dcfce7" if passed else "#fee2e2"
            color = "#15803d" if passed else "#b91c1c"
            border_col = "#86efac" if passed else "#fca5a5"
            g_cols[i].markdown(
                f"<div style='background:{bg};border:1px solid {border_col};border-radius:10px;"
                f"text-align:center;padding:10px 4px'>"
                f"<div style='font-size:1.3rem'>{icon}</div>"
                f"<div style='font-size:0.65rem;color:{color};font-weight:600;line-height:1.3'>{label}</div>"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("#### Model Feature Values (top 16)")
        feat_vals = {}
        for col in model_features[:16]:
            v = float(dd_last.get(col, 0) or 0)
            feat_vals[col.replace("feat_", "")] = round(v, 4)

        feat_df = pd.DataFrame([feat_vals])
        feat_fig = go.Figure(go.Heatmap(
            z=feat_df.values, x=list(feat_vals.keys()), y=[""],
            colorscale="RdYlGn", showscale=True,
            text=[[f"{v:.3f}" for v in feat_df.values[0]]], texttemplate="%{text}",
        ))
        feat_fig.update_layout(
            template="plotly_white", paper_bgcolor="#ffffff",
            height=110, margin=dict(l=10, r=10, t=10, b=40),
            font=dict(color="#334155"),
        )
        st.plotly_chart(feat_fig, use_container_width=True)

        st.markdown("#### Rolling BUY Probability (90 days)")
        dd_trim = dd_df.tail(90 + 50).copy()
        feat_matrix = np.column_stack(
            [dd_trim.get(col, pd.Series(0, index=dd_trim.index)).fillna(0).values for col in model_features]
        )
        probs = model.predict_proba(feat_matrix)[:, 1]
        dd_trim["prob"] = probs
        dd_trim = dd_trim.tail(90)
        dd_trim["DateTime"] = pd.to_datetime(dd_trim["DateTime"])

        prob_fig = go.Figure()
        prob_fig.add_trace(go.Scatter(
            x=dd_trim["DateTime"], y=dd_trim["prob"] * 100,
            fill="tozeroy", fillcolor="rgba(22,163,74,0.1)",
            line=dict(color="#16a34a", width=1.5), name="BUY prob %",
        ))
        prob_fig.add_hline(y=BUY_PROBA * 100, line_dash="dot",
                           line_color="#16a34a", line_width=1.5,
                           annotation_text=f"BUY threshold {BUY_PROBA*100:.0f}%",
                           annotation_font_color="#16a34a")
        prob_fig.update_layout(
            template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
            height=210, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="BUY%", yaxis_range=[0, 100],
            font=dict(color="#334155"),
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        st.markdown("#### Trade Risk Calculator")
        rk1, rk2, rk3 = st.columns(3)
        trade_capital = rk1.number_input("Capital (₹)", value=100000, step=10000, min_value=10000, key="rk_cap")
        risk_pct      = rk2.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.25, key="rk_pct")
        custom_entry  = rk3.number_input("Entry price (₹)", value=float(round(dd_price, 1)), key="rk_entry")

        stop_price    = custom_entry - 1.5 * dd_atr
        target_price  = custom_entry + 3.0 * dd_atr
        risk_amount   = trade_capital * risk_pct / 100
        risk_per_share = custom_entry - stop_price
        shares_calc   = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        invest_total  = shares_calc * custom_entry
        potential     = shares_calc * (target_price - custom_entry)

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
    f"Model v{blob.get('version','2')} · trained {blob.get('trained_at','?')} · "
    f"{blob.get('n_train',0):,} samples · {len(model_features)} features · "
    f"val acc {blob.get('val_acc',0)*100:.1f}% · "
    f"Auto-refreshes every 30 min · ⚠️ Educational only — not financial advice"
)
