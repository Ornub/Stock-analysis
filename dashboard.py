"""
dashboard.py — NSE Swing Trading Dashboard v3  (Streamlit + Plotly)
Institutional-grade light theme with news panel, signal intelligence, and quick filters.
Educational only — not financial advice.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
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
    NIFTY_50, NIFTY_NEXT_50, FEATURE_COLS, MODEL_PATH,
    compute_features, _fetch_market_regime, classify_regime, rank_candidates,
    ADX_MIN, BUY_PROBA, BUY_PROBA_STRONG_NEWS, MIN_CONFIRMATIONS,
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

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

_DARK_CSS = """
<style>
/* ── Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root tokens ────────────────────────────────────────────────────── */
:root {
  --bg-card:       rgba(17,24,39,0.75);
  --border:        rgba(59,130,246,0.12);
  --border-hi:     rgba(59,130,246,0.28);
  --blue:          #3B82F6;
  --cyan:          #06B6D4;
  --green:         #10B981;
  --amber:         #F59E0B;
  --red:           #EF4444;
  --text-1:        #F8FAFC;
  --text-2:        #94A3B8;
  --text-3:        #64748B;
  --glow-b: 0 0 24px rgba(59,130,246,0.35);
  --glow-g: 0 0 20px rgba(16,185,129,0.30);
  --glow-r: 0 0 20px rgba(239,68,68,0.30);
  --glow-a: 0 0 20px rgba(245,158,11,0.25);
}

/* ── App shell ──────────────────────────────────────────────────────── */
.stApp {
  background: linear-gradient(145deg,#0B1020 0%,#0F172A 60%,#070D1A 100%) !important;
  color: var(--text-1) !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
}
.block-container { padding-top:1.4rem !important; padding-bottom:0.5rem !important; max-width:1600px !important; }

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(7,13,26,0.97) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(24px) !important;
}
[data-testid="stSidebar"]::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg,#3B82F6,#06B6D4,#10B981);
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 { color: var(--text-1) !important; font-family:'Space Grotesk',sans-serif !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p  { color: var(--text-2) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(17,24,39,0.9) !important;
  border-color: var(--border-hi) !important;
  color: var(--text-1) !important;
  border-radius: 8px !important;
}

/* ── Typography ─────────────────────────────────────────────────────── */
h1,h2,h3,h4 { color:var(--text-1) !important; font-weight:700 !important; font-family:'Space Grotesk',sans-serif !important; letter-spacing:-0.02em; }
.stMarkdown p { color:var(--text-2) !important; }
label { color:var(--text-2) !important; }

/* ── Tab pills ──────────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
  background: rgba(17,24,39,0.92) !important;
  border-radius: 50px !important;
  padding: 5px !important;
  border: 1px solid var(--border) !important;
  gap: 2px !important;
}
div[data-baseweb="tab-list"] button {
  border-radius: 50px !important;
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  color: var(--text-3) !important;
  transition: all 0.22s ease !important;
  padding: 6px 18px !important;
  border: none !important;
  background: transparent !important;
}
div[data-baseweb="tab-list"] button:hover { color:var(--text-2) !important; background:rgba(59,130,246,0.08) !important; }
div[data-baseweb="tab-list"] button[aria-selected="true"] {
  color:#fff !important;
  background: linear-gradient(135deg,#3B82F6,#06B6D4) !important;
  box-shadow: 0 0 22px rgba(59,130,246,0.45) !important;
}
div[data-baseweb="tab-highlight"] { background:transparent !important; height:0 !important; }
div[data-baseweb="tab-border"]    { display:none !important; }

/* ── Metrics ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  backdrop-filter: blur(12px) !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stMetric"]:hover { border-color:var(--border-hi) !important; box-shadow:var(--glow-b) !important; }
[data-testid="stMetricLabel"]  { color:var(--text-3) !important; font-size:0.68rem !important; text-transform:uppercase !important; letter-spacing:0.09em !important; font-weight:600 !important; }
[data-testid="stMetricValue"]  { color:var(--text-1) !important; font-size:1.05rem !important; font-weight:700 !important; font-family:'Space Grotesk',sans-serif !important; }
[data-testid="stMetricDelta"]  { font-size:0.78rem !important; }

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
  background: rgba(59,130,246,0.10) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 8px !important;
  color: #93C5FD !important;
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  letter-spacing: 0.02em !important;
  transition: all .2s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg,#3B82F6,#06B6D4) !important;
  color: #fff !important;
  border-color: transparent !important;
  box-shadow: 0 0 28px rgba(59,130,246,0.50) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform:translateY(0) !important; }

/* ── DataFrames ─────────────────────────────────────────────────────── */
.stDataFrame { border-radius:12px !important; overflow:hidden !important; border:1px solid var(--border) !important; }

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--bg-card) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"]:hover { border-color:var(--border-hi) !important; }
[data-testid="stExpander"] summary { color:var(--text-2) !important; font-weight:600 !important; }

/* ── Select / Dropdowns ─────────────────────────────────────────────── */
[data-baseweb="select"] > div { background:rgba(17,24,39,0.9) !important; border-color:var(--border-hi) !important; border-radius:8px !important; color:var(--text-1) !important; }
[data-baseweb="menu"]         { background:#111827 !important; border:1px solid var(--border-hi) !important; border-radius:10px !important; }
[data-baseweb="option"]:hover { background:rgba(59,130,246,0.14) !important; }

/* ── Sliders ────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] { background:#3B82F6 !important; box-shadow:0 0 10px rgba(59,130,246,0.6) !important; }

/* ── Alerts/Status ──────────────────────────────────────────────────── */
[data-testid="stInfo"]    { background:rgba(59,130,246,0.08)  !important; border:1px solid rgba(59,130,246,0.22)  !important; border-radius:10px !important; color:#93C5FD !important; }
[data-testid="stSuccess"] { background:rgba(16,185,129,0.08)  !important; border:1px solid rgba(16,185,129,0.22)  !important; border-radius:10px !important; color:#6EE7B7 !important; }
[data-testid="stWarning"] { background:rgba(245,158,11,0.08)  !important; border:1px solid rgba(245,158,11,0.22)  !important; border-radius:10px !important; color:#FCD34D !important; }
[data-testid="stError"]   { background:rgba(239,68,68,0.08)   !important; border:1px solid rgba(239,68,68,0.22)   !important; border-radius:10px !important; color:#FCA5A5 !important; }

/* ── Signal badges ──────────────────────────────────────────────────── */
.badge { display:inline-block; padding:2px 10px; border-radius:99px; font-weight:700; font-size:0.74rem; letter-spacing:0.4px; }
.badge-BUY   { background:rgba(16,185,129,0.18);  color:#34D399; border:1px solid rgba(16,185,129,0.40); }
.badge-WATCH { background:rgba(245,158,11,0.15);  color:#FCD34D; border:1px solid rgba(245,158,11,0.38); }
.badge-SELL  { background:rgba(239,68,68,0.15);   color:#FCA5A5; border:1px solid rgba(239,68,68,0.38); }
.badge-HOLD  { background:rgba(100,116,139,0.18); color:#94A3B8; border:1px solid rgba(100,116,139,0.32); }

/* ── Glass cards ────────────────────────────────────────────────────── */
.card {
  background: rgba(17,24,39,0.75);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 20px;
  margin: 6px 0;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  transition: border-color .2s, box-shadow .2s;
}
.card:hover { border-color:var(--border-hi); box-shadow:0 6px 28px rgba(0,0,0,0.45); }
.card-accent-green  { border-left:3px solid #10B981; box-shadow:-4px 0 18px rgba(16,185,129,0.12); }
.card-accent-yellow { border-left:3px solid #F59E0B; box-shadow:-4px 0 18px rgba(245,158,11,0.12); }
.card-accent-red    { border-left:3px solid #EF4444; box-shadow:-4px 0 18px rgba(239,68,68,0.12); }
.card-accent-blue   { border-left:3px solid #3B82F6; box-shadow:-4px 0 18px rgba(59,130,246,0.12); }
.card-accent-gray   { border-left:3px solid #475569; }

/* ── Regime banner ──────────────────────────────────────────────────── */
.regime-banner {
  background: rgba(17,24,39,0.85);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 22px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 24px;
  backdrop-filter: blur(16px);
}

/* ── Top signal cards ────────────────────────────────────────────────── */
/* ── Tier badges ─────────────────────────────────────────────────────── */
.tier-A { display:inline-flex;align-items:center;gap:4px;background:linear-gradient(135deg,rgba(251,191,36,0.18),rgba(245,158,11,0.10));
  color:#FCD34D;border:1px solid rgba(251,191,36,0.45);border-radius:6px;padding:2px 8px;
  font-size:0.68rem;font-weight:800;letter-spacing:.5px;text-transform:uppercase; }
.tier-B { display:inline-flex;align-items:center;gap:4px;background:rgba(148,163,184,0.12);
  color:#CBD5E1;border:1px solid rgba(148,163,184,0.30);border-radius:6px;padding:2px 8px;
  font-size:0.68rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase; }
.tier-C { display:inline-flex;align-items:center;gap:4px;background:rgba(100,116,139,0.10);
  color:#64748B;border:1px solid rgba(100,116,139,0.22);border-radius:6px;padding:2px 8px;
  font-size:0.68rem;font-weight:600;letter-spacing:.5px;text-transform:uppercase; }

/* ── Premium BUY signal card ─────────────────────────────────────────── */
@keyframes border-glow { 0%,100%{box-shadow:0 0 16px rgba(16,185,129,0.25)} 50%{box-shadow:0 0 28px rgba(16,185,129,0.45)} }
.signal-hub-card {
  background: linear-gradient(145deg,rgba(16,185,129,0.06) 0%,rgba(17,24,39,0.90) 50%);
  border: 1px solid rgba(16,185,129,0.35);
  border-left: 4px solid #10B981;
  border-radius: 14px;
  padding: 16px 20px;
  backdrop-filter: blur(20px);
  transition: all .25s ease;
  animation: border-glow 3s ease-in-out infinite;
  position: relative;
  overflow: hidden;
}
.signal-hub-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(16,185,129,0.6),transparent);
}
.signal-hub-card:hover { transform:translateY(-3px); box-shadow:0 12px 40px rgba(16,185,129,0.20); }
.signal-hub-card-watch {
  background: linear-gradient(145deg,rgba(245,158,11,0.05) 0%,rgba(17,24,39,0.90) 50%);
  border: 1px solid rgba(245,158,11,0.30); border-left: 4px solid #F59E0B;
  border-radius: 14px; padding: 16px 20px; backdrop-filter: blur(20px);
  transition: all .25s ease;
}
.signal-hub-card-watch:hover { transform:translateY(-2px); box-shadow:0 8px 28px rgba(245,158,11,0.15); }

/* ── Probability arc bar ─────────────────────────────────────────────── */
.prob-track { background:rgba(255,255,255,0.07); border-radius:99px; height:6px; overflow:hidden; margin:4px 0; }
.prob-fill-green { border-radius:99px; height:6px; background:linear-gradient(90deg,#10B981,#34D399); }
.prob-fill-amber { border-radius:99px; height:6px; background:linear-gradient(90deg,#F59E0B,#FCD34D); }

/* ── Risk/Stop pill ──────────────────────────────────────────────────── */
.stop-pill  { background:rgba(239,68,68,0.12); color:#FCA5A5; border:1px solid rgba(239,68,68,0.28);
  border-radius:6px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.tgt-pill   { background:rgba(16,185,129,0.12); color:#6EE7B7; border:1px solid rgba(16,185,129,0.28);
  border-radius:6px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.rr-pill    { background:rgba(59,130,246,0.12); color:#93C5FD; border:1px solid rgba(59,130,246,0.28);
  border-radius:6px; padding:2px 8px; font-size:0.72rem; font-weight:600; }

/* ── Gate dots ───────────────────────────────────────────────────────── */
.gate-dot-pass { display:inline-block;width:8px;height:8px;border-radius:50%;background:#10B981;
  box-shadow:0 0 6px rgba(16,185,129,0.5);margin:0 2px;vertical-align:middle; }
.gate-dot-fail { display:inline-block;width:8px;height:8px;border-radius:50%;background:rgba(239,68,68,0.4);
  border:1px solid rgba(239,68,68,0.5);margin:0 2px;vertical-align:middle; }

.top-signal-card {
  background: rgba(17,24,39,0.80);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 18px;
  backdrop-filter: blur(16px);
  transition: all .22s ease;
}
.top-signal-card:hover { border-color:var(--border-hi); box-shadow:0 8px 32px rgba(0,0,0,0.5); transform:translateY(-2px); }

/* ── News ────────────────────────────────────────────────────────────── */
.news-item { padding:9px 0; border-bottom:1px solid rgba(59,130,246,0.07); line-height:1.45; color:var(--text-2); }
.news-item:last-child { border-bottom:none; }
.news-source { display:inline-block; font-size:0.7rem; font-weight:700; padding:1px 7px; border-radius:99px; margin-right:6px; }
.src-ET      { background:rgba(59,130,246,0.15);  color:#93C5FD; }
.src-MC      { background:rgba(236,72,153,0.15);  color:#F9A8D4; }
.src-Reuters { background:rgba(245,158,11,0.15);  color:#FCD34D; }
.src-BS      { background:rgba(139,92,246,0.15);  color:#C4B5FD; }
.src-Mint    { background:rgba(139,92,246,0.14);  color:#C4B5FD; }
.src-HBL     { background:rgba(16,185,129,0.14);  color:#6EE7B7; }

/* ── Confidence bar ──────────────────────────────────────────────────── */
.conf-bar-wrap { background:rgba(255,255,255,0.06); border-radius:99px; height:5px; margin-top:4px; }
.conf-bar-fill { border-radius:99px; height:5px; }

/* ── Intelligence pills ──────────────────────────────────────────────── */
.intel-pill { display:inline-block; padding:3px 10px; border-radius:99px; font-size:0.7rem; font-weight:600; margin:2px 2px; }
.pill-green  { background:rgba(16,185,129,0.15);  color:#34D399; border:1px solid rgba(16,185,129,0.30); }
.pill-red    { background:rgba(239,68,68,0.15);   color:#FCA5A5; border:1px solid rgba(239,68,68,0.30); }
.pill-yellow { background:rgba(245,158,11,0.15);  color:#FCD34D; border:1px solid rgba(245,158,11,0.30); }
.pill-blue   { background:rgba(59,130,246,0.15);  color:#93C5FD; border:1px solid rgba(59,130,246,0.30); }
.pill-gray   { background:rgba(100,116,139,0.15); color:#94A3B8; border:1px solid rgba(100,116,139,0.28); }

/* ── Grade colors ────────────────────────────────────────────────────── */
.grade-A { color:#FCD34D; font-weight:800; }
.grade-B { color:#94A3B8; font-weight:700; }
.grade-C { color:#64748B; font-weight:600; }

/* ── Watchlist chips ─────────────────────────────────────────────────── */
.wl-chip { display:inline-block; background:rgba(59,130,246,0.10); border:1px solid rgba(59,130,246,0.25); border-radius:8px; padding:3px 10px; font-size:0.78rem; color:#93C5FD; font-weight:600; margin:2px; transition:all .15s; }
.wl-chip:hover { background:rgba(59,130,246,0.20); border-color:rgba(59,130,246,0.42); }

/* ── Misc ────────────────────────────────────────────────────────────── */
div[data-testid="stHorizontalBlock"] > div { padding:0 3px; }
hr { border-color:rgba(59,130,246,0.10) !important; }
.stCaption { color:var(--text-3) !important; }

/* ── Pulse animation ─────────────────────────────────────────────────── */
@keyframes pulse-glow { 0%,100%{opacity:1} 50%{opacity:0.45} }
.live-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#10B981; box-shadow:0 0 8px #10B981; animation:pulse-glow 2s ease-in-out infinite; vertical-align:middle; margin-right:5px; }

/* ── Premium news cards ──────────────────────────────────────────────── */
.news-card-prem { background:rgba(17,24,39,0.65);border:1px solid var(--border);border-radius:12px;padding:10px 14px;margin-bottom:8px;transition:border-color .2s; }
.news-card-prem:hover { border-color:var(--border-hi); }
.sent-bull { display:inline-block;font-size:0.6rem;font-weight:700;padding:1px 6px;border-radius:4px;background:rgba(16,185,129,0.12);color:#34D399;border:1px solid rgba(16,185,129,0.28);margin-left:4px;vertical-align:middle; }
.sent-bear { display:inline-block;font-size:0.6rem;font-weight:700;padding:1px 6px;border-radius:4px;background:rgba(239,68,68,0.12);color:#FCA5A5;border:1px solid rgba(239,68,68,0.28);margin-left:4px;vertical-align:middle; }
.sent-neut { display:inline-block;font-size:0.6rem;font-weight:700;padding:1px 6px;border-radius:4px;background:rgba(100,116,139,0.12);color:#94A3B8;border:1px solid rgba(100,116,139,0.22);margin-left:4px;vertical-align:middle; }

/* ── Historical insight mini-card ────────────────────────────────────── */
.insight-card { background:linear-gradient(145deg,rgba(59,130,246,0.06),rgba(17,24,39,0.85));border:1px solid rgba(59,130,246,0.20);border-radius:12px;padding:12px 16px;text-align:center;transition:border-color .2s; }
.insight-card:hover { border-color:rgba(59,130,246,0.40); }

/* ── Footnote / disclaimer box ───────────────────────────────────────── */
.footnote-box { background:rgba(239,68,68,0.04);border:1px solid rgba(239,68,68,0.16);border-left:3px solid rgba(239,68,68,0.55);border-radius:10px;padding:12px 16px;margin-top:12px;font-size:0.74rem;color:#94A3B8;line-height:1.65; }
.footnote-box b { color:#FCA5A5; }
.footnote-box h5 { color:#CBD5E1;font-size:0.78rem;margin:8px 0 4px; }

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#0B1020; }
::-webkit-scrollbar-thumb { background:rgba(59,130,246,0.30); border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background:rgba(59,130,246,0.50); }
</style>
"""

_LIGHT_CSS = """
<style>
/* ── Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root tokens ────────────────────────────────────────────────────── */
:root {
  --bg-card:       rgba(255,255,255,0.90);
  --border:        rgba(59,130,246,0.18);
  --border-hi:     rgba(59,130,246,0.40);
  --blue:          #2563EB;
  --cyan:          #0891B2;
  --green:         #059669;
  --amber:         #D97706;
  --red:           #DC2626;
  --text-1:        #0F172A;
  --text-2:        #334155;
  --text-3:        #64748B;
  --glow-b: 0 0 18px rgba(37,99,235,0.20);
  --glow-g: 0 0 14px rgba(5,150,105,0.18);
  --glow-r: 0 0 14px rgba(220,38,38,0.18);
  --glow-a: 0 0 14px rgba(217,119,6,0.15);
}

/* ── App shell ──────────────────────────────────────────────────────── */
.stApp {
  background: linear-gradient(145deg,#F1F5F9 0%,#EEF2FF 60%,#F8FAFC 100%) !important;
  color: var(--text-1) !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
}
.block-container { padding-top:1.4rem !important; padding-bottom:0.5rem !important; max-width:1600px !important; }

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.97) !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: 2px 0 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stSidebar"]::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg,#2563EB,#0891B2,#059669);
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 { color: var(--text-1) !important; font-family:'Space Grotesk',sans-serif !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p  { color: var(--text-2) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(241,245,249,0.9) !important;
  border-color: var(--border-hi) !important;
  color: var(--text-1) !important;
  border-radius: 8px !important;
}

/* ── Typography ─────────────────────────────────────────────────────── */
h1,h2,h3,h4 { color:var(--text-1) !important; font-weight:700 !important; font-family:'Space Grotesk',sans-serif !important; letter-spacing:-0.02em; }
.stMarkdown p { color:var(--text-2) !important; }
label { color:var(--text-2) !important; }

/* ── Tab pills ──────────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
  background: rgba(226,232,240,0.80) !important;
  border-radius: 50px !important;
  padding: 5px !important;
  border: 1px solid var(--border) !important;
  gap: 2px !important;
}
div[data-baseweb="tab-list"] button {
  border-radius: 50px !important;
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  color: var(--text-3) !important;
  transition: all 0.22s ease !important;
  padding: 6px 18px !important;
  border: none !important;
  background: transparent !important;
}
div[data-baseweb="tab-list"] button:hover { color:var(--text-2) !important; background:rgba(37,99,235,0.08) !important; }
div[data-baseweb="tab-list"] button[aria-selected="true"] {
  color:#fff !important;
  background: linear-gradient(135deg,#2563EB,#0891B2) !important;
  box-shadow: 0 0 16px rgba(37,99,235,0.30) !important;
}
div[data-baseweb="tab-highlight"] { background:transparent !important; height:0 !important; }
div[data-baseweb="tab-border"]    { display:none !important; }

/* ── Metrics ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stMetric"]:hover { border-color:var(--border-hi) !important; box-shadow:var(--glow-b) !important; }
[data-testid="stMetricLabel"]  { color:var(--text-3) !important; font-size:0.68rem !important; text-transform:uppercase !important; letter-spacing:0.09em !important; font-weight:600 !important; }
[data-testid="stMetricValue"]  { color:var(--text-1) !important; font-size:1.05rem !important; font-weight:700 !important; font-family:'Space Grotesk',sans-serif !important; }
[data-testid="stMetricDelta"]  { font-size:0.78rem !important; }

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
  background: rgba(37,99,235,0.08) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 8px !important;
  color: #1D4ED8 !important;
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  letter-spacing: 0.02em !important;
  transition: all .2s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg,#2563EB,#0891B2) !important;
  color: #fff !important;
  border-color: transparent !important;
  box-shadow: 0 0 20px rgba(37,99,235,0.30) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform:translateY(0) !important; }

/* ── DataFrames ─────────────────────────────────────────────────────── */
.stDataFrame { border-radius:12px !important; overflow:hidden !important; border:1px solid var(--border) !important; }

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--bg-card) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"]:hover { border-color:var(--border-hi) !important; }
[data-testid="stExpander"] summary { color:var(--text-2) !important; font-weight:600 !important; }

/* ── Select / Dropdowns ─────────────────────────────────────────────── */
[data-baseweb="select"] > div { background:rgba(241,245,249,0.9) !important; border-color:var(--border-hi) !important; border-radius:8px !important; color:var(--text-1) !important; }
[data-baseweb="menu"]         { background:#fff !important; border:1px solid var(--border-hi) !important; border-radius:10px !important; }
[data-baseweb="option"]:hover { background:rgba(37,99,235,0.08) !important; }

/* ── Sliders ────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] { background:#2563EB !important; box-shadow:0 0 8px rgba(37,99,235,0.4) !important; }

/* ── Alerts/Status ──────────────────────────────────────────────────── */
[data-testid="stInfo"]    { background:rgba(37,99,235,0.06)   !important; border:1px solid rgba(37,99,235,0.20)   !important; border-radius:10px !important; color:#1E40AF !important; }
[data-testid="stSuccess"] { background:rgba(5,150,105,0.06)   !important; border:1px solid rgba(5,150,105,0.20)   !important; border-radius:10px !important; color:#065F46 !important; }
[data-testid="stWarning"] { background:rgba(217,119,6,0.06)   !important; border:1px solid rgba(217,119,6,0.20)   !important; border-radius:10px !important; color:#92400E !important; }
[data-testid="stError"]   { background:rgba(220,38,38,0.06)   !important; border:1px solid rgba(220,38,38,0.20)   !important; border-radius:10px !important; color:#991B1B !important; }

/* ── Signal badges ──────────────────────────────────────────────────── */
.badge { display:inline-block; padding:2px 10px; border-radius:99px; font-weight:700; font-size:0.74rem; letter-spacing:0.4px; }
.badge-BUY   { background:rgba(5,150,105,0.12);   color:#065F46; border:1px solid rgba(5,150,105,0.32); }
.badge-WATCH { background:rgba(217,119,6,0.10);   color:#92400E; border:1px solid rgba(217,119,6,0.30); }
.badge-SELL  { background:rgba(220,38,38,0.10);   color:#991B1B; border:1px solid rgba(220,38,38,0.30); }
.badge-HOLD  { background:rgba(100,116,139,0.10); color:#475569; border:1px solid rgba(100,116,139,0.25); }

/* ── Glass cards ────────────────────────────────────────────────────── */
.card {
  background: rgba(255,255,255,0.90);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 20px;
  margin: 6px 0;
  box-shadow: 0 1px 8px rgba(0,0,0,0.06);
  transition: border-color .2s, box-shadow .2s;
}
.card:hover { border-color:var(--border-hi); box-shadow:0 4px 18px rgba(0,0,0,0.10); }
.card-accent-green  { border-left:3px solid #059669; }
.card-accent-yellow { border-left:3px solid #D97706; }
.card-accent-red    { border-left:3px solid #DC2626; }
.card-accent-blue   { border-left:3px solid #2563EB; }
.card-accent-gray   { border-left:3px solid #94A3B8; }

/* ── Regime banner ──────────────────────────────────────────────────── */
.regime-banner {
  background: rgba(255,255,255,0.88);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 22px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 24px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}

/* ── Top signal cards ────────────────────────────────────────────────── */
.top-signal-card {
  background: rgba(255,255,255,0.92);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 18px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.06);
  transition: all .22s ease;
}
.top-signal-card:hover { border-color:var(--border-hi); box-shadow:0 6px 20px rgba(0,0,0,0.10); transform:translateY(-2px); }

/* ── News ────────────────────────────────────────────────────────────── */
.news-item { padding:9px 0; border-bottom:1px solid rgba(37,99,235,0.10); line-height:1.45; color:var(--text-2); }
.news-item:last-child { border-bottom:none; }
.news-source { display:inline-block; font-size:0.7rem; font-weight:700; padding:1px 7px; border-radius:99px; margin-right:6px; }
.src-ET      { background:rgba(37,99,235,0.10);  color:#1D4ED8; }
.src-MC      { background:rgba(219,39,119,0.10); color:#9D174D; }
.src-Reuters { background:rgba(217,119,6,0.10);  color:#92400E; }
.src-BS      { background:rgba(109,40,217,0.10); color:#5B21B6; }

/* ── Confidence bar ──────────────────────────────────────────────────── */
.conf-bar-wrap { background:rgba(0,0,0,0.08); border-radius:99px; height:5px; margin-top:4px; }
.conf-bar-fill { border-radius:99px; height:5px; }

/* ── Intelligence pills ──────────────────────────────────────────────── */
.intel-pill { display:inline-block; padding:3px 10px; border-radius:99px; font-size:0.7rem; font-weight:600; margin:2px 2px; }
.pill-green  { background:rgba(5,150,105,0.10);   color:#065F46; border:1px solid rgba(5,150,105,0.25); }
.pill-red    { background:rgba(220,38,38,0.10);   color:#991B1B; border:1px solid rgba(220,38,38,0.25); }
.pill-yellow { background:rgba(217,119,6,0.10);   color:#92400E; border:1px solid rgba(217,119,6,0.25); }
.pill-blue   { background:rgba(37,99,235,0.10);   color:#1D4ED8; border:1px solid rgba(37,99,235,0.25); }
.pill-gray   { background:rgba(100,116,139,0.10); color:#475569; border:1px solid rgba(100,116,139,0.22); }

/* ── Grade colors ────────────────────────────────────────────────────── */
.grade-A { color:#B45309; font-weight:800; }
.grade-B { color:#374151; font-weight:700; }
.grade-C { color:#6B7280; font-weight:600; }

/* ── Watchlist chips ─────────────────────────────────────────────────── */
.wl-chip { display:inline-block; background:rgba(37,99,235,0.07); border:1px solid rgba(37,99,235,0.22); border-radius:8px; padding:3px 10px; font-size:0.78rem; color:#1D4ED8; font-weight:600; margin:2px; transition:all .15s; }
.wl-chip:hover { background:rgba(37,99,235,0.14); border-color:rgba(37,99,235,0.38); }

/* ── Misc ────────────────────────────────────────────────────────────── */
div[data-testid="stHorizontalBlock"] > div { padding:0 3px; }
hr { border-color:rgba(37,99,235,0.12) !important; }
.stCaption { color:var(--text-3) !important; }

/* ── Pulse animation ─────────────────────────────────────────────────── */
@keyframes pulse-glow { 0%,100%{opacity:1} 50%{opacity:0.45} }
.live-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#059669; box-shadow:0 0 6px #059669; animation:pulse-glow 2s ease-in-out infinite; vertical-align:middle; margin-right:5px; }

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#F1F5F9; }
::-webkit-scrollbar-thumb { background:rgba(37,99,235,0.25); border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background:rgba(37,99,235,0.45); }

/* ── Streamlit base overrides for dark→light ───────────────────────── */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
  background-color: #F1F5F9 !important;
  color: #0F172A !important;
}
[data-testid="stVerticalBlock"] > div > p,
[data-testid="stVerticalBlock"] > div > span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li { color: #334155 !important; }
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
  background: #fff !important; color: #0F172A !important;
  border-color: rgba(37,99,235,0.25) !important;
}
[data-baseweb="checkbox"] span { color: #0F172A !important; }
[data-testid="stCheckbox"] label span { color: #334155 !important; }
div[data-baseweb="tab-panel"] { background:transparent !important; }
[data-testid="stDataFrame"] { background:#fff !important; }
[data-testid="stExpander"] summary span { color:#334155 !important; }
</style>
"""

_css = _DARK_CSS if st.session_state.get("theme", "dark") == "dark" else _LIGHT_CSS
st.markdown(_css, unsafe_allow_html=True)


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
    {
        "name": "Livemint",
        "cls":  "Mint",
        "url":  "https://www.livemint.com/rss/markets",
    },
    {
        "name": "Hindu Business",
        "cls":  "HBL",
        "url":  "https://www.thehindubusinessline.com/markets/?service=rss",
    },
]

# Sentiment keyword sets for tagging news articles
_BULL_WORDS = {"rally", "surge", "jump", "gain", "high", "record", "beat", "outperform",
               "bull", "boom", "rise", "positive", "upgrade", "profit", "growth", "breakout"}
_BEAR_WORDS = {"fall", "drop", "crash", "loss", "sell-off", "selloff", "bear", "decline",
               "weak", "miss", "downgrade", "recession", "risk", "cut", "worry", "fear",
               "plunge", "tumble", "slump", "disappoints", "warning"}


# ─────────────────────────────────────────────────────────────────────────────
# Cached data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    import __main__
    from swing_v2 import LGBMEnsemble
    __main__.LGBMEnsemble = LGBMEnsemble   # allow joblib unpickling from any context
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
    stock_tiers    = blob.get("stock_tiers", {})   # v5.1 per-stock BUY precision tiers

    # Expand scan universe: Nifty 50 + any Next-50 stocks promoted to Tier A
    tier_a_next50  = [s for s in NIFTY_NEXT_50 if stock_tiers.get(s) == "A"]
    scan_universe  = list(dict.fromkeys(NIFTY_50 + tier_a_next50))

    today     = date.today()
    from_date = today - timedelta(days=420)

    market = get_market_data()
    trend_regime, vol_regime = classify_regime(market)

    symbol_data: dict[str, pd.DataFrame] = {}
    for sym in scan_universe:
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

        # Stock tier + tier-adjusted probability bar
        stock_tier  = stock_tiers.get(sym, "C")
        _tier_bars  = {"A": BUY_PROBA, "B": BUY_PROBA + 0.02, "C": BUY_PROBA + 0.05}
        tier_bar    = _tier_bars[stock_tier]

        confirmations = {
            "EMA200":   (not REQUIRE_EMA200)   or gv("feat_ema200_ratio") > 0,
            "EMA50":    (not REQUIRE_EMA50)    or gv("feat_ema50_ratio")  > 0,
            "EMA20":    (not REQUIRE_EMA20)    or gv("feat_ema20_ratio")  > 0,
            "ADX":      (not REQUIRE_ADX)      or gv("feat_adx")          >= ADX_MIN,
            "MACD":     (not REQUIRE_MACD)     or gv("feat_macd_hist")    >= 0,
            "Breakout": (not REQUIRE_BREAKOUT) or gv("feat_dist_20d_high") > 0,
            "Volume":   (not REQUIRE_VOLUME)   or gv("feat_volume_ratio") >= VOLUME_THRESHOLD,
            # Gate 8: momentum — block when BOTH stock AND Nifty in 3-day decline
            "Momentum": (gv("feat_return_3d") >= -0.015 or
                         gv("feat_nifty_return_3d") >= -0.015),
        }
        n_pass = sum(confirmations.values())
        passed  = [k for k, v in confirmations.items() if v]
        failed  = [k for k, v in confirmations.items() if not v]

        grade = ("A" if n_pass >= GRADE_A_MIN else
                 "B" if n_pass >= GRADE_B_MIN else
                 "C" if n_pass >= GRADE_C_MIN else "D")

        ema200_ok = confirmations["EMA200"]
        tier_ok   = stock_tier in ("A", "B")
        gate_pass = (ema200_ok and sym in ranked_set and not regime_blocked
                     and tier_ok and n_pass >= MIN_CONFIRMATIONS)

        if buy_p >= tier_bar and gate_pass:
            signal = "BUY"
        elif buy_p >= tier_bar:
            signal = "WATCH"
        elif sell_p >= tier_bar:
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
            "Tier":     stock_tier,
            "BUY%":     round(buy_p * 100, 1),
            "Confs":    f"{n_pass}/8",
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
def fetch_news(max_per_feed: int = 4) -> list[dict]:
    """Fetch headlines from RSS feeds with sentiment tagging. Newest-first, max 7 days old."""
    import html as html_mod
    articles: list[dict] = []
    cutoff = datetime.now() - timedelta(days=7)

    _pub_fmts = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S",
        "%a, %d %b %Y %H:%M",
        "%d %b %Y %H:%M:%S %z",
    ]

    def _sentiment(title: str) -> str:
        words = set(title.lower().split())
        bull  = len(words & _BULL_WORDS)
        bear  = len(words & _BEAR_WORDS)
        if bull > bear:  return "bull"
        if bear > bull:  return "bear"
        return "neut"

    for feed in NEWS_FEEDS:
        try:
            resp = requests.get(feed["url"], timeout=7,
                                headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"})
            if resp.status_code != 200:
                continue
            root  = ET.fromstring(resp.content)
            items = root.findall(".//item")
            count = 0
            for item in items:
                if count >= max_per_feed:
                    break
                title = html_mod.unescape((item.findtext("title") or "").strip())
                link  = (item.findtext("link") or "").strip()
                pub   = (item.findtext("pubDate") or "").strip()
                desc_raw = html_mod.unescape((item.findtext("description") or "").strip())
                # strip HTML tags from description
                import re as _re
                desc = _re.sub(r"<[^>]+>", "", desc_raw)[:200]
                if not title or not link:
                    continue

                pub_dt = None
                for fmt in _pub_fmts:
                    try:
                        pub_dt = datetime.strptime(pub[:31].strip(), fmt).replace(tzinfo=None)
                        break
                    except Exception:
                        continue

                if pub_dt and pub_dt < cutoff:
                    continue

                articles.append({
                    "title":     title[:130],
                    "desc":      desc.strip() or "",
                    "link":      link,
                    "source":    feed["name"],
                    "cls":       feed["cls"],
                    "pub":       pub_dt.strftime("%d %b %H:%M") if pub_dt else "recent",
                    "pub_dt":    pub_dt or datetime.min,
                    "sentiment": _sentiment(title),
                })
                count += 1
        except Exception:
            continue

    articles.sort(key=lambda a: a["pub_dt"], reverse=True)
    for a in articles:
        a.pop("pub_dt", None)
    return articles


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty_insights() -> dict:
    """Return historical Nifty 50 stats: YTD, 1Y, 3Y returns, monthly seasonality, yearly table."""
    try:
        today     = date.today()
        from_date = today - timedelta(days=365 * 8)
        nifty_raw = fetch_historical_data("^NSEI", from_date, today)
        if nifty_raw is None or len(nifty_raw) < 100:
            return {}

        nifty = nifty_raw.copy()
        nifty["date"] = pd.to_datetime(nifty["DateTime"])
        nifty = nifty.sort_values("date").set_index("date")
        close = nifty["Close"]

        def _ret(days: int) -> float:
            sub = close[close.index >= str(today - timedelta(days=days))]
            return round((close.iloc[-1] / sub.iloc[0] - 1) * 100, 1) if len(sub) > 1 else 0.0

        ytd_start = close[close.index.year == today.year]
        ytd_ret   = round((close.iloc[-1] / ytd_start.iloc[0] - 1) * 100, 1) if len(ytd_start) > 1 else 0.0

        # Monthly seasonality — avg % return per calendar month across all years
        monthly     = close.resample("ME").last()
        monthly_ret = monthly.pct_change() * 100
        seasonality = (
            monthly_ret.groupby(monthly_ret.index.month).mean().round(2).to_dict()
        )

        # Yearly calendar returns
        yearly     = close.resample("YE").last()
        yearly_ret = yearly.pct_change() * 100
        yearly_data = [
            {"year": int(yr.year), "ret": round(float(ret), 1)}
            for yr, ret in yearly_ret.tail(8).items()
            if not np.isnan(ret)
        ]

        # 52-week high/low positioning
        last_252 = close.tail(252)
        high52   = float(last_252.max())
        low52    = float(last_252.min())
        cur_p    = float(close.iloc[-1])

        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        best_m  = max(seasonality, key=seasonality.get)
        worst_m = min(seasonality, key=seasonality.get)

        return {
            "ytd_ret":    ytd_ret,
            "one_y_ret":  _ret(365),
            "three_y_ret":_ret(365*3),
            "five_y_ret": _ret(365*5),
            "seasonality":seasonality,
            "cur_month":  today.month,
            "high52":     round(high52, 0),
            "low52":      round(low52, 0),
            "pct_from_hi":round((cur_p / high52 - 1) * 100, 1),
            "pct_from_lo":round((cur_p / low52  - 1) * 100, 1),
            "yearly_data":yearly_data,
            "best_month": month_names.get(best_m, "?"),
            "best_ret":   seasonality[best_m],
            "worst_month":month_names.get(worst_m, "?"),
            "worst_ret":  seasonality[worst_m],
            "cur_price":  round(cur_p, 0),
            "month_names":month_names,
        }
    except Exception:
        return {}


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

def _chart_theme() -> dict:
    """Return Plotly layout kwargs for current theme."""
    dark = st.session_state.get("theme", "dark") == "dark"
    if dark:
        return dict(
            template="plotly_dark",
            paper_bgcolor="rgba(11,16,32,0)",
            plot_bgcolor="rgba(17,24,39,0.35)",
            gridcolor="rgba(59,130,246,0.08)",
            linecolor="rgba(59,130,246,0.15)",
            tickcolor="#94a3b8",
            font_color="#94A3B8",
            legend_bg="rgba(17,24,39,0.85)",
            legend_border="rgba(59,130,246,0.20)",
            hover_bg="#111827",
            hover_border="rgba(59,130,246,0.30)",
            hover_font="#F8FAFC",
            title_color="#F8FAFC",
            ann_bgcolor="rgba(255,255,255,0.85)",
        )
    else:
        return dict(
            template="plotly_white",
            paper_bgcolor="rgba(241,245,249,0)",
            plot_bgcolor="rgba(255,255,255,0.80)",
            gridcolor="rgba(100,116,139,0.12)",
            linecolor="rgba(100,116,139,0.20)",
            tickcolor="#475569",
            font_color="#334155",
            legend_bg="rgba(255,255,255,0.92)",
            legend_border="rgba(37,99,235,0.18)",
            hover_bg="#ffffff",
            hover_border="rgba(37,99,235,0.25)",
            hover_font="#0F172A",
            title_color="#0F172A",
            ann_bgcolor="rgba(255,255,255,0.92)",
        )


def make_chart(
    df: pd.DataFrame, symbol: str, lookback: int,
    show_ema9: bool, show_ema20: bool, show_ema50: bool, show_ema200: bool,
    show_bb: bool, show_supertrend: bool, show_pivots: bool, show_sr: bool,
    model, model_features: list[str],
) -> go.Figure:

    ct = _chart_theme()
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
            bgcolor=ct["ann_bgcolor"],
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
        template=ct["template"],
        paper_bgcolor=ct["paper_bgcolor"], plot_bgcolor=ct["plot_bgcolor"],
        margin=dict(l=10, r=110, t=36, b=10),
        height=720,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top", y=0.99,
            xanchor="left", x=1.01,
            font=dict(size=9, color=ct["font_color"]),
            bgcolor=ct["legend_bg"],
            bordercolor=ct["legend_border"], borderwidth=1,
            tracegroupgap=2,
        ),
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color=ct["font_color"]),
        title=dict(
            text=(f"<b style='color:{ct['title_color']}'>{symbol}</b>"
                  f"  <span style='color:{ct['font_color']}'>₹{last_price:,.1f}</span>"
                  f"  <span style='color:{signal_color};font-weight:700'> {signal_label}</span>"
                  f"  <span style='color:{ct['font_color']};font-size:10px'>  BUY% {last_buy_p*100:.0f} · {last_conf}/7 gates</span>"),
            x=0.01, font=dict(size=14),
        ),
        hoverlabel=dict(bgcolor=ct["hover_bg"], bordercolor=ct["hover_border"],
                        font=dict(size=11, color=ct["hover_font"])),
    )
    fig.update_xaxes(
        gridcolor=ct["gridcolor"], showgrid=True, zeroline=False,
        linecolor=ct["linecolor"], tickfont=dict(color=ct["tickcolor"], size=10),
        showticklabels=False,  # hide on all rows except bottom
    )
    fig.update_xaxes(showticklabels=True, tickfont=dict(color=ct["tickcolor"], size=10), row=4, col=1)
    fig.update_yaxes(
        gridcolor=ct["gridcolor"], showgrid=True, zeroline=False,
        linecolor=ct["linecolor"], tickfont=dict(color=ct["tickcolor"], size=10),
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
    _theme_now = st.session_state.get("theme", "dark")
    _hdr_accent = "#3B82F6" if _theme_now == "dark" else "#2563EB"
    _hdr_text   = "#F8FAFC" if _theme_now == "dark" else "#0F172A"
    _hdr_sub    = "#06B6D4" if _theme_now == "dark" else "#0891B2"
    st.markdown(f"""<div style='padding:8px 0 12px;border-bottom:1px solid rgba(59,130,246,0.15);margin-bottom:12px'>
<div style='font-size:0.65rem;font-weight:700;color:{_hdr_accent};letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px'>NSE · AI Terminal</div>
<div style='font-size:1.1rem;font-weight:800;color:{_hdr_text};font-family:Space Grotesk,sans-serif;line-height:1.2'>Swing<span style='color:{_hdr_sub}'> Intelligence</span></div>
</div>""", unsafe_allow_html=True)

    _toggle_label = "☀️ Light Mode" if _theme_now == "dark" else "🌙 Dark Mode"
    if st.button(_toggle_label, use_container_width=True):
        st.session_state["theme"] = "light" if _theme_now == "dark" else "dark"
        st.rerun()

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
    ov1, ov2, ov3, ov4 = st.columns(4)
    show_ema9   = ov1.checkbox("EMA9",  value=True)
    show_ema20  = ov2.checkbox("EMA20", value=True)
    show_ema50  = ov3.checkbox("EMA50", value=True)
    show_ema200 = ov4.checkbox("EMA200",value=True)
    ov5, ov6, ov7, ov8 = st.columns(4)
    show_bb         = ov5.checkbox("BB",    value=False)
    show_supertrend = ov6.checkbox("ST",    value=False)
    show_pivots     = ov7.checkbox("Pivot", value=True)
    show_sr         = ov8.checkbox("S/R",   value=False)

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
    st.markdown("""<div class='footnote-box'>
<b>Setup required:</b> The swing signal model has not been trained yet.<br>
Run <code>python swing_v2.py train</code> in the terminal, then refresh this page.<br>
Training takes ~5–10 minutes on NIFTY 50 + NEXT 50 (93 stocks, 7-year history).
</div>""", unsafe_allow_html=True)
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
    <div style='font-size:1.2rem;font-weight:800;color:#F8FAFC;white-space:nowrap'>NSE Swing Intelligence</div>
    <div style='font-size:0.72rem;color:#64748b'>{date.today().strftime("%A, %d %B %Y")}</div>
  </div>
  <div style='display:flex;gap:20px;flex-wrap:wrap;align-items:center;justify-content:flex-end;flex:1'>
    <div style='text-align:center;min-width:70px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>NIFTY 50</div>
      <div style='font-size:1rem;font-weight:700;color:#F8FAFC;white-space:nowrap'>{_nifty_str} <span style='color:{chg_color};font-size:0.82rem'>{_chg_str}</span></div>
    </div>
    <div style='text-align:center;min-width:90px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>REGIME</div>
      <div style='font-size:0.88rem;font-weight:700;color:#F8FAFC;white-space:nowrap'>{regime_icon} {regime_label}{_regime_asterisk}</div>
    </div>
    <div style='text-align:center;min-width:55px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>VIX</div>
      <div style='font-size:0.88rem;font-weight:700;color:#F8FAFC'>{vix_icon} {_vix_str}</div>
    </div>
    <div style='text-align:center;min-width:60px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>EMA50</div>
      <div style='font-size:0.88rem;font-weight:600;color:#F8FAFC'>{_dma50_str}</div>
    </div>
    <div style='text-align:center;min-width:60px'>
      <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>EMA200</div>
      <div style='font-size:0.88rem;font-weight:600;color:#F8FAFC'>{_dma200_str}</div>
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Portfolio", "Sectors", "Deep Dive", "Intraday"])


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

    # ── Top 3 Actionable Signals (Premium Hub) ───────────────────────
    if not scan_df.empty:
        _buys = scan_df[scan_df["Signal"] == "BUY"].copy()
        # sort: Tier A first, then by BUY% desc; quality filter overbought/thin volume
        _tier_order = {"A": 0, "B": 1, "C": 2}
        _buys["_tier_rank"] = _buys.get("Tier", pd.Series("C", index=_buys.index)).map(_tier_order).fillna(2)
        _buys_sorted   = _buys.sort_values(["_tier_rank", "BUY%"], ascending=[True, False])
        _buys_filtered = _buys_sorted[(_buys_sorted["RSI"] <= 72) & (_buys_sorted["VolRatio"] >= 1.3)]
        top3 = _buys_filtered.head(3) if not _buys_filtered.empty else _buys_sorted.head(3)
        _top3_is_watch = top3.empty
        if _top3_is_watch:
            top3 = scan_df[scan_df["Signal"] == "WATCH"].head(3)

        if not top3.empty:
            _hub_label = "Best Watch Candidates" if _top3_is_watch else "Top BUY Signals Today"
            _hub_sub   = ("<span style='font-size:0.72rem;color:#94a3b8;margin-left:10px'>"
                          "No qualifying BUYs — showing WATCH setups</span>") if _top3_is_watch else ""
            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:14px'>"
                f"<span style='font-size:1.05rem;font-weight:700;color:#F8FAFC'>{_hub_label}</span>"
                f"{_hub_sub}</div>",
                unsafe_allow_html=True
            )
            t_cols = st.columns(len(top3))
            for col, (_, row) in zip(t_cols, top3.iterrows()):
                sig    = row["Signal"]
                grade  = row.get("Grade", "")
                tier   = str(row.get("Tier", "C"))
                conf_w = int(row["BUY%"])
                sector = str(row.get("Sector", "")) or "—"
                confs  = str(row.get("Confs", "0/8"))

                chg_color = "#10B981" if row["1D%"] >= 0 else "#EF4444"

                # gate dots from Confs e.g. "7/8"
                try:
                    n_pass_g, n_total_g = [int(x) for x in confs.split("/")]
                except Exception:
                    n_pass_g, n_total_g = 0, 8
                gate_dots = (
                    "".join(["<span class='gate-dot-pass'></span>"] * n_pass_g) +
                    "".join(["<span class='gate-dot-fail'></span>"] * max(0, n_total_g - n_pass_g))
                )

                # risk pills
                has_stop = pd.notna(row.get("Stop"))
                has_tgt  = pd.notna(row.get("Target"))
                stop_html = f"<span class='stop-pill'>⬇ ₹{row['Stop']:,.0f}</span>" if has_stop else ""
                tgt_html  = f"<span class='tgt-pill'>⬆ ₹{row['Target']:,.0f}</span>" if has_tgt else ""
                rr_html   = ""
                if has_stop and has_tgt:
                    _risk   = row["Price"] - row["Stop"]
                    _reward = row["Target"] - row["Price"]
                    if _risk > 0:
                        rr_html = f"<span class='rr-pill'>R:R {_reward/_risk:.1f}×</span>"

                # warning tags
                warn_html = ""
                if row.get("RSI_warn"):
                    warn_html += ("<span style='font-size:0.65rem;background:rgba(251,191,36,0.12);"
                                  "color:#FCD34D;border-radius:4px;padding:2px 6px;margin-right:4px'>⚠ RSI&gt;70</span>")
                if row.get("ATR_warn"):
                    warn_html += ("<span style='font-size:0.65rem;background:rgba(239,68,68,0.10);"
                                  "color:#FCA5A5;border-radius:4px;padding:2px 6px'>⚠ Wide ATR</span>")

                tier_labels = {"A": "★ TIER A", "B": "◈ TIER B", "C": "· TIER C"}
                tier_label  = tier_labels.get(tier, "TIER C")
                card_cls    = "signal-hub-card-watch" if _top3_is_watch else "signal-hub-card"
                prob_cls    = "prob-fill-amber" if _top3_is_watch else "prob-fill-green"

                col.markdown(f"""<div class='{card_cls}'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px'>
    <div>
      <div style='font-size:1.12rem;font-weight:800;color:#F8FAFC;letter-spacing:.5px'>{row["Symbol"]}</div>
      <div style='font-size:0.63rem;color:#64748b;margin-top:2px;text-transform:uppercase;letter-spacing:.5px'>{sector}</div>
    </div>
    <div style='display:flex;flex-direction:column;align-items:flex-end;gap:5px'>
      <span class='tier-{tier}'>{tier_label}</span>
      <span class='badge badge-{sig}'>{sig}{" " + grade if grade else ""}</span>
    </div>
  </div>
  <div style='display:flex;align-items:baseline;gap:8px;margin-bottom:10px'>
    <span style='font-size:1.18rem;font-weight:700;color:#F8FAFC'>₹{row["Price"]:,.1f}</span>
    <span style='font-size:0.82rem;font-weight:600;color:{chg_color}'>{row["1D%"]:+.2f}%</span>
  </div>
  <div style='margin-bottom:10px'>
    <div style='display:flex;justify-content:space-between;font-size:0.65rem;color:#64748b;margin-bottom:3px'>
      <span style='letter-spacing:.5px'>BUY PROBABILITY</span>
      <span style='color:#10B981;font-weight:700;font-size:0.72rem'>{conf_w}%</span>
    </div>
    <div class='prob-track'><div class='{prob_cls}' style='width:{conf_w}%'></div></div>
  </div>
  <div style='margin-bottom:10px'>
    <div style='font-size:0.63rem;color:#475569;letter-spacing:.5px;margin-bottom:5px'>GATES PASSED &nbsp;<b style='color:#CBD5E1'>{confs}</b></div>
    <div style='display:flex;gap:5px'>{gate_dots}</div>
  </div>
  <div style='display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px'>{stop_html}{tgt_html}{rr_html}</div>
  <div style='font-size:0.68rem;color:#475569'>
    RSI <b style='color:#CBD5E1'>{row["RSI"]:.0f}</b>&nbsp;·&nbsp;ADX <b style='color:#CBD5E1'>{row["ADX"]:.0f}</b>&nbsp;·&nbsp;Vol× <b style='color:#CBD5E1'>{row["VolRatio"]:.1f}</b>
  </div>
  {f"<div style='margin-top:6px'>{warn_html}</div>" if warn_html else ""}
</div>""", unsafe_allow_html=True)

    # ── Signal count summary row (full width) ────────────────────────
    if not scan_df.empty:
        sig_counts = scan_df["Signal"].value_counts()
        _sig_colors = {"BUY":"#10B981","WATCH":"#F59E0B","HOLD":"#64748b","SELL":"#EF4444"}
        _sig_bg     = {"BUY":"rgba(16,185,129,0.10)","WATCH":"rgba(245,158,11,0.10)",
                       "HOLD":"rgba(100,116,139,0.10)","SELL":"rgba(239,68,68,0.10)"}
        _sig_border = {"BUY":"rgba(16,185,129,0.30)","WATCH":"rgba(245,158,11,0.30)",
                       "HOLD":"rgba(100,116,139,0.20)","SELL":"rgba(239,68,68,0.30)"}
        _cnt_cols = st.columns([1, 1, 1, 1, 6])
        for col_m, sig in zip(_cnt_cols[:4], ["BUY","WATCH","HOLD","SELL"]):
            n = sig_counts.get(sig, 0)
            col_m.markdown(
                f"<div style='background:{_sig_bg[sig]};border:1px solid {_sig_border[sig]};"
                f"border-radius:10px;padding:10px 6px;text-align:center'>"
                f"<div style='font-size:1.5rem;font-weight:800;color:{_sig_colors[sig]}'>{n}</div>"
                f"<div style='font-size:0.68rem;font-weight:700;color:{_sig_colors[sig]};letter-spacing:.5px'>{sig}</div>"
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
<div style='background:rgba(17,24,39,0.8);border:1px solid rgba(59,130,246,0.15);border-top:3px solid {col_m};
     border-radius:8px;padding:7px 10px;text-align:center;cursor:pointer'>
  <div style='font-weight:700;font-size:0.82rem;color:#F8FAFC'>{sym}</div>
  <div style='font-size:0.78rem;color:#CBD5E1'>₹{price_m:,.0f}</div>
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

        # Signal card with explanation (premium styled)
        if not scan_df.empty:
            sr_row = scan_df[scan_df["Symbol"] == selected_symbol]
            if not sr_row.empty:
                sr     = sr_row.iloc[0]
                sig    = sr["Signal"]
                color  = SIGNAL_COLORS.get(sig, "#64748b")
                grade  = sr.get("Grade","")
                tier   = str(sr.get("Tier", "C"))
                expl   = explain_signal(sig, sr.get("Passed",""), sr.get("Blockers",""), sr["BUY%"], grade)
                confs  = str(sr.get("Confs", "0/8"))

                # gate dots
                try:
                    _np, _nt = [int(x) for x in confs.split("/")]
                except Exception:
                    _np, _nt = 0, 8
                _gate_dots = (
                    "".join(["<span class='gate-dot-pass'></span>"] * _np) +
                    "".join(["<span class='gate-dot-fail'></span>"] * max(0, _nt - _np))
                )

                # stop / tgt / R:R pills
                _has_stop = pd.notna(sr.get("Stop"))
                _has_tgt  = pd.notna(sr.get("Target"))
                _stop_p   = f"<span class='stop-pill'>⬇ ₹{sr['Stop']:,.0f}</span>" if _has_stop else ""
                _tgt_p    = f"<span class='tgt-pill'>⬆ ₹{sr['Target']:,.0f}</span>" if _has_tgt else ""
                _rr_p     = ""
                if _has_stop and _has_tgt:
                    _ri, _re = sr["Price"] - sr["Stop"], sr["Target"] - sr["Price"]
                    if _ri > 0:
                        _rr_p = f"<span class='rr-pill'>R:R {_re/_ri:.1f}×</span>"

                # risk warnings
                risk_tags = ""
                if sr.get("RSI_warn"):
                    risk_tags += ("<span style='font-size:0.7rem;background:rgba(251,191,36,0.12);"
                                  "color:#FCD34D;border-radius:4px;padding:2px 8px;margin-right:4px'>⚠ RSI &gt;70 — overbought entry</span>")
                if sr.get("ATR_warn"):
                    risk_tags += ("<span style='font-size:0.7rem;background:rgba(239,68,68,0.10);"
                                  "color:#FCA5A5;border-radius:4px;padding:2px 8px;margin-right:4px'>⚠ High ATR — stop wider than normal</span>")
                if sr.get("WeakBUY"):
                    risk_tags += ("<span style='font-size:0.7rem;background:rgba(59,130,246,0.08);"
                                  "color:#94A3B8;border-radius:4px;padding:2px 8px'>Grade C + RSI&gt;65 + low volume — HOLD acceptable</span>")
                sideways_note = ""
                if trend_regime == "SIDEWAYS" and sig == "BUY":
                    sideways_note = ("<div style='font-size:0.7rem;background:rgba(245,158,11,0.08);"
                                     "color:#FCD34D;border-radius:6px;padding:4px 10px;margin-top:6px'>"
                                     "⚠ Sideways regime — lower signal reliability, tighten stop</div>")

                tier_labels = {"A": "★ TIER A", "B": "◈ TIER B", "C": "· TIER C"}
                tier_label  = tier_labels.get(tier, "TIER C")
                _card_cls   = "signal-hub-card" if sig == "BUY" else ("signal-hub-card-watch" if sig == "WATCH" else "card")
                st.markdown(f"""
<div class='{_card_cls}' style='margin-bottom:10px'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px'>
    <div style='display:flex;align-items:center;gap:8px'>
      <span class='badge badge-{sig}'>{sig}{" " + grade if grade else ""}</span>
      <span class='tier-{tier}'>{tier_label}</span>
    </div>
    <span style='font-size:0.78rem;color:#64748b'>Rank <b style='color:#CBD5E1'>#{sr["Rank"]}</b></span>
  </div>
  <div style='margin-bottom:10px'>
    <div style='display:flex;justify-content:space-between;font-size:0.65rem;color:#64748b;margin-bottom:3px'>
      <span style='letter-spacing:.5px'>BUY PROBABILITY</span>
      <span style='color:{color};font-weight:700'>{sr["BUY%"]:.1f}%</span>
    </div>
    <div class='prob-track'><div class='prob-fill-{"green" if sig=="BUY" else "amber"}' style='width:{min(int(sr["BUY%"]),100)}%'></div></div>
  </div>
  <div style='margin-bottom:10px'>
    <div style='font-size:0.63rem;color:#475569;letter-spacing:.5px;margin-bottom:5px'>GATES PASSED &nbsp;<b style='color:#CBD5E1'>{confs}</b></div>
    <div style='display:flex;gap:5px'>{_gate_dots}</div>
  </div>
  <div style='display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px'>{_stop_p}{_tgt_p}{_rr_p}</div>
  <div style='font-size:0.78rem;color:#94A3B8;margin-bottom:6px'>{expl}</div>
  {f"<div style='display:flex;gap:5px;flex-wrap:wrap;margin-top:4px'>{risk_tags}</div>" if risk_tags else ""}
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
                    elif col == "Tier":
                        fg = {"A":"#FCD34D","B":"#94A3B8","C":"#475569"}.get(row.get("Tier",""),"#475569")
                        styles.append(f"color:{fg};font-weight:700")
                    else:
                        styles.append("color:#CBD5E1")
                return styles

            show_cols = ["Symbol","Price","1D%","Signal","Tier","Grade","BUY%","Confs","RSI"]
            fmt = {"Price": "₹{:,.1f}", "1D%": "{:+.2f}%", "BUY%": "{:.1f}"}
            _scan_disp = filtered_df.copy() if not filtered_df.empty else pd.DataFrame(columns=show_cols)
            if "intraday_score" in _scan_disp.columns:
                _scan_disp = _scan_disp.rename(columns={"intraday_score": "Intra"})
                show_cols = show_cols + ["Intra"]
                fmt["Intra"] = "{:+.2f}"
            disp = _scan_disp[show_cols] if not _scan_disp.empty else pd.DataFrame(columns=show_cols)

            def _style_row_ext(row):
                styles = _style_row(row)
                if "Intra" in row.index:
                    v = row.get("Intra", 0)
                    styles[list(row.index).index("Intra")] = (
                        "color:#16a34a;font-weight:700" if v > 0
                        else ("color:#dc2626;font-weight:700" if v < 0 else "color:#94a3b8")
                    )
                return styles

            styled = (
                disp.style
                .apply(_style_row_ext, axis=1)
                .format(fmt)
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

    # ── Market Historical Insights ─────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 Historical Market Insights — Nifty 50", expanded=False):
        with st.spinner("Computing historical stats…"):
            _ins = get_nifty_insights()

        if not _ins:
            st.info("Historical data unavailable — market data fetch failed.")
        else:
            _mnames = _ins["month_names"]
            _seas   = _ins["seasonality"]
            _cur_m  = _ins["cur_month"]

            # ── Row 1: Return metrics ──
            _ic1, _ic2, _ic3, _ic4, _ic5 = st.columns(5)
            for _col, _label, _val in [
                (_ic1, "YTD Return",   _ins["ytd_ret"]),
                (_ic2, "1-Year",       _ins["one_y_ret"]),
                (_ic3, "3-Year",       _ins["three_y_ret"]),
                (_ic4, "5-Year",       _ins["five_y_ret"]),
                (_ic5, "52W from High",_ins["pct_from_hi"]),
            ]:
                _c = "#10B981" if _val >= 0 else "#EF4444"
                _sign = "+" if _val >= 0 else ""
                _col.markdown(
                    f"<div class='insight-card'>"
                    f"<div style='font-size:0.62rem;color:#64748b;letter-spacing:.5px;text-transform:uppercase'>{_label}</div>"
                    f"<div style='font-size:1.3rem;font-weight:800;color:{_c};margin-top:4px'>{_sign}{_val:.1f}%</div>"
                    f"</div>", unsafe_allow_html=True
                )

            st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

            # ── Row 2: Seasonality bar chart ──
            _seas_l, _seas_r = st.columns([2, 1])
            with _seas_l:
                _seas_months = [_mnames.get(m, str(m)) for m in range(1, 13)]
                _seas_vals   = [_seas.get(m, 0) for m in range(1, 13)]
                _seas_colors = ["rgba(16,185,129,0.7)" if v >= 0 else "rgba(239,68,68,0.65)"
                                for v in _seas_vals]
                _seas_fig = go.Figure(go.Bar(
                    x=_seas_months, y=_seas_vals,
                    marker_color=_seas_colors,
                    text=[f"{v:+.1f}%" for v in _seas_vals],
                    textposition="outside",
                    textfont=dict(size=9),
                ))
                _ct2 = _chart_theme()
                _seas_fig.update_layout(
                    template=_ct2["template"],
                    paper_bgcolor=_ct2["paper_bgcolor"],
                    plot_bgcolor=_ct2["plot_bgcolor"],
                    height=220,
                    margin=dict(l=10, r=10, t=28, b=10),
                    title=dict(text="Monthly Seasonality (avg % return, 8yr)", font=dict(color=_ct2["title_color"], size=11)),
                    yaxis=dict(tickformat="+.1f", gridcolor=_ct2["gridcolor"], tickfont=dict(size=9, color=_ct2["font_color"])),
                    xaxis=dict(tickfont=dict(size=9, color=_ct2["font_color"])),
                    bargap=0.25,
                    shapes=[{
                        "type": "rect", "xref": "x", "yref": "paper",
                        "x0": _mnames.get(_cur_m, "") + " -0.5" if False else str(_seas_months[_cur_m-1]),
                        "x1": _seas_months[_cur_m-1],
                        "y0": 0, "y1": 1,
                        "fillcolor": "rgba(59,130,246,0.10)",
                        "line": {"color": "rgba(59,130,246,0.40)", "width": 1},
                    }] if _cur_m in _seas else [],
                )
                st.plotly_chart(_seas_fig, use_container_width=True)

            with _seas_r:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                # Current month seasonality
                _cur_sea = _seas.get(_cur_m, 0)
                _cur_sea_c = "#10B981" if _cur_sea >= 0 else "#EF4444"
                st.markdown(f"""<div class='insight-card' style='margin-bottom:8px'>
<div style='font-size:0.62rem;color:#64748b;letter-spacing:.5px'>THIS MONTH ({_mnames.get(_cur_m,"")})</div>
<div style='font-size:1.1rem;font-weight:800;color:{_cur_sea_c};margin-top:4px'>Avg {_cur_sea:+.1f}%</div>
<div style='font-size:0.68rem;color:#64748b;margin-top:2px'>historical avg return</div>
</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='insight-card' style='margin-bottom:8px'>
<div style='font-size:0.62rem;color:#64748b;letter-spacing:.5px'>BEST MONTH (hist.)</div>
<div style='font-size:1.1rem;font-weight:800;color:#10B981;margin-top:4px'>{_ins["best_month"]} +{_ins["best_ret"]:.1f}%</div>
</div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='insight-card'>
<div style='font-size:0.62rem;color:#64748b;letter-spacing:.5px'>WORST MONTH (hist.)</div>
<div style='font-size:1.1rem;font-weight:800;color:#EF4444;margin-top:4px'>{_ins["worst_month"]} {_ins["worst_ret"]:+.1f}%</div>
</div>""", unsafe_allow_html=True)

            # ── Row 3: Yearly returns ──
            if _ins["yearly_data"]:
                st.markdown("<div style='margin-top:8px;font-size:0.75rem;color:#64748b;font-weight:600;letter-spacing:.4px'>NIFTY 50 YEARLY RETURNS</div>", unsafe_allow_html=True)
                _yr_cols = st.columns(len(_ins["yearly_data"]))
                for _ycol, _yr in zip(_yr_cols, _ins["yearly_data"]):
                    _yc  = "#10B981" if _yr["ret"] >= 0 else "#EF4444"
                    _sign = "+" if _yr["ret"] >= 0 else ""
                    _ycol.markdown(
                        f"<div style='text-align:center;padding:6px 4px;border-radius:8px;"
                        f"background:rgba(59,130,246,0.05);border:1px solid rgba(59,130,246,0.12)'>"
                        f"<div style='font-size:0.6rem;color:#64748b'>{_yr['year']}</div>"
                        f"<div style='font-size:0.88rem;font-weight:700;color:{_yc}'>{_sign}{_yr['ret']}%</div>"
                        f"</div>", unsafe_allow_html=True
                    )

    # ── News panel (enhanced) ────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>"
        "<span style='font-size:1.0rem;font-weight:700;color:#F8FAFC'>Market News</span>"
        "<span style='font-size:0.68rem;color:#64748b'>· live from ET, MC, BS, Reuters, Livemint</span>"
        "</div>",
        unsafe_allow_html=True
    )
    with st.spinner("Fetching latest headlines…"):
        articles = fetch_news(max_per_feed=4)

    if articles:
        _bull_news = [a for a in articles if a["sentiment"] == "bull"]
        _bear_news = [a for a in articles if a["sentiment"] == "bear"]
        _neut_news = [a for a in articles if a["sentiment"] == "neut"]

        # Sentiment tally pill
        _s_html = (
            f"<span class='sent-bull'>▲ {len(_bull_news)} bullish</span>"
            f"<span class='sent-bear' style='margin-left:6px'>▼ {len(_bear_news)} bearish</span>"
            f"<span class='sent-neut' style='margin-left:6px'>— {len(_neut_news)} neutral</span>"
        )
        st.markdown(f"<div style='margin-bottom:10px'>{_s_html}</div>", unsafe_allow_html=True)

        n_cols = st.columns(2)
        mid    = max(1, len(articles) // 2)
        for _col_n, _chunk in zip(n_cols, [articles[:mid], articles[mid:]]):
            if not _chunk:
                continue
            with _col_n:
                _news_html = ""
                for _a in _chunk:
                    _sent_class = f"sent-{_a['sentiment']}"
                    _sent_label = {"bull": "▲ Bullish", "bear": "▼ Bearish", "neut": "Neutral"}.get(_a["sentiment"], "")
                    _news_html += f"""
<div class='news-card-prem'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px'>
    <span class='news-source src-{_a["cls"]}'>{_a["source"]}</span>
    <div style='display:flex;align-items:center;gap:6px'>
      <span class='{_sent_class}'>{_sent_label}</span>
      <span style='font-size:0.65rem;color:#475569'>{_a["pub"]}</span>
    </div>
  </div>
  <a href='{_a["link"]}' target='_blank'
     style='color:#F8FAFC;font-size:0.82rem;font-weight:600;text-decoration:none;line-height:1.4;display:block'>
    {_a["title"]}
  </a>
  {f"<div style='font-size:0.72rem;color:#64748b;margin-top:4px;line-height:1.4'>{_a['desc'][:120]}…</div>" if _a.get("desc") else ""}
</div>"""
                st.markdown(_news_html, unsafe_allow_html=True)
    else:
        st.info("No news from the last 7 days — feeds may be temporarily unavailable. Refresh to retry.")

    # ── Footnotes / Disclaimers ──────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 Model Notes, Gate Legend & Risk Disclaimer", expanded=False):
        st.markdown("""<div class='footnote-box'>
<h5>⚠️ Important Disclaimer</h5>
This tool is for <b>educational purposes only</b>. It does <b>not</b> constitute financial advice, investment
recommendation, or solicitation to buy or sell securities. Past model performance on historical data
does not guarantee future results. Markets can and do behave in ways that no model can predict.
Always conduct your own due diligence before placing any trade.

<h5>🔢 Signal Grades Explained</h5>
<b>Grade A</b> = 8/8 gates passed (highest conviction) &nbsp;·&nbsp;
<b>Grade B</b> = 7/8 gates &nbsp;·&nbsp;
<b>Grade C</b> = 6/8 gates &nbsp;·&nbsp;
<b>Grade D</b> = fewer than 6 — model score shown for monitoring only.

<h5>⭐ Tier System (OOF BUY Precision)</h5>
<b>Tier A ★</b> = stock historically had ≥65% BUY precision at the signal threshold (gold badge) —
signals from these stocks use base probability threshold. &nbsp;·&nbsp;
<b>Tier B ◈</b> = 55–65% precision — threshold raised +2% as cushion. &nbsp;·&nbsp;
<b>Tier C ·</b> = &lt;55% precision or insufficient samples — threshold raised +5%, WATCH only unless
all 8 gates pass. Tier C stocks <b>rarely generate BUY signals</b>.

<h5>🚪 The 8 Gates</h5>
<b>EMA200</b> — price above 200-day EMA (primary trend filter) &nbsp;·&nbsp;
<b>EMA50</b> — price above 50-day EMA (medium trend) &nbsp;·&nbsp;
<b>EMA20</b> — price above 20-day EMA (short trend) &nbsp;·&nbsp;
<b>ADX</b> — ADX ≥ 20 (trend strength) &nbsp;·&nbsp;
<b>MACD</b> — MACD histogram ≥ 0 (momentum direction) &nbsp;·&nbsp;
<b>Breakout</b> — within 3% of 20-day high (price structure) &nbsp;·&nbsp;
<b>Volume</b> — volume ≥ 1.2× 20-day average (institutional participation) &nbsp;·&nbsp;
<b>Momentum</b> — blocks BUY when both stock AND Nifty are in simultaneous 3-day decline (bear tape filter).

<h5>📐 R:R (Risk-to-Reward)</h5>
Stop = Entry − 1.5 × ATR(14) &nbsp;·&nbsp; Target = Entry + 2.5 × ATR(14) &nbsp;·&nbsp; Natural R:R ≈ 1:1.67.
Actual realized R:R depends on execution price and exit discipline.

<h5>⚡ Known Failure Modes</h5>
<b>1. Gap-down opens</b>: Model scores are computed on prior-day close. If the stock gaps down at open,
the stop may already be hit before entry — check pre-market levels. &nbsp;·&nbsp;
<b>2. Earnings surprises</b>: The model has no earnings calendar awareness. Avoid entering positions
within 5 trading days of scheduled results. &nbsp;·&nbsp;
<b>3. Broad market shocks</b> (macro events, geopolitical): The momentum gate partially mitigates this
but cannot protect against sudden systemic moves. Reduce position sizes during HIGH VIX. &nbsp;·&nbsp;
<b>4. Data staleness</b>: Signals are computed on last available EOD price. Intraday price action is
not reflected until after market close. &nbsp;·&nbsp;
<b>5. March–April seasonality</b>: Historically elevated false-positive rate (end-of-FY institutional
rebalancing creates misleading volume spikes). Cross-check with broader market context. &nbsp;·&nbsp;
<b>6. Low-liquidity stocks</b>: Even within Nifty 50, stocks with ATR/Price &gt; 2.5% have wider spreads
— the ATR_warn flag is triggered for these.

<h5>🔄 Refresh Logic</h5>
Signal scan: cached 15 min &nbsp;·&nbsp; Market regime: cached 30 min &nbsp;·&nbsp;
News: cached 5 min &nbsp;·&nbsp; Historical insights: cached 1 hr.
Use the sidebar "🔄 Refresh Data" button to force-clear all caches.
</div>""", unsafe_allow_html=True)


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
    <div><span style='font-size:0.7rem;color:#64748b'>CAPITAL</span><br><b style='color:#F8FAFC'>₹{capital:,.0f}</b></div>
    <div><span style='font-size:0.7rem;color:#64748b'>DEPLOYED</span><br><b style='color:#F8FAFC'>₹{total_invested:,.0f} ({deployed_pct:.1f}%)</b></div>
    <div><span style='font-size:0.7rem;color:#64748b'>CASH RESERVE</span><br><b style='color:#F8FAFC'>₹{capital-total_invested:,.0f}</b></div>
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
                _ct = _chart_theme()
                pie_fig.update_layout(
                    template=_ct["template"], paper_bgcolor=_ct["paper_bgcolor"],
                    height=320, margin=dict(l=10, r=10, t=30, b=10),
                    title=dict(text="Portfolio Allocation", font=dict(color=_ct["title_color"], size=13)),
                    showlegend=False,
                )
                st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("""
<div class='card' style='margin-top:12px'>
  <b style='color:#F8FAFC'>Exit rules:</b>
  <span style='color:#CBD5E1;font-size:0.85rem'>
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
                        styles.append("color:#15803d;font-weight:700" if row["BUY"] > 0 else "color:#CBD5E1")
                    elif col == "SELL":
                        styles.append("color:#b91c1c;font-weight:700" if row["SELL"] > 0 else "color:#CBD5E1")
                    elif col == "BUY_rate%":
                        pct = row["BUY_rate%"]
                        styles.append(f"color:{'#15803d' if pct>30 else '#a16207' if pct>0 else '#64748b'};font-weight:700")
                    elif col == "Avg_1D":
                        styles.append(f"color:{'#16a34a' if row['Avg_1D']>=0 else '#dc2626'}")
                    else:
                        styles.append("color:#CBD5E1")
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
            _ct = _chart_theme()
            bar_fig.update_layout(
                template=_ct["template"], paper_bgcolor=_ct["paper_bgcolor"], plot_bgcolor=_ct["plot_bgcolor"],
                height=420, barmode="stack",
                margin=dict(l=10, r=60, t=20, b=10),
                legend=dict(orientation="h", y=1.05, font=dict(color=_ct["font_color"])),
                xaxis_title="Number of stocks",
                font=dict(color=_ct["font_color"]),
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
    # ── Deep Dive header row ──────────────────────────────────────────
    dd_h1, dd_h2 = st.columns([2, 1])
    with dd_h1:
        st.markdown("### Deep Dive Analysis")
        st.caption("Full technical breakdown · chart · risk calculator · model diagnostics")
    with dd_h2:
        dd_symbol = st.selectbox("Stock", NIFTY_50,
                                  index=NIFTY_50.index(selected_symbol) if selected_symbol in NIFTY_50 else 0,
                                  key="dd_sym", label_visibility="collapsed")

    with st.spinner(f"Loading {dd_symbol}…"):
        dd_df = get_stock_data(dd_symbol, 365 + 200)

    if dd_df is None or dd_df.empty:
        st.error(f"No data for {dd_symbol}.")
    else:
        dd_last  = dd_df.iloc[-1]
        dd_price = float(dd_last["Close"])
        dd_atr   = float(dd_last.get("ATR", 0) or 0)
        dd_rsi   = float(dd_last.get("feat_rsi", 50) or 50)
        dd_adx   = float(dd_last.get("feat_adx", 0) or 0) * 100
        dd_volr  = float(dd_last.get("feat_volume_ratio", 1) or 1)
        dd_prev  = float(dd_df.iloc[-2]["Close"]) if len(dd_df) >= 2 else dd_price
        dd_chg   = (dd_price / dd_prev - 1) * 100
        high52   = float(dd_df["High"].tail(252).max())
        low52    = float(dd_df["Low"].tail(252).min())
        pct_hi   = (dd_price / high52 - 1) * 100
        pct_lo   = (dd_price / low52  - 1) * 100
        pos52_pct = int((dd_price - low52) / (high52 - low52) * 100) if high52 > low52 else 50

        def gv_dd(col, fb=0.0):
            return float(dd_last.get(col, fb) or fb)

        # ── Row 1: price hero + intel pills ──────────────────────────
        hero_col, intel_col = st.columns([1, 2])

        with hero_col:
            chg_col = "#16a34a" if dd_chg >= 0 else "#dc2626"
            dd_scan_row = scan_df[scan_df["Symbol"] == dd_symbol] if not scan_df.empty else pd.DataFrame()
            dd_sig    = dd_scan_row.iloc[0]["Signal"] if not dd_scan_row.empty else "HOLD"
            dd_grade  = dd_scan_row.iloc[0].get("Grade","") if not dd_scan_row.empty else ""
            dd_buy_p  = dd_scan_row.iloc[0]["BUY%"] if not dd_scan_row.empty else 0
            dd_confs  = dd_scan_row.iloc[0]["Confs"] if not dd_scan_row.empty else 0
            sig_col   = SIGNAL_COLORS.get(dd_sig, "#64748b")
            sig_bg    = SIGNAL_BG.get(dd_sig, "#f1f5f9")
            st.markdown(f"""
<div class='card' style='padding:16px 20px'>
  <div style='font-size:0.72rem;color:#94a3b8;font-weight:600;letter-spacing:.5px'>{dd_symbol} · NSE</div>
  <div style='font-size:2rem;font-weight:800;color:#F8FAFC;line-height:1.1'>₹{dd_price:,.1f}</div>
  <div style='font-size:1rem;font-weight:600;color:{chg_col};margin-bottom:8px'>{dd_chg:+.2f}% today</div>
  <div style='display:flex;gap:8px;align-items:center;margin-bottom:10px'>
    <span class='badge badge-{dd_sig}' style='font-size:0.82rem;padding:4px 10px'>{dd_sig}{' ' + dd_grade if dd_grade else ''}</span>
    <span style='font-size:0.78rem;color:#64748b'>BUY% {dd_buy_p:.0f} · {dd_confs}/7 gates</span>
  </div>
  <div style='font-size:0.72rem;color:#64748b;line-height:1.8'>
    <span style='color:#94a3b8'>ATR</span> ₹{dd_atr:,.1f} &nbsp;·&nbsp;
    <span style='color:#94a3b8'>RSI</span> {dd_rsi:.1f} &nbsp;·&nbsp;
    <span style='color:#94a3b8'>ADX</span> {dd_adx:.1f} &nbsp;·&nbsp;
    <span style='color:#94a3b8'>Vol×</span> {dd_volr:.2f}
  </div>
  <div style='margin-top:10px'>
    <div style='font-size:0.65rem;color:#94a3b8;margin-bottom:3px;font-weight:600'>52-WEEK POSITION — {pos52_pct}%</div>
    <div style='background:rgba(59,130,246,0.12);border-radius:4px;height:6px;position:relative'>
      <div style='background:{"#16a34a" if pos52_pct>60 else ("#ca8a04" if pos52_pct>30 else "#dc2626")};
           height:6px;border-radius:4px;width:{pos52_pct}%'></div>
    </div>
    <div style='display:flex;justify-content:space-between;font-size:0.63rem;color:#94a3b8;margin-top:2px'>
      <span>₹{low52:,.0f}</span><span>₹{high52:,.0f}</span>
    </div>
    <div style='font-size:0.65rem;color:#64748b;margin-top:2px'>
      {pct_hi:.1f}% from 52W high &nbsp;·&nbsp; +{pct_lo:.1f}% from 52W low
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        with intel_col:
            intel = classify_stock_state(dd_last)
            i_html = ""
            for label, (text, cls) in [("Trend", intel["trend"]), ("Momentum", intel["momentum"]),
                                         ("Volatility", intel["volatility"]), ("State", intel["state"])]:
                i_html += (f"<div style='flex:1;text-align:center;padding:8px'>"
                           f"<div style='font-size:0.6rem;color:#94a3b8;font-weight:700;letter-spacing:.5px;margin-bottom:4px'>{label.upper()}</div>"
                           f"<span class='intel-pill {cls}' style='font-size:0.8rem;padding:5px 12px'>{text}</span></div>")

            # Return profile grid
            ret_rows = ""
            for label, days in [("1D",1),("1W",5),("1M",22),("3M",66),("6M",132),("1Y",252)]:
                if len(dd_df) > days:
                    chg = (dd_price / float(dd_df.iloc[-(days+1)]["Close"]) - 1) * 100
                    rc_col = "#16a34a" if chg >= 0 else "#dc2626"
                    ret_rows += (f"<div style='text-align:center;padding:6px 4px'>"
                                 f"<div style='font-size:0.65rem;color:#94a3b8;font-weight:600'>{label}</div>"
                                 f"<div style='font-size:0.88rem;font-weight:700;color:{rc_col}'>{chg:+.2f}%</div>"
                                 f"</div>")

            st.markdown(f"""
<div class='card' style='padding:14px 16px;margin-bottom:8px'>
  <div style='font-size:0.7rem;color:#64748b;font-weight:700;margin-bottom:8px;letter-spacing:.3px'>MARKET STATE</div>
  <div style='display:flex;gap:0'>{i_html}</div>
</div>
<div class='card' style='padding:14px 16px'>
  <div style='font-size:0.7rem;color:#64748b;font-weight:700;margin-bottom:8px;letter-spacing:.3px'>RETURN PROFILE</div>
  <div style='display:flex;justify-content:space-between'>{ret_rows}</div>
</div>""", unsafe_allow_html=True)

        # ── Row 2: 7-Gate confirmation ────────────────────────────────
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#64748b;letter-spacing:.4px;margin:12px 0 8px'>7-GATE CONFIRMATION</div>", unsafe_allow_html=True)
        gate_defs = [
            ("EMA 200", "Price above 200d MA — long-term uptrend",    gv_dd("feat_ema200_ratio") > 0),
            ("EMA 50",  "Price above 50d MA — medium-term trend",     gv_dd("feat_ema50_ratio")  > 0),
            ("EMA 20",  "Price above 20d MA — short-term momentum",   gv_dd("feat_ema20_ratio")  > 0),
            (f"ADX {ADX_MIN*100:.0f}",  f"ADX ≥ {ADX_MIN*100:.0f} — trend has strength",      gv_dd("feat_adx") >= ADX_MIN),
            ("MACD",    "MACD histogram ≥ 0 — momentum rising",       gv_dd("feat_macd_hist") >= 0),
            ("Breakout","Near 20d high — price breaking resistance",   gv_dd("feat_dist_20d_high") > 0),
            ("Volume",  f"Vol ≥ {VOLUME_THRESHOLD}× avg — confirmed participation", gv_dd("feat_volume_ratio") >= VOLUME_THRESHOLD),
        ]
        n_pass = sum(1 for _, _, p in gate_defs if p)
        g_cols = st.columns(7)
        for i, (name, tooltip, passed) in enumerate(gate_defs):
            icon      = "✓" if passed else "✗"
            bg        = "#dcfce7" if passed else "#fee2e2"
            color     = "#15803d" if passed else "#b91c1c"
            border_c  = "#86efac" if passed else "#fca5a5"
            g_cols[i].markdown(
                f"<div title='{tooltip}' style='background:{bg};border:1px solid {border_c};"
                f"border-radius:10px;text-align:center;padding:10px 4px;cursor:help'>"
                f"<div style='font-size:1.2rem;font-weight:800;color:{color}'>{icon}</div>"
                f"<div style='font-size:0.68rem;color:{color};font-weight:700;margin-top:2px'>{name}</div>"
                f"</div>", unsafe_allow_html=True
            )
        gate_summary = f"{n_pass}/7 gates passed"
        gate_qual    = "Grade A" if n_pass >= 7 else ("Grade B" if n_pass >= 6 else ("Grade C" if n_pass >= 5 else "Insufficient"))
        gq_col       = "#16a34a" if n_pass >= 6 else ("#ca8a04" if n_pass >= 5 else "#dc2626")
        st.markdown(f"<div style='font-size:0.72rem;color:{gq_col};font-weight:600;margin-top:4px'>{gate_summary} — {gate_qual}</div>", unsafe_allow_html=True)

        # ── Row 3: Chart (full width) ─────────────────────────────────
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#64748b;letter-spacing:.4px;margin:16px 0 8px'>PRICE CHART</div>", unsafe_allow_html=True)
        dd_fig = make_chart(
            dd_df, dd_symbol, lookback,
            True, True, True, True,   # EMA9, EMA20, EMA50, EMA200 all on
            show_bb, True, True, True, # BB from sidebar, Supertrend on, Pivots on, S/R on
            model, model_features,
        )
        st.plotly_chart(dd_fig, use_container_width=True)

        # ── Row 4: Rolling probability + Trade calculator ─────────────
        prob_col, calc_col = st.columns([3, 2], gap="large")

        with prob_col:
            st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#64748b;letter-spacing:.4px;margin-bottom:8px'>ROLLING BUY PROBABILITY — 90 DAYS</div>", unsafe_allow_html=True)
            dd_trim = dd_df.tail(90 + 50).copy()
            feat_matrix = np.column_stack(
                [dd_trim.get(col, pd.Series(0, index=dd_trim.index)).fillna(0).values for col in model_features]
            )
            probs = model.predict_proba(feat_matrix)[:, 1]
            dd_trim["prob"] = probs
            dd_trim = dd_trim.tail(90)
            dd_trim["DateTime"] = pd.to_datetime(dd_trim["DateTime"])
            cur_prob = float(dd_trim["prob"].iloc[-1]) * 100
            avg_prob = float(dd_trim["prob"].mean()) * 100

            prob_fig = go.Figure()
            # Color-coded area: above threshold = green, below = amber
            above = dd_trim["prob"] * 100
            prob_fig.add_trace(go.Scatter(
                x=dd_trim["DateTime"], y=above,
                fill="tozeroy",
                fillcolor="rgba(22,163,74,0.08)",
                line=dict(color="#16a34a", width=2),
                name="BUY prob %",
                hovertemplate="<b>%{x|%d %b}</b><br>BUY prob: %{y:.1f}%<extra></extra>",
            ))
            prob_fig.add_hline(y=BUY_PROBA * 100, line_dash="dot",
                               line_color="#16a34a", line_width=1.5,
                               annotation_text=f"Threshold {BUY_PROBA*100:.0f}%",
                               annotation_font_color="#16a34a", annotation_font_size=10)
            prob_fig.add_hline(y=avg_prob, line_dash="dash",
                               line_color="#94a3b8", line_width=1,
                               annotation_text=f"90d avg {avg_prob:.1f}%",
                               annotation_font_color="#94a3b8", annotation_font_size=10)
            # Current value annotation
            _ct = _chart_theme()
            prob_fig.add_annotation(
                x=dd_trim["DateTime"].iloc[-1], y=cur_prob,
                text=f"<b>{cur_prob:.1f}%</b>",
                showarrow=True, arrowhead=2,
                font=dict(size=10, color="#16a34a" if cur_prob >= BUY_PROBA * 100 else "#dc2626"),
                bgcolor=_ct["hover_bg"], bordercolor=_ct["hover_border"], borderwidth=1,
            )
            prob_fig.update_layout(
                template=_ct["template"], paper_bgcolor=_ct["paper_bgcolor"], plot_bgcolor=_ct["plot_bgcolor"],
                height=220, margin=dict(l=10, r=90, t=10, b=30),
                yaxis=dict(title="", range=[0, 100], ticksuffix="%",
                           tickfont=dict(size=9, color=_ct["tickcolor"]), gridcolor=_ct["gridcolor"]),
                xaxis=dict(tickfont=dict(size=9, color=_ct["tickcolor"]), gridcolor=_ct["gridcolor"]),
                showlegend=False,
                font=dict(color=_ct["font_color"], size=10),
                hoverlabel=dict(bgcolor=_ct["hover_bg"], bordercolor=_ct["hover_border"]),
            )
            st.plotly_chart(prob_fig, use_container_width=True)

        with calc_col:
            st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#64748b;letter-spacing:.4px;margin-bottom:8px'>TRADE RISK CALCULATOR</div>", unsafe_allow_html=True)
            rk1, rk2 = st.columns(2)
            trade_capital = rk1.number_input("Capital (₹)", value=100000, step=10000, min_value=10000, key="rk_cap")
            risk_pct      = rk2.slider("Risk %", 0.5, 5.0, 1.0, 0.25, key="rk_pct")
            custom_entry  = st.number_input("Entry price (₹)", value=float(round(dd_price, 1)), key="rk_entry")

            stop_price    = custom_entry - ATR_STOP_MULT  * dd_atr
            target_price  = custom_entry + ATR_TARGET_MULT * dd_atr
            risk_amount   = trade_capital * risk_pct / 100
            risk_per_share = custom_entry - stop_price
            shares_calc   = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            invest_total  = shares_calc * custom_entry
            potential_p   = shares_calc * (target_price - custom_entry)
            pct_port      = invest_total / trade_capital * 100 if trade_capital > 0 else 0
            rr            = ATR_TARGET_MULT / ATR_STOP_MULT

            st.markdown(f"""
<div class='card' style='padding:14px 16px'>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
    <div style='text-align:center;padding:8px;background:rgba(17,24,39,0.6);border-radius:8px'>
      <div style='font-size:0.62rem;color:#94a3b8;font-weight:600'>SHARES</div>
      <div style='font-size:1.1rem;font-weight:800;color:#F8FAFC'>{shares_calc}</div>
    </div>
    <div style='text-align:center;padding:8px;background:rgba(17,24,39,0.6);border-radius:8px'>
      <div style='font-size:0.62rem;color:#94a3b8;font-weight:600'>INVESTED</div>
      <div style='font-size:1.1rem;font-weight:800;color:#F8FAFC'>₹{invest_total:,.0f}</div>
      <div style='font-size:0.65rem;color:#64748b'>{pct_port:.1f}% of capital</div>
    </div>
    <div style='text-align:center;padding:8px;background:#fee2e2;border-radius:8px'>
      <div style='font-size:0.62rem;color:#b91c1c;font-weight:600'>STOP LOSS</div>
      <div style='font-size:1.1rem;font-weight:800;color:#dc2626'>₹{stop_price:,.1f}</div>
      <div style='font-size:0.65rem;color:#b91c1c'>-₹{risk_amount:,.0f} max loss</div>
    </div>
    <div style='text-align:center;padding:8px;background:#dcfce7;border-radius:8px'>
      <div style='font-size:0.62rem;color:#15803d;font-weight:600'>TARGET</div>
      <div style='font-size:1.1rem;font-weight:800;color:#16a34a'>₹{target_price:,.1f}</div>
      <div style='font-size:0.65rem;color:#15803d'>+₹{potential_p:,.0f} potential</div>
    </div>
  </div>
  <div style='text-align:center;margin-top:10px;padding:6px;background:rgba(59,130,246,0.07);border-radius:6px'>
    <span style='font-size:0.7rem;color:#64748b'>Reward:Risk = </span>
    <span style='font-size:0.9rem;font-weight:800;color:#16a34a'>1 : {rr:.1f}</span>
    <span style='font-size:0.7rem;color:#94a3b8'> &nbsp;({ATR_STOP_MULT}× / {ATR_TARGET_MULT}× ATR)</span>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Row 5: Intraday Intelligence (yfinance free, or Zerodha if configured) ─
        _zerodha_ready = False
        try:
            from zerodha_data.config import api_key as _zk
            _zerodha_ready = _zk != "YOUR_API_KEY"
        except Exception:
            pass

        _intra_src = "Zerodha · live" if _zerodha_ready else "yfinance · ~15-min delay"
        with st.expander("Intraday Intelligence", expanded=True):
            st.markdown(
                f"<div style='font-size:0.72rem;color:#64748b;margin-bottom:10px'>"
                f"VWAP · Opening Range Breakout · Volume Surge &nbsp;"
                f"<span style='background:rgba(59,130,246,0.07);border-radius:4px;padding:1px 7px;"
                f"font-weight:600;font-size:0.68rem'>{_intra_src}</span></div>",
                unsafe_allow_html=True,
            )
            try:
                if _zerodha_ready:
                    from zerodha_data import get_kite, get_intraday_features, TokenExpiredError
                    with st.spinner("Fetching intraday data…"):
                        kite_inst = get_kite()
                        id_feats  = get_intraday_features(dd_symbol, kite=kite_inst)
                else:
                    from intraday import get_intraday_features as _yfget
                    with st.spinner("Fetching intraday data…"):
                        id_feats = _yfget(dd_symbol)

                if id_feats.get("data_ok"):
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    vwap_r = id_feats["vwap_ratio"]
                    orb    = id_feats["orb_signal"]
                    vsurge = id_feats["intraday_vol_surge"]
                    mrng   = id_feats["morning_range_pos"]
                    depth  = id_feats.get("depth_score", float("nan"))
                    fhr    = id_feats.get("first_hour_return")

                    vwap_col = "#16a34a" if vwap_r > 1.005 else ("#dc2626" if vwap_r < 0.995 else "#64748b")
                    orb_col  = "#16a34a" if orb > 0 else ("#dc2626" if orb < 0 else "#64748b")
                    orb_lbl  = "Above ORB ▲" if orb > 0 else ("Below ORB ▼" if orb < 0 else "Inside Range")

                    ic1.markdown(f"""<div class='card' style='text-align:center;padding:12px'>
<div style='font-size:0.62rem;color:#94a3b8;font-weight:700'>VWAP RATIO</div>
<div style='font-size:1.3rem;font-weight:800;color:{vwap_col}'>{vwap_r:.3f}</div>
<div style='font-size:0.68rem;color:#64748b'>{"Above VWAP" if vwap_r>1 else "Below VWAP"}</div>
</div>""", unsafe_allow_html=True)

                    ic2.markdown(f"""<div class='card' style='text-align:center;padding:12px'>
<div style='font-size:0.62rem;color:#94a3b8;font-weight:700'>ORB SIGNAL</div>
<div style='font-size:1.1rem;font-weight:800;color:{orb_col}'>{orb_lbl}</div>
<div style='font-size:0.68rem;color:#64748b'>Range pos {mrng*100:.0f}%</div>
</div>""", unsafe_allow_html=True)

                    vol_col = "#16a34a" if vsurge >= 1.5 else ("#ca8a04" if vsurge >= 1.0 else "#64748b")
                    ic3.markdown(f"""<div class='card' style='text-align:center;padding:12px'>
<div style='font-size:0.62rem;color:#94a3b8;font-weight:700'>VOL SURGE</div>
<div style='font-size:1.3rem;font-weight:800;color:{vol_col}'>{vsurge:.2f}×</div>
<div style='font-size:0.68rem;color:#64748b'>vs 20d avg same time</div>
</div>""", unsafe_allow_html=True)

                    if not pd.isna(depth):
                        dep_col = "#16a34a" if depth >= 0.55 else ("#dc2626" if depth <= 0.45 else "#64748b")
                        dep_lbl = "Buyers dominant" if depth >= 0.55 else ("Sellers dominant" if depth <= 0.45 else "Balanced")
                        ic4.markdown(f"""<div class='card' style='text-align:center;padding:12px'>
<div style='font-size:0.62rem;color:#94a3b8;font-weight:700'>DEPTH SCORE</div>
<div style='font-size:1.3rem;font-weight:800;color:{dep_col}'>{depth*100:.0f}%</div>
<div style='font-size:0.68rem;color:#64748b'>{dep_lbl}</div>
</div>""", unsafe_allow_html=True)
                    elif fhr is not None:
                        fhr_col = "#16a34a" if fhr > 0.5 else ("#dc2626" if fhr < -0.5 else "#64748b")
                        ic4.markdown(f"""<div class='card' style='text-align:center;padding:12px'>
<div style='font-size:0.62rem;color:#94a3b8;font-weight:700'>1H RETURN</div>
<div style='font-size:1.3rem;font-weight:800;color:{fhr_col}'>{fhr:+.2f}%</div>
<div style='font-size:0.68rem;color:#64748b'>First 60-min change</div>
</div>""", unsafe_allow_html=True)
                    else:
                        ic4.caption("Depth N/A")
                else:
                    st.info("No intraday data today — market may be closed or data unavailable.")

            except Exception as _ex:
                if "token" in str(_ex).lower() or "TokenExpired" in type(_ex).__name__:
                    st.warning(f"Zerodha token expired — {_ex}")
                else:
                    st.error(f"Intraday fetch error: {_ex}")

        # ── Row 6: Feature importance bar chart ───────────────────────
        with st.expander("Model Feature Diagnostics", expanded=False):
            st.markdown("<div style='font-size:0.72rem;color:#64748b;margin-bottom:8px'>Current feature values used by the XGBoost model. Green = bullish direction, red = bearish.</div>", unsafe_allow_html=True)
            feat_vals = {}
            for col in model_features[:20]:
                v = float(dd_last.get(col, 0) or 0)
                feat_vals[col.replace("feat_", "").replace("_", " ")] = round(v, 4)

            bar_fig = go.Figure(go.Bar(
                x=list(feat_vals.values()),
                y=list(feat_vals.keys()),
                orientation="h",
                marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in feat_vals.values()],
                marker_opacity=0.75,
                text=[f"{v:.3f}" for v in feat_vals.values()],
                textposition="outside",
                textfont=dict(size=9, color="#64748b"),
            ))
            _ct = _chart_theme()
            bar_fig.update_layout(
                template=_ct["template"], paper_bgcolor=_ct["paper_bgcolor"], plot_bgcolor=_ct["plot_bgcolor"],
                height=380, margin=dict(l=140, r=60, t=10, b=20),
                xaxis=dict(gridcolor=_ct["gridcolor"], tickfont=dict(size=9, color=_ct["tickcolor"])),
                yaxis=dict(tickfont=dict(size=9, color=_ct["font_color"])),
                font=dict(color=_ct["font_color"], size=10),
            )
            st.plotly_chart(bar_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 5 — INTRADAY PICKS
# ═══════════════════════════════════════════════════════════════════════

with tab5:
    from intraday_scan import scan_intraday, market_status, get_today_bars
    from pathlib import Path as _Path

    _ms = market_status()
    _ms_col = "#16a34a" if _ms["open"] else "#64748b"
    _ms_lbl = "OPEN" if _ms["open"] else "CLOSED"
    _ml_ready = _Path("models/intraday_v3.pkl").exists()

    # ── Header row ────────────────────────────────────────────────────
    _hdr_c1, _hdr_c2, _hdr_c3 = st.columns([3, 1, 1])
    _hdr_c1.markdown(
        f"<div style='padding:6px 0'>"
        f"<span style='background:{_ms_col};color:#fff;border-radius:4px;"
        f"padding:2px 10px;font-size:0.72rem;font-weight:700'>{_ms_lbl}</span>"
        f"&nbsp; <span style='font-size:0.8rem;color:#64748b'>"
        f"NSE · {_ms['time_ist']} · {_ms['date_ist']}</span>"
        f"&nbsp; <span style='background:{'#dcfce7' if _ml_ready else '#fef9c3'};"
        f"color:{'#15803d' if _ml_ready else '#92400e'};border-radius:4px;"
        f"padding:1px 7px;font-size:0.68rem;font-weight:600'>"
        f"{'ML model ready' if _ml_ready else 'ML not trained'}</span></div>",
        unsafe_allow_html=True,
    )
    _do_scan  = _hdr_c2.button("🔄 Refresh", use_container_width=True)
    _do_train = _hdr_c3.button("🧠 Train ML", use_container_width=True,
                                help="Train intraday v4 model (3-stage: HOLD filter → direction ensemble → Premium BUY rule + Meta-SELL) on 60d of 5-min data (~5 min)")

    if _do_train:
        with st.spinner("Training intraday v4 model (3-stage architecture, 60 days × 5-min bars)…"):
            try:
                import subprocess, sys as _sys
                result = subprocess.run(
                    [_sys.executable, "intraday_model_v3.py", "train"],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    st.success("v4 model trained! Refresh the scan to use ML signals.")
                    st.code(result.stdout[-2000:])
                else:
                    st.error("Training failed.")
                    st.code(result.stderr[-1000:])
            except Exception as _te:
                st.error(f"Training error: {_te}")

    if _ms["last_hour"]:
        st.info("Last hour of trading — intraday entries are high-risk.")

    # ── Scan ──────────────────────────────────────────────────────────
    _scan_symbols = list(st.session_state.get("watchlist", []) or []) + []
    _extra = [s for s in NIFTY_50 if s not in _scan_symbols]
    _scan_symbols = _scan_symbols + _extra[:max(0, 20 - len(_scan_symbols))]

    _cache_key = "intraday_scan_df"
    if _do_scan or _cache_key not in st.session_state:
        with st.spinner(f"Scanning {len(_scan_symbols)} symbols in parallel…"):
            _scan_result = scan_intraday(_scan_symbols, use_ml=False)
            if _ml_ready and not _scan_result.empty:
                try:
                    import intraday_model_v3 as _imv3
                    # Parallel predict — all symbols fetched concurrently
                    _v4_preds = _imv3.batch_predict_parallel(
                        _scan_result["symbol"].tolist(), max_workers=10
                    )
                    _v4_rows = [
                        {
                            "symbol":     _sv4,
                            "v4_signal":  _rv4.get("signal",    "HOLD"),
                            "v4_premium": _rv4.get("premium",   False),
                            "v4_stage2":  _rv4.get("stage2",    "HOLD"),
                            "v4_dir_p":   round(_rv4.get("dir_proba",  0.0), 3),
                            "v4_meta_p":  round(_rv4.get("meta_proba", 0.0), 3),
                            "v4_hold_p":  round(_rv4.get("hold_proba", 0.0), 3),
                        }
                        for _sv4, _rv4 in _v4_preds.items()
                    ]
                    _scan_result = _scan_result.merge(
                        pd.DataFrame(_v4_rows), on="symbol", how="left"
                    )
                    # Auto-log any new signals (deduplicated within 30 min)
                    try:
                        import signal_log as _slog
                        for _sv4, _rv4 in _v4_preds.items():
                            if _rv4.get("signal", "HOLD") != "HOLD" and _rv4.get("data_ok", True):
                                _slog.log_signal(
                                    symbol=_sv4,
                                    signal=_rv4["signal"],
                                    premium=bool(_rv4.get("premium", False)),
                                    dir_p=float(_rv4.get("dir_proba", 0.0)),
                                    meta_p=float(_rv4.get("meta_proba", 0.0)),
                                )
                    except Exception:
                        pass
                except Exception:
                    pass
            st.session_state[_cache_key] = _scan_result

    _idf = st.session_state.get(_cache_key, pd.DataFrame())

    if _idf.empty:
        st.info("No intraday data — market may be closed or data unavailable.")
    else:
        # ── Summary counts ────────────────────────────────────────────
        _long_n   = (_idf["direction"] == "long").sum()
        _short_n  = (_idf["direction"] == "short").sum()
        _none_n   = (_idf["direction"] == "none").sum()
        _ml_buy_n  = int((_idf["v4_signal"] == "BUY").sum())  if "v4_signal" in _idf.columns else 0
        _ml_sell_n = int((_idf["v4_signal"] == "SELL").sum()) if "v4_signal" in _idf.columns else 0
        _ml_prem_n = int(_idf["v4_premium"].sum())            if "v4_premium" in _idf.columns else 0
        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        _sc1.metric("Long Setups",   _long_n)
        _sc2.metric("Avoid / Short", _short_n)
        _sc3.metric("No Setup",      _none_n)
        if _ml_ready and "v4_signal" in _idf.columns:
            _sc4.metric("v4 Signals", f"★{_ml_prem_n} BUY · {_ml_sell_n} SELL")
        else:
            _sc4.metric("v4 Signals", "—")

        st.markdown("---")

        # ── Picks table ───────────────────────────────────────────────
        st.markdown("#### Intraday Picks")
        _show_all = st.checkbox("Show all (including No Setup)", value=False)
        _disp_df  = _idf if _show_all else _idf[_idf["direction"] != "none"].copy()

        def _style_intra(row):
            styles = []
            for col in row.index:
                if col == "setup":
                    if "Strong ORB" in str(row.get("setup", "")):
                        styles.append("color:#15803d;font-weight:700")
                    elif "ORB Breakout" in str(row.get("setup", "")):
                        styles.append("color:#16a34a;font-weight:600")
                    elif any(x in str(row.get("setup", "")) for x in ["Momentum", "VWAP Hold"]):
                        styles.append("color:#0369a1;font-weight:600")
                    elif any(x in str(row.get("setup", "")) for x in ["Avoid", "Selling", "Breakdown"]):
                        styles.append("color:#b91c1c;font-weight:600")
                    else:
                        styles.append("color:#94a3b8")
                elif col in ("score", "combined"):
                    v = row.get(col, 50)
                    styles.append(
                        "color:#15803d;font-weight:700" if v >= 70
                        else ("color:#0369a1;font-weight:600" if v >= 55 else "color:#94a3b8")
                    )
                elif col == "v4_signal":
                    sig  = row.get("v4_signal",  "HOLD")
                    prem = bool(row.get("v4_premium", False))
                    if prem:
                        styles.append("color:#d97706;font-weight:800")
                    elif sig == "BUY":
                        styles.append("color:#15803d;font-weight:700")
                    elif sig == "SELL":
                        styles.append("color:#b91c1c;font-weight:700")
                    else:
                        styles.append("color:#94a3b8")
                elif col == "rr":
                    v = row.get("rr") or 0
                    styles.append("color:#16a34a;font-weight:600" if v >= 1.5 else "color:#CBD5E1")
                else:
                    styles.append("color:#CBD5E1")
            return styles

        _base_cols = ["symbol", "combined", "score", "setup", "cur_price",
                      "entry", "stop", "target", "rr",
                      "vwap_ratio", "intraday_vol_surge", "first_hour_return"]
        if _ml_ready and "v4_signal" in _disp_df.columns:
            _base_cols = ["symbol", "v4_signal", "v4_dir_p", "v4_meta_p", "combined", "score",
                          "setup", "cur_price", "entry", "stop", "target", "rr",
                          "intraday_vol_surge"]
        _tcols = [c for c in _base_cols if c in _disp_df.columns]
        _tfmt  = {
            "cur_price": "₹{:,.1f}", "entry": "₹{:,.1f}",
            "stop": "₹{:,.1f}", "target": "₹{:,.1f}",
            "vwap_ratio": "{:.3f}", "intraday_vol_surge": "{:.2f}×",
            "first_hour_return": "{:+.2f}%",
            "v4_dir_p": "{:.0%}", "v4_meta_p": "{:.0%}",
        }

        _styled_intra = (
            _disp_df[_tcols].style
            .apply(_style_intra, axis=1)
            .format({k: v for k, v in _tfmt.items() if k in _tcols}, na_rep="—")
        )
        try:
            _evt = st.dataframe(
                _styled_intra, use_container_width=True, height=320,
                on_select="rerun", selection_mode="single-row", key="intra_table",
            )
            _sel_row = (
                _disp_df.iloc[_evt.selection.rows[0]]
                if _evt.selection and _evt.selection.rows and not _disp_df.empty
                else None
            )
        except Exception:
            st.dataframe(_styled_intra, use_container_width=True, height=320)
            _sel_row = None

        # ── Selected symbol detail ─────────────────────────────────────
        if _sel_row is not None:
            _sym = _sel_row["symbol"]
        else:
            _sym = _disp_df.iloc[0]["symbol"] if not _disp_df.empty else None

        if _sym:
            st.markdown(f"---\n#### {_sym} — Intraday Chart")
            _detail_row = _idf[_idf["symbol"] == _sym].iloc[0] if not _idf[_idf["symbol"] == _sym].empty else None

            _detail_c1, _detail_c2 = st.columns([1, 3])

            if _detail_row is not None:
                _entry    = _detail_row.get("entry")
                _stop     = _detail_row.get("stop")
                _target   = _detail_row.get("target")
                _rr       = _detail_row.get("rr")
                _setup    = _detail_row.get("setup", "—")
                _score    = _detail_row.get("score", 0)
                _combined = _detail_row.get("combined", _score)
                _sc_col   = "#15803d" if _combined >= 70 else ("#0369a1" if _combined >= 55 else "#64748b")
                _v4_sig    = str(_detail_row.get("v4_signal",  "—"))
                _v4_prem   = bool(_detail_row.get("v4_premium", False))
                _v4_stage2 = str(_detail_row.get("v4_stage2",  "—"))
                _v4_dir_p  = float(_detail_row.get("v4_dir_p",  0.0))
                _v4_meta_p = float(_detail_row.get("v4_meta_p", 0.0))
                _v4_hold_p = float(_detail_row.get("v4_hold_p", 0.0))
                _v4_sig_col = (
                    "#d97706" if _v4_prem else
                    ("#15803d" if _v4_sig == "BUY" else
                     ("#b91c1c" if _v4_sig == "SELL" else "#64748b"))
                )

                # v4 3-stage breakdown panel
                _bar_html = ""
                if _ml_ready and _v4_sig != "—":
                    _prem_badge = (
                        "<span style='background:#fef3c7;color:#92400e;border-radius:3px;"
                        "padding:1px 5px;font-size:0.58rem;font-weight:700'>★ PREMIUM</span>"
                        if _v4_prem else ""
                    )
                    _dir_pct  = max(2, min(98, round(_v4_dir_p * 100)))
                    _bear_pct = 100 - _dir_pct
                    _dir_col  = "#15803d" if _v4_dir_p >= 0.60 else ("#b91c1c" if _v4_dir_p <= 0.40 else "#64748b")
                    _hold_filtered = _v4_hold_p >= 0.52
                    _stage1_txt = (
                        f"<span style='color:#64748b'>HOLD filter: PASS</span>"
                        if not _hold_filtered else
                        f"<span style='color:#b45309'>HOLD filter: FILTERED ({_v4_hold_p:.0%})</span>"
                    )
                    _meta_line = (
                        f"<div style='font-size:0.58rem;color:#64748b;margin-top:2px'>"
                        f"Meta confidence: {_v4_meta_p:.0%}</div>"
                        if _v4_sig != "HOLD" else ""
                    )
                    _bar_html = f"""
<div style='margin:8px 0 4px;font-size:0.6rem;color:#94a3b8;font-weight:700'>V4 MODEL — 3 STAGES</div>
<div style='font-size:0.58rem;margin-bottom:5px'>{_stage1_txt}</div>
<div style='font-size:0.6rem;color:#94a3b8;margin-bottom:2px'>Stage 2 · Direction confidence</div>
<div style='display:flex;border-radius:3px;overflow:hidden;height:10px;margin-bottom:2px'>
  <div style='width:{_bear_pct}%;background:#fca5a5'></div>
  <div style='width:{_dir_pct}%;background:#86efac'></div>
</div>
<div style='font-size:0.58rem;color:{_dir_col};margin-bottom:6px'>{_v4_stage2} · {_v4_dir_p:.0%} bull</div>
<div style='font-size:0.6rem;color:#94a3b8;margin-bottom:2px'>Stage 3 · Final signal</div>
<div style='font-size:0.9rem;font-weight:800;color:{_v4_sig_col}'>{_v4_sig} {_prem_badge}</div>
{_meta_line}"""

                _detail_c1.markdown(f"""<div class='card' style='padding:14px'>
<div style='font-size:0.62rem;color:#94a3b8;font-weight:700;margin-bottom:4px'>SETUP</div>
<div style='font-size:0.82rem;font-weight:700;color:#F8FAFC;margin-bottom:8px'>{_setup}</div>
<div style='display:flex;gap:10px;margin-bottom:4px'>
  <div>
    <div style='font-size:0.58rem;color:#94a3b8;font-weight:700'>SCORE</div>
    <div style='font-size:1.3rem;font-weight:800;color:{_sc_col}'>{_combined}</div>
  </div>
  {'<div><div style="font-size:0.58rem;color:#94a3b8;font-weight:700">v4 SIGNAL</div>' +
   f'<div style="font-size:1.1rem;font-weight:800;color:{_v4_sig_col}">{_v4_sig}</div></div>'
   if _ml_ready else ''}
</div>
{_bar_html}
<hr style='border:none;border-top:1px solid rgba(59,130,246,0.12);margin:10px 0'>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px'>
  <div><div style='font-size:0.58rem;color:#94a3b8;font-weight:700'>ENTRY</div>
       <div style='font-size:0.88rem;font-weight:700'>{"₹{:,.1f}".format(_entry) if _entry else "—"}</div></div>
  <div><div style='font-size:0.58rem;color:#dc2626;font-weight:700'>STOP</div>
       <div style='font-size:0.88rem;font-weight:700;color:#dc2626'>{"₹{:,.1f}".format(_stop) if _stop else "—"}</div></div>
  <div><div style='font-size:0.58rem;color:#16a34a;font-weight:700'>TARGET</div>
       <div style='font-size:0.88rem;font-weight:700;color:#16a34a'>{"₹{:,.1f}".format(_target) if _target else "—"}</div></div>
  <div><div style='font-size:0.58rem;color:#94a3b8;font-weight:700'>R:R</div>
       <div style='font-size:0.88rem;font-weight:700'>{"1 : " + str(_rr) if _rr else "—"}</div></div>
</div></div>""", unsafe_allow_html=True)

            # 5-min chart
            with _detail_c2:
                _bars = get_today_bars(_sym)
                if _bars is not None and not _bars.empty and _detail_row is not None:
                    from plotly.subplots import make_subplots
                    _cfig = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04,
                    )
                    _cfig.add_trace(go.Candlestick(
                        x=_bars["DateTime"], open=_bars["Open"], high=_bars["High"],
                        low=_bars["Low"], close=_bars["Close"],
                        name=_sym, showlegend=False,
                        increasing_fillcolor="#bbf7d0", decreasing_fillcolor="#fecaca",
                        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
                        whiskerwidth=0.4, line=dict(width=1),
                    ), row=1, col=1)

                    # VWAP line
                    _vwap_v = _detail_row.get("vwap")
                    if _vwap_v and not (isinstance(_vwap_v, float) and math.isnan(_vwap_v)):
                        _cfig.add_hline(
                            y=_vwap_v, line_color="#6366f1", line_width=1.5,
                            line_dash="dash",
                            annotation_text=f"VWAP {_vwap_v:,.0f}",
                            annotation_position="right",
                            annotation_font_size=9, annotation_font_color="#6366f1",
                            row=1, col=1,
                        )

                    # ORB band
                    _oh = _detail_row.get("orb_high")
                    _ol = _detail_row.get("orb_low")
                    if _oh and _ol and not (isinstance(_oh, float) and math.isnan(_oh)):
                        _cfig.add_hrect(
                            y0=_ol, y1=_oh, fillcolor="#fef9c3",
                            opacity=0.3, line_width=0,
                            row=1, col=1,
                        )
                        _cfig.add_hline(
                            y=_oh, line_color="#ca8a04", line_width=1,
                            line_dash="dot",
                            annotation_text=f"ORB H {_oh:,.0f}",
                            annotation_position="right",
                            annotation_font_size=9, annotation_font_color="#ca8a04",
                            row=1, col=1,
                        )
                        _cfig.add_hline(
                            y=_ol, line_color="#ca8a04", line_width=1,
                            line_dash="dot",
                            annotation_text=f"ORB L {_ol:,.0f}",
                            annotation_position="right",
                            annotation_font_size=9, annotation_font_color="#ca8a04",
                            row=1, col=1,
                        )

                    # Volume bars
                    _vol_colors = [
                        "#bbf7d0" if c >= o else "#fecaca"
                        for c, o in zip(_bars["Close"], _bars["Open"])
                    ]
                    _cfig.add_trace(go.Bar(
                        x=_bars["DateTime"], y=_bars["Volume"],
                        marker_color=_vol_colors, name="Vol", showlegend=False,
                        marker_opacity=0.7,
                    ), row=2, col=1)

                    _ct = _chart_theme()
                    _cfig.update_layout(
                        template=_ct["template"],
                        paper_bgcolor=_ct["paper_bgcolor"], plot_bgcolor=_ct["plot_bgcolor"],
                        height=380,
                        margin=dict(l=10, r=120, t=20, b=10),
                        xaxis_rangeslider_visible=False,
                        font=dict(color=_ct["font_color"], size=10),
                    )
                    _cfig.update_yaxes(tickfont=dict(size=9), gridcolor=_ct["gridcolor"])
                    _cfig.update_xaxes(tickfont=dict(size=9), gridcolor=_ct["gridcolor"])
                    st.plotly_chart(_cfig, use_container_width=True)
                else:
                    st.caption("No 5-min data available for today.")

        # ── Signal Log ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Signal Log")
        try:
            import signal_log as _slog
            _log_c1, _log_c2 = st.columns([3, 1])
            with _log_c2:
                if st.button("Mark outcomes", use_container_width=True,
                             help="Auto-fetch current prices and mark WIN/LOSS for pending signals"):
                    try:
                        _updated = _slog.auto_update_outcomes()
                        st.success(f"Updated {_updated} signal(s)")
                    except Exception as _oe:
                        st.error(str(_oe))

            _log_summary = _slog.summary()
            _ls1, _ls2, _ls3, _ls4 = st.columns(4)
            _ls1.metric("Total logged",  _log_summary["total"])
            _ls2.metric("Pending",       _log_summary["pending"])
            _ls3.metric("Wins / Losses", f"{_log_summary['wins']} / {_log_summary['losses']}")
            _ls4.metric("Win rate",
                        f"{_log_summary['win_rate']:.0%}" if _log_summary["win_rate"] is not None else "—")

            _log_df = _slog.get_recent(40)
            if _log_df.empty:
                st.caption("No signals logged yet — run a scan during market hours.")
            else:
                def _style_log(row):
                    styles = []
                    for col in row.index:
                        if col == "signal":
                            styles.append("color:#15803d;font-weight:700" if row["signal"] == "BUY"
                                          else "color:#b91c1c;font-weight:700")
                        elif col == "outcome":
                            v = row.get("outcome")
                            styles.append(
                                "color:#15803d;font-weight:700" if v == "WIN"
                                else ("color:#b91c1c;font-weight:700" if v == "LOSS"
                                      else "color:#94a3b8")
                            )
                        elif col == "premium":
                            styles.append("color:#d97706;font-weight:700" if row.get("premium") else "color:#94a3b8")
                        else:
                            styles.append("color:#CBD5E1")
                    return styles

                _log_show_cols = ["ts", "symbol", "signal", "premium", "dir_p",
                                  "meta_p", "entry_price", "outcome", "exit_price"]
                _log_show_cols = [c for c in _log_show_cols if c in _log_df.columns]
                _log_fmt = {"dir_p": "{:.0%}", "meta_p": "{:.0%}",
                            "entry_price": "₹{:,.1f}", "exit_price": "₹{:,.1f}"}
                st.dataframe(
                    _log_df[_log_show_cols].style
                    .apply(_style_log, axis=1)
                    .format({k: v for k, v in _log_fmt.items() if k in _log_show_cols},
                            na_rep="—"),
                    use_container_width=True, height=240,
                )
        except Exception:
            st.caption("Signal log unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
_stock_tiers_blob = blob.get("stock_tiers", {})
_tier_a_count = sum(1 for t in _stock_tiers_blob.values() if t == "A")
_tier_b_count = sum(1 for t in _stock_tiers_blob.values() if t == "B")
st.markdown(f"""
<div style='display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:10px 0;
     border-top:1px solid rgba(59,130,246,0.10);font-size:0.72rem;color:#475569'>
  <span><b style='color:#64748b'>Model</b> v{blob.get("version","5.1")} · {blob.get("n_train",0):,} samples · {len(model_features)} features</span>
  <span><b style='color:#64748b'>Trained</b> {blob.get("trained_at","—")}</span>
  <span><b style='color:#64748b'>Val acc</b> {blob.get("val_acc",0)*100:.1f}%</span>
  <span><b style='color:#64748b'>Universe</b> {len(_stock_tiers_blob)} stocks · <span style='color:#FCD34D'>Tier A: {_tier_a_count}</span> · <span style='color:#94A3B8'>Tier B: {_tier_b_count}</span></span>
  <span><b style='color:#64748b'>Cache</b> scan 15 min · market 30 min · news 5 min</span>
  <span style='color:#EF4444;font-weight:600'>⚠ Educational only — not financial advice. Not SEBI registered.</span>
</div>""", unsafe_allow_html=True)
