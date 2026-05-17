"""
intraday_model.py — ML intraday signal engine for NSE stocks.

Implements 25 theories as XGBoost features (57 features total).

Architecture — TWO-STAGE MODEL:
  Stage 1 — Direction model (binary: BUY vs SELL)
    Trained ONLY on bars where price moves ≥0.3% (clean directional examples).
    Ignores HOLD bars entirely → no confounding with ambiguous bars.
    Target: ≥65% directional accuracy (BUY precision+recall ≥60%).

  Stage 2 — HOLD filter (binary: is-this-bar-worth-trading vs HOLD)
    Trained on all bars to learn when a directional move is likely.
    High threshold (≥0.48) means only confident bars emit a signal.

  Combined: emit BUY/SELL only when HOLD-filter says "trade" AND
            direction model has ≥0.52 confidence; otherwise HOLD.

Theories:
  Original (1-7): ORB, VWAP, Supertrend, CPR, EMA cross, Bollinger, RSI+MACD
  Round 2  (8-15): Microstructure, ATR-bars, Gap/prev-day, Stochastic,
                   Heikin-Ashi, Time-of-day, Volume-acceleration, Candle patterns
  Round 3 (16-20): MFI, ADX/+DI, OBV trend, Chaikin MF, Fast-RSI-5
  Round 4     (21): Market-regime — Nifty 50 return (1-bar, 6-bar), Nifty RSI,
                    relative strength of stock vs Nifty

CLI:
  python intraday_model.py train                # full train, Nifty-50
  python intraday_model.py train --test         # quick test, 8 symbols
  python intraday_model.py predict RELIANCE     # live signal
  python intraday_model.py importance           # feature importances
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from swing_v2 import NIFTY_50

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = Path("models/intraday_v1.pkl")
TARGET_BARS  = 6       # 6 × 5-min = 30 min lookahead
BUY_THRESH   = 0.005   # +0.5% → BUY  (high-water mark; cleaner signal)
SELL_THRESH  = -0.005  # −0.5% → SELL (high-water mark; cleaner signal)

# Stage-1 direction model threshold (BUY/SELL confidence floor)
# 0.58 → ~44% emit, ~67% accuracy  |  0.65 → ~12% emit, ~72%  |  0.70 → ~7%, ~74%
DIR_THRESHOLD  = 0.62
# Stage-2 HOLD filter threshold (non-HOLD confidence floor)
HOLD_THRESHOLD = 0.55

FEATURE_COLS = [
    # ── Theory 1: ORB ─────────────────────────────────────────────────
    "orb_signal", "morning_range_pos",
    # ── Theory 2: VWAP ────────────────────────────────────────────────
    "vwap_pct", "vwap_dev_norm",
    # ── Theory 3: Supertrend ──────────────────────────────────────────
    "supertrend_sig",
    # ── Theory 4: CPR ─────────────────────────────────────────────────
    "cpr_pos",
    # ── Theory 5: EMA cross ───────────────────────────────────────────
    "ema9_21_sig", "ema21_50_sig",
    # ── Theory 6: Bollinger ───────────────────────────────────────────
    "bb_pct_b", "bb_squeeze", "bb_width",
    # ── Theory 7: RSI + MACD ──────────────────────────────────────────
    "rsi", "rsi_zone", "macd_hist_norm", "macd_cross",
    # ── Theory 8: Market Microstructure ───────────────────────────────
    "buy_pressure", "upper_wick_pct", "lower_wick_pct",
    # ── Theory 9: ATR-normalized bar quality ──────────────────────────
    "bar_atr_ratio", "ret_1bar_atr", "ret_3bar", "ret_6bar",
    # ── Theory 10: Gap dynamics + prev-day levels ─────────────────────
    "gap_pct", "above_pd_high", "dist_pd_high",
    # ── Theory 11: Stochastic ─────────────────────────────────────────
    "stoch_k", "stoch_d",
    # ── Theory 12: Heikin-Ashi ────────────────────────────────────────
    "ha_color", "ha_streak",
    # ── Theory 13: Time-of-day (is_first_30min dropped: 0 importance) ─
    "is_power_hour", "time_sin", "time_cos",
    # ── Theory 14: Volume acceleration ───────────────────────────────
    "vol_ratio", "vol_accel",
    # ── Theory 15: Candlestick patterns ──────────────────────────────
    "engulfing", "inside_bar", "bar_body_pct",
    # ── Theory 16: Money Flow Index ───────────────────────────────────
    "mfi",
    # ── Theory 17: ADX + Directional Movement ────────────────────────
    "adx_norm", "plus_di_norm",
    # ── Theory 18: OBV trend ─────────────────────────────────────────
    "obv_trend",
    # ── Theory 19: Chaikin Money Flow ────────────────────────────────
    "cmf",
    # ── Theory 20: Fast RSI ──────────────────────────────────────────
    "rsi_5",
    # ── Session + streak ─────────────────────────────────────────────
    "session_pct", "consec_dir",
    # ── Theory 21: Market-regime (Nifty 50 index context) ────────────
    "nifty_ret_1bar", "nifty_ret_6bar", "nifty_rsi",
    "rel_strength_1bar", "rel_strength_6bar",
    # ── Theory 22: Intraday position (session high/low distance) ─────
    "session_high_dist", "session_low_dist",
    # ── Theory 23: Opening drive (first-3-bar momentum) ──────────────
    "opening_drive", "opening_drive_vol",
    # ── Theory 24: Nifty regime (ADX + trend direction) ──────────────
    "nifty_adx", "nifty_ema_sig",
    # ── Theory 25: Fast 3-bar MACD (non-lagging momentum) ────────────
    "macd_fast_hist",
]


# ── Data fetch ────────────────────────────────────────────────────────────────

def _yf_sym(sym: str) -> str:
    return f"{sym}.NS"


# Cache for Nifty index data (shared across all symbols in a single training run)
_NIFTY_CACHE: dict = {}


def fetch_5min(symbol: str, days: int = 58) -> pd.DataFrame | None:
    try:
        tk = yf.Ticker(_yf_sym(symbol))
        df = tk.history(period=f"{days}d", interval="5m", auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.rename(columns={"Datetime": "DateTime"}, inplace=True)
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
        return df[["DateTime", "Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as e:
        print(f"[WARN] fetch_5min({symbol}): {e}")
        return None


def fetch_nifty_5min(days: int = 58) -> pd.DataFrame | None:
    """Fetch Nifty 50 index 5-min data; cached per process."""
    key = f"nifty_{days}"
    if key in _NIFTY_CACHE:
        return _NIFTY_CACHE[key]
    try:
        tk = yf.Ticker("^NSEI")
        df = tk.history(period=f"{days}d", interval="5m", auto_adjust=True)
        if df is None or df.empty:
            _NIFTY_CACHE[key] = None
            return None
        df = df.reset_index()
        df.rename(columns={"Datetime": "DateTime"}, inplace=True)
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
        df = df[["DateTime", "Close"]].rename(columns={"Close": "nifty_close"}).copy()
        _NIFTY_CACHE[key] = df
        return df
    except Exception as e:
        print(f"[WARN] fetch_nifty_5min: {e}")
        _NIFTY_CACHE[key] = None
        return None


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 7, mult: float = 3.0) -> pd.Series:
    atr     = _atr(high, low, close, period)
    hl2     = (high + low) / 2
    upper   = (hl2 + mult * atr).values.copy()
    lower   = (hl2 - mult * atr).values.copy()
    close_v = close.values
    n       = len(close_v)
    sig     = np.ones(n)
    st      = lower.copy()
    for i in range(1, n):
        if close_v[i - 1] <= upper[i - 1]:
            upper[i] = min(upper[i], upper[i - 1])
        if close_v[i - 1] >= lower[i - 1]:
            lower[i] = max(lower[i], lower[i - 1])
        if sig[i - 1] == 1:
            if close_v[i] < lower[i]:
                sig[i] = -1; st[i] = upper[i]
            else:
                st[i] = lower[i]
        else:
            if close_v[i] > upper[i]:
                sig[i] = 1; st[i] = lower[i]
            else:
                st[i] = upper[i]
    return pd.Series(sig, index=close.index)


def _vwap_daily(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """VWAP + VWAP std dev, resetting each calendar day."""
    tp       = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv      = tp * df["Volume"]
    tp2v     = tp ** 2 * df["Volume"]
    date_col = df["DateTime"].dt.date
    vwap     = pd.Series(np.nan, index=df.index)
    vwap_std = pd.Series(np.nan, index=df.index)
    for d, grp in df.groupby(date_col):
        idx      = grp.index
        cum_vol  = df.loc[idx, "Volume"].cumsum().replace(0, np.nan)
        cum_tpv  = tpv.loc[idx].cumsum()
        cum_tp2v = tp2v.loc[idx].cumsum()
        vw       = cum_tpv / cum_vol
        var      = (cum_tp2v / cum_vol - vw ** 2).clip(lower=0)
        vwap.loc[idx]     = vw
        vwap_std.loc[idx] = var.apply(np.sqrt)
    return vwap, vwap_std


def _bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    mid    = close.rolling(period).mean()
    sigma  = close.rolling(period).std(ddof=0)
    upper  = mid + std * sigma
    lower  = mid - std * sigma
    pct_b  = (close - lower) / (upper - lower).replace(0, np.nan)
    bwidth = (sigma / mid.replace(0, np.nan)).fillna(0)
    squeeze = (bwidth < bwidth.rolling(50).mean()).astype(float)
    return pct_b, squeeze, bwidth


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14, sk: int = 3, sd: int = 3):
    lo_n  = low.rolling(period).min()
    hi_n  = high.rolling(period).max()
    raw_k = (close - lo_n) / (hi_n - lo_n).replace(0, np.nan) * 100
    k     = raw_k.rolling(sk).mean()
    d     = k.rolling(sd).mean()
    return (k / 100).clip(0, 1).fillna(0.5), (d / 100).clip(0, 1).fillna(0.5)


def _heikin_ashi(open_: pd.Series, high: pd.Series,
                 low: pd.Series, close: pd.Series):
    ha_close = (open_ + high + low + close) / 4
    ha_open  = pd.Series(np.nan, index=open_.index, dtype=float)
    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(open_)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    return ha_close, ha_open


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI. Returns 0-1 normalized."""
    tp  = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    return (1 - 1 / (1 + pos_sum / neg_sum.replace(0, np.nan))).clip(0, 1).fillna(0.5)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> tuple[pd.Series, pd.Series]:
    """ADX (trend strength) and net directional index (+DI - -DI)/100."""
    atr14    = _atr(high, low, close, period)
    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    # Zero out whichever is smaller
    cond     = plus_dm >= minus_dm
    plus_dm  = plus_dm.where(cond, 0.0)
    minus_dm = minus_dm.where(~cond, 0.0)
    plus_di  = (plus_dm.ewm(span=period, adjust=False).mean()
                / atr14.replace(0, np.nan) * 100).fillna(0)
    minus_di = (minus_dm.ewm(span=period, adjust=False).mean()
                / atr14.replace(0, np.nan) * 100).fillna(0)
    dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100).fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()
    return (adx / 100).clip(0, 1), ((plus_di - minus_di) / 100).clip(-1, 1)


def _obv_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV direction vs 21-bar EMA: +1 bullish / -1 bearish."""
    obv     = (np.sign(close.diff()) * volume).cumsum()
    obv_ema = obv.ewm(span=21, adjust=False).mean()
    return np.sign(obv - obv_ema).fillna(0)


def _chaikin_mf(high: pd.Series, low: pd.Series,
                close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow: buying/selling pressure over `period` bars. Returns -1..+1."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return ((clv * volume).rolling(period).sum()
            / volume.rolling(period).sum().replace(0, np.nan)).clip(-1, 1).fillna(0)


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return 45-feature DataFrame aligned with df's index."""
    df       = df.sort_values("DateTime").reset_index(drop=True)
    date_col = df["DateTime"].dt.date
    close    = df["Close"]
    high     = df["High"]
    low      = df["Low"]
    volume   = df["Volume"]
    open_    = df["Open"]
    out      = pd.DataFrame(index=df.index)

    # ── Whole-series indicators ───────────────────────────────────────

    # Theory 3
    out["supertrend_sig"] = _supertrend(high, low, close)

    # Theory 5
    ema9  = _ema(close, 9);  ema21 = _ema(close, 21);  ema50 = _ema(close, 50)
    out["ema9_21_sig"]  = np.sign(ema9  - ema21).fillna(0)
    out["ema21_50_sig"] = np.sign(ema21 - ema50).fillna(0)

    # Theory 7: RSI (14) + RSI (5) + MACD
    rsi14 = _rsi(close, 14)
    rsi5  = _rsi(close, 5)
    out["rsi"]      = (rsi14 / 100).clip(0, 1)
    out["rsi_5"]    = (rsi5  / 100).clip(0, 1)
    out["rsi_zone"] = pd.cut(rsi14, bins=[-1, 30, 70, 101],
                             labels=[0.0, 1.0, 2.0]).astype(float).fillna(1.0)
    macd_line = _ema(close, 12) - _ema(close, 26)
    macd_sig  = _ema(macd_line, 9)
    macd_hist = macd_line - macd_sig
    roll_std  = close.rolling(20).std(ddof=0).replace(0, np.nan)
    out["macd_hist_norm"] = (macd_hist / roll_std).clip(-3, 3).fillna(0)
    out["macd_cross"] = (
        ((macd_line > macd_sig) & (macd_line.shift() <= macd_sig.shift())).astype(float)
        - ((macd_line < macd_sig) & (macd_line.shift() >= macd_sig.shift())).astype(float)
    ).fillna(0)

    # Theory 6: Bollinger
    bb_pct_b, bb_squeeze, bb_width = _bollinger(close, 20, 2.0)
    out["bb_pct_b"]   = bb_pct_b.clip(-0.5, 1.5).fillna(0.5)
    out["bb_squeeze"] = bb_squeeze.fillna(0)
    out["bb_width"]   = bb_width.clip(0, 0.05).fillna(0)

    # Theory 9: ATR-normalized
    atr14 = _atr(high, low, close, 14)
    out["bar_atr_ratio"] = ((high - low) / atr14.replace(0, np.nan)).clip(0, 5).fillna(1)
    out["ret_1bar_atr"]  = (close.pct_change(1) / (atr14 / close).replace(0, np.nan)).clip(-5, 5).fillna(0)
    out["ret_3bar"]      = close.pct_change(3).clip(-0.05, 0.05).fillna(0)
    out["ret_6bar"]      = close.pct_change(6).clip(-0.08, 0.08).fillna(0)

    # Theory 11: Stochastic
    stoch_k, stoch_d = _stochastic(high, low, close)
    out["stoch_k"] = stoch_k;  out["stoch_d"] = stoch_d

    # Theory 12: Heikin-Ashi
    ha_close, ha_open = _heikin_ashi(open_, high, low, close)
    ha_dir = np.sign(ha_close - ha_open).fillna(0)
    out["ha_color"] = ha_dir
    ha_streak_arr = np.zeros(len(ha_dir)); cur_s = 0.0
    for i, d in enumerate(ha_dir):
        if np.isnan(d) or d == 0: cur_s = 0.0
        elif d == np.sign(cur_s) or cur_s == 0: cur_s += d
        else: cur_s = d
        ha_streak_arr[i] = cur_s
    out["ha_streak"] = np.clip(ha_streak_arr, -10, 10)

    # Theory 14: Volume
    out["vol_ratio"] = (volume / volume.rolling(20).mean().replace(0, np.nan)).clip(0, 10).fillna(1)
    out["vol_accel"] = (volume / volume.rolling(3).mean().shift(1).replace(0, np.nan)).clip(0, 8).fillna(1)

    # Theory 15: Candlestick patterns + microstructure (Theory 8)
    body       = close - open_
    hl_range   = (high - low).replace(0, np.nan)
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
    out["buy_pressure"]   = ((close - low) / hl_range).clip(0, 1).fillna(0.5)
    out["upper_wick_pct"] = (upper_wick / hl_range).clip(0, 1).fillna(0)
    out["lower_wick_pct"] = (lower_wick / hl_range).clip(0, 1).fillna(0)
    out["bar_body_pct"]   = (body.abs() / hl_range).clip(0, 1).fillna(0.5)
    prev_body = body.shift(1)
    bull_eng  = (body > 0) & (open_ <= close.shift(1)) & (close >= open_.shift(1)) & (body.abs() > prev_body.abs())
    bear_eng  = (body < 0) & (open_ >= close.shift(1)) & (close <= open_.shift(1)) & (body.abs() > prev_body.abs())
    out["engulfing"]  = (bull_eng.astype(float) - bear_eng.astype(float)).fillna(0)
    out["inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(float).fillna(0)

    # Consecutive close direction streak
    direction = np.sign(close.values - close.shift().values)
    streak    = np.zeros(len(direction)); cur_d = 0.0
    for i in range(len(direction)):
        d = direction[i]
        if np.isnan(d) or d == 0: cur_d = 0.0
        elif d == np.sign(cur_d) or cur_d == 0: cur_d += d
        else: cur_d = d
        streak[i] = cur_d
    out["consec_dir"] = np.clip(streak, -10, 10)

    # Theory 16: MFI
    out["mfi"] = _mfi(high, low, close, volume, 14)

    # Theory 17: ADX + DI
    adx_s, plus_di_s = _adx(high, low, close, 14)
    out["adx_norm"]   = adx_s.fillna(0)
    out["plus_di_norm"] = plus_di_s.fillna(0)

    # Theory 18: OBV trend
    out["obv_trend"] = _obv_trend(close, volume)

    # Theory 19: CMF
    out["cmf"] = _chaikin_mf(high, low, close, volume, 20)

    # ── Per-day features ─────────────────────────────────────────────
    vwap_all, vwap_std_all = _vwap_daily(df)

    orb_signal_s      = pd.Series(0.0,  index=df.index)
    morning_range_s   = pd.Series(0.5,  index=df.index)
    vwap_pct_s        = pd.Series(0.0,  index=df.index)
    vwap_dev_norm_s   = pd.Series(0.0,  index=df.index)
    cpr_pos_s         = pd.Series(0.0,  index=df.index)
    session_pct_s     = pd.Series(0.5,  index=df.index)
    gap_pct_s         = pd.Series(0.0,  index=df.index)
    above_pd_high_s   = pd.Series(0.0,  index=df.index)
    dist_pd_high_s    = pd.Series(0.0,  index=df.index)
    is_power_hr_s     = pd.Series(0.0,  index=df.index)
    time_sin_s        = pd.Series(0.0,  index=df.index)
    time_cos_s        = pd.Series(0.0,  index=df.index)
    session_hi_dist_s = pd.Series(0.0,  index=df.index)
    session_lo_dist_s = pd.Series(0.0,  index=df.index)
    opening_drive_s   = pd.Series(0.0,  index=df.index)
    opening_dvol_s    = pd.Series(1.0,  index=df.index)

    prev_ohlc: dict = {}
    TOTAL_BARS = 75  # NSE 5-min bars per day

    for d, grp in df.groupby(date_col):
        idx = grp.index;  n = len(grp)
        cur = grp["Close"].values

        # ORB
        orb_h = grp["High"].iloc[:min(3, n)].max()
        orb_l = grp["Low"].iloc[:min(3, n)].min()
        rng   = orb_h - orb_l
        if rng > 0:
            morning_range_s.loc[idx] = np.clip((cur - orb_l) / rng, 0, 1)
        orb_signal_s.loc[idx] = ((cur > orb_h).astype(float) - (cur < orb_l).astype(float))

        # VWAP deviation
        vw  = vwap_all.loc[idx].values
        vws = vwap_std_all.loc[idx].values
        vwap_pct_s.loc[idx]      = np.clip((cur / np.where(vw > 0, vw, np.nan) - 1) * 100, -5, 5)
        vwap_dev_norm_s.loc[idx] = np.clip((cur - vw) / np.where(vws > 0, vws, np.nan), -4, 4)

        # CPR + gap + prev-day levels
        if prev_ohlc:
            ph, pl, pc = list(prev_ohlc.values())[-1]
            pivot = (ph + pl + pc) / 3;  bc = (ph + pl) / 2;  tc = 2 * pivot - bc
            cpr_mid = (tc + bc) / 2
            if cpr_mid > 0:
                cpr_pos_s.loc[idx] = np.clip((cur / cpr_mid - 1) * 100, -5, 5)
            gap = (grp["Open"].iloc[0] / pc - 1) * 100 if pc > 0 else 0
            gap_pct_s.loc[idx]       = np.clip(gap, -5, 5)
            above_pd_high_s.loc[idx] = ((cur > ph).astype(float) - (cur < pl).astype(float))
            dist_pd_high_s.loc[idx]  = np.clip((cur / ph - 1) * 100 if ph > 0 else 0, -5, 5)

        # Time-of-day
        bar_nums = np.arange(n)
        is_power_hr_s.loc[idx] = (bar_nums >= TOTAL_BARS - 15).astype(float)
        angle = bar_nums / max(n - 1, 1) * 2 * np.pi
        time_sin_s.loc[idx] = np.sin(angle)
        time_cos_s.loc[idx] = np.cos(angle)
        session_pct_s.loc[idx] = bar_nums / max(n - 1, 1)

        # Theory 22: Session high/low distance (non-lagging real-time position)
        sess_h = grp["High"].expanding().max().values
        sess_l = grp["Low"].expanding().min().values
        with np.errstate(divide="ignore", invalid="ignore"):
            session_hi_dist_s.loc[idx] = np.clip(
                np.where(sess_h > 0, (cur / sess_h - 1) * 100, 0), -5, 0)
            session_lo_dist_s.loc[idx] = np.clip(
                np.where(sess_l > 0, (cur / sess_l - 1) * 100, 0), 0, 5)

        # Theory 23: Opening drive — first 3-bar direction × ATR-normalized magnitude
        atr14_day = atr14.loc[idx].values
        first3_h  = grp["High"].iloc[:min(3, n)].max()
        first3_l  = grp["Low"].iloc[:min(3, n)].min()
        open0     = grp["Open"].iloc[0]
        avg_atr   = atr14_day[:min(3, n)].mean() if min(3, n) > 0 else 1.0
        if avg_atr > 0 and open0 > 0:
            drive_mag = (first3_h - first3_l) / avg_atr
            drive_dir = np.sign(grp["Close"].iloc[min(2, n-1)] - open0)
        else:
            drive_mag, drive_dir = 0.0, 0.0
        opening_drive_s.loc[idx]  = float(np.clip(drive_dir * drive_mag, -5, 5))
        avg_vol = grp["Volume"].mean()
        first3_vol = grp["Volume"].iloc[:min(3, n)].mean()
        opening_dvol_s.loc[idx] = float(np.clip(
            first3_vol / avg_vol if avg_vol > 0 else 1.0, 0, 5))

        prev_ohlc[d] = (grp["High"].max(), grp["Low"].min(), grp["Close"].iloc[-1])

    out["orb_signal"]        = orb_signal_s
    out["morning_range_pos"] = morning_range_s
    out["vwap_pct"]          = vwap_pct_s
    out["vwap_dev_norm"]     = vwap_dev_norm_s.fillna(0)
    out["cpr_pos"]           = cpr_pos_s
    out["session_pct"]       = session_pct_s
    out["gap_pct"]           = gap_pct_s
    out["above_pd_high"]     = above_pd_high_s
    out["dist_pd_high"]      = dist_pd_high_s
    out["is_power_hour"]      = is_power_hr_s
    out["time_sin"]           = time_sin_s
    out["time_cos"]           = time_cos_s
    out["session_high_dist"]  = session_hi_dist_s
    out["session_low_dist"]   = session_lo_dist_s
    out["opening_drive"]      = opening_drive_s
    out["opening_drive_vol"]  = opening_dvol_s

    # ── Theory 21: Nifty 50 market-regime features ───────────────────
    nifty_df = fetch_nifty_5min(days=60)
    if nifty_df is not None and not nifty_df.empty:
        merged = df[["DateTime"]].merge(nifty_df, on="DateTime", how="left")
        nc     = merged["nifty_close"].ffill().bfill()
        nr1    = nc.pct_change(1).clip(-0.03, 0.03).fillna(0)
        nr6    = nc.pct_change(6).clip(-0.05, 0.05).fillna(0)
        nrsi   = (_rsi(nc, 14) / 100).clip(0, 1)
        out["nifty_ret_1bar"]    = nr1.values
        out["nifty_ret_6bar"]    = nr6.values
        out["nifty_rsi"]         = nrsi.values
        sr1 = close.pct_change(1).clip(-0.03, 0.03).fillna(0)
        sr6 = close.pct_change(6).clip(-0.05, 0.05).fillna(0)
        out["rel_strength_1bar"] = (sr1 - nr1.values).clip(-0.03, 0.03)
        out["rel_strength_6bar"] = (sr6 - nr6.values).clip(-0.05, 0.05)
        # Theory 24: Nifty regime — ADX strength + EMA trend direction
        nc_hi  = nc;  nc_lo = nc   # index has no H/L; approximate ADX via close
        n_adx, _ = _adx(nc_hi, nc_lo, nc, 14)
        nc_ema21  = _ema(nc, 21)
        out["nifty_adx"]     = n_adx.fillna(0).values
        out["nifty_ema_sig"] = np.sign(nc - nc_ema21).fillna(0).values
    else:
        for col in ["nifty_ret_1bar", "nifty_ret_6bar", "nifty_rsi",
                    "rel_strength_1bar", "rel_strength_6bar",
                    "nifty_adx", "nifty_ema_sig"]:
            out[col] = 0.0

    # Theory 25: Fast 3-bar MACD histogram (non-lagging momentum fingerprint)
    fast_macd  = _ema(close, 3) - _ema(close, 8)
    fast_sig   = _ema(fast_macd, 5)
    out["macd_fast_hist"] = ((fast_macd - fast_sig) / roll_std.replace(0, np.nan)
                             ).clip(-3, 3).fillna(0)

    return out[FEATURE_COLS]


def make_labels(df: pd.DataFrame) -> pd.Series:
    """Vectorized path-aware high-water mark labels over next TARGET_BARS bars.

    BUY  (1)  : price hits +BUY_THRESH% (via High) before SELL_THRESH% (via Low)
    SELL (-1) : price hits SELL_THRESH% (via Low) before +BUY_THRESH% (via High)
    HOLD (0)  : neither threshold reached in the window

    Vectorized: O(n * TARGET_BARS) NumPy — fast for large datasets.
    """
    close_v = df["Close"].values.astype(float)
    high_v  = df["High"].values.astype(float)
    low_v   = df["Low"].values.astype(float)
    n       = len(close_v)
    m       = n - TARGET_BARS                 # bars that have a full lookahead window
    lbl     = np.zeros(n, dtype=int)
    decided = np.zeros(m, dtype=bool)         # has bar i already hit a threshold?
    idx     = np.arange(m)

    for k in range(1, TARGET_BARS + 1):
        fi     = idx + k                      # future bar index
        ref    = close_v[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_h  = np.where(ref > 0, high_v[fi] / ref - 1, 0.0)
            ret_l  = np.where(ref > 0, low_v[fi]  / ref - 1, 0.0)

        still  = ~decided
        hit_b  = still & (ret_h >= BUY_THRESH)
        hit_s  = still & (ret_l <= SELL_THRESH)
        # On bars where BOTH thresholds fire on the same candle, larger move wins
        both   = hit_b & hit_s
        hit_b  = hit_b & ~(both & (ret_h < np.abs(ret_l)))
        hit_s  = hit_s & ~(both & (ret_h >= np.abs(ret_l)))

        lbl[idx[hit_b]] = 1
        lbl[idx[hit_s]] = -1
        decided        |= hit_b | hit_s

    return pd.Series(lbl, index=df.index)


def _eval_two_stage(X: pd.DataFrame, y_raw: pd.Series,
                    model_dir, model_hold) -> dict:
    """Evaluate the two-stage model on a held-out set. Returns accuracy metrics."""
    # Stage 1 — direction (BUY vs SELL bars only)
    bs_mask   = y_raw.isin([1, -1])
    X_bs      = X[bs_mask]
    y_bs      = (y_raw[bs_mask] == 1).astype(int)   # 1=BUY, 0=SELL
    dir_proba = model_dir.predict_proba(X_bs)[:, 1]  # P(BUY)
    dir_pred  = (dir_proba >= 0.5).astype(int)
    dir_acc   = float(np.mean(dir_pred == y_bs.values))

    buy_mask  = y_bs == 1
    sell_mask = y_bs == 0
    buy_acc   = float(np.mean(dir_pred[buy_mask.values]  == 1)) if buy_mask.any()  else 0.0
    sell_acc  = float(np.mean(dir_pred[sell_mask.values] == 0)) if sell_mask.any() else 0.0

    # Stage 2 — HOLD filter (all bars)
    y_nhold     = (y_raw != 0).astype(int)   # 1=BUY or SELL, 0=HOLD
    hold_proba  = model_hold.predict_proba(X)[:, 1]   # P(non-HOLD)
    hold_pred   = (hold_proba >= 0.5).astype(int)
    hold_acc    = float(np.mean(hold_pred == y_nhold.values))

    # Combined signal accuracy (on bars where model actually emits BUY/SELL)
    emit_mask   = (hold_proba >= HOLD_THRESHOLD)
    n_emit      = emit_mask.sum()
    combined_acc = 0.0
    if n_emit > 0:
        dir_emit  = (model_dir.predict_proba(X[emit_mask])[:, 1] >= DIR_THRESHOLD).astype(int)
        true_dir  = (y_raw[emit_mask].values == 1).astype(int)
        combined_acc = float(np.mean(dir_emit == true_dir))

    return {
        "dir_acc":      dir_acc,
        "buy_acc":      buy_acc,
        "sell_acc":     sell_acc,
        "hold_acc":     hold_acc,
        "combined_acc": combined_acc,
        "n_emit":       int(n_emit),
        "n_total":      len(X),
    }


# ── Walk-forward cross-validation ─────────────────────────────────────────────

def walk_forward_cv(X_all: pd.DataFrame, y_all: pd.Series,
                    n_folds: int = 3) -> list[dict]:
    """3-fold temporal walk-forward CV — proves non-overfitting."""
    n         = len(X_all)
    fold_size = n // (n_folds + 1)
    results   = []

    for fold in range(n_folds):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        X_tr    = X_all.iloc[:train_end];   y_tr = y_all.iloc[:train_end]
        X_te    = X_all.iloc[test_start:test_end]
        y_te    = y_all.iloc[test_start:test_end]

        # Direction model on BUY/SELL bars
        bs_tr   = y_tr.isin([1, -1])
        X_bs_tr = X_tr[bs_tr];  y_bs_tr = (y_tr[bs_tr] == 1).astype(int)
        m_dir   = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.75, colsample_bytree=0.65, min_child_weight=15,
            gamma=1.5, reg_alpha=0.15, reg_lambda=1.5,
            objective="binary:logistic", tree_method="hist",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m_dir.fit(X_bs_tr, y_bs_tr)

        # HOLD filter on all bars
        y_nh_tr = (y_tr != 0).astype(int)
        m_hold  = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.06,
            subsample=0.75, colsample_bytree=0.70, min_child_weight=20,
            gamma=2.0, objective="binary:logistic",
            tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
        )
        m_hold.fit(X_tr, y_nh_tr)

        r = _eval_two_stage(X_te, y_te, m_dir, m_hold)
        results.append({**r, "fold": fold + 1, "n_train": train_end})
        print(f"    Fold {fold+1}: dir={r['dir_acc']*100:.1f}%  "
              f"BUY={r['buy_acc']*100:.1f}%  SELL={r['sell_acc']*100:.1f}%  "
              f"combined={r['combined_acc']*100:.1f}%  "
              f"emit={r['n_emit']}/{r['n_total']}")

    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train(symbols: list[str] | None = None, days: int = 58,
          test_mode: bool = False) -> None:
    """Train the two-stage intraday model.

    Stage 1 — Direction (binary BUY/SELL, trained on non-HOLD bars only):
      Trained on bars where future return ≥0.3% or ≤-0.3%.
      Pure binary problem → no confounding from ambiguous HOLD bars.
      Target: BUY accuracy ≥60%, directional accuracy ≥65%.

    Stage 2 — HOLD filter (binary: worth-trading vs HOLD):
      Trained on all bars to identify when to enter.
      Combined signal emitted only when both stages are confident.

    test_mode=True: 8 symbols, models NOT saved.
    """
    if test_mode:
        symbols = ["RELIANCE", "HDFCBANK", "TCS", "INFY", "SBIN",
                   "ICICIBANK", "HINDUNILVR", "BAJFINANCE"]
        print("=" * 62)
        print("TEST TRAIN — 8 symbols · two-stage · 57 features")
        print("=" * 62)
    else:
        symbols = symbols or NIFTY_50
        print("=" * 62)
        print(f"FULL TRAIN — {len(symbols)} symbols · {days}d · two-stage")
        print("=" * 62)

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series]    = []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:2}/{len(symbols)}] {sym}", end=" … ", flush=True)
        df = fetch_5min(sym, days=days)
        if df is None or len(df) < 150:
            print("skipped (no data)"); continue
        try:
            feats  = compute_features(df)
            labels = make_labels(df)
        except Exception as e:
            print(f"skipped ({e})"); continue

        date_col   = df["DateTime"].dt.date
        bar_num    = df.groupby(date_col).cumcount()
        mask = (bar_num >= 6) & labels.notna() & (labels.index < len(labels) - TARGET_BARS) \
               & feats.notna().all(axis=1)

        X = feats[mask][FEATURE_COLS].astype(float)
        y = labels[mask]
        if len(X) < 50:
            print(f"skipped ({len(X)} bars)"); continue
        all_X.append(X); all_y.append(y)
        print(f"{len(X):,} bars  (B:{(y==1).sum()} S:{(y==-1).sum()} H:{(y==0).sum()})")

    if not all_X:
        print("No training data."); return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    n_buy  = (y_all == 1).sum();  n_sell = (y_all == -1).sum()
    n_hold = (y_all == 0).sum()
    print(f"\nTotal: {len(X_all):,} bars  "
          f"BUY={n_buy:,} ({n_buy/len(y_all)*100:.0f}%)  "
          f"SELL={n_sell:,} ({n_sell/len(y_all)*100:.0f}%)  "
          f"HOLD={n_hold:,} ({n_hold/len(y_all)*100:.0f}%)")

    # ── Walk-forward CV ──────────────────────────────────────────────
    print("\n── Walk-forward CV (3-fold, temporal) ──")
    cv_results = walk_forward_cv(X_all, y_all, n_folds=3)
    cv_dir     = np.mean([r["dir_acc"]   for r in cv_results])
    cv_buy     = np.mean([r["buy_acc"]   for r in cv_results])
    cv_sell    = np.mean([r["sell_acc"]  for r in cv_results])
    cv_comb    = np.mean([r["combined_acc"] for r in cv_results])
    print(f"  CV mean → dir={cv_dir*100:.1f}%  BUY={cv_buy*100:.1f}%  "
          f"SELL={cv_sell*100:.1f}%  combined={cv_comb*100:.1f}%")

    # ── Explicit OOT test (train on first 80%, test on last 20%) ─────
    # This simulates deploying today: model never saw last 20% of data.
    print("\n── Out-of-Time (OOT) validation — last 20% strictly held out ──")
    oot_split   = int(len(X_all) * 0.80)
    X_oot_tr    = X_all.iloc[:oot_split];  y_oot_tr = y_all.iloc[:oot_split]
    X_oot_te    = X_all.iloc[oot_split:];  y_oot_te = y_all.iloc[oot_split:]

    bs_oot      = y_oot_tr.isin([1, -1])
    X_bs_oot    = X_oot_tr[bs_oot]
    y_bs_oot    = (y_oot_tr[bs_oot] == 1).astype(int)
    n_bo = (y_bs_oot==1).sum(); n_so = (y_bs_oot==0).sum()
    w_oot = y_bs_oot.apply(lambda v: (n_bo+n_so)/(2*n_bo) if v==1
                            else (n_bo+n_so)/(2*n_so)).values
    m_oot_dir = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.65, min_child_weight=15,
        gamma=1.5, reg_alpha=0.15, reg_lambda=1.5,
        objective="binary:logistic", tree_method="hist",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    m_oot_dir.fit(X_bs_oot, y_bs_oot, sample_weight=w_oot)

    y_nh_oot    = (y_oot_tr != 0).astype(int)
    n_nho = (y_nh_oot==1).sum(); n_ho = (y_nh_oot==0).sum()
    w_nh_oot = y_nh_oot.apply(lambda v: (n_nho+n_ho)/(2*n_nho) if v==1
                               else (n_nho+n_ho)/(2*n_ho)*0.8).values
    m_oot_hold = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.04,
        subsample=0.75, colsample_bytree=0.70, min_child_weight=20,
        gamma=2.5, reg_alpha=0.20, reg_lambda=2.5,
        objective="binary:logistic", tree_method="hist",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    m_oot_hold.fit(X_oot_tr, y_nh_oot, sample_weight=w_nh_oot)

    oot_r = _eval_two_stage(X_oot_te, y_oot_te, m_oot_dir, m_oot_hold)
    print(f"  OOT dir accuracy : {oot_r['dir_acc']*100:.1f}%  (CV mean: {cv_dir*100:.1f}%)")
    print(f"  OOT BUY  recall  : {oot_r['buy_acc']*100:.1f}%  ← target ≥60%")
    print(f"  OOT SELL recall  : {oot_r['sell_acc']*100:.1f}%")
    print(f"  OOT combined     : {oot_r['combined_acc']*100:.1f}%  (on emitted signals)")
    print(f"  OOT emit rate    : {oot_r['n_emit']}/{oot_r['n_total']} "
          f"({oot_r['n_emit']/oot_r['n_total']*100:.1f}%)")
    gap = abs(oot_r['dir_acc'] - cv_dir)
    print(f"  OOT vs CV gap    : {gap*100:.1f}pp  "
          f"{'✓ No overfit' if gap < 0.05 else '⚠ Gap > 5pp — check for overfit'}")

    # ── Final models on 90/10 split ──────────────────────────────────
    split       = int(len(X_all) * 0.90)
    X_tr, X_val = X_all.iloc[:split], X_all.iloc[split:]
    y_tr, y_val = y_all.iloc[:split], y_all.iloc[split:]

    # ── Stage 1: Direction model (BUY/SELL only) ─────────────────────
    bs_mask     = y_tr.isin([1, -1])
    X_bs        = X_tr[bs_mask]
    y_bs        = (y_tr[bs_mask] == 1).astype(int)   # 1=BUY, 0=SELL

    # Balanced BUY/SELL weights for direction model
    n_b = (y_bs == 1).sum(); n_s = (y_bs == 0).sum()
    w_dir = y_bs.apply(lambda v: (n_b + n_s) / (2 * n_b) if v == 1
                       else (n_b + n_s) / (2 * n_s)).values

    print(f"\n── Stage 1: Direction model  "
          f"(train on {len(X_bs):,} BUY/SELL bars) ──")
    model_dir = XGBClassifier(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.020,
        subsample=0.75,
        colsample_bytree=0.65,
        colsample_bylevel=0.80,
        min_child_weight=12,
        gamma=1.2,
        reg_alpha=0.12,
        reg_lambda=1.2,
        eval_metric="logloss",
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=60,
    )
    bs_val  = y_val.isin([1, -1])
    X_bs_v  = X_val[bs_val]
    y_bs_v  = (y_val[bs_val] == 1).astype(int)
    model_dir.fit(X_bs, y_bs, sample_weight=w_dir,
                  eval_set=[(X_bs_v, y_bs_v)], verbose=200)

    dir_preds = model_dir.predict(X_bs_v)
    dir_acc   = float(np.mean(dir_preds == y_bs_v.values))
    buy_acc   = float(np.mean(dir_preds[y_bs_v == 1] == 1)) if (y_bs_v == 1).any() else 0.0
    sell_acc  = float(np.mean(dir_preds[y_bs_v == 0] == 0)) if (y_bs_v == 0).any() else 0.0
    print(f"\nStage 1 val → dir={dir_acc*100:.1f}%  "
          f"BUY-recall={buy_acc*100:.1f}%  SELL-recall={sell_acc*100:.1f}%  "
          f"(random=50%)")
    print(f"Best iteration: {model_dir.best_iteration}")

    print("\n── Stage 1 per-class report ──")
    print(classification_report(y_bs_v, dir_preds,
                                 target_names=["SELL","BUY"], digits=3))

    # ── Stage 2: HOLD filter (all bars) ──────────────────────────────
    print("── Stage 2: HOLD filter  (train on all bars) ──")
    y_nh_tr = (y_tr != 0).astype(int)   # 1=non-HOLD, 0=HOLD
    y_nh_v  = (y_val != 0).astype(int)

    # Upweight non-HOLD bars (they're the minority and the interesting case)
    n_nh  = (y_nh_tr == 1).sum(); n_h = (y_nh_tr == 0).sum()
    w_hold = y_nh_tr.apply(lambda v: (n_nh + n_h) / (2 * n_nh) if v == 1
                            else (n_nh + n_h) / (2 * n_h) * 0.8).values

    model_hold = XGBClassifier(
        n_estimators=900,
        max_depth=4,
        learning_rate=0.025,
        subsample=0.75,
        colsample_bytree=0.70,
        min_child_weight=20,
        gamma=2.0,
        reg_alpha=0.18,
        reg_lambda=2.0,
        eval_metric="logloss",
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=30,
    )
    model_hold.fit(X_tr, y_nh_tr, sample_weight=w_hold,
                   eval_set=[(X_val, y_nh_v)], verbose=200)
    hold_preds = model_hold.predict(X_val)
    hold_acc   = float(np.mean(hold_preds == y_nh_v.values))
    print(f"\nStage 2 val → HOLD-filter acc={hold_acc*100:.1f}%")
    print(f"Best iteration: {model_hold.best_iteration}")

    # ── Combined evaluation ───────────────────────────────────────────
    print("\n── Combined two-stage evaluation ──")
    combined = _eval_two_stage(X_val, y_val, model_dir, model_hold)
    emit_pct  = combined['n_emit'] / combined['n_total'] * 100
    print(f"  Emit rate  : {combined['n_emit']:,}/{combined['n_total']:,} bars "
          f"({emit_pct:.1f}%) — bars where model signals BUY or SELL")
    print(f"  BUY  recall: {combined['buy_acc']*100:.1f}%  ← target ≥60%")
    print(f"  SELL recall: {combined['sell_acc']*100:.1f}%")
    print(f"  Directional: {combined['dir_acc']*100:.1f}%  (50%=random)")
    print(f"  Combined   : {combined['combined_acc']*100:.1f}%  "
          f"(on emitted signals only)")

    # Threshold sensitivity — show how combined accuracy and emit rate trade off
    print("\n── Threshold sensitivity (precision vs coverage) ──")
    hold_proba_all = model_hold.predict_proba(X_val)[:, 1]
    dir_proba_all  = model_dir.predict_proba(X_val)[:, 1]
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        emit   = (hold_proba_all >= thr) & ((dir_proba_all >= thr) | (dir_proba_all <= 1-thr))
        n_e    = emit.sum()
        if n_e == 0:
            print(f"  thr={thr:.2f}: no signals"); continue
        dir_em = (dir_proba_all[emit] >= thr).astype(int)
        true_d = (y_val.values[emit] == 1).astype(int)
        acc_e  = float(np.mean(dir_em == true_d))
        print(f"  thr={thr:.2f}: emit={n_e:4d}/{combined['n_total']:,} "
              f"({n_e/combined['n_total']*100:4.1f}%)  "
              f"accuracy={acc_e*100:.1f}%")

    # Feature importance from direction model (the more discriminating one)
    imp = pd.Series(model_dir.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n── Top-20 direction-model feature importances ──")
    for feat, score in imp.head(20).items():
        bar = "█" * int(score * 180)
        print(f"  {feat:<25} {score:.4f}  {bar}")

    if test_mode:
        print("\n[TEST MODE] Models NOT saved. Remove --test for full train.")
        return

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model":          model_dir,   # kept as "model" for dashboard compat
        "model_dir":      model_dir,
        "model_hold":     model_hold,
        "features":       FEATURE_COLS,
        "val_acc":        dir_acc,
        "binary_acc":     combined["combined_acc"],
        "buy_acc":        combined["buy_acc"],
        "sell_acc":       combined["sell_acc"],
        "cv_dir_acc":     cv_dir,
        "cv_buy_acc":     cv_buy,
        "trained_at":     str(date.today()),
        "n_symbols":      len(all_X),
        "n_samples":      len(X_all),
        "target_bars":    TARGET_BARS,
        "buy_thresh":     BUY_THRESH,
        "feature_imp":    imp.to_dict(),
        "version":        "4",
    }, MODEL_PATH)
    print(f"\nSaved → {MODEL_PATH}")


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(symbol: str) -> dict:
    default = {"signal": "HOLD", "buy_prob": 0.33, "sell_prob": 0.33,
               "hold_prob": 0.34, "confidence": 0.33, "data_ok": False}

    if not MODEL_PATH.exists():
        return {**default, "error": "Model not trained. Run: python intraday_model.py train"}

    blob       = joblib.load(MODEL_PATH)
    model_dir  = blob.get("model_dir",  blob["model"])
    model_hold = blob.get("model_hold", None)
    feat_cols  = blob.get("features", FEATURE_COLS)

    df = fetch_5min(symbol, days=5)
    if df is None or df.empty:
        return default

    try:
        feats = compute_features(df)
    except Exception:
        return default

    valid_cols = [c for c in feat_cols if c in feats.columns]
    valid = feats[valid_cols].dropna()
    if valid.empty:
        return default

    X = valid.tail(1).astype(float)

    # Stage 2: HOLD filter — is this bar worth trading?
    if model_hold is not None:
        hold_proba = float(model_hold.predict_proba(X)[0][1])  # P(non-HOLD)
        if hold_proba < HOLD_THRESHOLD:
            return {
                **default,
                "hold_prob":  round(1.0 - hold_proba, 3),
                "confidence": round(1.0 - hold_proba, 3),
                "data_ok": True,
            }
    else:
        hold_proba = 1.0

    # Stage 1: Direction — BUY or SELL?
    dir_proba = float(model_dir.predict_proba(X)[0][1])  # P(BUY)

    if dir_proba >= DIR_THRESHOLD:
        signal = "BUY";  conf = dir_proba
    elif (1.0 - dir_proba) >= DIR_THRESHOLD:
        signal = "SELL"; conf = 1.0 - dir_proba
    else:
        signal = "HOLD"; conf = max(dir_proba, 1.0 - dir_proba)

    return {
        "signal":     signal,
        "buy_prob":   round(dir_proba, 3),
        "sell_prob":  round(1.0 - dir_proba, 3),
        "hold_prob":  round(1.0 - hold_proba, 3),
        "confidence": round(conf, 3),
        "data_ok":    True,
    }


def batch_predict(symbols: list[str]) -> dict[str, dict]:
    return {sym: predict(sym) for sym in symbols}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd   = sys.argv[1] if len(sys.argv) > 1 else "help"
    flags = set(sys.argv[2:])

    if cmd == "train":
        test_mode = "--test" in flags
        syms = [s for s in sys.argv[2:] if not s.startswith("--")] or None
        train(syms, test_mode=test_mode)

    elif cmd == "predict":
        sym = sys.argv[2] if len(sys.argv) > 2 else "RELIANCE"
        res = predict(sym)
        print(f"\n{'─'*44}\n  {sym} → {res['signal']}\n{'─'*44}")
        for k, v in res.items():
            print(f"  {k:<14} {v}")

    elif cmd == "importance":
        if not MODEL_PATH.exists():
            print("No model. Run: python intraday_model.py train")
        else:
            blob = joblib.load(MODEL_PATH)
            imp  = blob.get("feature_imp",
                            dict(zip(blob.get("features", FEATURE_COLS),
                                     blob["model"].feature_importances_)))
            print(f"\nFeature importances · trained {blob.get('trained_at','?')}")
            print(f"Val 3-class: {blob.get('val_acc',0)*100:.1f}%  "
                  f"Binary-dir: {blob.get('binary_acc',0)*100:.1f}%  "
                  f"CV-binary: {blob.get('cv_binary_acc',0)*100:.1f}%\n{'─'*55}")
            for feat, score in sorted(imp.items(), key=lambda x: -x[1]):
                bar = "█" * int(score * 180)
                print(f"  {feat:<25} {score:.4f}  {bar}")

    elif cmd == "scan":
        syms  = sys.argv[2:] or NIFTY_50[:10]
        preds = batch_predict(syms)
        buys  = [(s, p) for s, p in preds.items() if p["signal"] == "BUY"]
        sells = [(s, p) for s, p in preds.items() if p["signal"] == "SELL"]
        print(f"\n{'─'*50}")
        print(f"  BUY  ({len(buys)})")
        for s, p in sorted(buys,  key=lambda x: -x[1]["buy_prob"]):
            print(f"    {s:<14}  {p['buy_prob']*100:.0f}%")
        print(f"  SELL ({len(sells)})")
        for s, p in sorted(sells, key=lambda x: -x[1]["sell_prob"]):
            print(f"    {s:<14}  {p['sell_prob']*100:.0f}%")
        print(f"{'─'*50}")

    else:
        print(__doc__)
