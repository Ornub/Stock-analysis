"""
intraday_model.py — ML intraday signal engine for NSE stocks.

Implements 15 theories as XGBoost features:

  Original (7):
  1. Opening Range Breakout (ORB)   — Raschke / Crabel: buy/sell the 15-min range break
  2. VWAP Deviation                 — institutional benchmark; price above = buy pressure
  3. Supertrend                     — Wilder ATR-based trend filter (period=7, mult=3)
  4. Central Pivot Range (CPR)      — floor trader pivots from previous day OHLC
  5. EMA 9/21 Cross                 — fast trend alignment on 5-min bars
  6. Bollinger Band Squeeze         — volatility contraction signals breakout
  7. RSI + MACD Momentum            — classic combo: zone + histogram direction

  New (8):
  8.  Market Microstructure         — buy pressure (Williams %R raw), wick rejection signals
  9.  ATR-normalized Bar Quality    — breakout-bar detection, volatility-normalized returns
  10. Gap Dynamics                  — open gap vs prev close, prev-day high/low levels
  11. Stochastic Oscillator         — %K/%D overbought/oversold on 5-min
  12. Heikin-Ashi Trend             — noise-filtered candle direction + consecutive streak
  13. Time-of-Day Structure         — first-30-min ORB zone, power-hour, cyclic encoding
  14. Volume Acceleration           — vol vs prior 3-bar avg; detects thrust before moves
  15. Candlestick Patterns          — engulfing + inside bar (volatility contraction)

Label (3-class):
  BUY  — close rises ≥ 0.3% within next 6 bars (30 min)
  SELL — close falls ≥ 0.3% within next 6 bars
  HOLD — neither

CLI:
  python intraday_model.py train              # train on full Nifty-50
  python intraday_model.py train RELIANCE TCS # train on specific symbols
  python intraday_model.py predict RELIANCE   # predict current signal
  python intraday_model.py importance         # print top feature importances
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from swing_v2 import NIFTY_50

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = Path("models/intraday_v1.pkl")
TARGET_BARS = 6       # 6 × 5-min = 30 min lookahead
BUY_THRESH  = 0.003   # +0.3% → BUY
SELL_THRESH = -0.003  # −0.3% → SELL
MIN_BUY_PROB = 0.40   # minimum probability to emit a BUY/SELL signal

FEATURE_COLS = [
    # ── Theory 1: ORB ────────────────────────────────────────────────
    "orb_signal", "morning_range_pos",
    # ── Theory 2: VWAP ───────────────────────────────────────────────
    "vwap_pct", "vwap_dev_norm",
    # ── Theory 3: Supertrend ─────────────────────────────────────────
    "supertrend_sig",
    # ── Theory 4: CPR ────────────────────────────────────────────────
    "cpr_pos",
    # ── Theory 5: EMA cross ──────────────────────────────────────────
    "ema9_21_sig", "ema21_50_sig",
    # ── Theory 6: Bollinger ──────────────────────────────────────────
    "bb_pct_b", "bb_squeeze", "bb_width",
    # ── Theory 7: RSI + MACD ─────────────────────────────────────────
    "rsi", "rsi_zone", "macd_hist_norm", "macd_cross",
    # ── Theory 8: Market Microstructure ──────────────────────────────
    "buy_pressure", "upper_wick_pct", "lower_wick_pct",
    # ── Theory 9: ATR-normalized bar quality ─────────────────────────
    "bar_atr_ratio", "ret_1bar_atr", "ret_3bar", "ret_6bar",
    # ── Theory 10: Gap dynamics + prev-day levels ─────────────────────
    "gap_pct", "above_pd_high", "dist_pd_high",
    # ── Theory 11: Stochastic ────────────────────────────────────────
    "stoch_k", "stoch_d",
    # ── Theory 12: Heikin-Ashi ───────────────────────────────────────
    "ha_color", "ha_streak",
    # ── Theory 13: Time-of-day ───────────────────────────────────────
    "is_first_30min", "is_power_hour", "time_sin", "time_cos",
    # ── Theory 14: Volume acceleration ───────────────────────────────
    "vol_ratio", "vol_accel",
    # ── Theory 15: Candlestick patterns ──────────────────────────────
    "engulfing", "inside_bar", "bar_body_pct",
    # ── Session + streak ─────────────────────────────────────────────
    "session_pct", "consec_dir",
]


# ── Data fetch ────────────────────────────────────────────────────────────────

def _yf_sym(sym: str) -> str:
    return f"{sym}.NS"


def fetch_5min(symbol: str, days: int = 58) -> pd.DataFrame | None:
    """Fetch up to `days` days of 5-min OHLCV from yfinance."""
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


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 7, mult: float = 3.0) -> pd.Series:
    """Supertrend: +1 = bullish, -1 = bearish."""
    atr     = _atr(high, low, close, period)
    hl2     = (high + low) / 2
    upper   = (hl2 + mult * atr).values.copy()
    lower   = (hl2 - mult * atr).values.copy()
    close_v = close.values
    n       = len(close_v)

    sig = np.ones(n)
    st  = lower.copy()

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
    """Return (vwap, vwap_std) both resetting each calendar day."""
    tp     = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv    = tp * df["Volume"]
    date_col = df["DateTime"].dt.date

    vwap     = pd.Series(np.nan, index=df.index)
    vwap_std = pd.Series(np.nan, index=df.index)

    for d, grp in df.groupby(date_col):
        idx      = grp.index
        cum_tpv  = tpv.loc[idx].cumsum()
        cum_vol  = df.loc[idx, "Volume"].cumsum().replace(0, np.nan)
        cum_tp2v = (tp.loc[idx] ** 2 * df.loc[idx, "Volume"]).cumsum()

        vwap_d = cum_tpv / cum_vol
        var_d  = (cum_tp2v / cum_vol) - vwap_d ** 2
        std_d  = var_d.clip(lower=0).apply(np.sqrt)

        vwap.loc[idx]     = vwap_d
        vwap_std.loc[idx] = std_d

    return vwap, vwap_std


def _bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    """Return (pct_b, squeeze, bandwidth). Squeeze = bandwidth < 50-bar avg."""
    mid    = close.rolling(period).mean()
    sigma  = close.rolling(period).std(ddof=0)
    upper  = mid + std * sigma
    lower  = mid - std * sigma
    pct_b  = (close - lower) / (upper - lower).replace(0, np.nan)
    bwidth = (sigma / mid.replace(0, np.nan)).fillna(0)
    squeeze = (bwidth < bwidth.rolling(50).mean()).astype(float)
    return pct_b, squeeze, bwidth


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    """Traditional stochastic %K and %D, normalised 0-1."""
    lo_n  = low.rolling(period).min()
    hi_n  = high.rolling(period).max()
    raw_k = (close - lo_n) / (hi_n - lo_n).replace(0, np.nan) * 100
    k     = raw_k.rolling(smooth_k).mean()
    d     = k.rolling(smooth_d).mean()
    return (k / 100).clip(0, 1), (d / 100).clip(0, 1)


def _heikin_ashi(open_: pd.Series, high: pd.Series,
                 low: pd.Series, close: pd.Series):
    """Return (ha_close, ha_open) for Heikin-Ashi candles."""
    ha_close = (open_ + high + low + close) / 4
    ha_open  = pd.Series(np.nan, index=open_.index)
    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(open_)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    return ha_close, ha_open


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame aligned with df's index (37 features)."""
    df = df.sort_values("DateTime").reset_index(drop=True)
    date_col = df["DateTime"].dt.date

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"]

    out = pd.DataFrame(index=df.index)

    # ── Theory 3: Supertrend ─────────────────────────────────────────
    out["supertrend_sig"] = _supertrend(high, low, close)

    # ── Theory 5: EMA cross ─────────────────────────────────────────
    ema9  = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    out["ema9_21_sig"]  = np.sign(ema9  - ema21).fillna(0)
    out["ema21_50_sig"] = np.sign(ema21 - ema50).fillna(0)

    # ── Theory 7a: RSI ───────────────────────────────────────────────
    rsi_raw = _rsi(close, 14)
    out["rsi"]      = (rsi_raw / 100).clip(0, 1).fillna(0.5)
    out["rsi_zone"] = pd.cut(rsi_raw, bins=[-1, 30, 70, 101],
                             labels=[0.0, 1.0, 2.0]).astype(float).fillna(1.0)

    # ── Theory 7b: MACD ──────────────────────────────────────────────
    macd_line = _ema(close, 12) - _ema(close, 26)
    macd_sig  = _ema(macd_line, 9)
    macd_hist = macd_line - macd_sig
    roll_std  = close.rolling(20).std(ddof=0).replace(0, np.nan)
    out["macd_hist_norm"] = (macd_hist / roll_std).clip(-3, 3).fillna(0)
    out["macd_cross"] = (
        ((macd_line > macd_sig) & (macd_line.shift() <= macd_sig.shift())).astype(float)
        - ((macd_line < macd_sig) & (macd_line.shift() >= macd_sig.shift())).astype(float)
    ).fillna(0)

    # ── Theory 6: Bollinger Bands ────────────────────────────────────
    bb_pct_b, bb_squeeze, bb_width = _bollinger(close, 20, 2.0)
    out["bb_pct_b"]   = bb_pct_b.clip(-0.5, 1.5).fillna(0.5)
    out["bb_squeeze"] = bb_squeeze.fillna(0)
    out["bb_width"]   = bb_width.clip(0, 0.05).fillna(0)

    # ── Theory 9: ATR-normalized bar quality ─────────────────────────
    atr14 = _atr(high, low, close, 14)
    bar_range = (high - low)
    out["bar_atr_ratio"] = (bar_range / atr14.replace(0, np.nan)).clip(0, 5).fillna(1)
    out["ret_1bar_atr"]  = (close.pct_change(1) / (atr14 / close).replace(0, np.nan)).clip(-5, 5).fillna(0)
    out["ret_3bar"]      = close.pct_change(3).clip(-0.05, 0.05).fillna(0)
    out["ret_6bar"]      = close.pct_change(6).clip(-0.08, 0.08).fillna(0)

    # ── Theory 11: Stochastic ────────────────────────────────────────
    stoch_k, stoch_d = _stochastic(high, low, close, 14, 3, 3)
    out["stoch_k"] = stoch_k.fillna(0.5)
    out["stoch_d"] = stoch_d.fillna(0.5)

    # ── Theory 12: Heikin-Ashi ───────────────────────────────────────
    ha_close, ha_open = _heikin_ashi(open_, high, low, close)
    ha_dir = np.sign(ha_close - ha_open).fillna(0)
    out["ha_color"] = ha_dir

    ha_streak_arr = np.zeros(len(ha_dir))
    cur_streak = 0.0
    for i, d in enumerate(ha_dir):
        if np.isnan(d) or d == 0:
            cur_streak = 0.0
        elif d == np.sign(cur_streak) or cur_streak == 0:
            cur_streak += d
        else:
            cur_streak = d
        ha_streak_arr[i] = cur_streak
    out["ha_streak"] = np.clip(ha_streak_arr, -10, 10)

    # ── Theory 14: Volume acceleration ──────────────────────────────
    out["vol_ratio"]  = (volume / volume.rolling(20).mean().replace(0, np.nan)).clip(0, 10).fillna(1)
    out["vol_accel"]  = (volume / volume.rolling(3).mean().shift(1).replace(0, np.nan)).clip(0, 8).fillna(1)

    # ── Theory 15: Candlestick patterns ──────────────────────────────
    body     = close - open_
    hl_range = (high - low).replace(0, np.nan)
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low

    # Theory 8: Market microstructure
    out["buy_pressure"]   = ((close - low) / hl_range).clip(0, 1).fillna(0.5)
    out["upper_wick_pct"] = (upper_wick / hl_range).clip(0, 1).fillna(0)
    out["lower_wick_pct"] = (lower_wick / hl_range).clip(0, 1).fillna(0)
    out["bar_body_pct"]   = (body.abs() / hl_range).clip(0, 1).fillna(0.5)

    # Engulfing: +1 bullish (body engulfs prior body), -1 bearish, 0 none
    prev_body = body.shift(1)
    bull_eng  = ((body > 0) & (open_ <= close.shift(1)) & (close >= open_.shift(1)) & (body.abs() > prev_body.abs()))
    bear_eng  = ((body < 0) & (open_ >= close.shift(1)) & (close <= open_.shift(1)) & (body.abs() > prev_body.abs()))
    out["engulfing"] = bull_eng.astype(float) - bear_eng.astype(float)

    # Inside bar: H < prior H AND L > prior L (volatility contraction)
    out["inside_bar"] = (
        (high < high.shift(1)) & (low > low.shift(1))
    ).astype(float).fillna(0)

    # Consecutive direction streak
    direction = np.sign(close.values - close.shift().values)
    streak    = np.zeros(len(direction))
    cur       = 0.0
    for i in range(len(direction)):
        d = direction[i]
        if np.isnan(d) or d == 0:
            cur = 0.0
        elif d == np.sign(cur) or cur == 0:
            cur = cur + d
        else:
            cur = d
        streak[i] = cur
    out["consec_dir"] = np.clip(streak, -10, 10)

    # ── Per-day features (ORB, VWAP, CPR, gap, prev-day, session, time) ──
    vwap_all, vwap_std_all = _vwap_daily(df)

    orb_signal_s    = pd.Series(0.0,  index=df.index)
    morning_range_s = pd.Series(0.5,  index=df.index)
    vwap_pct_s      = pd.Series(0.0,  index=df.index)
    vwap_dev_norm_s = pd.Series(0.0,  index=df.index)
    cpr_pos_s       = pd.Series(0.0,  index=df.index)
    session_pct_s   = pd.Series(0.5,  index=df.index)
    gap_pct_s       = pd.Series(0.0,  index=df.index)
    above_pd_high_s = pd.Series(0.0,  index=df.index)
    dist_pd_high_s  = pd.Series(0.0,  index=df.index)
    is_first_30_s   = pd.Series(0.0,  index=df.index)
    is_power_hr_s   = pd.Series(0.0,  index=df.index)
    time_sin_s      = pd.Series(0.0,  index=df.index)
    time_cos_s      = pd.Series(0.0,  index=df.index)

    prev_ohlc: dict = {}

    for d, grp in df.groupby(date_col):
        idx = grp.index
        n   = len(grp)
        cur = grp["Close"].values
        h_g = grp["High"].values
        l_g = grp["Low"].values

        # Theory 1: ORB (first 3 bars = 15 min)
        orb_h = grp["High"].iloc[:min(3, n)].max()
        orb_l = grp["Low"].iloc[:min(3, n)].min()
        rng   = orb_h - orb_l
        if rng > 0:
            pos = np.clip((cur - orb_l) / rng, 0, 1)
            morning_range_s.loc[idx] = pos
        orb_signal_s.loc[idx] = (
            (cur > orb_h).astype(float) - (cur < orb_l).astype(float)
        )

        # Theory 2: VWAP deviation (% and normalized by std dev)
        vwap_day     = vwap_all.loc[idx].values
        vwap_std_day = vwap_std_all.loc[idx].values
        vwap_pct_s.loc[idx]      = np.clip((cur / np.where(vwap_day > 0, vwap_day, np.nan) - 1) * 100, -5, 5)
        vwap_dev_norm_s.loc[idx] = np.clip(
            (cur - vwap_day) / np.where(vwap_std_day > 0, vwap_std_day, np.nan),
            -4, 4,
        )

        # Theory 4: CPR from previous day
        if prev_ohlc:
            ph, pl, pc = list(prev_ohlc.values())[-1]
            pivot   = (ph + pl + pc) / 3
            bc      = (ph + pl) / 2
            tc      = 2 * pivot - bc
            cpr_mid = (tc + bc) / 2
            if cpr_mid > 0:
                cpr_pos_s.loc[idx] = np.clip((cur / cpr_mid - 1) * 100, -5, 5)

            # Theory 10: Gap dynamics + previous day levels
            prev_close = pc
            today_open = grp["Open"].iloc[0]
            gap = (today_open / prev_close - 1) * 100 if prev_close > 0 else 0
            gap_pct_s.loc[idx] = np.clip(gap, -5, 5)

            above_pd_high_s.loc[idx] = (
                (cur > ph).astype(float) - (cur < pl).astype(float)
            )
            dist_pd_high_s.loc[idx] = np.clip(
                (cur / ph - 1) * 100 if ph > 0 else 0, -5, 5
            )

        # Theory 13: Time-of-day structure
        bar_nums = np.arange(n)
        # NSE: 09:15–15:30 = 75 bars of 5-min
        TOTAL_BARS = 75
        is_first_30_s.loc[idx] = (bar_nums < 6).astype(float)    # first 30 min
        is_power_hr_s.loc[idx] = (bar_nums >= TOTAL_BARS - 15).astype(float)  # last 75 min
        # Cyclic time encoding (maps bar position to circle, captures cyclical patterns)
        angle = bar_nums / max(n - 1, 1) * 2 * np.pi
        time_sin_s.loc[idx] = np.sin(angle)
        time_cos_s.loc[idx] = np.cos(angle)

        # Session position 0→1
        session_pct_s.loc[idx] = bar_nums / max(n - 1, 1)

        prev_ohlc[d] = (
            grp["High"].max(),
            grp["Low"].min(),
            grp["Close"].iloc[-1],
        )

    out["orb_signal"]        = orb_signal_s
    out["morning_range_pos"] = morning_range_s
    out["vwap_pct"]          = vwap_pct_s
    out["vwap_dev_norm"]     = vwap_dev_norm_s.fillna(0)
    out["cpr_pos"]           = cpr_pos_s
    out["session_pct"]       = session_pct_s
    out["gap_pct"]           = gap_pct_s
    out["above_pd_high"]     = above_pd_high_s
    out["dist_pd_high"]      = dist_pd_high_s
    out["is_first_30min"]    = is_first_30_s
    out["is_power_hour"]     = is_power_hr_s
    out["time_sin"]          = time_sin_s
    out["time_cos"]          = time_cos_s

    return out[FEATURE_COLS]


def make_labels(df: pd.DataFrame) -> pd.Series:
    """3-class: 1=BUY, -1=SELL, 0=HOLD based on 30-min forward return."""
    fwd = df["Close"].shift(-TARGET_BARS) / df["Close"] - 1
    lbl = pd.Series(0, index=df.index, dtype=int)
    lbl[fwd >=  BUY_THRESH]  = 1
    lbl[fwd <= SELL_THRESH] = -1
    return lbl


# ── Training ──────────────────────────────────────────────────────────────────

def train(symbols: list[str] | None = None, days: int = 58) -> None:
    symbols = symbols or NIFTY_50
    print(f"Training intraday model on {len(symbols)} symbols · {days}d of 5-min data")
    print(f"15 theories → {len(FEATURE_COLS)} features · label = ±0.3% in 30 min\n")

    all_X, all_y = [], []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:2}/{len(symbols)}] {sym}", end=" … ", flush=True)
        df = fetch_5min(sym, days=days)
        if df is None or len(df) < 150:
            print("skipped (no data)")
            continue

        try:
            feats  = compute_features(df)
            labels = make_labels(df)
        except Exception as e:
            print(f"skipped ({e})")
            continue

        # Exclude first 6 bars per day (ORB context not yet established)
        date_col = df["DateTime"].dt.date
        bar_num  = df.groupby(date_col).cumcount()
        context_ok = bar_num >= 6

        # Exclude last TARGET_BARS rows (no forward label) and NaN rows
        label_ok = labels.notna() & (labels.index < len(labels) - TARGET_BARS)
        feat_ok  = feats.notna().all(axis=1)
        mask     = context_ok & label_ok & feat_ok

        X = feats[mask][FEATURE_COLS].astype(float)
        y = labels[mask]
        if len(X) < 50:
            print(f"skipped (only {len(X)} bars after filtering)")
            continue
        all_X.append(X)
        all_y.append(y)
        print(f"{len(X):,} bars  (B:{(y==1).sum()} S:{(y==-1).sum()} H:{(y==0).sum()})")

    if not all_X:
        print("No training data collected.")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    # Map labels: -1→0 (SELL), 0→1 (HOLD), 1→2 (BUY) for XGBoost multi-class
    y_xgb = y_all.map({-1: 0, 0: 1, 1: 2})

    # Balanced class weights
    counts = y_all.value_counts()
    total  = len(y_all)
    w_map  = {v: total / (len(counts) * counts[v]) for v in counts.index}
    weights = y_all.map(w_map).values

    print(f"\nTotal: {len(X_all):,} bars  "
          f"SELL={(y_all==-1).sum():,}  HOLD={(y_all==0).sum():,}  BUY={(y_all==1).sum():,}")

    # Time-ordered train/val split (no shuffling — preserves temporal integrity)
    split  = int(len(X_all) * 0.88)
    X_tr, X_val = X_all.iloc[:split], X_all.iloc[split:]
    y_tr, y_val = y_xgb.iloc[:split], y_xgb.iloc[split:]
    w_tr        = weights[:split]

    model = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.70,
        colsample_bylevel=0.85,
        min_child_weight=20,
        gamma=1.5,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="mlogloss",
        tree_method="hist",
        num_class=3,
        objective="multi:softprob",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=30,
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    val_preds = model.predict(X_val)
    val_acc   = (val_preds == y_val.values).mean()

    print(f"\nOverall val accuracy: {val_acc*100:.1f}%  (random baseline: 33.3%)")
    print(f"Best iteration: {model.best_iteration}")
    for lbl, name in [(0, "SELL"), (1, "HOLD"), (2, "BUY")]:
        mask_c = y_val == lbl
        if mask_c.sum() > 0:
            acc_c = (val_preds[mask_c] == lbl).mean()
            print(f"  {name} accuracy: {acc_c*100:.1f}%  (n={mask_c.sum()})")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    print(f"\nTop-15 feature importances:")
    for feat, score in imp.head(15).items():
        bar = "█" * int(score * 200)
        print(f"  {feat:<25} {score:.4f}  {bar}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model":        model,
        "features":     FEATURE_COLS,
        "val_acc":      val_acc,
        "trained_at":   str(date.today()),
        "n_symbols":    len(all_X),
        "n_samples":    len(X_all),
        "target_bars":  TARGET_BARS,
        "buy_thresh":   BUY_THRESH,
        "feature_imp":  imp.to_dict(),
    }, MODEL_PATH)
    print(f"\nSaved → {MODEL_PATH}")


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(symbol: str) -> dict:
    """Return intraday BUY/SELL/HOLD signal for a single symbol."""
    default = {
        "signal": "HOLD", "buy_prob": 0.33, "sell_prob": 0.33,
        "hold_prob": 0.34, "confidence": 0.33, "data_ok": False,
    }

    if not MODEL_PATH.exists():
        return {**default, "error": "Model not trained. Run: python intraday_model.py train"}

    blob  = joblib.load(MODEL_PATH)
    model: XGBClassifier = blob["model"]
    feat_cols = blob.get("features", FEATURE_COLS)

    df = fetch_5min(symbol, days=5)
    if df is None or df.empty:
        return default

    try:
        feats = compute_features(df)
    except Exception:
        return default

    # Align to saved feature set (handles version upgrades gracefully)
    valid_cols = [c for c in feat_cols if c in feats.columns]
    valid = feats[valid_cols].dropna()
    if valid.empty:
        return default

    X     = valid.tail(1).astype(float)
    proba = model.predict_proba(X)[0]   # [sell_prob, hold_prob, buy_prob]
    sell_p, hold_p, buy_p = float(proba[0]), float(proba[1]), float(proba[2])

    if buy_p >= sell_p and buy_p >= hold_p and buy_p >= MIN_BUY_PROB:
        signal = "BUY";  conf = buy_p
    elif sell_p >= buy_p and sell_p >= hold_p and sell_p >= MIN_BUY_PROB:
        signal = "SELL"; conf = sell_p
    else:
        signal = "HOLD"; conf = hold_p

    return {
        "signal":     signal,
        "buy_prob":   round(buy_p,  3),
        "sell_prob":  round(sell_p, 3),
        "hold_prob":  round(hold_p, 3),
        "confidence": round(conf,   3),
        "data_ok":    True,
    }


def batch_predict(symbols: list[str]) -> dict[str, dict]:
    """Predict intraday signal for multiple symbols."""
    return {sym: predict(sym) for sym in symbols}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "train":
        syms = sys.argv[2:] or None
        train(syms)

    elif cmd == "predict":
        sym = sys.argv[2] if len(sys.argv) > 2 else "RELIANCE"
        res = predict(sym)
        print(f"\n{'─'*44}")
        print(f"  {sym} intraday signal: {res['signal']}")
        print(f"{'─'*44}")
        for k, v in res.items():
            print(f"  {k:<14} {v}")
        print(f"{'─'*44}")

    elif cmd == "importance":
        if not MODEL_PATH.exists():
            print("Model not found. Run: python intraday_model.py train")
        else:
            blob = joblib.load(MODEL_PATH)
            imp  = blob.get("feature_imp", {})
            if imp:
                print(f"\nFeature importances ({blob.get('trained_at','?')}):\n{'─'*50}")
                for feat, score in sorted(imp.items(), key=lambda x: -x[1]):
                    bar = "█" * int(score * 200)
                    print(f"  {feat:<25} {score:.4f}  {bar}")
            else:
                model = blob["model"]
                imp   = dict(zip(blob.get("features", FEATURE_COLS), model.feature_importances_))
                for feat, score in sorted(imp.items(), key=lambda x: -x[1]):
                    print(f"  {feat:<25} {score:.4f}")

    elif cmd == "scan":
        syms  = sys.argv[2:] or NIFTY_50[:10]
        preds = batch_predict(syms)
        buys  = [(s, p) for s, p in preds.items() if p["signal"] == "BUY"]
        sells = [(s, p) for s, p in preds.items() if p["signal"] == "SELL"]
        print(f"\n{'─'*50}")
        print(f"  BUY signals  ({len(buys)})")
        for s, p in sorted(buys, key=lambda x: -x[1]["buy_prob"]):
            print(f"    {s:<14}  {p['buy_prob']*100:.0f}% conf")
        print(f"  SELL signals ({len(sells)})")
        for s, p in sorted(sells, key=lambda x: -x[1]["sell_prob"]):
            print(f"    {s:<14}  {p['sell_prob']*100:.0f}% conf")
        print(f"{'─'*50}")

    else:
        print(__doc__)
