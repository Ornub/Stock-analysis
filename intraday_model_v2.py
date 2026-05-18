"""
intraday_model_v2.py — Clean-slate intraday ML engine for NSE 5-min data.

V2 design principles (learned from v1 failures):
  1. Only 22 features — all non-lagging or structural (no slow RSI/MACD/Bollinger)
  2. nifty_day_ret: Nifty's cumulative return from yesterday's close → today
     This is the strongest missing signal: bull-day vs bear-day context
  3. rel_strength_day: how this stock is doing vs Nifty TODAY (intraday alpha)
  4. session_trend: ATR-normalised move from session open — momentum direction
  5. LightGBM ensemble (3 diverse seeds) — better generalisation than XGBoost
  6. Two-stage architecture preserved (HOLD filter + direction model)
  7. Per-symbol 70/30 temporal split — no contamination
  8. Time-decay sample weights (recent bars 3× more important)
  9. BUY_CLASS_THRESH = 0.47 to counter bearish-regime SELL bias

CLI:
  python intraday_model_v2.py train          # full 44-symbol train
  python intraday_model_v2.py train --test   # quick 8-symbol test
  python intraday_model_v2.py predict SYMBOL
  python intraday_model_v2.py importance
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import classification_report, precision_score

sys.path.insert(0, str(Path(__file__).parent))
from swing_v2 import NIFTY_50

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH_V2  = Path("models/intraday_v2.pkl")
TARGET_BARS    = 8       # 8 × 5-min = 40-min lookahead [v3.1: wider window, higher quality labels]
BUY_THRESH     = 0.007   # +0.7% high-water mark → BUY [v3.1: tighter — filter noise]
SELL_THRESH    = -0.007  # −0.7% high-water mark → SELL
DIR_THRESHOLD  = 0.67    # emit BUY/SELL if confidence ≥ this [v3.1: raised from 0.65]
HOLD_THRESHOLD = 0.52
BUY_CLASS_THRESH = 0.48  # neutral threshold for BUY vs SELL classification
# 70/30 per-symbol temporal split (for honest test reporting).
# Early stopping uses the last 15% of the CONCATENATED training pool —
# avoids the "5-day single-symbol bias" where one bullish window gives
# 58% BUY in the ES set, making the model stop at iteration 1.
TRAIN_FRAC = 0.70   # per-symbol: first 70% → training pool, last 30% → test
ES_POOL_FRAC = 0.85 # within training pool: first 85% → actual fit, last 15% → ES val

FEATURE_COLS = [
    # ── Nifty regime / daily context (most important cluster) ────────
    "nifty_day_ret",        # Nifty cumulative return today vs yesterday's close
    "nifty_ret_6bar",       # Nifty 30-min momentum
    "nifty_rsi",            # Nifty RSI(14) — trend maturity
    "nifty_adx",            # Nifty trend strength
    "nifty_ema_sig",        # Nifty above/below 21-EMA (+1/-1)
    "rel_strength_day",     # stock intraday return − Nifty intraday return
    # ── Timing / session context ──────────────────────────────────────
    "is_power_hour",        # last 75 min of session
    "session_pct",          # 0→1 over the session
    "time_cos",
    "time_sin",
    # ── Opening dynamics ─────────────────────────────────────────────
    "gap_pct",              # opening gap %
    "gap_atr_norm",         # gap / ATR — gap size adjusted for volatility
    "opening_drive",        # first-15-min direction × ATR magnitude
    "prev_close_pos",       # where yesterday closed in its range (0=low, 1=high)
    # ── Session position ─────────────────────────────────────────────
    "vwap_pct",             # % from daily VWAP
    "dist_pd_high",         # % from previous day's high
    "session_high_dist",    # % from session high (≤ 0)
    # ── Intraday momentum ────────────────────────────────────────────
    "ret_3bar",             # 15-min return (ATR normalised)
    "macd_fast_hist",       # fast 3/8-bar MACD histogram
    "session_trend",        # ATR-normalised drift from session open
    "buy_pressure",         # (close-low)/(high-low) — bar buying pressure
    # ── Volume ───────────────────────────────────────────────────────
    "rvol_tod",             # volume vs historical same-time-slot average
    # ── v3.0 additions ───────────────────────────────────────────────
    "nifty_ret_3bar",       # Nifty 15-min momentum — faster pivot detection
    "session_vol_accel",    # volume EMA3/EMA10 ratio — accumulation acceleration
    # ── v3.1 additions ───────────────────────────────────────────────
    "stock_rsi_5m",         # RSI(9) of stock on 5-min bars — individual momentum
    "bb_squeeze",           # BB(20) width / ATR(14) — low = volatility squeeze
    "stock_ema_align",      # EMA9>EMA21>EMA55 alignment score (0→3, scaled /3)
]

assert len(FEATURE_COLS) == 27

# ── Data fetch ────────────────────────────────────────────────────────────────

_NIFTY_CACHE: dict = {}


def _yf_sym(sym: str) -> str:
    return f"{sym}.NS"


def fetch_5min(symbol: str, days: int = 90) -> pd.DataFrame | None:
    try:
        tk = yf.Ticker(_yf_sym(symbol))
        fetch_days = min(days, 60)  # yfinance 5-min data capped at 60 days
        df = tk.history(period=f"{fetch_days}d", interval="5m", auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.rename(columns={"Datetime": "DateTime"}, inplace=True)
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
        return df[["DateTime", "Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as e:
        print(f"[WARN] fetch_5min({symbol}): {e}")
        return None


def fetch_nifty_5min(days: int = 92) -> pd.DataFrame | None:
    key = f"nifty_{days}"
    if key in _NIFTY_CACHE:
        return _NIFTY_CACHE[key]
    try:
        tk = yf.Ticker("^NSEI")
        fetch_days = min(days, 60)  # yfinance 5-min data capped at 60 days
        df = tk.history(period=f"{fetch_days}d", interval="5m", auto_adjust=True)
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


# ── Indicator helpers ─────────────────────────────────────────────────────────

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


def _adx_close_only(close: pd.Series, period: int = 14) -> pd.Series:
    """ADX approximation from close-only series (for Nifty index)."""
    atr_c   = close.diff().abs().ewm(span=period, adjust=False).mean()
    plus_dm = close.diff().clip(lower=0)
    minus_dm = (-close.diff()).clip(lower=0)
    plus_di  = (plus_dm.ewm(span=period, adjust=False).mean()
                / atr_c.replace(0, np.nan) * 100).fillna(0)
    minus_di = (minus_dm.ewm(span=period, adjust=False).mean()
                / atr_c.replace(0, np.nan) * 100).fillna(0)
    dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100).fillna(0)
    return (dx.ewm(span=period, adjust=False).mean() / 100).clip(0, 1)


def _vwap_daily(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    tp   = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv  = tp * df["Volume"]
    tp2v = tp ** 2 * df["Volume"]
    date_col = df["DateTime"].dt.date
    vwap = pd.Series(np.nan, index=df.index)
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


# ── Labels ────────────────────────────────────────────────────────────────────

def make_labels(df: pd.DataFrame) -> pd.Series:
    """Path-aware high-water mark labels over next TARGET_BARS bars."""
    close_v = df["Close"].values.astype(float)
    high_v  = df["High"].values.astype(float)
    low_v   = df["Low"].values.astype(float)
    n, m    = len(close_v), len(close_v) - TARGET_BARS
    lbl     = np.zeros(n, dtype=int)
    decided = np.zeros(m, dtype=bool)
    idx     = np.arange(m)
    for k in range(1, TARGET_BARS + 1):
        fi    = idx + k
        ref   = close_v[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_h = np.where(ref > 0, high_v[fi] / ref - 1, 0.0)
            ret_l = np.where(ref > 0, low_v[fi]  / ref - 1, 0.0)
        still = ~decided
        hit_b = still & (ret_h >= BUY_THRESH)
        hit_s = still & (ret_l <= SELL_THRESH)
        both  = hit_b & hit_s
        hit_b = hit_b & ~(both & (ret_h < np.abs(ret_l)))
        hit_s = hit_s & ~(both & (ret_h >= np.abs(ret_l)))
        lbl[idx[hit_b]] = 1
        lbl[idx[hit_s]] = -1
        decided        |= hit_b | hit_s
    return pd.Series(lbl, index=df.index)


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return 27-feature DataFrame aligned to df's index."""
    df       = df.sort_values("DateTime").reset_index(drop=True)
    date_col = df["DateTime"].dt.date
    close    = df["Close"]
    high     = df["High"]
    low      = df["Low"]
    volume   = df["Volume"]
    open_    = df["Open"]
    out      = pd.DataFrame(index=df.index)
    n        = len(df)

    # ── Whole-series computations ────────────────────────────────────
    atr14     = _atr(high, low, close, 14)
    vwap_all, _ = _vwap_daily(df)

    # Fast 3/8 MACD
    fast_macd = _ema(close, 3) - _ema(close, 8)
    fast_sig  = _ema(fast_macd, 5)
    roll_std  = close.rolling(20).std(ddof=0).replace(0, np.nan)
    out["macd_fast_hist"] = ((fast_macd - fast_sig) / roll_std).clip(-3, 3).fillna(0)

    # 3-bar return
    out["ret_3bar"] = close.pct_change(3).clip(-0.05, 0.05).fillna(0)

    # Bar buying pressure
    hl_range = (high - low).replace(0, np.nan)
    out["buy_pressure"] = ((close - low) / hl_range).clip(0, 1).fillna(0.5)

    # ── v3.1 features ─────────────────────────────────────────────────
    # stock_rsi_5m: RSI(9) of the stock on 5-min bars (most important missing signal)
    out["stock_rsi_5m"] = (_rsi(close, 9) / 100).clip(0, 1).fillna(0.5)

    # bb_squeeze: Bollinger Band(20) width / ATR(14) — squeeze = low value = breakout pending
    bb_mid  = close.rolling(20).mean()
    bb_std  = close.rolling(20).std(ddof=0).replace(0, np.nan)
    bb_width = (4 * bb_std) / close.replace(0, np.nan)   # (upper-lower)/close
    atr_pct  = atr14 / close.replace(0, np.nan)
    out["bb_squeeze"] = (bb_width / atr_pct.replace(0, np.nan)).clip(0, 5).fillna(2.0)

    # stock_ema_align: count of alignments EMA9>EMA21, EMA21>EMA55, price>EMA9 (scaled 0→1)
    ema9  = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema55 = _ema(close, 55)
    align = ((close > ema9).astype(float) +
             (ema9   > ema21).astype(float) +
             (ema21  > ema55).astype(float)) / 3.0
    out["stock_ema_align"] = align.fillna(0.5)

    # ── Per-day features ─────────────────────────────────────────────
    is_power_hr_s     = pd.Series(0.0, index=df.index)
    session_pct_s     = pd.Series(0.5, index=df.index)
    time_sin_s        = pd.Series(0.0, index=df.index)
    time_cos_s        = pd.Series(0.0, index=df.index)
    vwap_pct_s        = pd.Series(0.0, index=df.index)
    gap_pct_s         = pd.Series(0.0, index=df.index)
    gap_atr_norm_s    = pd.Series(0.0, index=df.index)
    dist_pd_high_s    = pd.Series(0.0, index=df.index)
    prev_close_pos_s  = pd.Series(0.5, index=df.index)
    session_hi_dist_s = pd.Series(0.0, index=df.index)
    opening_drive_s   = pd.Series(0.0, index=df.index)
    session_trend_s   = pd.Series(0.0, index=df.index)
    rvol_tod_s        = pd.Series(1.0, index=df.index)
    stock_day_ret_s   = pd.Series(0.0, index=df.index)

    prev_ohlc: dict = {}
    TOTAL_BARS = 75
    _bar_vol_history: dict[int, list[float]] = {}

    for d, grp in df.groupby(date_col):
        idx = grp.index
        k   = len(grp)
        cur = grp["Close"].values
        bar_nums = np.arange(k)

        # Time features
        is_power_hr_s.loc[idx] = (bar_nums >= TOTAL_BARS - 15).astype(float)
        session_pct_s.loc[idx] = bar_nums / max(k - 1, 1)
        angle = bar_nums / max(k - 1, 1) * 2 * np.pi
        time_sin_s.loc[idx] = np.sin(angle)
        time_cos_s.loc[idx] = np.cos(angle)

        # VWAP
        vw = vwap_all.loc[idx].values
        vwap_pct_s.loc[idx] = np.clip((cur / np.where(vw > 0, vw, np.nan) - 1) * 100, -5, 5)

        # Session high distance
        sess_h = grp["High"].expanding().max().values
        with np.errstate(divide="ignore", invalid="ignore"):
            session_hi_dist_s.loc[idx] = np.clip(
                np.where(sess_h > 0, (cur / sess_h - 1) * 100, 0), -5, 0)

        # Session trend: ATR-normalised drift from open
        open0   = grp["Open"].iloc[0]
        atr_day = atr14.loc[idx].mean()
        if atr_day > 0 and open0 > 0:
            session_trend_s.loc[idx] = ((grp["Close"] - open0) / atr_day).clip(-5, 5).values
        else:
            session_trend_s.loc[idx] = 0.0

        # Gap, prev-day levels, opening drive
        if prev_ohlc:
            ph, pl, pc = list(prev_ohlc.values())[-1]
            gap = (grp["Open"].iloc[0] / pc - 1) * 100 if pc > 0 else 0.0
            gap_pct_s.loc[idx] = np.clip(gap, -5, 5)
            atr_pct = (atr_day / grp["Close"].mean() * 100) if grp["Close"].mean() > 0 else 1.0
            gap_atr_norm_s.loc[idx] = float(np.clip(gap / atr_pct if atr_pct > 0 else 0, -5, 5))
            dist_pd_high_s.loc[idx] = np.clip((cur / ph - 1) * 100 if ph > 0 else 0, -5, 5)
            prev_rng = ph - pl
            if prev_rng > 0:
                prev_close_pos_s.loc[idx] = float(np.clip((pc - pl) / prev_rng, 0, 1))
            # Stock intraday return from yesterday's close
            if pc > 0:
                stock_day_ret_s.loc[idx] = np.clip((cur / pc - 1) * 100, -5, 5)
            # Opening drive (first 3 bars)
            first3_h  = grp["High"].iloc[:min(3, k)].max()
            first3_l  = grp["Low"].iloc[:min(3, k)].min()
            avg_atr   = atr14.loc[idx].values[:min(3, k)].mean() if k > 0 else 1.0
            if avg_atr > 0:
                drive_dir = np.sign(grp["Close"].iloc[min(2, k - 1)] - grp["Open"].iloc[0])
                drive_mag = (first3_h - first3_l) / avg_atr
                opening_drive_s.loc[idx] = float(np.clip(drive_dir * drive_mag, -5, 5))

        # RVOL_TOD
        vols = grp["Volume"].values
        rvol_vals = np.ones(k)
        for slot in range(k):
            hist = _bar_vol_history.get(slot, [])
            if len(hist) >= 3:
                avg_sv = np.mean(hist[-20:])
                if avg_sv > 0:
                    rvol_vals[slot] = np.clip(vols[slot] / avg_sv, 0, 10)
            _bar_vol_history.setdefault(slot, []).append(float(vols[slot]))
        rvol_tod_s.loc[idx] = rvol_vals

        prev_ohlc[d] = (grp["High"].max(), grp["Low"].min(), grp["Close"].iloc[-1])

    out["is_power_hour"]   = is_power_hr_s
    out["session_pct"]     = session_pct_s
    out["time_sin"]        = time_sin_s
    out["time_cos"]        = time_cos_s
    out["vwap_pct"]        = vwap_pct_s
    out["gap_pct"]         = gap_pct_s
    out["gap_atr_norm"]    = gap_atr_norm_s
    out["dist_pd_high"]    = dist_pd_high_s
    out["prev_close_pos"]  = prev_close_pos_s
    out["session_high_dist"] = session_hi_dist_s
    out["opening_drive"]   = opening_drive_s
    out["session_trend"]   = session_trend_s
    out["rvol_tod"]        = rvol_tod_s
    out["_stock_day_ret"]  = stock_day_ret_s   # intermediate; removed after Nifty join

    # ── Nifty market-regime features ────────────────────────────────
    nifty_df = fetch_nifty_5min(days=92)
    if nifty_df is not None and not nifty_df.empty:
        merged = df[["DateTime"]].merge(nifty_df, on="DateTime", how="left")
        nc     = merged["nifty_close"].ffill().bfill()

        # --- nifty_day_ret: cumulative Nifty return today vs yesterday close ---
        nifty_date_col = merged["DateTime"].dt.date
        # build prev-day-close map from nifty data
        nifty_daily_close = (
            nifty_df.copy()
            .assign(date=pd.to_datetime(nifty_df["DateTime"]).dt.date)
            .groupby("date")["nifty_close"].last()
        )
        nifty_prev_close = nifty_daily_close.shift(1)
        # map each bar to its nifty prev-day close
        nifty_pdc_series = pd.Series(nifty_date_col).map(nifty_prev_close)
        nifty_pdc_series.index = df.index
        nifty_pdc_series = nifty_pdc_series.ffill().bfill()
        with np.errstate(divide="ignore", invalid="ignore"):
            nifty_day_ret = ((nc.values - nifty_pdc_series.values)
                             / np.where(nifty_pdc_series.values > 0,
                                        nifty_pdc_series.values, np.nan))
        out["nifty_day_ret"] = np.clip(nifty_day_ret, -0.05, 0.05)

        # Standard Nifty features
        nr6  = nc.pct_change(6).clip(-0.05, 0.05).fillna(0)
        nrsi = (_rsi(nc, 14) / 100).clip(0, 1)
        nadx = _adx_close_only(nc, 14)
        nema21 = _ema(nc, 21)
        out["nifty_ret_6bar"] = nr6.values
        out["nifty_rsi"]      = nrsi.values
        out["nifty_adx"]      = nadx.fillna(0).values
        out["nifty_ema_sig"]  = np.sign(nc - nema21).fillna(0).values

        # rel_strength_day: stock's intraday drift − Nifty's intraday drift
        out["rel_strength_day"] = np.clip(
            out["_stock_day_ret"].values - out["nifty_day_ret"].values * 100,
            -5, 5)

        # v3.0: faster 3-bar Nifty momentum (15-min pivot detection)
        nr3 = nc.pct_change(3).clip(-0.03, 0.03).fillna(0)
        out["nifty_ret_3bar"] = nr3.values
    else:
        for col in ["nifty_day_ret", "nifty_ret_6bar", "nifty_rsi",
                    "nifty_adx", "nifty_ema_sig", "rel_strength_day",
                    "nifty_ret_3bar"]:
            out[col] = 0.0

    # v3.0: session volume acceleration (accumulation trend within session)
    vol_s = out["volume"] if "volume" in out.columns else pd.Series(1.0, index=out.index)
    vol_e3  = vol_s.ewm(span=3,  adjust=False).mean()
    vol_e10 = vol_s.ewm(span=10, adjust=False).mean().replace(0, np.nan)
    out["session_vol_accel"] = (vol_e3 / vol_e10 - 1).clip(-2, 2).fillna(0)

    # drop intermediate column
    out.drop(columns=["_stock_day_ret"], inplace=True, errors="ignore")

    return out[FEATURE_COLS]


# ── Model helpers ─────────────────────────────────────────────────────────────

def _dir_proba_ens(models: list, X: pd.DataFrame) -> np.ndarray:
    """Average P(BUY) across ensemble."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def _lgbm_best_logloss(m: lgb.LGBMClassifier) -> float:
    """Return validation logloss from best iteration (lower = better)."""
    try:
        return list(list(m.best_score_.values())[0].values())[0]
    except Exception:
        return float("inf")


def _make_lgbm_dir(seed: int, col_frac: float, row_frac: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.008,    # slower learning → later, cleaner convergence
        max_depth=5,
        num_leaves=31,
        min_child_samples=60,   # more conservative, less overfitting
        feature_fraction=col_frac,
        bagging_fraction=row_frac,
        bagging_freq=5,
        lambda_l1=0.2,
        lambda_l2=2.0,          # stronger L2 regularisation
        objective="binary",
        metric="binary_logloss",
        verbose=-1,
        n_jobs=-1,
        random_state=seed,
    )


def _make_lgbm_hold(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.008,
        max_depth=4,
        num_leaves=20,
        min_child_samples=70,
        feature_fraction=0.6,
        bagging_fraction=0.75,
        bagging_freq=5,
        lambda_l1=0.2,
        lambda_l2=1.5,
        objective="binary",
        metric="binary_logloss",
        verbose=-1,
        n_jobs=-1,
        random_state=seed,
    )


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv(X_all: pd.DataFrame, y_all: pd.Series, n_folds: int = 3) -> list[dict]:
    n     = len(X_all)
    fold  = n // (n_folds + 1)
    results = []
    for i in range(n_folds):
        train_end  = fold * (i + 1)
        test_start = train_end
        test_end   = min(train_end + fold, n)
        X_tr = X_all.iloc[:train_end];  y_tr = y_all.iloc[:train_end]
        X_te = X_all.iloc[test_start:test_end]
        y_te = y_all.iloc[test_start:test_end]

        bs   = y_tr.isin([1, -1])
        X_bs = X_tr[bs];  y_bs = (y_tr[bs] == 1).astype(int)
        n_b  = (y_bs == 1).sum(); n_s = (y_bs == 0).sum()
        w    = y_bs.apply(lambda v: (n_b + n_s) / (2 * n_b) if v == 1
                           else (n_b + n_s) / (2 * n_s)).values

        m = _make_lgbm_dir(42, 0.70, 0.80)
        bs_te = y_te.isin([1, -1])
        X_bs_te = X_te[bs_te];  y_bs_te = (y_te[bs_te] == 1).astype(int)
        if len(X_bs_te) < 50:
            continue
        val_pct = int(len(X_bs) * 0.20)
        X_cv_val = X_bs.iloc[-val_pct:]; y_cv_val = y_bs.iloc[-val_pct:]
        X_cv_tr  = X_bs.iloc[:-val_pct]; y_cv_tr  = y_bs.iloc[:-val_pct]
        w_tr     = w[:-val_pct]
        cb = lgb.early_stopping(120, verbose=False)
        m.fit(X_cv_tr, y_cv_tr, sample_weight=w_tr,
              eval_set=[(X_cv_val, y_cv_val)], callbacks=[cb])

        y_nh_tr = (y_tr != 0).astype(int)
        n_nh = (y_nh_tr == 1).sum(); n_h = (y_nh_tr == 0).sum()
        w_h = y_nh_tr.apply(lambda v: (n_nh + n_h) / (2 * n_nh) if v == 1
                              else (n_nh + n_h) / (2 * n_h) * 0.8).values
        mh = _make_lgbm_hold(42)
        y_nh_cv_val = (y_te != 0).astype(int)
        cb_h = lgb.early_stopping(120, verbose=False)
        mh.fit(X_tr, y_nh_tr, sample_weight=w_h,
               eval_set=[(X_te, y_nh_cv_val)], callbacks=[cb_h])

        hold_p = mh.predict_proba(X_te)[:, 1]
        emit   = hold_p >= HOLD_THRESHOLD
        n_emit = emit.sum()

        dir_p  = m.predict_proba(X_bs_te)[:, 1]
        pred   = (dir_p >= BUY_CLASS_THRESH).astype(int)
        dir_a  = float(np.mean(pred == y_bs_te.values))
        buy_a  = float(np.mean(pred[y_bs_te == 1] == 1)) if (y_bs_te == 1).any() else 0.0
        sell_a = float(np.mean(pred[y_bs_te == 0] == 0)) if (y_bs_te == 0).any() else 0.0

        comb_a = 0.0
        if n_emit > 0:
            dp = m.predict_proba(X_te[emit])[:, 1]
            de = (dp >= BUY_CLASS_THRESH).astype(int)
            td = (y_te[emit].values == 1).astype(int)
            comb_a = float(np.mean(de == td))

        print(f"    Fold {i+1}: dir={dir_a*100:.1f}%  BUY={buy_a*100:.1f}%  "
              f"SELL={sell_a*100:.1f}%  combined={comb_a*100:.1f}%  "
              f"emit={n_emit}/{len(X_te)}")
        results.append({"dir_acc": dir_a, "buy_acc": buy_a, "sell_acc": sell_a,
                         "combined_acc": comb_a})
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train(symbols: list[str], test_mode: bool = False) -> None:
    print("=" * 62)
    print(f"FULL TRAIN V3.1 — {len(symbols)} symbols · 90d · LightGBM two-stage")
    print("=" * 62)

    all_X_tr:   list[pd.DataFrame] = []   # 70% training pool per symbol
    all_y_tr:   list[pd.Series]    = []
    all_X_oot:  list[pd.DataFrame] = []   # 30% honest test per symbol
    all_y_oot:  list[pd.Series]    = []
    sym_oot_map: dict[str, tuple]  = {}   # symbol → (Xte, yte) for per-sym tier [v3.0]

    for si, sym in enumerate(symbols, 1):
        df = fetch_5min(sym)
        if df is None or len(df) < 500:
            print(f"  [{si:2d}/{len(symbols)}] {sym:<15} skipped (no data)")
            continue

        feats  = compute_features(df)
        labels = make_labels(df)
        valid  = feats.notna().all(axis=1)
        feats  = feats[valid]; labels = labels[valid]

        date_col     = df.loc[feats.index, "DateTime"].dt.date
        unique_dates = sorted(date_col.unique())
        if len(unique_dates) < 12:
            print(f"  [{si:2d}/{len(symbols)}] {sym:<15} skipped (too few days)")
            continue

        nd      = len(unique_dates)
        cutoff  = unique_dates[int(nd * TRAIN_FRAC)]
        is_tr   = date_col < cutoff
        is_test = date_col >= cutoff

        Xtr, ytr = feats[is_tr],   labels[is_tr]
        Xte, yte = feats[is_test], labels[is_test]
        tr_d = int(nd * TRAIN_FRAC)
        te_d = nd - tr_d

        print(f"  [{si:2d}/{len(symbols)}] {sym:<15} "
              f"tr={len(Xtr):,}({tr_d}d)  test={len(Xte):,}({te_d}d)  "
              f"(B:{(ytr==1).sum()} S:{(ytr==-1).sum()} H:{(ytr==0).sum()})")

        all_X_tr.append(Xtr); all_y_tr.append(ytr)
        if len(Xte) > 50:
            all_X_oot.append(Xte); all_y_oot.append(yte)
            sym_oot_map[sym] = (Xte, yte)   # v3.0: per-symbol tier tracking

    if not all_X_tr:
        print("No training data."); return

    X_all = pd.concat(all_X_tr,  ignore_index=True)
    y_all = pd.concat(all_y_tr,  ignore_index=True)
    X_oot = pd.concat(all_X_oot, ignore_index=True) if all_X_oot else None
    y_oot = pd.concat(all_y_oot, ignore_index=True) if all_X_oot else None

    # Early-stop val: last 15% of the CONCATENATED training pool.
    # This is a mix of all symbols' last few training days → balanced classes
    # and in-distribution (same calendar window as actual training).
    n_pool  = len(X_all)
    es_cut  = int(n_pool * ES_POOL_FRAC)
    X_val   = X_all.iloc[es_cut:]
    y_val   = y_all.iloc[es_cut:]
    X_tr    = X_all.iloc[:es_cut]
    y_tr    = y_all.iloc[:es_cut]

    n_buy = (y_all == 1).sum(); n_sell = (y_all == -1).sum(); n_hold = (y_all == 0).sum()
    print(f"\nTrain pool (70%): {len(X_all):,} bars  "
          f"BUY={n_buy:,} ({n_buy/len(y_all)*100:.0f}%)  "
          f"SELL={n_sell:,} ({n_sell/len(y_all)*100:.0f}%)  "
          f"HOLD={n_hold:,} ({n_hold/len(y_all)*100:.0f}%)")
    nb = (y_val==1).sum(); ns = (y_val==-1).sum(); nh = (y_val==0).sum()
    print(f"ES val  (last 15% of pool): {len(X_val):,} bars  "
          f"BUY={nb:,} ({nb/len(y_val)*100:.0f}%)  "
          f"SELL={ns:,} ({ns/len(y_val)*100:.0f}%)  "
          f"HOLD={nh:,} ({nh/len(y_val)*100:.0f}%)")
    if X_oot is not None:
        nb = (y_oot==1).sum(); ns = (y_oot==-1).sum(); nh = (y_oot==0).sum()
        print(f"OOT test (30%): {len(X_oot):,} bars  "
              f"BUY={nb:,} ({nb/len(y_oot)*100:.0f}%)  "
              f"SELL={ns:,} ({ns/len(y_oot)*100:.0f}%)  "
              f"HOLD={nh:,} ({nh/len(y_oot)*100:.0f}%)")

    # Walk-forward CV
    print("\n── Walk-forward CV (3-fold, temporal) ──")
    cv_res  = walk_forward_cv(X_all, y_all, n_folds=3)
    cv_dir  = np.mean([r["dir_acc"] for r in cv_res])
    cv_buy  = np.mean([r["buy_acc"] for r in cv_res])
    cv_sell = np.mean([r["sell_acc"] for r in cv_res])
    cv_comb = np.mean([r["combined_acc"] for r in cv_res])
    print(f"  CV mean → dir={cv_dir*100:.1f}%  BUY={cv_buy*100:.1f}%  "
          f"SELL={cv_sell*100:.1f}%  combined={cv_comb*100:.1f}%")

    # Time-decay weights scoped to X_tr (first 85% of training pool)
    n_tr    = len(X_tr)
    decay   = np.exp(np.linspace(-np.log(3), 0, n_tr))
    _w_time = decay / decay.mean()

    # ── Stage 1: Direction ensemble (3 diverse LightGBM models) ─────
    bs_mask = y_tr.isin([1, -1])
    X_bs    = X_tr[bs_mask]
    y_bs    = (y_tr[bs_mask] == 1).astype(int)
    n_b     = (y_bs == 1).sum(); n_s = (y_bs == 0).sum()
    w_cls   = y_bs.apply(lambda v: (n_b + n_s) / (2 * n_b) if v == 1
                          else (n_b + n_s) / (2 * n_s)).values
    w_dir   = w_cls * _w_time[bs_mask.values]
    w_dir  /= w_dir.mean()

    bs_val  = y_val.isin([1, -1])
    X_bs_v  = X_val[bs_val]
    y_bs_v  = (y_val[bs_val] == 1).astype(int)

    _ens_cfgs = [
        (42, 0.70, 0.80),
        (43, 0.65, 0.75),
        (44, 0.75, 0.85),
    ]
    cb_es  = lgb.early_stopping(150, verbose=False)   # more patience — slower lr
    cb_log = lgb.log_evaluation(period=300)

    print(f"\n── Stage 1: Direction ensemble  "
          f"(train on {len(X_bs):,} BUY/SELL bars × 3 models) ──")
    models_dir: list = []
    for seed, col_f, row_f in _ens_cfgs:
        print(f"  Training model (seed={seed}) …", end="", flush=True)
        m = _make_lgbm_dir(seed, col_f, row_f)
        m.fit(X_bs, y_bs, sample_weight=w_dir,
              eval_set=[(X_bs_v, y_bs_v)], callbacks=[cb_es])
        models_dir.append(m)
        ep = m.best_iteration_
        ens_p = _dir_proba_ens(models_dir, X_bs_v)
        ens_a = float(np.mean((ens_p >= BUY_CLASS_THRESH).astype(int) == y_bs_v.values))
        print(f"  iter={ep:4d}  ensemble val dir={ens_a*100:.1f}%")

    ens_p   = _dir_proba_ens(models_dir, X_bs_v)
    ens_pred = (ens_p >= BUY_CLASS_THRESH).astype(int)
    dir_acc  = float(np.mean(ens_pred == y_bs_v.values))
    buy_acc  = float(np.mean(ens_pred[y_bs_v == 1] == 1)) if (y_bs_v == 1).any() else 0.0
    sell_acc = float(np.mean(ens_pred[y_bs_v == 0] == 0)) if (y_bs_v == 0).any() else 0.0
    print(f"\nEnsemble val → dir={dir_acc*100:.1f}%  "
          f"BUY-recall={buy_acc*100:.1f}%  SELL-recall={sell_acc*100:.1f}%")
    print("\n── Stage 1 per-class report ──")
    print(classification_report(y_bs_v, ens_pred, target_names=["SELL", "BUY"], digits=3))

    model_dir = min(models_dir, key=_lgbm_best_logloss)  # lowest val logloss = best

    # ── Stage 2: HOLD filter ─────────────────────────────────────────
    print("── Stage 2: HOLD filter ──")
    y_nh_tr = (y_tr != 0).astype(int)
    y_nh_v  = (y_val != 0).astype(int)
    n_nh    = (y_nh_tr == 1).sum(); n_h = (y_nh_tr == 0).sum()
    w_cls_h = y_nh_tr.apply(lambda v: (n_nh + n_h) / (2 * n_nh) if v == 1
                               else (n_nh + n_h) / (2 * n_h) * 0.8).values
    w_hold  = w_cls_h * _w_time  # _w_time already sized to len(X_tr)
    w_hold /= w_hold.mean()

    mh = _make_lgbm_hold(42)
    mh.fit(X_tr, y_nh_tr, sample_weight=w_hold,
           eval_set=[(X_val, y_nh_v)], callbacks=[cb_es, cb_log])
    model_hold = mh
    hold_p_val = mh.predict_proba(X_val)[:, 1]
    hold_pred  = (hold_p_val >= 0.5).astype(int)
    hold_acc   = float(np.mean(hold_pred == y_nh_v.values))
    print(f"\nStage 2 val → HOLD-filter acc={hold_acc*100:.1f}%")

    # ── Combined evaluation ──────────────────────────────────────────
    if X_oot is not None and len(X_oot) >= 100:
        N_val    = len(X_oot)
        hold_po  = model_hold.predict_proba(X_oot)[:, 1]
        dir_po   = _dir_proba_ens(models_dir, X_oot)
        emit_mask = hold_po >= HOLD_THRESHOLD
        n_emit    = emit_mask.sum()

        bs_oot   = y_oot.isin([1, -1])
        dir_po_bs = _dir_proba_ens(models_dir, X_oot[bs_oot])
        pred_bs   = (dir_po_bs >= BUY_CLASS_THRESH).astype(int)
        y_bs_oot  = (y_oot[bs_oot] == 1).astype(int)
        raw_dir   = float(np.mean(pred_bs == y_bs_oot.values))
        raw_buy   = float(np.mean(pred_bs[y_bs_oot == 1] == 1)) if (y_bs_oot == 1).any() else 0.0
        raw_sell  = float(np.mean(pred_bs[y_bs_oot == 0] == 0)) if (y_bs_oot == 0).any() else 0.0

        # Combined: on emitted BUY/SELL bars only
        emit_bs = emit_mask & y_oot.isin([1, -1])
        if emit_bs.sum() > 10:
            dp_e  = _dir_proba_ens(models_dir, X_oot[emit_bs])
            pred_e = (dp_e >= BUY_CLASS_THRESH).astype(int)
            true_e = (y_oot[emit_bs].values == 1).astype(int)
            comb_acc = float(np.mean(pred_e == true_e))
            # Precision on emitted BUY signals
            buy_emit = emit_mask & (dir_po >= DIR_THRESHOLD)
            sell_emit = emit_mask & (dir_po <= 1 - DIR_THRESHOLD)
            n_buy_sig  = buy_emit.sum()
            n_sell_sig = sell_emit.sum()
            buy_prec = sell_prec = 0.0
            if n_buy_sig > 0:
                buy_prec  = float(np.mean((y_oot[buy_emit].values == 1).astype(int)))
            if n_sell_sig > 0:
                sell_prec = float(np.mean((y_oot[sell_emit].values == -1).astype(int)))
        else:
            comb_acc = buy_prec = sell_prec = 0.0
            n_buy_sig = n_sell_sig = 0

        print(f"\n── OOT / Val evaluation (last 30% per stock) ──")
        print(f"  Bars            : {N_val:,} total, {n_emit:,} emitted ({n_emit/N_val*100:.1f}%)")
        print(f"  Raw dir accuracy: {raw_dir*100:.1f}%  (BUY: {raw_buy*100:.1f}%  SELL: {raw_sell*100:.1f}%)")
        print(f"  Combined (dir+hold emitted BUY/SELL): {comb_acc*100:.1f}%")
        print(f"  BUY  signals: {n_buy_sig:,}  precision={buy_prec*100:.1f}%")
        print(f"  SELL signals: {n_sell_sig:,}  precision={sell_prec*100:.1f}%")

        # ── Threshold sweep (HOLD × direction joint threshold) ───────
        est_days = max(1, N_val // (75 * max(1, len(all_X_oot))))
        print(f"\n── Threshold sweep (joint hold × dir threshold) ──")
        print(f"  {'thr':>5}  {'BUY sig':>8}  {'BUY prec':>9}  "
              f"{'SELL sig':>8}  {'SELL prec':>9}  signals/day")
        for thr in [0.55, 0.58, 0.60, 0.62, 0.65, 0.67, 0.70, 0.73, 0.75]:
            buy_e  = (hold_po >= thr) & (dir_po >= thr)
            sell_e = (hold_po >= thr) & (dir_po <= 1 - thr)
            nb_e = buy_e.sum(); ns_e = sell_e.sum()
            if nb_e + ns_e == 0:
                print(f"  {thr:.2f}:   no signals"); continue
            bp = float(np.mean((y_oot[buy_e].values == 1).astype(int))) if nb_e > 0 else 0.0
            sp = float(np.mean((y_oot[sell_e].values == -1).astype(int))) if ns_e > 0 else 0.0
            spd = (nb_e + ns_e) / max(1, est_days)
            mark = " ◄ 70%+" if (bp >= 0.70 or sp >= 0.70) else (
                   " ◄ 65%+" if (bp >= 0.65 or sp >= 0.65) else "")
            print(f"  {thr:.2f}:  {nb_e:6d}  {bp*100:8.1f}%  "
                  f"{ns_e:6d}  {sp*100:8.1f}%  {spd:.0f}/day{mark}")

    # ── Feature importances ──────────────────────────────────────────
    imp = pd.Series(model_dir.feature_importances_,
                    index=FEATURE_COLS).sort_values(ascending=False)
    print("\n── Feature importances (direction model) ──")
    for feat, score in imp.items():
        bar = "█" * int(score / imp.max() * 30)
        print(f"  {feat:<25} {score:6.0f}  {bar}")

    # ── Per-symbol OOT BUY precision → tier assignment  [v3.0] ──────────────
    sym_tiers: dict[str, dict] = {}
    for sym, (Xte, yte) in sym_oot_map.items():
        try:
            hold_po_s = model_hold.predict_proba(Xte)[:, 1]
            dir_po_s  = _dir_proba_ens(models_dir, Xte)
            buy_sig_s = (hold_po_s >= HOLD_THRESHOLD) & (dir_po_s >= DIR_THRESHOLD)
            n_buy_s   = int(buy_sig_s.sum())
            if n_buy_s < 3:
                tier_s = "C"; prec_s = float("nan")
            else:
                prec_s = float(np.mean((yte.values[buy_sig_s] == 1).astype(int)))
                tier_s = "A" if prec_s >= 0.68 else ("B" if prec_s >= 0.55 else "C")
            sym_tiers[sym] = {"tier": tier_s, "buy_precision": prec_s, "n_signals": n_buy_s}
        except Exception:
            sym_tiers[sym] = {"tier": "C", "buy_precision": float("nan"), "n_signals": 0}

    ta = [s for s, v in sym_tiers.items() if v["tier"] == "A"]
    tb = [s for s, v in sym_tiers.items() if v["tier"] == "B"]
    tc = [s for s, v in sym_tiers.items() if v["tier"] == "C"]
    print(f"\nPer-symbol OOT BUY precision tiers:")
    print(f"  Tier A (≥68%): {len(ta)} — {', '.join(sorted(ta))}")
    print(f"  Tier B (55-68%): {len(tb)} — {', '.join(sorted(tb))}")
    print(f"  Tier C (<55%): {len(tc)} — {', '.join(sorted(tc))} [signals need higher threshold]")

    if test_mode:
        print("\n[TEST MODE] Models NOT saved."); return

    MODEL_PATH_V2.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model":      model_dir,
        "model_dir":  model_dir,
        "models_dir": models_dir,
        "model_hold": model_hold,
        "features":   FEATURE_COLS,
        "version":    "3.1",
        "trained_at": __import__("datetime").date.today().isoformat(),
        "dir_threshold":  DIR_THRESHOLD,
        "hold_threshold": HOLD_THRESHOLD,
        "buy_class_thresh": BUY_CLASS_THRESH,
        "sym_tiers":  sym_tiers,   # v3.0: per-symbol BUY precision tiers
    }, MODEL_PATH_V2)
    print(f"\nSaved → {MODEL_PATH_V2}")


# ── Live predict ──────────────────────────────────────────────────────────────

def predict(symbol: str) -> dict:
    if not MODEL_PATH_V2.exists():
        return {"error": f"Model not found: {MODEL_PATH_V2}"}
    blob       = joblib.load(MODEL_PATH_V2)
    models_dir = blob.get("models_dir", [blob.get("model_dir", blob["model"])])
    model_hold = blob.get("model_hold")
    feat_cols  = blob.get("features", FEATURE_COLS)
    dir_thr    = blob.get("dir_threshold", DIR_THRESHOLD)
    hold_thr   = blob.get("hold_threshold", HOLD_THRESHOLD)
    buy_thr    = blob.get("buy_class_thresh", BUY_CLASS_THRESH)
    sym_tiers  = blob.get("sym_tiers", {})

    # Apply tier-adjusted threshold [v3.0]: Tier A=0.65, Tier B=0.68, Tier C=0.72
    tier_info  = sym_tiers.get(symbol, {"tier": "C"})
    stock_tier = tier_info.get("tier", "C")
    _tier_dir  = {"A": dir_thr, "B": dir_thr + 0.03, "C": dir_thr + 0.07}
    eff_dir_thr = _tier_dir[stock_tier]

    df = fetch_5min(symbol, days=5)
    default = {"signal": "HOLD", "buy_prob": 0.5, "sell_prob": 0.5,
               "hold_prob": 1.0, "confidence": 0.0, "data_ok": False}
    if df is None or len(df) < 30:
        return {**default, "error": "no data"}

    feats = compute_features(df)
    if feats is None or len(feats) == 0:
        return {**default, "error": "feature error"}

    X = feats[feat_cols].tail(1)
    if X.isna().any().any():
        return {**default, "error": "NaN features"}

    if model_hold is not None:
        hold_proba = float(model_hold.predict_proba(X)[0][1])
        if hold_proba < hold_thr:
            return {**default, "signal": "HOLD", "hold_prob": round(1 - hold_proba, 3),
                    "data_ok": True}
    else:
        hold_proba = 1.0

    dir_proba = float(_dir_proba_ens(models_dir, X)[0])

    if dir_proba >= eff_dir_thr:
        signal = "BUY";  conf = dir_proba
    elif (1.0 - dir_proba) >= eff_dir_thr:
        signal = "SELL"; conf = 1.0 - dir_proba
    else:
        signal = "HOLD"; conf = max(dir_proba, 1.0 - dir_proba)

    return {
        "signal":     signal,
        "buy_prob":   round(dir_proba, 3),
        "sell_prob":  round(1.0 - dir_proba, 3),
        "hold_prob":  round(1.0 - hold_proba, 3),
        "confidence": round(conf, 3),
        "tier":       stock_tier,
        "data_ok":    True,
    }


# ── Importance display ────────────────────────────────────────────────────────

def show_importance() -> None:
    if not MODEL_PATH_V2.exists():
        print(f"Model not found: {MODEL_PATH_V2}"); return
    blob  = joblib.load(MODEL_PATH_V2)
    m     = blob.get("model_dir", blob["model"])
    feats = blob.get("features", FEATURE_COLS)
    imp   = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    print("Direction model — feature importances:")
    for f, s in imp.items():
        print(f"  {f:<30} {s:6.0f}  {'█' * int(s / imp.max() * 30)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(0)

    cmd = args[0].lower()

    if cmd == "train":
        test_mode = "--test" in args
        syms = NIFTY_50[:8] if test_mode else NIFTY_50
        train(syms, test_mode=test_mode)

    elif cmd == "predict":
        if len(args) < 2:
            print("Usage: python intraday_model_v2.py predict SYMBOL")
        else:
            import json
            print(json.dumps(predict(args[1]), indent=2))

    elif cmd == "importance":
        show_importance()

    else:
        print(f"Unknown command: {cmd}")
