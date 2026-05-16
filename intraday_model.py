"""
intraday_model.py — ML intraday signal engine for NSE stocks.

Implements 7 classic theories as XGBoost features:

  1. Opening Range Breakout (ORB)   — Raschke / Crabel: buy/sell the 15-min range break
  2. VWAP Deviation                 — institutional benchmark; price above = buy pressure
  3. Supertrend                     — Wilder ATR-based trend filter (period=7, mult=3)
  4. Central Pivot Range (CPR)      — floor trader pivots from previous day OHLC
  5. EMA 9/21 Cross                 — fast trend alignment on 5-min bars
  6. Bollinger Band Squeeze         — volatility contraction signals breakout
  7. RSI + MACD Momentum            — classic combo: zone + histogram direction

Label (3-class):
  BUY  — close rises ≥ 0.3% within next 6 bars (30 min)
  SELL — close falls ≥ 0.3% within next 6 bars
  HOLD — neither

CLI:
  python intraday_model.py train              # train on full Nifty-50
  python intraday_model.py train RELIANCE TCS # train on specific symbols
  python intraday_model.py predict RELIANCE   # predict current signal
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
    # ORB
    "orb_signal", "morning_range_pos",
    # VWAP
    "vwap_pct",
    # Supertrend
    "supertrend_sig",
    # CPR
    "cpr_pos",
    # EMA cross
    "ema9_21_sig", "ema21_50_sig",
    # RSI
    "rsi", "rsi_zone",
    # MACD
    "macd_hist_norm", "macd_cross",
    # Bollinger
    "bb_pct_b", "bb_squeeze",
    # Volume
    "vol_ratio",
    # Session
    "session_pct",
    # Price momentum
    "ret_1bar", "ret_3bar", "ret_6bar",
    # Candle quality
    "bar_body_pct",
    # Streak
    "consec_dir",
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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 7) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 7, mult: float = 3.0) -> pd.Series:
    """Supertrend: +1 = bullish, -1 = bearish. Vectorised where possible."""
    atr   = _atr(high, low, close, period)
    hl2   = (high + low) / 2
    upper = (hl2 + mult * atr).values.copy()
    lower = (hl2 - mult * atr).values.copy()
    close_v = close.values
    n = len(close_v)

    sig = np.ones(n)  # 1 = bullish
    st  = lower.copy()

    for i in range(1, n):
        # Adjust bands using previous close
        if close_v[i - 1] <= upper[i - 1]:
            upper[i] = min(upper[i], upper[i - 1])
        if close_v[i - 1] >= lower[i - 1]:
            lower[i] = max(lower[i], lower[i - 1])

        if sig[i - 1] == 1:
            if close_v[i] < lower[i]:
                sig[i] = -1
                st[i]  = upper[i]
            else:
                st[i] = lower[i]
        else:
            if close_v[i] > upper[i]:
                sig[i] = 1
                st[i]  = lower[i]
            else:
                st[i]  = upper[i]

    return pd.Series(sig, index=close.index)


def _vwap_daily(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP resetting at each calendar day."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = tp * df["Volume"]
    date_col = df["DateTime"].dt.date

    vwap = pd.Series(np.nan, index=df.index)
    for d, grp in df.groupby(date_col):
        cum_tpv = tpv.loc[grp.index].cumsum()
        cum_vol = df.loc[grp.index, "Volume"].cumsum().replace(0, np.nan)
        vwap.loc[grp.index] = cum_tpv / cum_vol
    return vwap


def _bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    """Return (pct_b, squeeze). Squeeze = current bandwidth < 50-bar avg bandwidth."""
    mid    = close.rolling(period).mean()
    sigma  = close.rolling(period).std(ddof=0)
    upper  = mid + std * sigma
    lower  = mid - std * sigma
    pct_b  = (close - lower) / (upper - lower).replace(0, np.nan)
    bwidth = sigma / mid.replace(0, np.nan)
    squeeze = (bwidth < bwidth.rolling(50).mean()).astype(float)
    return pct_b, squeeze


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame aligned with df's index."""
    df = df.sort_values("DateTime").reset_index(drop=True)
    date_col = df["DateTime"].dt.date

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"]

    out = pd.DataFrame(index=df.index)

    # ── Whole-series indicators (computed once across all days) ───────
    # Theory 3: Supertrend
    out["supertrend_sig"] = _supertrend(high, low, close)

    # Theory 5: EMA cross
    ema9  = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    out["ema9_21_sig"]  = np.sign(ema9  - ema21).fillna(0)
    out["ema21_50_sig"] = np.sign(ema21 - ema50).fillna(0)

    # Theory 7a: RSI
    rsi_raw = _rsi(close, 14)
    out["rsi"]      = (rsi_raw / 100).clip(0, 1).fillna(0.5)
    out["rsi_zone"] = pd.cut(rsi_raw, bins=[-1, 30, 70, 101],
                             labels=[0.0, 1.0, 2.0]).astype(float).fillna(1.0)

    # Theory 7b: MACD
    macd_line = _ema(close, 12) - _ema(close, 26)
    macd_sig  = _ema(macd_line, 9)
    macd_hist = macd_line - macd_sig
    roll_std  = close.rolling(20).std(ddof=0).replace(0, np.nan)
    out["macd_hist_norm"] = (macd_hist / roll_std).clip(-3, 3).fillna(0)
    out["macd_cross"] = (
        ((macd_line > macd_sig) & (macd_line.shift() <= macd_sig.shift())).astype(float)
        - ((macd_line < macd_sig) & (macd_line.shift() >= macd_sig.shift())).astype(float)
    ).fillna(0)

    # Theory 6: Bollinger Bands
    bb_pct_b, bb_squeeze = _bollinger(close, 20, 2.0)
    out["bb_pct_b"]   = bb_pct_b.clip(-0.5, 1.5).fillna(0.5)
    out["bb_squeeze"] = bb_squeeze.fillna(0)

    # Volume ratio (bar vol vs rolling 20-bar avg)
    out["vol_ratio"] = (volume / volume.rolling(20).mean().replace(0, np.nan)).clip(0, 10).fillna(1)

    # Price momentum
    out["ret_1bar"] = close.pct_change(1).clip(-0.02,  0.02).fillna(0)
    out["ret_3bar"] = close.pct_change(3).clip(-0.05,  0.05).fillna(0)
    out["ret_6bar"] = close.pct_change(6).clip(-0.08,  0.08).fillna(0)

    # Candle body quality
    body     = (close - open_).abs()
    hl_range = (high - low).replace(0, np.nan)
    out["bar_body_pct"] = (body / hl_range).clip(0, 1).fillna(0.5)

    # Consecutive direction streak (positive = up streak, negative = down streak)
    direction = np.sign(close.values - close.shift().values)
    streak    = np.zeros(len(direction))
    cur       = 0
    for i in range(len(direction)):
        d = direction[i]
        if np.isnan(d) or d == 0:
            cur = 0
        elif d == np.sign(cur) or cur == 0:
            cur = cur + d
        else:
            cur = d
        streak[i] = cur
    out["consec_dir"] = np.clip(streak, -10, 10)

    # ── Per-day features (ORB, VWAP, CPR, session %) ─────────────────
    vwap_all = _vwap_daily(df)

    orb_signal_s    = pd.Series(0.0,  index=df.index)
    morning_range_s = pd.Series(0.5,  index=df.index)
    vwap_pct_s      = pd.Series(0.0,  index=df.index)
    cpr_pos_s       = pd.Series(0.0,  index=df.index)
    session_pct_s   = pd.Series(0.5,  index=df.index)

    prev_ohlc: dict = {}   # date → (H, L, C)

    for d, grp in df.groupby(date_col):
        idx = grp.index
        n   = len(grp)
        cur = grp["Close"]

        # Theory 1: ORB (first 3 bars = 15 min)
        orb_h = grp["High"].iloc[:min(3, n)].max()
        orb_l = grp["Low"].iloc[:min(3, n)].min()
        rng   = orb_h - orb_l
        if rng > 0:
            pos = ((cur - orb_l) / rng).clip(0, 1)
            morning_range_s.loc[idx] = pos.values
        orb_signal_s.loc[idx] = (
            (cur > orb_h).astype(float) - (cur < orb_l).astype(float)
        ).values

        # Theory 2: VWAP deviation %
        vwap_day = vwap_all.loc[idx]
        vwap_pct_s.loc[idx] = ((cur.values / vwap_day.values - 1) * 100).clip(-5, 5)

        # Theory 4: CPR from previous day
        if prev_ohlc:
            ph, pl, pc = list(prev_ohlc.values())[-1]
            pivot   = (ph + pl + pc) / 3
            bc      = (ph + pl) / 2
            tc      = 2 * pivot - bc
            cpr_mid = (tc + bc) / 2
            if cpr_mid > 0:
                cpr_pos_s.loc[idx] = ((cur.values / cpr_mid - 1) * 100).clip(-5, 5)

        # Session position 0→1
        session_pct_s.loc[idx] = np.linspace(0, 1, n)

        # Record OHLC for next day's CPR
        prev_ohlc[d] = (
            grp["High"].max(),
            grp["Low"].min(),
            grp["Close"].iloc[-1],
        )

    out["orb_signal"]        = orb_signal_s
    out["morning_range_pos"] = morning_range_s
    out["vwap_pct"]          = vwap_pct_s
    out["cpr_pos"]           = cpr_pos_s
    out["session_pct"]       = session_pct_s

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
    print(f"7 theories → {len(FEATURE_COLS)} features · label = ±0.3% in 30 min\n")

    all_X, all_y = [], []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:2}/{len(symbols)}] {sym}", end=" … ", flush=True)
        df = fetch_5min(sym, days=days)
        if df is None or len(df) < 150:
            print("skipped (no data)")
            continue

        feats  = compute_features(df)
        labels = make_labels(df)

        # Only bars after 10:00 (ORB context established)
        hour_ok = df["DateTime"].dt.hour >= 10
        # Exclude last TARGET_BARS rows (no label)
        label_ok = labels.notna()
        feat_ok  = feats.notna().all(axis=1)
        mask     = hour_ok & label_ok & feat_ok

        X = feats[mask][FEATURE_COLS].astype(float)
        y = labels[mask]
        all_X.append(X)
        all_y.append(y)
        print(f"{len(X):,} bars  (B:{(y==1).sum()} S:{(y==-1).sum()} H:{(y==0).sum()})")

    if not all_X:
        print("No training data collected.")
        return

    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    # Map labels: -1→0 (SELL), 0→1 (HOLD), 1→2 (BUY) for XGBoost
    y_xgb = y_all.map({-1: 0, 0: 1, 1: 2})

    # Compute sample weights to balance classes
    counts = y_all.value_counts()
    total  = len(y_all)
    w_map  = {v: total / (len(counts) * counts[v]) for v in counts.index}
    weights = y_all.map(w_map).values

    print(f"\nTotal: {len(X_all):,} bars  "
          f"SELL={( y_all==-1).sum():,}  HOLD={(y_all==0).sum():,}  BUY={(y_all==1).sum():,}")

    # Train / val split (last 10% as val)
    split  = int(len(X_all) * 0.9)
    X_tr, X_val = X_all.iloc[:split], X_all.iloc[split:]
    y_tr, y_val = y_xgb.iloc[:split], y_xgb.iloc[split:]
    w_tr        = weights[:split]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=15,
        gamma=1,
        eval_metric="mlogloss",
        tree_method="hist",
        num_class=3,
        objective="multi:softprob",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    val_preds = model.predict(X_val)
    val_acc   = (val_preds == y_val.values).mean()

    # Per-class accuracy
    for lbl, name in [(0, "SELL"), (1, "HOLD"), (2, "BUY")]:
        mask_c = y_val == lbl
        if mask_c.sum() > 0:
            acc_c = (val_preds[mask_c] == lbl).mean()
            print(f"  {name} accuracy: {acc_c*100:.1f}%  (n={mask_c.sum()})")

    print(f"\nOverall val accuracy: {val_acc*100:.1f}%")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model":      model,
        "features":   FEATURE_COLS,
        "val_acc":    val_acc,
        "trained_at": str(date.today()),
        "n_symbols":  len(all_X),
        "n_samples":  len(X_all),
        "target_bars": TARGET_BARS,
        "buy_thresh":  BUY_THRESH,
    }, MODEL_PATH)
    print(f"Saved → {MODEL_PATH}")


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

    df = fetch_5min(symbol, days=5)
    if df is None or df.empty:
        return default

    feats = compute_features(df)
    valid = feats.dropna(subset=FEATURE_COLS)
    if valid.empty:
        return default

    X     = valid[FEATURE_COLS].tail(1).astype(float)
    proba = model.predict_proba(X)[0]   # [sell_prob, hold_prob, buy_prob]
    sell_p, hold_p, buy_p = float(proba[0]), float(proba[1]), float(proba[2])

    max_p = max(buy_p, sell_p, hold_p)
    if buy_p >= sell_p and buy_p >= hold_p and buy_p >= MIN_BUY_PROB:
        signal = "BUY"
        conf   = buy_p
    elif sell_p >= buy_p and sell_p >= hold_p and sell_p >= MIN_BUY_PROB:
        signal = "SELL"
        conf   = sell_p
    else:
        signal = "HOLD"
        conf   = hold_p

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
