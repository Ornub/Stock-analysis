"""
intraday.py — Free intraday feature engine using yfinance.

Works out of the box with zero API keys or subscriptions.
Uses NSE 5-min data (free, ~15-min delayed) to compute:

  vwap_ratio        — close / today's VWAP
  orb_signal        — Opening Range Breakout vs first-15-min high/low
  morning_range_pos — price position in ORB range [0, 1]
  intraday_vol_surge— today's vol vs 20-day avg at same bar index
  first_hour_return — first-hour price change %

Optional upgrade: set DEPTH_PROVIDER = "zerodha" or "angel" in config to
also get market depth score (Level 2 bid/ask pressure).
"""

from __future__ import annotations
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# ── helpers ───────────────────────────────────────────────────────────────────

def _yf_symbol(nse_symbol: str) -> str:
    return f"{nse_symbol}.NS"


def _fetch_5min(symbol: str, days: int = 5) -> pd.DataFrame | None:
    """Fetch 5-min OHLCV via yfinance. Returns tz-naive UTC+5:30 DataFrame."""
    try:
        ticker = yf.Ticker(_yf_symbol(symbol))
        df = ticker.history(period=f"{days}d", interval="5m", auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.rename(columns={"Datetime": "DateTime"}, inplace=True)
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
        return df[["DateTime", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"[WARN] _fetch_5min({symbol}): {e}")
        return None


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    cv  = df["Volume"].cumsum().replace(0, np.nan)
    return (tp * df["Volume"]).cumsum() / cv


def _today_bars(df: pd.DataFrame) -> pd.DataFrame:
    today = str(date.today())
    mask  = df["DateTime"].dt.date.astype(str) == today
    return df[mask].reset_index(drop=True)


# ── public API ────────────────────────────────────────────────────────────────

def get_today_bars(symbol: str) -> pd.DataFrame | None:
    """Return today's 5-min OHLCV bars for a symbol (for charting)."""
    df = _fetch_5min(symbol, days=2)
    if df is None or df.empty:
        return None
    today = _today_bars(df)
    return today if not today.empty else None


def get_intraday_features(symbol: str) -> dict:
    """Compute free intraday features for one NSE symbol.

    Returns dict with keys:
        vwap_ratio        float   close / VWAP (>1 = above, bullish)
        vwap              float   actual VWAP price level
        orb_signal        int     +1 above ORB high / -1 below ORB low / 0
        orb_high          float   opening-range high price
        orb_low           float   opening-range low price
        cur_price         float   latest close price
        morning_range_pos float   position in first-15-min range [0, 1]
        intraday_vol_surge float  today vol vs 20d avg at same time (ratio)
        first_hour_return float   % change from open to end of first hour
        data_ok           bool    True if intraday data was successfully fetched
    """
    result = dict(
        vwap_ratio=1.0, vwap=float("nan"), orb_signal=0,
        orb_high=float("nan"), orb_low=float("nan"), cur_price=float("nan"),
        morning_range_pos=0.5, intraday_vol_surge=1.0,
        first_hour_return=0.0, data_ok=False,
    )

    df = _fetch_5min(symbol, days=2)
    if df is None or df.empty:
        return result

    today = _today_bars(df)
    if today.empty:
        return result

    cur_price = float(today["Close"].iloc[-1])
    result["cur_price"] = round(cur_price, 2)

    # ── VWAP ──────────────────────────────────────────────────────────
    vwap_val = float(_vwap(today).iloc[-1])
    if vwap_val > 0:
        result["vwap"]       = round(vwap_val, 2)
        result["vwap_ratio"] = round(cur_price / vwap_val, 4)

    # ── Opening Range Breakout (first 15 min = first 3 × 5-min bars) ──
    orb_bars = today.head(3)
    orb_high = float(orb_bars["High"].max())
    orb_low  = float(orb_bars["Low"].min())
    orb_rng  = orb_high - orb_low
    result["orb_high"] = round(orb_high, 2)
    result["orb_low"]  = round(orb_low, 2)
    if orb_rng > 0:
        pos = (cur_price - orb_low) / orb_rng
        result["morning_range_pos"] = round(max(0.0, min(1.0, pos)), 3)
    if cur_price > orb_high:
        result["orb_signal"] = 1
    elif cur_price < orb_low:
        result["orb_signal"] = -1

    # ── First-hour return (first 12 × 5-min bars = 60 min) ────────────
    open_price = float(today["Open"].iloc[0])
    fh_bars    = today.head(12)
    if len(fh_bars) == 12 and open_price > 0:
        fh_close = float(fh_bars["Close"].iloc[-1])
        result["first_hour_return"] = round((fh_close / open_price - 1) * 100, 3)

    # ── Intraday volume surge vs 20-day avg at same bar count ─────────
    bar_idx = len(today)
    hist_df = _fetch_5min(symbol, days=30)
    if hist_df is not None and not hist_df.empty:
        hist_df["date_only"] = hist_df["DateTime"].dt.date
        groups   = [g.reset_index(drop=True) for _, g in hist_df.groupby("date_only")]
        avg_vols = [
            float(g.iloc[:bar_idx]["Volume"].sum())
            for g in groups[-20:]
            if len(g) >= bar_idx
        ]
        if avg_vols:
            avg_vol   = np.mean(avg_vols)
            today_vol = float(today["Volume"].sum())
            result["intraday_vol_surge"] = round(
                today_vol / avg_vol if avg_vol > 0 else 1.0, 2
            )

    result["data_ok"] = True
    return result


def get_batch_intraday_features(symbols: list[str]) -> dict[str, dict]:
    """Fetch intraday features for multiple symbols. Returns {symbol: dict}."""
    return {sym: get_intraday_features(sym) for sym in symbols}


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(f"\nIntraday features for {sym} (free via yfinance):\n{'─'*44}")
    feats = get_intraday_features(sym)
    for k, v in feats.items():
        print(f"  {k:<22} {v}")
    print(f"{'─'*44}")
