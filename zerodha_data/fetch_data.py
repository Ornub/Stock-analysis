"""
zerodha_data/fetch_data.py — Intraday data + Phase 2 features via Kite Connect.

Provides:
  get_intraday_features(symbol)  → dict of fast features for a single stock
  get_market_depth_score(symbol) → float 0-1 (buy pressure)
  get_intraday_df(symbol, interval, days) → OHLCV DataFrame

Phase 2 features computed here:
  vwap_ratio        — close / VWAP (intraday mean price benchmark)
  morning_range_pos — where price sits in first-15-min range (0-1)
  orb_signal        — Opening Range Breakout: +1 above, -1 below, 0 neutral
  depth_score       — bid qty / (bid + ask qty): buy pressure 0-1
  intraday_vol_surge— current 15-min volume vs 20-day avg 15-min volume
"""

from __future__ import annotations
import sys
from datetime import date, timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from auth import get_kite, TokenExpiredError

# NSE symbol → Kite instrument token lookup cache
_INSTRUMENT_CACHE: dict[str, int] | None = None

# Kite interval labels
INTERVAL_5MIN  = "5minute"
INTERVAL_15MIN = "15minute"
INTERVAL_DAY   = "day"


def _get_instruments(kite) -> dict[str, int]:
    """Build symbol → instrument_token map for NSE EQ segment."""
    global _INSTRUMENT_CACHE
    if _INSTRUMENT_CACHE is not None:
        return _INSTRUMENT_CACHE
    df = pd.DataFrame(kite.instruments("NSE"))
    eq = df[(df["segment"] == "NSE") & (df["instrument_type"] == "EQ")]
    _INSTRUMENT_CACHE = dict(zip(eq["tradingsymbol"], eq["instrument_token"]))
    return _INSTRUMENT_CACHE


def get_intraday_df(
    symbol: str,
    interval: str = INTERVAL_15MIN,
    days: int = 5,
    kite=None,
) -> pd.DataFrame | None:
    """Fetch intraday OHLCV DataFrame for a symbol.

    Args:
        symbol:   NSE symbol, e.g. "RELIANCE"
        interval: Kite interval string — "5minute", "15minute", "60minute", "day"
        days:     How many calendar days back to fetch (max 60 for intraday)
        kite:     Pre-authenticated KiteConnect instance (optional)

    Returns:
        DataFrame with columns [DateTime, Open, High, Low, Close, Volume]
        or None if symbol not found / data unavailable.
    """
    try:
        if kite is None:
            kite = get_kite()
        instruments = _get_instruments(kite)
        token = instruments.get(symbol)
        if token is None:
            print(f"[WARN] {symbol} not found in NSE instruments list.")
            return None

        to_date   = datetime.now()
        from_date = to_date - timedelta(days=days)

        raw = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=False,
            oi=False,
        )
        if not raw:
            return None

        df = pd.DataFrame(raw)
        df.rename(columns={"date": "DateTime"}, inplace=True)
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
        return df[["DateTime", "open", "high", "low", "close", "volume"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low",
                     "close": "Close", "volume": "Volume"}
        )
    except TokenExpiredError:
        raise
    except Exception as e:
        print(f"[ERROR] get_intraday_df({symbol}): {e}")
        return None


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP from intraday OHLCV df (typical price × volume / cumvol)."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (tp * df["Volume"]).cumsum() / cum_vol


def get_intraday_features(symbol: str, kite=None) -> dict:
    """Compute Phase 2 intraday features for a single symbol.

    Returns a dict with keys:
        vwap_ratio        — close / VWAP (>1 = above VWAP, bullish)
        morning_range_pos — position in first-15-min range [0, 1]
        orb_signal        — +1 above ORB high, -1 below ORB low, 0 neutral
        intraday_vol_surge— today's vol vs 20-day avg vol at same time (ratio)
        depth_score       — buy pressure from market depth [0, 1]  (NaN if unavailable)
        last_price        — current price
        data_ok           — True if intraday data was available
    """
    result = {
        "vwap_ratio": 1.0,
        "morning_range_pos": 0.5,
        "orb_signal": 0,
        "intraday_vol_surge": 1.0,
        "depth_score": float("nan"),
        "last_price": float("nan"),
        "data_ok": False,
    }

    try:
        if kite is None:
            kite = get_kite()

        # ── 15-min bars for today ────────────────────────────────────
        df = get_intraday_df(symbol, interval=INTERVAL_15MIN, days=2, kite=kite)
        if df is None or df.empty:
            return result

        today_str = str(date.today())
        today_bars = df[df["DateTime"].dt.date.astype(str) == today_str].copy()
        if today_bars.empty:
            return result

        today_bars = today_bars.reset_index(drop=True)
        last_bar   = today_bars.iloc[-1]
        cur_price  = float(last_bar["Close"])
        result["last_price"] = cur_price

        # ── VWAP ─────────────────────────────────────────────────────
        vwap_series = _compute_vwap(today_bars)
        vwap_now    = float(vwap_series.iloc[-1])
        result["vwap_ratio"] = round(cur_price / vwap_now, 4) if vwap_now > 0 else 1.0

        # ── Opening Range (first 15-min bar) ─────────────────────────
        orb_high = float(today_bars.iloc[0]["High"])
        orb_low  = float(today_bars.iloc[0]["Low"])
        orb_rng  = orb_high - orb_low
        if orb_rng > 0:
            result["morning_range_pos"] = round(
                max(0.0, min(1.0, (cur_price - orb_low) / orb_rng)), 3
            )
        if cur_price > orb_high:
            result["orb_signal"] = 1
        elif cur_price < orb_low:
            result["orb_signal"] = -1

        # ── Intraday volume surge ─────────────────────────────────────
        # Compare today's cumulative volume to 20-day historical avg at same bar #
        bar_idx = len(today_bars)
        hist_df = get_intraday_df(symbol, interval=INTERVAL_15MIN, days=30, kite=kite)
        if hist_df is not None and not hist_df.empty:
            hist_df["date_only"] = hist_df["DateTime"].dt.date
            daily_groups = [g for _, g in hist_df.groupby("date_only")]
            bar_vols = []
            for g in daily_groups[-20:]:
                g = g.reset_index(drop=True)
                if len(g) >= bar_idx:
                    bar_vols.append(float(g.iloc[:bar_idx]["Volume"].sum()))
            if bar_vols:
                avg_vol = np.mean(bar_vols)
                today_vol = float(today_bars["Volume"].sum())
                result["intraday_vol_surge"] = round(
                    today_vol / avg_vol if avg_vol > 0 else 1.0, 2
                )

        # ── Market depth score ────────────────────────────────────────
        try:
            instruments = _get_instruments(kite)
            token = instruments.get(symbol)
            if token:
                q = kite.quote([f"NSE:{symbol}"])
                depth = q.get(f"NSE:{symbol}", {}).get("depth", {})
                buy_qty  = sum(d.get("quantity", 0) for d in depth.get("buy",  []))
                sell_qty = sum(d.get("quantity", 0) for d in depth.get("sell", []))
                total    = buy_qty + sell_qty
                if total > 0:
                    result["depth_score"] = round(buy_qty / total, 3)
        except Exception:
            pass  # depth unavailable outside market hours

        result["data_ok"] = True

    except TokenExpiredError:
        raise
    except Exception as e:
        print(f"[ERROR] get_intraday_features({symbol}): {e}")

    return result


def get_batch_intraday_features(
    symbols: list[str],
    kite=None,
) -> dict[str, dict]:
    """Fetch intraday features for a list of symbols.
    Returns {symbol: feature_dict} mapping.
    """
    if kite is None:
        kite = get_kite()

    results = {}
    for sym in symbols:
        results[sym] = get_intraday_features(sym, kite=kite)
    return results


# ── CLI quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(f"\nFetching intraday features for {sym}…")
    try:
        feats = get_intraday_features(sym)
        print(f"\n{'─'*40}")
        for k, v in feats.items():
            print(f"  {k:<22} {v}")
        print(f"{'─'*40}")
    except TokenExpiredError as e:
        print(e)
