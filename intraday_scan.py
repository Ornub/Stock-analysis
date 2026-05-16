"""
intraday_scan.py — Intraday trade setup scanner for NSE stocks.

Scores every symbol on the watchlist / Nifty-50 list using free
yfinance 5-min data and classifies them into actionable setups:

  Strong ORB Breakout   — price broke opening range with volume surge + above VWAP
  ORB Breakout          — price broke opening range high
  VWAP Momentum         — trending above VWAP with volume confirmation
  Avoid / Breakdown     — below VWAP or ORB low, selling pressure active

Entry, stop, and target are computed from actual price levels.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pandas as pd

from intraday import get_intraday_features, get_today_bars  # noqa: E402


# ── Market-hours helper ───────────────────────────────────────────────────────

IST = timezone(timedelta(hours=5, minutes=30))

def market_status() -> dict:
    """Return dict with 'open' bool, 'time_ist' str, 'session_pct' float 0-1."""
    now = datetime.now(IST)
    open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_open  = open_t <= now <= close_t and now.weekday() < 5
    session_secs = (close_t - open_t).total_seconds()
    elapsed_secs = max(0.0, min(session_secs, (now - open_t).total_seconds()))
    return {
        "open":        is_open,
        "time_ist":    now.strftime("%H:%M IST"),
        "date_ist":    now.strftime("%d %b %Y"),
        "session_pct": round(elapsed_secs / session_secs, 3) if is_open else 0.0,
        "last_hour":   now >= close_t - timedelta(hours=1),
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_and_classify(f: dict) -> tuple[int, str, str]:
    """Return (score 0-100, setup_label, direction 'long'|'short'|'none')."""
    orb    = f["orb_signal"]
    vsurge = f["intraday_vol_surge"]
    vwap_r = f["vwap_ratio"]
    fhr    = f.get("first_hour_return", 0.0) or 0.0
    mrng   = f["morning_range_pos"]

    score = 50  # neutral baseline

    # ORB contribution (±40)
    score += orb * 40

    # Volume confirmation (0 to +30)
    if vsurge >= 2.0:
        score += 30
    elif vsurge >= 1.5:
        score += 20
    elif vsurge >= 1.2:
        score += 10

    # VWAP alignment (±20)
    if vwap_r > 1.005:
        score += 20
    elif vwap_r < 0.995:
        score -= 20

    # Momentum bonus (0 to +10)
    if fhr > 2.0:
        score += 10
    elif fhr > 1.0:
        score += 5
    elif fhr < -2.0:
        score -= 10

    # Range position tiebreaker (±5)
    if mrng > 0.85:
        score += 5
    elif mrng < 0.15:
        score -= 5

    score = max(0, min(100, score))

    # ── Setup label ──────────────────────────────────────────────────
    above_vwap = vwap_r > 1.0
    if orb == 1 and vsurge >= 1.5 and above_vwap:
        label, direction = "Strong ORB Breakout", "long"
    elif orb == 1 and vsurge >= 1.2:
        label, direction = "ORB Breakout", "long"
    elif orb == 1:
        label, direction = "ORB Breakout (low vol)", "long"
    elif above_vwap and vsurge >= 1.5:
        label, direction = "VWAP Momentum", "long"
    elif above_vwap and vsurge >= 1.2:
        label, direction = "VWAP Hold", "long"
    elif orb == -1 and vsurge >= 1.2:
        label, direction = "ORB Breakdown (avoid)", "short"
    elif not above_vwap and vsurge >= 1.5:
        label, direction = "Selling Pressure (avoid)", "short"
    else:
        label, direction = "No Setup", "none"

    return score, label, direction


# ── Entry / Stop / Target ─────────────────────────────────────────────────────

def _compute_levels(f: dict, direction: str) -> dict:
    """Return entry, stop, target, rr for the given setup direction."""
    price    = f.get("cur_price")
    orb_high = f.get("orb_high")
    orb_low  = f.get("orb_low")
    vwap     = f.get("vwap")

    if not price or math.isnan(price):
        return dict(entry=None, stop=None, target=None, rr=None)

    if direction == "long":
        if orb_high and orb_low and not math.isnan(orb_high):
            rng    = orb_high - orb_low
            entry  = round(price, 2)
            stop   = round(orb_low * 0.998, 2)          # just below ORB low
            target = round(orb_high + rng * 1.5, 2)     # 1.5× range projection
        elif vwap and not math.isnan(vwap):
            entry  = round(price, 2)
            stop   = round(vwap * 0.995, 2)
            target = round(price * 1.015, 2)
        else:
            return dict(entry=None, stop=None, target=None, rr=None)
    elif direction == "short":
        if orb_high and orb_low and not math.isnan(orb_low):
            rng    = orb_high - orb_low
            entry  = round(price, 2)
            stop   = round(orb_high * 1.002, 2)
            target = round(orb_low - rng * 1.5, 2)
        else:
            return dict(entry=None, stop=None, target=None, rr=None)
    else:
        return dict(entry=None, stop=None, target=None, rr=None)

    risk   = abs(entry - stop)
    reward = abs(target - entry)
    rr     = round(reward / risk, 1) if risk > 0 else None
    return dict(entry=entry, stop=stop, target=target, rr=rr)


# ── Public API ────────────────────────────────────────────────────────────────

def score_symbol(symbol: str) -> dict | None:
    """Score a single symbol for intraday. Returns None if data unavailable."""
    f = get_intraday_features(symbol)
    if not f.get("data_ok"):
        return None

    score, setup, direction = _score_and_classify(f)
    levels = _compute_levels(f, direction)

    return {
        "symbol":    symbol,
        "score":     score,
        "setup":     setup,
        "direction": direction,
        **levels,
        **f,
    }


def scan_intraday(symbols: list[str]) -> pd.DataFrame:
    """Score all symbols and return a ranked DataFrame (best score first).

    Columns: symbol, score, setup, direction, entry, stop, target, rr,
             vwap_ratio, orb_signal, intraday_vol_surge, first_hour_return, ...
    """
    rows = [r for sym in symbols if (r := score_symbol(sym)) is not None]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from swing_v2 import NIFTY_50

    syms = sys.argv[1:] if len(sys.argv) > 1 else NIFTY_50[:10]
    ms   = market_status()
    print(f"\nNSE Market: {'OPEN' if ms['open'] else 'CLOSED'} — {ms['time_ist']}, {ms['date_ist']}")
    print(f"Scanning {len(syms)} symbols…\n{'─'*80}")

    df = scan_intraday(syms)
    if df.empty:
        print("No data available.")
    else:
        cols = ["symbol", "score", "setup", "entry", "stop", "target", "rr",
                "vwap_ratio", "intraday_vol_surge"]
        print(df[cols].to_string(index=False))
    print(f"{'─'*80}")
