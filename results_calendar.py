"""
results_calendar.py — Suppress signals on quarterly/annual results days.

Data source: Yahoo Finance (yfinance) earnings dates — one fetch per symbol,
cached in data/results_calendar.json for 24 h.

Usage:
    from results_calendar import is_results_day, get_results_today
    is_results_day("RELIANCE")          # True if RELIANCE announces today
    get_results_today()                 # ["RELIANCE", "TCS", ...]

    python results_calendar.py [YYYY-MM-DD]   # print today's results list
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

_CACHE_PATH = Path("data/results_calendar.json")
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Module-level state
_CALENDAR: dict[str, list[str]] = {}   # {SYMBOL: ["YYYY-MM-DD", ...]}
_LOADED = False


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _fetch_one(symbol: str) -> tuple[str, list[str]]:
    """Return (symbol, [iso_date, ...]) of known earnings dates."""
    try:
        import yfinance as yf
        t   = yf.Ticker(f"{symbol}.NS")
        cal = t.calendar            # dict or None
        dates: list[str] = []
        if cal and "Earnings Date" in cal:
            for d in cal["Earnings Date"]:
                try:
                    if isinstance(d, date):
                        dates.append(d.isoformat())
                    else:
                        dates.append(date.fromisoformat(str(d)[:10]).isoformat())
                except Exception:
                    pass
        return symbol, dates
    except Exception:
        return symbol, []


def _fetch_all(symbols: list[str], max_workers: int = 8) -> dict[str, list[str]]:
    """Fetch earnings dates for all symbols in parallel."""
    cal: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, dates = fut.result()
            if dates:
                cal[sym] = dates
    return cal


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    try:
        if _CACHE_PATH.exists():
            raw = json.loads(_CACHE_PATH.read_text())
            cached_at = datetime.fromisoformat(raw.get("cached_at", "2000-01-01"))
            if datetime.utcnow() - cached_at < timedelta(hours=20):
                return raw
    except Exception:
        pass
    return {}


def _save_cache(cal: dict) -> None:
    try:
        _CACHE_PATH.write_text(
            json.dumps({"cached_at": datetime.utcnow().isoformat(), "calendar": cal},
                       ensure_ascii=False)
        )
    except Exception:
        pass


# ── Calendar loader ───────────────────────────────────────────────────────────

def _ensure_loaded(force: bool = False) -> None:
    global _CALENDAR, _LOADED
    if _LOADED and not force:
        return

    cache = _load_cache()
    if not force and cache.get("calendar"):
        _CALENDAR = cache["calendar"]
        _LOADED   = True
        return

    # Fetch fresh from yfinance
    try:
        from swing_v2 import NIFTY_50
        symbols = NIFTY_50
    except Exception:
        symbols = []

    if symbols:
        cal = _fetch_all(symbols)
    elif cache.get("calendar"):
        cal = cache["calendar"]      # keep stale rather than wipe
    else:
        cal = {}

    if cal:
        _CALENDAR = cal
        _save_cache(cal)
    _LOADED = True


# ── Public API ────────────────────────────────────────────────────────────────

def is_results_day(symbol: str, on_date: date | None = None) -> bool:
    """
    True if `symbol` has a results announcement on `on_date` (default: today).
    Never raises — returns False on any error so signals are never blocked by a
    calendar failure.
    """
    try:
        _ensure_loaded()
        check = (on_date or date.today()).isoformat()
        return check in _CALENDAR.get(symbol.upper(), [])
    except Exception:
        return False


def get_results_today(on_date: date | None = None) -> list[str]:
    """Sorted list of symbols with results today."""
    try:
        _ensure_loaded()
        check = (on_date or date.today()).isoformat()
        return sorted(sym for sym, dates in _CALENDAR.items() if check in dates)
    except Exception:
        return []


def refresh(symbols: list[str] | None = None) -> int:
    """Force-refresh from yfinance. Returns number of symbols with known dates."""
    global _LOADED
    _LOADED = False
    _CACHE_PATH.unlink(missing_ok=True)
    if symbols:
        cal = _fetch_all(symbols)
        _CALENDAR.update(cal)
        _save_cache(_CALENDAR)
        _LOADED = True
        return len(_CALENDAR)
    _ensure_loaded(force=True)
    return len(_CALENDAR)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    check_date = date.today()
    if len(sys.argv) > 1:
        try:
            check_date = date.fromisoformat(sys.argv[1])
        except ValueError:
            print("Usage: python results_calendar.py [YYYY-MM-DD]"); sys.exit(1)

    print("Fetching earnings calendar from Yahoo Finance…")
    n = refresh()
    print(f"Loaded {n} symbols with upcoming results.\n")

    syms = get_results_today(check_date)
    if syms:
        print(f"Results on {check_date}: {', '.join(syms)}")
    else:
        print(f"No results announcements found for {check_date}")

    if len(sys.argv) > 2:
        sym = sys.argv[2].upper()
        print(f"\nis_results_day('{sym}', {check_date}) = {is_results_day(sym, check_date)}")
