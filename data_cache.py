"""
data_cache.py — Pickle-based TTL cache for 5-min bar data.

Eliminates repeated yfinance calls on dashboard refresh.
TTL: 5 min while market is open, 2 hours when closed.
"""
from __future__ import annotations

import pickle
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_CACHE_DIR   = Path("data/cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MARKET_TTL  = timedelta(minutes=5)
_OFFLINE_TTL = timedelta(hours=2)
_LOCK        = threading.Lock()   # safe for ThreadPoolExecutor parallel fetches


def _ttl() -> timedelta:
    try:
        from intraday_scan import market_status
        return _MARKET_TTL if market_status()["open"] else _OFFLINE_TTL
    except Exception:
        return _MARKET_TTL


def _cache_path(symbol: str) -> Path:
    return _CACHE_DIR / f"{symbol}.pkl"


def get_bars(symbol: str, days: int = 5, force: bool = False) -> pd.DataFrame | None:
    """Return 5-min bars, from cache if fresh, else fetch from yfinance and cache."""
    path = _cache_path(symbol)

    if not force and path.exists():
        try:
            with _LOCK:
                with open(path, "rb") as f:
                    cached = pickle.load(f)
            age = datetime.utcnow() - cached["fetched_at"]
            if age < _ttl():
                return cached["df"]
        except Exception:
            pass

    from intraday_model_v2 import fetch_5min
    df = fetch_5min(symbol, days=days)
    if df is not None and not df.empty:
        try:
            with _LOCK:
                with open(path, "wb") as f:
                    # Also drop cached features on new bar data
                    fp = _feat_path(symbol)
                    if fp.exists():
                        fp.unlink()
                    pickle.dump({"df": df, "fetched_at": datetime.utcnow()}, f)
        except Exception:
            pass
    return df


def _feat_path(symbol: str) -> Path:
    return _CACHE_DIR / f"{symbol}_feats.pkl"


def get_features(symbol: str, days: int = 5) -> "pd.DataFrame | None":
    """
    Return computed v3 features for symbol, caching the result.
    Features are invalidated whenever bar data is refreshed.
    """
    fpath = _feat_path(symbol)
    if fpath.exists():
        try:
            with _LOCK:
                with open(fpath, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass

    df = get_bars(symbol, days=days)
    if df is None:
        return None

    from intraday_model_v3 import compute_features_v3
    feats = compute_features_v3(df)
    if feats is not None and not feats.empty:
        try:
            with _LOCK:
                with open(fpath, "wb") as f:
                    pickle.dump(feats, f)
        except Exception:
            pass
    return feats


def get_nifty_bars(days: int = 5, force: bool = False) -> pd.DataFrame | None:
    """Return Nifty 5-min bars, cached."""
    path = _CACHE_DIR / "NIFTY50_5m.pkl"

    if not force and path.exists():
        try:
            with _LOCK:
                with open(path, "rb") as f:
                    cached = pickle.load(f)
            if datetime.utcnow() - cached["fetched_at"] < _ttl():
                return cached["df"]
        except Exception:
            pass

    from intraday_model_v2 import fetch_nifty_5min
    df = fetch_nifty_5min(days=days)
    if df is not None and not df.empty:
        try:
            with _LOCK:
                with open(path, "wb") as f:
                    pickle.dump({"df": df, "fetched_at": datetime.utcnow()}, f)
        except Exception:
            pass
    return df


def invalidate(symbol: str | None = None) -> None:
    """Delete cached file(s) to force fresh fetch on next call."""
    if symbol:
        p = _cache_path(symbol)
        if p.exists():
            p.unlink()
    else:
        for p in _CACHE_DIR.glob("*.pkl"):
            p.unlink()


def cache_info() -> dict:
    """Return per-symbol cache age in seconds, for debugging."""
    info = {}
    now = datetime.utcnow()
    for p in sorted(_CACHE_DIR.glob("*.pkl")):
        try:
            with open(p, "rb") as f:
                cached = pickle.load(f)
            info[p.stem] = round((now - cached["fetched_at"]).total_seconds())
        except Exception:
            info[p.stem] = -1
    return info
