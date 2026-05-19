"""
angel_one.py — Real-time 5-min OHLCV data via Angel One SmartAPI.

Credentials (add to .env):
  ANGEL_API_KEY       = your SmartAPI key  (create at smartapi.angelbroking.com)
  ANGEL_CLIENT_ID     = your login ID      (e.g. A12345)
  ANGEL_MPIN          = your 4-digit MPIN
  ANGEL_TOTP_SECRET   = TOTP base-32 secret from the QR code

If credentials are absent or login fails, all functions return None
and data_cache falls back to yfinance automatically.
"""
from __future__ import annotations

import os
import json
import pickle
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── Load .env ────────────────────────────────────────────────────────────────
_env = Path(".env")
if _env.exists():
    for _ln in _env.read_text().splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _, _v = _ln.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

API_KEY      = os.getenv("ANGEL_API_KEY",      "")
CLIENT_ID    = os.getenv("ANGEL_CLIENT_ID",    "")
MPIN         = os.getenv("ANGEL_MPIN",         "")
TOTP_SECRET  = os.getenv("ANGEL_TOTP_SECRET",  "")

# ── Session state ─────────────────────────────────────────────────────────────
_SESSION_PATH = Path("data/angel_session.pkl")
_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
_SESSION_LOCK = threading.Lock()
_obj          = None          # SmartConnect instance
_jwt_token    = None
_feed_token   = None
_session_ts   = None          # datetime of last login

# ── ScripMaster token map ─────────────────────────────────────────────────────
_SCRIP_MASTER_PATH = Path("data/angel_scrip_master.pkl")
_SCRIP_CACHE: dict[str, str] = {}   # symbol → numeric token
_SCRIP_LOCK  = threading.Lock()
_SCRIP_URL   = ("https://margincalculator.angelbroking.com"
                "/OpenAPI_File/files/OpenAPIScripMaster.json")

# Known Nifty 50 index token (NSE segment)
_NIFTY_TOKEN = "99926000"
_NIFTY_EXCH  = "NSE"


def is_configured() -> bool:
    return bool(API_KEY and CLIENT_ID and MPIN and TOTP_SECRET)


# ── Auth ──────────────────────────────────────────────────────────────────────

def _login() -> bool:
    global _obj, _jwt_token, _feed_token, _session_ts
    if not is_configured():
        return False
    try:
        import pyotp
        from SmartApi import SmartConnect
        obj  = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_SECRET).now()
        data = obj.generateSession(CLIENT_ID, MPIN, totp)
        if not data or not data.get("status"):
            return False
        _obj        = obj
        _jwt_token  = data["data"]["jwtToken"]
        _feed_token = obj.getfeedToken()
        _session_ts = datetime.utcnow()
        return True
    except Exception as exc:
        print(f"[angel_one] login failed: {exc}")
        return False


def _ensure_session() -> bool:
    """Return True if a valid session exists, refreshing if stale (>6 h)."""
    global _session_ts
    with _SESSION_LOCK:
        if _obj and _session_ts and (datetime.utcnow() - _session_ts) < timedelta(hours=6):
            return True
        return _login()


# ── ScripMaster ───────────────────────────────────────────────────────────────

def _load_scrip_master() -> None:
    """Download and cache the full scrip master, refreshed daily."""
    with _SCRIP_LOCK:
        if _SCRIP_CACHE:
            return
        # Try local pickle first (valid for 24 h)
        if _SCRIP_MASTER_PATH.exists():
            age = datetime.utcnow() - datetime.utcfromtimestamp(_SCRIP_MASTER_PATH.stat().st_mtime)
            if age < timedelta(hours=24):
                try:
                    with open(_SCRIP_MASTER_PATH, "rb") as f:
                        _SCRIP_CACHE.update(pickle.load(f))
                    return
                except Exception:
                    pass
        # Fetch fresh
        try:
            import urllib.request
            with urllib.request.urlopen(_SCRIP_URL, timeout=20) as r:
                data = json.loads(r.read())
            mapping = {}
            for row in data:
                exch  = row.get("exch_seg", "")
                sym   = row.get("symbol",   "")
                token = row.get("token",    "")
                if exch == "NSE" and sym and token:
                    # Strip -EQ suffix to match plain symbols like SBIN, RELIANCE
                    clean = sym.replace("-EQ", "").replace("-BE", "")
                    mapping[clean] = token
                    mapping[sym]   = token          # also keep original
            _SCRIP_CACHE.update(mapping)
            with open(_SCRIP_MASTER_PATH, "wb") as f:
                pickle.dump(dict(_SCRIP_CACHE), f)
        except Exception as exc:
            print(f"[angel_one] scrip master load failed: {exc}")


def _get_token(symbol: str) -> str | None:
    _load_scrip_master()
    return _SCRIP_CACHE.get(symbol) or _SCRIP_CACHE.get(f"{symbol}-EQ")


# ── Data fetch ────────────────────────────────────────────────────────────────

def _to_df(raw_data: list) -> pd.DataFrame | None:
    if not raw_data:
        return None
    df = pd.DataFrame(raw_data, columns=["DateTime", "Open", "High", "Low", "Close", "Volume"])
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
    df = df.sort_values("DateTime").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["DateTime", "Open", "High", "Low", "Close", "Volume"]]


def fetch_5min(symbol: str, days: int = 5) -> pd.DataFrame | None:
    """Return 5-min OHLCV bars for an NSE equity symbol."""
    if not _ensure_session():
        return None
    token = _get_token(symbol)
    if not token:
        print(f"[angel_one] token not found for {symbol}")
        return None
    try:
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=min(days, 90))
        params  = {
            "exchange":    "NSE",
            "symboltoken": token,
            "interval":    "FIVE_MINUTE",
            "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate":      to_dt.strftime("%Y-%m-%d %H:%M"),
        }
        resp = _obj.getCandleData(params)
        if not resp or not resp.get("status"):
            return None
        return _to_df(resp.get("data") or [])
    except Exception as exc:
        print(f"[angel_one] fetch_5min({symbol}): {exc}")
        return None


def fetch_nifty_5min(days: int = 5) -> pd.DataFrame | None:
    """Return 5-min close bars for Nifty 50 index."""
    if not _ensure_session():
        return None
    try:
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=min(days, 90))
        params  = {
            "exchange":    _NIFTY_EXCH,
            "symboltoken": _NIFTY_TOKEN,
            "interval":    "FIVE_MINUTE",
            "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate":      to_dt.strftime("%Y-%m-%d %H:%M"),
        }
        resp = _obj.getCandleData(params)
        if not resp or not resp.get("status"):
            return None
        df = _to_df(resp.get("data") or [])
        if df is None:
            return None
        return df[["DateTime", "Close"]].rename(columns={"Close": "nifty_close"})
    except Exception as exc:
        print(f"[angel_one] fetch_nifty_5min: {exc}")
        return None


def ltp(symbol: str) -> float | None:
    """Return last traded price for a symbol (faster than full candle fetch)."""
    if not _ensure_session():
        return None
    token = _get_token(symbol)
    if not token:
        return None
    try:
        resp = _obj.ltpData("NSE", f"{symbol}-EQ", token)
        if resp and resp.get("status"):
            return float(resp["data"]["ltp"])
    except Exception as exc:
        print(f"[angel_one] ltp({symbol}): {exc}")
    return None
