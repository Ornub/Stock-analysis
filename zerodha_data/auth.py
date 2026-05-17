"""
zerodha_data/auth.py — Daily token manager for Kite Connect.

Usage (run once each morning):
    python zerodha_data/auth.py login

Or from code:
    from zerodha_data.auth import get_kite
    kite = get_kite()          # returns ready-to-use KiteConnect instance
"""

from __future__ import annotations
import json
import sys
from datetime import date, datetime
from pathlib import Path

from kiteconnect import KiteConnect

# ── import credentials ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import api_key, api_secret
except ImportError:
    raise RuntimeError(
        "zerodha_data/config.py not found. Copy config.py and fill in your credentials."
    )

_TOKEN_FILE = Path(__file__).parent / ".token_cache"


def _save_token(access_token: str) -> None:
    _TOKEN_FILE.write_text(json.dumps({
        "access_token": access_token,
        "date": str(date.today()),
    }))


def _load_token() -> str | None:
    """Return today's cached access_token, or None if stale/missing."""
    if not _TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(_TOKEN_FILE.read_text())
        if data.get("date") == str(date.today()):
            return data["access_token"]
    except Exception:
        pass
    return None


def get_kite(access_token: str | None = None) -> KiteConnect:
    """Return an authenticated KiteConnect instance.

    Priority:
      1. access_token argument (if provided)
      2. Today's cached token from .token_cache
      3. Raises TokenExpiredError with instructions
    """
    kite = KiteConnect(api_key=api_key)

    token = access_token or _load_token()
    if token:
        kite.set_access_token(token)
        return kite

    raise TokenExpiredError(
        f"\n{'─'*60}\n"
        f"Zerodha access_token is missing or expired (refreshes daily at 6 AM).\n\n"
        f"  1. Open this URL in your browser:\n"
        f"     {kite.login_url()}\n\n"
        f"  2. Log in with your Zerodha credentials.\n\n"
        f"  3. After redirect, copy the 'request_token' from the URL:\n"
        f"     https://127.0.0.1/?request_token=XXXX&action=login&status=success\n\n"
        f"  4. Run:  python zerodha_data/auth.py <request_token>\n"
        f"{'─'*60}"
    )


class TokenExpiredError(Exception):
    pass


def generate_access_token(request_token: str) -> str:
    """Exchange a request_token for an access_token and cache it."""
    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]
    _save_token(access_token)
    print(f"✓ Access token saved. Valid until 6 AM tomorrow.")
    return access_token


def interactive_login() -> KiteConnect:
    """Full interactive login flow for terminal use."""
    kite = KiteConnect(api_key=api_key)
    print(f"\n{'─'*60}")
    print("Zerodha Kite Connect — Login")
    print(f"{'─'*60}")
    print(f"\nOpen this URL in your browser:\n  {kite.login_url()}\n")
    request_token = input("Paste request_token from redirect URL: ").strip()
    token = generate_access_token(request_token)
    kite.set_access_token(token)
    print("✓ Authenticated successfully.\n")
    return kite


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "login":
        interactive_login()
    elif len(sys.argv) == 2 and sys.argv[1] not in ("login",):
        # Called with request_token directly
        request_token = sys.argv[1]
        generate_access_token(request_token)
    else:
        print("Usage:")
        print("  python zerodha_data/auth.py login           # interactive login")
        print("  python zerodha_data/auth.py <request_token> # exchange token")
