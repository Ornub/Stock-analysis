"""
alerts.py (imported as telegram_alert) — WhatsApp push alerts via CallMeBot.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 WHATSAPP SETUP  (free, ~3 min, no payment needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. Save this number in your contacts (name it anything):
      +34 644 59 73 97

 2. Send this exact message to that number on WhatsApp:
      I allow callmebot to send me messages

 3. You'll receive a reply with your API key, e.g.:
      Your CallMeBot API Key is: 1234567

 4. Create a .env file in this folder:
      WHATSAPP_PHONE=91XXXXXXXXXX    ← your number, country code, no +
      WHATSAPP_APIKEY=1234567

 5. Test: python telegram_alert.py
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from urllib.parse import quote

# ── Load .env if present ──────────────────────────────────────────────────────
_env = Path(".env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

WA_PHONE  = os.getenv("WHATSAPP_PHONE",  "")
WA_APIKEY = os.getenv("WHATSAPP_APIKEY", "")


# ── Formatters ────────────────────────────────────────────────────────────────

def format_signal(
    symbol: str,
    signal: str,
    premium: bool = False,
    dir_p: float = 0.0,
    meta_p: float = 0.0,
    nifty_ret: float = 0.0,
    suppressed: bool = False,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    rr: float | None = None,
) -> str:
    """Return a concise alert message for WhatsApp."""
    if suppressed:
        return ""

    emoji  = "📈" if signal == "BUY" else "📉"
    prem   = "⭐ PREMIUM " if premium else ""
    conf_s = f"dir {dir_p:.0%}" + (f"  meta {meta_p:.0%}" if meta_p else "")

    lines = [
        f"{emoji} {prem}{signal}: {symbol}",
        conf_s,
    ]
    if entry_price is not None:
        entry_s = f"Entry ₹{entry_price:,.1f}"
        if stop_price is not None and target_price is not None:
            entry_s += f"  |  SL ₹{stop_price:,.1f}  |  Target ₹{target_price:,.1f}"
            if rr is not None:
                entry_s += f"  |  R:R 1:{rr:.2f}"
        lines.append(entry_s)
    if nifty_ret != 0.0:
        lines.append(f"Nifty {nifty_ret:+.1%}")
    if premium:
        lines.append("Contrarian oversold bounce — power hour")
    return "\n".join(lines)


# ── WhatsApp (CallMeBot — free personal API) ─────────────────────────────────

def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _wa_send(text: str, retries: int = 2) -> bool:
    if not WA_PHONE or not WA_APIKEY or not text:
        return False
    plain = _strip_html(text)
    url = (
        f"https://api.callmebot.com/whatsapp.php"
        f"?phone={WA_PHONE}&text={quote(plain)}&apikey={WA_APIKEY}"
    )
    for attempt in range(retries + 1):
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                if "error" in body.lower() and "message" not in body.lower():
                    print(f"[whatsapp] API error: {body[:120]}")
                    return False
                return True
        except Exception as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                print(f"[whatsapp] send failed after {retries+1} attempts: {exc}")
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def send(text: str) -> dict[str, bool]:
    """Send raw text to WhatsApp. Returns {channel: success}."""
    return {"whatsapp": _wa_send(text)}


def send_signal(
    symbol: str,
    signal: str,
    premium: bool = False,
    dir_p: float = 0.0,
    meta_p: float = 0.0,
    nifty_ret: float = 0.0,
    suppressed: bool = False,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    rr: float | None = None,
) -> dict[str, bool]:
    """Format and send a signal alert. Returns {} if suppressed or unconfigured."""
    msg = format_signal(symbol, signal, premium, dir_p, meta_p, nifty_ret, suppressed,
                        entry_price, stop_price, target_price, rr)
    if not msg:
        return {}
    return send(msg)


def is_configured() -> bool:
    return bool(WA_PHONE and WA_APIKEY)


def test() -> None:
    """Send a test message to WhatsApp."""
    if not WA_PHONE or not WA_APIKEY:
        print("WhatsApp not configured.")
        print("Add WHATSAPP_PHONE and WHATSAPP_APIKEY to .env")
        return
    msg = "✅ Stock-analysis alert test\nNotifications are working."
    ok = _wa_send(msg)
    if ok:
        print("✓ WhatsApp: message sent successfully")
    else:
        print("✗ WhatsApp: send failed — check WHATSAPP_PHONE (country code, no +) and WHATSAPP_APIKEY")


if __name__ == "__main__":
    test()
