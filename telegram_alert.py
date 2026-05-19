"""
telegram_alert.py — Send intraday signal alerts via Telegram and/or WhatsApp.

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TELEGRAM SETUP  (optional, also free)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. Message @BotFather → /newbot → copy the token
 2. Message your bot once, then open:
      https://api.telegram.org/bot<TOKEN>/getUpdates
    Copy the chat_id from the JSON response
 3. Add to .env:
      TELEGRAM_BOT_TOKEN=123456:ABCdef...
      TELEGRAM_CHAT_ID=987654321
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

TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN",  "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",    "")
WA_PHONE   = os.getenv("WHATSAPP_PHONE",       "")
WA_APIKEY  = os.getenv("WHATSAPP_APIKEY",      "")


# ── Formatters ────────────────────────────────────────────────────────────────

def format_signal(
    symbol: str,
    signal: str,
    premium: bool = False,
    dir_p: float = 0.0,
    meta_p: float = 0.0,
    nifty_ret: float = 0.0,
    suppressed: bool = False,
) -> str:
    """Return a concise, emoji-rich alert message."""
    if suppressed:
        return ""   # regime-suppressed: don't alert

    emoji   = "📈" if signal == "BUY" else "📉"
    prem    = "⭐ PREMIUM " if premium else ""
    nifty_s = f"Nifty {nifty_ret:+.1%}" if nifty_ret != 0.0 else ""
    conf_s  = f"dir {dir_p:.0%}" + (f"  meta {meta_p:.0%}" if meta_p else "")

    lines = [
        f"{emoji} <b>{prem}{signal}: {symbol}</b>",
        conf_s,
    ]
    if nifty_s:
        lines.append(nifty_s)
    if premium:
        lines.append("Contrarian oversold bounce — power hour")
    return "\n".join(lines)


# ── Telegram ──────────────────────────────────────────────────────────────────

def _tg_send(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT_ID or not text:
        return False
    try:
        import urllib.request, json
        payload = json.dumps({
            "chat_id":    TG_CHAT_ID,
            "text":       text,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[telegram] send failed: {exc}")
        return False


# ── WhatsApp (CallMeBot — free personal API) ─────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove HTML tags; CallMeBot requires plain text."""
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
                # CallMeBot always returns HTTP 200; check body for errors
                if "error" in body.lower() and "message" not in body.lower():
                    print(f"[whatsapp] API error: {body[:120]}")
                    return False
                return True
        except Exception as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)   # 1s, 2s back-off
            else:
                print(f"[whatsapp] send failed after {retries+1} attempts: {exc}")
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def send(text: str) -> dict[str, bool]:
    """Send raw text to all configured channels. Returns {channel: success}."""
    return {
        "telegram":  _tg_send(text),
        "whatsapp":  _wa_send(text),
    }


def send_signal(
    symbol: str,
    signal: str,
    premium: bool = False,
    dir_p: float = 0.0,
    meta_p: float = 0.0,
    nifty_ret: float = 0.0,
    suppressed: bool = False,
) -> dict[str, bool]:
    """Format and send a signal alert. Returns {} if suppressed or unconfigured."""
    msg = format_signal(symbol, signal, premium, dir_p, meta_p, nifty_ret, suppressed)
    if not msg:
        return {}
    return send(msg)


def test() -> None:
    """Send a test message to all configured channels."""
    msg = "✅ Stock-analysis alert test\nNotifications are working."
    if not TG_TOKEN and not WA_PHONE:
        print("No channels configured.")
        print("WhatsApp: add WHATSAPP_PHONE and WHATSAPP_APIKEY to .env")
        print("Telegram: add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")
        return
    results = send(msg)
    for ch, ok in results.items():
        configured = (ch == "telegram" and TG_TOKEN) or (ch == "whatsapp" and WA_PHONE)
        if not configured:
            continue
        if ok:
            print(f"✓ {ch}: message sent successfully")
        else:
            hints = {
                "whatsapp": "check WHATSAPP_PHONE (country code, no +) and WHATSAPP_APIKEY",
                "telegram": "check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID",
            }
            print(f"✗ {ch}: send failed — {hints[ch]}")


def is_configured() -> bool:
    return bool(TG_TOKEN and TG_CHAT_ID) or bool(WA_PHONE and WA_APIKEY)


if __name__ == "__main__":
    test()
