"""
telegram_alert.py — Send intraday signal alerts via Telegram (and optionally WhatsApp).

Setup (one-time):
  Telegram:
    1. Message @BotFather → /newbot → copy the token
    2. Message your bot once, then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
       Copy the "chat_id" from the response
    3. Set env vars or add to .env:
         TELEGRAM_BOT_TOKEN=123456:ABCdef...
         TELEGRAM_CHAT_ID=987654321

  WhatsApp (via CallMeBot — free):
    1. Add +34 644 59 73 97 to your WhatsApp contacts (name: CallMeBot)
    2. Send "I allow callmebot to send me messages" to that number
    3. You'll receive an API key — save it:
         WHATSAPP_PHONE=91XXXXXXXXXX   (with country code, no +)
         WHATSAPP_APIKEY=your_key_here

Usage:
  import telegram_alert as ta
  ta.send_signal("SBIN", "SELL", premium=False, dir_p=0.33, meta_p=0.67, nifty_ret=0.003)
  ta.test()   # sends a test message to verify config
"""
from __future__ import annotations

import os
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


# ── WhatsApp (CallMeBot) ──────────────────────────────────────────────────────

def _wa_send(text: str) -> bool:
    if not WA_PHONE or not WA_APIKEY or not text:
        return False
    # CallMeBot requires plain text (no HTML tags)
    plain = text.replace("<b>", "").replace("</b>", "")
    try:
        import urllib.request
        url = (
            f"https://api.callmebot.com/whatsapp.php"
            f"?phone={WA_PHONE}&text={quote(plain)}&apikey={WA_APIKEY}"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[whatsapp] send failed: {exc}")
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
    """Send a test message to verify Telegram / WhatsApp config."""
    msg = "✅ <b>Stock-analysis alert test</b>\nTelegram notifications are working."
    results = send(msg)
    if not results.get("telegram") and not results.get("whatsapp"):
        if not TG_TOKEN:
            print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        if not WA_PHONE:
            print("WhatsApp not configured. Set WHATSAPP_PHONE and WHATSAPP_APIKEY in .env")
    else:
        for ch, ok in results.items():
            if ok:
                print(f"✓ {ch}: message sent")
            elif (ch == "telegram" and TG_TOKEN) or (ch == "whatsapp" and WA_PHONE):
                print(f"✗ {ch}: send failed — check token/chat_id")


def is_configured() -> bool:
    return bool(TG_TOKEN and TG_CHAT_ID) or bool(WA_PHONE and WA_APIKEY)


if __name__ == "__main__":
    test()
