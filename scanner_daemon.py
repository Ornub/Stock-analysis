"""
scanner_daemon.py — Background signal scanner + position monitor + EOD report.

Runs every INTERVAL minutes during NSE market hours.
  • Scans all symbols for new v4 signals → logs + WhatsApp alert
  • Monitors open positions → auto-closes on stop/target → WhatsApp exit alert
  • Sends EOD WhatsApp report once per day at 15:30

Usage:
  python scanner_daemon.py                # scan every 5 min (default)
  python scanner_daemon.py --interval 3   # scan every 3 min
  python scanner_daemon.py --once         # single scan then exit (for cron)
  python scanner_daemon.py --symbols RELIANCE,TCS,INFY  # custom list
  python scanner_daemon.py --no-positions # skip position monitoring
  python scanner_daemon.py --no-eod       # skip EOD report

Run at startup (Linux):
  Add to crontab:  @reboot cd /path/to/Stock-analysis && python scanner_daemon.py &
  Or use systemd / tmux / screen for persistent background running.
"""
from __future__ import annotations

import argparse
import signal as _signal
import sys
import time
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from intraday_scan import market_status
from intraday_model_v3 import batch_predict_parallel
from signal_log import log_signal
from telegram_alert import send_signal, is_configured
from swing_v2 import NIFTY_50

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Intraday signal scanner daemon")
_parser.add_argument("--interval",      type=int, default=5)
_parser.add_argument("--once",          action="store_true")
_parser.add_argument("--symbols",       type=str, default="")
_parser.add_argument("--quiet",         action="store_true")
_parser.add_argument("--no-positions",  action="store_true", help="Disable position monitoring")
_parser.add_argument("--no-eod",        action="store_true", help="Disable EOD WhatsApp report")
args = _parser.parse_args()

INTERVAL_S   = args.interval * 60
ONCE         = args.once
QUIET        = args.quiet
DO_POSITIONS = not args.no_positions
DO_EOD       = not args.no_eod

# ── Symbol list ───────────────────────────────────────────────────────────────
try:
    import joblib, __main__
    from swing_v2 import LGBMEnsemble
    __main__.LGBMEnsemble = LGBMEnsemble
    _blob = joblib.load("models/intraday_v3.pkl")
    BASE_SYMBOLS = _blob.get("sym_list", NIFTY_50)
except Exception:
    BASE_SYMBOLS = NIFTY_50

SYMBOLS = (
    [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols else BASE_SYMBOLS
)

# ── Graceful shutdown ─────────────────────────────────────────────────────────
_running = True

def _handle_sigterm(signum, frame):
    global _running
    print("\n[daemon] shutting down…")
    _running = False

_signal.signal(_signal.SIGTERM, _handle_sigterm)
_signal.signal(_signal.SIGINT,  _handle_sigterm)


# ── Single signal scan ────────────────────────────────────────────────────────

def run_scan() -> list[dict]:
    """Scan all symbols, log new signals, send WhatsApp alerts."""
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fired  = []

    results = batch_predict_parallel(SYMBOLS, max_workers=10)

    for sym, r in results.items():
        sig = r.get("signal", "HOLD")
        if sig == "HOLD" or not r.get("data_ok", True):
            continue

        sid = log_signal(
            symbol       = sym,
            signal       = sig,
            premium      = bool(r.get("premium",      False)),
            dir_p        = float(r.get("dir_proba",   0.0)),
            meta_p       = float(r.get("meta_proba",  0.0)),
            entry_price  = r.get("entry_price"),
            stop_price   = r.get("stop_price"),
            target_price = r.get("target_price"),
            rr           = r.get("rr"),
            atr_5m       = r.get("atr_5m"),
            ts           = ts_now,
            nifty_ret    = float(r.get("nifty_day_ret", 0.0)),
            alert        = True,   # send_signal is called inside log_signal
        )

        if sid is None:
            continue   # duplicate within 30-min window

        fired.append({
            "sym":     sym,
            "signal":  sig,
            "premium": r.get("premium", False),
            "dir_p":   r.get("dir_proba", 0.0),
        })

    return fired


# ── Position exit monitor ─────────────────────────────────────────────────────

def run_position_check() -> int:
    """Check all open positions; auto-close those that hit stop or target."""
    try:
        from position_tracker import auto_check_exits
        return auto_check_exits(send_alert=True)
    except Exception as exc:
        print(f"[daemon] position check error: {exc}")
        return 0


# ── EOD report ────────────────────────────────────────────────────────────────

_eod_sent_date: date | None = None

def maybe_send_eod() -> None:
    """Send the EOD report once, at or after 15:30 IST, once per calendar day."""
    global _eod_sent_date
    now = datetime.now()
    if now.hour < 15 or (now.hour == 15 and now.minute < 30):
        return
    today = now.date()
    if _eod_sent_date == today:
        return
    _eod_sent_date = today
    try:
        from daily_report import send_report
        send_report()
        print(f"[daemon] EOD report sent")
    except Exception as exc:
        print(f"[daemon] EOD report error: {exc}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parts = [f"{len(SYMBOLS)} symbols", f"interval={args.interval}m",
             f"alerts={'ON' if is_configured() else 'OFF'}",
             f"positions={'ON' if DO_POSITIONS else 'OFF'}",
             f"eod={'ON' if DO_EOD else 'OFF'}"]
    print(f"[daemon] started — {', '.join(parts)}")
    if not is_configured():
        print("[daemon] tip: add WHATSAPP_PHONE + WHATSAPP_APIKEY to .env")

    while _running:
        ms = market_status()

        if not ms["open"]:
            # Still run EOD check even when market is closed
            if DO_EOD:
                maybe_send_eod()
            if not ONCE:
                if not QUIET:
                    print(f"[{datetime.now():%H:%M}] market closed — sleeping {args.interval}m")
                time.sleep(INTERVAL_S)
                continue

        ts = datetime.now().strftime("%H:%M")

        # 1. Signal scan
        try:
            fired = run_scan()
        except Exception as exc:
            print(f"[{ts}] scan error: {exc}")
            fired = []

        # 2. Position exit monitor
        exits = 0
        if DO_POSITIONS:
            exits = run_position_check()

        # 3. EOD report
        if DO_EOD:
            maybe_send_eod()

        if not QUIET:
            signal_str = " ".join(
                f"{'★' if f['premium'] else ''}{f['signal']} {f['sym']}"
                for f in fired
            ) or "no new signals"
            exit_str = f"  exits={exits}" if exits else ""
            print(f"[{ts}] {signal_str}{exit_str}")

        if ONCE:
            break

        time.sleep(INTERVAL_S)

    print("[daemon] stopped")


if __name__ == "__main__":
    main()
