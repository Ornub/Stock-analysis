"""
scanner_daemon.py — Background signal scanner with Telegram alerts.

Runs every INTERVAL seconds during NSE market hours.
Sends Telegram / WhatsApp alerts when new v4 signals fire.
Deduplication is handled by signal_log (30-min window per symbol+signal).

Usage:
  python scanner_daemon.py                # scan every 5 min (default)
  python scanner_daemon.py --interval 3   # scan every 3 min
  python scanner_daemon.py --once         # single scan then exit (for cron)
  python scanner_daemon.py --symbols RELIANCE,TCS,INFY  # custom list

Run at startup (Linux):
  Add to crontab:  @reboot cd /path/to/Stock-analysis && python scanner_daemon.py &

Or systemd / tmux / screen for persistent background running.
"""
from __future__ import annotations

import argparse
import signal as _signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from intraday_scan import market_status
from intraday_model_v3 import batch_predict_parallel
from signal_log import log_signal
from telegram_alert import send_signal, is_configured, format_signal
from swing_v2 import NIFTY_50

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Intraday signal scanner daemon")
_parser.add_argument("--interval", type=int, default=5,
                     help="Scan interval in minutes (default 5)")
_parser.add_argument("--once",    action="store_true",
                     help="Run a single scan then exit")
_parser.add_argument("--symbols", type=str, default="",
                     help="Comma-separated symbol list (default: all 44 trained symbols)")
_parser.add_argument("--quiet",   action="store_true",
                     help="Suppress per-scan stdout output")

args = _parser.parse_args()

INTERVAL_S = args.interval * 60
ONCE       = args.once
QUIET      = args.quiet

# Load the trained symbol list from the pkl, fallback to Nifty-50
try:
    import joblib, __main__
    from swing_v2 import LGBMEnsemble
    __main__.LGBMEnsemble = LGBMEnsemble
    _blob = joblib.load("models/intraday_v3.pkl")
    BASE_SYMBOLS = _blob.get("sym_list", NIFTY_50)
except Exception:
    BASE_SYMBOLS = NIFTY_50

if args.symbols:
    SYMBOLS = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
else:
    SYMBOLS = BASE_SYMBOLS


# ── Graceful shutdown ─────────────────────────────────────────────────────────
_running = True

def _handle_sigterm(signum, frame):
    global _running
    print("\n[daemon] shutting down…")
    _running = False

_signal.signal(_signal.SIGTERM, _handle_sigterm)
_signal.signal(_signal.SIGINT,  _handle_sigterm)


# ── Single scan ───────────────────────────────────────────────────────────────

def run_scan() -> list[dict]:
    """
    Scan all symbols, log new signals, send alerts.
    Returns list of new (non-deduped) signal dicts.
    """
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fired  = []

    results = batch_predict_parallel(SYMBOLS, max_workers=10)

    for sym, r in results.items():
        sig = r.get("signal", "HOLD")
        if sig == "HOLD" or not r.get("data_ok", True):
            continue

        sid = log_signal(
            symbol   = sym,
            signal   = sig,
            premium  = bool(r.get("premium",  False)),
            dir_p    = float(r.get("dir_proba",  0.0)),
            meta_p   = float(r.get("meta_proba", 0.0)),
            ts       = ts_now,
        )

        if sid is None:
            continue   # duplicate within 30-min window, skip alert

        ndr = float(r.get("nifty_day_ret", 0.0))
        send_signal(
            symbol    = sym,
            signal    = sig,
            premium   = bool(r.get("premium", False)),
            dir_p     = float(r.get("dir_proba",  0.0)),
            meta_p    = float(r.get("meta_proba", 0.0)),
            nifty_ret = ndr,
        )
        fired.append({"sym": sym, "signal": sig,
                      "premium": r.get("premium", False),
                      "dir_p": r.get("dir_proba", 0.0)})

    return fired


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[daemon] started — {len(SYMBOLS)} symbols, "
          f"interval={args.interval}m, alerts={'ON' if is_configured() else 'OFF (no .env)'}")
    if not is_configured():
        print("[daemon] tip: create .env with TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable alerts")

    while _running:
        ms = market_status()

        if not ms["open"]:
            if not ONCE:
                if not QUIET:
                    print(f"[{datetime.now():%H:%M}] market closed — sleeping {args.interval}m")
                time.sleep(INTERVAL_S)
                continue

        ts = datetime.now().strftime("%H:%M")
        try:
            fired = run_scan()
        except Exception as exc:
            print(f"[{ts}] scan error: {exc}")
            fired = []

        if not QUIET:
            if fired:
                for f in fired:
                    prem = "★ " if f["premium"] else ""
                    print(f"[{ts}] SIGNAL  {prem}{f['signal']:<5} {f['sym']}  dir={f['dir_p']:.0%}")
            else:
                syms_str = f"{len(SYMBOLS)} syms"
                print(f"[{ts}] scan OK — {syms_str}, no new signals")

        if ONCE:
            break

        time.sleep(INTERVAL_S)

    print("[daemon] stopped")


if __name__ == "__main__":
    main()
