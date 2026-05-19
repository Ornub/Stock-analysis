"""
signal_log.py — Persistent SQLite log for intraday v4 signals with P&L tracking.

Every non-HOLD signal stores entry, stop, target, and R:R from the model.
Outcomes (WIN/LOSS) are marked when price hits stop or target.
P&L is calculated as percentage of entry price.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_DB_PATH = Path("data/signal_log.db")
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _init() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           TEXT    NOT NULL,
                symbol       TEXT    NOT NULL,
                signal       TEXT    NOT NULL,
                premium      INTEGER NOT NULL DEFAULT 0,
                dir_p        REAL,
                meta_p       REAL,
                entry_price  REAL,
                stop_price   REAL,
                target_price REAL,
                rr           REAL,
                atr_5m       REAL,
                outcome      TEXT,
                exit_price   REAL,
                pnl_pct      REAL,
                outcome_ts   TEXT
            )
        """)
        # Migrate older DB that lacks the new columns
        existing = {r[1] for r in con.execute("PRAGMA table_info(signals)").fetchall()}
        for col, typedef in [
            ("stop_price",   "REAL"),
            ("target_price", "REAL"),
            ("rr",           "REAL"),
            ("atr_5m",       "REAL"),
            ("pnl_pct",      "REAL"),
        ]:
            if col not in existing:
                con.execute(f"ALTER TABLE signals ADD COLUMN {col} {typedef}")


_init()


# ── Write ─────────────────────────────────────────────────────────────────────

def log_signal(
    symbol: str,
    signal: str,
    premium: bool = False,
    dir_p: float = 0.0,
    meta_p: float = 0.0,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    rr: float | None = None,
    atr_5m: float | None = None,
    ts: str | None = None,
    dedup_minutes: int = 30,
    nifty_ret: float = 0.0,
    alert: bool = True,
) -> int | None:
    """
    Insert a signal row. Returns the new row id, or None if deduplicated.
    When alert=True and the signal is new, sends Telegram/WhatsApp notification.
    """
    ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M")
    cutoff = (datetime.now() - timedelta(minutes=dedup_minutes)).strftime("%Y-%m-%d %H:%M")

    with _conn() as con:
        existing = con.execute(
            "SELECT id FROM signals WHERE symbol=? AND signal=? AND ts>=? LIMIT 1",
            (symbol, signal, cutoff),
        ).fetchone()
        if existing:
            return None

        cur = con.execute(
            """INSERT INTO signals
               (ts, symbol, signal, premium, dir_p, meta_p,
                entry_price, stop_price, target_price, rr, atr_5m)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, symbol, signal, int(premium),
             _r(dir_p), _r(meta_p),
             _r(entry_price), _r(stop_price), _r(target_price),
             _r(rr), _r(atr_5m)),
        )
        row_id = cur.lastrowid

    if alert and row_id:
        try:
            from telegram_alert import send_signal
            send_signal(symbol=symbol, signal=signal, premium=premium,
                        dir_p=dir_p, meta_p=meta_p, nifty_ret=nifty_ret,
                        entry_price=entry_price, stop_price=stop_price,
                        target_price=target_price, rr=rr)
        except Exception:
            pass

    return row_id


def _r(v, digits: int = 4):
    """Round to digits if numeric, else None."""
    try:
        return round(float(v), digits) if v is not None else None
    except (TypeError, ValueError):
        return None


def update_outcome(
    signal_id: int,
    outcome: str,
    exit_price: float | None = None,
) -> None:
    """Mark a signal WIN or LOSS and compute P&L."""
    pnl_pct = None
    if exit_price:
        with _conn() as con:
            row = con.execute(
                "SELECT signal, entry_price FROM signals WHERE id=?", (signal_id,)
            ).fetchone()
        if row and row["entry_price"]:
            entry = float(row["entry_price"])
            move  = (float(exit_price) - entry) / entry * 100
            # P&L is positive for correct direction, negative for wrong
            pnl_pct = round(move if row["signal"] == "BUY" else -move, 3)

    with _conn() as con:
        con.execute(
            """UPDATE signals
               SET outcome=?, exit_price=?, pnl_pct=?, outcome_ts=?
               WHERE id=?""",
            (outcome, _r(exit_price, 2), pnl_pct,
             datetime.now().strftime("%Y-%m-%d %H:%M"), signal_id),
        )


def auto_update_outcomes() -> int:
    """
    Fetch current prices for pending signals and mark WIN/LOSS using
    the stored stop_price / target_price levels.
    Falls back to ±0.7% / ±0.5% if prices weren't stored.
    Returns number of signals updated.
    """
    pending = get_pending()
    if pending.empty:
        return 0

    import yfinance as yf
    syms = pending["symbol"].unique().tolist()
    tickers = " ".join(f"{s}.NS" for s in syms)
    try:
        prices = yf.download(tickers, period="1d", interval="1m",
                             progress=False, group_by="ticker")
    except Exception:
        return 0

    def _cur_price(sym: str) -> float | None:
        try:
            if len(syms) == 1:
                return float(prices["Close"].iloc[-1])
            return float(prices[f"{sym}.NS"]["Close"].iloc[-1])
        except Exception:
            return None

    updated = 0
    for _, row in pending.iterrows():
        entry  = row.get("entry_price")
        stop   = row.get("stop_price")
        target = row.get("target_price")
        if not entry:
            continue

        cur = _cur_price(row["symbol"])
        if cur is None:
            continue

        sig = row["signal"]
        outcome = None

        if stop and target:
            # Use model-defined levels
            if sig == "BUY":
                if cur >= target:
                    outcome = "WIN"
                elif cur <= stop:
                    outcome = "LOSS"
            else:  # SELL
                if cur <= target:
                    outcome = "WIN"
                elif cur >= stop:
                    outcome = "LOSS"
        else:
            # Fallback: fixed % thresholds
            ret = (cur - entry) / entry * 100
            if sig == "BUY":
                outcome = "WIN" if ret >= 0.7 else ("LOSS" if ret <= -0.5 else None)
            else:
                outcome = "WIN" if ret <= -0.7 else ("LOSS" if ret >= 0.5 else None)

        if outcome:
            update_outcome(int(row["id"]), outcome, cur)
            updated += 1

    return updated


# ── Read ──────────────────────────────────────────────────────────────────────

def get_recent(n: int = 50) -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", con, params=(n,)
        )


def get_today() -> pd.DataFrame:
    today = datetime.now().strftime("%Y-%m-%d")
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM signals WHERE ts LIKE ? ORDER BY id DESC",
            con, params=(f"{today}%",),
        )


def get_pending() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM signals WHERE outcome IS NULL ORDER BY id DESC", con,
        )


def summary() -> dict:
    with _conn() as con:
        rows = con.execute("SELECT outcome, COUNT(*) AS n FROM signals GROUP BY outcome").fetchall()
        pnl  = con.execute(
            "SELECT SUM(pnl_pct) as total_pnl, AVG(pnl_pct) as avg_pnl "
            "FROM signals WHERE outcome IS NOT NULL"
        ).fetchone()
    counts  = {r["outcome"]: r["n"] for r in rows}
    total   = sum(counts.values())
    wins    = counts.get("WIN",  0)
    losses  = counts.get("LOSS", 0)
    closed  = wins + losses
    return {
        "total":     total,
        "wins":      wins,
        "losses":    losses,
        "pending":   counts.get(None, 0),
        "win_rate":  round(wins / closed, 3) if closed else None,
        "total_pnl": round(float(pnl["total_pnl"]), 2) if pnl["total_pnl"] else 0.0,
        "avg_pnl":   round(float(pnl["avg_pnl"]),   2) if pnl["avg_pnl"]   else 0.0,
    }
