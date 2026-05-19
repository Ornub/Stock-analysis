"""
signal_log.py — Persistent SQLite log for intraday v4 signals.

Every non-HOLD signal that fires during a scan is stored here.
Outcomes (WIN/LOSS) can be marked manually or via update_outcomes().
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
                outcome      TEXT,
                exit_price   REAL,
                outcome_ts   TEXT
            )
        """)


_init()


# ── Write ─────────────────────────────────────────────────────────────────────

def log_signal(
    symbol: str,
    signal: str,
    premium: bool = False,
    dir_p: float = 0.0,
    meta_p: float = 0.0,
    entry_price: float | None = None,
    ts: str | None = None,
    dedup_minutes: int = 30,
) -> int | None:
    """
    Insert a signal row. Returns the new row id, or None if deduplicated.

    Deduplication: skips if same symbol+signal was logged within dedup_minutes.
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
            """INSERT INTO signals (ts, symbol, signal, premium, dir_p, meta_p, entry_price)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ts, symbol, signal, int(premium),
             round(dir_p, 4), round(meta_p, 4), entry_price),
        )
        return cur.lastrowid


def update_outcome(signal_id: int, outcome: str, exit_price: float | None = None) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE signals SET outcome=?, exit_price=?, outcome_ts=? WHERE id=?",
            (outcome, exit_price,
             datetime.now().strftime("%Y-%m-%d %H:%M"), signal_id),
        )


def auto_update_outcomes(win_pct: float = 0.7, loss_pct: float = 0.5) -> int:
    """
    Fetch current prices for pending signals and mark WIN/LOSS.
    WIN  : price moved ≥ win_pct% in signal direction.
    LOSS : price moved ≥ loss_pct% against signal direction.
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

    updated = 0
    for _, row in pending.iterrows():
        sym = row["symbol"]
        entry = row["entry_price"]
        if not entry:
            continue
        try:
            if len(syms) == 1:
                cur = float(prices["Close"].iloc[-1])
            else:
                cur = float(prices[f"{sym}.NS"]["Close"].iloc[-1])
            ret = (cur - entry) / entry * 100
            if row["signal"] == "BUY":
                if ret >= win_pct:
                    update_outcome(row["id"], "WIN",  cur); updated += 1
                elif ret <= -loss_pct:
                    update_outcome(row["id"], "LOSS", cur); updated += 1
            elif row["signal"] == "SELL":
                if ret <= -win_pct:
                    update_outcome(row["id"], "WIN",  cur); updated += 1
                elif ret >= loss_pct:
                    update_outcome(row["id"], "LOSS", cur); updated += 1
        except Exception:
            pass
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
    """Signals without an outcome yet."""
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM signals WHERE outcome IS NULL ORDER BY id DESC",
            con,
        )


def summary() -> dict:
    with _conn() as con:
        rows = con.execute("SELECT outcome, COUNT(*) AS n FROM signals GROUP BY outcome").fetchall()
    counts = {r["outcome"]: r["n"] for r in rows}
    total  = sum(counts.values())
    wins   = counts.get("WIN",  0)
    losses = counts.get("LOSS", 0)
    closed = wins + losses
    return {
        "total":    total,
        "wins":     wins,
        "losses":   losses,
        "pending":  counts.get(None, 0),
        "win_rate": round(wins / closed, 3) if closed else None,
    }
