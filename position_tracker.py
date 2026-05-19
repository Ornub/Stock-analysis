"""
position_tracker.py — Persistent SQLite store for open/closed positions.

Positions are opened from a signal (or manually), tracked with real-time
P&L via Angel One LTP (falling back to yfinance), and auto-closed when
stop or target is hit — with a WhatsApp alert on exit.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

_DB_PATH = Path("data/positions.db")
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _init() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id     INTEGER,
                symbol        TEXT    NOT NULL,
                direction     TEXT    NOT NULL,   -- BUY / SELL
                qty           INTEGER NOT NULL,
                entry_price   REAL    NOT NULL,
                stop_price    REAL,
                target_price  REAL,
                rr            REAL,
                capital_risked REAL,
                invested      REAL,
                status        TEXT    NOT NULL DEFAULT 'OPEN',  -- OPEN / CLOSED
                exit_price    REAL,
                exit_reason   TEXT,               -- HIT_TARGET / HIT_STOP / MANUAL
                exit_ts       TEXT,
                pnl_pct       REAL,
                pnl_inr       REAL,
                entry_ts      TEXT    NOT NULL
            )
        """)


_init()


def _r(v, d: int = 4):
    try:
        return round(float(v), d) if v is not None else None
    except (TypeError, ValueError):
        return None


# ── Write ─────────────────────────────────────────────────────────────────────

def open_position(
    symbol: str,
    direction: str,
    qty: int,
    entry_price: float,
    stop_price: float | None = None,
    target_price: float | None = None,
    rr: float | None = None,
    capital_risked: float | None = None,
    signal_id: int | None = None,
) -> int:
    """Insert an open position. Returns new row id."""
    invested = round(qty * entry_price, 2) if qty and entry_price else None
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO positions
               (signal_id, symbol, direction, qty, entry_price, stop_price,
                target_price, rr, capital_risked, invested, entry_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (signal_id, symbol, direction, qty,
             _r(entry_price, 2), _r(stop_price, 2), _r(target_price, 2),
             _r(rr), _r(capital_risked, 2), _r(invested, 2), ts),
        )
        return cur.lastrowid


def close_position(
    pos_id: int,
    exit_price: float,
    reason: str = "MANUAL",
) -> None:
    """Close a position and compute P&L."""
    with _conn() as con:
        row = con.execute(
            "SELECT direction, qty, entry_price FROM positions WHERE id=?", (pos_id,)
        ).fetchone()
    if not row:
        return

    entry = float(row["entry_price"])
    move  = (float(exit_price) - entry) / entry * 100
    pnl_pct = round(move if row["direction"] == "BUY" else -move, 3)
    pnl_inr = round(row["qty"] * (float(exit_price) - entry) * (1 if row["direction"] == "BUY" else -1), 2)

    with _conn() as con:
        con.execute(
            """UPDATE positions
               SET status=?, exit_price=?, exit_reason=?, exit_ts=?,
                   pnl_pct=?, pnl_inr=?
               WHERE id=?""",
            ("CLOSED", _r(exit_price, 2), reason,
             datetime.now().strftime("%Y-%m-%d %H:%M"),
             pnl_pct, pnl_inr, pos_id),
        )


# ── Read ──────────────────────────────────────────────────────────────────────

def get_open() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM positions WHERE status='OPEN' ORDER BY id DESC", con
        )


def get_closed(n: int = 50) -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM positions WHERE status='CLOSED' ORDER BY id DESC LIMIT ?",
            con, params=(n,),
        )


def get_all(n: int = 100) -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM positions ORDER BY id DESC LIMIT ?", con, params=(n,)
        )


def portfolio_summary() -> dict:
    open_df = get_open()
    with _conn() as con:
        closed = con.execute(
            "SELECT COUNT(*) as n, SUM(pnl_inr) as total_pnl, AVG(pnl_pct) as avg_pnl "
            "FROM positions WHERE status='CLOSED'"
        ).fetchone()
        wins   = con.execute(
            "SELECT COUNT(*) as n FROM positions WHERE status='CLOSED' AND pnl_pct > 0"
        ).fetchone()["n"]
        losses = con.execute(
            "SELECT COUNT(*) as n FROM positions WHERE status='CLOSED' AND pnl_pct <= 0"
        ).fetchone()["n"]

    total_closed = closed["n"] or 0
    return {
        "open_count":   len(open_df),
        "total_invested": round(float(open_df["invested"].sum()), 2) if not open_df.empty else 0.0,
        "closed_count": total_closed,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins / total_closed, 3) if total_closed else None,
        "total_pnl_inr": round(float(closed["total_pnl"]), 2) if closed["total_pnl"] else 0.0,
        "avg_pnl_pct":   round(float(closed["avg_pnl"]),   3) if closed["avg_pnl"]   else 0.0,
    }


# ── Live price + auto-exit ─────────────────────────────────────────────────────

def _ltp(symbol: str) -> float | None:
    """Fetch last traded price: Angel One LTP → yfinance snapshot."""
    try:
        import angel_one as _ao
        if _ao.is_configured():
            p = _ao.ltp(symbol)
            if p:
                return p
    except Exception:
        pass
    try:
        import yfinance as yf
        tk = yf.Ticker(f"{symbol}.NS")
        info = tk.fast_info
        return float(info.last_price)
    except Exception:
        return None


def enrich_open_with_ltp() -> pd.DataFrame:
    """Return open positions DataFrame with current price and unrealized P&L columns."""
    df = get_open()
    if df.empty:
        return df
    ltps = {sym: _ltp(sym) for sym in df["symbol"].unique()}
    df["ltp"] = df["symbol"].map(ltps)
    df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")

    def _upnl(row):
        try:
            entry = float(row["entry_price"])
            cur   = float(row["ltp"])
            move  = (cur - entry) / entry * 100
            pct   = move if row["direction"] == "BUY" else -move
            inr   = row["qty"] * (cur - entry) * (1 if row["direction"] == "BUY" else -1)
            return round(pct, 3), round(inr, 2)
        except Exception:
            return None, None

    df[["unrealized_pct", "unrealized_inr"]] = df.apply(
        lambda r: pd.Series(_upnl(r)), axis=1
    )
    return df


def auto_check_exits(send_alert: bool = True) -> int:
    """
    For each open position, fetch LTP and close it if stop or target is hit.
    Sends a WhatsApp alert for each auto-exit.
    Returns number of positions closed.
    """
    df = get_open()
    if df.empty:
        return 0

    closed = 0
    for _, row in df.iterrows():
        cur = _ltp(row["symbol"])
        if cur is None:
            continue

        stop   = row.get("stop_price")
        target = row.get("target_price")
        direction = row["direction"]
        pos_id = int(row["id"])
        reason = None

        if stop and target:
            if direction == "BUY":
                if cur >= float(target):
                    reason = "HIT_TARGET"
                elif cur <= float(stop):
                    reason = "HIT_STOP"
            else:  # SELL
                if cur <= float(target):
                    reason = "HIT_TARGET"
                elif cur >= float(stop):
                    reason = "HIT_STOP"
        if not reason:
            continue

        close_position(pos_id, cur, reason)
        closed += 1

        if send_alert:
            try:
                from telegram_alert import send
                entry = float(row["entry_price"])
                pnl_pct = (cur - entry) / entry * 100
                if direction == "SELL":
                    pnl_pct = -pnl_pct
                emoji = "✅" if reason == "HIT_TARGET" else "🛑"
                msg = (
                    f"{emoji} {reason}: {row['symbol']}\n"
                    f"{direction} @ ₹{entry:,.1f} → ₹{cur:,.1f}\n"
                    f"P&L: {pnl_pct:+.2f}% (₹{row['qty'] * abs(cur - entry):,.0f})"
                )
                send(msg)
            except Exception:
                pass

    return closed
