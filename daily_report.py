"""
daily_report.py — Generate and send an EOD WhatsApp summary.

Intended to be called at ~15:30 IST by scanner_daemon.py or a cron job.
Can also be run manually: python daily_report.py
"""
from __future__ import annotations

from datetime import datetime


def build_report() -> str:
    """Compile today's trading summary into a plain-text WhatsApp message."""
    today = datetime.now().strftime("%d %b %Y")
    lines = [f"📊 Daily Report — {today}"]

    # ── Signal log summary ──────────────────────────────────────────────────
    try:
        import signal_log as sl
        df = sl.get_today()
        total   = len(df)
        buys    = int((df["signal"] == "BUY").sum())  if total else 0
        sells   = int((df["signal"] == "SELL").sum()) if total else 0
        premium = int(df["premium"].sum())             if total else 0
        wins    = int((df["outcome"] == "WIN").sum())  if total else 0
        losses  = int((df["outcome"] == "LOSS").sum()) if total else 0
        pending = int(df["outcome"].isna().sum())       if total else 0
        pnl_closed = df[df["outcome"].notna()]["pnl_pct"].sum() if total else 0.0

        lines.append(
            f"\n📡 Signals: {total} total  ({buys} BUY · {sells} SELL"
            + (f" · ⭐{premium} premium" if premium else "") + ")"
        )
        if wins + losses:
            wr = wins / (wins + losses)
            lines.append(f"  Outcomes: {wins}W / {losses}L  win rate {wr:.0%}")
            lines.append(f"  Closed P&L: {pnl_closed:+.2f}%")
        if pending:
            lines.append(f"  Pending: {pending} signal(s) still open")
    except Exception as exc:
        lines.append(f"\n[signals unavailable: {exc}]")

    # ── Position tracker summary ─────────────────────────────────────────────
    try:
        import position_tracker as pt
        summary = pt.portfolio_summary()
        open_cnt = summary["open_count"]
        if open_cnt:
            lines.append(
                f"\n💼 Open positions: {open_cnt}  "
                f"(invested ₹{summary['total_invested']:,.0f})"
            )
        closed_today = pt.get_closed(n=100)
        if not closed_today.empty:
            today_str = datetime.now().strftime("%Y-%m-%d")
            ct = closed_today[closed_today["exit_ts"].str.startswith(today_str, na=False)]
            if not ct.empty:
                pnl_sum = ct["pnl_inr"].sum()
                lines.append(
                    f"\n✅ Closed today: {len(ct)} position(s)  "
                    f"P&L ₹{pnl_sum:+,.0f}"
                )
                # Best and worst
                best = ct.loc[ct["pnl_pct"].idxmax()]
                worst = ct.loc[ct["pnl_pct"].idxmin()]
                lines.append(
                    f"  Best:  {best['symbol']} {best['pnl_pct']:+.2f}%\n"
                    f"  Worst: {worst['symbol']} {worst['pnl_pct']:+.2f}%"
                )
    except Exception as exc:
        lines.append(f"\n[positions unavailable: {exc}]")

    lines.append("\n⚠ Educational only — not financial advice.")
    return "\n".join(lines)


def send_report() -> bool:
    """Build and send the report. Returns True if sent successfully."""
    msg = build_report()
    try:
        from telegram_alert import send
        results = send(msg)
        ok = any(results.values())
        if ok:
            print("[daily_report] sent successfully")
        else:
            print("[daily_report] no channel delivered the message")
            print(msg)
        return ok
    except Exception as exc:
        print(f"[daily_report] send failed: {exc}")
        print(msg)
        return False


if __name__ == "__main__":
    send_report()
