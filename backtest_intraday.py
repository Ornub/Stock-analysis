"""
backtest_intraday.py — Walk-forward intraday backtest with ₹100 starting capital.

Simulates every signal fired by the v6.0 model on historical 5-min data:
  - Entry  : close price of signal bar
  - Exit   : first touch of target OR stop in next TARGET_BARS bars;
             if neither hit, exit at bar-8 close (time-based)
  - Broker : 0.10 % round-trip (Zerodha intraday + STT + charges)
  - Capital: ₹100 start, fully deployed per trade, compounded

Outputs a weekly P&L table + summary stats.

Usage:
    python backtest_intraday.py                        # all Nifty-50 symbols
    python backtest_intraday.py --symbols RELIANCE,TCS
    python backtest_intraday.py --no-dedup             # allow clustered signals
    python backtest_intraday.py --capital 10000        # change starting capital
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# ── CLI ───────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--symbols",  default="")
_p.add_argument("--capital",  type=float, default=100.0)
_p.add_argument("--no-dedup", action="store_true", help="allow >1 signal per symbol/day")
_p.add_argument("--all-data", action="store_true",
                help="use full 60-day window (note: first 90%% is in-sample)")
_p.add_argument("--json", action="store_true",
                help="output JSON result to stdout (for dashboard integration)")
args = _p.parse_args()

BROKERAGE_RT = 0.0010   # 0.10% round-trip (Zerodha intraday + STT + charges)
CAPITAL_0    = args.capital

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/intraday_v3.pkl")
if not MODEL_PATH.exists():
    print("Model not found — run: python intraday_model_v3.py train"); sys.exit(1)

import __main__
from swing_v2 import LGBMEnsemble, NIFTY_50
__main__.LGBMEnsemble = LGBMEnsemble
blob = joblib.load(MODEL_PATH)

from intraday_model_v3 import (
    compute_features_v3, make_labels_atr, _apply_pipeline_batch,
    V3_FEATURE_COLS, TRAIN_FRAC, META_FRAC, VAL_FRAC, TARGET_BARS,
    BUY_ATR_MULT, SELL_ATR_MULT, ATR_LABEL_MIN,
)
from intraday_model_v2 import _dir_proba_ens, HOLD_THRESHOLD, DIR_THRESHOLD

SYMBOLS = (
    [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols else blob.get("sym_list", NIFTY_50)
)

oot_note = "TRUE OOT (last 10%)" if not args.all_data else "FULL DATA (90% in-sample)"
if not args.json:
    print("=" * 65)
    print(f"  Intraday Backtest  v{blob.get('version','?')}  |  "
          f"Capital ₹{CAPITAL_0:,.0f}  |  {len(SYMBOLS)} symbols")
    print(f"  Brokerage: {BROKERAGE_RT:.2%} round-trip  |  "
          f"Exit: target/stop/8-bar-close")
    print(f"  Window: {oot_note}")
    print("=" * 65)


# ── Trade simulation ──────────────────────────────────────────────────────────

def _sim_trade(
    direction: str,
    entry: float,
    atr: float,
    future_bars: pd.DataFrame,
) -> tuple[float, str]:
    """
    Walk forward up to TARGET_BARS bars.
    Returns (exit_price, exit_reason) where reason = TARGET | STOP | TIME.
    """
    if direction == "BUY":
        stop_p   = entry - 1.0 * atr
        target_p = entry + 1.5 * atr
    else:
        stop_p   = entry + 1.5 * atr
        target_p = entry - 2.0 * atr

    for _, bar in future_bars.iterrows():
        h, l = float(bar["High"]), float(bar["Low"])
        if direction == "BUY":
            if h >= target_p: return target_p, "TARGET"
            if l <= stop_p:   return stop_p,   "STOP"
        else:
            if l <= target_p: return target_p, "TARGET"
            if h >= stop_p:   return stop_p,   "STOP"

    # Time exit: last bar close
    return float(future_bars["Close"].iloc[-1]), "TIME"


# ── Per-symbol backtest ───────────────────────────────────────────────────────

all_trades: list[dict] = []
skipped = 0

for sym in SYMBOLS:
    try:
        from data_cache import get_bars
        raw = get_bars(sym, days=60)
    except Exception:
        try:
            from intraday_model_v2 import fetch_5min
            raw = fetch_5min(sym, days=60)
        except Exception:
            skipped += 1; continue

    if raw is None or len(raw) < 200:
        skipped += 1; continue

    feats = compute_features_v3(raw)
    if feats is None or feats.empty:
        skipped += 1; continue

    raw_a = raw.iloc[-len(feats):].reset_index(drop=True)
    feats  = feats.reset_index(drop=True)

    # ATR for stop/target sizing
    closes = raw_a["Close"].astype(float)
    highs  = raw_a["High"].astype(float)
    lows   = raw_a["Low"].astype(float)
    prev_c = closes.shift(1).fillna(closes)
    tr     = pd.concat([highs - lows,
                        (highs - prev_c).abs(),
                        (lows  - prev_c).abs()], axis=1).max(axis=1)
    atr14  = tr.rolling(14, min_periods=5).mean().bfill()

    # Evaluation window
    n  = len(feats)
    if args.all_data:
        start_i = 0
    else:
        start_i = int(n * (TRAIN_FRAC + META_FRAC + VAL_FRAC))

    eval_f   = feats.iloc[start_i : n - TARGET_BARS]
    eval_raw = raw_a.iloc[start_i : n - TARGET_BARS]

    if len(eval_f) < 10:
        skipped += 1; continue

    sig_df = _apply_pipeline_batch(eval_f, blob)

    # Dedup: keep first signal per symbol per date
    seen_dates: set[str] = set()

    for pos, (_, row) in enumerate(sig_df.iterrows()):
        sig = row["signal"]
        if sig == "HOLD":
            continue

        bar_raw  = eval_raw.iloc[pos]
        dt_val   = bar_raw.get("DateTime", None)
        if dt_val is None and "DateTime" in raw_a.columns:
            dt_val = raw_a["DateTime"].iloc[start_i + pos]
        dt_str   = str(dt_val)[:10] if dt_val is not None else "unknown"

        if not args.no_dedup:
            key = f"{sym}_{dt_str}_{sig}"
            if key in seen_dates:
                continue
            seen_dates.add(key)

        entry_price = float(bar_raw["Close"])
        atr_val     = float(atr14.iloc[start_i + pos])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        future_slice = raw_a.iloc[start_i + pos + 1 : start_i + pos + 1 + TARGET_BARS]
        if len(future_slice) < 1:
            continue

        exit_price, exit_reason = _sim_trade(sig, entry_price, atr_val, future_slice)

        # P&L
        if sig == "BUY":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        net_pct = pnl_pct - BROKERAGE_RT

        # ISO week
        try:
            d = date.fromisoformat(dt_str)
            iso_week = f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"
        except Exception:
            iso_week = "unknown"

        all_trades.append({
            "symbol":   sym,
            "date":     dt_str,
            "week":     iso_week,
            "signal":   sig,
            "premium":  bool(row["premium"]),
            "entry":    round(entry_price, 2),
            "exit":     round(exit_price,  2),
            "exit_why": exit_reason,
            "pnl_pct":  round(pnl_pct  * 100, 3),
            "net_pct":  round(net_pct  * 100, 3),
            "win":      net_pct > 0,
        })

if not all_trades:
    if args.json:
        import json as _json
        print(_json.dumps({"error": "No trades generated"}))
    else:
        print("\n  No trades generated. Check data availability.")
    sys.exit(0)

df = pd.DataFrame(all_trades)

# ── Weekly summary ────────────────────────────────────────────────────────────

if not args.json:
    print(f"\n── Per-Trade Log ({'dedup: 1st signal/sym/day' if not args.no_dedup else 'all signals'}) ──")
    print(f"  {'Date':<12} {'Sym':<13} {'Sig':<5} {'Entry':>8} {'Exit':>8} "
          f"{'Why':<7} {'Gross%':>7} {'Net%':>7}")
    print("  " + "-" * 72)
    for _, r in df.iterrows():
        prem = "⭐" if r["premium"] else ""
        print(f"  {r['date']:<12} {r['symbol']:<13} {prem}{r['signal']:<4} "
              f"{r['entry']:>8,.1f} {r['exit']:>8,.1f} "
              f"{r['exit_why']:<7} {r['pnl_pct']:>+6.2f}% {r['net_pct']:>+6.2f}%")

# ── Daily portfolio returns ───────────────────────────────────────────────────
# Equal-weight allocation across all signals that day.
# If 5 signals fire on one day, ₹100 is split equally → each gets ₹20.
# Day return = average of all trade net_pct values that day.

daily_stats = (
    df.groupby("date")
      .agg(
          n_trades = ("net_pct", "count"),
          n_wins   = ("win",     "sum"),
          day_ret  = ("net_pct", "mean"),   # equal-weight portfolio return
      )
      .reset_index()
      .sort_values("date")
)

capital   = CAPITAL_0
cum_cap   = []
for _, row in daily_stats.iterrows():
    capital *= (1 + row["day_ret"] / 100)
    cum_cap.append(capital)
daily_stats["capital"] = cum_cap
daily_stats["day_ret_cap"] = daily_stats["capital"].diff().fillna(
    daily_stats["capital"].iloc[0] - CAPITAL_0
)

# ── Weekly rollup using corrected daily compounding ───────────────────────────
df["date_dt"] = pd.to_datetime(df["date"])
daily_stats["date_dt"] = pd.to_datetime(daily_stats["date"])
daily_stats["week"] = daily_stats["date_dt"].dt.strftime("%G-W%V")

if not args.json:
    print(f"\n── Daily P&L  (₹{CAPITAL_0:,.0f} split equally across all signals each day) ──")
    print(f"  {'Date':<12} {'Trades':>7} {'Wins':>5} {'WinRate':>8} "
          f"{'DayRet%':>9} {'Capital':>10}")
    print("  " + "-" * 55)
    for _, row in daily_stats.iterrows():
        wr  = row["n_wins"] / row["n_trades"] if row["n_trades"] else 0
        ind = "▲" if row["day_ret"] >= 0 else "▼"
        print(f"  {row['date']:<12} {int(row['n_trades']):>7} {int(row['n_wins']):>5} "
              f"{wr:>8.0%} {ind}{abs(row['day_ret']):>7.2f}% ₹{row['capital']:>9,.2f}")

    print(f"\n── Weekly Summary ───────────────────────────────────────────────────────")
    print(f"  {'Week':<12} {'Days':>5} {'Trades':>7} {'WinRate':>8} {'Week%':>8} {'Capital':>10}")
    print("  " + "-" * 55)
    for wk, wdf_d in daily_stats.groupby("week"):
        wdf_t    = df[df["date"].isin(wdf_d["date"])]
        n_t      = len(wdf_t)
        n_w      = wdf_t["win"].sum()
        start_c  = wdf_d["capital"].iloc[0] / (1 + wdf_d["day_ret"].iloc[0] / 100)
        end_c    = wdf_d["capital"].iloc[-1]
        wk_ret   = (end_c - start_c) / start_c * 100
        print(f"  {wk:<12} {len(wdf_d):>5} {n_t:>7} {n_w/n_t:>8.0%} "
              f"{wk_ret:>+7.2f}% ₹{end_c:>9,.2f}")

# ── Overall summary ───────────────────────────────────────────────────────────
total_trades = len(df)
total_wins   = df["win"].sum()
avg_trade    = df["net_pct"].mean()
best_trade   = df.loc[df["net_pct"].idxmax()]
worst_trade  = df.loc[df["net_pct"].idxmin()]
final_cap    = daily_stats["capital"].iloc[-1]
total_return = (final_cap - CAPITAL_0) / CAPITAL_0 * 100
n_weeks      = daily_stats["week"].nunique()
avg_weekly   = total_return / n_weeks if n_weeks else 0
exits        = df["exit_why"].value_counts()

if not args.json:
    print(f"\n{'─'*65}")
    print(f"  Symbols: {len(SYMBOLS)}  |  Skipped: {skipped}  |  "
          f"Period: {df['date'].min()} → {df['date'].max()}")
    print(f"  Total trades  : {total_trades}  "
          f"({exits.get('TARGET',0)} targets  {exits.get('STOP',0)} stops  "
          f"{exits.get('TIME',0)} time-exits)")
    print(f"  Win rate      : {int(total_wins)}/{total_trades} = {total_wins/total_trades:.1%}")
    print(f"  Avg per trade : {avg_trade:+.2f}% net")
    print(f"  Best trade    : {best_trade['symbol']} {best_trade['signal']} "
          f"{best_trade['net_pct']:+.2f}%  ({best_trade['date']})")
    print(f"  Worst trade   : {worst_trade['symbol']} {worst_trade['signal']} "
          f"{worst_trade['net_pct']:+.2f}%  ({worst_trade['date']})")
    print(f"\n  Starting capital : ₹{CAPITAL_0:,.2f}")
    print(f"  Final capital    : ₹{final_cap:,.2f}")
    print(f"  Total return     : {total_return:+.2f}%  over {n_weeks} week(s)")
    print(f"  Avg per week     : {avg_weekly:+.2f}%")
    if not args.all_data:
        print(f"\n  ⚠  Evaluated on TRUE OOT (last 10%) only — ~{n_weeks} week(s) of data.")
        print(f"     Run with --all-data for more weeks (but first 90% is in-sample).")

# ── JSON output (for dashboard) ──────────────────────────────────────────────
if args.json:
    import json as _json

    _weekly_rows = []
    for wk, wdf_d in daily_stats.groupby("week"):
        wdf_t   = df[df["date"].isin(wdf_d["date"])]
        n_t     = len(wdf_t)
        n_w     = int(wdf_t["win"].sum())
        start_c = wdf_d["capital"].iloc[0] / (1 + wdf_d["day_ret"].iloc[0] / 100)
        end_c   = float(wdf_d["capital"].iloc[-1])
        wk_ret  = (end_c - start_c) / start_c * 100
        _weekly_rows.append({
            "week": wk, "days": len(wdf_d), "trades": n_t,
            "win_rate_pct": round(n_w / n_t * 100, 1) if n_t else 0,
            "week_ret": round(wk_ret, 2), "capital": round(end_c, 2),
        })

    _daily_rows = [
        {
            "date": row["date"],
            "trades": int(row["n_trades"]),
            "wins": int(row["n_wins"]),
            "win_rate_pct": round(row["n_wins"] / row["n_trades"] * 100, 1) if row["n_trades"] else 0,
            "day_ret": round(float(row["day_ret"]), 3),
            "capital": round(float(row["capital"]), 2),
        }
        for _, row in daily_stats.iterrows()
    ]

    _trade_rows = [
        {
            "date": r["date"], "symbol": r["symbol"], "signal": r["signal"],
            "premium": r["premium"], "entry": r["entry"], "exit_price": r["exit"],
            "exit_why": r["exit_why"],
            "gross_pct": round(r["pnl_pct"], 3), "net_pct": round(r["net_pct"], 3),
            "win": bool(r["win"]),
        }
        for _, r in df.iterrows()
    ]

    _out = {
        "summary": {
            "n_trades": total_trades,
            "n_wins": int(total_wins),
            "win_rate_pct": round(total_wins / total_trades * 100, 1),
            "avg_trade_pct": round(avg_trade, 3),
            "starting_capital": CAPITAL_0,
            "final_capital": round(final_cap, 2),
            "total_return_pct": round(total_return, 2),
            "avg_week_pct": round(avg_weekly, 2),
            "period": f"{df['date'].min()} → {df['date'].max()}",
            "n_weeks": n_weeks,
        },
        "daily": _daily_rows,
        "weekly": _weekly_rows,
        "trades": _trade_rows,
    }
    print(_json.dumps(_out))
    sys.exit(0)

# ── Chart ─────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    dates   = daily_stats["date"].tolist()
    day_ret = daily_stats["day_ret"].tolist()
    cap     = daily_stats["capital"].tolist()
    colors  = ["#26a69a" if r >= 0 else "#ef5350" for r in day_ret]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [1, 1.6]})
    fig.patch.set_facecolor("#1e1e2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="#cdd6f4", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

    # ── Top: daily return bars ────────────────────────────────────────────────
    x = range(len(dates))
    ax1.bar(x, day_ret, color=colors, width=0.6, zorder=3)
    ax1.axhline(0, color="#45475a", linewidth=0.8)
    ax1.set_ylabel("Daily Return %", color="#cdd6f4", fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right")
    ax1.set_title(
        f"Intraday Backtest  v{blob.get('version','?')}  |  "
        f"{len(SYMBOLS)} Nifty-50 symbols  |  "
        f"{'OOT (last 10%)' if not args.all_data else 'Full data'}",
        color="#cdd6f4", fontsize=11, pad=8,
    )
    ax1.grid(axis="y", color="#313244", linewidth=0.5, zorder=0)

    # annotate bar values
    for xi, ret in zip(x, day_ret):
        ax1.text(xi, ret + (0.02 if ret >= 0 else -0.05),
                 f"{ret:+.2f}%", ha="center", va="bottom" if ret >= 0 else "top",
                 fontsize=7.5, color="#cdd6f4")

    # ── Bottom: cumulative capital ────────────────────────────────────────────
    cap_full = [CAPITAL_0] + cap
    x_full   = range(len(cap_full))
    ax2.plot(x_full, cap_full, color="#89b4fa", linewidth=2, zorder=3)
    ax2.fill_between(x_full, CAPITAL_0, cap_full,
                     where=[c >= CAPITAL_0 for c in cap_full],
                     color="#26a69a", alpha=0.15)
    ax2.fill_between(x_full, CAPITAL_0, cap_full,
                     where=[c < CAPITAL_0 for c in cap_full],
                     color="#ef5350", alpha=0.15)
    ax2.axhline(CAPITAL_0, color="#f38ba8", linewidth=0.8, linestyle="--", label=f"Start ₹{CAPITAL_0:.0f}")
    ax2.set_ylabel("Capital (₹)", color="#cdd6f4", fontsize=9)
    ax2.set_xticks(list(x_full))
    ax2.set_xticklabels(["Start"] + [d[5:] for d in dates], rotation=45, ha="right")
    ax2.grid(color="#313244", linewidth=0.5, zorder=0)

    # mark final capital
    ax2.annotate(
        f"₹{final_cap:,.2f}  ({total_return:+.1f}%)",
        xy=(len(cap_full) - 1, final_cap),
        xytext=(-60, 12), textcoords="offset points",
        color="#a6e3a1", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#a6e3a1", lw=0.8),
    )

    # stats box
    stats_text = (
        f"Trades: {total_trades}  |  Win: {total_wins/total_trades:.0%}\n"
        f"Avg/trade: {avg_trade:+.2f}%  |  Brokerage: {BROKERAGE_RT:.2%} RT\n"
        f"Total: {total_return:+.2f}%  |  Avg/week: {avg_weekly:+.2f}%"
    )
    ax2.text(0.02, 0.96, stats_text, transform=ax2.transAxes,
             fontsize=8, color="#cdd6f4", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#313244", alpha=0.8))

    plt.tight_layout(pad=1.5)
    out_path = Path("backtest_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {out_path.resolve()}")
except Exception as exc:
    print(f"\n  [chart] skipped: {exc}")

if not args.all_data:
    print(f"\n  ⚠  Evaluated on TRUE OOT (last 10%) only — ~{n_weeks} week(s) of data.")
    print(f"     Run with --all-data for more weeks (but first 90% is in-sample).")
