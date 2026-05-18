"""
verify_intraday.py — Multi-day walkforward signal accuracy test for v3.1 intraday model.

Tests Tier A/B stocks across recent trading days:
  - Replays every 5-min bar, asks model "what signal would it emit here?"
  - Checks if signal was correct using make_labels ground truth (+0.7% in 40 min)
  - Reports per-day and aggregate precision/recall/signal-count

Usage:
  python verify_intraday.py           # all Tier A/B stocks, last 15 trading days
  python verify_intraday.py --days 20 # last 20 days
  python verify_intraday.py --sell    # include SELL signal verification too
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from intraday_model_v2 import (
    MODEL_PATH_V2, FEATURE_COLS, BUY_THRESH, SELL_THRESH, TARGET_BARS,
    HOLD_THRESHOLD, BUY_CLASS_THRESH,
    fetch_5min, fetch_nifty_5min, compute_features, make_labels,
    _dir_proba_ens,
)

# ── Config ──────────────────────────────────────────────────────────────────
N_DAYS       = 15   # trailing trading days to verify
MIN_SIGNALS  = 3    # skip days with < this many signal bars (too thin)
SELL_MODE    = "--sell" in sys.argv
N_DAYS_ARG   = next((int(sys.argv[i+1]) for i, a in enumerate(sys.argv)
                      if a == "--days" and i+1 < len(sys.argv)), None)
if N_DAYS_ARG:
    N_DAYS = N_DAYS_ARG

# ── Load model ───────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH_V2}...")
blob          = joblib.load(MODEL_PATH_V2)
models_dir    = blob.get("models_dir", [blob.get("model_dir", blob["model"])])
model_hold    = blob.get("model_hold")
sym_tiers     = blob.get("sym_tiers", {})
thresh_over   = blob.get("signal_threshold_override", {"A": 0.67, "B": 0.70, "C": 0.73})
dir_threshold = blob.get("dir_threshold", 0.67)
feat_cols     = blob.get("features") or FEATURE_COLS

print(f"  version={blob.get('version','?')}  trained={blob.get('trained_at','?')}")
print(f"  features={len(feat_cols)}  ensemble={len(models_dir)} models")

# ── Determine which stocks to test ───────────────────────────────────────────
BUY_TIER_AB  = [s for s, info in sym_tiers.items()
                if info.get("buy_tier", "C") in ("A", "B")]
SELL_TIER_AB = [s for s, info in sym_tiers.items()
                if info.get("sell_tier", "C") in ("A", "B")]
TEST_STOCKS  = sorted(set(BUY_TIER_AB) | (set(SELL_TIER_AB) if SELL_MODE else set()))

print(f"\nTier A/B BUY:  {sorted(BUY_TIER_AB)}")
if SELL_MODE:
    print(f"Tier A/B SELL: {sorted(SELL_TIER_AB)}")
print(f"Testing {len(TEST_STOCKS)} stocks: {TEST_STOCKS}\n")

# ── Helper: effective threshold per stock/direction ──────────────────────────
def eff_threshold(sym: str, direction: str) -> float:
    """Return the confidence threshold to emit a signal for this stock."""
    tier_info = sym_tiers.get(sym, {})
    t = tier_info.get(f"{direction.lower()}_tier", tier_info.get("tier", "C"))
    return thresh_over.get(t, 0.73)

# ── Fetch data once per stock ─────────────────────────────────────────────────
print("Fetching 5-min data (60 days max via yfinance)...")
stock_dfs: dict[str, pd.DataFrame] = {}
for sym in TEST_STOCKS:
    df = fetch_5min(sym, days=60)
    if df is not None and not df.empty:
        stock_dfs[sym] = df
        print(f"  {sym}: {len(df)} bars  "
              f"[{df['DateTime'].min().date()} → {df['DateTime'].max().date()}]")
    else:
        print(f"  {sym}: NO DATA — skipping")

print("\nFetching Nifty 5-min data...")
nifty_df = fetch_nifty_5min(days=60)
if nifty_df is not None:
    print(f"  Nifty: {len(nifty_df)} bars  "
          f"[{nifty_df['DateTime'].min().date()} → {nifty_df['DateTime'].max().date()}]")

# ── Determine trading days to test ───────────────────────────────────────────
# Use union of dates available in stock data
all_dates: set[date] = set()
for df in stock_dfs.values():
    all_dates |= set(df["DateTime"].dt.date.unique())

sorted_dates = sorted(all_dates)
# Exclude the very last date (partial day / today if markets open)
if sorted_dates:
    sorted_dates = sorted_dates[:-1]
test_dates = sorted_dates[-N_DAYS:]
print(f"\nTesting {len(test_dates)} days: {test_dates[0]} → {test_dates[-1]}\n")

# ── Per-day replay ────────────────────────────────────────────────────────────
results: list[dict] = []
day_summaries: list[dict] = []

for sym in sorted(stock_dfs.keys()):
    df_full  = stock_dfs[sym]
    tier_info = sym_tiers.get(sym, {})
    buy_tier  = tier_info.get("buy_tier", "C")
    sell_tier = tier_info.get("sell_tier", "C")

    print(f"{'='*60}")
    print(f"{sym}  [BUY-Tier:{buy_tier}  SELL-Tier:{sell_tier}]"
          f"  buy_precision={tier_info.get('buy_precision',0):.1%}"
          f"  sell_precision={tier_info.get('sell_precision',0):.1%}")

    # Compute features for all available data
    feats_all = compute_features(df_full)
    if feats_all is None or feats_all.empty:
        print(f"  [SKIP] feature computation failed")
        continue

    labels_all = make_labels(df_full)

    # Align features + labels on index
    common_idx = feats_all.index.intersection(labels_all.index)
    feats_all  = feats_all.loc[common_idx]
    labels_all = labels_all.loc[common_idx]
    datetimes  = df_full.loc[common_idx, "DateTime"]

    sym_buy_n = sym_buy_win = 0
    sym_sell_n = sym_sell_win = 0

    for tday in test_dates:
        day_mask = datetimes.dt.date == tday
        if day_mask.sum() < TARGET_BARS + 5:
            continue

        X_day   = feats_all.loc[day_mask, feat_cols]
        lbl_day = labels_all.loc[day_mask]
        dt_day  = datetimes.loc[day_mask]

        if X_day.isna().all().all():
            continue

        # Drop trailing bars that can't have valid labels (last TARGET_BARS bars)
        n_valid = max(0, len(X_day) - TARGET_BARS)
        if n_valid < MIN_SIGNALS:
            continue
        X_day   = X_day.iloc[:n_valid]
        lbl_day = lbl_day.iloc[:n_valid]
        dt_day  = dt_day.iloc[:n_valid]

        # Hold filter
        if model_hold is not None:
            hold_p  = model_hold.predict_proba(X_day)[:, 1]
            active  = hold_p >= HOLD_THRESHOLD
        else:
            active = np.ones(len(X_day), dtype=bool)

        # Direction probabilities for active bars
        if active.sum() == 0:
            continue
        dir_p = np.full(len(X_day), 0.5)
        dir_p[active] = _dir_proba_ens(models_dir, X_day[active])

        # Determine signals
        buy_thr_eff  = eff_threshold(sym, "BUY")
        sell_thr_eff = eff_threshold(sym, "SELL")

        buy_mask  = active & (dir_p >= buy_thr_eff)
        sell_mask = active & ((1.0 - dir_p) >= sell_thr_eff)

        # Only count BUY for BUY-Tier stocks, SELL for SELL-Tier stocks
        if buy_tier not in ("A", "B"):
            buy_mask[:] = False
        if sell_tier not in ("A", "B") or not SELL_MODE:
            sell_mask[:] = False

        n_buy  = buy_mask.sum()
        n_sell = sell_mask.sum()
        if n_buy + n_sell == 0:
            continue

        lbl_arr = lbl_day.values
        buy_wins  = int((buy_mask & (lbl_arr == 1)).sum())
        sell_wins = int((sell_mask & (lbl_arr == -1)).sum())

        sym_buy_n  += n_buy;  sym_buy_win  += buy_wins
        sym_sell_n += n_sell; sym_sell_win += sell_wins

        # First signal details
        first_buy_idx  = int(np.argmax(buy_mask))  if n_buy > 0  else None
        first_sell_idx = int(np.argmax(sell_mask)) if n_sell > 0 else None

        row_detail = []
        if n_buy > 0:
            fbi   = first_buy_idx
            row_detail.append(
                f"    BUY@{dt_day.iloc[fbi].strftime('%H:%M')} "
                f"conf={dir_p[fbi]:.3f}  lbl={lbl_arr[fbi]:+d}  "
                f"{'WIN ✓' if lbl_arr[fbi]==1 else 'MISS ✗'}"
            )
        if n_sell > 0 and SELL_MODE:
            fsi = first_sell_idx
            row_detail.append(
                f"    SELL@{dt_day.iloc[fsi].strftime('%H:%M')} "
                f"conf={1-dir_p[fsi]:.3f}  lbl={lbl_arr[fsi]:+d}  "
                f"{'WIN ✓' if lbl_arr[fsi]==-1 else 'MISS ✗'}"
            )

        print(f"  {tday}  BUY:{n_buy}  SELL:{n_sell}  "
              f"buy_acc={buy_wins/n_buy:.0%}" if n_buy > 0 else
              f"  {tday}  BUY:0  SELL:{n_sell}", end="")
        if n_sell > 0 and SELL_MODE:
            print(f"  sell_acc={sell_wins/n_sell:.0%}", end="")
        print()
        for rd in row_detail:
            print(rd)

        results.append({
            "sym": sym, "date": tday,
            "n_buy": n_buy, "buy_win": buy_wins,
            "n_sell": n_sell, "sell_win": sell_wins,
            "buy_prec": buy_wins/n_buy if n_buy else np.nan,
            "sell_prec": sell_wins/n_sell if n_sell else np.nan,
        })

    # Stock summary
    buy_prec  = sym_buy_win/sym_buy_n  if sym_buy_n  > 0 else float("nan")
    sell_prec = sym_sell_win/sym_sell_n if sym_sell_n > 0 else float("nan")
    print(f"  → {sym} BUY: {sym_buy_win}/{sym_buy_n} = {buy_prec:.1%}"
          f"  SELL: {sym_sell_win}/{sym_sell_n} = {sell_prec:.1%}\n")

# ── Aggregate summary ─────────────────────────────────────────────────────────
if not results:
    print("\nNo signals fired — data too sparse or model filtered everything.")
    sys.exit(0)

df_res = pd.DataFrame(results)
tot_buy_n   = int(df_res["n_buy"].sum())
tot_buy_win = int(df_res["buy_win"].sum())
tot_sell_n  = int(df_res["n_sell"].sum())
tot_sell_win = int(df_res["sell_win"].sum())

print("\n" + "="*60)
print("AGGREGATE RESULTS")
print("="*60)
print(f"  Period          : {test_dates[0]} → {test_dates[-1]}")
print(f"  Stocks tested   : {len(stock_dfs)}")
print(f"  Trading days    : {len(test_dates)}")
print()
if tot_buy_n > 0:
    print(f"  BUY  signals    : {tot_buy_win}/{tot_buy_n} = {tot_buy_win/tot_buy_n:.1%}  precision")
    # Expected random rate: base rate of BUY labels
    print(f"  (random baseline ≈ 10-15% for +0.7% in 40-min)")
if tot_sell_n > 0:
    print(f"  SELL signals    : {tot_sell_win}/{tot_sell_n} = {tot_sell_win/tot_sell_n:.1%}  precision")

# Per-stock breakdown table
print()
print(f"{'Stock':<12}  {'BUY_N':>6}  {'BUY_WIN':>7}  {'BUY%':>6}  {'SELL_N':>7}  {'SELL%':>6}")
print("-"*60)
for sym in sorted(df_res["sym"].unique()):
    sub = df_res[df_res["sym"] == sym]
    bn  = int(sub["n_buy"].sum())
    bw  = int(sub["buy_win"].sum())
    sn  = int(sub["n_sell"].sum())
    sw  = int(sub["sell_win"].sum())
    bp  = f"{bw/bn:.1%}" if bn > 0 else "-"
    sp  = f"{sw/sn:.1%}" if sn > 0 else "-"
    print(f"{sym:<12}  {bn:>6}  {bw:>7}  {bp:>6}  {sn:>7}  {sp:>6}")

print()
if tot_buy_n >= 20:
    lift = (tot_buy_win/tot_buy_n) / 0.12  # vs 12% random
    print(f"  Model BUY lift over random  : {lift:.1f}×")
    if lift >= 2.5:
        print(f"  Verdict: STRONG edge (≥2.5× lift)")
    elif lift >= 1.8:
        print(f"  Verdict: MODERATE edge (1.8-2.5× lift)")
    else:
        print(f"  Verdict: WEAK edge (<1.8× lift) — retrain or filter harder")
else:
    print(f"  (Too few signals for lift calculation; need ≥20 BUY signals)")

print(f"\nDone. {len(test_dates)} days × {len(TEST_STOCKS)} stocks evaluated.")
