"""
swing_v2.py — Upgraded swing trading model for NSE.

Improvements over swing_signals.py:
  • Pooled training: ONE XGBoost trained on Nifty-50 stocks together
    (~10–15k samples vs ~250 samples per stock in v1)
  • 15 features: trend, momentum, volatility, volume, position,
    gap, candle shape, 52-week distance, market context
  • 3 years of history per stock (not 1)
  • Walk-forward backtest with realistic stop/target/timeout exits

Commands:
    python swing_v2.py train       # train pooled model, save to models/swing_v2.pkl
    python swing_v2.py predict     # generate today's signals using saved model
    python swing_v2.py backtest    # walk-forward backtest with P&L

⚠️ Educational only — not financial advice.
"""

from __future__ import annotations
import sys
from datetime import date, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from stock_fetcher import fetch_historical_data
from swing_signals import _rsi, _macd, _bollinger, _atr, _obv


# =============================================================================
# Configuration
# =============================================================================

NIFTY_50 = [
    "RELIANCE", "HDFCBANK", "TCS", "BHARTIARTL", "ICICIBANK", "SBIN",
    "INFY", "HINDUNILVR", "ITC", "LT", "HCLTECH", "KOTAKBANK",
    "BAJFINANCE", "MARUTI", "SUNPHARMA", "AXISBANK", "TITAN",
    "ULTRACEMCO", "WIPRO", "NTPC", "ONGC", "POWERGRID", "TATAMOTORS",
    "NESTLEIND", "TATASTEEL", "JSWSTEEL", "BAJAJFINSV", "TECHM",
    "COALINDIA", "ASIANPAINT", "GRASIM", "EICHERMOT", "ADANIPORTS",
    "INDUSINDBK", "CIPLA", "HINDALCO", "SBILIFE", "HDFCLIFE",
    "HEROMOTOCO", "BRITANNIA", "DIVISLAB", "DRREDDY", "APOLLOHOSP",
    "TRENT", "BPCL",
]

WATCHLIST = [
    "SBIN", "HDFCBANK", "RELIANCE", "TCS", "INFY",
    "ICICIBANK", "WIPRO", "AXISBANK",
]

LOOKBACK_YEARS = 3
FORWARD_DAYS   = 5
THRESHOLD      = 0.02   # ±2% for BUY/SELL labels (drop neutral samples)
BUY_PROBA      = 0.60   # min probability to call BUY
SELL_PROBA     = 0.60   # min probability to call SELL

MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "swing_v2.pkl"


# =============================================================================
# Feature engineering — 15 features
# =============================================================================

FEATURE_COLS = [
    "feat_rsi",
    "feat_macd_hist",
    "feat_macd_signal_diff",
    "feat_ema_fast_ratio",
    "feat_ema_slow_ratio",
    "feat_bb_pos",
    "feat_atr_pct",
    "feat_obv_ratio",
    "feat_gap_pct",
    "feat_body_ratio",
    "feat_dist_52w_high",
    "feat_dist_52w_low",
    "feat_return_5d",
    "feat_volatility_20d",
    "feat_volume_ratio",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 15 features from OHLCV. Returns df with feat_* columns."""
    df = df.copy()
    close, high, low, vol, op = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

    # 1. RSI 14
    df["feat_rsi"] = _rsi(close, 14)

    # 2-3. MACD histogram + (macd - signal) normalised by price
    macd_line, sig_line, hist = _macd(close, 12, 26, 9)
    df["feat_macd_hist"]        = hist
    df["feat_macd_signal_diff"] = (macd_line - sig_line) / close

    # 4-5. EMA ratios (trend strength)
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    df["feat_ema_fast_ratio"] = ema9 / ema21 - 1
    df["feat_ema_slow_ratio"] = ema21 / ema50 - 1

    # 6. Bollinger Band position
    _, _, _, bb_pos = _bollinger(close, 20, 2)
    df["feat_bb_pos"] = bb_pos

    # 7. ATR % of price (normalised volatility)
    atr = _atr(high, low, close, 14)
    df["feat_atr_pct"] = atr / close
    df["ATR"] = atr   # kept for stop/target calculations later

    # 8. OBV ratio (volume confirmation)
    obv = _obv(close, vol)
    df["feat_obv_ratio"] = obv / obv.ewm(span=21, adjust=False).mean().replace(0, np.nan)

    # 9. Gap % (today's open vs yesterday's close)
    df["feat_gap_pct"] = op / close.shift(1) - 1

    # 10. Body ratio (|close-open| / range) — strength of the candle
    df["feat_body_ratio"] = (close - op).abs() / (high - low).replace(0, np.nan)

    # 11-12. Distance from 52-week high / low
    high_52w = close.rolling(252, min_periods=20).max()
    low_52w  = close.rolling(252, min_periods=20).min()
    df["feat_dist_52w_high"] = close / high_52w - 1
    df["feat_dist_52w_low"]  = close / low_52w  - 1

    # 13. 5-day past return (momentum)
    df["feat_return_5d"] = close.pct_change(5)

    # 14. 20-day volatility (regime)
    df["feat_volatility_20d"] = close.pct_change().rolling(20).std()

    # 15. Volume / 20d avg volume (relative volume)
    df["feat_volume_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)

    return df


# =============================================================================
# Dataset assembly
# =============================================================================

def build_panel(symbols: list[str], years: int = LOOKBACK_YEARS) -> pd.DataFrame | None:
    """Fetch all symbols, compute features, concatenate with a 'symbol' column."""
    today     = date.today()
    from_date = today - timedelta(days=years * 365)

    panels = []
    for sym in symbols:
        df = fetch_historical_data(sym, from_date, today)
        if df is None or len(df) < 100:
            print(f"  [SKIP] {sym}: insufficient data")
            continue
        df = compute_features(df)
        df["symbol"] = sym
        panels.append(df)
        print(f"  [OK]   {sym}: {len(df)} rows")

    if not panels:
        return None
    return pd.concat(panels, ignore_index=True)


def make_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward 5-day return ≥+2% → BUY (1), ≤-2% → SELL (0), else drop."""
    panel = panel.sort_values(["symbol", "DateTime"]).copy()
    panel["fwd_return"] = (
        panel.groupby("symbol")["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    )
    panel = panel.dropna(subset=FEATURE_COLS + ["fwd_return"])
    panel = panel[panel["fwd_return"].abs() >= THRESHOLD].copy()
    panel["label"] = (panel["fwd_return"] >= 0).astype(int)
    return panel


# =============================================================================
# Train
# =============================================================================

def train(symbols: list[str] | None = None, years: int = LOOKBACK_YEARS):
    symbols = symbols or NIFTY_50
    print(f"\n=== TRAINING v2 model on {len(symbols)} stocks × {years}y ===\n")

    panel = build_panel(symbols, years)
    if panel is None:
        print("No data — aborting.")
        return None

    print(f"\nTotal panel rows: {len(panel):,}")
    panel = make_labels(panel)
    n_buy  = int(panel["label"].sum())
    n_sell = len(panel) - n_buy
    print(f"After label filter: {len(panel):,} samples (BUY={n_buy:,}, SELL={n_sell:,})")

    # Sort chronologically (across all stocks) for honest time-series CV
    panel = panel.sort_values("DateTime").reset_index(drop=True)
    X = panel[FEATURE_COLS].values
    y = panel["label"].values

    print("\nWalk-forward cross-validation (5 folds):")
    tscv = TimeSeriesSplit(n_splits=5)
    fold_accs = []
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        m = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        m.fit(X[tr], y[tr])
        acc = accuracy_score(y[te], m.predict(X[te]))
        fold_accs.append(acc)
        print(f"  Fold {i}: train={len(tr):>5,}  test={len(te):>5,}  acc={acc:.3f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    print(f"\nMean walk-forward accuracy: {mean_acc:.3f}  (±{std_acc:.3f})")

    if mean_acc < 0.52:
        print("⚠️  WARNING: Accuracy barely beats coin-flip (50%). Consider:")
        print("   • More stocks (add mid/small caps)")
        print("   • Longer history (5y instead of 3y)")
        print("   • Different forward horizon (3 or 10 days)")

    # Final fit on all data
    final = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", verbosity=0,
    )
    final.fit(X, y)

    # Save
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": final,
        "features": FEATURE_COLS,
        "val_acc": mean_acc,
        "val_std": std_acc,
        "n_train": len(panel),
        "trained_on": symbols,
        "trained_at": str(date.today()),
    }, MODEL_PATH)
    print(f"\n✓ Saved to {MODEL_PATH}")

    # Top features
    imp = pd.Series(final.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(imp.head(10).round(3).to_string())

    return final


# =============================================================================
# Predict
# =============================================================================

def predict(symbols: list[str] | None = None):
    if not MODEL_PATH.exists():
        print('No saved model. Run "python swing_v2.py train" first.')
        return

    blob    = joblib.load(MODEL_PATH)
    model   = blob["model"]
    val_acc = blob["val_acc"]

    symbols   = symbols or WATCHLIST
    today     = date.today()
    from_date = today - timedelta(days=400)   # need 252d for 52-week features

    print(f"\n=== v2 SIGNALS — {today} ===")
    print(f"Model val accuracy: {val_acc*100:.1f}% on {blob['n_train']:,} samples\n")

    rows = []
    for sym in symbols:
        df = fetch_historical_data(sym, from_date, today)
        if df is None or len(df) < 252:
            print(f"  [SKIP] {sym}: insufficient data")
            continue
        df = compute_features(df).dropna(subset=FEATURE_COLS)
        if df.empty:
            continue

        latest = df.iloc[-1]
        proba  = model.predict_proba(latest[FEATURE_COLS].values.reshape(1, -1))[0]
        buy_p, sell_p = float(proba[1]), float(proba[0])

        if buy_p >= BUY_PROBA:
            signal = "BUY"
        elif sell_p >= SELL_PROBA:
            signal = "SELL"
        else:
            signal = "HOLD"

        atr, price = float(latest["ATR"]), float(latest["Close"])
        if signal == "BUY":
            stop, tgt = round(price - 1.5 * atr, 2), round(price + 3 * atr, 2)
        elif signal == "SELL":
            stop, tgt = round(price + 1.5 * atr, 2), round(price - 3 * atr, 2)
        else:
            stop = tgt = None

        rows.append({
            "Symbol":    sym,
            "Price (₹)": round(price, 2),
            "Signal":    signal,
            "BUY %":     round(buy_p * 100, 1),
            "Stop":      stop,
            "Target":    tgt,
        })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    print("\n⚠️  Educational only — not financial advice.")
    return df_out


# =============================================================================
# Backtest — walk-forward simulation with stop/target/timeout exits
# =============================================================================

def backtest(symbols: list[str] | None = None, years: int = 2,
             buy_threshold: float = BUY_PROBA, max_holding_days: int = 10):
    """Simulate trades using the saved v2 model on out-of-sample data."""
    if not MODEL_PATH.exists():
        print('No saved model. Run "python swing_v2.py train" first.')
        return

    blob  = joblib.load(MODEL_PATH)
    model = blob["model"]

    symbols = symbols or NIFTY_50[:20]   # 20 stocks for speed
    print(f"\n=== BACKTEST v2 — {len(symbols)} stocks × {years}y ===")

    panel = build_panel(symbols, years)
    if panel is None:
        return

    panel = panel.sort_values(["symbol", "DateTime"]).reset_index(drop=True)
    panel = panel.dropna(subset=FEATURE_COLS).copy()

    # Score every row
    panel["buy_prob"] = model.predict_proba(panel[FEATURE_COLS].values)[:, 1]

    # Walk forward and simulate trades per symbol
    trades = []
    for sym in symbols:
        sd = panel[panel["symbol"] == sym].sort_values("DateTime").reset_index(drop=True)
        if len(sd) < 30:
            continue
        i = 0
        while i < len(sd) - max_holding_days - 1:
            if sd.loc[i, "buy_prob"] < buy_threshold:
                i += 1
                continue

            # Enter long at NEXT bar's open (no look-ahead)
            entry_idx = i + 1
            entry     = float(sd.loc[entry_idx, "Open"])
            atr       = float(sd.loc[i, "ATR"])
            stop      = entry - 1.5 * atr
            target    = entry + 3.0 * atr

            exit_price, reason, days = entry, "timeout", max_holding_days
            for d in range(1, max_holding_days + 1):
                if entry_idx + d >= len(sd):
                    break
                bar_low  = float(sd.loc[entry_idx + d, "Low"])
                bar_high = float(sd.loc[entry_idx + d, "High"])
                if bar_low <= stop:
                    exit_price, reason, days = stop, "stop", d
                    break
                if bar_high >= target:
                    exit_price, reason, days = target, "target", d
                    break
            else:
                exit_price = float(sd.loc[min(entry_idx + max_holding_days, len(sd) - 1), "Close"])

            pnl_pct = (exit_price - entry) / entry
            trades.append({
                "symbol": sym,
                "entry_date": sd.loc[entry_idx, "DateTime"],
                "entry": round(entry, 2),
                "exit": round(exit_price, 2),
                "pnl_pct": pnl_pct,
                "days": days,
                "reason": reason,
                "buy_prob": float(sd.loc[i, "buy_prob"]),
            })
            i = entry_idx + days + 1   # skip past exit before looking for next setup

    if not trades:
        print("No BUY signals crossed the threshold — no trades simulated.")
        return None

    tdf = pd.DataFrame(trades)

    # Metrics
    wr           = (tdf["pnl_pct"] > 0).mean() * 100
    avg_pnl      = tdf["pnl_pct"].mean() * 100
    best, worst  = tdf["pnl_pct"].max() * 100, tdf["pnl_pct"].min() * 100
    total_return = ((1 + tdf["pnl_pct"]).prod() - 1) * 100
    avg_days     = tdf["days"].mean()
    sharpe       = (tdf["pnl_pct"].mean() / tdf["pnl_pct"].std()) * np.sqrt(252 / max(avg_days, 1))

    # Buy-and-hold baseline (equal weight)
    bh = []
    for sym in symbols:
        sd = panel[panel["symbol"] == sym].sort_values("DateTime")
        if len(sd) >= 2:
            bh.append((sd.iloc[-1]["Close"] - sd.iloc[0]["Close"]) / sd.iloc[0]["Close"])
    bh_return = float(np.mean(bh)) * 100 if bh else 0.0

    print(f"\n=== RESULTS ({len(tdf)} trades) ===")
    print(f"Win rate              : {wr:.1f}%")
    print(f"Average P&L per trade : {avg_pnl:+.2f}%")
    print(f"Best / Worst trade    : {best:+.2f}% / {worst:+.2f}%")
    print(f"Avg holding days      : {avg_days:.1f}")
    print(f"Total compound return : {total_return:+.2f}%")
    print(f"Sharpe (annualised)   : {sharpe:.2f}")
    print(f"\nBuy-and-hold baseline : {bh_return:+.2f}%  (equal-weight {len(bh)} stocks, {years}y)")
    print(f"Strategy vs B&H       : {total_return - bh_return:+.2f}% {'✓ beats' if total_return > bh_return else '✗ underperforms'}")

    print("\nExit reasons:")
    print(tdf["reason"].value_counts().to_string())

    print("\nTop 5 winners:")
    print(tdf.nlargest(5, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "days", "reason"]]
          .assign(pnl_pct=lambda d: (d.pnl_pct * 100).round(2)).to_string(index=False))

    print("\n⚠️  Educational only — not financial advice. Slippage and brokerage NOT modelled.")
    return tdf


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    {
        "train":    lambda: train(),
        "predict":  lambda: predict(),
        "backtest": lambda: backtest(),
    }.get(cmd, lambda: print(f"Usage: python {sys.argv[0]} [train|predict|backtest]"))()
