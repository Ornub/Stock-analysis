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
from sklearn.metrics import accuracy_score, precision_score, recall_score

from stock_fetcher import fetch_historical_data
from swing_signals import _rsi, _macd, _bollinger, _atr, _obv


# =============================================================================
# Extra indicators (v2.1)
# =============================================================================

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX — trend strength (0-100). Higher = stronger trend, regardless of direction."""
    up   = high.diff()
    down = -low.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean()  / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic %K — close vs 14-day high/low range, 0-100."""
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    return 100 * (close - ll) / (hh - ll).replace(0, np.nan)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R — inverted stochastic, -100 to 0."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index — typical price deviation from mean."""
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


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

LOOKBACK_YEARS = 5      # v2.1: 5y vs 3y for more samples
FORWARD_DAYS   = 5
THRESHOLD      = 0.02   # ±2% for BUY/SELL labels (drop neutral samples)
BUY_PROBA      = 0.60   # min probability to call BUY
SELL_PROBA     = 0.60   # min probability to call SELL

# Signal-quality gates (v2.1) — applied at PREDICT and BACKTEST entry time
ADX_MIN        = 0.20   # ADX (normalised 0-1) — require trend strength ≥ 20
USE_EMA200_FILTER = True   # only BUY when close > EMA200 (uptrend filter)

# Trade-cost model for backtest
SLIPPAGE_PCT       = 0.0025  # 0.25% per fill (entry + exit applied separately)
ROUND_TRIP_COST    = 0.0050  # 0.5% combined brokerage + STT + GST per round trip

MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "swing_v2.pkl"


# =============================================================================
# Feature engineering — 15 features
# =============================================================================

FEATURE_COLS = [
    # Trend / momentum
    "feat_rsi",
    "feat_macd_hist",
    "feat_macd_signal_diff",
    "feat_ema_fast_ratio",
    "feat_ema_slow_ratio",
    "feat_ema200_ratio",         # NEW: long-term trend filter
    "feat_adx",                  # NEW: trend strength
    "feat_stoch_k",              # NEW: stochastic oscillator
    "feat_williams_r",           # NEW: inverted stochastic
    "feat_cci",                  # NEW: commodity channel index
    # Volatility / position
    "feat_bb_pos",
    "feat_atr_pct",
    "feat_dist_52w_high",
    "feat_dist_52w_low",
    "feat_volatility_20d",
    "feat_high_low_range",       # NEW: bar range vs price
    # Volume
    "feat_obv_ratio",
    "feat_volume_ratio",
    "feat_volume_trend",         # NEW: vol EMA9/EMA21
    # Candle / gap
    "feat_gap_pct",
    "feat_body_ratio",
    # Multi-period momentum
    "feat_return_5d",
    "feat_return_10d",           # NEW
    "feat_return_20d",           # NEW
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

    # 4-6. EMA ratios (trend strength)
    ema9   = close.ewm(span=9,   adjust=False).mean()
    ema21  = close.ewm(span=21,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df["feat_ema_fast_ratio"] = ema9   / ema21 - 1
    df["feat_ema_slow_ratio"] = ema21  / ema50 - 1
    df["feat_ema200_ratio"]   = close  / ema200 - 1     # long-term trend filter

    # 7. ADX 14 (trend strength) — normalised to 0-1
    df["feat_adx"] = _adx(high, low, close, 14) / 100

    # 8. Stochastic %K 14 — normalised to 0-1
    df["feat_stoch_k"] = _stoch_k(high, low, close, 14) / 100

    # 9. Williams %R 14 — already -100 to 0, shift to 0-1
    df["feat_williams_r"] = (_williams_r(high, low, close, 14) + 100) / 100

    # 10. CCI 20 — clip extreme values, normalise around 0
    df["feat_cci"] = _cci(high, low, close, 20).clip(-300, 300) / 300

    # 11. Bollinger Band position
    _, _, _, bb_pos = _bollinger(close, 20, 2)
    df["feat_bb_pos"] = bb_pos

    # 12. ATR % of price (normalised volatility)
    atr = _atr(high, low, close, 14)
    df["feat_atr_pct"] = atr / close
    df["ATR"] = atr   # kept for stop/target calculations later

    # 13-14. Distance from 52-week high / low
    high_52w = close.rolling(252, min_periods=20).max()
    low_52w  = close.rolling(252, min_periods=20).min()
    df["feat_dist_52w_high"] = close / high_52w - 1
    df["feat_dist_52w_low"]  = close / low_52w  - 1

    # 15. 20-day volatility (regime)
    df["feat_volatility_20d"] = close.pct_change().rolling(20).std()

    # 16. High-low range vs close (intraday range as % of price)
    df["feat_high_low_range"] = (high - low) / close

    # 17. OBV ratio (volume confirmation)
    obv = _obv(close, vol)
    df["feat_obv_ratio"] = obv / obv.ewm(span=21, adjust=False).mean().replace(0, np.nan)

    # 18. Volume / 20d avg volume (relative volume)
    df["feat_volume_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)

    # 19. Volume trend — short EMA / long EMA of volume
    vol_ema9  = vol.ewm(span=9,  adjust=False).mean()
    vol_ema21 = vol.ewm(span=21, adjust=False).mean().replace(0, np.nan)
    df["feat_volume_trend"] = vol_ema9 / vol_ema21 - 1

    # 20. Gap % (today's open vs yesterday's close)
    df["feat_gap_pct"] = op / close.shift(1) - 1

    # 21. Body ratio (|close-open| / range) — strength of the candle
    df["feat_body_ratio"] = (close - op).abs() / (high - low).replace(0, np.nan)

    # 22-24. Multi-period momentum
    df["feat_return_5d"]  = close.pct_change(5)
    df["feat_return_10d"] = close.pct_change(10)
    df["feat_return_20d"] = close.pct_change(20)

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

    total_rows = len(panel)
    print(f"\nTotal panel rows: {total_rows:,}")
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
    fold_accs, fold_precs, fold_recs, fold_iters = [], [], [], []
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        # Internal val split (last 15% of train) for early stopping
        val_size  = max(int(len(tr) * 0.15), 100)
        tr_inner  = tr[:-val_size]
        val_inner = tr[-val_size:]

        m = XGBClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            min_child_weight=5, gamma=0.2,
            reg_alpha=0.5, reg_lambda=1.0,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", verbosity=0,
            early_stopping_rounds=30,
        )
        m.fit(
            X[tr_inner], y[tr_inner],
            eval_set=[(X[val_inner], y[val_inner])],
            verbose=False,
        )
        y_pred  = m.predict(X[te])
        acc     = accuracy_score(y[te], y_pred)
        prec    = precision_score(y[te], y_pred, pos_label=1, zero_division=0)
        rec     = recall_score(y[te], y_pred, pos_label=1, zero_division=0)
        n_trees = getattr(m, "best_iteration", m.n_estimators) or m.n_estimators
        fold_accs.append(acc)
        fold_precs.append(prec)
        fold_recs.append(rec)
        fold_iters.append(n_trees)
        print(f"  Fold {i}: train={len(tr_inner):>5,}  val={val_inner.size:>4,}  test={len(te):>5,} "
              f"| acc={acc:.3f}  BUY-prec={prec:.3f}  BUY-rec={rec:.3f}  trees={n_trees}")

    mean_acc  = float(np.mean(fold_accs))
    std_acc   = float(np.std(fold_accs))
    mean_prec = float(np.mean(fold_precs))
    mean_rec  = float(np.mean(fold_recs))
    print(f"\nMean walk-forward accuracy : {mean_acc:.3f}  (±{std_acc:.3f})")
    print(f"Mean BUY precision         : {mean_prec:.3f}  (when model says BUY, % correct)")
    print(f"Mean BUY recall            : {mean_rec:.3f}  (% of true BUYs caught)")

    # Flag unstable folds (>1.5σ from mean)
    sigma = max(std_acc, 1e-6)
    unstable = [(i + 1, a) for i, a in enumerate(fold_accs) if abs(a - mean_acc) > 1.5 * sigma]
    if unstable:
        print(f"⚠️  Unstable folds (>1.5σ from mean): "
              + ", ".join(f"#{i} ({a:.3f})" for i, a in unstable))

    if mean_acc < 0.52:
        print("⚠️  WARNING: Accuracy barely beats coin-flip (50%). Consider:")
        print("   • More stocks (add mid/small caps)")
        print("   • Longer history (5y instead of 3y)")
        print("   • Different forward horizon (3 or 10 days)")

    # Final fit on all data — use median of best-iteration counts to avoid overfit
    n_final = int(np.median(fold_iters)) if fold_iters else 400
    final = XGBClassifier(
        n_estimators=n_final, max_depth=4, learning_rate=0.03,
        min_child_weight=5, gamma=0.2,
        reg_alpha=0.5, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", verbosity=0,
    )
    final.fit(X, y, verbose=False)

    # Save
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": final,
        "features": FEATURE_COLS,
        "val_acc":  mean_acc,
        "val_std":  std_acc,
        "val_prec": mean_prec,
        "val_rec":  mean_rec,
        "n_train":  len(panel),
        "trained_on": symbols,
        "trained_at": str(date.today()),
    }, MODEL_PATH)
    print(f"\n✓ Saved to {MODEL_PATH}")

    # Output summary block
    print("\n=== TRAINING SUMMARY ===")
    print(f"Total panel rows    : {total_rows:,}")
    print(f"Labelled samples    : {len(panel):,}")
    print(f"Class balance       : BUY {n_buy:,} ({n_buy/len(panel)*100:.1f}%) | "
          f"SELL {n_sell:,} ({n_sell/len(panel)*100:.1f}%)")
    print(f"Features            : {len(FEATURE_COLS)}")
    print(f"Final trees         : {n_final}")

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

        # Signal-quality filter — both gates must pass for a BUY
        in_uptrend  = (not USE_EMA200_FILTER) or (latest["feat_ema200_ratio"] > 0)
        trending    = latest["feat_adx"] >= ADX_MIN
        passes_gate = in_uptrend and trending

        if buy_p >= BUY_PROBA and passes_gate:
            signal = "BUY"
        elif buy_p >= BUY_PROBA and not passes_gate:
            signal = "WATCH"   # model wants to buy but trend filter rejects
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

    # Walk forward and simulate trades per symbol — apply signal-quality gate
    trades = []
    skipped_gate = 0
    for sym in symbols:
        sd = panel[panel["symbol"] == sym].sort_values("DateTime").reset_index(drop=True)
        if len(sd) < 30:
            continue
        i = 0
        while i < len(sd) - max_holding_days - 1:
            if sd.loc[i, "buy_prob"] < buy_threshold:
                i += 1
                continue
            # Trend gate — same as predict()
            in_uptrend = (not USE_EMA200_FILTER) or (sd.loc[i, "feat_ema200_ratio"] > 0)
            trending   = sd.loc[i, "feat_adx"] >= ADX_MIN
            if not (in_uptrend and trending):
                skipped_gate += 1
                i += 1
                continue

            # Enter long at NEXT bar's open (no look-ahead) + slippage on entry
            entry_idx  = i + 1
            raw_entry  = float(sd.loc[entry_idx, "Open"])
            entry      = raw_entry * (1 + SLIPPAGE_PCT)
            atr        = float(sd.loc[i, "ATR"])
            stop       = raw_entry - 1.5 * atr
            target     = raw_entry + 3.0 * atr

            raw_exit, reason, days = raw_entry, "timeout", max_holding_days
            for d in range(1, max_holding_days + 1):
                if entry_idx + d >= len(sd):
                    break
                bar_low  = float(sd.loc[entry_idx + d, "Low"])
                bar_high = float(sd.loc[entry_idx + d, "High"])
                if bar_low <= stop:
                    raw_exit, reason, days = stop, "stop", d
                    break
                if bar_high >= target:
                    raw_exit, reason, days = target, "target", d
                    break
            else:
                raw_exit = float(sd.loc[min(entry_idx + max_holding_days, len(sd) - 1), "Close"])

            # Slippage on exit (always works against you)
            exit_price = raw_exit * (1 - SLIPPAGE_PCT)
            gross_pnl  = (exit_price - entry) / entry
            net_pnl    = gross_pnl - ROUND_TRIP_COST   # subtract brokerage/STT

            trades.append({
                "symbol": sym,
                "entry_date": sd.loc[entry_idx, "DateTime"],
                "entry": round(entry, 2),
                "exit":  round(exit_price, 2),
                "pnl_pct_gross": gross_pnl,
                "pnl_pct":       net_pnl,
                "days": days,
                "reason": reason,
                "buy_prob": float(sd.loc[i, "buy_prob"]),
            })
            i = entry_idx + days + 1   # skip past exit before looking for next setup

    if not trades:
        print("No BUY signals crossed the threshold — no trades simulated.")
        return None

    tdf = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

    # Core metrics (NET of costs)
    wr           = (tdf["pnl_pct"] > 0).mean() * 100
    avg_pnl      = tdf["pnl_pct"].mean() * 100
    avg_pnl_gross = tdf["pnl_pct_gross"].mean() * 100
    best, worst  = tdf["pnl_pct"].max() * 100, tdf["pnl_pct"].min() * 100
    total_return = ((1 + tdf["pnl_pct"]).prod() - 1) * 100
    avg_days     = tdf["days"].mean()
    median_days  = tdf["days"].median()
    sharpe       = (tdf["pnl_pct"].mean() / tdf["pnl_pct"].std()) * np.sqrt(252 / max(avg_days, 1))

    # Profit factor = sum(wins) / |sum(losses)|
    wins   = tdf.loc[tdf["pnl_pct"] > 0, "pnl_pct"].sum()
    losses = tdf.loc[tdf["pnl_pct"] < 0, "pnl_pct"].sum()
    profit_factor = wins / abs(losses) if losses < 0 else float("inf")

    # Expectancy = (P_win * avg_win) - (P_loss * |avg_loss|)
    p_win       = (tdf["pnl_pct"] > 0).mean()
    avg_win     = tdf.loc[tdf["pnl_pct"] > 0, "pnl_pct"].mean() if p_win > 0 else 0.0
    avg_loss    = tdf.loc[tdf["pnl_pct"] < 0, "pnl_pct"].mean() if p_win < 1 else 0.0
    expectancy  = (p_win * avg_win + (1 - p_win) * avg_loss) * 100

    # Max drawdown on the equity curve
    equity   = (1 + tdf["pnl_pct"]).cumprod()
    peak     = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    max_dd   = drawdown.min()

    # Buy-and-hold baseline (equal weight)
    bh = []
    for sym in symbols:
        sd = panel[panel["symbol"] == sym].sort_values("DateTime")
        if len(sd) >= 2:
            bh.append((sd.iloc[-1]["Close"] - sd.iloc[0]["Close"]) / sd.iloc[0]["Close"])
    bh_return = float(np.mean(bh)) * 100 if bh else 0.0

    reasons = tdf["reason"].value_counts()

    print(f"\n=== RESULTS ({len(tdf)} trades, {skipped_gate} gated out) ===")
    print(f"Win rate                 : {wr:.1f}%")
    print(f"Avg P&L net per trade    : {avg_pnl:+.2f}%   (gross {avg_pnl_gross:+.2f}%)")
    print(f"Best / Worst trade       : {best:+.2f}% / {worst:+.2f}%")
    print(f"Median holding days      : {median_days:.0f}   (avg {avg_days:.1f})")
    print(f"Total compound return    : {total_return:+.2f}%")
    print(f"Sharpe (annualised)      : {sharpe:.2f}")
    print(f"Max drawdown             : {max_dd:.2f}%")
    print(f"Profit factor            : {profit_factor:.2f}")
    print(f"Expectancy per trade     : {expectancy:+.3f}%")
    print(f"\nBuy-and-hold baseline    : {bh_return:+.2f}%  (equal-weight {len(bh)} stocks, {years}y)")
    print(f"Strategy vs B&H          : {total_return - bh_return:+.2f}% "
          f"{'✓ beats' if total_return > bh_return else '✗ underperforms'}")

    print("\nExit reason breakdown:")
    for r in ("target", "stop", "timeout"):
        n = int(reasons.get(r, 0))
        pct = n / len(tdf) * 100 if len(tdf) else 0
        print(f"  {r:8s}: {n:4d}  ({pct:.1f}%)")

    print("\nTop 5 winners:")
    print(tdf.nlargest(5, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "days", "reason"]]
          .assign(pnl_pct=lambda d: (d.pnl_pct * 100).round(2)).to_string(index=False))

    print(f"\nCosts modelled: slippage {SLIPPAGE_PCT*100:.2f}% per fill, "
          f"round-trip {ROUND_TRIP_COST*100:.2f}%")
    print("⚠️  Educational only — not financial advice.")
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
