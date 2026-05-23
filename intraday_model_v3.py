"""
intraday_model_v3.py -- Precision-maximizing intraday model v6.0.

Architecture (3-stage):
  Stage 1 (HOLD filter):   pooled LightGBM -- trained on first 70% of every symbol
  Stage 2 (Direction):     pooled LightGBM ensemble (5 000 est) -- BUY vs SELL
  Stage 3 (Meta):          TWO meta-classifiers + TWO rule-based premium overrides
                           -> meta-BUY  / Premium BUY rule
                           -> meta-SELL / Premium SELL rule (new in v6)
  Threshold:               fine-grained 0.01-step sweep on holdout OOT

v6.0 changes vs v5.0:
  1. Asymmetric ATR labels: BUY = 0.5*ATR, SELL = 0.7*ATR
     -> SELL requires a larger genuine move; cleanses noisy SELL labels
  2. Direction ensemble: n_estimators 3 000 -> 5 000 (was hitting ceiling)
  3. 5 new features (37 -> 42):
       candle_upper_shadow  upper wick / range -- rejection candle (bearish)
       candle_lower_shadow  lower wick / range -- buying tail (bullish)
       vwap_slope           VWAP deviation 3-bar rate-of-change
       vol_up_frac          session up-bar volume fraction − 0.5
       price_accel          2nd derivative of ret_3bar -- momentum acceleration
  4. Premium SELL rule: overbought contrarian short (analogous to Premium BUY)
  5. Meta-SELL trained with FP-penalty weighting (false-positive cost = 1.5*)
  6. Fine-grained threshold sweep (0.01 steps) + MIN_SIGNALS=25 floor on SELL

Split per symbol (temporal, no look-ahead):
  ── 70 % ── Stage 1+2 training pool
  ── 15 % ── Stage 3 meta training
  ── 15 % ── Holdout OOT (never touched)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score

sys.path.insert(0, str(Path(__file__).parent))
from swing_v2 import NIFTY_50
from intraday_model_v2 import (
    fetch_5min, fetch_nifty_5min, make_labels,
    _dir_proba_ens, _ema, _rsi,
    HOLD_THRESHOLD, BUY_CLASS_THRESH, DIR_THRESHOLD,
    TARGET_BARS, BUY_THRESH, SELL_THRESH,
    _make_lgbm_dir, _make_lgbm_hold,
)

# ── Module-level model cache (loaded once, shared across threads) ─────────────
_BLOB_CACHE: dict | None = None
_BLOB_LOCK  = __import__("threading").Lock()


def _load_blob() -> dict:
    """Load the v4 pkl once and keep it in module memory. Thread-safe."""
    global _BLOB_CACHE
    if _BLOB_CACHE is None:
        with _BLOB_LOCK:
            if _BLOB_CACHE is None:
                import __main__
                from swing_v2 import LGBMEnsemble
                __main__.LGBMEnsemble = LGBMEnsemble
                _BLOB_CACHE = joblib.load(MODEL_PATH_V3)
    return _BLOB_CACHE

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH_V3   = Path("models/intraday_v3.pkl")

REGIME_THRESHOLD = 0.015   # 1.5% Nifty day move suppresses opposite-direction signals

TRAIN_FRAC      = 0.70   # first 70% per symbol -> pooled Stage 1+2 training
META_FRAC       = 0.10   # next 10% per symbol  -> pooled Stage 3 meta training
VAL_FRAC        = 0.10   # next 10% per symbol  -> threshold tuning ONLY (never evaluated on)
# Holdout = last 10% per symbol  -- never used in any training or tuning decision

BASE_PRE_FILTER = 0.55   # dir_p ≥ this -> include bar in meta-BUY training data
ES_POOL_FRAC    = 0.85   # within training pool: first 85% = fit, last 15% = ES val

# v6.0 -- asymmetric ATR label thresholds
BUY_ATR_MULT   = 0.50    # BUY  requires  ≥ 0.5*ATR upside move
SELL_ATR_MULT  = 0.70    # SELL requires  ≥ 0.7*ATR downside move (harder -> cleaner labels)
ATR_LABEL_MIN  = 0.0040  # floor: always require at least 0.40% move
ATR_LABEL_MAX  = 0.0150  # ceiling: cap at 1.50%
META_SELL_FLOOR = 0.62   # minimum meta-SELL threshold
FP_PENALTY     = 1.50    # false-positive cost multiplier in meta-SELL training
MIN_SELL_SIGS  = 25      # minimum holdout signals required when sweeping SELL threshold

V3_FEATURE_COLS = [
    # ── Nifty regime / daily context ──────────────────────────────────────
    "nifty_day_ret",
    "nifty_ret_6bar",
    "nifty_rsi",
    "nifty_adx",
    "nifty_ema_sig",
    "rel_strength_day",
    # ── Timing / session ──────────────────────────────────────────────────
    "is_power_hour",
    "session_pct",
    "time_cos",
    "time_sin",
    # ── Opening dynamics ──────────────────────────────────────────────────
    "gap_pct",
    "gap_atr_norm",
    "opening_drive",
    "prev_close_pos",
    # ── Session position ──────────────────────────────────────────────────
    "vwap_pct",
    "dist_pd_high",
    "session_high_dist",
    # ── Intraday momentum ─────────────────────────────────────────────────
    "ret_3bar",
    "macd_fast_hist",
    "session_trend",
    "buy_pressure",
    # ── Volume ────────────────────────────────────────────────────────────
    "rvol_tod",
    # ── v3.0 ──────────────────────────────────────────────────────────────
    "nifty_ret_3bar",
    "session_vol_accel",
    # ── v3.1 ──────────────────────────────────────────────────────────────
    "stock_rsi_5m",
    "bb_squeeze",
    "stock_ema_align",
    # ── v4.0 ──────────────────────────────────────────────────────────────
    "body_ratio",           # bar conviction: |close-open|/(high-low)
    "consecutive_bars",     # consecutive same-direction bars, scaled /5
    "rsi_slope",            # RSI(9) 3-bar slope, normalized
    "vol_zscore",           # volume z-score within session
    "close_to_open_atr",    # intraday drift in ATR units from day open
    "nifty_accel",          # Nifty 6-bar momentum 2nd derivative
    # ── v5.0 ──────────────────────────────────────────────────────────────
    "session_low_dist",     # (close - session_low) / ATR -- above day low
    "open_range_pos",       # price position in opening 15-min range [-1.5, 1.5]
    "intraday_range_pct",   # (session_high - session_low) / ATR -- day expansion
    "price_mom_rel_nifty",  # stock ret_3bar - nifty_ret_3bar -- relative momentum
    # ── v6.0 (new) ────────────────────────────────────────────────────────
    "candle_upper_shadow",  # upper wick / range -- rejection of higher prices (bearish)
    "candle_lower_shadow",  # lower wick / range -- buying tail (bullish)
    "vwap_slope",           # VWAP deviation 3-bar rate of change
    "vol_up_frac",          # session up-bar volume fraction − 0.5
    "price_accel",          # 2nd derivative of ret_3bar -- momentum acceleration
]
assert len(V3_FEATURE_COLS) == 42

# Meta features: base 33 + Stage 1+2 model outputs
META_FEATURE_COLS = V3_FEATURE_COLS + ["dir_proba", "hold_proba"]  # 35 total


# ── Feature engineering ──────────────────────────────────────────────────────
def compute_features_v3(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return 33-feature DataFrame. Extends v3.1 with 6 new conviction features."""
    from intraday_model_v2 import compute_features as _base_compute
    base = _base_compute(df)
    if base is None or base.empty:
        return None

    df = df.sort_values("DateTime").reset_index(drop=True)
    df = df.loc[base.index]

    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    open_  = df["Open"].astype(float)
    vol    = df["Volume"].astype(float)
    date_col = df["DateTime"].dt.date

    # ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().replace(0, np.nan)

    # body_ratio: bar conviction [0,1]
    base["body_ratio"] = ((close - open_).abs() / (high - low + 1e-8)).clip(0, 1).fillna(0.5)

    # consecutive_bars: ±1.0 for ≥5 consecutive same-direction bars
    direction = np.sign(close.values - open_.values)
    consec = np.zeros(len(direction))
    count  = 0
    for i in range(len(direction)):
        d = direction[i]
        if d == 0:
            count = 0
        elif i == 0 or d != direction[i-1]:
            count = d
        else:
            count += d
        consec[i] = count
    base["consecutive_bars"] = np.clip(consec / 5.0, -1, 1)

    # rsi_slope: RSI(9) 3-bar slope, normalized
    rsi9 = _rsi(close, 9)
    base["rsi_slope"] = (rsi9.diff(3).fillna(0) / 30.0).clip(-1, 1)

    # vol_zscore: z-score within session
    daily_mean = vol.groupby(date_col.values).transform("mean")
    daily_std  = vol.groupby(date_col.values).transform("std").replace(0, np.nan)
    base["vol_zscore"] = ((vol - daily_mean) / daily_std).clip(-3, 3).fillna(0)

    # close_to_open_atr: intraday drift in ATR units
    day_open = df.groupby(date_col)["Open"].transform("first")
    base["close_to_open_atr"] = ((close - day_open) / atr14).clip(-5, 5).fillna(0)

    # nifty_accel: 2nd derivative of Nifty 6-bar momentum
    base["nifty_accel"] = (base["nifty_ret_6bar"].diff(2).fillna(0) * 100).clip(-2, 2)

    # ── v5.0 features ─────────────────────────────────────────────────────
    # session_low_dist: distance from session low in ATR units
    session_low = df.groupby(date_col)["Low"].transform("min")
    base["session_low_dist"] = ((close - session_low) / atr14).clip(0, 10).fillna(0)

    # open_range_pos: price position in opening 15-min (3 bars) range
    _or_h = df.groupby(date_col)["High"].transform(lambda x: x.iloc[:min(3, len(x))].max())
    _or_l = df.groupby(date_col)["Low"].transform(lambda x: x.iloc[:min(3, len(x))].min())
    _or_rng = (_or_h - _or_l).replace(0, np.nan)
    base["open_range_pos"] = ((close - _or_l) / _or_rng - 0.5).clip(-1.5, 1.5).fillna(0)

    # intraday_range_pct: session range relative to ATR
    session_high_v = df.groupby(date_col)["High"].transform("max")
    session_low_v  = df.groupby(date_col)["Low"].transform("min")
    base["intraday_range_pct"] = ((session_high_v - session_low_v) / atr14).clip(0, 5).fillna(1.0)

    # price_mom_rel_nifty: stock vs market 3-bar momentum
    base["price_mom_rel_nifty"] = (base["ret_3bar"] - base["nifty_ret_3bar"]).clip(-0.05, 0.05)

    # ── v6.0 features ─────────────────────────────────────────────────────
    _bar_rng = (high - low).replace(0, np.nan)
    # candle_upper_shadow: upper wick proportion -- rejection of higher prices
    base["candle_upper_shadow"] = (
        (high - pd.concat([close, open_], axis=1).max(axis=1)) / _bar_rng
    ).clip(0, 1).fillna(0)
    # candle_lower_shadow: lower wick proportion -- buying tail
    base["candle_lower_shadow"] = (
        (pd.concat([close, open_], axis=1).min(axis=1) - low) / _bar_rng
    ).clip(0, 1).fillna(0)

    # vwap_slope: 3-bar rate-of-change of the VWAP deviation (already in base)
    base["vwap_slope"] = (base["vwap_pct"].diff(3).fillna(0) * 100).clip(-2, 2)

    # vol_up_frac: fraction of session volume in up-bars, centered at 0
    _is_up      = (close >= open_).astype(float)
    _vol_up_cum = ((_is_up * vol).groupby(date_col.values, group_keys=False)
                   .cumsum())
    _vol_cum    = (vol.groupby(date_col.values, group_keys=False).cumsum())
    base["vol_up_frac"] = (
        (_vol_up_cum / _vol_cum.replace(0, np.nan)) - 0.5
    ).clip(-0.5, 0.5).fillna(0)

    # price_accel: 2nd derivative of ret_3bar -- momentum acceleration
    base["price_accel"] = (base["ret_3bar"].diff(3).fillna(0) * 100).clip(-3, 3)

    return base[V3_FEATURE_COLS]


# ── ATR-relative labeling (v5.0) ─────────────────────────────────────────────
def make_labels_atr(df: pd.DataFrame) -> pd.Series:
    """
    Path-aware labels with asymmetric ATR-relative thresholds (v6.0).
    BUY  threshold = max(ATR_LABEL_MIN, BUY_ATR_MULT  * ATR(14)/close)
    SELL threshold = max(ATR_LABEL_MIN, SELL_ATR_MULT * ATR(14)/close)
    SELL requires a harder move (0.7 vs 0.5 ATR mult) -- cleaner negative labels.
    """
    close_v = df["Close"].values.astype(float)
    high_v  = df["High"].values.astype(float)
    low_v   = df["Low"].values.astype(float)

    prev_c  = np.concatenate([[close_v[0]], close_v[:-1]])
    tr_vals = np.maximum.reduce([
        high_v - low_v,
        np.abs(high_v - prev_c),
        np.abs(low_v  - prev_c),
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_frac = np.where(close_v > 0, tr_vals / close_v, np.nan)
    atr_frac = pd.Series(atr_frac).rolling(14, min_periods=5).mean().bfill().fillna(ATR_LABEL_MIN).values

    buy_thr  = np.clip(BUY_ATR_MULT  * atr_frac, ATR_LABEL_MIN, ATR_LABEL_MAX)
    sell_thr = np.clip(SELL_ATR_MULT * atr_frac, ATR_LABEL_MIN, ATR_LABEL_MAX)

    n, m    = len(close_v), len(close_v) - TARGET_BARS
    lbl     = np.zeros(n, dtype=int)
    decided = np.zeros(m, dtype=bool)
    idx     = np.arange(m)

    for k in range(1, TARGET_BARS + 1):
        fi    = idx + k
        ref   = close_v[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_h = np.where(ref > 0, high_v[fi] / ref - 1, 0.0)
            ret_l = np.where(ref > 0, low_v[fi]  / ref - 1, 0.0)
        still = ~decided
        hit_b = still & (ret_h >=  buy_thr[idx])
        hit_s = still & (ret_l <= -sell_thr[idx])
        both  = hit_b & hit_s
        hit_b = hit_b & ~(both & (ret_h < np.abs(ret_l)))
        hit_s = hit_s & ~(both & (ret_h >= np.abs(ret_l)))
        lbl[idx[hit_b]] =  1
        lbl[idx[hit_s]] = -1
        decided        |= hit_b | hit_s
    return pd.Series(lbl, index=df.index)


# ── Training helpers ─────────────────────────────────────────────────────────
def _make_lgbm_dir_v6(seed: int, col_frac: float, row_frac: float) -> lgb.LGBMClassifier:
    """Direction model with higher capacity (5 000 est) -- v6.0."""
    return lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.006,
        max_depth=6,
        num_leaves=40,
        min_child_samples=50,
        feature_fraction=col_frac,
        bagging_fraction=row_frac,
        bagging_freq=5,
        lambda_l1=0.2,
        lambda_l2=2.0,
        objective="binary",
        metric="binary_logloss",
        verbose=-1,
        n_jobs=-1,
        random_state=seed,
    )


def _make_lgbm_meta(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.006,
        max_depth=5,
        num_leaves=24,
        min_child_samples=20,
        feature_fraction=0.70,
        bagging_fraction=0.80,
        bagging_freq=5,
        lambda_l1=0.5,
        lambda_l2=3.0,
        objective="binary",
        metric="binary_logloss",
        verbose=-1,
        n_jobs=-1,
        random_state=seed,
    )


def _train_pooled_base(X_tr: pd.DataFrame, y_tr: pd.Series
                       ) -> tuple[list, object]:
    """Train pooled Stage 1 + Stage 2 on concatenated multi-symbol data."""
    es_cut = int(len(X_tr) * ES_POOL_FRAC)
    X_fit  = X_tr.iloc[:es_cut];  y_fit = y_tr.iloc[:es_cut]
    X_val  = X_tr.iloc[es_cut:];  y_val = y_tr.iloc[es_cut:]

    # Stage 2: direction model (BUY vs SELL)
    bs   = y_fit.isin([1, -1])
    X_bs = X_fit[bs]; y_bs = (y_fit[bs] == 1).astype(int)
    bv   = y_val.isin([1, -1])
    X_bv = X_val[bv]; y_bv = (y_val[bv] == 1).astype(int)
    n_b  = (y_bs == 1).sum(); n_s = (y_bs == 0).sum()
    if n_b < 50 or n_s < 50:
        return [], None
    w_dir = y_bs.map({1: (n_b+n_s)/(2*n_b), 0: (n_b+n_s)/(2*n_s)}).values
    models_dir = []
    for seed, colf, rowf in [(42, 0.70, 0.80), (137, 0.65, 0.75), (911, 0.75, 0.85)]:
        m = _make_lgbm_dir_v6(seed, colf, rowf)
        cb = lgb.early_stopping(150, verbose=False)
        m.fit(X_bs, y_bs, sample_weight=w_dir,
              eval_set=[(X_bv, y_bv)], callbacks=[cb])
        models_dir.append(m)
        print(f"    dir[seed={seed}] best_iter={m.best_iteration_}")

    # Stage 1: HOLD filter
    y_nh_fit = (y_fit != 0).astype(int)
    y_nh_val = (y_val != 0).astype(int)
    n_nh = y_nh_fit.sum(); n_h = (y_nh_fit == 0).sum()
    w_hold = y_nh_fit.map(
        {1: (n_nh+n_h)/(2*n_nh), 0: (n_nh+n_h)/(2*n_h) * 0.8}
    ).values
    mh = _make_lgbm_hold(42)
    cb_h = lgb.early_stopping(120, verbose=False)
    mh.fit(X_fit, y_nh_fit, sample_weight=w_hold,
           eval_set=[(X_val, y_nh_val)], callbacks=[cb_h])
    print(f"    hold best_iter={mh.best_iteration_}")

    return models_dir, mh


def _get_probas(models_dir: list, model_hold, X: pd.DataFrame
                ) -> tuple[np.ndarray, np.ndarray]:
    hold_p = model_hold.predict_proba(X)[:, 1]
    dir_p  = np.full(len(X), 0.5)
    active = hold_p >= HOLD_THRESHOLD
    if active.sum() > 0:
        dir_p[active] = _dir_proba_ens(models_dir, X[active])
    return dir_p, hold_p


def _train_meta_model(X: np.ndarray, y: np.ndarray, name: str,
                      fp_penalty: float = 1.0) -> tuple | tuple[None, None]:
    """
    Train a meta-classifier.  fp_penalty > 1.0 increases cost of false positives,
    pushing the model toward higher precision at the expense of recall.
    """
    print(f"\n  {name}: {len(X)} rows  pos={y.sum()}  neg={(y==0).sum()}"
          f"  base_rate={y.mean():.1%}"
          + (f"  fp_penalty={fp_penalty:.1f}" if fp_penalty != 1.0 else ""))
    if len(X) < 80 or y.sum() < 15 or (y == 0).sum() < 15:
        print(f"  {name}: insufficient data -- skip"); return None, None

    val_cut = int(len(X) * 0.80)
    Xf, Xv = X[:val_cut], X[val_cut:]
    yf, yv = y[:val_cut], y[val_cut:]
    n_p = yf.sum(); n_n = (yf == 0).sum()
    # FP penalty: multiply non-signal class weight (reduces false positives)
    w   = np.where(yf == 1,
                   (n_p+n_n)/(2*n_p),
                   (n_p+n_n)/(2*n_n) * fp_penalty)
    m   = _make_lgbm_meta(42)
    cb  = lgb.early_stopping(120, verbose=False)
    m.fit(Xf, yf, sample_weight=w, eval_set=[(Xv, yv)], callbacks=[cb])

    raw_v = m.predict_proba(Xv)[:, 1]
    iso   = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_v, yv)
    cal_v = iso.predict(raw_v)

    # Precision sweep on meta validation set
    print(f"  {name} (meta-val sweep):")
    print(f"    {'Thr':>5}  {'N':>6}  {'Prec':>8}  {'Recall':>8}")
    best_thr = 0.75; best_prec = 0.0
    for thr in np.arange(0.50, 0.96, 0.05):
        emit = cal_v >= thr
        if emit.sum() == 0: break
        pr   = precision_score(yv, emit.astype(int), zero_division=0)
        rec  = emit[yv == 1].mean() if yv.sum() > 0 else 0
        print(f"    {thr:>5.2f}  {emit.sum():>6}  {pr:>8.1%}  {rec:>8.1%}")
        if pr > best_prec:
            best_prec = pr; best_thr = thr
    print(f"    best_iter={m.best_iteration_}  val_best={best_prec:.1%}@{best_thr:.2f}")
    return m, iso


# ── Main training ─────────────────────────────────────────────────────────────
def train(syms: list[str], test_mode: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"intraday_model_v3  v6.0 train  {'(TEST MODE)' if test_mode else ''}")
    print(f"Symbols: {len(syms)}  Features: {len(V3_FEATURE_COLS)}  Split: 70/10/10/10")
    print(f"ATR labels  |  meta-BUY + meta-SELL  |  thr tuned on val  |  unbiased holdout")
    print(f"{'='*60}\n")

    fetch_nifty_5min(days=60)  # warm Nifty cache

    # ── Phase 1: Fetch & compute features per symbol ───────────────────────
    print("── Phase 1: Fetch & compute features ────────────────────────────────")
    all_data: dict[str, tuple] = {}
    for sym in syms:
        df = fetch_5min(sym, days=60)
        if df is None or len(df) < 300:
            print(f"  {sym}: skip"); continue
        feats = compute_features_v3(df)
        if feats is None or feats.empty:
            print(f"  {sym}: feature error"); continue
        labels = make_labels_atr(df)
        common = feats.index.intersection(labels.index)
        feats  = feats.loc[common]; labels = labels.loc[common]
        dt_s   = pd.Series(df.loc[common, "DateTime"].values, index=common)
        all_data[sym] = (feats, labels, dt_s)
        n = len(feats)
        buy_n = (labels == 1).sum(); sell_n = (labels == -1).sum()
        print(f"  {sym}: {n} bars  BUY={buy_n}  SELL={sell_n}  HOLD={(labels==0).sum()}")

    if len(all_data) < 5:
        print("Too few symbols -- aborting."); return

    # ── Phase 2: Temporal splits (4-way: train | meta | val | holdout) ───────
    # val  = threshold tuning set  (influences meta thresholds, nothing else)
    # holdout = truly untouched    (no training, no tuning decision used it)
    train_Xs: list[pd.DataFrame] = []
    train_ys: list[pd.Series]    = []
    meta_Xs: list[pd.DataFrame]  = []
    meta_ys: list[pd.Series]     = []
    val_data: list[tuple]        = []
    holdout:  list[tuple]        = []

    for sym, (feats, labels, dt_s) in all_data.items():
        n   = len(feats)
        t1  = int(n * TRAIN_FRAC)
        t2  = int(n * (TRAIN_FRAC + META_FRAC))
        t3  = int(n * (TRAIN_FRAC + META_FRAC + VAL_FRAC))
        train_Xs.append(feats.iloc[:t1]);  train_ys.append(labels.iloc[:t1])
        meta_Xs.append(feats.iloc[t1:t2]); meta_ys.append(labels.iloc[t1:t2])
        val_data.append((sym, feats.iloc[t2:t3][V3_FEATURE_COLS],
                         labels.iloc[t2:t3], dt_s.iloc[t2:t3]))
        holdout.append((sym, feats.iloc[t3:][V3_FEATURE_COLS],
                        labels.iloc[t3:], dt_s.iloc[t3:]))

    X_tr_all = pd.concat(train_Xs, ignore_index=True)
    y_tr_all = pd.concat(train_ys, ignore_index=True)
    X_mt_all = pd.concat(meta_Xs, ignore_index=True)
    y_mt_all = pd.concat(meta_ys, ignore_index=True)

    print(f"\n  Training pool : {len(X_tr_all):,} rows")
    print(f"  Meta pool     : {len(X_mt_all):,} rows")
    print(f"  Val (thr tune): {sum(len(v[1]) for v in val_data):,} rows")
    print(f"  Holdout (OOT) : {sum(len(h[1]) for h in holdout):,} rows")

    # ── Phase 3: Train pooled Stage 1+2 ──────────────────────────────────
    print("\n── Phase 3: Train pooled Stage 1+2 ──────────────────────────────────")
    X_feat_tr = X_tr_all[V3_FEATURE_COLS]
    models_dir, model_hold = _train_pooled_base(X_feat_tr, y_tr_all)
    if not models_dir:
        print("Base model training failed."); return

    # Stage 2 OOT report on the meta pool (naive, as sanity check)
    print(f"\n  Sanity check -- Stage 2 on meta pool (n={len(X_mt_all):,}):")
    dir_pm, hold_pm = _get_probas(models_dir, model_hold, X_mt_all[V3_FEATURE_COLS])
    active_m = hold_pm >= HOLD_THRESHOLD
    buy_s2   = active_m & (dir_pm >= DIR_THRESHOLD)
    sell_s2  = active_m & ((1 - dir_pm) >= DIR_THRESHOLD)
    lbl_m    = y_mt_all.values
    s2_bp    = (lbl_m[buy_s2] == 1).mean() if buy_s2.sum() > 0 else 0
    s2_sp    = (lbl_m[sell_s2] == -1).mean() if sell_s2.sum() > 0 else 0
    print(f"    BUY  {buy_s2.sum():>5} signals  precision={s2_bp:.1%}")
    print(f"    SELL {sell_s2.sum():>5} signals  precision={s2_sp:.1%}")

    # ── Phase 4: Build meta-train rows ────────────────────────────────────
    print("\n── Phase 4: Build Stage 3 meta training rows ────────────────────────")
    meta_rows: list[dict] = []
    for i in range(len(X_mt_all)):
        if not active_m[i]: continue
        dp = float(dir_pm[i]); hp = float(hold_pm[i])
        if dp >= BASE_PRE_FILTER or dp <= (1 - BASE_PRE_FILTER):
            row = {f: X_mt_all.iloc[i][f] for f in V3_FEATURE_COLS}
            row["dir_proba"]  = dp
            row["hold_proba"] = hp
            row["label"]      = int(y_mt_all.iloc[i])
            meta_rows.append(row)

    print(f"  Meta-train rows: {len(meta_rows)}"
          f"  ({len(meta_rows)/len(X_mt_all):.1%} of meta pool)")

    df_meta = pd.DataFrame(meta_rows)
    Xm_all  = df_meta[META_FEATURE_COLS].values.astype(float)
    lbl_m2  = df_meta["label"].values

    meta_dp  = dir_pm
    meta_lbl = y_mt_all.values
    meta_rsi = X_mt_all["stock_rsi_5m"].values
    meta_pwr = X_mt_all["is_power_hour"].values
    active_m_arr = hold_pm >= HOLD_THRESHOLD

    # ── Phase 5a: Premium BUY rule calibration ─────────────────────────────
    print("\n── Phase 5a: Premium BUY rule calibration ────────────────────────────")
    print(f"  {'Dir_thr':>8}  {'RSI_max':>8}  {'Power':>6}  {'N':>6}  {'Prec':>8}")
    best_buy_rule = {"dir_min": 0.75, "rsi_max": 0.40, "power": True, "prec": 0.0}
    for dp_thr in [0.70, 0.75, 0.80]:
        for rsi_max in [0.25, 0.30, 0.35, 0.40, 0.45]:
            for pw in [True, False]:
                mask = (active_m_arr & (meta_dp >= dp_thr) &
                        (meta_rsi < rsi_max) &
                        ((meta_pwr > 0) if pw else True))
                n = mask.sum()
                if n < 5: continue
                prec = (meta_lbl[mask] == 1).mean()
                flag = " ← 80%+" if prec >= 0.80 else ""
                print(f"  dp>={dp_thr:.2f}  rsi<{rsi_max:.2f}  pow={'Y' if pw else 'N'}  "
                      f"N={n:>6}  {prec:>8.1%}{flag}")
                if prec >= best_buy_rule["prec"] and n >= 5:
                    best_buy_rule = {"dir_min": dp_thr, "rsi_max": rsi_max, "power": pw,
                                     "prec": prec, "n": n}

    print(f"\n  Best Premium BUY: dir>={best_buy_rule['dir_min']:.2f}  "
          f"rsi<{best_buy_rule['rsi_max']:.2f}  "
          f"power={'Y' if best_buy_rule['power'] else 'N'}  "
          f"-> {best_buy_rule['prec']:.1%}  (N={best_buy_rule.get('n',0)})")

    # ── Phase 5a': Premium SELL rule calibration ────────────────────────────
    # Mirror of Premium BUY: overbought stock (RSI>70), very confident SELL signal,
    # Nifty momentum positive (so SELL is contrarian)
    print("\n── Phase 5a': Premium SELL rule calibration ──────────────────────────")
    meta_nifty6 = X_mt_all["nifty_ret_6bar"].values
    print(f"  {'Dir_thr':>8}  {'RSI_min':>8}  {'Power':>6}  {'N':>6}  {'Prec':>8}")
    best_sell_rule = {"dir_max": 0.25, "rsi_min": 0.60, "power": True, "prec": 0.0}
    for dp_ceil in [0.30, 0.25, 0.20]:
        for rsi_min in [0.55, 0.60, 0.65, 0.70]:
            for pw in [True, False]:
                mask = (active_m_arr & (meta_dp <= dp_ceil) &
                        (meta_rsi > rsi_min) &
                        ((meta_pwr > 0) if pw else True))
                n = mask.sum()
                if n < 5: continue
                prec = (meta_lbl[mask] == -1).mean()
                flag = " ← 75%+" if prec >= 0.75 else ""
                print(f"  dp<={dp_ceil:.2f}  rsi>{rsi_min:.2f}  pow={'Y' if pw else 'N'}  "
                      f"N={n:>6}  {prec:>8.1%}{flag}")
                if prec >= best_sell_rule["prec"] and n >= 5:
                    best_sell_rule = {"dir_max": dp_ceil, "rsi_min": rsi_min, "power": pw,
                                      "prec": prec, "n": n}

    print(f"\n  Best Premium SELL: dir<={best_sell_rule['dir_max']:.2f}  "
          f"rsi>{best_sell_rule['rsi_min']:.2f}  "
          f"power={'Y' if best_sell_rule['power'] else 'N'}  "
          f"-> {best_sell_rule['prec']:.1%}  (N={best_sell_rule.get('n',0)})")

    # ── Phase 5b: Train Stage 3 meta-BUY and meta-SELL classifiers ───────
    print("\n── Phase 5b: Meta-BUY classifier ────────────────────────────────────")
    buy_mask  = df_meta["dir_proba"].values >= BASE_PRE_FILTER
    X_mbuy    = Xm_all[buy_mask]
    y_mbuy    = (lbl_m2[buy_mask] == 1).astype(int)
    meta_buy_model, meta_buy_cal = _train_meta_model(X_mbuy, y_mbuy, "Meta-BUY")

    print("\n── Phase 5c: Meta-SELL classifier (FP-penalty) ──────────────────────")
    sell_mask = df_meta["dir_proba"].values <= (1 - BASE_PRE_FILTER)
    X_msell   = Xm_all[sell_mask]
    y_msell   = (lbl_m2[sell_mask] == -1).astype(int)
    meta_sell_model, meta_sell_cal = _train_meta_model(
        X_msell, y_msell, "Meta-SELL", fp_penalty=FP_PENALTY
    )

    # ── Phase 6a: Threshold tuning on VAL set (separate from holdout) ────────
    # Thresholds are chosen on val_data ONLY -- holdout is never touched here.
    print("\n── Phase 6a: Threshold tuning on VAL set ────────────────────────────")

    def _collect_meta_probas(dataset, label_val: int):
        """Run Stage 3 meta model over dataset; return (proba_list, label_list)."""
        all_p: list[float] = []; all_y: list[int] = []
        for sym, X_ds, y_ds, _ in dataset:
            if len(X_ds) < 20: continue
            dir_pd, hold_pd = _get_probas(models_dir, model_hold, X_ds)
            lbl_d  = y_ds.values
            active = hold_pd >= HOLD_THRESHOLD
            if label_val == 1:
                pre = active & (dir_pd >= BASE_PRE_FILTER)
                model, cal = meta_buy_model, meta_buy_cal
            else:
                pre = active & (dir_pd <= 1 - BASE_PRE_FILTER)
                model, cal = meta_sell_model, meta_sell_cal
            if model is None or pre.sum() == 0: continue
            Xm  = np.column_stack([X_ds[pre].values, dir_pd[pre], hold_pd[pre]])
            raw = model.predict_proba(Xm)[:, 1]
            c   = cal.predict(raw)
            all_p.extend(c.tolist())
            all_y.extend((lbl_d[pre] == label_val).astype(int).tolist())
        return all_p, all_y

    def _collect_stage2(dataset):
        """Collect Stage 2 stats for a dataset (sanity baseline)."""
        buy_n = sell_n = 0
        buy_wins = sell_wins = 0
        for sym, X_ds, y_ds, _ in dataset:
            if len(X_ds) < 20: continue
            dir_pd, hold_pd = _get_probas(models_dir, model_hold, X_ds)
            lbl_d = y_ds.values
            active = hold_pd >= HOLD_THRESHOLD
            bs = active & (dir_pd >= DIR_THRESHOLD)
            ss = active & ((1 - dir_pd) >= DIR_THRESHOLD)
            buy_n   += int(bs.sum()); buy_wins  += int((lbl_d[bs] == 1).sum())
            sell_n  += int(ss.sum()); sell_wins += int((lbl_d[ss] == -1).sum())
        return buy_n, buy_wins, sell_n, sell_wins

    val_buy_p,  val_buy_y  = _collect_meta_probas(val_data,  1)
    val_sell_p, val_sell_y = _collect_meta_probas(val_data, -1)

    vbn, vbw, vsn, vsw = _collect_stage2(val_data)
    print(f"\n  Stage 2 on val set:  BUY {vbn} sigs {vbw/vbn:.1%}" if vbn else "  BUY: 0", end="")
    print(f"  |  SELL {vsn} sigs {vsw/vsn:.1%}" if vsn else "  |  SELL: 0")

    def _sweep(proba_arr, label_arr, name, min_sigs: int = 10) -> float:
        if not proba_arr:
            print(f"\n  {name}: no data"); return 0.75
        p  = np.array(proba_arr); y_ = np.array(label_arr)
        print(f"\n  {name} ({len(p)} val bars, base_rate={y_.mean():.1%}, min_sigs={min_sigs}):")
        print(f"    {'Thr':>5}  {'N':>6}  {'Precision':>10}  {'Recall':>8}")
        best_thr = 0.75; best_prec = 0.0; found_80 = False
        for thr in list(np.arange(0.50, 0.97, 0.01)):
            emit = p >= thr; n = int(emit.sum())
            if n < min_sigs: break
            pr  = precision_score(y_, emit.astype(int), zero_division=0)
            rec = emit[y_ == 1].mean() if y_.sum() > 0 else 0
            flag = "  <- 80% ✓" if pr >= 0.80 and not found_80 else ""
            if thr in np.arange(0.50, 0.97, 0.05) or pr >= 0.78 or n <= min_sigs + 20:
                print(f"    {thr:>5.2f}  {n:>6}  {pr:>10.1%}  {rec:>8.1%}{flag}")
            if pr >= 0.80 and not found_80: found_80 = True
            if pr > best_prec: best_prec = pr; best_thr = thr
        if not found_80:
            print(f"    [80% NOT reached -- best: {best_prec:.1%} @ {best_thr:.2f}]")
        return best_thr

    best_buy_thr  = _sweep(val_buy_p,  val_buy_y,  "Meta-BUY  val",  min_sigs=10)
    best_sell_thr = max(META_SELL_FLOOR,
                        _sweep(val_sell_p, val_sell_y, "Meta-SELL val",
                               min_sigs=MIN_SELL_SIGS))

    # ── Phase 6b: Unbiased evaluation on TRUE holdout ────────────────────────
    # Thresholds are now fixed. Holdout never influenced any prior decision.
    print("\n── Phase 6b: Unbiased evaluation on TRUE holdout ───────────────────")

    hd_buy_p,  hd_buy_y  = _collect_meta_probas(holdout,  1)
    hd_sell_p, hd_sell_y = _collect_meta_probas(holdout, -1)

    hbn, hbw, hsn, hsw = _collect_stage2(holdout)
    print(f"\n  Stage 2 on holdout:  BUY {hbn} sigs {hbw/hbn:.1%}" if hbn else "  BUY: 0", end="")
    print(f"  |  SELL {hsn} sigs {hsw/hsn:.1%}" if hsn else "  |  SELL: 0")

    def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """95% Wilson confidence interval for a proportion k/n."""
        if n == 0: return 0.0, 1.0
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
        return max(0.0, centre - margin), min(1.0, centre + margin)

    def _report_holdout(proba_arr, label_arr, thr, name):
        if not proba_arr:
            print(f"\n  {name}: no holdout data"); return 0, 0, 0.0
        p = np.array(proba_arr); y_ = np.array(label_arr)
        emit = p >= thr; n = int(emit.sum())
        win  = int((emit & (y_ == 1)).sum())
        prec = win / n if n > 0 else 0.0
        lo, hi = _wilson_ci(win, n)
        ok = "✓" if prec >= 0.80 else "✗"
        print(f"\n  {ok} {name}: {win}/{n} = {prec:.1%}  "
              f"95% CI [{lo:.1%}, {hi:.1%}]  (thr={thr:.2f})")
        return n, win, prec

    buy_n, buy_win, buy_prec   = _report_holdout(hd_buy_p,  hd_buy_y,  best_buy_thr,  "BUY ")
    sell_n, sell_win, sell_prec = _report_holdout(hd_sell_p, hd_sell_y, best_sell_thr, "SELL")

    # ── Save ──────────────────────────────────────────────────────────────
    blob = {
        "version":           "6.0",
        "trained_at":        date.today().isoformat(),
        "feature_cols":      V3_FEATURE_COLS,
        "meta_feature_cols": META_FEATURE_COLS,
        "models_dir":        models_dir,
        "model_hold":        model_hold,
        # BUY: meta-classifier + rule-based premium override
        "premium_buy_rule":   best_buy_rule,
        "premium_sell_rule":  best_sell_rule,
        "meta_buy_model":    meta_buy_model,
        "meta_buy_cal":      meta_buy_cal,
        "meta_sell_model":   meta_sell_model,
        "meta_sell_cal":     meta_sell_cal,
        "meta_buy_thresh":   best_buy_thr,
        "meta_sell_thresh":  best_sell_thr,
        "dir_threshold":     DIR_THRESHOLD,
        "hold_threshold":    HOLD_THRESHOLD,
        "base_pre_filter":   BASE_PRE_FILTER,
        "holdout_buy_prec":  buy_prec,
        "holdout_sell_prec": sell_prec,
        "n_syms":            len(all_data),
        "sym_list":          list(all_data.keys()),
    }
    MODEL_PATH_V3.parent.mkdir(exist_ok=True)
    joblib.dump(blob, MODEL_PATH_V3, compress=3)
    print(f"\n✓ Saved: {MODEL_PATH_V3}"
          f"  ({MODEL_PATH_V3.stat().st_size/1e6:.1f} MB)")
    print(f"  BUY  unbiased holdout: {buy_prec:.1%}  ({buy_win}/{buy_n})  "
          f"@ meta_thr={best_buy_thr:.2f}  (thr tuned on val set)")
    print(f"  SELL unbiased holdout: {sell_prec:.1%}  ({sell_win}/{sell_n})  "
          f"@ meta_thr={best_sell_thr:.2f}  (thr tuned on val set)")
    lo_b, hi_b   = _wilson_ci(buy_win,  buy_n)
    lo_s, hi_s   = _wilson_ci(sell_win, sell_n)
    print(f"  95% CI:  BUY [{lo_b:.1%}, {hi_b:.1%}]  |  SELL [{lo_s:.1%}, {hi_s:.1%}]")
    if buy_prec >= 0.80:
        print(f"\n  ✓ 80% BUY precision ACHIEVED on unbiased holdout!")
    else:
        print(f"\n  Best BUY: {buy_prec:.1%}  (gap to 80%: {0.80-buy_prec:.1%})")


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(symbol: str) -> dict:
    if not MODEL_PATH_V3.exists():
        return {"error": f"Model not found: {MODEL_PATH_V3}"}

    blob = _load_blob()
    models_dir    = blob["models_dir"]
    model_hold    = blob["model_hold"]
    meta_buy_m    = blob.get("meta_buy_model")
    meta_buy_cal  = blob.get("meta_buy_cal")
    meta_sell_m   = blob.get("meta_sell_model")
    meta_sell_cal = blob.get("meta_sell_cal")
    feat_cols     = blob.get("feature_cols",      V3_FEATURE_COLS)
    meta_fcols    = blob.get("meta_feature_cols",  META_FEATURE_COLS)
    buy_thr       = blob.get("meta_buy_thresh",    0.75)
    sell_thr      = blob.get("meta_sell_thresh",   0.75)
    dir_thr       = blob.get("dir_threshold",      DIR_THRESHOLD)
    hold_thr      = blob.get("hold_threshold",     HOLD_THRESHOLD)
    pre_flt       = blob.get("base_pre_filter",    BASE_PRE_FILTER)

    try:
        from data_cache import get_features, get_bars
        feats    = get_features(symbol, days=5)
        raw_bars = get_bars(symbol, days=5)
    except ImportError:
        raw_bars = fetch_5min(symbol, days=5)
        feats    = compute_features_v3(raw_bars) if raw_bars is not None and len(raw_bars) >= 40 else None

    if feats is None or feats.empty:
        return {"signal": "HOLD", "error": "no data / feature error", "data_ok": False}

    # Current price and ATR(14) from 5-min bars for stop/target calculation
    entry_price = stop_price = target_price = atr_5m = None
    if raw_bars is not None and len(raw_bars) >= 15:
        closes = raw_bars["Close"].astype(float)
        highs  = raw_bars["High"].astype(float)
        lows   = raw_bars["Low"].astype(float)
        entry_price = round(float(closes.iloc[-1]), 2)
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows  - closes.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_5m = float(tr.rolling(14).mean().iloc[-1])

    X = feats[feat_cols].tail(1)
    if X.isna().any().any():
        return {"signal": "HOLD", "error": "NaN features", "data_ok": False}

    hold_p = float(model_hold.predict_proba(X)[0][1])
    if hold_p < hold_thr:
        return {"signal": "HOLD", "hold_prob": round(1-hold_p, 3), "data_ok": True}

    dir_p = float(_dir_proba_ens(models_dir, X)[0])

    s2 = ("BUY" if dir_p >= dir_thr else
          "SELL" if (1 - dir_p) >= dir_thr else "HOLD")

    meta_signal = "HOLD"; meta_conf = 0.0; premium = False
    latest_feats = feats[feat_cols].iloc[-1]
    Xm = np.concatenate([X.values[0], [dir_p, hold_p]]).reshape(1, -1)
    Xm_df = pd.DataFrame(Xm, columns=meta_fcols)

    # Stage 3 BUY: meta-classifier for regular BUY signals
    if meta_signal == "HOLD" and dir_p >= pre_flt and meta_buy_m is not None:
        raw = meta_buy_m.predict_proba(Xm_df)[:, 1][0]
        cal = float(meta_buy_cal.predict([raw])[0])
        if cal >= buy_thr:
            meta_signal = "BUY"; meta_conf = cal; premium = False

    # Stage 3 BUY override: Premium rule (oversold contrarian bounce)
    # dir_p≥0.80 + RSI<30% + power_hour + prec≥75% required (OOT2 showed 72.1% overfits)
    prem_rule = blob.get("premium_buy_rule", {})
    if prem_rule and prem_rule.get("prec", 0) >= 0.75 and dir_p >= prem_rule.get("dir_min", 0.80):
        stock_rsi  = float(latest_feats.get("stock_rsi_5m", 1.0))
        is_pow     = float(latest_feats.get("is_power_hour", 0.0))
        nifty_ret6 = float(latest_feats.get("nifty_ret_6bar", 0.0))
        rsi_ok     = stock_rsi < prem_rule.get("rsi_max", 0.30)
        pow_ok     = (not prem_rule.get("power", True)) or (is_pow > 0)
        nifty_ok   = nifty_ret6 < prem_rule.get("nifty_trend_max", 0.0)
        if rsi_ok and pow_ok and nifty_ok:
            meta_signal = "BUY"; meta_conf = dir_p; premium = True

    # Stage 3 SELL: meta-classifier
    if meta_signal == "HOLD" and dir_p <= (1 - pre_flt) and meta_sell_m is not None:
        raw = meta_sell_m.predict_proba(Xm_df)[:, 1][0]
        cal = float(meta_sell_cal.predict([raw])[0])
        if cal >= sell_thr:
            meta_signal = "SELL"; meta_conf = cal

    # Stage 3 SELL override: Premium SELL rule (overbought contrarian short)
    # dir_p≤0.25 + RSI>60% + power_hour -> high-conviction reversal short
    prem_sell_rule = blob.get("premium_sell_rule", {})
    if prem_sell_rule and (1 - dir_p) >= (1 - prem_sell_rule.get("dir_max", 0.25)):
        stock_rsi  = float(latest_feats.get("stock_rsi_5m", 0.0))
        is_pow     = float(latest_feats.get("is_power_hour", 0.0))
        rsi_ok     = stock_rsi > prem_sell_rule.get("rsi_min", 0.60)
        pow_ok     = (not prem_sell_rule.get("power", True)) or (is_pow > 0)
        if rsi_ok and pow_ok and prem_sell_rule.get("prec", 0) >= 0.70:
            meta_signal = "SELL"; meta_conf = 1 - dir_p; premium = True

    # Regime filter: suppress signals that fight the Nifty trend
    nifty_day = float(latest_feats.get("nifty_day_ret", 0.0))
    regime_suppressed = False
    if meta_signal == "BUY" and nifty_day < -REGIME_THRESHOLD:
        meta_signal = "HOLD"; meta_conf = 0.0; regime_suppressed = True
    elif meta_signal == "SELL" and nifty_day > REGIME_THRESHOLD:
        meta_signal = "HOLD"; meta_conf = 0.0; regime_suppressed = True

    # Stop loss and target (ATR-based)
    # Premium BUY  : tight stop (1.0* ATR), generous target (1.5* ATR) -> R:R 1:1.5
    # Meta-SELL    : wider stop (1.5* ATR), target (2.0* ATR)           -> R:R 1:1.33
    rr = None
    if entry_price and atr_5m and not np.isnan(atr_5m) and meta_signal != "HOLD":
        if meta_signal == "BUY":
            stop_price   = round(entry_price - 1.0 * atr_5m, 2)
            target_price = round(entry_price + 1.5 * atr_5m, 2)
            rr = 1.5
        else:  # SELL
            stop_price   = round(entry_price + 1.5 * atr_5m, 2)
            target_price = round(entry_price - 2.0 * atr_5m, 2)
            rr = round((entry_price - target_price) / (stop_price - entry_price), 2)

    return {
        "signal":       meta_signal,
        "stage2":       s2,
        "dir_proba":    round(dir_p, 3),
        "hold_proba":   round(hold_p, 3),
        "meta_proba":   round(meta_conf, 3),
        "premium":      premium,
        "entry_price":  entry_price,
        "stop_price":   stop_price,
        "target_price": target_price,
        "atr_5m":       round(atr_5m, 2) if atr_5m and not np.isnan(atr_5m) else None,
        "rr":           rr,
        "nifty_day_ret":     round(nifty_day, 4),
        "regime_suppressed": regime_suppressed,
        "buy_thr":      round(buy_thr, 3),
        "sell_thr":     round(sell_thr, 3),
        "data_ok":      True,
    }


def batch_predict_parallel(symbols: list[str], max_workers: int = 10) -> dict[str, dict]:
    """
    Run predict() for all symbols in parallel using a thread pool.
    Returns {symbol: result_dict}. Never raises -- errors appear as HOLD signals.

    max_workers=10 is a safe default; yfinance handles ~10 concurrent requests
    without rate-limiting. Increase to 15-20 on fast connections.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(predict, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                results[sym] = future.result()
            except Exception as exc:
                results[sym] = {"signal": "HOLD", "error": str(exc),
                                "data_ok": False, "dir_proba": 0.0,
                                "meta_proba": 0.0, "hold_proba": 0.0,
                                "premium": False, "stage2": "HOLD"}
    return results



# ── OOT2: bar-by-bar pipeline replay on holdout slice ─────────────────────────

def _apply_pipeline_batch(
    feats: pd.DataFrame,
    blob: dict,
) -> pd.DataFrame:
    """
    Run the full 3-stage pipeline on every row of feats.
    Returns a DataFrame with columns: [signal, dir_p, hold_p, meta_p, premium].
    """
    feat_cols  = blob.get("feature_cols",     V3_FEATURE_COLS)
    meta_fcols = blob.get("meta_feature_cols", META_FEATURE_COLS)
    model_hold = blob["model_hold"]
    models_dir = blob["models_dir"]
    meta_buy_m = blob.get("meta_buy_model")
    meta_buy_c = blob.get("meta_buy_cal")
    meta_sell_m= blob.get("meta_sell_model")
    meta_sell_c= blob.get("meta_sell_cal")
    buy_thr    = blob.get("meta_buy_thresh",  0.75)
    sell_thr   = blob.get("meta_sell_thresh", 0.75)
    dir_thr    = blob.get("dir_threshold",    DIR_THRESHOLD)
    hold_thr   = blob.get("hold_threshold",   HOLD_THRESHOLD)
    pre_flt    = blob.get("base_pre_filter",  BASE_PRE_FILTER)
    prem_buy   = blob.get("premium_buy_rule", {})
    prem_sell  = blob.get("premium_sell_rule", {})

    X      = feats[feat_cols].fillna(0).values
    hold_p = model_hold.predict_proba(X)[:, 1]
    dir_p  = _dir_proba_ens(models_dir, feats[feat_cols].fillna(0))

    n = len(feats)
    signals = ["HOLD"] * n
    premiums = [False] * n
    meta_p   = np.zeros(n)

    # Meta feature matrix (feat_cols + dir_p + hold_p)
    Xm = np.concatenate([X, dir_p.reshape(-1, 1), hold_p.reshape(-1, 1)], axis=1)
    Xm_df = pd.DataFrame(Xm, columns=meta_fcols, index=feats.index)

    # Pre-compute auxiliary columns for premium rules
    rsi_col  = feats["stock_rsi_5m"].values   if "stock_rsi_5m"   in feats.columns else np.full(n, 0.5)
    pow_col  = feats["is_power_hour"].values  if "is_power_hour"  in feats.columns else np.zeros(n)
    nrd6_col = feats["nifty_ret_6bar"].values if "nifty_ret_6bar" in feats.columns else np.zeros(n)
    nday_col = feats["nifty_day_ret"].values  if "nifty_day_ret"  in feats.columns else np.zeros(n)

    # Stage 1: HOLD filter
    active = hold_p >= hold_thr

    # Stage 2: direction (on active bars only)
    buy_mask  = active & (dir_p >= dir_thr)
    sell_mask = active & ((1 - dir_p) >= dir_thr)

    # Stage 3: meta-BUY
    if meta_buy_m is not None:
        buy_cand = active & (dir_p >= pre_flt)
        if buy_cand.any():
            raw_b = meta_buy_m.predict_proba(Xm_df[buy_cand])[:, 1]
            cal_b = meta_buy_c.predict(raw_b)
            fire  = cal_b >= buy_thr
            idx   = np.where(buy_cand)[0][fire]
            for i, ci in enumerate(idx):
                signals[ci]  = "BUY"
                meta_p[ci]   = cal_b[fire][i]
                premiums[ci] = False

    # Stage 3: meta-SELL
    if meta_sell_m is not None:
        sell_cand = active & (dir_p <= (1 - pre_flt))
        if sell_cand.any():
            raw_s = meta_sell_m.predict_proba(Xm_df[sell_cand])[:, 1]
            cal_s = meta_sell_c.predict(raw_s)
            fire  = cal_s >= sell_thr
            idx   = np.where(sell_cand)[0][fire]
            for i, ci in enumerate(idx):
                if signals[ci] == "HOLD":   # don't overwrite BUY
                    signals[ci]  = "SELL"
                    meta_p[ci]   = cal_s[fire][i]
                    premiums[ci] = False

    # Premium BUY override (only if calibrated prec≥75%; 72.1% rule is excluded)
    if prem_buy and prem_buy.get("prec", 0) >= 0.75:
        pmask = (active & (dir_p >= prem_buy.get("dir_min", 0.80)) &
                 (rsi_col < prem_buy.get("rsi_max", 0.25)))
        if prem_buy.get("power", True):
            pmask &= pow_col > 0
        if prem_buy.get("nifty_trend_max") is not None:
            pmask &= nrd6_col < prem_buy["nifty_trend_max"]
        for ci in np.where(pmask)[0]:
            signals[ci]  = "BUY"
            meta_p[ci]   = dir_p[ci]
            premiums[ci] = True

    # Premium SELL override
    if prem_sell and prem_sell.get("prec", 0) >= 0.70:
        smask = (active & (dir_p <= prem_sell.get("dir_max", 0.25)) &
                 (rsi_col > prem_sell.get("rsi_min", 0.65)))
        if prem_sell.get("power", False):
            smask &= pow_col > 0
        for ci in np.where(smask)[0]:
            if not premiums[ci]:
                signals[ci]  = "SELL"
                meta_p[ci]   = 1 - dir_p[ci]
                premiums[ci] = True

    # Regime filter
    for ci in range(n):
        if signals[ci] == "BUY"  and nday_col[ci] < -REGIME_THRESHOLD:
            signals[ci] = "HOLD"; meta_p[ci] = 0.0; premiums[ci] = False
        elif signals[ci] == "SELL" and nday_col[ci] >  REGIME_THRESHOLD:
            signals[ci] = "HOLD"; meta_p[ci] = 0.0; premiums[ci] = False

    return pd.DataFrame({
        "signal":  signals,
        "dir_p":   dir_p,
        "hold_p":  hold_p,
        "meta_p":  meta_p,
        "premium": premiums,
    }, index=feats.index)


def _cmd_oot2(syms: list[str] | None = None) -> None:
    """
    Bar-by-bar pipeline replay on the holdout slice (last 15%) of each symbol.
    Evaluates actual next-8-bar outcome against the model's fired signals.
    """
    import __main__
    from swing_v2 import LGBMEnsemble
    __main__.LGBMEnsemble = LGBMEnsemble
    blob = joblib.load(Path("models/intraday_v3.pkl"))

    version  = blob.get("version", "?")
    sym_list = syms or blob.get("sym_list", NIFTY_50)

    print("=" * 65)
    print(f"  OOT2 — intraday_model_v3 v{version}  bar-by-bar holdout replay")
    print("=" * 65)
    print(f"  Symbols: {len(sym_list)}  |  Holdout = last 10% (thresholds tuned on val)  |  "
          f"TARGET_BARS={TARGET_BARS}\n")

    all_rows: list[dict] = []
    skipped = 0

    for sym in sym_list:
        try:
            from data_cache import get_bars
            raw = get_bars(sym, days=60)
        except Exception:
            try:
                raw = fetch_5min(sym, days=60)
            except Exception:
                skipped += 1; continue

        if raw is None or len(raw) < 200:
            skipped += 1; continue

        feats = compute_features_v3(raw)
        if feats is None or feats.empty:
            skipped += 1; continue

        # Align raw bars to feats index
        raw_a = raw.loc[feats.index] if len(raw) == len(feats) else raw.iloc[-len(feats):]

        # ATR-relative labels on full data
        labels = make_labels_atr(raw_a)

        # True holdout = last 10% (same split as training)
        n      = len(feats)
        t3     = int(n * (TRAIN_FRAC + META_FRAC + VAL_FRAC))
        hd_f   = feats.iloc[t3:]
        hd_raw = raw_a.iloc[t3:]
        hd_lbl = labels.iloc[t3:]

        if len(hd_f) < TARGET_BARS + 5:
            skipped += 1; continue

        # Run pipeline on holdout bars (exclude last TARGET_BARS — no outcome yet)
        eval_f   = hd_f.iloc[:-TARGET_BARS]
        eval_raw = hd_raw.iloc[:-TARGET_BARS]
        eval_lbl = hd_lbl.iloc[:-TARGET_BARS]
        # Actual future bars for P&L calculation
        future_raw = hd_raw

        sig_df = _apply_pipeline_batch(eval_f, blob)

        # Reset integer position index for aligned slicing
        eval_raw_r = eval_raw.reset_index(drop=True)
        eval_lbl_r = eval_lbl.reset_index(drop=True)
        future_raw_r = future_raw.reset_index(drop=True)

        # Grab DateTime column for display
        dt_col = None
        if "DateTime" in eval_raw.columns:
            dt_col = eval_raw["DateTime"].reset_index(drop=True)
        elif hasattr(eval_raw.index, "dtype") and str(eval_raw.index.dtype).startswith("datetime"):
            dt_col = pd.Series(eval_raw.index, name="DateTime").reset_index(drop=True)

        for pos, (_, row) in enumerate(sig_df.iterrows()):
            sig = row["signal"]
            if sig == "HOLD":
                continue

            actual_lbl = int(eval_lbl_r.iloc[pos])
            entry_c    = float(eval_raw_r["Close"].iloc[pos])

            # Walk forward TARGET_BARS to find actual high/low
            fwd_slice  = future_raw_r.iloc[pos + 1 : pos + 1 + TARGET_BARS]
            if fwd_slice.empty:
                continue
            max_h = float(fwd_slice["High"].max())
            min_l = float(fwd_slice["Low"].min())

            if sig == "BUY":
                predicted_correct = (actual_lbl == 1)
                best_ret = (max_h - entry_c) / entry_c * 100
                worst_ret= (min_l - entry_c) / entry_c * 100
            else:  # SELL
                predicted_correct = (actual_lbl == -1)
                best_ret = (entry_c - min_l) / entry_c * 100
                worst_ret= (entry_c - max_h) / entry_c * 100

            ts = str(dt_col.iloc[pos])[:16] if dt_col is not None else f"bar_{pos}"
            all_rows.append({
                "symbol":   sym,
                "ts":       ts,
                "signal":   sig,
                "premium":  bool(row["premium"]),
                "dir_p":    round(float(row["dir_p"]),  3),
                "meta_p":   round(float(row["meta_p"]), 3),
                "correct":  predicted_correct,
                "best_ret": round(best_ret,  2),
                "worst_ret":round(worst_ret, 2),
                "actual":   {1: "BUY", -1: "SELL", 0: "HOLD"}.get(actual_lbl, "?"),
            })

    if not all_rows:
        print("  No signals found in holdout. Check data availability.")
        return

    df = pd.DataFrame(all_rows)

    # ── Per-signal table ──────────────────────────────────────────────────────
    print(f"  {'Symbol':<13} {'Time':<17} {'Sig':<5} {'Prem':<5} "
          f"{'Dir':>5} {'Meta':>5} {'Actual':<6} {'OK':>3} {'Best%':>7} {'Worst%':>8}")
    print("  " + "-" * 80)
    for _, r in df.iterrows():
        ok   = "✓" if r["correct"] else "✗"
        prem = "⭐" if r["premium"] else ""
        mp_s = f"{r['meta_p']:.0%}" if r["meta_p"] > 0 else "—"
        print(f"  {r['symbol']:<13} {r['ts']:<17} {r['signal']:<5} {prem:<5} "
              f"{r['dir_p']:>4.0%} {mp_s:>5} {r['actual']:<6} {ok:>3} "
              f"{r['best_ret']:>+6.2f}% {r['worst_ret']:>+7.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("── OOT2 Summary ─────────────────────────────────────────────────────")

    def _wilson(k, n, z=1.96):
        if n == 0: return 0.0, 1.0
        p = k / n; d = 1 + z**2 / n
        c = (p + z**2 / (2*n)) / d
        m = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / d
        return max(0.0, c-m), min(1.0, c+m)

    def _report(label: str, sub: pd.DataFrame) -> None:
        if sub.empty:
            print(f"  — {label}: no signals")
            return
        n_ok  = int(sub["correct"].sum())
        n_sig = len(sub)
        prec  = n_ok / n_sig
        avg_b = sub["best_ret"].mean()
        avg_w = sub["worst_ret"].mean()
        lo, hi = _wilson(n_ok, n_sig)
        ok_s  = "✓" if prec >= 0.80 else "✗"
        print(f"  {ok_s} {label:<25} {n_ok:>2}/{n_sig:<2} = {prec:>5.1%}  "
              f"CI[{lo:.0%},{hi:.0%}]  best={avg_b:+.2f}%  worst={avg_w:+.2f}%")

    _report("BUY  (meta-classifier)",  df[(df["signal"] == "BUY")  & ~df["premium"]])
    _report("BUY  (⭐ Premium rule)",   df[(df["signal"] == "BUY")  &  df["premium"]])
    _report("SELL (meta-classifier)",  df[(df["signal"] == "SELL") & ~df["premium"]])
    _report("SELL (⭐ Premium rule)",   df[(df["signal"] == "SELL") &  df["premium"]])

    print()
    # Deduplicated: keep first signal per symbol per session date
    df["date"] = df["ts"].str[:10]
    dedup = df.groupby(["symbol", "signal", "date"]).first().reset_index()
    total_d = len(dedup)
    n_ok_d  = dedup["correct"].sum()
    print(f"  De-duplicated (1st signal/symbol/day):")
    _report("  BUY",  dedup[dedup["signal"] == "BUY"])
    _report("  SELL", dedup[dedup["signal"] == "SELL"])

    total  = len(df)
    n_ok   = df["correct"].sum()
    print(f"\n  All signals : {n_ok}/{total} = {n_ok/total:.1%}  "
          f"({total-n_ok} misses)  |  skipped {skipped} symbols")


# ── Entry point ───────────────────────────────────────────────────────────────
def _cmd_eval(sample_syms: list[str] | None = None) -> None:
    """Load saved model, print stored holdout metrics, run live predictions."""
    import json

    # ── Load blob ────────────────────────────────────────────────────────────
    model_path = Path("models/intraday_v3.pkl")
    if not model_path.exists():
        print("No model found — run: python intraday_model_v3.py train")
        return

    import __main__
    from swing_v2 import LGBMEnsemble
    __main__.LGBMEnsemble = LGBMEnsemble
    blob = joblib.load(model_path)

    version       = blob.get("version", "?")
    n_feats       = len(blob.get("feature_cols", []))
    sym_list      = blob.get("sym_list", [])
    buy_thr       = blob.get("meta_buy_thresh",  0.0)
    sell_thr      = blob.get("meta_sell_thresh", 0.0)
    buy_prec      = blob.get("holdout_buy_prec",  0.0)
    sell_prec     = blob.get("holdout_sell_prec", 0.0)
    prem_buy      = blob.get("premium_buy_rule",  {})
    prem_sell     = blob.get("premium_sell_rule", {})

    print("=" * 60)
    print(f"  intraday_model_v3  v{version}  — saved model report")
    print("=" * 60)
    print(f"  Trained symbols : {len(sym_list)}")
    print(f"  Feature columns : {n_feats}")
    print()
    print("── Holdout OOT precision (never-seen data) ──────────────────")
    buy_ok  = "✓" if buy_prec  >= 0.80 else "✗"
    sell_ok = "✓" if sell_prec >= 0.80 else "✗"
    print(f"  {buy_ok}  BUY  precision: {buy_prec:.1%}  @ meta_thr={buy_thr:.2f}")
    print(f"  {sell_ok}  SELL precision: {sell_prec:.1%}  @ meta_thr={sell_thr:.2f}")
    print()

    if prem_buy:
        print("── Premium BUY rule ──────────────────────────────────────────")
        print(f"  dir >= {prem_buy.get('dir_min', '?')}  |  RSI < {prem_buy.get('rsi_max', '?')}  |  "
              f"power={'Y' if prem_buy.get('power') else 'N'}  |  "
              f"prec={prem_buy.get('prec', 0):.1%}  (N={prem_buy.get('n', 0)})")

    if prem_sell:
        print("── Premium SELL rule ─────────────────────────────────────────")
        print(f"  dir <= {prem_sell.get('dir_max', '?')}  |  RSI > {prem_sell.get('rsi_min', '?')}  |  "
              f"power={'Y' if prem_sell.get('power') else 'N'}  |  "
              f"prec={prem_sell.get('prec', 0):.1%}  (N={prem_sell.get('n', 0)})")

    # ── Live predict sample ──────────────────────────────────────────────────
    syms = sample_syms or sym_list[:10]
    print()
    print(f"── Live predictions (sample: {len(syms)} symbols) ────────────────")
    print(f"  {'Symbol':<14} {'Signal':<6} {'Premium':<8} {'Dir%':>6} {'Meta%':>6}  Entry")

    signals = {"BUY": 0, "SELL": 0, "HOLD": 0, "ERR": 0}
    for sym in syms:
        try:
            r = predict(sym)
            sig  = r.get("signal", "ERR")
            prem = "⭐" if r.get("premium") else ""
            dp   = r.get("dir_proba",  0.0) or 0.0
            mp   = r.get("meta_proba", 0.0) or 0.0
            ep   = r.get("entry_price")
            ep_s = f"₹{ep:,.1f}" if ep else "—"
            signals[sig if sig in signals else "ERR"] += 1
            print(f"  {sym:<14} {sig:<6} {prem:<8} {dp:>5.0%} {mp:>5.0%}  {ep_s}")
        except Exception as exc:
            signals["ERR"] += 1
            print(f"  {sym:<14} ERROR: {exc}")

    print()
    print(f"  Summary: {signals['BUY']} BUY  {signals['SELL']} SELL  "
          f"{signals['HOLD']} HOLD  {signals['ERR']} ERR  (of {len(syms)} sampled)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(0)
    cmd = args[0].lower()
    if cmd == "train":
        test_mode = "--test" in args
        syms = NIFTY_50[:12] if test_mode else NIFTY_50
        train(syms, test_mode=test_mode)
    elif cmd == "eval":
        custom = [s.strip().upper() for s in args[1].split(",") if s.strip()] if len(args) > 1 else None
        _cmd_eval(custom)
    elif cmd == "oot2":
        custom = [s.strip().upper() for s in args[1].split(",") if s.strip()] if len(args) > 1 else None
        _cmd_oot2(custom)
    elif cmd == "predict":
        if len(args) < 2:
            print("Usage: python intraday_model_v3.py predict SYMBOL")
        else:
            import json
            print(json.dumps(predict(args[1]), indent=2))
    else:
        print(f"Unknown command: {cmd}")
