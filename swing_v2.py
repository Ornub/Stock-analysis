"""
swing_v2.py — Institutional-grade NSE swing trading model (v3.1).

Architecture v3.0 upgrades over v2.3:
  1. Market regime filter — Nifty 50/200 DMA + VIX: long entries only in BULL/SIDEWAYS
  2. Cross-sectional ranking — RS20, MOM60, volume expansion, breakout proximity
  3. 7-gate strict entry confirmation — EMA20/50/200, ADX, MACD, breakout, volume
  4. ML meta-filter — configurable probability threshold + min confirmations
  5. Event filter — earnings ±3-day blackout via live news detection
  6. Risk management — ATR trailing stop, max open positions, sector exposure limit
  7. Purged walk-forward CV — embargo between train/test folds prevents leakage
  8. Richer evaluation — yearly P&L, regime-segmented win rates, profit factor
  9. 28 features (4 new: ema20_ratio, ema50_ratio, dist_20d_high, return_60d)
 10. Default watchlist = full Nifty-50

Architecture v3.1 upgrades (Phase 1 — faster indicators):
 11. feat_mfi            — Money Flow Index (price × volume, faster than RSI)
 12. feat_rsi_divergence — RSI vs price divergence (+1 bull / -1 bear)
 13. feat_price_accel    — Price acceleration (ROC of 5d ROC)
 14. feat_candle_pattern — Candle pattern score: engulfing / hammer / shooting-star
 15. feat_obv_slope      — OBV EMA5/EMA20 cross (accumulation momentum)
     Total features: 33

Commands:
    python swing_v2.py train       # retrain pooled XGBoost → models/swing_v2.pkl
    python swing_v2.py predict     # live signals with regime/rank/gate filters
    python swing_v2.py backtest    # backtest with trailing stops + regime breakdown

Default hyperparameters (all configurable at top of file):
    BUY_PROBA = 0.62        MIN_CONFIRMATIONS = 5 / 7
    REGIME_BULL_ONLY = True  TOP_N_CANDIDATES = 20
    ATR_STOP = 1.5×          ATR_TRAIL = 2.0×   ATR_TARGET = 3.0×
    MAX_OPEN_POSITIONS = 5   EMBARGO_SIZE = 10 days

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
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score

from stock_fetcher import fetch_historical_data
from swing_signals import _rsi, _macd, _bollinger, _atr, _obv


# =============================================================================
# Extra indicators
# =============================================================================

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    return 100 * (close - ll) / (hh - ll).replace(0, np.nan)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         vol: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index — combines price movement and volume (faster than RSI alone)."""
    tp  = (high + low + close) / 3
    mf  = tp * vol
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    return 100 * pos / (pos + neg).replace(0, np.nan)


def _rsi_divergence(close: pd.Series, rsi: pd.Series, window: int = 10) -> pd.Series:
    """Detect RSI divergence.
    +1 = bullish (price lower low, RSI higher low — potential reversal up)
    -1 = bearish (price higher high, RSI lower high — potential reversal down)
     0 = no divergence
    """
    def _slope(s: np.ndarray) -> float:
        if len(s) < 2:
            return 0.0
        x = np.arange(len(s), dtype=float)
        return float(np.polyfit(x, s, 1)[0])

    price_slope = close.rolling(window).apply(_slope, raw=True)
    rsi_slope   = rsi.rolling(window).apply(_slope, raw=True)
    # Divergence: slopes have opposite signs
    result = pd.Series(0.0, index=close.index)
    bull = (price_slope < 0) & (rsi_slope > 0)
    bear = (price_slope > 0) & (rsi_slope < 0)
    result[bull] =  1.0
    result[bear] = -1.0
    return result


def _candle_pattern(open_: pd.Series, high: pd.Series,
                    low: pd.Series, close: pd.Series) -> pd.Series:
    """Candle pattern score in [-1, +1].
    +1 = strong bullish pattern (engulfing / hammer)
    -1 = strong bearish pattern (engulfing / shooting star)
     0 = no pattern
    """
    body      = close - open_
    prev_body = body.shift(1)
    hl        = (high - low).replace(0, np.nan)
    lower_wick = (open_.combine(close, min) - low)  / hl
    upper_wick = (high - open_.combine(close, max)) / hl
    body_pct   = body.abs() / hl

    bull_eng = (body > 0) & (prev_body < 0) & (body.abs() > prev_body.abs() * 1.1)
    bear_eng = (body < 0) & (prev_body > 0) & (body.abs() > prev_body.abs() * 1.1)
    hammer   = (body_pct < 0.35) & (lower_wick > 0.5)   # bullish reversal
    shooting = (body_pct < 0.35) & (upper_wick > 0.5)   # bearish reversal

    score = (bull_eng.astype(float)
             + hammer.astype(float) * 0.6
             - bear_eng.astype(float)
             - shooting.astype(float) * 0.6)
    return score.clip(-1.0, 1.0)


def _obv_slope(close: pd.Series, vol: pd.Series,
               fast: int = 5, slow: int = 20) -> pd.Series:
    """OBV momentum: EMAfast / EMAslow - 1. Positive = accumulation accelerating."""
    obv      = _obv(close, vol)
    fast_ema = obv.ewm(span=fast,  adjust=False).mean()
    slow_ema = obv.ewm(span=slow, adjust=False).mean().replace(0, np.nan)
    return fast_ema / slow_ema - 1


# =============================================================================
# Universe & sector map
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

NIFTY_NEXT_50 = [
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM",
    "BAJAJHLDNG", "BANKBARODA", "BERGEPAINT", "BOSCHLTD", "CANBK",
    "CHOLAFIN", "COLPAL", "DABUR", "DLF", "DMART",
    "GAIL", "GODREJCP", "HAL", "HAVELLS", "ICICIGI",
    "ICICIPRULI", "INDIGO", "INDUSTOWER", "IOC", "IRCTC",
    "JINDALSTEL", "LICI", "LODHA", "MARICO", "MOTHERSON",
    "NAUKRI", "NMDC", "PAYTM", "PFC", "PIDILITIND",
    "PNB", "RECLTD", "SBICARD", "SHREECEM", "SIEMENS",
    "SRF", "TATAPOWER", "TORNTPHARM", "TVSMOTOR", "VBL",
    "VEDL", "ZOMATO", "ZYDUSLIFE",
]

TRAIN_UNIVERSE = NIFTY_50 + NIFTY_NEXT_50
WATCHLIST      = NIFTY_50   # default predict universe

SECTOR_MAP: dict[str, str] = {
    "RELIANCE": "Energy",  "ONGC": "Energy",    "BPCL": "Energy",   "COALINDIA": "Energy",
    "NTPC":    "Power",    "POWERGRID": "Power",
    "LT":      "Infra",    "ADANIPORTS": "Infra",
    "HDFCBANK":"Banking",  "ICICIBANK": "Banking", "SBIN": "Banking",
    "AXISBANK":"Banking",  "KOTAKBANK": "Banking", "INDUSINDBK": "Banking",
    "BAJFINANCE":"FinServ","BAJAJFINSV": "FinServ",
    "SBILIFE": "Insurance","HDFCLIFE": "Insurance",
    "TCS":     "IT",       "INFY": "IT",    "WIPRO": "IT",
    "HCLTECH": "IT",       "TECHM": "IT",
    "BHARTIARTL": "Telecom",
    "HINDUNILVR":"FMCG",  "ITC": "FMCG",   "NESTLEIND": "FMCG",  "BRITANNIA": "FMCG",
    "TITAN":   "Consumer", "TRENT": "Consumer",
    "MARUTI":  "Auto",     "HEROMOTOCO": "Auto", "TATAMOTORS": "Auto", "EICHERMOT": "Auto",
    "SUNPHARMA":"Pharma",  "CIPLA": "Pharma",  "DRREDDY": "Pharma",
    "DIVISLAB":"Pharma",   "APOLLOHOSP": "Healthcare",
    "TATASTEEL":"Metal",   "JSWSTEEL": "Metal",  "HINDALCO": "Metal",
    "GRASIM":  "Cement",   "ULTRACEMCO": "Cement",
    "ASIANPAINT":"Paints",
}


# =============================================================================
# Configuration — all hyperparameters in one place
# =============================================================================

# Training
LOOKBACK_YEARS = 5
FORWARD_DAYS   = 5
THRESHOLD      = 0.03   # ±3% label band — tighter than v3.1 to reduce noise
EMBARGO_SIZE   = 10     # trading days purged between train/test in purged CV

# ── Market regime filter ────────────────────────────────────────────────────
# Reasoning: institutional desks only open longs when broad market trend is
# supportive. Nifty 50/200 DMA golden/death cross is the most durable regime
# signal; VIX overlays the fear level.
REGIME_BULL_ONLY   = True    # block new longs in BEAR regime
NIFTY_FAST_DMA     = 50      # EMA period for fast DMA
NIFTY_SLOW_DMA     = 200     # SMA period for slow DMA
VIX_HIGH_THRESHOLD = 20.0    # above: high volatility → tighten gates
VIX_VERY_HIGH      = 25.0    # above: extreme fear → halt new longs

# ── Cross-sectional ranking ─────────────────────────────────────────────────
# Reasoning: only evaluating top-ranked candidates removes low-quality setups
# before the ML model even sees them, improving effective precision.
TOP_N_CANDIDATES = 20   # evaluate only top N by composite rank
RS_PERIOD        = 20   # relative-strength lookback (trading days)
MOMENTUM_PERIOD  = 60   # momentum lookback (trading days)

# ── 7-gate entry confirmation ───────────────────────────────────────────────
# Reasoning: each gate represents an independent evidence dimension. Requiring
# 5/7 ensures multi-dimensional confluence without demanding perfection.
REQUIRE_EMA200   = True   # [HARD GATE] price > EMA200 — mandatory for longs
REQUIRE_EMA50    = True
REQUIRE_EMA20    = True
REQUIRE_ADX      = True
ADX_MIN          = 0.20   # feat_adx = raw_ADX/100 → 0.20 = ADX ≥ 20
REQUIRE_MACD     = True
REQUIRE_BREAKOUT = True   # close > 20-day rolling high (price breakout)
REQUIRE_VOLUME   = True   # volume > N × 20-day average
VOLUME_THRESHOLD = 1.0    # 1.0 = current volume ≥ 100% of 20d avg

MIN_CONFIRMATIONS = 5   # of 7 gates must pass (raised from 4/6 in v2.3)
GRADE_A_MIN = 7         # 7/7 confirmations
GRADE_B_MIN = 6         # 6/7 confirmations
GRADE_C_MIN = 5         # 5/7 confirmations (minimum for BUY)

# ── ML meta-filter ──────────────────────────────────────────────────────────
# Raised from 0.60 to 0.62 for precision. Strong news relaxes to 0.57.
BUY_PROBA             = 0.62
BUY_PROBA_STRONG_NEWS = 0.57
SELL_PROBA            = 0.62

# ── Event / news filters ────────────────────────────────────────────────────
USE_NEWS_GATE          = True
USE_EVENT_FILTER       = True
EARNINGS_BLACKOUT_DAYS = 3   # block entry within N days of detected earnings news

# ── Risk management ─────────────────────────────────────────────────────────
ATR_STOP_MULT      = 1.5   # initial stop: entry - 1.5 × ATR
ATR_TRAIL_MULT     = 2.0   # trailing stop: running_high - 2.0 × ATR (wider = breathes more)
ATR_TARGET_MULT    = 3.0   # target: entry + 3.0 × ATR (1:2 R:R)
MAX_HOLDING_DAYS   = 10    # time stop after N bars
MAX_OPEN_POSITIONS = 5     # max concurrent portfolio positions in backtest
MAX_SECTOR_EXPOSURE = 2    # max concurrent positions in same sector

# Trade-cost model
SLIPPAGE_PCT    = 0.0025   # 0.25% per fill (entry and exit separately)
ROUND_TRIP_COST = 0.0050   # 0.50% combined brokerage + STT + GST

MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "swing_v2.pkl"


# =============================================================================
# Ensemble wrapper — module-level so joblib can pickle it
# =============================================================================

class LGBMEnsemble:
    """Average probability across diverse LightGBM models."""
    def __init__(self, estimators: list):
        self.estimators_ = estimators

    def predict_proba(self, X) -> np.ndarray:
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.estimators_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.mean([m.feature_importances_ for m in self.estimators_], axis=0)


# =============================================================================
# Purged walk-forward CV with embargo
# =============================================================================

class PurgedTimeSeriesSplit:
    """Walk-forward CV that removes an embargo gap from the end of each train fold.

    The embargo prevents rolling-window features (e.g. 20-day return) computed
    near the fold boundary from leaking information about the test period into
    training, a subtle but real source of overfit in time-series CV.

    Default embargo = 10 trading days, covering our longest rolling label
    horizon (FORWARD_DAYS=5) plus a safety buffer.
    """

    def __init__(self, n_splits: int = 5, embargo_size: int = EMBARGO_SIZE):
        self.n_splits     = n_splits
        self.embargo_size = embargo_size
        self._base        = TimeSeriesSplit(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        for tr_idx, te_idx in self._base.split(X):
            purged = tr_idx[:-self.embargo_size] if len(tr_idx) > self.embargo_size else tr_idx
            yield purged, te_idx


# =============================================================================
# Feature engineering — 33 features (28 base + 5 faster indicators v3.1)
# =============================================================================

FEATURE_COLS = [
    # Trend / EMA-relative position
    "feat_rsi",
    "feat_macd_hist",
    "feat_macd_signal_diff",
    "feat_ema_fast_ratio",      # ema9 / ema21 - 1
    "feat_ema_slow_ratio",      # ema21 / ema50 - 1
    "feat_ema20_ratio",         # close / ema20 - 1  [v3.0]
    "feat_ema50_ratio",         # close / ema50 - 1  [v3.0]
    "feat_ema200_ratio",        # close / ema200 - 1
    "feat_adx",
    "feat_stoch_k",
    "feat_williams_r",
    "feat_cci",
    # Volatility / price position
    "feat_bb_pos",
    "feat_atr_pct",
    "feat_dist_52w_high",
    "feat_dist_52w_low",
    "feat_dist_20d_high",       # close / 20d_rolling_max - 1  [v3.0]
    "feat_volatility_20d",
    "feat_high_low_range",
    # Volume
    "feat_obv_ratio",
    "feat_volume_ratio",
    "feat_volume_trend",
    # Candle / gap
    "feat_gap_pct",
    "feat_body_ratio",
    # Multi-period momentum
    "feat_return_5d",
    "feat_return_10d",
    "feat_return_20d",
    "feat_return_60d",          # 60-day momentum  [v3.0]
    # ── Phase 1: faster / leading indicators  [v3.1] ──────────────────
    "feat_mfi",                 # Money Flow Index: price × volume pressure
    "feat_rsi_divergence",      # RSI vs price divergence: +1 bull / -1 bear
    "feat_price_accel",         # ROC acceleration: 5d ROC change over 5 days
    "feat_candle_pattern",      # Engulfing / hammer / shooting-star score
    "feat_obv_slope",           # OBV EMA5/EMA20 cross: accumulation momentum
    # ── Phase 2: market-regime context  [v4.0] ────────────────────────
    "feat_market_return_1d",    # Nifty 1-day return — bull/bear day context
    "feat_market_return_5d",    # Nifty 5-day return — trend strength
    "feat_market_volatility_20d", # Nifty realised vol — fear proxy
    "feat_vix_level",           # India VIX / 100
    "feat_vix_change_5d",       # VIX momentum — rising = fear expanding
    "feat_relative_return_5d",  # stock − Nifty 5d: relative strength
    "feat_nifty_rsi",           # Nifty RSI(14) / 100 — overbought/oversold
    "feat_nifty_trend",         # 50DMA/200DMA − 1: golden/death cross strength
    "feat_nifty_above_200dma",  # +1 bull / −1 bear long-term regime
]

# Auxiliary market-regime columns — joined from the Nifty fetch.
# All are now included in FEATURE_COLS above.
MARKET_AUX_COLS = [
    "feat_market_return_1d",
    "feat_market_return_5d",
    "feat_market_volatility_20d",
    "feat_vix_level",
    "feat_vix_change_5d",
    "feat_relative_return_5d",
    "feat_nifty_rsi",
    "feat_nifty_trend",
    "feat_nifty_above_200dma",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 33 features from OHLCV. Returns df with feat_* columns."""
    df = df.copy()
    close, high, low, vol, op = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

    df["feat_rsi"] = _rsi(close, 14)

    macd_line, sig_line, hist = _macd(close, 12, 26, 9)
    df["feat_macd_hist"]        = hist
    df["feat_macd_signal_diff"] = (macd_line - sig_line) / close

    ema9   = close.ewm(span=9,   adjust=False).mean()
    ema20  = close.ewm(span=20,  adjust=False).mean()   # [v3.0]
    ema21  = close.ewm(span=21,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    df["feat_ema_fast_ratio"] = ema9   / ema21  - 1
    df["feat_ema_slow_ratio"] = ema21  / ema50  - 1
    df["feat_ema20_ratio"]    = close  / ema20  - 1     # [v3.0] gate: price > EMA20
    df["feat_ema50_ratio"]    = close  / ema50  - 1     # [v3.0] gate: price > EMA50
    df["feat_ema200_ratio"]   = close  / ema200 - 1     # long-term trend

    df["feat_adx"]       = _adx(high, low, close, 14) / 100
    df["feat_stoch_k"]   = _stoch_k(high, low, close, 14) / 100
    df["feat_williams_r"]= (_williams_r(high, low, close, 14) + 100) / 100
    df["feat_cci"]       = _cci(high, low, close, 20).clip(-300, 300) / 300

    _, _, _, bb_pos = _bollinger(close, 20, 2)
    df["feat_bb_pos"] = bb_pos

    atr = _atr(high, low, close, 14)
    df["feat_atr_pct"] = atr / close
    df["ATR"] = atr

    high_52w = close.rolling(252, min_periods=20).max()
    low_52w  = close.rolling(252, min_periods=20).min()
    df["feat_dist_52w_high"] = close / high_52w - 1
    df["feat_dist_52w_low"]  = close / low_52w  - 1

    # 20-day breakout proximity: positive = price above previous 20d high (breakout)
    # .shift(1) on the rolling max excludes the current bar → avoids lookahead
    close_20d_max = close.rolling(20, min_periods=10).max().shift(1)
    df["feat_dist_20d_high"] = close / close_20d_max.replace(0, np.nan) - 1   # [v3.0]

    df["feat_volatility_20d"]  = close.pct_change().rolling(20).std()
    df["feat_high_low_range"]  = (high - low) / close

    obv = _obv(close, vol)
    df["feat_obv_ratio"]    = obv / obv.ewm(span=21, adjust=False).mean().replace(0, np.nan)
    df["feat_volume_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)

    vol_ema9  = vol.ewm(span=9,  adjust=False).mean()
    vol_ema21 = vol.ewm(span=21, adjust=False).mean().replace(0, np.nan)
    df["feat_volume_trend"] = vol_ema9 / vol_ema21 - 1

    df["feat_gap_pct"]   = op / close.shift(1) - 1
    df["feat_body_ratio"]= (close - op).abs() / (high - low).replace(0, np.nan)

    df["feat_return_5d"]  = close.pct_change(5)
    df["feat_return_10d"] = close.pct_change(10)
    df["feat_return_20d"] = close.pct_change(20)
    df["feat_return_60d"] = close.pct_change(60)    # [v3.0] 60-day momentum

    # ── Phase 1: faster / leading indicators  [v3.1] ──────────────────
    rsi_series = df["feat_rsi"]
    df["feat_mfi"]           = _mfi(high, low, close, vol, 14) / 100
    df["feat_rsi_divergence"]= _rsi_divergence(close, rsi_series, window=10)
    roc_5                    = close.pct_change(5)
    df["feat_price_accel"]   = (roc_5 - roc_5.shift(5)).clip(-0.15, 0.15)
    df["feat_candle_pattern"]= _candle_pattern(op, high, low, close)
    df["feat_obv_slope"]     = _obv_slope(close, vol, fast=5, slow=20).clip(-0.5, 0.5)

    return df


# =============================================================================
# Market regime — fetch + classify
# =============================================================================

_MARKET_CACHE: pd.DataFrame | None = None


def _fetch_market_regime(years: int) -> pd.DataFrame:
    """Fetch ^NSEI + ^INDIAVIX, compute regime features including 50/200 DMA."""
    global _MARKET_CACHE
    if _MARKET_CACHE is not None:
        return _MARKET_CACHE

    import yfinance as yf
    today     = date.today()
    from_date = today - timedelta(days=years * 365 + 60)

    nifty = yf.download("^NSEI",     start=from_date, end=today,
                        auto_adjust=True, progress=False)
    vix   = yf.download("^INDIAVIX", start=from_date, end=today,
                        auto_adjust=True, progress=False)

    if nifty is None or len(nifty) == 0:
        print("[WARN] Could not fetch ^NSEI; market regime features will be zero.")
        return pd.DataFrame()

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if vix is not None and isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    df = pd.DataFrame(index=nifty.index)
    df["nifty_close"]                = nifty["Close"]
    # DMAs for regime classification
    df["nifty_50dma"]  = nifty["Close"].ewm(span=NIFTY_FAST_DMA, adjust=False).mean()
    df["nifty_200dma"] = nifty["Close"].rolling(NIFTY_SLOW_DMA, min_periods=100).mean()

    df["feat_market_return_1d"]      = nifty["Close"].pct_change(1)
    df["feat_market_return_5d"]      = nifty["Close"].pct_change(5)
    df["feat_market_volatility_20d"] = nifty["Close"].pct_change().rolling(20).std()

    if vix is not None and len(vix):
        v = vix["Close"].reindex(df.index, method="ffill")
        df["feat_vix_level"]     = v / 100
        df["feat_vix_change_5d"] = v.pct_change(5).fillna(0)
    else:
        df["feat_vix_level"]     = 0.15
        df["feat_vix_change_5d"] = 0.0

    nc = nifty["Close"].reindex(df.index)
    df["feat_nifty_rsi"]          = _rsi(nc, 14).fillna(50) / 100
    df["feat_nifty_trend"]        = (df["nifty_50dma"] / df["nifty_200dma"].replace(0, np.nan) - 1).fillna(0)
    df["feat_nifty_above_200dma"] = ((nc > df["nifty_200dma"]).astype(float) * 2 - 1).fillna(0)

    df = df.reset_index().rename(columns={"Date": "DateTime"})
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None).dt.normalize()
    _MARKET_CACHE = df
    return df


def classify_regime(market_df: pd.DataFrame) -> tuple[str, str]:
    """Classify the latest market observation into trend and volatility regimes.

    Trend regimes:
        BULL     — Nifty > 50DMA > 200DMA (golden cross, all aligned)
        SIDEWAYS — Mixed alignment; no clear direction
        BEAR     — Nifty < 200DMA or 50DMA < 200DMA (death cross)

    Vol regimes (raw VIX points):
        LOW      — VIX < 12
        NORMAL   — 12 ≤ VIX < 20
        HIGH     — 20 ≤ VIX < 25
        EXTREME  — VIX ≥ 25

    Default hyperparameters: VIX_HIGH_THRESHOLD=20, VIX_VERY_HIGH=25
    """
    if market_df is None or market_df.empty:
        return "SIDEWAYS", "NORMAL"

    latest   = market_df.sort_values("DateTime").iloc[-1]
    nifty    = float(latest.get("nifty_close", 0) or 0)
    fast_dma = float(latest.get("nifty_50dma",  nifty) or nifty)
    slow_dma = float(latest.get("nifty_200dma", nifty) or nifty)
    vix      = float(latest.get("feat_vix_level", 0.15) or 0.15) * 100

    if nifty > fast_dma and fast_dma > slow_dma:
        trend = "BULL"
    elif nifty < slow_dma or fast_dma < slow_dma:
        trend = "BEAR"
    else:
        trend = "SIDEWAYS"

    if vix >= VIX_VERY_HIGH:
        vol = "EXTREME"
    elif vix >= VIX_HIGH_THRESHOLD:
        vol = "HIGH"
    elif vix < 12.0:
        vol = "LOW"
    else:
        vol = "NORMAL"

    return trend, vol


def _attach_market_features(panel: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if market is None or market.empty:
        for c in MARKET_AUX_COLS:
            panel[c] = 0.0
        return panel

    panel = panel.copy()
    panel["_date_key"] = pd.to_datetime(panel["DateTime"]).dt.tz_localize(None).dt.normalize()
    direct_cols = [c for c in MARKET_AUX_COLS if c != "feat_relative_return_5d"
                   and c in market.columns]
    join_cols   = ["DateTime"] + direct_cols
    market_join = market[join_cols].rename(columns={"DateTime": "_date_key"})
    panel = panel.merge(market_join, on="_date_key", how="left").drop(columns=["_date_key"])
    panel["feat_relative_return_5d"] = (
        panel["feat_return_5d"].fillna(0) - panel["feat_market_return_5d"].fillna(0)
    )
    return panel


# =============================================================================
# Cross-sectional ranking
# =============================================================================

def rank_candidates(
    symbol_data: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    top_n: int = TOP_N_CANDIDATES,
) -> list[str]:
    """Score every stock on 4 dimensions, return top N symbols.

    Dimensions (equal-weighted percentile rank, higher = better):
        1. 20-day relative strength vs Nifty  (recent outperformance)
        2. 60-day price momentum              (trend persistence)
        3. 5-day avg volume / 20-day avg vol  (volume expansion → institutional interest)
        4. Close proximity to 20-day high     (closeness to breakout level)

    Rationale: institutional systematic strategies select from a pre-filtered
    opportunity set rather than scanning the full universe through the full
    model pipeline — this removes statistically weak setups before ML scoring.
    """
    nifty_ret_rs  = 0.0
    nifty_ret_mom = 0.0
    if market_df is not None and not market_df.empty:
        ns = market_df.sort_values("DateTime").set_index("DateTime")["nifty_close"]
        if len(ns) >= RS_PERIOD:
            nifty_ret_rs  = float(ns.pct_change(RS_PERIOD).iloc[-1]  or 0)
        if len(ns) >= MOMENTUM_PERIOD:
            nifty_ret_mom = float(ns.pct_change(MOMENTUM_PERIOD).iloc[-1] or 0)

    scores = []
    for sym, df in symbol_data.items():
        if df is None or len(df) < MOMENTUM_PERIOD + 5:
            continue
        c, h, v = df["Close"], df["High"], df["Volume"]

        rs_20d   = float(c.pct_change(RS_PERIOD).iloc[-1]  or 0) - nifty_ret_rs
        mom_60d  = float(c.pct_change(MOMENTUM_PERIOD).iloc[-1] or 0) - nifty_ret_mom

        vol_avg  = float(v.rolling(20).mean().iloc[-1] or 1)
        vol_exp  = float(v.tail(5).mean() or 0) / vol_avg

        # proximity to 20d high: higher ratio = closer to/above breakout level
        h20 = float(h.iloc[-21:-1].max() or c.iloc[-1])
        breakout_prox = float(c.iloc[-1]) / h20 if h20 > 0 else 1.0

        scores.append({
            "symbol":        sym,
            "rs_20d":        rs_20d,
            "mom_60d":       mom_60d,
            "vol_expansion": vol_exp,
            "breakout_prox": breakout_prox,
        })

    if not scores:
        return list(symbol_data.keys())

    df_s = pd.DataFrame(scores)
    df_s["rank_rs"]       = df_s["rs_20d"].rank()
    df_s["rank_mom"]      = df_s["mom_60d"].rank()
    df_s["rank_vol"]      = df_s["vol_expansion"].rank()
    df_s["rank_breakout"] = df_s["breakout_prox"].rank()
    df_s["composite"]     = (df_s["rank_rs"] + df_s["rank_mom"] +
                             df_s["rank_vol"] + df_s["rank_breakout"])

    return df_s.nlargest(min(top_n, len(df_s)), "composite")["symbol"].tolist()


# =============================================================================
# Dataset assembly
# =============================================================================

def build_panel(symbols: list[str], years: int = LOOKBACK_YEARS) -> pd.DataFrame | None:
    today     = date.today()
    from_date = today - timedelta(days=years * 365)

    market = _fetch_market_regime(years)
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
    panel = pd.concat(panels, ignore_index=True)
    return _attach_market_features(panel, market)


def make_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward 5-day return ≥+2% → BUY (1), ≤-2% → SELL (0), else drop neutral."""
    panel = panel.sort_values(["symbol", "DateTime"]).copy()
    panel["fwd_return"] = (
        panel.groupby("symbol")["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    )
    panel = panel.dropna(subset=FEATURE_COLS + ["fwd_return"])
    panel = panel[panel["fwd_return"].abs() >= THRESHOLD].copy()
    panel["label"] = (panel["fwd_return"] >= 0).astype(int)
    return panel


# =============================================================================
# Train — purged walk-forward CV
# =============================================================================

def train(symbols: list[str] | None = None, years: int = LOOKBACK_YEARS):
    symbols = symbols or NIFTY_50
    print(f"\n=== TRAINING v3 model on {len(symbols)} stocks × {years}y ===")
    print(f"    Purged CV embargo: {EMBARGO_SIZE} days | Features: {len(FEATURE_COLS)}\n")

    panel = build_panel(symbols, years)
    if panel is None:
        print("No data — aborting.")
        return None

    total_rows = len(panel)
    print(f"\nTotal panel rows: {total_rows:,}")
    panel = make_labels(panel)
    n_buy  = int(panel["label"].sum())
    n_sell = len(panel) - n_buy
    print(f"After label filter: {len(panel):,} samples  BUY={n_buy:,}  SELL={n_sell:,}")

    panel = panel.sort_values("DateTime").reset_index(drop=True)
    X = panel[FEATURE_COLS].values
    y = panel["label"].values

    # Class-balance weight
    spw = n_sell / max(n_buy, 1)
    print(f"\nClass balance weight (SELL/BUY): {spw:.3f}")

    # ── CV: fixed-depth LightGBM — no early stopping so small early folds stay stable ──
    def _cv_lgbm(seed: int) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03,
            max_depth=5, num_leaves=31, min_child_samples=40,
            feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=5,
            lambda_l1=0.2, lambda_l2=1.0,
            scale_pos_weight=spw,
            objective="binary", metric="binary_logloss",
            verbose=-1, n_jobs=-1, random_state=seed,
        )

    print(f"\nWalk-forward CV (PurgedTimeSeriesSplit, 5 folds, embargo=10d):")
    ptscv = PurgedTimeSeriesSplit(n_splits=5, embargo_size=EMBARGO_SIZE)
    fold_accs, fold_precs, fold_recs, fold_srecs = [], [], [], []

    for i, (tr, te) in enumerate(ptscv.split(X), 1):
        m = _cv_lgbm(42)
        m.fit(X[tr], y[tr])
        y_pred  = m.predict(X[te])
        acc     = accuracy_score(y[te], y_pred)
        prec    = precision_score(y[te], y_pred, pos_label=1, zero_division=0)
        rec     = recall_score(y[te], y_pred, pos_label=1, zero_division=0)
        sell_rc = recall_score(y[te], y_pred, pos_label=0, zero_division=0)
        fold_accs.append(acc); fold_precs.append(prec)
        fold_recs.append(rec); fold_srecs.append(sell_rc)
        print(f"  Fold {i}: train={len(tr):>5,}  test={len(te):>5,} | "
              f"acc={acc:.3f}  BUY-prec={prec:.3f}  "
              f"BUY-rec={rec:.3f}  SELL-rec={sell_rc:.3f}")

    mean_acc  = float(np.mean(fold_accs))
    std_acc   = float(np.std(fold_accs))
    mean_prec = float(np.mean(fold_precs))
    mean_rec  = float(np.mean(fold_recs))
    mean_srec = float(np.mean(fold_srecs))
    print(f"\nMean walk-forward accuracy : {mean_acc:.3f}  (±{std_acc:.3f})")
    print(f"Mean BUY precision         : {mean_prec:.3f}")
    print(f"Mean BUY recall            : {mean_rec:.3f}")
    print(f"Mean SELL recall           : {mean_srec:.3f}")

    sigma    = max(std_acc, 1e-6)
    unstable = [(i+1, a) for i, a in enumerate(fold_accs) if abs(a - mean_acc) > 1.5*sigma]
    if unstable:
        print(f"⚠  Unstable folds (>1.5σ): "
              + ", ".join(f"#{i} ({a:.3f})" for i, a in unstable))
    if mean_acc < 0.52:
        print("⚠  Accuracy near coin-flip — consider retraining with more data or "
              "adjusting FORWARD_DAYS / THRESHOLD.")

    # --- OOT split: hold out last 30 trading days ---
    all_dates    = pd.to_datetime(panel["DateTime"]).dt.date.values
    unique_dates = np.sort(np.unique(all_dates))
    oot_cutoff   = unique_dates[-30] if len(unique_dates) >= 30 else unique_dates[0]
    oot_mask     = all_dates >= oot_cutoff

    X_train_final = X[~oot_mask]
    y_train_final = y[~oot_mask]
    X_oot         = X[oot_mask]
    y_oot         = y[oot_mask]

    print(f"\nOOT holdout : {oot_mask.sum():,} samples  "
          f"[{unique_dates[-30]} → {unique_dates[-1]}]")
    print(f"Final train : {(~oot_mask).sum():,} samples  "
          f"[{unique_dates[0]} → {unique_dates[-31]}]")

    # ── 3-model LightGBM ensemble, fixed 500 trees ──
    # Early stopping is skipped: the last 15% of training can sit in a different
    # market regime and cause spurious early abort (iter=2-9). Fixed depth is
    # more stable; 500 trees at lr=0.01 is well-regularised for this data size.
    _ens_cfgs  = [(42, 0.70, 0.80), (43, 0.65, 0.75), (44, 0.75, 0.85)]
    models_ens = []
    for seed, col_frac, row_frac in _ens_cfgs:
        m = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.01,
            max_depth=5, num_leaves=31, min_child_samples=40,
            feature_fraction=col_frac, bagging_fraction=row_frac, bagging_freq=5,
            lambda_l1=0.2, lambda_l2=1.0,
            scale_pos_weight=spw,
            objective="binary", metric="binary_logloss",
            verbose=-1, n_jobs=-1, random_state=seed,
        )
        m.fit(X_train_final, y_train_final)
        print(f"  Ensemble seed={seed}: done")
        models_ens.append(m)

    final = LGBMEnsemble(models_ens)

    # Honest OOT evaluation
    oot_acc = oot_prec = oot_rec = float("nan")
    if len(X_oot) > 0:
        y_oot_pred = final.predict(X_oot)
        y_oot_prob = final.predict_proba(X_oot)[:, 1]
        oot_acc  = accuracy_score(y_oot, y_oot_pred)
        oot_prec  = precision_score(y_oot, y_oot_pred, pos_label=1, zero_division=0)
        oot_rec   = recall_score(y_oot, y_oot_pred, pos_label=1, zero_division=0)
        oot_srec  = recall_score(y_oot, y_oot_pred, pos_label=0, zero_division=0)
        print(f"\nOOT test (last 30 trading days, n={len(X_oot):,}):")
        print(f"  Accuracy      : {oot_acc:.3f}")
        print(f"  BUY precision : {oot_prec:.3f}")
        print(f"  BUY recall    : {oot_rec:.3f}")
        print(f"  SELL recall   : {oot_srec:.3f}")
        print("\nOOT probability calibration:")
        for p in (0.50, 0.55, 0.60, 0.62, 0.65, 0.70):
            pmask = y_oot_prob >= p
            n     = int(pmask.sum())
            hit   = float(y_oot[pmask].mean()) if n > 0 else 0.0
            print(f"  prob ≥ {p:.2f}: {n:>5,} samples  empirical BUY rate = {hit*100:5.1f}%")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model":      final,
        "features":   FEATURE_COLS,
        "val_acc":    mean_acc,
        "val_std":    std_acc,
        "val_prec":   mean_prec,
        "val_rec":    mean_rec,
        "val_srec":   mean_srec,
        "oot_acc":    oot_acc,
        "oot_prec":   oot_prec,
        "oot_rec":    oot_rec,
        "n_train":    int((~oot_mask).sum()),
        "trained_on": symbols,
        "trained_at": str(date.today()),
    }, MODEL_PATH)
    print(f"\n✓ Saved to {MODEL_PATH}")

    print("\n=== TRAINING SUMMARY ===")
    print(f"Total panel rows  : {total_rows:,}")
    print(f"Labelled samples  : {len(panel):,}")
    print(f"Class balance     : BUY {n_buy/len(panel)*100:.1f}%  SELL {n_sell/len(panel)*100:.1f}%")
    print(f"scale_pos_weight  : {spw:.3f}")
    print(f"Features          : {len(FEATURE_COLS)}")
    print(f"Final model       : 3-model LightGBM ensemble")
    print(f"CV accuracy       : {mean_acc:.3f}  (±{std_acc:.3f})")
    print(f"CV SELL recall    : {mean_srec:.3f}")
    print(f"OOT accuracy      : {oot_acc:.3f}")

    imp = pd.Series(final.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(imp.head(10).round(3).to_string())

    return final


# =============================================================================
# Predict — regime → rank → 7-gate → ML meta-filter
# =============================================================================

def _add_intraday_overlay(df_out: pd.DataFrame) -> None:
    """Enrich df_out in-place with intraday features.

    Tries Zerodha first (if configured); falls back to free yfinance data.
    Adds: vwap_ratio, orb_signal, intraday_vol_surge, depth_score, intraday_score.
    """
    def _apply(feats: dict) -> None:
        df_out["vwap_ratio"]         = df_out["Symbol"].map(lambda s: feats.get(s, {}).get("vwap_ratio",         1.0))
        df_out["orb_signal"]         = df_out["Symbol"].map(lambda s: feats.get(s, {}).get("orb_signal",         0))
        df_out["intraday_vol_surge"] = df_out["Symbol"].map(lambda s: feats.get(s, {}).get("intraday_vol_surge", 1.0))
        df_out["depth_score"]        = df_out["Symbol"].map(lambda s: feats.get(s, {}).get("depth_score",        float("nan")))
        def _score(row):
            v = 1.0 if row["vwap_ratio"] > 1.005 else (-1.0 if row["vwap_ratio"] < 0.995 else 0.0)
            o = float(row["orb_signal"])
            s = 1.0 if row["intraday_vol_surge"] >= 1.5 else (0.0 if row["intraday_vol_surge"] >= 1.0 else -1.0)
            return round((v + o + s) / 3, 2)
        df_out["intraday_score"] = df_out.apply(_score, axis=1)

    # Try Zerodha first
    try:
        from zerodha_data.config import api_key
        if api_key != "YOUR_API_KEY":
            from zerodha_data import get_kite, get_batch_intraday_features, TokenExpiredError
            kite    = get_kite()
            symbols = df_out["Symbol"].tolist()
            print("[Zerodha] Fetching intraday features…")
            feats   = get_batch_intraday_features(symbols, kite=kite)
            _apply(feats)
            print("[Zerodha] Intraday overlay applied.")
            return
    except Exception:
        pass

    # Free fallback: yfinance (always works, ~15-min delayed)
    try:
        from intraday import get_batch_intraday_features as _yf_batch
        symbols = df_out["Symbol"].tolist()
        print("[yfinance] Fetching intraday features…")
        feats   = _yf_batch(symbols)
        _apply(feats)
        print("[yfinance] Intraday overlay applied.")
    except Exception:
        pass  # never break the main predict() if intraday is unavailable


def predict(symbols: list[str] | None = None):
    if not MODEL_PATH.exists():
        print('No saved model. Run "python swing_v2.py train" first.')
        return

    blob    = joblib.load(MODEL_PATH)
    model   = blob["model"]
    val_acc = blob["val_acc"]
    model_features = blob.get("features", FEATURE_COLS)   # backward-compat

    symbols   = symbols or WATCHLIST
    today     = date.today()
    from_date = today - timedelta(days=420)   # 420d for 252d 52-week + 60d momentum

    print(f"\n=== v3 SIGNALS — {today} ===")
    print(f"Model val accuracy: {val_acc*100:.1f}% on {blob['n_train']:,} samples")

    # ── Step 1: Market regime ────────────────────────────────────────────────
    market       = _fetch_market_regime(LOOKBACK_YEARS)
    trend_regime, vol_regime = classify_regime(market)

    latest_market = {}
    if market is not None and not market.empty:
        row = market.sort_values("DateTime").iloc[-1]
        latest_market = row.to_dict()

    nifty_close  = latest_market.get("nifty_close", 0)
    nifty_50dma  = latest_market.get("nifty_50dma",  0)
    nifty_200dma = latest_market.get("nifty_200dma", 0)
    vix_raw      = latest_market.get("feat_vix_level", 0.15) * 100

    print(f"\nMarket regime : {trend_regime}  |  Vol regime: {vol_regime}")
    print(f"Nifty: {nifty_close:>8.0f}  |  50DMA: {nifty_50dma:>8.0f}  "
          f"|  200DMA: {nifty_200dma:>8.0f}  |  VIX: {vix_raw:.1f}")

    regime_blocked = REGIME_BULL_ONLY and trend_regime == "BEAR"
    vix_extreme    = vol_regime == "EXTREME"

    if regime_blocked:
        print("⚠  BEAR regime — long entries blocked. Showing scores for monitoring only.")
    if vol_regime in ("HIGH", "EXTREME"):
        print(f"⚠  VIX {vol_regime} ({vix_raw:.1f}) — reduce position sizes or skip C-grade signals.")

    # ── Step 2: Fetch all stock data ─────────────────────────────────────────
    print()
    symbol_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_historical_data(sym, from_date, today)
        if df is None or len(df) < 252:
            continue
        df = compute_features(df)
        # Attach market aux features for relative-strength display
        if market is not None and not market.empty:
            mj = market[["DateTime", "feat_market_return_5d"]].rename(
                columns={"DateTime": "_dk"})
            df["_dk"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None).dt.normalize()
            df = df.merge(mj, on="_dk", how="left").drop(columns=["_dk"])
            df["feat_relative_return_5d"] = (
                df.get("feat_return_5d", pd.Series(0, index=df.index)).fillna(0) -
                df.get("feat_market_return_5d", pd.Series(0, index=df.index)).fillna(0)
            )
        df = df.dropna(subset=[c for c in model_features if c in df.columns])
        if not df.empty:
            symbol_data[sym] = df

    # ── Step 3: Cross-sectional ranking ──────────────────────────────────────
    ranked   = rank_candidates(symbol_data, market, top_n=TOP_N_CANDIDATES)
    ranked_set = set(ranked)
    print(f"Top {len(ranked)} ranked: {', '.join(ranked[:8])}{'...' if len(ranked) > 8 else ''}\n")

    # ── Step 4: Evaluate each stock ──────────────────────────────────────────
    rows = []
    for sym, df in symbol_data.items():
        latest = df.iloc[-1]

        # Safe feature extraction (handle old model missing new features)
        feat_vals = []
        for c in model_features:
            feat_vals.append(float(latest.get(c, 0.0) or 0.0))
        proba    = model.predict_proba(np.array(feat_vals).reshape(1, -1))[0]
        buy_p    = float(proba[1])
        sell_p   = float(proba[0])

        # News + event filter
        news_feats     = {"news_score": 0.0, "news_count": 0, "news_event_type": "none",
                         "recency_days": 99}
        news_blocked   = False
        strong_news_up = False
        ngrade         = "B"
        event_blocked  = False
        event_reason   = ""

        if USE_NEWS_GATE:
            try:
                from news_features import (get_news_features, news_gate_passes,
                                           news_strong_positive, news_grade)
                news_feats     = get_news_features(sym)
                news_blocked   = not news_gate_passes(news_feats)
                strong_news_up = news_strong_positive(news_feats)
                ngrade         = news_grade(news_feats)

                if USE_EVENT_FILTER:
                    etype   = news_feats.get("news_event_type", "none")
                    recency = int(news_feats.get("recency_days", 99))
                    if etype == "earnings" and recency <= EARNINGS_BLACKOUT_DAYS:
                        event_blocked = True
                        event_reason  = f"earnings({recency}d)"
                    elif etype in ("downgrade", "regulatory") and recency <= 1:
                        event_blocked = True
                        event_reason  = etype
            except Exception:
                pass

        # 7 confirmation gates
        def gv(col, fallback=0.0):
            return float(latest.get(col, fallback) or fallback)

        confirmations = {
            "ema200":   (not REQUIRE_EMA200)   or gv("feat_ema200_ratio") > 0,
            "ema50":    (not REQUIRE_EMA50)    or gv("feat_ema50_ratio")  > 0,
            "ema20":    (not REQUIRE_EMA20)    or gv("feat_ema20_ratio")  > 0,
            "adx":      (not REQUIRE_ADX)      or gv("feat_adx")          >= ADX_MIN,
            "macd":     (not REQUIRE_MACD)     or gv("feat_macd_hist")    >= 0,
            "breakout": (not REQUIRE_BREAKOUT) or gv("feat_dist_20d_high") > 0,
            "volume":   (not REQUIRE_VOLUME)   or gv("feat_volume_ratio") >= VOLUME_THRESHOLD,
        }
        n_pass = sum(confirmations.values())
        failed = [k for k, v in confirmations.items() if not v]

        if n_pass >= GRADE_A_MIN:
            grade = "A"
        elif n_pass >= GRADE_B_MIN:
            grade = "B"
        elif n_pass >= GRADE_C_MIN:
            grade = "C"
        else:
            grade = "D"

        bar = BUY_PROBA_STRONG_NEWS if strong_news_up else BUY_PROBA

        # Hard gates (must ALL pass for BUY)
        ema200_ok  = confirmations["ema200"]
        regime_ok  = not regime_blocked and not vix_extreme
        rank_ok    = sym in ranked_set

        gate_pass = (
            ema200_ok and regime_ok and rank_ok
            and not news_blocked and not event_blocked
            and n_pass >= MIN_CONFIRMATIONS
        )

        if buy_p >= bar and gate_pass:
            signal = "BUY"
        elif buy_p >= bar:
            signal = "WATCH"
        elif sell_p >= SELL_PROBA:
            signal = "SELL"
        else:
            signal = "HOLD"

        blockers = []
        if signal == "WATCH":
            if not regime_ok:   blockers.append("bear_regime")
            if not ema200_ok:   blockers.append("ema200")
            if not rank_ok:     blockers.append("low_rank")
            if news_blocked:    blockers.append("bad_news")
            if event_blocked:   blockers.append(event_reason)
            if n_pass < MIN_CONFIRMATIONS:
                blockers += [f for f in failed if f not in ("ema200",)]

        atr, price = float(latest.get("ATR", 0) or 0), float(latest["Close"])
        if signal == "BUY":
            stop = round(price - ATR_STOP_MULT  * atr, 2)
            tgt  = round(price + ATR_TARGET_MULT * atr, 2)
        elif signal == "SELL":
            stop = round(price + ATR_STOP_MULT  * atr, 2)
            tgt  = round(price - ATR_TARGET_MULT * atr, 2)
        else:
            stop = tgt = None

        # Composite rank position (1-based)
        rank_pos = (ranked.index(sym) + 1) if sym in ranked_set else len(symbol_data)

        rows.append({
            "Symbol":   sym,
            "Price":    round(price, 2),
            "Signal":   signal,
            "Grade":    grade if signal == "BUY" else "",
            "BUY%":     round(buy_p * 100, 1),
            "Confs":    f"{n_pass}/7",
            "Rank":     rank_pos,
            "News":     ngrade if news_feats.get("news_count", 0) > 0 else "—",
            "Event":    news_feats.get("news_event_type", "none"),
            "Stop":     stop,
            "Target":   tgt,
            "Blockers": ",".join(blockers) if signal == "WATCH" else "",
        })

    df_out = pd.DataFrame(rows)

    # ── Phase 2: Zerodha intraday overlay (optional) ─────────────────────────
    # Adds vwap_ratio, orb_signal, intraday_vol_surge, depth_score columns.
    # Silently skipped if Zerodha credentials are not configured.
    _add_intraday_overlay(df_out)

    # Sort: BUY (grade A→C) first, then WATCH by BUY%, then HOLD, SELL last
    signal_order = {"BUY": 0, "WATCH": 1, "HOLD": 2, "SELL": 3}
    grade_order  = {"A": 0, "B": 1, "C": 2, "D": 3, "": 4}
    df_out["_so"] = df_out["Signal"].map(signal_order).fillna(4)
    df_out["_go"] = df_out["Grade"].map(grade_order).fillna(4)
    df_out = (df_out.sort_values(["_so", "_go", "BUY%"], ascending=[True, True, False])
              .drop(columns=["_so", "_go"]).reset_index(drop=True))

    print(df_out.to_string(index=False))

    buys = df_out[df_out["Signal"] == "BUY"]
    if not buys.empty:
        print("\nBUY RATIONALE:")
        for _, r in buys.iterrows():
            parts = [f"EMA200 uptrend", f"{r['Confs']} confirmations"]
            if r["News"] not in ("—", ""):
                parts.append(f"news {r['News']} ({r['Event']})")
            if not rank_ok:  # shouldn't happen if BUY, included for safety
                pass
            print(f"  • {r['Symbol']:<12} (Grade {r['Grade']}, BUY% {r['BUY%']}): "
                  + " + ".join(parts))

    watches = df_out[df_out["Signal"] == "WATCH"]
    if not watches.empty:
        print("\nWATCH (blocked — monitor for gate clearance):")
        for _, r in watches.iterrows():
            print(f"  • {r['Symbol']:<12} BUY% {r['BUY%']}  blocked by: {r['Blockers']}")

    print(f"\nLegend: Confs = gates passed (7 total) | Grade A=7/7, B=6/7, C=5/7")
    print(f"        Regime={trend_regime} | VIX={vol_regime} ({vix_raw:.1f})")
    print("⚠  Educational only — not financial advice.")
    return df_out


# =============================================================================
# Backtest — ATR trailing stop + sector/position limits + regime breakdown
# =============================================================================

def backtest(
    symbols: list[str] | None = None,
    years: int = 2,
    buy_threshold: float = BUY_PROBA,
    max_holding_days: int = MAX_HOLDING_DAYS,
):
    """Simulate trades on the saved model with institutional-grade exit logic.

    Exit hierarchy (checked each bar):
        1. Trailing stop hit (bar_low ≤ trailing_stop)       → stop exit
        2. Target hit        (bar_high ≥ target)             → target exit
        3. Time stop         (days held ≥ max_holding_days)  → timeout exit

    Trailing stop starts at entry - ATR_STOP_MULT×ATR and ratchets up as the
    trade moves in our favour (never moves down), giving winners room to run
    while protecting accumulated gains.

    Global position limits: MAX_OPEN_POSITIONS across all stocks simultaneously;
    MAX_SECTOR_EXPOSURE per sector.  Regime is computed per-day using live market
    data and recorded alongside each trade for regime-segmented analysis.
    """
    if not MODEL_PATH.exists():
        print('No saved model. Run "python swing_v2.py train" first.')
        return

    blob  = joblib.load(MODEL_PATH)
    model = blob["model"]
    model_features = blob.get("features", FEATURE_COLS)

    symbols = symbols or NIFTY_50[:20]
    print(f"\n=== BACKTEST v3 — {len(symbols)} stocks × {years}y ===")
    print(f"    Threshold={buy_threshold:.2f}  Gates: EMA200+ADX+MACD+EMA50+Volume  "
          f"Trail={ATR_TRAIL_MULT}×ATR  MaxPos={MAX_OPEN_POSITIONS}\n")

    panel = build_panel(symbols, years)
    if panel is None:
        return

    # Fetch market data for regime annotation per trade
    market = _fetch_market_regime(years)
    market_idx = None
    if market is not None and not market.empty:
        market_idx = market.set_index("DateTime")[["nifty_close", "nifty_50dma",
                                                    "nifty_200dma", "feat_vix_level"]]

    panel = panel.sort_values(["symbol", "DateTime"]).reset_index(drop=True)
    panel = panel.dropna(subset=[c for c in model_features if c in panel.columns]).copy()

    # Score every row with the model
    feat_matrix = np.column_stack([
        panel.get(c, pd.Series(0, index=panel.index)).values for c in model_features
    ])
    panel["buy_prob"] = model.predict_proba(feat_matrix)[:, 1]

    # ── Pass 1: collect all candidate signals per stock ──────────────────────
    candidates = []  # (entry_date, symbol, entry_idx in panel, sector)
    skipped_gate = 0

    for sym in symbols:
        sd  = panel[panel["symbol"] == sym].sort_values("DateTime").reset_index(drop=True)
        sec = SECTOR_MAP.get(sym, "Other")
        if len(sd) < 30:
            continue
        i = 0
        while i < len(sd) - max_holding_days - 1:
            if sd.loc[i, "buy_prob"] < buy_threshold:
                i += 1
                continue

            row = sd.loc[i]
            def rg(col): return float(row.get(col, 0) or 0)

            # Backtest gates: EMA200 (hard) + ADX + MACD + EMA50 + Volume
            # Not using full 7 gates in backtest since hist_20d_high may alias
            # to current-day data in some edge cases; keep proven simpler gates
            ema200_ok = rg("feat_ema200_ratio") > 0
            adx_ok    = rg("feat_adx") >= ADX_MIN
            macd_ok   = rg("feat_macd_hist") >= 0
            ema50_ok  = rg("feat_ema50_ratio") > 0      # [v3.0]
            vol_ok    = rg("feat_volume_ratio") >= VOLUME_THRESHOLD  # [v3.0]

            # Need EMA200 (hard) + 3 of remaining 4
            soft_pass = sum([adx_ok, macd_ok, ema50_ok, vol_ok])
            if not ema200_ok or soft_pass < 3:
                skipped_gate += 1
                i += 1
                continue

            entry_idx = i + 1
            if entry_idx >= len(sd):
                break

            candidates.append({
                "sym":       sym,
                "sector":    sec,
                "sig_idx":   i,
                "entry_idx": entry_idx,
                "entry_date": pd.Timestamp(sd.loc[entry_idx, "DateTime"]),
                "sd":        sd,
            })
            i = entry_idx + max_holding_days + 1  # don't re-enter until current expires

    if not candidates:
        print("No candidates passed all gates — no trades simulated.")
        return None

    candidates.sort(key=lambda x: x["entry_date"])

    # ── Pass 2: portfolio-level position limit enforcement ───────────────────
    trades = []
    open_positions: list[dict] = []   # {symbol, sector, exit_date}

    for cand in candidates:
        entry_date = cand["entry_date"]

        # Remove positions that have already exited
        open_positions = [p for p in open_positions if p["exit_date"] > entry_date]

        # Global position limit
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        # Sector exposure limit
        sector_count = sum(1 for p in open_positions if p["sector"] == cand["sector"])
        if sector_count >= MAX_SECTOR_EXPOSURE:
            continue

        sym       = cand["sym"]
        sd        = cand["sd"]
        i         = cand["sig_idx"]
        entry_idx = cand["entry_idx"]

        raw_entry  = float(sd.loc[entry_idx, "Open"])
        entry      = raw_entry * (1 + SLIPPAGE_PCT)
        atr        = float(sd.loc[i, "ATR"])
        stop       = raw_entry - ATR_STOP_MULT  * atr
        target     = raw_entry + ATR_TARGET_MULT * atr
        trail_stop = stop            # trailing stop starts at initial stop
        running_high = raw_entry

        # Determine regime at entry date for segmented analysis
        entry_regime = "UNKNOWN"
        if market_idx is not None:
            key = pd.Timestamp(entry_date).normalize()
            try:
                mrow = market_idx.loc[key] if key in market_idx.index else None
                if mrow is not None:
                    nc = float(mrow["nifty_close"] or 0)
                    f  = float(mrow["nifty_50dma"]  or nc)
                    s  = float(mrow["nifty_200dma"] or nc)
                    entry_regime = ("BULL" if nc > f > s else
                                   "BEAR" if nc < s or f < s else "SIDEWAYS")
            except Exception:
                pass

        raw_exit, reason, days = raw_entry, "timeout", max_holding_days
        for d in range(1, max_holding_days + 1):
            idx = entry_idx + d
            if idx >= len(sd):
                break
            bar_high = float(sd.loc[idx, "High"])
            bar_low  = float(sd.loc[idx, "Low"])

            # Ratchet trailing stop upward as trade moves in our favour
            if bar_high > running_high:
                running_high = bar_high
                trail_stop   = max(trail_stop, running_high - ATR_TRAIL_MULT * atr)

            if bar_low <= trail_stop:
                raw_exit, reason, days = trail_stop, "stop", d
                break
            if bar_high >= target:
                raw_exit, reason, days = target, "target", d
                break
        else:
            raw_exit = float(sd.loc[min(entry_idx + max_holding_days, len(sd)-1), "Close"])

        exit_price = raw_exit * (1 - SLIPPAGE_PCT)
        gross_pnl  = (exit_price - entry) / entry
        net_pnl    = gross_pnl - ROUND_TRIP_COST

        approx_exit_date = entry_date + timedelta(days=int(days * 1.4))
        open_positions.append({"symbol": sym, "sector": cand["sector"],
                               "exit_date": approx_exit_date})

        trades.append({
            "symbol":     sym,
            "sector":     cand["sector"],
            "regime":     entry_regime,
            "entry_date": entry_date,
            "entry":      round(entry, 2),
            "exit":       round(exit_price, 2),
            "pnl_pct":    net_pnl,
            "pnl_gross":  gross_pnl,
            "days":       days,
            "reason":     reason,
            "buy_prob":   float(sd.loc[i, "buy_prob"]),
        })

    if not trades:
        print("No trades executed after position-limit filtering.")
        return None

    tdf = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    tdf["year"] = pd.to_datetime(tdf["entry_date"]).dt.year

    # ── Core metrics ─────────────────────────────────────────────────────────
    wr          = (tdf["pnl_pct"] > 0).mean() * 100
    avg_pnl     = tdf["pnl_pct"].mean() * 100
    avg_gross   = tdf["pnl_gross"].mean() * 100
    best        = tdf["pnl_pct"].max() * 100
    worst       = tdf["pnl_pct"].min() * 100
    total_ret   = ((1 + tdf["pnl_pct"]).prod() - 1) * 100
    avg_days    = tdf["days"].mean()
    median_days = tdf["days"].median()
    sharpe      = (tdf["pnl_pct"].mean() /
                   tdf["pnl_pct"].std()) * np.sqrt(252 / max(avg_days, 1))

    wins   = tdf.loc[tdf["pnl_pct"] > 0, "pnl_pct"].sum()
    losses = tdf.loc[tdf["pnl_pct"] < 0, "pnl_pct"].sum()
    pf     = wins / abs(losses) if losses < 0 else float("inf")

    p_win      = (tdf["pnl_pct"] > 0).mean()
    avg_win    = tdf.loc[tdf["pnl_pct"] > 0, "pnl_pct"].mean() if p_win > 0 else 0.0
    avg_loss   = tdf.loc[tdf["pnl_pct"] < 0, "pnl_pct"].mean() if p_win < 1 else 0.0
    expectancy = (p_win * avg_win + (1 - p_win) * avg_loss) * 100

    equity  = (1 + tdf["pnl_pct"]).cumprod()
    peak    = equity.cummax()
    max_dd  = ((equity / peak - 1) * 100).min()

    # Buy-and-hold baseline
    bh = []
    for sym in symbols:
        sd = panel[panel["symbol"] == sym].sort_values("DateTime")
        if len(sd) >= 2:
            bh.append((sd.iloc[-1]["Close"] - sd.iloc[0]["Close"]) / sd.iloc[0]["Close"])
    bh_return = float(np.mean(bh)) * 100 if bh else 0.0

    reasons = tdf["reason"].value_counts()

    print(f"\n=== RESULTS ({len(tdf)} trades | {skipped_gate} gated out) ===")
    print(f"Win rate              : {wr:.1f}%")
    print(f"Avg P&L net/trade     : {avg_pnl:+.2f}%   (gross {avg_gross:+.2f}%)")
    print(f"Best / Worst trade    : {best:+.2f}%  /  {worst:+.2f}%")
    print(f"Median hold           : {median_days:.0f}d  (avg {avg_days:.1f}d)")
    print(f"Total compound return : {total_ret:+.2f}%")
    print(f"Buy-and-hold baseline : {bh_return:+.2f}%  (equal-weight {len(bh)} stocks, {years}y)")
    print(f"Alpha vs B&H          : {total_ret - bh_return:+.2f}%  "
          f"{'✓ beats' if total_ret > bh_return else '✗ underperforms'}")
    print(f"Sharpe (annualised)   : {sharpe:.2f}")
    print(f"Max drawdown          : {max_dd:.2f}%")
    print(f"Profit factor         : {pf:.2f}")
    print(f"Expectancy/trade      : {expectancy:+.3f}%")

    print("\nExit breakdown:")
    for r in ("target", "stop", "timeout"):
        n   = int(reasons.get(r, 0))
        pct = n / len(tdf) * 100 if len(tdf) else 0
        print(f"  {r:8s}: {n:4d}  ({pct:.1f}%)")

    # ── Yearly performance breakdown ─────────────────────────────────────────
    print("\nYearly performance:")
    yearly = (tdf.groupby("year").apply(lambda g: pd.Series({
        "trades":   len(g),
        "win_rate": (g["pnl_pct"] > 0).mean() * 100,
        "avg_pnl":  g["pnl_pct"].mean() * 100,
        "total":    ((1 + g["pnl_pct"]).prod() - 1) * 100,
    }), include_groups=False).round(2))
    print(yearly.to_string())

    # ── Regime-segmented performance ─────────────────────────────────────────
    if tdf["regime"].nunique() > 1:
        print("\nPerformance by market regime at entry:")
        regime_perf = (tdf.groupby("regime").apply(lambda g: pd.Series({
            "trades":   len(g),
            "win_rate": (g["pnl_pct"] > 0).mean() * 100,
            "avg_pnl":  g["pnl_pct"].mean() * 100,
        }), include_groups=False).round(2))
        print(regime_perf.to_string())

    # ── Sector breakdown ─────────────────────────────────────────────────────
    print("\nSector breakdown:")
    sector_perf = (tdf.groupby("sector").apply(lambda g: pd.Series({
        "trades":   len(g),
        "win_rate": (g["pnl_pct"] > 0).mean() * 100,
        "avg_pnl":  g["pnl_pct"].mean() * 100,
    }), include_groups=False).sort_values("trades", ascending=False).round(2))
    print(sector_perf.to_string())

    print("\nTop 5 winners:")
    print(tdf.nlargest(5, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "days", "reason",
                                       "regime"]]
          .assign(pnl_pct=lambda d: (d.pnl_pct * 100).round(2)).to_string(index=False))

    print(f"\nCosts: slippage {SLIPPAGE_PCT*100:.2f}%/fill  "
          f"round-trip {ROUND_TRIP_COST*100:.2f}%  trail={ATR_TRAIL_MULT}×ATR")
    print("⚠  Educational only — not financial advice.")
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
