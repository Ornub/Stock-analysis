"""
stock_fetcher.py

Fetch free historical OHLCV data for NSE/BSE stocks — no account or API key required.

Data sources:
    • jugaad-data  ← daily candles via NSE's official bhavcopy files (most reliable)
    • yfinance     ← intraday candles via Yahoo Finance (1m / 5m / 15m / 30m / 1h)

Install:
    pip install -r requirements.txt

Run:
    python stock_fetcher.py
"""

from datetime import date, datetime
import pandas as pd
from jugaad_data.nse import stock_df as _nse_stock_df
import yfinance as yf


# ---------------------------------------------------------------------------
# Interval configuration
# ---------------------------------------------------------------------------

# Mapping from human-friendly names to yfinance interval strings
YF_INTERVAL_MAP = {
    "ONE_MINUTE":     "1m",
    "TWO_MINUTE":     "2m",
    "FIVE_MINUTE":    "5m",
    "FIFTEEN_MINUTE": "15m",
    "THIRTY_MINUTE":  "30m",
    "ONE_HOUR":       "1h",
    "ONE_DAY":        "1d",
    "ONE_WEEK":       "1wk",
    "ONE_MONTH":      "1mo",
}

# Intervals that require the intraday source (jugaad-data only has end-of-day)
INTRADAY_INTERVALS = {
    "ONE_MINUTE", "TWO_MINUTE", "FIVE_MINUTE",
    "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR",
}

# Yahoo Finance hard limits on how far back intraday data goes
YF_LOOKBACK_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(d: "str | date") -> date:
    """Accept 'YYYY-MM-DD' string or datetime.date; always return date."""
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def _normalize_jugaad_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jugaad-data mirrors the raw NSE bhavcopy column names which can vary
    slightly across library versions. Map all known variants to the canonical
    set [DateTime, Open, High, Low, Close, Volume].
    """
    # NSE bhavcopy uses TOTTRDQTY for traded quantity (volume).
    # Newer jugaad-data versions may alias it to VOLUME.
    rename_map = {
        "DATE":      "DateTime",
        "TIMESTAMP": "DateTime",
        "OPEN":      "Open",
        "HIGH":      "High",
        "LOW":       "Low",
        "CLOSE":     "Close",
        "TOTTRDQTY": "Volume",
        "VOLUME":    "Volume",
        "TTQ":       "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"DateTime", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Could not map jugaad-data columns. "
            f"Missing: {missing}. Available: {list(df.columns)}"
        )

    return df[["DateTime", "Open", "High", "Low", "Close", "Volume"]].copy()


# ---------------------------------------------------------------------------
# Daily data — jugaad-data (official NSE bhavcopy, most reliable)
# ---------------------------------------------------------------------------

def fetch_daily_data(
    symbol: str,
    from_date: "str | date",
    to_date: "str | date",
    series: str = "EQ",
) -> "pd.DataFrame | None":
    """
    Download daily OHLCV candles for an NSE-listed equity.

    jugaad-data downloads NSE's official bhavcopy ZIP files (one per trading
    day), concatenates them, and filters to the requested symbol. This is the
    most accurate free source for end-of-day Indian market data, with history
    going back to 1994 for most stocks.

    Args:
        symbol:    NSE ticker without exchange suffix, e.g. "SBIN", "HDFCBANK".
        from_date: Start date — "YYYY-MM-DD" string or datetime.date.
        to_date:   End date   — "YYYY-MM-DD" string or datetime.date.
        series:    NSE series code. "EQ" covers all regular equity shares.
                   Use "BE" for trade-to-trade, "SM" for SME stocks, etc.

    Returns:
        DataFrame[DateTime, Open, High, Low, Close, Volume] sorted ascending,
        or None on failure.
    """
    from_date = _parse_date(from_date)
    to_date   = _parse_date(to_date)

    try:
        print(
            f"[INFO] Fetching daily data: {symbol} ({series}) "
            f"{from_date} → {to_date} via jugaad-data (NSE bhavcopy) …"
        )

        raw = _nse_stock_df(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            series=series,
        )

        if raw is None or raw.empty:
            print(f"[WARNING] jugaad-data returned no rows for {symbol}.")
            return None

        df = _normalize_jugaad_columns(raw)

        df["DateTime"] = pd.to_datetime(df["DateTime"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # bhavcopy is newest-first; sort to chronological order
        df = df.sort_values("DateTime").reset_index(drop=True)

        print(f"[INFO] Fetched {len(df)} daily candles.")
        return df

    except Exception as exc:
        print(f"[ERROR] jugaad-data fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Intraday data — yfinance (Yahoo Finance)
# ---------------------------------------------------------------------------

def fetch_intraday_data(
    symbol: str,
    exchange: str,
    interval: str,
    from_date: "str | date",
    to_date: "str | date",
) -> "pd.DataFrame | None":
    """
    Download intraday OHLCV candles via Yahoo Finance (yfinance).

    Yahoo Finance appends ".NS" for NSE and ".BO" for BSE tickers.
    Prices are adjusted for splits and dividends automatically.

    Intraday lookback limits imposed by Yahoo Finance:
        ONE_MINUTE  → last 7 days only
        2m – 1h     → last 60 days

    Args:
        symbol:    NSE/BSE ticker without suffix, e.g. "SBIN".
        exchange:  "NSE" or "BSE".
        interval:  ONE_MINUTE, TWO_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
                   THIRTY_MINUTE, or ONE_HOUR.
        from_date: Start date — "YYYY-MM-DD" string or datetime.date.
        to_date:   End date   — "YYYY-MM-DD" string or datetime.date.

    Returns:
        DataFrame[DateTime, Open, High, Low, Close, Volume] sorted ascending,
        or None on failure.
    """
    from_date = _parse_date(from_date)
    to_date   = _parse_date(to_date)

    yf_interval = YF_INTERVAL_MAP.get(interval)
    if yf_interval is None:
        print(
            f"[ERROR] Unsupported interval '{interval}'. "
            f"Valid options: {', '.join(YF_INTERVAL_MAP)}"
        )
        return None

    # Clamp from_date to Yahoo Finance's retention window for this interval
    max_days = YF_LOOKBACK_DAYS.get(yf_interval)
    if max_days:
        import datetime as _dt
        earliest_available = date.today() - _dt.timedelta(days=max_days)
        if from_date < earliest_available:
            print(
                f"[WARNING] Yahoo Finance only keeps {yf_interval} data for "
                f"the last {max_days} days. Adjusting from_date to {earliest_available}."
            )
            from_date = earliest_available

    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    ticker = f"{symbol}{suffix}"

    try:
        print(
            f"[INFO] Fetching {interval} intraday data: {ticker} "
            f"{from_date} → {to_date} via yfinance …"
        )

        tk = yf.Ticker(ticker)
        raw = tk.history(
            start=str(from_date),
            end=str(to_date),
            interval=yf_interval,
            auto_adjust=True,   # adjust prices for splits/dividends
            prepost=False,      # exclude pre/post-market sessions
        )

        if raw is None or raw.empty:
            print(f"[WARNING] yfinance returned no data for {ticker}.")
            return None

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "DateTime"
        df = df.reset_index()

        # Drop timezone info so the type is consistent with jugaad-data output
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("DateTime").reset_index(drop=True)

        print(f"[INFO] Fetched {len(df)} intraday candles.")
        return df

    except Exception as exc:
        print(f"[ERROR] yfinance fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Unified fetch function — single entry point for callers
# ---------------------------------------------------------------------------

def fetch_historical_data(
    symbol: str,
    from_date: "str | date",
    to_date: "str | date",
    interval: str = "ONE_DAY",
    exchange: str = "NSE",
    series: str = "EQ",
) -> "pd.DataFrame | None":
    """
    Fetch historical OHLCV data for any NSE or BSE stock.

    Automatically routes to the best available free data source:
        • ONE_DAY / ONE_WEEK / ONE_MONTH on NSE → jugaad-data (official bhavcopy)
        • Intraday intervals, or any BSE request  → yfinance (Yahoo Finance)

    Args:
        symbol:    Ticker without exchange suffix, e.g. "SBIN", "RELIANCE".
        from_date: Start date — "YYYY-MM-DD" string or datetime.date object.
        to_date:   End date   — "YYYY-MM-DD" string or datetime.date object.
        interval:  Candle interval. Supported values:
                     ONE_MINUTE, TWO_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
                     THIRTY_MINUTE, ONE_HOUR,
                     ONE_DAY (default), ONE_WEEK, ONE_MONTH
        exchange:  "NSE" (default) or "BSE".
        series:    NSE equity series, default "EQ". Ignored for BSE/intraday.

    Returns:
        DataFrame with columns [DateTime, Open, High, Low, Close, Volume],
        sorted chronologically, or None if the fetch fails.

    Examples:
        # Daily prices for HDFC Bank over a full year
        df = fetch_historical_data("HDFCBANK", "2024-01-01", "2024-12-31")

        # 15-minute candles for Reliance
        df = fetch_historical_data(
            "RELIANCE", "2025-03-10", "2025-04-09", interval="FIFTEEN_MINUTE"
        )
    """
    if interval in INTRADAY_INTERVALS:
        # jugaad-data has no intraday support; always use yfinance here
        return fetch_intraday_data(symbol, exchange, interval, from_date, to_date)

    if exchange.upper() == "NSE":
        # NSE daily/weekly/monthly → jugaad-data (authoritative official source)
        return fetch_daily_data(symbol, from_date, to_date, series)

    # BSE daily → yfinance (jugaad-data only covers NSE)
    return fetch_intraday_data(symbol, exchange, interval, from_date, to_date)


# ---------------------------------------------------------------------------
# Main — example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Example 1: Daily OHLCV for SBI (full year, from NSE bhavcopy) ────
    print("=" * 60)
    print("Example 1: Daily data — SBIN on NSE")
    print("=" * 60)

    df_daily = fetch_historical_data(
        symbol="SBIN",
        from_date="2024-01-01",
        to_date="2024-12-31",
        interval="ONE_DAY",
        exchange="NSE",
    )

    if df_daily is not None:
        print(f"\nRows      : {len(df_daily)}")
        print(f"Date range: {df_daily['DateTime'].min().date()} → "
              f"{df_daily['DateTime'].max().date()}")
        print(df_daily.head(5).to_string(index=False))

        out = "SBIN_NSE_daily.csv"
        df_daily.to_csv(out, index=False)
        print(f"\n[INFO] Saved to '{out}'.")

    print()

    # ── Example 2: 15-minute intraday candles for HDFCBANK (via yfinance) ─
    print("=" * 60)
    print("Example 2: 15-minute intraday data — HDFCBANK on NSE")
    print("=" * 60)

    df_intraday = fetch_historical_data(
        symbol="HDFCBANK",
        from_date="2025-04-01",
        to_date="2025-04-30",
        interval="FIFTEEN_MINUTE",
        exchange="NSE",
    )

    if df_intraday is not None:
        print(f"\nRows      : {len(df_intraday)}")
        print(df_intraday.head(5).to_string(index=False))

        out = "HDFCBANK_NSE_15min.csv"
        df_intraday.to_csv(out, index=False)
        print(f"\n[INFO] Saved to '{out}'.")
