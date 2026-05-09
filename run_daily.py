"""
run_daily.py

Called automatically by GitHub Actions every weekday after market close.
Fetches today's OHLCV data for every symbol in WATCHLIST and appends it
to a per-symbol CSV under the data/ folder (duplicates are dropped).
"""

from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from stock_fetcher import fetch_historical_data

# ── Add or remove symbols here ───────────────────────────────────────────────
WATCHLIST = [
    {"symbol": "SBIN",     "exchange": "NSE", "interval": "ONE_DAY"},
    {"symbol": "HDFCBANK", "exchange": "NSE", "interval": "ONE_DAY"},
    {"symbol": "RELIANCE", "exchange": "NSE", "interval": "ONE_DAY"},
    {"symbol": "TCS",      "exchange": "NSE", "interval": "ONE_DAY"},
    {"symbol": "INFY",     "exchange": "NSE", "interval": "ONE_DAY"},
    {"symbol": "ICICIBANK","exchange": "NSE", "interval": "ONE_DAY"},
    {"symbol": "WIPRO",    "exchange": "NSE", "interval": "ONE_DAY"},
    {"symbol": "AXISBANK", "exchange": "NSE", "interval": "ONE_DAY"},
]

OUTPUT_DIR = Path("data")


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    today     = date.today()
    # Fetch the last 7 days so we also catch any days missed by a failed run
    from_date = today - timedelta(days=7)
    to_date   = today

    results = {"ok": [], "skipped": []}

    for item in WATCHLIST:
        symbol   = item["symbol"]
        exchange = item["exchange"]
        interval = item["interval"]

        df = fetch_historical_data(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            exchange=exchange,
        )

        if df is None or df.empty:
            print(f"[SKIP] No data returned for {symbol} — market holiday or API issue.")
            results["skipped"].append(symbol)
            continue

        out_path = OUTPUT_DIR / f"{symbol}_{exchange}_{interval}.csv"

        if out_path.exists():
            # Append new rows; drop duplicate dates to stay idempotent
            existing = pd.read_csv(out_path, parse_dates=["DateTime"])
            combined = (
                pd.concat([existing, df], ignore_index=True)
                .drop_duplicates(subset=["DateTime"])
                .sort_values("DateTime")
                .reset_index(drop=True)
            )
            combined.to_csv(out_path, index=False)
            print(f"[OK] Updated {out_path}  ({len(combined)} total rows)")
        else:
            df.to_csv(out_path, index=False)
            print(f"[OK] Created {out_path}  ({len(df)} rows)")

        results["ok"].append(symbol)

    # Print summary
    print("\n── Run summary ─────────────────────────────────────")
    print(f"Fetched : {', '.join(results['ok'])   or 'none'}")
    print(f"Skipped : {', '.join(results['skipped']) or 'none'}")


if __name__ == "__main__":
    run()
