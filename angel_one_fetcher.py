"""
angel_one_fetcher.py

Fetch free historical OHLCV data from Angel One's SmartAPI and save it to CSV.

Requirements:
    pip install smartapi-python pyotp pandas requests

Usage:
    python angel_one_fetcher.py
"""

import json
import requests
import pyotp
import pandas as pd
from SmartApi import SmartConnect

import config  # Local credentials file — see config.py


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def login():
    """
    Authenticate with Angel One SmartAPI using TOTP-based 2FA.

    Returns:
        tuple: (SmartConnect instance, auth_token str, feed_token str)
               Returns (None, None, None) on failure.
    """
    try:
        # Initialise the SmartAPI client with your API key
        smart_api = SmartConnect(api_key=config.API_KEY)

        # Generate the current 6-digit TOTP from the stored secret.
        # pyotp produces a fresh code every 30 seconds, matching what the
        # Angel One authenticator app shows.
        totp = pyotp.TOTP(config.TOTP_SECRET).now()

        # Perform the login; SmartAPI handles the REST call internally
        session_data = smart_api.generateSession(
            clientCode=config.CLIENT_ID,
            password=config.PASSWORD,
            totp=totp,
        )

        if not session_data or session_data.get("status") is False:
            print(f"[ERROR] Login failed: {session_data.get('message', 'unknown error')}")
            return None, None, None

        auth_token = session_data["data"]["jwtToken"]
        feed_token = smart_api.getfeedToken()

        print("[INFO] Login successful.")
        return smart_api, auth_token, feed_token

    except Exception as exc:
        print(f"[ERROR] Exception during login: {exc}")
        return None, None, None


# ---------------------------------------------------------------------------
# Symbol-token lookup
# ---------------------------------------------------------------------------

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)

def get_symbol_token(symbol_name: str, exchange: str = "NSE") -> str | None:
    """
    Look up the numeric symbol token for a stock by its trading symbol name.

    Angel One's historical-data API needs the token (e.g. "3045"), not the
    ticker string.  The OpenAPIScripMaster JSON is the authoritative source.

    Args:
        symbol_name: Trading symbol as it appears in the scrip master,
                     e.g. "SBIN-EQ", "HDFCBANK-EQ", "RELIANCE-EQ".
        exchange:    Exchange segment, e.g. "NSE", "BSE", "NFO".

    Returns:
        The token string if found, otherwise None.
    """
    try:
        print(f"[INFO] Downloading scrip master from Angel One …")
        response = requests.get(SCRIP_MASTER_URL, timeout=30)
        response.raise_for_status()

        scrip_master = response.json()

        # Each entry is a dict; filter by exchange and symbol name
        for entry in scrip_master:
            if (
                entry.get("exch_seg", "").upper() == exchange.upper()
                and entry.get("symbol", "").upper() == symbol_name.upper()
            ):
                token = entry["token"]
                print(f"[INFO] Found token {token} for {symbol_name} on {exchange}.")
                return token

        print(f"[WARNING] Symbol '{symbol_name}' not found on {exchange} in scrip master.")
        return None

    except requests.RequestException as exc:
        print(f"[ERROR] Failed to download scrip master: {exc}")
        return None
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to parse scrip master: {exc}")
        return None


# ---------------------------------------------------------------------------
# Historical data fetch
# ---------------------------------------------------------------------------

def fetch_historical_data(
    smart_api: SmartConnect,
    exchange: str,
    symbol_token: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame | None:
    """
    Fetch OHLCV candlestick data from Angel One's SmartAPI.

    Args:
        smart_api:    Authenticated SmartConnect instance (returned by login()).
        exchange:     Exchange segment, e.g. "NSE", "BSE", "NFO".
        symbol_token: Numeric token string for the instrument, e.g. "3045".
        interval:     Candle interval.  Supported values:
                        ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE,
                        TEN_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE,
                        ONE_HOUR, ONE_DAY
        from_date:    Start of the date range, format "YYYY-MM-DD HH:MM".
        to_date:      End   of the date range, format "YYYY-MM-DD HH:MM".

    Returns:
        A Pandas DataFrame with columns [DateTime, Open, High, Low, Close, Volume],
        or None if the request fails.
    """
    try:
        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }

        print(
            f"[INFO] Fetching {interval} data for token {symbol_token} "
            f"({exchange}) from {from_date} to {to_date} …"
        )

        response = smart_api.getCandleData(params)

        # The API returns {"status": True, "data": [[datetime, o, h, l, c, v], ...]}
        if not response or response.get("status") is False:
            print(f"[ERROR] API returned an error: {response.get('message', 'unknown')}")
            return None

        raw_data = response.get("data")
        if not raw_data:
            print("[WARNING] API returned no candle data for the given parameters.")
            return None

        # Build a clean DataFrame from the list of lists
        df = pd.DataFrame(
            raw_data,
            columns=["DateTime", "Open", "High", "Low", "Close", "Volume"],
        )

        # Parse the datetime column so callers can do time-series operations
        df["DateTime"] = pd.to_datetime(df["DateTime"])

        # Ensure numeric types for OHLCV columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])

        print(f"[INFO] Fetched {len(df)} candles.")
        return df

    except Exception as exc:
        print(f"[ERROR] Exception while fetching historical data: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main — example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Step 1: Login ---
    smart_api, auth_token, feed_token = login()
    if smart_api is None:
        raise SystemExit("Aborting: login failed.")

    # --- Step 2: Resolve the symbol token ---
    # Change SYMBOL_NAME to any NSE equity symbol, e.g. "HDFCBANK-EQ"
    SYMBOL_NAME = "SBIN-EQ"
    EXCHANGE = "NSE"

    symbol_token = get_symbol_token(SYMBOL_NAME, EXCHANGE)
    if symbol_token is None:
        raise SystemExit(f"Aborting: could not resolve token for {SYMBOL_NAME}.")

    # --- Step 3: Fetch historical data ---
    # Adjust the date range and interval as needed.
    # Note: ONE_DAY data is typically available for the past few years;
    # intraday intervals have a shorter lookback window (usually 30–60 days).
    df = fetch_historical_data(
        smart_api=smart_api,
        exchange=EXCHANGE,
        symbol_token=symbol_token,
        interval="ONE_DAY",
        from_date="2024-01-01 09:15",
        to_date="2024-12-31 15:30",
    )

    if df is None:
        raise SystemExit("Aborting: failed to fetch data.")

    # --- Step 4: Preview the data ---
    print("\n--- Sample rows ---")
    print(df.head(10).to_string(index=False))
    print(f"\nTotal rows : {len(df)}")
    print(f"Date range : {df['DateTime'].min()} → {df['DateTime'].max()}")

    # --- Step 5: Save to CSV ---
    output_file = f"{SYMBOL_NAME.replace('-', '_')}_{EXCHANGE}_ONE_DAY.csv"
    df.to_csv(output_file, index=False)
    print(f"\n[INFO] Data saved to '{output_file}'.")
