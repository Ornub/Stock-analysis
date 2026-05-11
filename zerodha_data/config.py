# zerodha_data/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Zerodha Kite Connect credentials
# Get these from: https://developers.kite.trade/
#
# Steps:
#   1. Log in to kite.trade → API → Create App
#   2. Note the api_key and api_secret below
#   3. Run:  python zerodha_data/auth.py login
#      → opens the login URL, paste the request_token after redirect
#   4. access_token is saved to zerodha_data/.token_cache and reused until 6 AM
# ─────────────────────────────────────────────────────────────────────────────

api_key    = "YOUR_API_KEY"        # from kite.trade developer console
api_secret = "YOUR_API_SECRET"     # from kite.trade developer console

# Optional: pre-fill your Zerodha user ID (only used in the login URL hint)
user_id    = "YOUR_ZERODHA_CLIENT_ID"   # e.g. ZZ1234
