# config.py
# Store your Angel One SmartAPI credentials here.
# NEVER commit this file with real credentials to version control.

# Your Angel One API key (from the SmartAPI developer portal)
API_KEY = "your_api_key_here"

# Your Angel One client/user ID (e.g. "A123456")
CLIENT_ID = "your_client_id_here"

# Your Angel One login password
PASSWORD = "your_password_here"

# The TOTP secret key shown when you enable TOTP in Angel One's settings.
# Pass this to pyotp to generate a live 6-digit one-time password at login time.
TOTP_SECRET = "your_totp_secret_here"
