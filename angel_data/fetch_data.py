from SmartApi import SmartConnect
import pyotp, pytz, pandas as pd, requests
from datetime import datetime

from config import apikey, username, pwd, token

obj = SmartConnect(api_key=apikey)

# --- Login ---
def login():
    totp = pyotp.TOTP(token).now()
    data = obj.generateSession(username, pwd, totp)
    return data['data']['jwtToken'], obj.getfeedToken()

# --- Look up symbol token ---
def get_token(symbol):
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    df = pd.DataFrame(requests.get(url).json())
    result = df[df['symbol'] == symbol]
    if result.empty:
        print(f"Symbol '{symbol}' not found.")
        return None
    return result.iloc[0]['token']

# --- Fetch historical data ---
def fetch_historical_data(exchange, symboltoken, interval, from_date, to_date):
    try:
        params = {
            "exchange": exchange,
            "symboltoken": symboltoken,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date
        }
        resp = obj.getCandleData(params)
        df = pd.DataFrame(resp['data'],
                          columns=['DateTime','Open','High','Low','Close','Volume'])
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- Main ---
if __name__ == "__main__":
    login()
    sym_token = get_token("SBIN-EQ")
    df = fetch_historical_data("NSE", sym_token, "ONE_DAY",
                               "2024-01-01 09:15", "2024-12-31 15:30")
    print(df.head())
    df.to_csv("SBIN_historical.csv", index=False)
    print("Saved to SBIN_historical.csv")
