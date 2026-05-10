---
name: upgrade-model
description: Retrain the pooled v2 XGBoost model on Nifty-50 stocks with 15 features and 3 years of history. Use when the user says "retrain", "improve the model", "upgrade", or accuracy is reported as poor.
---

# Retrain the v2 pooled model

## Steps

1. From the project root, run:
   ```bash
   python swing_v2.py train
   ```

2. This will:
   - Fetch 3 years of daily OHLCV for ~50 Nifty-50 stocks (yfinance)
   - Engineer 15 features (RSI, MACD, EMA ratios, BB pos, ATR%, OBV ratio, gap, body, 52w distances, 5d return, volatility, volume ratio)
   - Drop neutral samples (forward 5-day return between -2% and +2%)
   - Train pooled XGBoost with 5-fold TimeSeriesSplit walk-forward CV
   - Save to `models/swing_v2.pkl`

3. Report back:
   - **Walk-forward accuracy per fold** (5 numbers)
   - **Mean accuracy ± std**
   - **Top 10 feature importances**
   - **Number of training samples**
   - **Whether accuracy beats 50%** (the coin-flip baseline)

4. If accuracy < 55%, suggest:
   - Adding more stocks (mid/small cap)
   - Longer history (5 years)
   - Different forward horizon (3 or 10 days instead of 5)

## Notes

- Training takes ~5–10 minutes due to yfinance throttling
- Model is overwritten each run — there's no version history yet
- Run `python swing_v2.py predict` afterwards to use the new model
