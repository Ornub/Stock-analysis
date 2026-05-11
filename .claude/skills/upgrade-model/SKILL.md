---
name: upgrade-model
description: Retrain the pooled v2 XGBoost model on Nifty-50 stocks with 24 features and 5 years of history. Use when the user says "retrain", "improve the model", "upgrade", or accuracy is reported as poor.
---

# Retrain the v2.3 pooled model

## Steps

1. From the project root, run:
   ```bash
   python swing_v2.py train
   ```

2. This will:
   - Fetch **5 years** of daily OHLCV for ~45 Nifty-50 stocks (yfinance)
   - Engineer **24 features**: RSI, MACD histogram, MACD signal diff, EMA fast/slow/200 ratios, Bollinger Band position, ATR%, OBV ratio, gap%, body ratio, dist 52w high/low, 5d/10d/20d returns, volatility 20d, volume ratio, volume trend, ADX, Stochastic %K, Williams %R, CCI, high-low range%
   - Drop neutral samples (forward 5-day return between −2% and +2%)
   - Train pooled XGBoost with 5-fold TimeSeriesSplit walk-forward CV + early stopping (patience 30)
   - Save to `models/swing_v2.pkl`

3. Report back:
   - **Walk-forward accuracy per fold** (5 folds) with BUY precision/recall per fold
   - **Mean accuracy ± std**
   - **Top 10 feature importances**
   - **Number of training samples**
   - **Whether accuracy beats 50%** (the coin-flip baseline)

4. If accuracy < 55%, suggest:
   - Reviewing feature importance — drop near-zero features
   - Extending history further or adding more Nifty Next 50 stocks
   - Adjusting forward horizon (3 or 10 days instead of 5)
   - Checking for data quality issues (yfinance skipped stocks)

## Benchmark (v2.3)

- ~26,756 samples, 45 stocks, 5y history
- Mean walk-forward accuracy: **55.6%**, BUY precision: **56.5%**
- 57 trees (median of early-stopping across folds), max_depth=4, lr=0.03

## Notes

- Training takes ~8–12 minutes due to yfinance throttling + market regime fetch
- Model is overwritten each run; copy `models/swing_v2.pkl` before retraining to keep a backup
- Market regime features (Nifty return, VIX) are fetched for gate logic but are NOT in the model's feature set
- Run `python swing_v2.py predict` afterwards to use the new model
