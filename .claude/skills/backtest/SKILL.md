---
name: backtest
description: Walk-forward backtest of the v2 model with realistic trade simulation, win rate, P&L, and Sharpe ratio. Use when the user asks "backtest", "how would this perform", "is the strategy profitable", "validate the model".
---

# Backtest the v2.3 model

## Steps

1. Ensure a trained model exists at `models/swing_v2.pkl`. If not, run `python swing_v2.py train` first.

2. Run:
   ```bash
   python swing_v2.py backtest
   ```

3. The simulator will:
   - Generate a BUY signal at every row where `buy_prob > 0.6` AND EMA200 uptrend AND ADX ≥ 0.20
   - Enter long at the **next bar's open** (with 0.25% slippage)
   - Set stop at entry − 1.5 × ATR, target at entry + 3 × ATR
   - Exit at first of: stop hit, target hit, or 10 trading days elapsed
   - Deduct 0.50% round-trip brokerage + STT on exit
   - Record entry, exit, net P&L %, exit reason

4. Report:
   - **Total trades** and **win rate** (% closed at profit)
   - **Avg P&L per trade** and **expectancy**
   - **Best / worst trade**
   - **Total compounded return** vs buy-and-hold
   - **Sharpe ratio (annualised)** and **max drawdown**
   - **Profit factor** (gross wins / gross losses)
   - **Median holding period** (days)
   - **Exit-reason breakdown** (target / stop / timeout)

5. Verdict:
   - Win rate ≥ 50% AND beats buy-and-hold → strategy is viable
   - Sharpe ≥ 0.5 → acceptable risk-adjusted return
   - Profit factor ≥ 1.2 → edge over random entries
   - Otherwise → flag what to improve

## Benchmark results (v2.3, 5-year backtest)

- 67 trades, **55.2% win rate**, Sharpe 0.59, profit factor 1.25
- Expectancy +0.36%/trade, total compound +21.87%, beats B&H by +21.6%

## Notes

- Backtest uses EMA200 + ADX gates only (not news — historical news unavailable)
- Slippage (0.25%) and round-trip costs (0.50%) are fully modelled
- Walk-forward CV ensures no look-ahead bias in the trained model
