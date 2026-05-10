---
name: backtest
description: Walk-forward backtest of the v2 model with realistic trade simulation, win rate, P&L, and Sharpe ratio. Use when the user asks "backtest", "how would this perform", "is the strategy profitable", "validate the model".
---

# Backtest the v2 model

## Steps

1. Ensure a trained model exists at `models/swing_v2.pkl`. If not, run `python swing_v2.py train` first.

2. Run:
   ```bash
   python swing_v2.py backtest
   ```

3. The simulator will:
   - Generate a BUY signal at every row where `buy_prob > 0.6`
   - Enter long at the **next bar's open**
   - Set stop at entry − 1.5 × ATR, target at entry + 3 × ATR
   - Exit at first of: stop hit, target hit, or 10 trading days elapsed
   - Record entry, exit, P&L %, exit reason

4. Report:
   - **Total trades**
   - **Win rate** (% closed at profit)
   - **Avg P&L per trade**
   - **Best / worst trade**
   - **Total compounded return**
   - **Sharpe ratio (annualised)**
   - **Exit-reason breakdown** (target / stop / timeout)
   - **Comparison to buy-and-hold** of an equal-weight basket of the same stocks

5. Verdict:
   - If win rate ≥ 50% AND Sharpe ≥ 1.0 AND beats buy-and-hold → strategy is viable
   - Otherwise → flag what to improve (e.g. tighter probability threshold, longer holding period, feature additions)

## Notes

- Backtest uses look-ahead-free signal generation (model was trained on data ending before the test period)
- Slippage and brokerage are NOT modelled — real returns will be 0.5–1% lower per trade
- Run on 20 stocks by default for speed; pass `python swing_v2.py backtest_full` for full Nifty-50
