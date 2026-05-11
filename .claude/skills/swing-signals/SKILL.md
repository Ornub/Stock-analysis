---
name: swing-signals
description: Generate fresh BUY/SELL/HOLD swing trading signals for the watchlist by running the signal engine. Use when the user asks "any trade ideas", "what should I buy/sell today", "give me signals", or similar.
---

# Generate today's swing signals (v2.3)

## Steps

1. From the project root, run:
   ```bash
   python swing_v2.py predict
   ```

2. Parse the **SUMMARY TABLE** in stdout. Columns: Symbol, Price, Signal, Grade, BUY%, Confirms, News, Event, Stop, Target.

3. Present results:
   - **BUY Grade A/B** rows — highest priority; show entry, stop, target, grade, news event
   - **BUY Grade C** rows — lower confidence; mention to reduce size
   - **WATCH** rows — model says BUY but gates blocked; show which blockers and watch for resolution
   - **SELL** rows — note reason if available

4. After the table, read the **BUY RATIONALE** block and include it verbatim or summarised.

5. If walk-forward accuracy shown is below 52%, add a warning that the ML layer is unreliable.

## Confidence grades

- **Grade A** (6/6 gates): Full-size position
- **Grade B** (5/6 gates): Normal-size position
- **Grade C** (4/6 gates): Half-size or wait
- **WATCH**: Monitor; do not enter until blockers clear

## Output format

```markdown
## Signals — <date>

| Stock | Price | Signal | Grade | BUY% | Confirms | News | Stop | Target |
|---|---|---|---|---|---|---|---|---|
| ... |

### Top BUY: <SYMBOL> (Grade A/B)
- Entry ₹X, Stop ₹Y, Target ₹Z (R:R 1:2)
- Confirmations: 5/6 — EMA200 uptrend, ADX momentum, MACD positive, …
- News: Grade B (guidance)

### Watch list
- SYMBOL — <blocker 1>, <blocker 2> (revisit tomorrow)

⚠️ Educational only — not financial advice.
```

## Notes

- 6-factor confirmation gates: EMA200, ADX, MACD, relative strength vs Nifty, market regime, news
- News grade F (score ≤ −1.0) vetoes BUY regardless of model confidence
- For deep news check on a specific stock: `python news_features.py SYMBOL`
