---
name: swing-signals
description: Generate fresh BUY/SELL/HOLD swing trading signals for the watchlist by running the signal engine. Use when the user asks "any trade ideas", "what should I buy/sell today", "give me signals", or similar.
---

# Generate today's swing signals

## Steps

1. From the project root, run:
   ```bash
   python swing_signals.py
   ```
   (For the upgraded model with better accuracy: `python swing_v2.py predict`.)

2. Parse the **SUMMARY TABLE** in stdout.

3. Present results:
   - **🟢 BUY** rows — show entry price, stop loss, target, R:R, and trigger reasons
   - **🔴 SELL** rows — show why (which indicators)
   - **🟡 HOLD** rows with score within 1 of trigger — call out as "watch list"

4. If walk-forward accuracy is shown and is below 52%, add a warning that the ML layer is unreliable and to weight the rule-based score higher.

5. End with the educational-only disclaimer.

## Output format

```markdown
## Signals — <date>

| Stock | Price | Signal | Score | Stop | Target |
|---|---|---|---|---|---|
| ... |

### Top BUY: <SYMBOL>
- Entry ₹X, Stop ₹Y, Target ₹Z (R:R 1:2)
- Triggered by: <which indicators>

⚠️ Educational only — not financial advice.
```
