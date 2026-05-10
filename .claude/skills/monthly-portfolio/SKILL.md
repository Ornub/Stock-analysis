---
name: monthly-portfolio
description: Build a ₹1 lakh (or custom capital) monthly swing portfolio from live signals with ATR-based position sizing, stop losses, and targets. Use when the user asks "what should I buy this month", "build a portfolio for 1 lakh", "monthly picks", "how do I deploy 1L capital", "suggest stocks for X budget".
---

# Build a monthly swing portfolio

## Steps

1. Run the v2.3 signal engine:
   ```bash
   python swing_v2.py predict
   ```

2. **Filter** the output: keep only rows where Signal = BUY and Grade is A, B, or C.
   - Prefer Grade A/B. Include C only if fewer than 3 A/B signals exist.
   - Skip any stock with a News grade of F (strong negative news).

3. **Rank** the filtered BUYs:
   - Grade A first, then B, then C
   - Within the same grade, rank by BUY% descending

4. **Select** top 3–5 stocks (never more than 5 positions in a ₹1L portfolio).

5. **Allocate capital** using the grade-based sizing table:

   | Grade | Allocation |
   |---|---|
   | A | ₹25,000–₹30,000 |
   | B | ₹20,000–₹25,000 |
   | C | ₹10,000–₹15,000 |

   Total must not exceed the user's capital. Reserve ≥10% as cash buffer.

6. **Compute shares** for each position:
   ```
   shares = floor(allocation / entry_price)
   actual_invested = shares × entry_price
   stop_loss = entry − 1.5 × ATR  (from predict output)
   target    = entry + 3 × ATR
   risk_per_trade = shares × (entry − stop_loss)
   ```

7. **Present** a portfolio table and trade plan.

## Output format

```markdown
## Monthly Portfolio — <date>
**Capital**: ₹1,00,000 | **Deployed**: ₹X | **Cash reserve**: ₹Y

| # | Stock | Grade | Entry (₹) | Shares | Invested (₹) | Stop (₹) | Target (₹) | Max Risk (₹) |
|---|---|---|---|---|---|---|---|---|
| 1 | SYMBOL | A | XXX | NN | XX,XXX | XXX | XXX | X,XXX |
| 2 | ...    |   |     |    |        |     |     |      |

**Total portfolio risk**: ₹X,XXX (X% of capital)
**Expected R:R**: 1:2 (ATR-based)

### Trade notes
- **SYMBOL (Grade A)**: Brief rationale — e.g. EMA200 uptrend, 6/6 confirmations, news grade A (upgrade)
- ...

### Exit rules
- Hit target → close full position, book profit
- Hit stop → close full position, accept loss
- 10 trading days elapsed → close if not already stopped/targeted

⚠️ Educational only — not financial advice. Past performance does not guarantee future results.
```

## Portfolio risk rules

- **Max single-stock risk**: 2% of total capital (₹2,000 on ₹1L)
- **Max total portfolio risk**: 6% of total capital (₹6,000 on ₹1L)
- If risk_per_trade exceeds 2% of capital → reduce shares until within limit
- Never allocate > 30% to a single stock
- If fewer than 3 BUY signals pass all gates → present only those, do not force 5 picks

## When no signals pass

If 0 BUY signals after news-gate filtering:
- Check WATCH list — mention which stocks are close but blocked and why
- Recommend holding cash and re-running the scan in 2–3 trading days
- Do NOT suggest stocks just to fill positions

## Notes

- Re-run `python swing_v2.py predict` each Monday morning for fresh signals
- For a specific stock deep-dive: `python news_features.py SYMBOL`
- If capital is different from ₹1L, scale allocations proportionally (Grade A = 25–30% of capital)
