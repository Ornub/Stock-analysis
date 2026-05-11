---
name: news-signals
description: Generate news-confirmed, high-confidence BUY signals using the v2.3 system with 6-factor confirmation gates and A/B/C grading. Use when the user asks for "safe signals", "news-confirmed trades", "high confidence picks", "which stocks have good news", or wants an extra safety layer on top of technical signals.
---

# News-confirmed swing signals (v2.3)

## Quick start

```bash
# Full predict with news gate (recommended)
python swing_v2.py predict

# Check news for a specific stock
python news_features.py RELIANCE
python news_features.py HDFCBANK
```

## How the output columns work

| Column | Meaning |
|---|---|
| **Signal** | BUY / SELL / HOLD / WATCH |
| **Grade** | A / B / C confidence (see below) |
| **BUY%** | Raw XGBoost BUY probability (need ≥ 60%, or ≥ 55% if news is A-grade) |
| **Confirms** | How many of 6 gates are bullish (e.g. `5/6`) |
| **News** | News grade A/B/C/F for this symbol |
| **Event** | Detected news event type (earnings, guidance, order, upgrade, etc.) |
| **Stop** | Entry − 1.5 × ATR |
| **Target** | Entry + 3 × ATR |
| **Blockers** | (WATCH only) Which gates failed |

## Grade definitions

- **Grade A** — 6/6 confirmations: highest conviction, take full size
- **Grade B** — 5/6 confirmations: good confidence, normal size
- **Grade C** — 4/6 confirmations: minimum threshold, reduce size or wait for dip
- **WATCH** — Model says BUY but fewer than 4/6 gates pass; monitor for improvement

## The 6 confirmation gates

1. **EMA200 uptrend** — price > 200-day EMA (hard filter; always required)
2. **ADX trend strength** — ADX ≥ 0.20 (directional momentum present)
3. **MACD momentum** — MACD histogram positive
4. **Relative strength** — 5-day return beats Nifty 50 by > 0%
5. **Market regime** — Nifty is above its 20-day EMA (broad market not in downtrend)
6. **News gate** — News score ≥ 0 (no recent bad news)

BUY requires **at least 4 of 6** gates to pass.  
BUY is **blocked entirely** if news score ≤ −1.0 (F-grade news).

## News sources and scoring

Sources scraped (with 6-hour TTL cache):
- Moneycontrol RSS (markets, business, stocks feeds)
- Economic Times RSS (markets, stocks feeds)
- LiveMint RSS (markets, companies feeds)
- Yahoo Finance via yfinance Ticker.news (most reliable)
- Google News RSS (fallback)

Scoring pipeline:
1. VADER sentiment on each headline → compound score (−1 to +1)
2. Event type detection (10 types) → bias applied:
   - Positive bias: earnings beat, guidance raise, large order win, upgrade, promoter buy, capex, dividend
   - Negative bias: regulatory action, downgrade, block deal sell
3. Recency weight (today = 1.0×, 5 days ago = 0.2×)
4. Aggregated to `news_score` = weighted average across all headlines

News grade:
- **A** (score ≥ 1.0): Strong positive news → probability bar relaxed from 60% → 55%
- **B** (0 < score < 1.0): Mildly positive / neutral positive
- **C** (−1.0 < score ≤ 0): Mildly negative / neutral negative
- **F** (score ≤ −1.0): Strong negative news → **BUY vetoed** regardless of model

## Interpreting the rationale block

After the table, a BUY RATIONALE section explains each BUY in plain English:

```
BUY RATIONALE
  • DRREDDY   (Grade B, BUY% 61.2): EMA200 uptrend + 5/6 bullish confirmations + news grade B (guidance)
  • HDFCBANK  (Grade A, BUY% 63.5): EMA200 uptrend + 6/6 bullish confirmations + news grade A (upgrade)
```

WATCH entries show their specific blockers:

```
WATCH (blocked): TATASTEEL — MACD histogram negative, market regime bearish
```

## Position sizing suggestion (₹1 lakh capital)

| Grade | Allocation |
|---|---|
| A | ₹30,000–₹35,000 |
| B | ₹20,000–₹25,000 |
| C | ₹10,000–₹15,000 |
| WATCH | Paper trade only |

Split across 3–5 BUY signals; never put > 35% in a single position.

## Workflow for daily use

```
1. python swing_v2.py predict          # morning scan
2. Review Grade A/B first
3. python news_features.py <SYMBOL>    # deep-dive on any stock you're unsure about
4. Enter A/B grades at open, set stop/target from table
5. Review WATCH list — if a blocker clears intraday, it becomes a late entry candidate
```

## Notes

- News gate is applied in `predict` only — the `backtest` command uses EMA200 + ADX (historical news unavailable)
- News cache expires every 6 hours; force refresh: `python news_features.py SYMBOL --refresh`
- The model was trained on 24 technical features only (no news in training data) — news is a post-model safety filter
- Walk-forward accuracy: ~55.6%, BUY precision: ~56.5%, backtest win rate: ~55.2% (67 trades, 5 years)

---
⚠️ **Educational only — not financial advice. Past performance does not guarantee future results.**
