"""
news_features.py — Trustworthy financial news scraper for NSE stocks.

Sources (priority order):
    1. Moneycontrol news RSS  (per-symbol when available; Markets feed otherwise)
    2. Economic Times Markets RSS
    3. LiveMint Markets RSS

We avoid blogs / social / Telegram / low-quality scrapers.

Per stock we derive:
    news_score          ∈ [-2, +2]  scaled VADER-weighted material score
    news_recency_days   freshness of the most recent matching headline
    news_event_type     one of: earnings | guidance | order | regulatory |
                                downgrade | upgrade | promoter | block_deal |
                                capex | dividend | other | none

Used as a SAFETY GATE in predict() — strong negative news suppresses BUY,
strong positive news allows a slightly lower probability bar.

Cached on disk (data/news_cache.json) to avoid hammering RSS endpoints.

⚠️ Educational only — not financial advice.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


CACHE_PATH      = Path("data/news_cache.json")
CACHE_TTL_HOURS = 6
HEADLINES_LIMIT = 25
HTTP_TIMEOUT    = 8
LOOKBACK_DAYS   = 5

USER_AGENT = "Mozilla/5.0 (compatible; swing-v2/2.0)"

_analyzer = SentimentIntensityAnalyzer()


# =============================================================================
# Trusted RSS feeds
# =============================================================================

# Each feed: (label, url-template). {q} is replaced with the search query.
RSS_FEEDS = [
    ("moneycontrol", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("moneycontrol_markets", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("moneycontrol_business", "https://www.moneycontrol.com/rss/business.xml"),
    ("et_markets",   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("et_stocks",    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("livemint_markets", "https://www.livemint.com/rss/markets"),
    ("livemint_companies", "https://www.livemint.com/rss/companies"),
]


# =============================================================================
# Event-type classifier (keyword based, light-weight)
# =============================================================================

EVENT_KEYWORDS: dict[str, list[str]] = {
    "earnings":   ["q1 result", "q2 result", "q3 result", "q4 result", "earnings",
                   "profit", "revenue", "net profit", "loss", "ebitda",
                   "topline", "bottomline", "results"],
    "guidance":   ["guidance", "outlook", "forecast", "raises target",
                   "lowers target", "management commentary"],
    "order":      ["order win", "wins order", "bags order", "contract win",
                   "lc award", "secures contract", "bagged contract"],
    "regulatory": ["sebi", "rbi", "ed probe", "show cause", "penalty",
                   "investigation", "tax demand", "compliance", "inspection"],
    "downgrade":  ["downgrade", "cut rating", "lowered rating", "reduce rating",
                   "underperform", "sell rating"],
    "upgrade":    ["upgrade", "raised rating", "outperform", "buy rating",
                   "overweight"],
    "promoter":   ["promoter pledge", "promoter sell", "promoter stake",
                   "stake sale", "promoter buys"],
    "block_deal": ["block deal", "bulk deal", "stake sale", "stake purchase"],
    "capex":      ["capex", "capacity expansion", "new plant", "acquisition",
                   "acquires", "merger"],
    "dividend":   ["dividend", "buyback", "bonus issue", "stock split"],
}


def classify_event(title: str) -> str:
    t = title.lower()
    for event, kws in EVENT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return event
    return "other"


# =============================================================================
# RSS fetch helpers
# =============================================================================

def _http_get(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def _parse_rss(xml_bytes: bytes) -> list[dict[str, Any]]:
    """Extract <item> elements as {title, ts, link}. Tolerates malformed XML."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    items = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link  = (it.findtext("link")  or "").strip()
        ts    = (it.findtext("pubDate") or "").strip()
        if title:
            items.append({"title": title, "ts": ts, "link": link})
    return items


_FEED_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_FEED_TTL = 1800   # seconds


def _fetch_feed(label: str, url: str) -> list[dict[str, Any]]:
    now = time.time()
    cached = _FEED_CACHE.get(url)
    if cached and (now - cached[0]) < _FEED_TTL:
        return cached[1]
    raw = _http_get(url)
    items = _parse_rss(raw) if raw else []
    for x in items:
        x["src"] = label
    _FEED_CACHE[url] = (now, items)
    return items


def _fetch_all_feeds() -> list[dict[str, Any]]:
    out = []
    for label, url in RSS_FEEDS:
        out.extend(_fetch_feed(label, url))
    return out


# =============================================================================
# Symbol → company name map (for matching against headlines)
# =============================================================================

# Most NSE tickers contain the company name; for a few, the headline uses a
# fuller form. Add overrides here as needed.
SYMBOL_NAME_OVERRIDES = {
    "RELIANCE":   ["reliance industries", "reliance"],
    "HDFCBANK":   ["hdfc bank"],
    "ICICIBANK":  ["icici bank"],
    "AXISBANK":   ["axis bank"],
    "KOTAKBANK":  ["kotak mahindra bank", "kotak bank"],
    "INDUSINDBK": ["indusind bank"],
    "BAJFINANCE": ["bajaj finance"],
    "BAJAJFINSV": ["bajaj finserv"],
    "BAJAJ-AUTO": ["bajaj auto"],
    "MARUTI":     ["maruti suzuki", "maruti"],
    "M&M":        ["mahindra & mahindra", "m&m"],
    "TATAMOTORS": ["tata motors"],
    "TATASTEEL":  ["tata steel"],
    "TATACONSUM": ["tata consumer"],
    "TCS":        ["tata consultancy", "tcs"],
    "TECHM":      ["tech mahindra"],
    "HCLTECH":    ["hcl tech", "hcltech"],
    "LT":         ["larsen & toubro", "l&t "],
    "ONGC":       ["ongc", "oil and natural gas"],
    "BPCL":       ["bharat petroleum", "bpcl"],
    "IOC":        ["indian oil", "ioc "],
    "GAIL":       ["gail"],
    "POWERGRID":  ["power grid", "powergrid"],
    "NTPC":       ["ntpc"],
    "COALINDIA":  ["coal india"],
    "SBIN":       ["sbi", "state bank of india"],
    "SBILIFE":    ["sbi life"],
    "HDFCLIFE":   ["hdfc life"],
    "HEROMOTOCO": ["hero motocorp", "hero motors"],
    "EICHERMOT":  ["eicher motors", "royal enfield"],
    "TVSMOTOR":   ["tvs motor"],
    "ULTRACEMCO": ["ultratech cement", "ultratech"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "NESTLEIND":  ["nestle india"],
    "BRITANNIA":  ["britannia"],
    "DRREDDY":    ["dr reddy", "dr. reddy", "dr reddys"],
    "SUNPHARMA":  ["sun pharma"],
    "DIVISLAB":   ["divi's lab", "divis lab"],
    "ZYDUSLIFE":  ["zydus lifesciences", "zydus life"],
    "TORNTPHARM": ["torrent pharma"],
    "ADANIPORTS": ["adani ports"],
    "ADANIENSOL": ["adani energy"],
    "ADANIGREEN": ["adani green"],
    "ADANIPOWER": ["adani power"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "INDIGO":     ["indigo airlines", "interglobe aviation"],
    "DMART":      ["dmart", "avenue supermarts"],
    "ZOMATO":     ["zomato", "eternal"],
    "PAYTM":      ["paytm", "one97 communications"],
    "LICI":       ["lic ", "life insurance corporation"],
    "ICICIPRULI": ["icici prudential"],
    "ICICIGI":    ["icici lombard"],
}


def _candidate_names(symbol: str) -> list[str]:
    sym = symbol.upper()
    names = SYMBOL_NAME_OVERRIDES.get(sym, [])
    # Always include lowercase ticker as a weak fallback
    names = list(dict.fromkeys(names + [sym.lower()]))
    return names


def _matches_symbol(title: str, names: list[str]) -> bool:
    t = title.lower()
    return any(n in t for n in names)


# =============================================================================
# Symbol-aware filtering + scoring
# =============================================================================

def _parse_pubdate(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return parsedate_to_datetime(ts)
    except Exception:
        return None


def _fetch_yfinance_news(symbol: str) -> list[dict[str, Any]]:
    """Trusted: Yahoo Finance pulls headlines from Reuters / Bloomberg / WSJ / TOI."""
    try:
        items = yf.Ticker(f"{symbol}.NS").news or []
    except Exception:
        return []
    out = []
    for it in items[:HEADLINES_LIMIT]:
        c = it.get("content") or it
        title = c.get("title") or it.get("title") or ""
        if not title:
            continue
        # Yahoo timestamp may be unix seconds or ISO string
        ts = c.get("pubDate") or it.get("providerPublishTime") or c.get("displayTime")
        ts_str = ""
        if isinstance(ts, (int, float)):
            ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        elif isinstance(ts, str):
            ts_str = ts
        out.append({"title": title, "ts": ts_str, "src": "yahoo_finance"})
    return out


def _fetch_google_news(symbol: str) -> list[dict[str, Any]]:
    """Fallback: Google News RSS — broad but generally credible aggregator."""
    query = quote_plus(f"{symbol} NSE share price OR results OR order")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    raw = _http_get(url)
    items = _parse_rss(raw) if raw else []
    for x in items:
        x["src"] = "google_news"
    return items[:HEADLINES_LIMIT]


def fetch_headlines(symbol: str, lookback_days: int = LOOKBACK_DAYS) -> list[dict[str, Any]]:
    """
    Combined headline fetch:
      1. Curated RSS (Moneycontrol / ET / LiveMint) — when available
      2. yfinance Ticker.news (Reuters / Bloomberg / TOI via Yahoo)
      3. Google News RSS as final fallback

    Then filter to last N days and to those mentioning the symbol.
    """
    names = _candidate_names(symbol)
    raw   = _fetch_all_feeds() + _fetch_yfinance_news(symbol) + _fetch_google_news(symbol)

    now = datetime.now(tz=timezone.utc)
    matched = []
    seen_titles: set[str] = set()
    for it in raw:
        title = it["title"]
        norm  = re.sub(r"\s+", " ", title.strip().lower())[:120]
        if norm in seen_titles:
            continue
        if not _matches_symbol(title, names):
            continue
        dt  = _parse_pubdate(it.get("ts", ""))
        age = None
        if dt:
            age = (now - dt).total_seconds() / 86400
            if age > lookback_days or age < -1:   # tolerate ~1d clock skew
                continue
        seen_titles.add(norm)
        it["age_days"] = age
        it["event"]    = classify_event(title)
        matched.append(it)
        if len(matched) >= HEADLINES_LIMIT:
            break
    return matched


# Event-type sentiment weighting (positive events get a small upweight)
EVENT_BIAS = {
    "earnings":   0.0,
    "guidance":   0.10,
    "order":      0.20,
    "upgrade":    0.30,
    "downgrade": -0.30,
    "regulatory": -0.25,
    "promoter":  -0.10,
    "block_deal": 0.0,
    "capex":      0.10,
    "dividend":   0.10,
    "other":      0.0,
}


def score_headlines(headlines: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Returns dict with:
        news_score        ∈ [-2, +2]   weighted sum
        news_count        int
        news_neg_share    fraction of negative-sentiment items
        news_pos_share    fraction of positive-sentiment items
        news_recency_days  age of most recent matched item (None if unknown)
        news_event_type   most prominent event type (mode)
    """
    if not headlines:
        return {
            "news_score":        0.0,
            "news_count":        0,
            "news_neg_share":    0.0,
            "news_pos_share":    0.0,
            "news_recency_days": None,
            "news_event_type":   "none",
        }

    scores, events = [], []
    for h in headlines:
        s = _analyzer.polarity_scores(h["title"])["compound"]
        s += EVENT_BIAS.get(h.get("event", "other"), 0.0)
        scores.append(max(-1.0, min(1.0, s)))
        events.append(h.get("event", "other"))

    n = len(scores)
    avg = sum(scores) / n
    raw = avg * 2.0   # scale to [-2, +2]
    score = max(-2.0, min(2.0, raw))

    neg_share = sum(1 for s in scores if s <= -0.10) / n
    pos_share = sum(1 for s in scores if s >=  0.10) / n
    ages = [h["age_days"] for h in headlines if h.get("age_days") is not None]
    recency = min(ages) if ages else None

    # Most common non-"other" event takes precedence
    event_counts: dict[str, int] = {}
    for ev in events:
        event_counts[ev] = event_counts.get(ev, 0) + 1
    sorted_events = sorted(event_counts.items(),
                           key=lambda kv: (kv[0] == "other", -kv[1]))
    event_type = sorted_events[0][0] if sorted_events else "other"

    return {
        "news_score":        round(float(score), 2),
        "news_count":        int(n),
        "news_neg_share":    round(float(neg_share), 2),
        "news_pos_share":    round(float(pos_share), 2),
        "news_recency_days": None if recency is None else round(float(recency), 1),
        "news_event_type":   event_type,
    }


# =============================================================================
# Cache
# =============================================================================

def _load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))


def _is_fresh(entry: dict[str, Any]) -> bool:
    fetched = entry.get("fetched_at")
    if not fetched:
        return False
    try:
        ts = datetime.fromisoformat(fetched).replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age_h = (datetime.now(tz=timezone.utc) - ts).total_seconds() / 3600
    return age_h < CACHE_TTL_HOURS


def get_news_features(symbol: str, *, force_refresh: bool = False) -> dict[str, Any]:
    """Cached, trusted-source news features for a symbol."""
    cache = _load_cache()
    entry = cache.get(symbol)
    keys  = ("news_score", "news_count", "news_neg_share", "news_pos_share",
             "news_recency_days", "news_event_type")
    if entry and not force_refresh and _is_fresh(entry):
        return {k: entry.get(k) for k in keys}

    headlines = fetch_headlines(symbol)
    feats     = score_headlines(headlines)
    cache[symbol] = {
        **feats,
        "fetched_at":  datetime.now(tz=timezone.utc).isoformat(),
        "n_headlines": len(headlines),
    }
    _save_cache(cache)
    return feats


# =============================================================================
# Gate logic for predict()
# =============================================================================

def news_grade(feats: dict[str, Any]) -> str:
    """A=positive, B=neutral, C=mildly negative, F=strong negative."""
    s = feats.get("news_score", 0.0) or 0.0
    if s >= 1.0:
        return "A"
    if s >= 0.0:
        return "B"
    if s >= -1.0:
        return "C"
    return "F"


def news_gate_passes(feats: dict[str, Any]) -> bool:
    """Block BUY only on strong negative news (score <= -1.0)."""
    return (feats.get("news_score", 0.0) or 0.0) > -1.0


def news_strong_positive(feats: dict[str, Any]) -> bool:
    """News is materially supportive — allow slightly lower probability bar."""
    return (feats.get("news_score", 0.0) or 0.0) >= 1.0


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    h = fetch_headlines(sym)
    print(f"{sym}: {len(h)} headlines from trusted feeds (last {LOOKBACK_DAYS}d)")
    for x in h[:8]:
        age = f"{x['age_days']:.1f}d" if x.get("age_days") is not None else "?"
        print(f"  - [{x['src']:>22s} | {x['event']:>12s} | {age:>5s}] {x['title'][:90]}")
    print("\nScored:", score_headlines(h))
