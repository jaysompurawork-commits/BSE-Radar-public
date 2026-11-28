"""
sentiment.py — BSE Radar Bot v7.0
News sentiment + FII-DII institutional flow analysis.
"""

import logging
import os
import re
from datetime import datetime, timedelta

import requests

from config import NEWS_API_KEY, IST

log = logging.getLogger("bse_radar")

__all__ = [
    "fetch_news_sentiment", "fetch_fii_dii_flow", "sentiment_score_for_ticker",
    "market_wide_sentiment",
]

# Market-wide / sector keywords mapped to sectors in the watchlist
SECTOR_KEYWORDS = {
    "Banking": ["bank", "rbi", "repo rate", "npa", "credit", "hdfc bank", "icici", "sbi", "axis", "kotak"],
    "IT": ["it services", "tcs", "infosys", "wipro", "hcl", "tech mahindra", "us revenue", "bps", "outsourcing"],
    "Energy": ["reliance", "ongc", "oil", "crude", "ongc", "coal india", "nv", "petrol", "diesel"],
    "Auto": ["maruti", "tata motors", "mahindra", "auto", "ev", "electric vehicle", "passenger vehicle"],
    "FMCG": ["fmcg", "hindustan unilever", "itc", "nestle", "consumer", "inflation", "rural demand"],
    "Pharma": ["pharma", "sun pharma", "cipla", "dr reddy", "drug", "fda", "usfda", "api"],
    "Metals": ["tata steel", "hindalco", "steel", "aluminium", "iron ore", "metal"],
    "Infra": ["adani", "infra", "roads", "ports", "capex", "government spending"],
    "Power": ["ntpc", "power grid", "power", "electricity", "renewable", "solar"],
    "Cement": ["ultratech", "cement", "shree cement", "construction"],
    "Consumer": ["titan", "asian paints", "consumer discretionary", "jewelry", "paint"],
    "Telecom": ["bharti airtel", "telecom", "5g", "tariff", "mobile"],
    "Finance": ["bajaj finance", "nbfc", "lending", "housing finance", "credit"],
}

CATEGORISED_TICKERS = {
    "Banking": ["HDFCBANK.BO", "SBIN.BO", "ICICIBANK.BO", "AXISBANK.BO", "KOTAKBANK.BO"],
    "IT": ["TCS.BO", "INFY.BO", "WIPRO.BO", "TECHM.BO", "HCLTECH.BO"],
    "Energy": ["RELIANCE.BO", "ONGC.BO", "COALINDIA.BO"],
    "Auto": ["MARUTI.BO", "M&M.BO", "TATAMOTORS.BO"],
    "FMCG": ["HINDUNILVR.BO", "ITC.BO", "NESTLEIND.BO"],
    "Pharma": ["SUNPHARMA.BO", "DRREDDY.BO", "CIPLA.BO"],
    "Metals": ["TATASTEEL.BO", "HINDALCO.BO"],
    "Infra": ["ADANIENT.BO", "ADANIPORTS.BO", "LT.BO"],
    "Power": ["NTPC.BO", "POWERGRID.BO"],
    "Cement": ["ULTRACEMCO.BO"],
    "Consumer": ["TITAN.BO", "ASIANPAINT.BO"],
    "Telecom": ["BHARTIARTL.BO"],
    "Finance": ["BAJFINANCE.BO", "BAJAJFINSV.BO"],
}


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── News sentiment ────────────────────────────────────────────────────────────

def fetch_news_headlines(query: str, limit: int = 8) -> list:
    """Fetch recent headlines for a query via NewsAPI (if key configured)."""
    if not NEWS_API_KEY:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            log.warning("NewsAPI %s for %s: %s", resp.status_code, query, resp.text[:120])
            return []
        data = resp.json()
        articles = data.get("articles", [])
        return [_clean_text(a.get("title", "")) for a in articles if a.get("title")]
    except Exception as exc:
        log.warning("News fetch failed for %s: %s", query, exc)
        return []


_POSITIVE = {
    "surge", "rally", "gain", "up", "rise", "jump", "boost", "beat", "beats",
    "record", "high", "growth", "profit", "profits", "upgrade", "bullish",
    "outperform", "buy", "strong", "positive", "momentum", "breakout",
    "all-time", "all time", "peak", "expand", "expansion", "deal", "wins",
    "award", "approval", "approved", "green light", "surpass", "exceed",
    "dividend", "bonus", "buyback", "demerger", "merger", "acquisition",
}
_NEGATIVE = {
    "fall", "falls", "drop", "drops", "decline", "declines", "slump", "slumps",
    "crash", "plunge", "plunges", "weak", "weakness", "bearish", "sell",
    "cut", "cuts", "miss", "misses", "loss", "losses", "downgrade", "underperform",
    "negative", "concern", "concerns", "worry", "worries", "risk", "risks",
    "delay", "delays", "probe", "investigation", "penalty", "fine", "ban",
    "suspend", "suspension", "default", "bankruptcy", "insolvency", "disappoint",
    "disappointing", "weaker", "soft", "softness", "headwind", "headwinds",
    "inflation", "rate hike", "tightening", "recession", "slowdown",
}


def _score_headline(headline: str) -> int:
    """Return -1, 0, or +1 for a headline based on keyword sentiment."""
    h = headline.lower()
    pos = sum(1 for w in _POSITIVE if w in h)
    neg = sum(1 for w in _NEGATIVE if w in h)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def sentiment_score_for_ticker(ticker: str) -> tuple:
    """
    Returns (score, label, confidence) where score in [-1, 1].
    score > 0 → bullish news, < 0 → bearish news.
    """
    from config import WATCHLIST
    info = WATCHLIST.get(ticker, {})
    name = info.get("name", ticker)
    sector = info.get("sector", "")

    queries = [name]
    if sector:
        queries.append(sector)
    # Keep queries short & relevant
    all_headlines = []
    for q in queries[:2]:
        all_headlines.extend(fetch_news_headlines(q, limit=5))

    if not all_headlines:
        return 0.0, "Neutral", 0.0

    scores = [_score_headline(h) for h in all_headlines if h]
    if not scores:
        return 0.0, "Neutral", 0.0

    avg = sum(scores) / len(scores)
    # Normalize to [-1, 1]
    score = max(-1.0, min(1.0, avg))
    if score > 0.15:
        label = "Bullish"
    elif score < -0.15:
        label = "Bearish"
    else:
        label = "Neutral"
    confidence = min(0.9, len(scores) / 8.0)
    return round(score, 2), label, round(confidence, 2)


def market_wide_sentiment() -> tuple:
    """Aggregate market-wide sentiment from broad India market headlines."""
    headlines = fetch_news_headlines("Nifty Sensex stock market India today", limit=10)
    if not headlines:
        return 0.0, "Neutral", 0.0
    scores = [_score_headline(h) for h in headlines if h]
    if not scores:
        return 0.0, "Neutral", 0.0
    avg = sum(scores) / len(scores)
    score = max(-1.0, min(1.0, avg))
    label = "Bullish" if score > 0.15 else ("Bearish" if score < -0.15 else "Neutral")
    return round(score, 2), label, round(min(0.9, len(scores) / 10.0), 2)


# ── FII / DII flow ────────────────────────────────────────────────────────────

def fetch_fii_dii_flow() -> tuple:
    """
    Fetch latest FII/DII net flow.
    Returns (net_flow_crores, label) where label in {"Bullish","Bearish","Neutral"}.
    Falls back to neutral if no data source configured.
    """
    # Placeholder: wire up NSE FII-DII endpoint or Alpha Vantage here when available.
    # For now return neutral with a clear flag so the scorer doesn't over-weight.
    return 0.0, "Neutral"
