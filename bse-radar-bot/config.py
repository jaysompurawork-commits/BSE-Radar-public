"""
config.py — BSE Radar Bot v7.0 configuration
Watchlist, constants, thresholds, file paths, and time helpers.
"""

import os
from datetime import date, time as dtime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ── Market hours (IST) ────────────────────────────────────────────────────────
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
PRE_MARKET_START = dtime(9, 0)
POST_MARKET_END  = dtime(15, 30)

# ── NSE/BSE trading calendar (fixed-date holidays) ────────────────────────────
# Add exchange-declared holidays (Holi, Diwali, Muharram, etc.) as announced.
# Format: "MM-DD". Year is derived at runtime.
NSE_HOLIDAYS_MD = {
    "01-26",  # Republic Day
    "03-14",  # Holi (approx)
    "04-10",  # Good Friday (approx)
    "08-15",  # Independence Day
    "10-02",  # Gandhi Jayanti
    "10-20",  # Diwali (Laxmi Pujan) — approximate
    "12-25",  # Christmas
}


def now_ist():
    from datetime import datetime
    return datetime.now(IST)


def ist_today():
    return now_ist().date()


def is_trading_day(d: date | None = None) -> bool:
    if d is None:
        d = ist_today()
    if d.weekday() >= 5:  # Sat=5, Sun=6
        return False
    return d.strftime("%m-%d") not in NSE_HOLIDAYS_MD


def next_trading_day(d: date | None = None) -> date:
    if d is None:
        d = ist_today()
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def next_market_open_str() -> str:
    nd = next_trading_day()
    return nd.strftime("%A, %d %b %Y") + " at 09:15 AM IST"


def is_market_open() -> bool:
    t = now_ist().time()
    return is_trading_day() and MARKET_OPEN <= t <= MARKET_CLOSE


# ── Stock universe (yfinance ticker -> {name, bse_code, sector}) ──────────────
# Broad liquid universe across sectors for scanning; top picks ranked from it.
WATCHLIST: dict[str, dict] = {
    # Large-cap / index heavyweights
    "RELIANCE.BO":  {"name": "Reliance Industries",   "bse_code": "500325", "sector": "Energy"},
    "TCS.BO":       {"name": "TCS",                   "bse_code": "532540", "sector": "IT"},
    "INFY.BO":      {"name": "Infosys",               "bse_code": "500209", "sector": "IT"},
    "HDFCBANK.BO":  {"name": "HDFC Bank",             "bse_code": "500180", "sector": "Banking"},
    "SBIN.BO":      {"name": "SBI",                   "bse_code": "500112", "sector": "Banking"},
    "ICICIBANK.BO": {"name": "ICICI Bank",            "bse_code": "532174", "sector": "Banking"},
    "AXISBANK.BO":  {"name": "Axis Bank",             "bse_code": "532215", "sector": "Banking"},
    "KOTAKBANK.BO": {"name": "Kotak Bank",            "bse_code": "500247", "sector": "Banking"},
    "LT.BO":        {"name": "L&T",                   "bse_code": "500510", "sector": "Infra"},
    "HINDUNILVR.BO":{"name": "Hindustan Unilever",    "bse_code": "500696", "sector": "FMCG"},
    "ITC.BO":       {"name": "ITC",                   "bse_code": "500875", "sector": "FMCG"},
    "NESTLEIND.BO": {"name": "Nestle India",          "bse_code": "500790", "sector": "FMCG"},
    # Auto
    "MARUTI.BO":    {"name": "Maruti Suzuki",         "bse_code": "532500", "sector": "Auto"},
    "M&M.BO":       {"name": "M&M",                   "bse_code": "500520", "sector": "Auto"},
    "BAJFINANCE.BO":{"name": "Bajaj Finance",         "bse_code": "500034", "sector": "Finance"},
    "BAJAJFINSV.BO":{"name": "Bajaj Finserv",         "bse_code": "532978", "sector": "Finance"},
    # Pharma / Healthcare
    "SUNPHARMA.BO": {"name": "Sun Pharma",            "bse_code": "532349", "sector": "Pharma"},
    "DRREDDY.BO":   {"name": "Dr Reddy's",            "bse_code": "500124", "sector": "Pharma"},
    "CIPLA.BO":     {"name": "Cipla",                 "bse_code": "500087", "sector": "Pharma"},
    # Metals / Energy / Cement / Power
    "HINDALCO.BO":  {"name": "Hindalco",              "bse_code": "500440", "sector": "Metals"},
    "TATAMOTORS.BO":{"name": "Tata Motors",           "bse_code": "500570", "sector": "Auto"},
    "TATASTEEL.BO": {"name": "Tata Steel",            "bse_code": "500470", "sector": "Metals"},
    "ULTRACEMCO.BO":{"name": "UltraTech Cement",      "bse_code": "532538", "sector": "Cement"},
    "POWERGRID.BO": {"name": "Power Grid",            "bse_code": "532898", "sector": "Power"},
    "NTPC.BO":      {"name": "NTPC",                  "bse_code": "500156", "sector": "Power"},
    "ONGC.BO":      {"name": "ONGC",                  "bse_code": "500312", "sector": "Energy"},
    "COALINDIA.BO": {"name": "Coal India",            "bse_code": "533278", "sector": "Energy"},
    # Telecom / Cement / Infra / Cons. Durables
    "BHARTIARTL.BO":{"name": "Bharti Airtel",         "bse_code": "532454", "sector": "Telecom"},
    "ADANIENT.BO":  {"name": "Adani Enterprises",     "bse_code": "512400", "sector": "Infra"},
    "ADANIPORTS.BO":{"name": "Adani Ports",           "bse_code": "532921", "sector": "Infra"},
    "ASIANPAINT.BO":{"name": "Asian Paints",          "bse_code": "523127", "sector": "Consumer"},
    "TITAN.BO":     {"name": "Titan",                 "bse_code": "500114", "sector": "Consumer"},
}

NIFTY_TICKER  = "^NSEI"
SENSEX_TICKER = "^BSESN"
BANKNIFTY_TICKER = "^NSEBANK"

# Core index constituents (for breadth / regime)
INDEX_LIST = [NIFTY_TICKER, SENSEX_TICKER, BANKNIFTY_TICKER]

# ── Signal constants ──────────────────────────────────────────────────────────
TARGET1_PCT = 0.5   # Target 1 = ±0.5%
TARGET2_PCT = 1.0   # Target 2 = ±1.0%
STOPLOSS_PCT = 0.5  # Stop Loss = ±0.5%

MIN_CONFIDENCE_SHOW = 70      # only show picks >= 70% confidence
MIN_RR_RATIO = 1.5            # minimum risk:reward ratio
INTRADAY_MAX_PICKS = 6        # max picks in intraday report
PRE_MARKET_MAX_PICKS = 8      # max picks in pre-market report
POST_MARKET_SETUPS = 6        # max setups in post-market report

# Scoring weights (must sum sensibly; higher = more influence)
WEIGHTS = {
    "ema": 2,
    "rsi": 2,
    "macd": 2,
    "trend": 2,
    "momentum": 2,
    "volume": 1,
    "bb": 1,
    "vwap": 1,
    "mtf_bias": 2,
    "stoch": 1,
    "williams": 1,
    "cci": 1,
    "roc": 1,
    "obv": 1,
    "adx": 1,
    "sar": 1,
    "ichimoku": 2,
    "news_sentiment": 2,
    "fii_dii": 1,
}

# Confidence calibration
BASE_CONFIDENCE = 50
CONF_PER_FACTOR = 3
CONF_MAX = 94
CONF_STRONG_BONUS = 4   # |score| >= 5
CONF_VERY_STRONG_BONUS = 4  # |score| >= 8

# ── File / storage paths ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SIGNAL_LOG = os.path.join(DATA_DIR, "signals.jsonl")
PERFORMANCE_LOG = os.path.join(DATA_DIR, "performance.jsonl")
PREDICTION_LOG = os.path.join(DATA_DIR, "predictions.jsonl")

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_IDS_RAW = os.environ.get("TELEGRAM_CHAT_IDS", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
REPORT_TYPE = os.environ.get("REPORT_TYPE", "pre").strip().lower()
DRY_RUN = os.environ.get("DRY_RUN", "").strip() in ("1", "true", "yes", "on")
SEND_CLOSED_NOTICE = os.environ.get("SEND_CLOSED_NOTICE", "").strip() in ("1", "true", "yes", "on")
PAPER_TRADING = os.environ.get("PAPER_TRADING", "").strip() in ("1", "true", "yes", "on")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "").strip()

# ── Request / retry settings ──────────────────────────────────────────────────
YFINANCE_TIMEOUT = 25
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds
