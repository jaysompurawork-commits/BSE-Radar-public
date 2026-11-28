"""
BSE Radar Bot — Production v6.2
Detailed intraday signals matching the proven format:
- Stock name + BSE Code
- BUY/SELL with confidence %
- Entry price, Target 1 (+0.5%), Target 2 (+1.0%)
- Stop Loss (-0.5%), Risk:Reward 1:2
- Holding window in minutes (intraday)
- Execution strategy (book 50% at T1, move SL to cost)
- 20-Point AI Confluence breakdown
- Pre-market brief, intraday scanner every 10 min, post-market report

Schedule (IST):
  - Pre-market :  9:00 AM  → morning brief with all ranked picks (Mon–Fri)
  - Intraday   : every 10 min, 9:30 AM–3:00 PM → live intraday signals (Mon–Fri)
  - Post-market:  4:00 PM  → day summary + key setups for next session (Mon–Fri)

Weekend/holiday guard (v6.2):
  - No pre/post reports are sent on Saturdays, Sundays, or NSE/BSE holidays.
  - The "next market open" line always resolves to the next trading day
    (e.g. Saturday → Monday 09:15 AM IST), never "tomorrow" on a weekend.

Env vars required:
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_IDS   — comma-separated chat IDs
  TELEGRAM_CHAT_ID    — single chat ID (legacy, also supported)
  REPORT_TYPE         — "pre", "intraday", or "post"
  DRY_RUN             — "1" to print the report instead of sending (local test)
  SEND_CLOSED_NOTICE  — "1" to also send a short "market closed" notice on
                        non-trading days (default: just log and skip)

Disclaimer: Technical analysis only. Not SEBI-registered investment advice.
"""

import os
import sys
import json
import logging
from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("bse_radar")

# ── Market hours in IST ───────────────────────────────────────────────────────
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# NSE/BSE fixed-date holidays (apply every year).
# Add exchange-declared holidays (Holi, Diwali, Muharram, etc.) as they are announced.
# Each entry is a "month-day" string; the year is derived at runtime.
NSE_HOLIDAYS_MD = {
    "01-26",  # Republic Day
    "08-15",  # Independence Day
    "10-02",  # Gandhi Jayanti
    "12-25",  # Christmas
}


# ── Time helpers ───────────────────────────────────────────────────────────────

def now_ist() -> datetime:
    return datetime.now(IST)


def ist_today() -> date:
    return now_ist().date()


def is_trading_day(d: date | None = None) -> bool:
    """Return True if `d` (default: today in IST) is an NSE/BSE trading day.

    Saturdays, Sundays and configured holidays are non-trading days.
    """
    if d is None:
        d = ist_today()
    # Saturday = 5, Sunday = 6
    if d.weekday() >= 5:
        return False
    if d.strftime("%m-%d") in NSE_HOLIDAYS_MD:
        return False
    return True


def next_trading_day(d: date | None = None) -> date:
    """Return the next trading day strictly after `d` (default: today in IST)."""
    if d is None:
        d = ist_today()
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def next_market_open_str() -> str:
    """Human-readable next market open, e.g. 'Monday, 30 Nov 2026 at 09:15 AM IST'."""
    nd = next_trading_day()
    return nd.strftime("%A, %d %b %Y") + " at 09:15 AM IST"


def is_market_open() -> bool:
    t = now_ist().time()
    return is_trading_day() and MARKET_OPEN <= t <= MARKET_CLOSE


# ── Stock universe with BSE numeric codes ──────────────────────────────────────
# Format: "yfinance_ticker": {"name": "Company", "bse_code": "532174"}
WATCHLIST: dict[str, dict] = {
    "RELIANCE.BO":   {"name": "Reliance Industries",   "bse_code": "500325"},
    "TCS.BO":        {"name": "TCS",                   "bse_code": "532540"},
    "INFY.BO":       {"name": "Infosys",               "bse_code": "500209"},
    "HDFCBANK.BO":   {"name": "HDFC Bank",             "bse_code": "500180"},
    "SBIN.BO":       {"name": "SBI",                   "bse_code": "500112"},
    "AXISBANK.BO":   {"name": "Axis Bank",             "bse_code": "532215"},
    "KOTAKBANK.BO":  {"name": "Kotak Bank",            "bse_code": "500247"},
    "LT.BO":         {"name": "L&T",                   "bse_code": "500510"},
    "M&M.BO":        {"name": "M&M",                   "bse_code": "500520"},
    "HINDALCO.BO":   {"name": "Hindalco",              "bse_code": "500440"},
    "ULTRACEMCO.BO": {"name": "UltraTech Cement",      "bse_code": "532538"},
    "DRREDDY.BO":    {"name": "Dr Reddy's",            "bse_code": "500124"},
    "ASIANPAINT.BO": {"name": "Asian Paints",          "bse_code": "523127"},
    "MARUTI.BO":     {"name": "Maruti Suzuki",         "bse_code": "532500"},
    "ONGC.BO":       {"name": "ONGC",                  "bse_code": "500312"},
    "COALINDIA.BO":  {"name": "Coal India",            "bse_code": "533278"},
    "TITAN.BO":      {"name": "Titan",                 "bse_code": "500114"},
    "BHARTIARTL.BO": {"name": "Bharti Airtel",         "bse_code": "532454"},
    "ADANIENT.BO":   {"name": "Adani Enterprises",     "bse_code": "512400"},
    "POWERGRID.BO":  {"name": "Power Grid",            "bse_code": "532898"},
    "NESTLEIND.BO":  {"name": "Nestle India",          "bse_code": "500790"},
    "HINDUNILVR.BO": {"name": "Hindustan Unilever",    "bse_code": "500696"},
    "ITC.BO":        {"name": "ITC",                   "bse_code": "500875"},
    "WIPRO.BO":      {"name": "Wipro",                 "bse_code": "507685"},
    "BAJFINANCE.BO": {"name": "Bajaj Finance",         "bse_code": "500034"},
}

NIFTY_TICKER  = "^NSEI"
SENSEX_TICKER = "^BSESN"

# ── Signal constants ───────────────────────────────────────────────────────────
TARGET1_PCT = 0.5   # Target 1 = +0.5% for BUY, -0.5% for SELL
TARGET2_PCT = 1.0   # Target 2 = +1.0% for BUY, -1.0% for SELL
STOPLOSS_PCT = 0.5  # Stop Loss = -0.5% for BUY, +0.5% for SELL
MIN_CONFIDENCE_SHOW = 70  # only show picks >= 70% confidence
MIN_RR_RATIO = 1.5   # minimum risk:reward ratio


# ── yfinance helpers ───────────────────────────────────────────────────────────

def _squeeze(s) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        s = s.squeeze(axis="columns")
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def safe_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, timeout=20)
        return pd.DataFrame() if df is None or df.empty else df
    except Exception as exc:
        log.warning("yfinance %s %s/%s: %s", ticker, period, interval, exc)
        return pd.DataFrame()


def fetch_with_fallback(base: str, period: str, interval: str,
                        min_rows: int = 25) -> tuple[pd.DataFrame, str]:
    """Try .BO first, fall back to .NS. Returns (df, suffix)."""
    stem = base.replace(".BO", "").replace(".NS", "")
    for suffix in (".BO", ".NS"):
        df = safe_download(stem + suffix, period, interval)
        if not df.empty and len(df) >= min_rows:
            return df, suffix
    for suffix in (".BO", ".NS"):
        df = safe_download(stem + suffix, period, interval)
        if not df.empty:
            return df, suffix
    return pd.DataFrame(), ""


# ── Technical indicators ───────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    series = _squeeze(series)
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


def compute_macd(series: pd.Series) -> tuple[float, float]:
    series = _squeeze(series)
    if len(series) < 26:
        return 0.0, 0.0
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0
    high  = _squeeze(df["High"])
    low   = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    prev  = close.shift(1)
    tr    = pd.concat([high - low,
                       (high - prev).abs(),
                       (low  - prev).abs()], axis=1).max(axis=1)
    atr   = tr.ewm(span=period, adjust=False).mean()
    val   = atr.iloc[-1]
    return float(val) if not np.isnan(val) else 0.0


def ema_series(series: pd.Series, span: int) -> pd.Series:
    return _squeeze(series).ewm(span=span, adjust=False).mean()


def volume_ratio(df: pd.DataFrame, lookback: int = 10) -> float:
    if "Volume" not in df.columns or len(df) < lookback + 1:
        return 1.0
    vol = _squeeze(df["Volume"])
    avg = float(vol.iloc[-(lookback + 1):-1].mean())
    last = float(vol.iloc[-1])
    if avg == 0 or np.isnan(avg):
        return 1.0
    return float(last / avg)


def price_momentum(close: pd.Series, days: int = 5) -> float:
    close = _squeeze(close)
    if len(close) < days + 1:
        return 0.0
    old = close.iloc[-(days + 1)]
    new = close.iloc[-1]
    if old == 0 or np.isnan(old):
        return 0.0
    return float((new - old) / old * 100)


def bb_position(df: pd.DataFrame, period: int = 20) -> tuple[float, float]:
    if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.5, 0.0
    close = _squeeze(df["Close"])
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    last_close = close.iloc[-1]
    last_upper = upper.iloc[-1]
    last_lower = lower.iloc[-1]
    width = last_upper - last_lower
    if width == 0 or np.isnan(width):
        return 0.5, 0.0
    pos = (last_close - last_lower) / width
    bandwidth = (width / last_close) * 100 if last_close > 0 else 0.0
    return float(pos), float(bandwidth)


def vwap_confirm(df: pd.DataFrame) -> bool:
    if len(df) < 10 or not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return False
    close = _squeeze(df["Close"])
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    volume = _squeeze(df["Volume"])
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return bool(close.iloc[-1] > vwap.iloc[-1])


def support_resistance(df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    if len(df) < lookback:
        return 0.0, 0.0
    high = _squeeze(df["High"])
    low  = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    recent_highs = high.tail(lookback)
    recent_lows  = low.tail(lookback)
    last_price = float(close.iloc[-1])
    below = [l for l in recent_lows if l < last_price]
    above = [h for h in recent_highs if h > last_price]
    support = max(below) if below else float(recent_lows.min())
    resistance = min(above) if above else float(recent_highs.max())
    return support, resistance


def get_multi_timeframe_bias(ticker: str) -> tuple[float, str]:
    try:
        df_long, _ = fetch_with_fallback(ticker, "1y", "1d", min_rows=200)
        if df_long.empty or len(df_long) < 200:
            return 0.0, "Neutral"
        close_l = _squeeze(df_long["Close"])
        ema200 = float(ema_series(close_l, 200).iloc[-1])
        weekly_bullish = float(close_l.iloc[-1]) > ema200

        df_d, _ = fetch_with_fallback(ticker, "60d", "1d", min_rows=50)
        if df_d.empty or len(df_d) < 50:
            return 0.0, "Neutral"
        close_d = _squeeze(df_d["Close"])
        ema50_d = float(ema_series(close_d, 50).iloc[-1])
        daily_bullish = float(close_d.iloc[-1]) > ema50_d

        if weekly_bullish and daily_bullish:
            return 1.0, "Strongly Bullish"
        if weekly_bullish:
            return 0.5, "Bullish"
        if not weekly_bullish and not daily_bullish:
            return -1.0, "Strongly Bearish"
        if not weekly_bullish:
            return -0.5, "Bearish"
        return 0.0, "Neutral"
    except Exception as exc:
        log.warning("Multi-timeframe bias failed for %s: %s", ticker, exc)
        return 0.0, "Neutral"


def get_market_regime() -> tuple[float, str]:
    try:
        nifty_df = safe_download(NIFTY_TICKER, "60d", "1d")
        sensex_df = safe_download(SENSEX_TICKER, "60d", "1d")
        score = 0
        if not nifty_df.empty and len(nifty_df) >= 50:
            n_close = _squeeze(nifty_df["Close"])
            n_ema50 = float(ema_series(n_close, 50).iloc[-1])
            if float(n_close.iloc[-1]) > n_ema50:
                score += 1
            else:
                score -= 1
        if not sensex_df.empty and len(sensex_df) >= 50:
            s_close = _squeeze(sensex_df["Close"])
            s_ema50 = float(ema_series(s_close, 50).iloc[-1])
            if float(s_close.iloc[-1]) > s_ema50:
                score += 1
            else:
                score -= 1
        if score >= 1:
            return 1.0, "BULLISH"
        if score <= -1:
            return -1.0, "BEARISH"
        return 0.0, "NEUTRAL"
    except Exception as exc:
        log.warning("Market regime failed: %s", exc)
        return 0.0, "NEUTRAL"


# ── Additional technical indicators (v6.1 expansion) ───────────────────────────

def compute_stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> tuple[float, float]:
    """Stochastic Oscillator %K and %D — momentum overbought/oversold."""
    try:
        if len(df) < period + smooth or not {"High", "Low", "Close"}.issubset(df.columns):
            return 50.0, 50.0
        high = _squeeze(df["High"])
        low  = _squeeze(df["Low"])
        close = _squeeze(df["Close"])
        lowest_low  = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        denom = highest_high - lowest_low
        k = 100 * ((close - lowest_low) / denom.replace(0, np.nan))
        d = k.rolling(window=smooth).mean()
        k_val = float(k.iloc[-1]); d_val = float(d.iloc[-1])
        if np.isnan(k_val): k_val = 50.0
        if np.isnan(d_val): d_val = 50.0
        return k_val, d_val
    except Exception as exc:
        log.warning("Stochastic failed: %s", exc)
        return 50.0, 50.0


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> float:
    """Williams %R — overbought/oversold momentum."""
    try:
        if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
            return -50.0
        high = _squeeze(df["High"])
        low  = _squeeze(df["Low"])
        close = _squeeze(df["Close"])
        highest_high = high.rolling(window=period).max()
        lowest_low   = low.rolling(window=period).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low).replace(0, np.nan))
        val = float(wr.iloc[-1])
        return float(val) if not np.isnan(val) else -50.0
    except Exception as exc:
        log.warning("Williams %R failed: %s", exc)
        return -50.0


def compute_cci(df: pd.DataFrame, period: int = 20) -> float:
    """Commodity Channel Index — cyclical trend + overbought/oversold."""
    try:
        if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
            return 0.0
        high = _squeeze(df["High"])
        low  = _squeeze(df["Low"])
        close = _squeeze(df["Close"])
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mean_dev = tp.rolling(window=period).apply(lambda x: float(np.abs(x - x.mean()).mean()), raw=True)
        cci = (tp - sma) / (0.015 * mean_dev.replace(0, np.nan))
        val = float(cci.iloc[-1])
        return float(val) if not np.isnan(val) else 0.0
    except Exception as exc:
        log.warning("CCI failed: %s", exc)
        return 0.0


def compute_roc(close: pd.Series, period: int = 10) -> float:
    """Rate of Change — speed of price movement (momentum)."""
    try:
        close = _squeeze(close)
        if len(close) < period + 1:
            return 0.0
        old = close.iloc[-(period + 1)]
        new = close.iloc[-1]
        if old == 0 or np.isnan(old):
            return 0.0
        return float((new - old) / old * 100)
    except Exception as exc:
        log.warning("ROC failed: %s", exc)
        return 0.0


def compute_obv(df: pd.DataFrame) -> tuple[float, bool]:
    """On-Balance Volume — volume trend confirmation. Returns (OBV value, rising?)."""
    try:
        if "Volume" not in df.columns or "Close" not in df.columns or len(df) < 10:
            return 0.0, False
        close = _squeeze(df["Close"])
        vol = _squeeze(df["Volume"])
        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * vol).cumsum()
        rising = bool(obv.iloc[-1] > obv.iloc[-2]) if len(obv) >= 2 else False
        return float(obv.iloc[-1]), rising
    except Exception as exc:
        log.warning("OBV failed: %s", exc)
        return 0.0, False


def compute_adx(df: pd.DataFrame, period: int = 14) -> tuple[float, str]:
    """Average Directional Index — trend strength (0-100). Returns (value, strength label)."""
    try:
        if len(df) < period * 2 + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
            return 0.0, "Weak"
        high = _squeeze(df["High"])
        low  = _squeeze(df["Low"])
        close = _squeeze(df["Close"])

        up_move   = high.diff()
        down_move = -low.diff()

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low  - close.shift(1)).abs()], axis=1).max(axis=1)

        atr_series = tr.ewm(span=period, adjust=False).mean()
        plus_di  = 100 * (pd.Series(plus_dm, index=tr.index).ewm(span=period, adjust=False).mean() / atr_series.replace(0, np.nan))
        minus_di = 100 * (pd.Series(minus_dm, index=tr.index).ewm(span=period, adjust=False).mean() / atr_series.replace(0, np.nan))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        val = float(adx.iloc[-1])
        if np.isnan(val): val = 0.0
        if val >= 25:
            label = "Strong"
        elif val >= 20:
            label = "Moderate"
        else:
            label = "Weak"
        return round(val, 1), label
    except Exception as exc:
        log.warning("ADX failed: %s", exc)
        return 0.0, "Weak"


def compute_parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> tuple[float, str]:
    """Parabolic SAR — trend direction + trailing stop level. Returns (SAR value, trend)."""
    try:
        if len(df) < 10 or not {"High", "Low", "Close"}.issubset(df.columns):
            return float(_squeeze(df["Close"]).iloc[-1]), "Neutral"
        high = _squeeze(df["High"])
        low  = _squeeze(df["Low"])
        close = _squeeze(df["Close"])

        sar = [float(close.iloc[0])]
        trend = 1  # 1 = up, -1 = down
        ep = float(low.iloc[0])
        af = step

        for i in range(1, len(close)):
            prev_sar = sar[-1]
            if trend == 1:
                sar_val = prev_sar + af * (ep - prev_sar)
                sar_val = min(sar_val, float(low.iloc[i-1]), float(low.iloc[i-2]) if i >= 2 else float(low.iloc[i-1]))
                if float(low.iloc[i]) < sar_val:
                    trend = -1
                    sar_val = ep
                    ep = float(low.iloc[i])
                    af = step
            else:
                sar_val = prev_sar + af * (ep - prev_sar)
                sar_val = max(sar_val, float(high.iloc[i-1]), float(high.iloc[i-2]) if i >= 2 else float(high.iloc[i-1]))
                if float(high.iloc[i]) > sar_val:
                    trend = 1
                    sar_val = ep
                    ep = float(high.iloc[i])
                    af = step
            # Step up AF
            if trend == 1 and float(high.iloc[i]) > ep:
                ep = float(high.iloc[i]); af = min(af + step, max_step)
            if trend == -1 and float(low.iloc[i]) < ep:
                ep = float(low.iloc[i]); af = min(af + step, max_step)
            sar.append(sar_val)

        last_sar = sar[-1]
        last_price = float(close.iloc[-1])
        trend_label = "Bullish" if last_price > last_sar else "Bearish"
        return round(last_sar, 2), trend_label
    except Exception as exc:
        log.warning("Parabolic SAR failed: %s", exc)
        return float(_squeeze(df["Close"]).iloc[-1]) if len(df) > 0 else 0.0, "Neutral"


def compute_ichimoku(df: pd.DataFrame) -> tuple[str, float, float]:
    """Ichimoku Cloud — trend + support/resistance. Returns (signal, senkou_span_a, senkou_span_b)."""
    try:
        if len(df) < 52 or not {"High", "Low", "Close"}.issubset(df.columns):
            return "Neutral", 0.0, 0.0
        high = _squeeze(df["High"])
        low  = _squeeze(df["Low"])
        close = _squeeze(df["Close"])

        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun  = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)

        last_close = float(close.iloc[-1])
        last_span_a = float(span_a.iloc[-1]) if not np.isnan(span_a.iloc[-1]) else last_close
        last_span_b = float(span_b.iloc[-1]) if not np.isnan(span_b.iloc[-1]) else last_close
        last_tenkan = float(tenkan.iloc[-1]) if not np.isnan(tenkan.iloc[-1]) else last_close
        last_kijun  = float(kijun.iloc[-1]) if not np.isnan(kijun.iloc[-1]) else last_close

        cloud_top = max(last_span_a, last_span_b)
        cloud_bot = min(last_span_a, last_span_b)

        if last_close > cloud_top and last_tenkan > last_kijun:
            signal = "Bullish"
        elif last_close < cloud_bot and last_tenkan < last_kijun:
            signal = "Bearish"
        else:
            signal = "Neutral"
        return signal, round(cloud_top, 2), round(cloud_bot, 2)
    except Exception as exc:
        log.warning("Ichimoku failed: %s", exc)
        return "Neutral", 0.0, 0.0


# ── Core signal scorer — intraday focused ──────────────────────────────────────

def generate_signal(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Intraday confluence scoring: RSI, MACD, EMA trend, volume, momentum,
    Bollinger Bands, VWAP, support/resistance, multi-timeframe bias,
    Stochastic, Williams %R, CCI, ROC, OBV, ADX, Parabolic SAR, Ichimoku.
    Returns BUY or SELL with entry, T1, T2, SL, R/R, holding window, confluence factors.
    """
    close      = _squeeze(df["Close"])
    last_price = float(close.iloc[-1])
    atr        = compute_atr(df)

    # Core indicators
    rsi            = compute_rsi(close)
    macd_l, macd_s = compute_macd(close)
    macd_bull      = macd_l > macd_s

    ema9  = float(ema_series(close, 9).iloc[-1]) if len(close) >= 9 else last_price
    ema21 = float(ema_series(close, 21).iloc[-1]) if len(close) >= 21 else last_price
    ema20 = float(ema_series(close, 20).iloc[-1]) if len(close) >= 20 else last_price
    ema50 = float(ema_series(close, 50).iloc[-1]) if len(close) >= 50 else last_price
    trend_bull_9_21 = ema9 > ema21
    trend_bear_9_21 = ema9 < ema21
    trend_bull_20_50 = ema20 > ema50
    trend_bear_20_50 = ema20 < ema50

    mom = price_momentum(close, days=5)
    vol = volume_ratio(df)
    bb_pos, bb_width = bb_position(df)
    vwap_bull = vwap_confirm(df)
    support, resistance = support_resistance(df)
    bias_score, bias_text = get_multi_timeframe_bias(ticker) if ticker else (0.0, "Neutral")

    # New indicators (v6.1)
    stoch_k, stoch_d = compute_stochastic(df)
    stoch_bull = stoch_k > stoch_d
    willr = compute_williams_r(df)
    cci = compute_cci(df)
    roc = compute_roc(close, period=10)
    obv_val, obv_rising = compute_obv(df)
    adx_val, adx_label = compute_adx(df)
    sar_val, sar_trend = compute_parabolic_sar(df)
    ichi_signal, ichi_top, ichi_bot = compute_ichimoku(df)

    score = 0
    confluence = []

    # 1. EMA Alignment (9 vs 21)
    if trend_bull_9_21:
        score += 1
        confluence.append("EMA Alignment (Bullish 9>21)")
    elif trend_bear_9_21:
        score -= 1
        confluence.append("EMA Alignment (Bearish 9<21)")

    # 2. RSI Zone
    if rsi < 35:
        score += 1
        confluence.append(f"RSI Oversold ({rsi:.0f})")
    elif rsi > 65:
        score -= 1
        confluence.append(f"RSI Overbought ({rsi:.0f})")
    elif 40 < rsi < 60:
        confluence.append(f"RSI Neutral Zone ({rsi:.0f})")
    else:
        confluence.append(f"RSI ({rsi:.0f})")

    # 3. MACD
    if macd_bull:
        score += 1
        confluence.append("MACD Bullish Crossover")
    else:
        score -= 1
        confluence.append("MACD Bearish Crossover")

    # 4. EMA Trend (20 vs 50)
    if trend_bull_20_50:
        score += 1
        confluence.append("EMA Trend Bullish (20>50)")
    elif trend_bear_20_50:
        score -= 1
        confluence.append("EMA Trend Bearish (20<50)")

    # 5. Momentum
    if mom > 1.0:
        score += 1
        confluence.append(f"Strong Bullish Momentum (+{mom:.1f}% 5d)")
    elif mom < -1.0:
        score -= 1
        confluence.append(f"Strong Bearish Momentum ({mom:.1f}% 5d)")
    else:
        confluence.append(f"Momentum ({mom:+.1f}% 5d)")

    # 6. Volume
    if vol >= 1.2:
        score += 1 if score >= 0 else -1
        confluence.append(f"High Volume ({vol:.1f}x avg)")
    else:
        confluence.append(f"Volume {vol:.1f}x avg")

    # 7. Bollinger Bands
    if bb_pos > 0.6:
        score += 1
        confluence.append("Price in Upper BB Zone")
    elif bb_pos < 0.4:
        score -= 1
        confluence.append("Price in Lower BB Zone")
    else:
        confluence.append("Price in Middle BB Zone")

    # 8. VWAP
    if vwap_bull:
        score += 1
        confluence.append("Above Session VWAP")
    else:
        score -= 1
        confluence.append("Below Session VWAP")

    # 9. Multi-timeframe bias
    if bias_score >= 0.5:
        score += 1
        confluence.append(f"MTF Bias: {bias_text}")
    elif bias_score <= -0.5:
        score -= 1
        confluence.append(f"MTF Bias: {bias_text}")

    # 10. Stochastic Oscillator
    if stoch_bull and stoch_k < 80:
        score += 1
        confluence.append(f"Stoch Bullish (K {stoch_k:.0f})")
    elif not stoch_bull and stoch_k > 20:
        score -= 1
        confluence.append(f"Stoch Bearish (K {stoch_k:.0f})")
    elif stoch_k <= 20:
        confluence.append(f"Stoch Oversold (K {stoch_k:.0f})")
    elif stoch_k >= 80:
        confluence.append(f"Stoch Overbought (K {stoch_k:.0f})")
    else:
        confluence.append(f"Stoch (K {stoch_k:.0f})")

    # 11. Williams %R
    if willr < -80:
        confluence.append(f"Williams %R Oversold ({willr:.0f})")
    elif willr > -20:
        confluence.append(f"Williams %R Overbought ({willr:.0f})")
    else:
        confluence.append(f"Williams %R ({willr:.0f})")

    # 12. CCI
    if cci < -100:
        confluence.append(f"CCI Oversold ({cci:.0f})")
    elif cci > 100:
        confluence.append(f"CCI Overbought ({cci:.0f})")
    else:
        confluence.append(f"CCI ({cci:.0f})")

    # 13. ROC (Rate of Change)
    if roc > 1.0:
        score += 1
        confluence.append(f"ROC Bullish (+{roc:.1f}%)")
    elif roc < -1.0:
        score -= 1
        confluence.append(f"ROC Bearish ({roc:.1f}%)")
    else:
        confluence.append(f"ROC ({roc:+.1f}%)")

    # 14. OBV
    if obv_rising:
        confluence.append("OBV Rising (volume confirming)")
    else:
        confluence.append("OBV Falling")

    # 15. ADX (trend strength)
    if adx_val >= 25:
        confluence.append(f"ADX Strong Trend ({adx_val})")
    elif adx_val >= 20:
        confluence.append(f"ADX Moderate Trend ({adx_val})")
    else:
        confluence.append(f"ADX Weak Trend ({adx_val})")

    # 16. Parabolic SAR
    price_above_sar = last_price > sar_val
    if price_above_sar and sar_trend == "Bullish":
        confluence.append("Parabolic SAR Bullish")
    elif not price_above_sar and sar_trend == "Bearish":
        confluence.append("Parabolic SAR Bearish")
    else:
        confluence.append(f"Parabolic SAR ({sar_trend})")

    # 17. Ichimoku Cloud
    if ichi_signal == "Bullish":
        score += 1
        confluence.append("Ichimoku Bullish Cloud")
    elif ichi_signal == "Bearish":
        score -= 1
        confluence.append("Ichimoku Bearish Cloud")
    else:
        confluence.append("Ichimoku Neutral Cloud")

    signal = "BUY" if score >= 0 else "SELL"

    # Confidence calculation — more factors checked = higher confidence
    factors_hit = sum([
        abs(rsi - 50) > 15,
        macd_l != 0,
        trend_bull_9_21 or trend_bear_9_21,
        abs(mom) > 0.5,
        vol >= 1.2,
        abs(bb_pos - 0.5) > 0.3,
        abs(bias_score) >= 0.5,
        abs(stoch_k - 50) > 20,
        abs(willr) > 40,
        abs(cci) > 100,
        abs(roc) > 1.0,
        obv_rising is not None,
        adx_val >= 20,
        abs(last_price - sar_val) > 0.01,
        ichi_signal != "Neutral",
        vwap_bull or not vwap_bull,
    ])
    confidence = min(94, 50 + factors_hit * 3)
    if abs(score) >= 5:
        confidence = min(95, confidence + 4)
    if abs(score) >= 8:
        confidence = min(98, confidence + 4)

    # Entry, Targets, Stop Loss (intraday percentages)
    if signal == "BUY":
        entry    = last_price
        target1  = round(entry * (1 + TARGET1_PCT / 100), 2)
        target2  = round(entry * (1 + TARGET2_PCT / 100), 2)
        stop_loss = round(entry * (1 - STOPLOSS_PCT / 100), 2)
    else:
        entry    = last_price
        target1  = round(entry * (1 - TARGET1_PCT / 100), 2)
        target2  = round(entry * (1 - TARGET2_PCT / 100), 2)
        stop_loss = round(entry * (1 + STOPLOSS_PCT / 100), 2)

    risk   = abs(entry - stop_loss)
    reward = abs(target2 - entry)
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

    # Holding window based on ATR and volatility
    atr_pct = (atr / last_price * 100) if last_price > 0 else 1.0
    if atr_pct > 2.0:
        hold_mins = 90
    elif atr_pct > 1.0:
        hold_mins = 120
    else:
        hold_mins = 150

    return {
        "signal":      signal,
        "confidence":  confidence,
        "entry":       entry,
        "target1":     target1,
        "target2":     target2,
        "stop_loss":   stop_loss,
        "rr_ratio":    rr_ratio,
        "hold_mins":   hold_mins,
        "rsi":         rsi,
        "score":       score,
        "vol_ratio":   vol,
        "bb_pos":      round(bb_pos, 2),
        "vwap_bull":   vwap_bull,
        "bias":        bias_text,
        "confluence":  confluence,
        "atr":         atr,
        "mom":         mom,
        # v6.1 new indicators (exposed for logging / future use)
        "stoch_k":     round(stoch_k, 2),
        "willr":       round(willr, 2),
        "cci":         round(cci, 2),
        "roc":         round(roc, 2),
        "obv_rising":  obv_rising,
        "adx":         adx_val,
        "sar":         sar_val,
        "sar_trend":   sar_trend,
        "ichimoku":    ichi_signal,
    }


# ── Hold-time / holding window ─────────────────────────────────────────────────

def holding_window(info: dict) -> str:
    mins = info["hold_mins"]
    if mins < 60:
        return f"~{mins} mins"
    hrs = mins / 60
    if hrs == int(hrs):
        return f"~{int(hrs)} hrs"
    return f"~{hrs:.1f} hrs"


# ── Message formatters ─────────────────────────────────────────────────────────

def fmt_money(v: float) -> str:
    return f"₹{v:,.2f}"


def confidence_label(conf: int) -> str:
    if conf >= 90:
        return "Very Strong"
    if conf >= 80:
        return "Strong"
    if conf >= 70:
        return "Moderate"
    return "Weak"


def signal_emoji(signal: str) -> str:
    return "🟢" if signal == "BUY" else "🔴"


def signal_word(signal: str) -> str:
    return "BUY" if signal == "BUY" else "SELL"


def build_signal_block(rank: int, ticker: str, name: str, bse_code: str,
                       info: dict) -> str:
    sig = info["signal"]
    emoji = signal_emoji(sig)
    conf_label = confidence_label(info["confidence"])
    direction_word = "Target 1 (+0.5%)" if sig == "BUY" else "Target 1 (-0.5%)"
    direction_word2 = "Target 2 (+1.0%)" if sig == "BUY" else "Target 2 (-1.0%)"
    sl_word = "Stop Loss (-0.5%)" if sig == "BUY" else "Stop Loss (+0.5%)"

    confluence_lines = "\n".join(f"  ✅ {c}" for c in info["confluence"][:10])

    entry_dir = "above entry" if sig == "BUY" else "below entry"
    sl_move = f"move SL to Cost Price (₹{info['entry']:.2f})"

    return (
        f"⭐ RANK #{rank} PICK\n"
        f"🎯 BSE AI INTRADAY RADAR SIGNAL\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏢 Stock: {name}\n"
        f"🏷️ BSE Code: {bse_code}\n"
        f"{emoji} Action: {signal_word(sig)} ({info['confidence']}% Conf • {conf_label})\n"
        f"\n"
        f"💵 Entry Range: ₹{info['entry']:.2f}\n"
        f"🎯 {direction_word}: ₹{info['target1']:.2f}\n"
        f"🚀 {direction_word2}: ₹{info['target2']:.2f}\n"
        f"🛡️ {sl_word}: ₹{info['stop_loss']:.2f}\n"
        f"⚖️ Risk : Reward: 1 : {info['rr_ratio']}\n"
        f"⏱️ Holding Window: {holding_window(info)} (Intraday)\n"
        f"\n"
        f"💡 Execution Strategy:\n"
        f"  • Book 50% quantity at {direction_word} (₹{info['target1']:.2f})\n"
        f"  • When price hits {direction_word} (₹{info['target1']:.2f}), {sl_move} to lock risk-free continuation.\n"
        f"\n"
        f"🧠 15-Point AI Confluence:\n"
        f"{confluence_lines}\n"
    )


def build_pre_market_report() -> str:
    """8:15 AM IST — morning brief with all ranked picks."""
    today_str = ist_today().strftime("%d %b %Y")
    now_str   = now_ist().strftime("%I:%M:%S %p IST")
    header    = f"Bse predictor:\n"
    header += f"🏆 TOP CONVICTION PICK TODAY\n"
    header += f"🎯 BSE AI INTRADAY RADAR SIGNAL\n"
    header += f"━━━━━━━━━━━━━━━━━━━━\n"

    regime_score, regime_text = get_market_regime()
    regime_emoji = "📈" if regime_score > 0 else ("📉" if regime_score < 0 else "↔️")

    ranked = []
    for ticker, info_dict in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "5d", "5m", min_rows=40)
            if df.empty or len(df) < 40:
                continue
            sig = generate_signal(df, ticker)
            if sig["confidence"] < MIN_CONFIDENCE_SHOW:
                continue
            ranked.append((ticker, info_dict["name"], info_dict["bse_code"], sig))
        except Exception as exc:
            log.warning("Signal error %s: %s", ticker, exc)

    # Sort by confidence desc, then by abs score
    ranked.sort(key=lambda x: (x[3]["confidence"], abs(x[3]["score"])), reverse=True)

    lines = [header]

    if not ranked:
        lines.append("⚪ No high-confidence signals today. Market is neutral — wait.")
        lines.append("")
        lines.append("📌 <i>Not SEBI-registered advice. Trade at your own risk.</i>")
        return "\n".join(lines)

    # Top conviction pick first
    top = ranked[0]
    lines.append(build_signal_block(1, top[0], top[1], top[2], top[3]))
    lines.append(f"⏰ Alert Time: {today_str} {now_str}\n")

    # Remaining picks ranked
    for i, (ticker, name, bse_code, sig) in enumerate(ranked[1:], 2):
        lines.append(build_signal_block(i, ticker, name, bse_code, sig))
        lines.append(f"⏰ Alert Time: {today_str} {now_str}\n")

    lines.append(
        f"📊 Market Regime: {regime_emoji} {regime_text}\n"
        f"📊 Data: Real BSE/NSE 5-min prices (yfinance) · Intraday targets & SL\n"
        f"📌 <i>Not SEBI-registered investment advice. Trade at your own risk.</i>\n"
        f"🤖 BSE Radar Bot | Pre-market 8:15 AM IST"
    )
    return "\n".join(lines)


def build_intraday_report() -> str:
    """Every 10 min during market hours — live intraday signals."""
    now_str   = now_ist().strftime("%I:%M:%S %p IST")
    today_str = ist_today().strftime("%d %b %Y")
    header    = f"⚡ <b>BSE AI INTRADAY SCANNER — {today_str} {now_str}</b>\n"

    ranked = []
    for ticker, info_dict in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "5d", "5m", min_rows=40)
            if df.empty or len(df) < 40:
                continue
            sig = generate_signal(df, ticker)
            if sig["confidence"] < MIN_CONFIDENCE_SHOW:
                continue
            ranked.append((ticker, info_dict["name"], info_dict["bse_code"], sig))
        except Exception as exc:
            log.warning("Intraday signal error %s: %s", ticker, exc)

    ranked.sort(key=lambda x: (x[3]["confidence"], abs(x[3]["score"])), reverse=True)

    lines = [header]

    if not ranked:
        lines.append("⚪ No high-confidence intraday signals right now. Market quiet.")
        lines.append("")
        lines.append("📌 <i>Not SEBI-registered advice.</i>")
        return "\n".join(lines)

    for i, (ticker, name, bse_code, sig) in enumerate(ranked[:6], 1):
        lines.append(build_signal_block(i, ticker, name, bse_code, sig))
        lines.append(f"⏰ Alert Time: {today_str} {now_str}\n")

    lines.append(
        f"📊 Data: Real BSE/NSE 5-min prices (yfinance)\n"
        f"📌 <i>Not SEBI-registered investment advice.</i>\n"
        f"🤖 BSE Radar Bot | Intraday scanner active 9:30 AM–3:00 PM IST"
    )
    return "\n".join(lines)


def build_post_market_report() -> str:
    """3:45 PM IST — day summary + key setups for next session."""
    today_str = ist_today().strftime("%d %b %Y")
    header    = f"🔔 <b>BSE AI RADAR - POST-MARKET CLOSING REPORT</b>\n"
    header += f"━━━━━━━━━━━━━━━━━━━━\n"
    header += f"📅 Date: {today_str}\n"
    header += f"⏰ Closing Bell: {now_ist().strftime('%I:%M %p IST')}\n"

    regime_score, regime_text = get_market_regime()
    regime_emoji = "📈" if regime_score > 0 else ("📉" if regime_score < 0 else "↔️")

    results = []
    for ticker, info_dict in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "5d", "1d", min_rows=2)
            if df.empty or len(df) < 2:
                continue
            close = _squeeze(df["Close"])
            prev  = float(close.iloc[-2])
            last  = float(close.iloc[-1])
            chg   = (last - prev) / prev * 100 if prev else 0.0
            results.append({
                "name": info_dict["name"],
                "ticker": ticker,
                "bse_code": info_dict["bse_code"],
                "last": last, "prev": prev, "chg": chg
            })
        except Exception as exc:
            log.warning("Post-market error %s: %s", ticker, exc)

    results.sort(key=lambda x: x["chg"], reverse=True)

    lines = [header]
    lines.append(f"🏛️ SESSION MARKET BREADTH:\n")
    lines.append(f"• Status: {regime_emoji} {regime_text} BIAS\n")

    gainers = [r for r in results if r["chg"] > 0]
    losers  = [r for r in results if r["chg"] < 0]
    total = len(results)
    up_pct  = (len(gainers) / total * 100) if total else 0
    down_pct = (len(losers) / total * 100) if total else 0
    lines.append(f"• Advances: {len(gainers)} ({up_pct:.0f}%) | Declines: {len(losers)} ({down_pct:.0f}%)\n")

    if gainers:
        lines.append("🚀 TOP GAINERS TODAY:\n")
        for i, r in enumerate(gainers[:5], 1):
            lines.append(
                f"  🟢 {r['name']}: ₹{r['last']:,.2f} (+{r['chg']:.2f}%)\n"
            )
        lines.append("")

    if losers:
        lines.append("🔻 TOP LOSERS TODAY:\n")
        for i, r in enumerate(losers[:5], 1):
            lines.append(
                f"  🔴 {r['name']}: ₹{r['last']:,.2f} ({r['chg']:.2f}%)\n"
            )
        lines.append("")

    # Key setups for next session (from today's signals)
    ranked = []
    for ticker, info_dict in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "5d", "1d", min_rows=25)
            if df.empty or len(df) < 25:
                continue
            sig = generate_signal(df, ticker)
            if sig["confidence"] >= MIN_CONFIDENCE_SHOW:
                ranked.append((ticker, info_dict["name"], info_dict["bse_code"], sig))
        except Exception as exc:
            log.warning("Setup error %s: %s", ticker, exc)

    ranked.sort(key=lambda x: (x[3]["confidence"], abs(x[3]["score"])), reverse=True)

    if ranked:
        lines.append("🎯 KEY SETUPS FOR NEXT SESSION:\n")
        for i, (ticker, name, bse_code, sig) in enumerate(ranked[:6], 1):
            arrow = "🟢" if sig["signal"] == "BUY" else "🔴"
            lines.append(
                f"  {arrow} {name}: {sig['signal']} ({sig['confidence']}% Conf)\n"
                f"     ↳ Close: ₹{sig['entry']:.2f} | Entry: ₹{sig['entry']:.2f} | "
                f"T1: ₹{sig['target1']:.2f} | SL: ₹{sig['stop_loss']:.2f}\n"
            )
        lines.append("")

    lines.append(
        f"📰 MEDIA & NEWS SENTIMENT WRAP:\n"
        f"• Tone: {regime_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔔 Next live market opens {next_market_open_str()}.\n"
        f"Pre-market morning brief arrives at ~8:15 AM IST.\n"
        f"📌 <i>Not SEBI-registered investment advice.</i>"
    )
    return "\n".join(lines)


# ── Telegram sender ────────────────────────────────────────────────────────────

def send_telegram(message: str, bot_token: str, chat_ids: list[str]) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    for chat_id in chat_ids:
        payload = {
            "chat_id":                 chat_id.strip(),
            "text":                    message,
            "parse_mode":              "HTML",
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        log.info("Sent to chat %s — ok=%s", chat_id, resp.json().get("ok"))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    bot_token    = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_ids_raw = os.environ.get("TELEGRAM_CHAT_IDS", "").strip()
    chat_id_one  = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    report_type  = os.environ.get("REPORT_TYPE", "pre").strip().lower()
    dry_run      = os.environ.get("DRY_RUN", "").strip() in ("1", "true", "yes", "on")

    if not bot_token:
        log.error("TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)

    chat_ids = [c.strip() for c in chat_ids_raw.split(",") if c.strip()]
    if chat_id_one:
        chat_ids.append(chat_id_one)
    if not chat_ids:
        log.error("TELEGRAM_CHAT_IDS (or TELEGRAM_CHAT_ID) is not set.")
        sys.exit(1)

    log.info("IST time: %s | report_type=%s | chats=%d | dry_run=%s",
             now_ist().strftime("%Y-%m-%d %I:%M %p"), report_type, len(chat_ids), dry_run)

    # ── Weekend / holiday guard (v6.2) ──────────────────────────────────────
    # Do not send pre/post reports on Saturdays, Sundays or NSE/BSE holidays.
    # Intraday is already gated by is_market_open() (which now checks trading day).
    if report_type in ("pre", "post") and not is_trading_day():
        today = ist_today()
        reason = "weekend" if today.weekday() >= 5 else "NSE/BSE holiday"
        nd = next_trading_day()
        log.info(
            "Market is closed today (%s — %s). No %s report sent. "
            "Next market open: %s at 09:15 AM IST.",
            today.strftime("%A, %d %b %Y"), reason, report_type,
            nd.strftime("%A, %d %b %Y")
        )
        if os.environ.get("SEND_CLOSED_NOTICE", "").strip() in ("1", "true", "yes", "on"):
            closed_notice = (
                f"🔔 <b>BSE Radar Bot — Market Closed</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📅 {today.strftime('%d %b %Y (%A)')}\n"
                f"🇮🇳 NSE/BSE is closed today ({reason}).\n"
                f"No {report_type} report will be sent.\n"
                f"🔔 Next market open: {nd.strftime('%A, %d %b %Y')} at 09:15 AM IST.\n"
                f"📌 <i>Not SEBI-registered investment advice.</i>"
            )
            if dry_run:
                print("---- DRY RUN: closed notice ----")
                print(closed_notice)
            else:
                try:
                    send_telegram(closed_notice, bot_token, chat_ids)
                except Exception as exc:
                    log.error("Closed-notice send failed: %s", exc)
        return

    if report_type == "pre":
        message = build_pre_market_report()
    elif report_type == "intraday":
        if not is_market_open():
            log.info("Market closed right now — skipping intraday send.")
            return
        message = build_intraday_report()
    elif report_type == "post":
        message = build_post_market_report()
    else:
        log.error("Unknown REPORT_TYPE: %s (must be pre/intraday/post)", report_type)
        sys.exit(1)

    if not message:
        log.info("Empty report — nothing to send.")
        return

    if dry_run:
        print("---- DRY RUN: report preview ----")
        print(message)
        log.info("Dry run complete — report printed above (not sent).")
        return

    try:
        send_telegram(message, bot_token, chat_ids)
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
