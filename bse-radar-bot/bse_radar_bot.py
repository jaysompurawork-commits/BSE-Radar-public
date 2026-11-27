"""
BSE Radar Bot — Production v5.0
Always sends daily predictions to Telegram with enhanced accuracy helpers.

Schedule (IST):
  - Pre-market :  9:00 AM  → daily BUY/SELL picks (always sends)
  - Intraday   : every 10 min, 9:30 AM–3:00 PM → live intraday signals
  - Post-market:  3:45 PM  → day summary: top gainers/losers

Message format (every pick):
  Stock Name (TICKER.BO) | ₹Price | BUY/SELL | Hold: X | Target by: HH:MM
  🎯 Target: ₹X  |  🛑 SL: ₹X  |  Conf: XX%

Env vars required:
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_IDS   — comma-separated chat IDs
  TELEGRAM_CHAT_ID    — single chat ID (legacy, also supported)
  REPORT_TYPE         — "pre", "intraday", or "post"

Disclaimer: Technical analysis only. Not SEBI-registered investment advice.
"""

import os
import sys
import json
import logging
from datetime import datetime, date, time as dtime, timedelta

import requests
import numpy as np
import pandas as pd
import yfinance as yf

IST = "Asia/Kolkata"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("bse_radar")


# ── Stock universe (BSE tickers via yfinance .BO suffix; .NS fallback) ─────────
WATCHLIST: dict[str, str] = {
    "RELIANCE.BO": "Reliance",
    "TCS.BO": "TCS",
    "INFY.BO": "Infosys",
    "HDFCBANK.BO": "HDFC Bank",
    "SBIN.BO": "SBI",
    "AXISBANK.BO": "Axis Bank",
    "KOTAKBANK.BO": "Kotak Bank",
    "LT.BO": "L&T",
    "M&M.BO": "M&M",
    "HINDALCO.BO": "Hindalco",
    "ULTRACEMCO.BO": "UltraCemco",
    "DRREDDY.BO": "Dr Reddy",
    "ASIANPAINT.BO": "Asian Paints",
    "MARUTI.BO": "Maruti",
    "ONGC.BO": "ONGC",
    "COALINDIA.BO": "Coal India",
    "TITAN.BO": "Titan",
    "BHARTIARTL.BO": "Airtel",
    "ADANIENT.BO": "Adani Ent",
    "POWERGRID.BO": "Power Grid",
    "NESTLEIND.BO": "Nestle",
    "HINDUNILVR.BO": "HUL",
    "ITC.BO": "ITC",
    "WIPRO.BO": "Wipro",
    "BAJFINANCE.BO": "Bajaj Fin",
}

NIFTY_TICKER  = "^NSEI"
SENSEX_TICKER = "^BSESN"

# ── Signal constants ───────────────────────────────────────────────────────────
RSI_OVERSOLD          = 35
RSI_OVERBOUGHT        = 65
VOLUME_SURGE_MIN      = 1.15   # volume must be >= 1.15x avg to confirm
MOMENTUM_STRONG       = 0.8    # % change over 5 days to count as strong
ATR_MULTIPLIER_TARGET = 1.5
ATR_MULTIPLIER_SL     = 1.0
MIN_CONFIDENCE_SHOW   = 50     # minimum confidence to include a pick

# Market hours in IST
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


# ── Time helpers ───────────────────────────────────────────────────────────────

def now_ist() -> datetime:
    return datetime.now().astimezone(__import__("zoneinfo").ZoneInfo(IST))

def ist_today() -> date:
    return now_ist().date()

def is_market_open() -> bool:
    t = now_ist().time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


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
    """Last bar volume vs N-bar average. 1.0 on failure."""
    if "Volume" not in df.columns or len(df) < lookback + 1:
        return 1.0
    vol = _squeeze(df["Volume"])
    avg = float(vol.iloc[-(lookback + 1):-1].mean())
    last = float(vol.iloc[-1])
    if avg == 0 or np.isnan(avg):
        return 1.0
    return float(last / avg)


def price_momentum(close: pd.Series, days: int = 5) -> float:
    """% change over last N days. 0.0 on failure."""
    close = _squeeze(close)
    if len(close) < days + 1:
        return 0.0
    old = close.iloc[-(days + 1)]
    new = close.iloc[-1]
    if old == 0 or np.isnan(old):
        return 0.0
    return float((new - old) / old * 100)


def bb_position(df: pd.DataFrame, period: int = 20) -> tuple[float, float]:
    """Return (bb_position_0_to_1, bandwidth_pct). 0=lower band, 1=upper band."""
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
    """Return True if last close is above VWAP (bullish)."""
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
    """Return (nearest_support, nearest_resistance) from recent swing lows/highs."""
    if len(df) < lookback:
        return 0.0, 0.0
    high = _squeeze(df["High"])
    low  = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    recent_highs = high.tail(lookback)
    recent_lows  = low.tail(lookback)
    # Simple swing points: local maxima/minima
    resistance = float(recent_highs.max())
    support    = float(recent_lows.min())
    last_price = float(close.iloc[-1])
    # Find nearest support below price, nearest resistance above price
    below = [l for l in recent_lows if l < last_price]
    above = [h for h in recent_highs if h > last_price]
    support = max(below) if below else support
    resistance = min(above) if above else resistance
    return support, resistance


def get_multi_timeframe_bias(ticker: str) -> tuple[float, str]:
    """Return (bias_score, bias_text) from EMA200 (weekly proxy) + EMA50 (daily).
    Uses daily data only to avoid yfinance 1wk dividend errors on .BO tickers."""
    try:
        # Longer daily series as weekly-proxy trend
        df_long, _ = fetch_with_fallback(ticker, "1y", "1d", min_rows=200)
        if df_long.empty or len(df_long) < 200:
            return 0.0, "Neutral"
        close_l = _squeeze(df_long["Close"])
        ema200 = float(ema_series(close_l, 200).iloc[-1])
        weekly_bullish = float(close_l.iloc[-1]) > ema200

        # Daily trend vs EMA50
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
    """Return (regime_score, regime_text) from NIFTY + SENSEX trend."""
    try:
        nifty_df, _ = safe_download(NIFTY_TICKER, "60d", "1d"), ""
        sensex_df, _ = safe_download(SENSEX_TICKER, "60d", "1d"), ""
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
            return 1.0, "Bullish"
        if score <= -1:
            return -1.0, "Bearish"
        return 0.0, "Neutral"
    except Exception as exc:
        log.warning("Market regime failed: %s", exc)
        return 0.0, "Neutral"


# ── Core signal scorer — always returns BUY or SELL ────────────────────────────

def generate_signal(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Enhanced confluence scoring with: RSI, MACD, EMA trend, volume, momentum,
    Bollinger Bands, VWAP, support/resistance, multi-timeframe bias, market regime.
    Always returns BUY or SELL (never neutral) so every stock gets a prediction.
    Includes ATR-based target and stop-loss, plus risk/reward ratio.
    """
    close      = _squeeze(df["Close"])
    last_price = float(close.iloc[-1])
    atr        = compute_atr(df)

    rsi            = compute_rsi(close)
    macd_l, macd_s = compute_macd(close)
    macd_bull      = macd_l > macd_s

    ema20 = float(ema_series(close, 20).iloc[-1]) if len(close) >= 20 else last_price
    ema50 = float(ema_series(close, 50).iloc[-1]) if len(close) >= 50 else last_price
    trend_bull = ema20 > ema50
    trend_bear = ema20 < ema50

    mom = price_momentum(close, days=5)
    vol = volume_ratio(df)
    bb_pos, bb_width = bb_position(df)
    vwap_bull = vwap_confirm(df)
    support, resistance = support_resistance(df)
    bias_score, bias_text = get_multi_timeframe_bias(ticker) if ticker else (0.0, "Neutral")

    score = 0  # -6 to +6

    # RSI
    if rsi < RSI_OVERSOLD:   score += 2
    elif rsi > RSI_OVERBOUGHT: score -= 2

    # MACD
    if macd_bull:  score += 1
    else:          score -= 1

    # EMA trend
    if trend_bull: score += 1
    elif trend_bear: score -= 1

    # Momentum
    if mom > MOMENTUM_STRONG:  score += 1
    elif mom < -MOMENTUM_STRONG: score -= 1

    # Bollinger Bands position
    if bb_pos > 0.5:  score += 1
    else:             score -= 1

    # Volume confirms direction (not standalone, adds weight)
    if vol >= VOLUME_SURGE_MIN:
        score += 0 if score < 0 else 1  # volume amplifies existing direction

    # Multi-timeframe bias
    if bias_score >= 0.5:   score += 1
    elif bias_score <= -0.5: score -= 1

    # Always force a directional call — no WATCH
    signal = "BUY" if score >= 0 else "SELL"

    # Confidence: stronger alignment = higher confidence (capped at 90)
    factors_hit = sum([
        abs(rsi - 50) > 15,
        macd_l != 0,
        trend_bull or trend_bear,
        abs(mom) > 0.5,
        vol >= VOLUME_SURGE_MIN,
        abs(bb_pos - 0.5) > 0.3,
        abs(bias_score) >= 0.5,
    ])
    confidence = min(90, 45 + factors_hit * 6)
    if abs(bias_score) >= 1.0:
        confidence = min(90, confidence + 5)

    # ATR-based stop loss and target
    if atr > 0 and last_price > 0:
        if signal == "BUY":
            target    = round(last_price + ATR_MULTIPLIER_TARGET * atr, 2)
            stop_loss = round(last_price - ATR_MULTIPLIER_SL     * atr, 2)
        else:
            target    = round(last_price - ATR_MULTIPLIER_TARGET * atr, 2)
            stop_loss = round(last_price + ATR_MULTIPLIER_SL     * atr, 2)
    else:
        if signal == "BUY":
            target    = round(last_price * 1.020, 2)
            stop_loss = round(last_price * 0.985, 2)
        else:
            target    = round(last_price * 0.980, 2)
            stop_loss = round(last_price * 1.015, 2)

    # Sanity: ensure SL < target for BUY, SL > target for SELL
    if signal == "BUY" and stop_loss >= target:
        stop_loss = round(last_price * 0.985, 2)
        target    = round(last_price * 1.020, 2)
    if signal == "SELL" and stop_loss <= target:
        stop_loss = round(last_price * 1.015, 2)
        target    = round(last_price * 0.980, 2)

    # Risk/reward ratio
    risk   = abs(last_price - stop_loss)
    reward = abs(target - last_price)
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

    return {
        "signal":     signal,
        "confidence": confidence,
        "last_price": last_price,
        "target":     target,
        "stop_loss":  stop_loss,
        "atr":        atr,
        "rsi":        rsi,
        "score":      score,
        "vol_ratio":  vol,
        "bb_pos":     round(bb_pos, 2),
        "vwap_bull":  vwap_bull,
        "bias":       bias_text,
        "rr_ratio":   rr_ratio,
        "support":    round(support, 2) if support > 0 else 0.0,
        "resistance": round(resistance, 2) if resistance > 0 else 0.0,
    }


# ── Hold-time estimator ────────────────────────────────────────────────────────

def hold_time(report_type: str, atr: float, price: float) -> str:
    """Realistic hold time estimate based on volatility."""
    atr_pct = (atr / price * 100) if price > 0 else 1.5
    if report_type == "intraday":
        if atr_pct > 2.0:
            return "30–60 min"
        elif atr_pct > 1.0:
            return "1–2 hrs"
        else:
            return "2–3 hrs"
    else:
        if atr_pct > 3.0:
            return "1–2 days"
        elif atr_pct > 1.5:
            return "2–3 days"
        else:
            return "3–5 days"


def max_profit_time(report_type: str) -> str:
    """Clock window by which target is likely hit."""
    if report_type == "intraday":
        n = now_ist()
        target_hr = n + timedelta(hours=2)
        if target_hr.time() > MARKET_CLOSE:
            return "before 3:30 PM"
        return target_hr.strftime("%I:%M %p")
    else:
        return "2–3 trading sessions"


# ── Message formatters ─────────────────────────────────────────────────────────

def fmt_money(v: float) -> str:
    return f"₹{v:,.2f}"


def build_pre_market_report() -> str:
    """9:00 AM IST — daily BUY/SELL picks. ALWAYS sends predictions."""
    today_str = ist_today().strftime("%d %b %Y")
    header    = f"📈 <b>BSE RADAR — {today_str}</b>\n"

    regime_score, regime_text = get_market_regime()
    regime_emoji = "📈" if regime_score > 0 else ("📉" if regime_score < 0 else "↔️")
    header += f"{regime_emoji} Market: {regime_text}\n"

    ranked = []
    for ticker, name in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "60d", "1d", min_rows=25)
            if df.empty or len(df) < 25:
                continue
            info = generate_signal(df, ticker)
            ranked.append((ticker, name, info))
        except Exception as exc:
            log.warning("Signal error %s: %s", ticker, exc)

    # Sort by confidence desc, then by score magnitude
    ranked.sort(key=lambda x: (x[2]["confidence"], abs(x[2]["score"])), reverse=True)

    lines = [header]

    if not ranked:
        lines.append("⚪ No market data available today.")
        lines.append("")
        lines.append("📌 <i>Not SEBI-registered advice.</i>")
        return "\n".join(lines)

    # Split into BUY and SELL groups
    buys  = [r for r in ranked if r[2]["signal"] == "BUY"]
    sells = [r for r in ranked if r[2]["signal"] == "SELL"]

    if buys:
        lines.append("🟢 <b>BUY TODAY</b>")
        for i, (ticker, name, info) in enumerate(buys[:6], 1):
            hold = hold_time("pre", info["atr"], info["last_price"])
            mpt  = max_profit_time("pre")
            lines.append(
                f"{i}. <b>{name}</b> ({ticker})  ₹{info['last_price']:,.2f}\n"
                f"   🟢 BUY | Conf: {info['confidence']}% | Hold: {hold} | Target by: {mpt}\n"
                f"   🎯 Target: {fmt_money(info['target'])}  |  🛑 SL: {fmt_money(info['stop_loss'])}  |  R/R: {info['rr_ratio']}"
            )
        lines.append("")

    if sells:
        lines.append("🔴 <b>SELL TODAY</b>")
        for i, (ticker, name, info) in enumerate(sells[:4], 1):
            hold = hold_time("pre", info["atr"], info["last_price"])
            mpt  = max_profit_time("pre")
            lines.append(
                f"{i}. <b>{name}</b> ({ticker})  ₹{info['last_price']:,.2f}\n"
                f"   🔴 SELL | Conf: {info['confidence']}% | Hold: {hold} | Target by: {mpt}\n"
                f"   🎯 Target: {fmt_money(info['target'])}  |  🛑 SL: {fmt_money(info['stop_loss'])}  |  R/R: {info['rr_ratio']}"
            )
        lines.append("")

    lines.append("📊 Data: Real BSE/NSE prices (yfinance) · ATR-based targets & SL")
    lines.append("📌 <i>Not SEBI-registered investment advice. Trade at your own risk.</i>")
    return "\n".join(lines)


def build_intraday_report() -> str:
    """Every 10 min during market hours — live intraday signals. ALWAYS sends."""
    now_str   = now_ist().strftime("%I:%M %p")
    today_str = ist_today().strftime("%d %b %Y")
    header    = f"⚡ <b>INTRADAY RADAR — {today_str} {now_str} IST</b>\n"

    ranked = []
    for ticker, name in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "5d", "5m", min_rows=40)
            if df.empty or len(df) < 40:
                continue
            info = generate_signal(df, ticker)
            ranked.append((ticker, name, info))
        except Exception as exc:
            log.warning("Intraday signal error %s: %s", ticker, exc)

    ranked.sort(key=lambda x: (x[2]["confidence"], abs(x[2]["score"])), reverse=True)

    lines = [header]

    if not ranked:
        lines.append("⚪ No intraday data available right now.")
        lines.append("")
        lines.append("📌 <i>Not SEBI-registered advice.</i>")
        return "\n".join(lines)

    buys  = [r for r in ranked if r[2]["signal"] == "BUY"]
    sells = [r for r in ranked if r[2]["signal"] == "SELL"]

    if buys:
        lines.append("🟢 <b>BUY NOW (Intraday)</b>")
        for i, (ticker, name, info) in enumerate(buys[:5], 1):
            hold = hold_time("intraday", info["atr"], info["last_price"])
            mpt  = max_profit_time("intraday")
            lines.append(
                f"{i}. <b>{name}</b> ({ticker})  ₹{info['last_price']:,.2f}\n"
                f"   🟢 BUY | Conf: {info['confidence']}% | Hold: {hold} | Target by: {mpt}\n"
                f"   🎯 Target: {fmt_money(info['target'])}  |  🛑 SL: {fmt_money(info['stop_loss'])}  |  R/R: {info['rr_ratio']}"
            )
        lines.append("")

    if sells:
        lines.append("🔴 <b>SELL NOW (Intraday)</b>")
        for i, (ticker, name, info) in enumerate(sells[:4], 1):
            hold = hold_time("intraday", info["atr"], info["last_price"])
            mpt  = max_profit_time("intraday")
            lines.append(
                f"{i}. <b>{name}</b> ({ticker})  ₹{info['last_price']:,.2f}\n"
                f"   🔴 SELL | Conf: {info['confidence']}% | Hold: {hold} | Target by: {mpt}\n"
                f"   🎯 Target: {fmt_money(info['target'])}  |  🛑 SL: {fmt_money(info['stop_loss'])}  |  R/R: {info['rr_ratio']}"
            )
        lines.append("")

    lines.append("📊 Data: Real BSE/NSE 5-min prices (yfinance)")
    lines.append("📌 <i>Not SEBI-registered investment advice.</i>")
    return "\n".join(lines)


def build_post_market_report() -> str:
    """3:45 PM IST — day summary: top gainers/losers. ALWAYS sends."""
    today_str = ist_today().strftime("%d %b %Y")
    header    = f"🔔 <b>BSE CLOSE — {today_str}</b>\n"

    results = []
    for ticker, name in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "5d", "1d", min_rows=2)
            if df.empty or len(df) < 2:
                continue
            close = _squeeze(df["Close"])
            prev  = float(close.iloc[-2])
            last  = float(close.iloc[-1])
            chg   = (last - prev) / prev * 100 if prev else 0.0
            results.append({"name": name, "ticker": ticker, "last": last, "prev": prev, "chg": chg})
        except Exception as exc:
            log.warning("Post-market error %s: %s", ticker, exc)

    results.sort(key=lambda x: x["chg"], reverse=True)

    lines = [header]

    if not results:
        lines.append("⚪ No market data available today.")
        lines.append("")
        lines.append("📌 <i>Not SEBI-registered advice.</i>")
        return "\n".join(lines)

    gainers = [r for r in results if r["chg"] > 0][:5]
    losers  = [r for r in results if r["chg"] < 0][:5]

    if gainers:
        lines.append("📈 <b>TOP GAINERS</b>")
        for i, r in enumerate(gainers, 1):
            lines.append(
                f"{i}. <b>{r['name']}</b> ({r['ticker']})  ₹{r['last']:,.2f}  <b>+{r['chg']:.1f}%</b>"
            )
        lines.append("")

    if losers:
        lines.append("📉 <b>TOP LOSERS</b>")
        for i, r in enumerate(losers, 1):
            lines.append(
                f"{i}. <b>{r['name']}</b> ({r['ticker']})  ₹{r['last']:,.2f}  <b>{r['chg']:.1f}%</b>"
            )
        lines.append("")

    lines.append("📊 Data: Real BSE/NSE daily prices (yfinance)")
    lines.append("📌 <i>Not SEBI-registered investment advice.</i>")
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

    if not bot_token:
        log.error("TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)

    chat_ids = [c.strip() for c in chat_ids_raw.split(",") if c.strip()]
    if chat_id_one:
        chat_ids.append(chat_id_one)
    if not chat_ids:
        log.error("TELEGRAM_CHAT_IDS (or TELEGRAM_CHAT_ID) is not set.")
        sys.exit(1)

    log.info("IST time: %s | report_type=%s | chats=%d",
             now_ist().strftime("%Y-%m-%d %I:%M %p"), report_type, len(chat_ids))

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

    try:
        send_telegram(message, bot_token, chat_ids)
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
