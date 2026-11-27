"""
BSE Radar Bot — Production v3.0
Sends clean, actionable predictions to Telegram.

Schedule (IST):
  - Pre-market :  9:00 AM  → BUY/SELL picks for the day (daily data)
  - Intraday   : every 30 min, 9:30 AM–3:00 PM → live intraday signals
  - Post-market:  4:00 PM  → day summary

Message format (minimal, actionable):
  Stock Name | ₹Price | BUY/SELL | Hold: Xh | SL: ₹X | Target by: HH:MM

Env vars required:
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_IDS   — comma-separated chat IDs (groups use negative numbers)
  TELEGRAM_CHAT_ID    — single chat ID (legacy, also supported)
  REPORT_TYPE         — "pre", "intraday", or "post"

Disclaimer: Technical analysis only. Not SEBI-registered advice. Trade at your own risk.
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


# ── Stock universe ─────────────────────────────────────────────────────────────
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

# ── Constants ──────────────────────────────────────────────────────────────────
RSI_OVERSOLD          = 38
RSI_OVERBOUGHT        = 62
MIN_CONFIDENCE        = 55   # only show picks at/above this
ATR_MULTIPLIER_TARGET = 1.5
ATR_MULTIPLIER_SL     = 1.0

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
    # last resort — whatever exists
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


# ── Core signal scorer ─────────────────────────────────────────────────────────

def generate_signal(df: pd.DataFrame) -> dict:
    """
    Score based on RSI, MACD, EMA trend, ATR for SL/target.
    Returns dict: signal, confidence, last_price, target, stop_loss, atr.
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

    # 5-day momentum
    mom = 0.0
    if len(close) >= 6:
        old = float(close.iloc[-6])
        if old > 0:
            mom = (last_price - old) / old * 100

    score = 0
    if rsi < RSI_OVERSOLD:   score += 2
    elif rsi > RSI_OVERBOUGHT: score -= 2
    if macd_bull:  score += 1
    else:          score -= 1
    if trend_bull: score += 1
    elif trend_bear: score -= 1
    if mom > 1.0:  score += 1
    elif mom < -1.0: score -= 1

    if score >= 3:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"
    elif score == 2:
        signal = "BUY"
    elif score == -2:
        signal = "SELL"
    else:
        signal = "WATCH"

    # Honest confidence
    factors_hit = sum([
        abs(rsi - 50) > 12,
        macd_l != 0,
        trend_bull or trend_bear,
        abs(mom) > 0.5,
    ])
    confidence = min(78, 48 + factors_hit * 7)
    if signal == "WATCH":
        confidence = 40

    # ATR-based stop loss and target
    if atr > 0:
        if signal == "BUY":
            target    = round(last_price + ATR_MULTIPLIER_TARGET * atr, 2)
            stop_loss = round(last_price - ATR_MULTIPLIER_SL     * atr, 2)
        elif signal == "SELL":
            target    = round(last_price - ATR_MULTIPLIER_TARGET * atr, 2)
            stop_loss = round(last_price + ATR_MULTIPLIER_SL     * atr, 2)
        else:
            target = stop_loss = last_price
    else:
        if signal == "BUY":
            target    = round(last_price * 1.020, 2)
            stop_loss = round(last_price * 0.985, 2)
        elif signal == "SELL":
            target    = round(last_price * 0.980, 2)
            stop_loss = round(last_price * 1.015, 2)
        else:
            target = stop_loss = last_price

    return {
        "signal":     signal,
        "confidence": confidence,
        "last_price": last_price,
        "target":     target,
        "stop_loss":  stop_loss,
        "atr":        atr,
        "rsi":        rsi,
    }


# ── Hold-time estimator ────────────────────────────────────────────────────────

def hold_time(report_type: str, atr: float, price: float) -> str:
    """
    Estimate realistic hold time.
    - Pre-market  → daily picks: 1-5 days
    - Intraday    → 30 min – 3 hrs
    - Post-market → reference only (positions held overnight)
    """
    if report_type == "intraday":
        # Volatility-adjusted intraday hold: low ATR% → shorter
        atr_pct = (atr / price * 100) if price > 0 else 1.0
        if atr_pct > 2.0:
            return "30–60 min"
        elif atr_pct > 1.0:
            return "1–2 hrs"
        else:
            return "2–3 hrs"
    else:
        # Daily picks: ATR drives how many sessions
        atr_pct = (atr / price * 100) if price > 0 else 1.5
        if atr_pct > 3.0:
            return "1–2 days"
        elif atr_pct > 1.5:
            return "2–3 days"
        else:
            return "3–5 days"


def max_profit_time(report_type: str) -> str:
    """Return the clock window by which target is likely hit."""
    if report_type == "intraday":
        # Next 1-3 hrs during market hours
        n = now_ist()
        target_hr = n + timedelta(hours=2)
        if target_hr.time() > MARKET_CLOSE:
            return "before 3:30 PM"
        return target_hr.strftime("%I:%M %p")
    else:
        # Day picks aim for next 2-3 sessions
        return "2–3 trading sessions"


# ── Message formatters ─────────────────────────────────────────────────────────

def fmt_money(v: float) -> str:
    return f"₹{v:,.2f}"


def build_pre_market_report(force_send: bool = False) -> str:
    """9:00 AM IST — daily BUY/SELL picks for today's session."""
    today_str = ist_today().strftime("%d %b %Y")  # e.g. 27 Nov 2026
    header    = f"📈 <b>BSE RADAR — {today_str}</b>\n"

    buy_picks  = []
    sell_picks = []

    for ticker, name in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "60d", "1d", min_rows=25)
            if df.empty or len(df) < 25:
                continue
            info = generate_signal(df)
            if info["signal"] in ("BUY",) and info["confidence"] >= MIN_CONFIDENCE:
                buy_picks.append((ticker, name, info))
            elif info["signal"] in ("SELL",) and info["confidence"] >= MIN_CONFIDENCE:
                sell_picks.append((ticker, name, info))
        except Exception as exc:
            log.warning("Signal error %s: %s", ticker, exc)

    buy_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)
    sell_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)

    lines = [header]

    if not buy_picks and not sell_picks:
        if force_send:
            lines.append("⚪ No strong signals today. Market is neutral — wait.")
        else:
            log.info("No strong signals today — nothing to send (pre-market).")
            return ""
        lines.append("")
        lines.append("📌 <i>Not SEBI-registered advice. Do your own research.</i>")
        return "\n".join(lines)

    # ── BUY picks ──
    if buy_picks:
        lines.append("🟢 <b>BUY TODAY</b>")
        for i, (ticker, name, info) in enumerate(buy_picks[:6], 1):
            hold = hold_time("pre", info["atr"], info["last_price"])
            mpt  = max_profit_time("pre")
            lines.append(
                f"{i}. <b>{name}</b> ({ticker})  ₹{info['last_price']:,.2f}\n"
                f"   🟢 BUY | Hold: {hold} | Target by: {mpt}\n"
                f"   🎯 Target: {fmt_money(info['target'])}  |  🛑 SL: {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    # ── SELL picks ──
    if sell_picks:
        lines.append("🔴 <b>SELL / SHORT TODAY</b>")
        for i, (ticker, name, info) in enumerate(sell_picks[:4], 1):
            hold = hold_time("pre", info["atr"], info["last_price"])
            mpt  = max_profit_time("pre")
            lines.append(
                f"{i}. <b>{name}</b> ({ticker})  ₹{info['last_price']:,.2f}\n"
                f"   🔴 SELL | Hold: {hold} | Target by: {mpt}\n"
                f"   🎯 Target: {fmt_money(info['target'])}  |  🛑 SL: {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    lines.append("📌 <i>Not SEBI-registered advice. Do your own research.</i>")
    return "\n".join(lines)


def build_intraday_report(force_send: bool = False) -> str:
    """Every 30 min during market hours — live intraday signals."""
    now_str   = now_ist().strftime("%I:%M %p")
    today_str = ist_today().strftime("%d %b %Y")
    header    = f"⚡ <b>INTRADAY RADAR — {today_str} {now_str} IST</b>\n"

    buy_picks  = []
    sell_picks = []

    for ticker, name in WATCHLIST.items():
        try:
            # 5-min candles over last 5 days; need >= 40 bars for indicators
            df, _ = fetch_with_fallback(ticker, "5d", "5m", min_rows=40)
            if df.empty or len(df) < 40:
                continue
            info = generate_signal(df)
            if info["signal"] == "BUY" and info["confidence"] >= MIN_CONFIDENCE:
                buy_picks.append((ticker, name, info))
            elif info["signal"] == "SELL" and info["confidence"] >= MIN_CONFIDENCE:
                sell_picks.append((ticker, name, info))
        except Exception as exc:
            log.warning("Intraday signal error %s: %s", ticker, exc)

    buy_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)
    sell_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)

    lines = [header]

    if not buy_picks and not sell_picks:
        if force_send:
            lines.append("⚪ No intraday signals right now. Market quiet.")
        else:
            log.info("No intraday signals — nothing to send.")
            return ""
        lines.append("")
        lines.append("📌 <i>Not SEBI-registered advice.</i>")
        return "\n".join(lines)

    if buy_picks:
        lines.append("🟢 <b>BUY NOW (Intraday)</b>")
        for i, (ticker, name, info) in enumerate(buy_picks[:5], 1):
            hold = hold_time("intraday", info["atr"], info["last_price"])
            mpt  = max_profit_time("intraday")
            lines.append(
                f"{i}. <b>{name}</b> ({ticker})  ₹{info['last_price']:,.2f}\n"
                f"   🟢 BUY | Hold: {hold} | Target by: {mpt}\n"
                f"   🎯 Target: {fmt_money(info['target'])}  |  🛑 SL: {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    if sell_picks:
        lines.append("🔴 <b>SELL NOW (Intraday)</b>")
        for i, (ticker, name, info) in enumerate(sell_picks[:4], 1):
            hold = hold_time("intraday", info["atr"], info["last_price"])
            mpt  = max_profit_time("intraday")
            lines.append(
                f"{i}. <b>{name}</b> ({ticker})  ₹{info['last_price']:,.2f}\n"
                f"   🔴 SELL | Hold: {hold} | Target by: {mpt}\n"
                f"   🎯 Target: {fmt_money(info['target'])}  |  🛑 SL: {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    lines.append("📌 <i>Not SEBI-registered advice.</i>")
    return "\n".join(lines)


def build_post_market_report(force_send: bool = False) -> str:
    """4:00 PM IST — day summary: top gainers/losers from watchlist."""
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
            results.append({"name": name, "last": last, "prev": prev, "chg": chg})
        except Exception as exc:
            log.warning("Post-market error %s: %s", ticker, exc)

    gainers = sorted([r for r in results if r["chg"] > 0],
                     key=lambda x: x["chg"], reverse=True)[:5]
    losers  = sorted([r for r in results if r["chg"] < 0],
                     key=lambda x: x["chg"])[:5]

    lines = [header]

    if not gainers and not losers:
        if force_send:
            lines.append("⚪ No data available for today.")
            lines.append("")
        else:
            log.info("No post-market data — nothing to send.")
            return ""

    if gainers:
        lines.append("📈 <b>TOP GAINERS</b>")
        for i, r in enumerate(gainers, 1):
            lines.append(
                f"{i}. <b>{r['name']}</b>  ₹{r['last']:,.2f}  <b>+{r['chg']:.1f}%</b>"
            )
        lines.append("")

    if losers:
        lines.append("📉 <b>TOP LOSERS</b>")
        for i, r in enumerate(losers, 1):
            lines.append(
                f"{i}. <b>{r['name']}</b>  ₹{r['last']:,.2f}  <b>{r['chg']:.1f}%</b>"
            )
        lines.append("")

    lines.append("📌 <i>Not SEBI-registered advice.</i>")
    return "\n".join(lines)


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

    force_send = os.environ.get("SEND_ALWAYS", "0").strip() == "1"

    if report_type == "pre":
        message = build_pre_market_report(force_send)
    elif report_type == "intraday":
        # Only send if market is open (prevents ghost messages on holidays)
        if not is_market_open():
            log.info("Market closed right now — skipping intraday send.")
            return
        message = build_intraday_report(force_send)
    elif report_type == "post":
        message = build_post_market_report(force_send)
    else:
        log.error("Unknown REPORT_TYPE: %s (must be pre/intraday/post)", report_type)
        sys.exit(1)

    try:
        send_telegram(message, bot_token, chat_ids)
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
