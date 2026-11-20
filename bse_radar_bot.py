"""
BSE Radar Bot — Production v2.0
Sends pre-market (9:00 AM IST) and post-market (4:00 PM IST) reports to Telegram.
Uses REAL BSE/NSE data via yfinance (.BO primary, .NS fallback). Runs on GitHub Actions — no PC required.

Env vars required:
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_ID    — group/channel chat ID (negative number for groups)
  REPORT_TYPE         — "pre" or "post"

Disclaimer: Technical analysis only. Not SEBI-registered investment advice. Trade at your own risk.
"""

import os
import sys
import logging
from datetime import datetime, date, timedelta

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
# Format: "BSE_TICKER": "Company Name"
WATCHLIST: dict[str, str] = {
    "RELIANCE.BO": "Reliance Industries",
    "TCS.BO": "TCS",
    "INFY.BO": "Infosys",
    "HDFCBANK.BO": "HDFC Bank",
    "SBIN.BO": "SBI",
    "AXISBANK.BO": "Axis Bank",
    "KOTAKBANK.BO": "Kotak Bank",
    "LT.BO": "L&T",
    "M&M.BO": "M&M",
    "HINDALCO.BO": "Hindalco",
    "ULTRACEMCO.BO": "UltraTech Cement",
    "DRREDDY.BO": "Dr Reddy's",
    "ASIANPAINT.BO": "Asian Paints",
    "MARUTI.BO": "Maruti Suzuki",
    "ONGC.BO": "ONGC",
    "COALINDIA.BO": "Coal India",
    "TITAN.BO": "Titan",
    "BHARTIARTL.BO": "Bharti Airtel",
    "ADANIENT.BO": "Adani Enterprises",
    "POWERGRID.BO": "Power Grid",
    "NESTLEIND.BO": "Nestle India",
    "HINDUNILVR.BO": "Hindustan Unilever",
    "ITC.BO": "ITC",
    "WIPRO.BO": "Wipro",
    "BAJFINANCE.BO": "Bajaj Finance",
}

# Index tickers for market context
NIFTY_TICKER = "^NSEI"
SENSEX_TICKER = "^BSESN"

# ── Signal thresholds ──────────────────────────────────────────────────────────
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
VOLUME_SURGE = 1.2          # volume must be >= 1.2x average to confirm
MOMENTUM_STRONG = 1.0       # % change over 5 days to count as strong momentum
MIN_CONFIDENCE_FOR_PICK = 55  # only show picks with confidence >= this


# ── Helpers ────────────────────────────────────────────────────────────────────

def now_ist() -> datetime:
    """Return current time in IST."""
    return datetime.now().astimezone(__import__("zoneinfo").ZoneInfo(IST))


def ist_today() -> date:
    """Return today's date in IST."""
    return now_ist().date()


def to_ist(ts) -> pd.Timestamp:
    """Convert a pandas Timestamp to IST (tz-aware)."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_convert(IST)
    return ts.tz_convert(IST)


def safe_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download yfinance data; return empty DataFrame on any failure."""
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, timeout=20)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as exc:
        log.warning("yfinance failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_with_fallback(base_ticker: str, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    """
    Try BSE (.BO) first, then NSE (.NS) fallback.
    BSE data via yfinance is often sparse, so a source is only accepted
    if it returns enough rows to compute indicators. BSE and NSE list the
    same stocks at nearly identical prices, so the .NS fallback keeps data
    accurate and matching what you see on BSE/AngleOne.
    Returns (dataframe, source_suffix_used).
    """
    min_rows = 25 if interval == "1d" else 40
    for suffix in (".BO", ".NS"):
        ticker = base_ticker.replace(".BO", "").replace(".NS", "") + suffix
        df = safe_download(ticker, period, interval)
        if not df.empty and len(df) >= min_rows:
            return df, suffix
    # Last resort: return whatever we got (even if sparse) so we don't skip silently
    for suffix in (".BO", ".NS"):
        ticker = base_ticker.replace(".BO", "").replace(".NS", "") + suffix
        df = safe_download(ticker, period, interval)
        if not df.empty:
            return df, suffix
    return pd.DataFrame(), ""


def last_close(df: pd.DataFrame) -> float:
    """Safely return the last close price."""
    close = df["Close"].squeeze()
    return float(close.iloc[-1])


# ── Technical indicators ───────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Return the latest RSI value; 50.0 if insufficient data."""
    series = _squeeze(series)
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


def compute_macd(series: pd.Series) -> tuple[float, float]:
    """Return (macd_line, signal_line). Both 0.0 on failure."""
    series = _squeeze(series)
    if len(series) < 26:
        return 0.0, 0.0
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return latest ATR (Average True Range) in absolute price points."""
    if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    val = atr.iloc[-1]
    return float(val) if not np.isnan(val) else 0.0


def _squeeze(series) -> pd.Series:
    """Ensure a plain 1-D Series (yfinance sometimes returns multi-index cols)."""
    if isinstance(series, pd.DataFrame):
        series = series.squeeze(axis="columns")
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series


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


def ema(series: pd.Series, span: int) -> pd.Series:
    series = _squeeze(series)
    return series.ewm(span=span, adjust=False).mean()


# ── Signal generation (confluence scoring, honest confidence) ──────────────────

SIGNALS = ("STRONG_BUY", "BUY", "WATCH", "SELL", "STRONG_SELL")


def generate_signal(df: pd.DataFrame) -> dict:
    """
    Confluence score 0-5 based on: RSI, MACD, EMA trend, volume, momentum.
    Returns dict with signal, confidence (honest, capped), rsi, macd_bull,
    vol_ratio, momentum_pct, last_price, atr, target, stop_loss.
    """
    close = _squeeze(df["Close"])
    last_price = float(close.iloc[-1])
    atr = compute_atr(df)

    rsi = compute_rsi(close)
    macd_line, macd_sig = compute_macd(close)
    macd_bull = macd_line > macd_sig

    vol = volume_ratio(df)
    mom = price_momentum(close, days=5)

    # EMA trend (20 vs 50)
    ema20 = float(ema(close, 20).iloc[-1]) if len(close) >= 20 else last_price
    ema50 = float(ema(close, 50).iloc[-1]) if len(close) >= 50 else last_price
    trend_bullish = ema20 > ema50
    trend_bearish = ema20 < ema50

    score = 0  # -3 to +3

    # RSI
    if rsi < RSI_OVERSOLD:
        score += 1
    elif rsi > RSI_OVERBOUGHT:
        score -= 1

    # MACD
    if macd_bull:
        score += 1
    else:
        score -= 1

    # EMA trend
    if trend_bullish:
        score += 1
    elif trend_bearish:
        score -= 1

    # Momentum
    if mom > MOMENTUM_STRONG:
        score += 1
    elif mom < -MOMENTUM_STRONG:
        score -= 1

    # Volume confirms direction (not a standalone signal)
    vol_ok = vol >= VOLUME_SURGE

    if score >= 3:
        signal = "STRONG_BUY"
    elif score == 2:
        signal = "BUY"
    elif score == 1:
        signal = "WATCH"
    elif score <= -3:
        signal = "STRONG_SELL"
    elif score == -2:
        signal = "SELL"
    else:
        signal = "WATCH"

    # Honest confidence: base 45, + per aligned factor, capped at 78.
    # Never overstate — technicals are probabilistic, not certain.
    aligned = sum([abs(rsi - 50) > 15, macd_line != 0, trend_bullish or trend_bearish,
                   vol_ok, abs(mom) > 0.5])
    confidence = min(78, 45 + aligned * 6)
    if signal == "WATCH":
        confidence = 40

    # ATR-based targets & stop-losses (realistic, not fixed 2%)
    if atr > 0 and last_price > 0:
        if "BUY" in signal:
            target = last_price + 1.5 * atr
            stop_loss = last_price - 1.0 * atr
        elif "SELL" in signal:
            target = last_price - 1.5 * atr
            stop_loss = last_price + 1.0 * atr
        else:
            target = last_price
            stop_loss = last_price
    else:
        if "BUY" in signal:
            target = last_price * 1.02
            stop_loss = last_price * 0.985
        elif "SELL" in signal:
            target = last_price * 0.98
            stop_loss = last_price * 1.015
        else:
            target = last_price
            stop_loss = last_price

    # Ensure SL < target for buys, SL > target for sells
    if "BUY" in signal and stop_loss >= target:
        stop_loss = last_price * 0.985
        target = last_price * 1.02
    if "SELL" in signal and stop_loss <= target:
        stop_loss = last_price * 1.015
        target = last_price * 0.98

    return {
        "signal": signal,
        "confidence": confidence,
        "rsi": rsi,
        "macd_bull": macd_bull,
        "vol_ratio": vol,
        "momentum_pct": mom,
        "last_price": last_price,
        "atr": atr,
        "target": round(target, 2),
        "stop_loss": round(stop_loss, 2),
        "trend_bullish": trend_bullish,
    }


# ── Index / market context ─────────────────────────────────────────────────────

def get_index_level(ticker: str, days: int = 5) -> tuple[float, float]:
    """Return (level, change_pct) for an index. Both 0.0 on failure."""
    try:
        # Index tickers (^NSEI, ^BSESN) are NOT BSE/NSE suffixed — fetch directly.
        df = safe_download(ticker, f"{days}d", "1d")
        if df.empty or len(df) < 2:
            return 0.0, 0.0
        close = df["Close"].squeeze()
        close = _squeeze(close)
        level = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        pct = (level - prev) / prev * 100 if prev != 0 else 0.0
        return level, pct
    except Exception as exc:
        log.warning("Index fetch failed for %s: %s", ticker, exc)
        return 0.0, 0.0


def get_nifty_level() -> tuple[float, float]:
    return get_index_level(NIFTY_TICKER)


def get_sensex_level() -> tuple[float, float]:
    return get_index_level(SENSEX_TICKER)


# ── Report builders ────────────────────────────────────────────────────────────

def arrow(val: float) -> str:
    return "▲" if val >= 0 else "▼"


def signal_emoji(signal: str) -> str:
    return {
        "STRONG_BUY": "🟢",
        "BUY": "🟢",
        "WATCH": "🟡",
        "SELL": "🔴",
        "STRONG_SELL": "🔴",
    }.get(signal, "⚪")


def fmt_money(val: float) -> str:
    return f"₹{val:,.2f}"


def build_pre_market_report() -> str:
    """Pre-market report: bias based on daily timeframe, sent at 9:00 AM IST."""
    today_str = ist_today().strftime("%d %b %Y")
    now = now_ist().strftime("%I:%M %p IST")

    nifty_level, nifty_pct = get_nifty_level()
    sensex_level, sensex_pct = get_sensex_level()

    buy_picks = []
    sell_picks = []
    watch_picks = []
    skipped = 0

    for ticker, name in WATCHLIST.items():
        try:
            df, source = fetch_with_fallback(ticker, "60d", "1d")
            if df.empty or len(df) < 25:
                skipped += 1
                log.warning("Skipping %s — insufficient data (source: %s)", ticker, source)
                continue
            info = generate_signal(df)
            if info["signal"] in ("STRONG_BUY", "BUY") and info["confidence"] >= MIN_CONFIDENCE_FOR_PICK:
                buy_picks.append((name, info))
            elif info["signal"] in ("STRONG_SELL", "SELL") and info["confidence"] >= MIN_CONFIDENCE_FOR_PICK:
                sell_picks.append((name, info))
            else:
                watch_picks.append((name, info))
        except Exception as exc:
            log.warning("Signal failed for %s: %s", ticker, exc)
            skipped += 1
            continue

    buy_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)
    sell_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)
    watch_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)

    lines = [
        f"🌅 <b>BSE PRE-MARKET RADAR | {today_str} | {now}</b>",
        "",
    ]

    # Market context
    ctx_lines = []
    if nifty_level > 0:
        ctx_lines.append(
            f"📊 <b>NIFTY50:</b> {nifty_level:,.0f} "
            f"{arrow(nifty_pct)} ({nifty_pct:+.2f}%)"
        )
    if sensex_level > 0:
        ctx_lines.append(
            f"📊 <b>SENSEX:</b> {sensex_level:,.0f} "
            f"{arrow(sensex_pct)} ({sensex_pct:+.2f}%)"
        )
    if ctx_lines:
        lines.extend(ctx_lines)
        lines.append("")

    if buy_picks:
        lines.append("🟢 <b>TOP BUY PICKS TODAY</b>")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        for i, (name, info) in enumerate(buy_picks[:6], 1):
            macd_txt = "MACD ✅" if info["macd_bull"] else "MACD ⚠️"
            rsi_txt = f"RSI {info['rsi']:.0f}"
            vol_txt = f"Vol {info['vol_ratio']:.1f}x"
            mom_txt = f"5d {info['momentum_pct']:+.1f}%"
            lines.append(
                f"{i}. <b>{name}</b> | {fmt_money(info['last_price'])} | "
                f"{info['signal']} | Conf: {info['confidence']}%"
            )
            lines.append(f"   {rsi_txt} · {vol_txt} · {macd_txt} · {mom_txt}")
            lines.append(
                f"   🎯 Target: {fmt_money(info['target'])} | "
                f"SL: {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    if sell_picks:
        lines.append("🔴 <b>SELL / AVOID TODAY</b>")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        for i, (name, info) in enumerate(sell_picks[:4], 1):
            rsi_txt = f"RSI {info['rsi']:.0f}"
            mom_txt = f"5d {info['momentum_pct']:+.1f}%"
            lines.append(
                f"{i}. <b>{name}</b> | {fmt_money(info['last_price'])} | "
                f"{info['signal']} | Conf: {info['confidence']}%"
            )
            lines.append(f"   ↘ {rsi_txt} · weakness {mom_txt}")
        lines.append("")

    if watch_picks and not buy_picks and not sell_picks:
        lines.append("🟡 <b>WATCH LIST (no strong signal)</b>")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        for i, (name, info) in enumerate(watch_picks[:5], 1):
            lines.append(
                f"{i}. <b>{name}</b> | {fmt_money(info['last_price'])} | "
                f"{info['signal']} | Conf: {info['confidence']}%"
            )
        lines.append("")

    if not buy_picks and not sell_picks and not watch_picks:
        lines.append("⚪ No strong signals today — market appears neutral.")
        lines.append("")

    # Mood
    bull_count = len(buy_picks)
    bear_count = len(sell_picks)
    if bull_count >= bear_count * 2 and bull_count > 0:
        mood = "Bullish 📈"
    elif bear_count >= bull_count * 2 and bear_count > 0:
        mood = "Bearish 📉"
    else:
        mood = "Cautiously Mixed ↔️"

    lines += [
        f"⚠️ <b>MARKET MOOD:</b> {mood}",
        "",
        "📊 Data: Real BSE/NSE prices (yfinance) · Targets/SL: ATR-based",
        "📌 <i>Technical analysis only. Not SEBI-registered advice. Do your own research.</i>",
        "🤖 BSE Radar Bot | Pre-market 9:00 AM IST",
    ]

    return "\n".join(lines)


def build_post_market_report() -> str:
    """Post-market report: real intraday results, sent at 4:00 PM IST."""
    today_str = ist_today().strftime("%d %b %Y")
    now = now_ist().strftime("%I:%M %p IST")

    nifty_level, nifty_pct = get_nifty_level()
    sensex_level, sensex_pct = get_sensex_level()

    stock_results = []

    for ticker, name in WATCHLIST.items():
        try:
            df_intra, src = fetch_with_fallback(ticker, "5d", "5m")
            df_daily, _ = fetch_with_fallback(ticker, "5d", "1d")
            if df_intra.empty or df_daily.empty:
                log.warning("Skipping %s post-market (no data, src=%s)", ticker, src)
                continue

            today = pd.Timestamp(ist_today())
            if df_intra.index.tz is None:
                today_intra = df_intra[df_intra.index.tz_localize("UTC").tz_convert(IST).normalize() == today]
            else:
                today_intra = df_intra[df_intra.index.tz_convert(IST).normalize() == today]

            if today_intra.empty:
                # Market may be closed today or no intraday yet — use last session
                today_intra = df_intra.tail(78)

            close_col = _squeeze(today_intra["Close"])
            open_col = _squeeze(today_intra["Open"])
            high_col = _squeeze(today_intra["High"])
            low_col = _squeeze(today_intra["Low"])
            open_price = float(open_col.iloc[0])
            close_price = float(close_col.iloc[-1])
            day_change_pct = (close_price - open_price) / open_price * 100 if open_price else 0.0

            vol_today = float(today_intra["Volume"].sum()) if "Volume" in today_intra.columns else 0.0
            vol_avg = float(df_daily["Volume"].mean()) if "Volume" in df_daily.columns else 0.0

            high_price = float(high_col.max()) if len(high_col) else close_price
            low_price = float(low_col.min()) if len(low_col) else close_price

            stock_results.append({
                "name": name,
                "ticker": ticker,
                "source": src,
                "open": open_price,
                "close": close_price,
                "high": high_price,
                "low": low_price,
                "change_pct": day_change_pct,
                "volume": vol_today,
                "vol_avg": vol_avg,
            })
        except Exception as exc:
            log.warning("Post-market calc failed for %s: %s", ticker, exc)
            continue

    gainers = sorted([s for s in stock_results if s["change_pct"] > 0],
                     key=lambda x: x["change_pct"], reverse=True)[:6]
    losers = sorted([s for s in stock_results if s["change_pct"] < 0],
                    key=lambda x: x["change_pct"])[:4]

    lines = [
        f"🔔 <b>BSE POST-MARKET REPORT | {today_str} | {now}</b>",
        "",
    ]

    # Market context
    ctx_lines = []
    if nifty_level > 0:
        pts = nifty_pct / 100 * nifty_level
        ctx_lines.append(
            f"📊 <b>NIFTY50 CLOSE:</b> {nifty_level:,.0f} "
            f"{arrow(nifty_pct)} {abs(pts):,.0f} pts ({nifty_pct:+.2f}%)"
        )
    if sensex_level > 0:
        pts = sensex_pct / 100 * sensex_level
        ctx_lines.append(
            f"📊 <b>SENSEX CLOSE:</b> {sensex_level:,.0f} "
            f"{arrow(sensex_pct)} {abs(pts):,.0f} pts ({sensex_pct:+.2f}%)"
        )
    if ctx_lines:
        lines.extend(ctx_lines)
        lines.append("")

    if gainers:
        lines.append("🏆 <b>TOP GAINERS TODAY</b>")
        lines.append("━━━━━━━━━━━━━━━━━━")
        for i, s in enumerate(gainers, 1):
            lines.append(
                f"{i}. <b>{s['name']}</b> +{s['change_pct']:.1f}% "
                f"| {fmt_money(s['open'])} → {fmt_money(s['close'])}"
            )
        lines.append("")

    if losers:
        lines.append("📉 <b>TOP LOSERS TODAY</b>")
        lines.append("━━━━━━━━━━━━━━━━━━")
        for i, s in enumerate(losers, 1):
            lines.append(
                f"{i}. <b>{s['name']}</b> {s['change_pct']:.1f}% "
                f"| {fmt_money(s['open'])} → {fmt_money(s['close'])}"
            )
        lines.append("")

    if not gainers and not losers:
        lines.append("⚪ No significant moves detected today.")
        lines.append("")

    # Overall sentiment
    total = len(stock_results)
    up_count = len([s for s in stock_results if s["change_pct"] > 0])
    down_count = total - up_count
    if total > 0:
        lines.append(
            f"📊 <b>ADVANCE/DECLINE (watchlist):</b> "
            f"{up_count} up ↑ | {down_count} down ↓"
        )
        lines.append("")

    lines += [
        "📊 Data: Real BSE/NSE intraday prices (yfinance)",
        "📌 <i>Past performance does not guarantee future results. Not SEBI-registered advice.</i>",
        "🤖 BSE Radar Bot | Post-market 4:00 PM IST",
    ]

    return "\n".join(lines)


# ── Telegram sender ────────────────────────────────────────────────────────────

def send_telegram(message: str, bot_token: str, chat_ids: list[str]) -> None:
    """Send the same message to multiple Telegram chat IDs."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    for chat_id in chat_ids:
        payload = {
            "chat_id": chat_id.strip(),
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        log.info("Message sent to chat %s. ok=%s", chat_id, resp.json().get("ok"))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_ids_raw = os.environ.get("TELEGRAM_CHAT_IDS", "")
    chat_id_single = os.environ.get("TELEGRAM_CHAT_ID", "")
    report_type = os.environ.get("REPORT_TYPE", "pre").strip().lower()

    if not bot_token:
        log.error("TELEGRAM_BOT_TOKEN must be set.")
        sys.exit(1)

    # Build list of chat IDs (comma-separated, supports both old and new secret names)
    chat_ids = []
    if chat_ids_raw:
        chat_ids.extend([c.strip() for c in chat_ids_raw.split(",") if c.strip()])
    if chat_id_single:
        chat_ids.append(chat_id_single.strip())

    if not chat_ids:
        log.error("TELEGRAM_CHAT_IDS (or TELEGRAM_CHAT_ID) must be set.")
        sys.exit(1)

    log.info("Building %s-market report for %d chat(s) (IST time: %s)...",
             report_type, len(chat_ids), now_ist().strftime("%Y-%m-%d %I:%M %p"))

    try:
        if report_type == "pre":
            message = build_pre_market_report()
        elif report_type == "post":
            message = build_post_market_report()
        else:
            log.error("REPORT_TYPE must be 'pre' or 'post', got: %s", report_type)
            sys.exit(1)
    except Exception as exc:
        log.exception("Report build failed: %s", exc)
        error_msg = (
            f"⚠️ <b>BSE Radar Bot Error</b>\n"
            f"Could not build {report_type}-market report.\n"
            f"Error: {exc}\n"
            f"Time: {now_ist().strftime('%Y-%m-%d %I:%M %p IST')}"
        )
        try:
            send_telegram(error_msg, bot_token, chat_ids)
        except Exception:
            pass
        sys.exit(1)

    try:
        send_telegram(message, bot_token, chat_ids)
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
