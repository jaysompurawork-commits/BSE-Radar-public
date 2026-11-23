"""
BSE Radar Bot — Production v3.0
Sends pre-market (9:00 AM IST), intraday updates (every 10 min during market
hours), and post-market (4:00 PM IST) reports to Telegram.
Uses REAL BSE/NSE data via yfinance (.BO primary, .NS fallback). Runs on GitHub Actions.

Env vars required:
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_ID    — group/channel chat ID (negative number for groups)
  REPORT_TYPE         — "pre", "intraday", or "post"

Optional (for committing prediction log back to repo so accuracy persists):
  GITHUB_TOKEN        — auto-provided by GitHub Actions
  GITHUB_REPOSITORY   — auto-provided by GitHub Actions

Disclaimer: Technical analysis only. Not SEBI-registered investment advice.
Trade at your own risk.
"""

import os
import sys
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta

import requests
import numpy as np
import pandas as pd
import yfinance as yf

IST = "Asia/Kolkata"

# Silence yfinance's noisy per-ticker stderr logs (its own logger + urllib3).
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("yfinance.tickers").setLevel(logging.ERROR)
try:
    import urllib3
    urllib3.disable_warnings()
except Exception:
    pass

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

# ── Accuracy tracking ──────────────────────────────────────────────────────────
# Logs each prediction so we can measure real accuracy over time.
# We commit this back to the repo after each run so it persists across
# GitHub Actions runs (each run gets a fresh checkout otherwise).
ACCURACY_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "prediction_log.jsonl")


def log_prediction(ticker: str, name: str, signal: str, confidence: int,
                   last_price: float, target: float, stop_loss: float,
                   report_type: str) -> None:
    """Append a prediction record to the local JSONL accuracy log.

    Each prediction is appended (pre-market and intraday can both log the
    same ticker at different price levels). Duplicate counting is handled at
    evaluation time in evaluate_open_predictions() by keeping the earliest
    record per ticker per day, so accuracy stats stay clean.
    """
    record = {
        "ticker": ticker,
        "name": name,
        "signal": signal,
        "confidence": confidence,
        "last_price": last_price,
        "target": target,
        "stop_loss": stop_loss,
        "report_type": report_type,
        "date": ist_today().isoformat(),
        "timestamp": now_ist().isoformat(),
        "outcome": None,          # "WIN" / "LOSS" / "OPEN"
        "outcome_price": None,    # price at which target or SL was hit
    }
    try:
        # Skip if an identical prediction (ticker + date + signal + levels)
        # already exists today — prevents intraday ticks from re-logging the
        # same pick every 10 minutes while allowing genuinely different
        # pre-market vs intraday predictions to coexist.
        today = ist_today().isoformat()
        try:
            existing = load_predictions()
            for r in existing:
                if (r.get("ticker") == ticker and r.get("date") == today
                        and r.get("signal") == signal
                        and abs(float(r.get("target", 0) or 0) - target) < 0.01
                        and abs(float(r.get("stop_loss", 0) or 0) - stop_loss) < 0.01):
                    log.debug("Skipping duplicate prediction log for %s", ticker)
                    return
        except Exception:
            pass
        with open(ACCURACY_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.warning("Could not write prediction log: %s", exc)


def load_predictions(days: int | None = None) -> list[dict]:
    """Load prediction records from the accuracy log. Optional day filter."""
    records = []
    if not os.path.exists(ACCURACY_LOG_FILE):
        return records
    try:
        with open(ACCURACY_LOG_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        log.warning("Could not read prediction log: %s", exc)
    if days is not None:
        cutoff = ist_today() - timedelta(days=days)
        filtered = []
        for r in records:
            try:
                r_date = date.fromisoformat(r.get("date", ""))
                if r_date >= cutoff:
                    filtered.append(r)
            except ValueError:
                continue
        records = filtered
    return records


def commit_and_push_log() -> None:
    """Commit prediction_log.jsonl back to the repo so accuracy persists across runs."""
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        log.debug("Skipping git commit — GITHUB_TOKEN or GITHUB_REPOSITORY not set")
        return
    if not os.path.exists(ACCURACY_LOG_FILE):
        return
    try:
        subprocess.run(["git", "config", "user.name", "GitHub Actions"],
                       check=False, capture_output=True)
        subprocess.run(["git", "config", "user.email", "github-actions@github.com"],
                       check=False, capture_output=True)
        subprocess.run(["git", "add", ACCURACY_LOG_FILE],
                       check=False, capture_output=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"],
                                capture_output=True)
        if result.returncode == 0:
            log.info("No changes to prediction log — nothing to commit")
            return
        subprocess.run(["git", "commit", "-m", f"Update prediction log {ist_today().isoformat()}"],
                       check=False, capture_output=True)
        # Push using token-based auth
        remote_url = f"https://x-access-token:{token}@github.com/{repo}.git"
        subprocess.run(["git", "push", remote_url, "HEAD:main"],
                       check=False, capture_output=True)
        log.info("Prediction log committed and pushed.")
    except Exception as exc:
        log.warning("Could not commit/push prediction log: %s", exc)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_predictions(records: list[dict]) -> None:
    """Rewrite the entire prediction log (used to persist resolved outcomes)."""
    try:
        with open(ACCURACY_LOG_FILE, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
    except Exception as exc:
        log.warning("Could not rewrite prediction log: %s", exc)


def now_ist() -> datetime:
    """Return current time in IST."""
    return datetime.now().astimezone(__import__("zoneinfo").ZoneInfo(IST))


def ist_today() -> date:
    """Return today's date in IST."""
    return now_ist().date()


def is_market_open() -> bool:
    """Return True if BSE/NSE is currently open (Mon-Fri 9:15 AM–3:30 PM IST)."""
    now = now_ist()
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def to_ist(ts) -> pd.Timestamp:
    """Convert a pandas Timestamp to IST (tz-aware)."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_convert(IST)
    return ts.tz_convert(IST)


def safe_download(ticker: str, period: str, interval: str, retries: int = 1) -> pd.DataFrame:
    """Download yfinance data with retries; return empty DataFrame on failure.

    Fails fast: only 2 attempts with a short sleep, so a single flaky ticker
    doesn't hold up the whole report.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, timeout=15)
            if df is None or df.empty:
                if attempt < retries:
                    time.sleep(1.0)
                    continue
                return pd.DataFrame()
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0)
                continue
            return pd.DataFrame()
    return pd.DataFrame()


def fetch_with_fallback(base_ticker: str, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    """
    Try BSE (.BO) first, then NSE (.NS) fallback.
    Returns (dataframe, source_suffix_used).
    """
    min_rows = 25 if interval == "1d" else 40
    for suffix in (".BO", ".NS"):
        ticker = base_ticker.replace(".BO", "").replace(".NS", "") + suffix
        df = safe_download(ticker, period, interval)
        if not df.empty and len(df) >= min_rows:
            return df, suffix
    for suffix in (".BO", ".NS"):
        ticker = base_ticker.replace(".BO", "").replace(".NS", "") + suffix
        df = safe_download(ticker, period, interval)
        if not df.empty:
            return df, suffix
    return pd.DataFrame(), ""


def fetch_all(tickers: list[str], period: str, interval: str,
              max_workers: int = 8) -> dict[str, tuple[pd.DataFrame, str]]:
    """
    Fetch data for many tickers concurrently. Returns {ticker: (df, suffix)}.
    Skips tickers that fail or return no data.
    """
    results: dict[str, tuple[pd.DataFrame, str]] = {}

    def _one(ticker: str):
        try:
            return ticker, fetch_with_fallback(ticker, period, interval)
        except Exception as exc:
            log.warning("Concurrent fetch failed for %s: %s", ticker, exc)
            return ticker, (pd.DataFrame(), "")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, t) for t in tickers]
        for fut in as_completed(futures):
            ticker, (df, suffix) = fut.result()
            if not df.empty:
                results[ticker] = (df, suffix)
    return results


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


def bb_position(df: pd.DataFrame, period: int = 20) -> tuple[float, float]:
    """Return (bb_position_0_to_1, bandwidth_pct). Position: 0=lower band, 1=upper band."""
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


def get_multi_timeframe_bias(ticker: str = "", df_w: pd.DataFrame | None = None,
                             df_d: pd.DataFrame | None = None) -> tuple[float, str]:
    """
    Return (bias_score, bias_text) from weekly + daily trend.
    Pass pre-fetched dataframes to avoid re-downloading; if not provided,
    they are fetched once per ticker.
    """
    try:
        if df_w is None or df_w.empty:
            df_w, _ = fetch_with_fallback(ticker, "1y", "1wk")
        if df_w is None or df_w.empty or len(df_w) < 10:
            return 0.0, "Neutral"
        close_w = _squeeze(df_w["Close"])
        ema20_w = float(ema(close_w, 20).iloc[-1]) if len(close_w) >= 20 else close_w.iloc[-1]
        weekly_bullish = float(close_w.iloc[-1]) > ema20_w

        if df_d is None or df_d.empty:
            df_d, _ = fetch_with_fallback(ticker, "60d", "1d")
        if df_d is None or df_d.empty or len(df_d) < 10:
            return 0.0, "Neutral"
        close_d = _squeeze(df_d["Close"])
        ema20_d = float(ema(close_d, 20).iloc[-1]) if len(close_d) >= 20 else close_d.iloc[-1]
        daily_bullish = float(close_d.iloc[-1]) > ema20_d

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


# ── Signal generation ──────────────────────────────────────────────────────────

SIGNALS = ("STRONG_BUY", "BUY", "WATCH", "SELL", "STRONG_SELL")


def generate_signal(df: pd.DataFrame, ticker: str = "",
                    bias: tuple[float, str] | None = None) -> dict:
    """
    Confluence scoring based on RSI, MACD, EMA trend, volume, momentum,
    Bollinger Bands, VWAP, and multi-timeframe bias.

    `bias` is an optional pre-computed (bias_score, bias_text) from
    get_multi_timeframe_bias(). When omitted it is fetched once per ticker.
    """
    close = _squeeze(df["Close"])
    last_price = float(close.iloc[-1])
    atr = compute_atr(df)

    rsi = compute_rsi(close)
    macd_line, macd_sig = compute_macd(close)
    macd_bull = macd_line > macd_sig

    vol = volume_ratio(df)
    mom = price_momentum(close, days=5)

    ema20 = float(ema(close, 20).iloc[-1]) if len(close) >= 20 else last_price
    ema50 = float(ema(close, 50).iloc[-1]) if len(close) >= 50 else last_price
    trend_bullish = ema20 > ema50
    trend_bearish = ema20 < ema50

    bb_pos, _ = bb_position(df)
    bb_bullish = bb_pos > 0.5
    bb_bearish = bb_pos < 0.5

    vwap_bull = vwap_confirm(df)

    bias_score, bias_text = bias if bias is not None else (
        get_multi_timeframe_bias(ticker) if ticker else (0.0, "Neutral"))

    score = 0  # -4 to +4
    if rsi < RSI_OVERSOLD:
        score += 1
    elif rsi > RSI_OVERBOUGHT:
        score -= 1
    if macd_bull:
        score += 1
    else:
        score -= 1
    if trend_bullish:
        score += 1
    elif trend_bearish:
        score -= 1
    if mom > MOMENTUM_STRONG:
        score += 1
    elif mom < -MOMENTUM_STRONG:
        score -= 1
    if bb_bullish:
        score += 1
    elif bb_bearish:
        score -= 1

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

    aligned = sum([
        abs(rsi - 50) > 15,
        macd_line != 0,
        trend_bullish or trend_bearish,
        vol_ok,
        abs(mom) > 0.5,
        abs(bb_pos - 0.5) > 0.3,
    ])
    confidence = min(75, 45 + aligned * 5)
    if abs(bias_score) >= 0.5:
        confidence = min(75, confidence + 5)
    if signal == "WATCH":
        confidence = 40

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
        "bb_pos": round(bb_pos, 2),
        "bias": bias_text,
        "bias_score": bias_score,
        "vwap_bull": vwap_bull,
    }


# ── Outcome checking for previous predictions ──────────────────────────────────

def check_prediction_outcome(record: dict, today_intra: pd.DataFrame) -> dict:
    """
    Given an old prediction and today's intraday data, determine if the target
    or stop-loss was hit during today's session. Returns an updated record copy.
    """
    result = dict(record)
    target = record.get("target")
    stop_loss = record.get("stop_loss")
    signal = record.get("signal", "")
    if target is None or stop_loss is None or today_intra.empty:
        result["outcome"] = "OPEN"
        return result

    high_col = _squeeze(today_intra["High"])
    low_col = _squeeze(today_intra["Low"])
    close_col = _squeeze(today_intra["Close"])

    hit_target = False
    hit_sl = False
    hit_price = None
    hit_time = None

    # Walk through today's bars in time order to see which level was hit first.
    for i in range(len(today_intra)):
        h = float(high_col.iloc[i])
        l = float(low_col.iloc[i])
        c = float(close_col.iloc[i])
        ts = today_intra.index[i]
        if "BUY" in signal:
            if h >= target:
                hit_target = True
                hit_price = target
                hit_time = ts
                break
            if l <= stop_loss:
                hit_sl = True
                hit_price = stop_loss
                hit_time = ts
                break
        elif "SELL" in signal:
            if l <= target:
                hit_target = True
                hit_price = target
                hit_time = ts
                break
            if h >= stop_loss:
                hit_sl = True
                hit_price = stop_loss
                hit_time = ts
                break

    if hit_target:
        result["outcome"] = "WIN"
    elif hit_sl:
        result["outcome"] = "LOSS"
    else:
        # Neither hit — use today's close to mark open
        result["outcome"] = "OPEN"
        result["outcome_price"] = float(close_col.iloc[-1])
    result["outcome_price"] = hit_price
    result["outcome_time"] = str(hit_time) if hit_time is not None else None
    return result


def evaluate_open_predictions() -> tuple[list[dict], int, int, int]:
    """
    Evaluate all predictions that still have outcome OPEN against the intraday
    bars of the day they were made. Returns (updated_records, wins, losses, open).
    Each prediction is checked against its own record date so a missed post-market
    run doesn't evaluate yesterday's picks against today's bars.
    """
    records = load_predictions()
    open_records = [r for r in records if r.get("outcome") in (None, "OPEN")]
    if not open_records:
        return [], 0, 0, 0

    wins = losses = still_open = 0
    updated = []
    all_records = load_predictions()  # full copy to rewrite with outcomes

    # Group open records by the date they were made.
    by_date: dict[str, list[dict]] = {}
    for r in open_records:
        by_date.setdefault(r.get("date", ""), []).append(r)

    # For each distinct date, fetch that day's 5m intraday data once per ticker.
    intraday_by_date: dict[str, dict[str, pd.DataFrame]] = {}
    for rec_date, recs in by_date.items():
        tickers = list({r["ticker"] for r in recs})
        fetched = fetch_all(tickers, "5d", "5m")
        per_ticker: dict[str, pd.DataFrame] = {}
        target = pd.Timestamp(rec_date)
        for ticker, (df, _suffix) in fetched.items():
            if df.empty:
                continue
            if df.index.tz is None:
                tod = df[df.index.tz_localize("UTC").tz_convert(IST).normalize() == target]
            else:
                tod = df[df.index.tz_convert(IST).normalize() == target]
            if not tod.empty:
                per_ticker[ticker] = tod
        intraday_by_date[rec_date] = per_ticker

    for rec_date, recs in by_date.items():
        tod_map = intraday_by_date.get(rec_date, {})
        for r in recs:
            ticker = r["ticker"]
            tod = tod_map.get(ticker, pd.DataFrame())
            try:
                r = check_prediction_outcome(r, tod)
                if r["outcome"] == "WIN":
                    wins += 1
                elif r["outcome"] == "LOSS":
                    losses += 1
                else:
                    still_open += 1
                updated.append(r)
            except Exception as exc:
                log.warning("Outcome check failed for %s (%s): %s", ticker, rec_date, exc)
                r["outcome"] = "OPEN"
                still_open += 1
                updated.append(r)

    # Persist resolved outcomes back to the log so they survive across runs.
    try:
        updated_key = {(r["ticker"], r.get("date", "")): r for r in updated}
        for i, r in enumerate(all_records):
            key = (r.get("ticker"), r.get("date", ""))
            if key in updated_key:
                all_records[i] = updated_key[key]
        _write_predictions(all_records)
    except Exception as exc:
        log.warning("Could not persist prediction outcomes: %s", exc)

    return updated, wins, losses, still_open


# ── Index / market context ─────────────────────────────────────────────────────


def get_index_level(ticker: str, days: int = 5) -> tuple[float, float]:
    """Return (level, change_pct) for an index. Both 0.0 on failure."""
    try:
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


def signal_label(signal: str) -> str:
    """Return a bold, visually dominant BUY/SELL label.

    Telegram's HTML mode doesn't support custom text colours, so BUY picks are
    framed with green circle emoji and SELL picks with red circle emoji, plus
    bold text — making them stand out as clearly as colour would.
    """
    emoji = signal_emoji(signal)
    label = f"{emoji} <b>{signal.replace('_', ' ')}</b> {emoji}"
    return label


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
    skipped = 0

    # Fetch daily data for all tickers concurrently (much faster than sequential).
    fetched = fetch_all(list(WATCHLIST.keys()), "60d", "1d")

    # Pre-fetch weekly data concurrently so multi-timeframe bias doesn't
    # re-download inside every generate_signal() call.
    weekly_fetched = fetch_all(list(WATCHLIST.keys()), "1y", "1wk")

    for ticker, name in WATCHLIST.items():
        try:
            result = fetched.get(ticker)
            if result is None:
                skipped += 1
                log.warning("Skipping %s — no data", ticker)
                continue
            df, source = result
            if len(df) < 25:
                skipped += 1
                log.warning("Skipping %s — insufficient data (source: %s)", ticker, source)
                continue

            # Compute bias from pre-fetched weekly + daily data (no extra downloads).
            wdf, _ = weekly_fetched.get(ticker, (pd.DataFrame(), ""))
            bias = get_multi_timeframe_bias(ticker, df_w=wdf, df_d=df)
            info = generate_signal(df, ticker, bias=bias)
            if info["signal"] in ("STRONG_BUY", "BUY") and info["confidence"] >= MIN_CONFIDENCE_FOR_PICK:
                buy_picks.append((name, info))
            elif info["signal"] in ("STRONG_SELL", "SELL") and info["confidence"] >= MIN_CONFIDENCE_FOR_PICK:
                sell_picks.append((name, info))
        except Exception as exc:
            log.warning("Signal failed for %s: %s", ticker, exc)
            skipped += 1
            continue

    buy_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)
    sell_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)

    # Log predictions for accuracy tracking
    for name, info in buy_picks + sell_picks:
        ticker_key = next((t for t, n in WATCHLIST.items() if n == name), "")
        log_prediction(ticker_key, name, info["signal"], info["confidence"],
                       info["last_price"], info["target"], info["stop_loss"],
                       "pre")

    lines = [
        f"🌅 <b>BSE PRE-MARKET | {today_str}</b>",
        "",
    ]

    if nifty_level > 0:
        lines.append(
            f"📊 NIFTY50: {nifty_level:,.0f} "
            f"{arrow(nifty_pct)} ({nifty_pct:+.2f}%)"
        )
    if sensex_level > 0:
        lines.append(
            f"📊 SENSEX: {sensex_level:,.0f} "
            f"{arrow(sensex_pct)} ({sensex_pct:+.2f}%)"
        )
    lines.append("")

    if buy_picks:
        lines.append("🟢 <b>BUY PICKS</b>")
        lines.append("━━━━━━━━━━━━")
        for i, (name, info) in enumerate(buy_picks[:5], 1):
            lines.append(
                f"{i}. <b>{name}</b> | {fmt_money(info['last_price'])} | "
                f"{signal_label(info['signal'])} | Conf {info['confidence']}% | "
                f"🎯 {fmt_money(info['target'])} | SL {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    if sell_picks:
        lines.append("🔴 <b>SELL / AVOID</b>")
        lines.append("━━━━━━━━━━━━")
        for i, (name, info) in enumerate(sell_picks[:4], 1):
            lines.append(
                f"{i}. <b>{name}</b> | {fmt_money(info['last_price'])} | "
                f"{signal_label(info['signal'])} | Conf {info['confidence']}% | "
                f"SL {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    if not buy_picks and not sell_picks:
        lines.append("⚪ No strong signals today — market neutral.")
        lines.append("")

    lines += [
        f"📊 Mood: {'Bullish 📈' if len(buy_picks) >= len(sell_picks) else 'Bearish 📉' if sell_picks else 'Mixed ↔️'}",
        "",
        "Not SEBI-registered advice.",
        "BSE Radar Bot | Pre-market",
    ]

    return "\n".join(lines)


def build_intraday_report() -> str:
    """Intraday update: BUY/SELL picks from reliable daily signals + live prices."""
    today_str = ist_today().strftime("%d %b %Y")
    now = now_ist().strftime("%I:%M %p IST")

    nifty_level, nifty_pct = get_nifty_level()
    sensex_level, sensex_pct = get_sensex_level()

    buy_picks = []
    sell_picks = []
    stock_results = []

    # Fetch daily data (for signals) and 5m data (for live price) concurrently.
    daily_data = fetch_all(list(WATCHLIST.keys()), "60d", "1d")
    intra_data = fetch_all(list(WATCHLIST.keys()), "5d", "5m")

    # Pre-fetch weekly data concurrently so multi-timeframe bias doesn't
    # re-download inside every generate_signal() call.
    weekly_data = fetch_all(list(WATCHLIST.keys()), "1y", "1wk")

    today = pd.Timestamp(ist_today())
    for ticker, name in WATCHLIST.items():
        try:
            result = daily_data.get(ticker)
            if result is None:
                continue
            df_daily, _ = result
            if len(df_daily) < 25:
                continue

            # Compute bias from pre-fetched weekly + daily data (no extra downloads).
            wdf, _ = weekly_data.get(ticker, (pd.DataFrame(), ""))
            bias = get_multi_timeframe_bias(ticker, df_w=wdf, df_d=df_daily)
            info = generate_signal(df_daily, ticker, bias=bias)

            # 5m data only for current price / today's change display.
            close_price = info["last_price"]
            change_pct = 0.0
            df_intra, _ = intra_data.get(ticker, (pd.DataFrame(), ""))
            if not df_intra.empty:
                if df_intra.index.tz is None:
                    tod = df_intra[df_intra.index.tz_localize("UTC").tz_convert(IST).normalize() == today]
                else:
                    tod = df_intra[df_intra.index.tz_convert(IST).normalize() == today]
                if not tod.empty and len(tod) >= 2:
                    close_price = float(_squeeze(tod["Close"]).iloc[-1])
                    open_price = float(_squeeze(tod["Open"]).iloc[0])
                    change_pct = (close_price - open_price) / open_price * 100 if open_price else 0.0

            stock_results.append({
                "name": name,
                "ticker": ticker,
                "close": close_price,
                "change_pct": change_pct,
            })

            # Same thresholds as pre-market so picks are consistent and accurate.
            if info["signal"] in ("STRONG_BUY", "BUY") and info["confidence"] >= MIN_CONFIDENCE_FOR_PICK:
                buy_picks.append((name, info))
            elif info["signal"] in ("STRONG_SELL", "SELL") and info["confidence"] >= MIN_CONFIDENCE_FOR_PICK:
                sell_picks.append((name, info))
        except Exception as exc:
            log.warning("Intraday signal failed for %s: %s", ticker, exc)
            continue

    buy_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)
    sell_picks.sort(key=lambda x: x[1]["confidence"], reverse=True)

    # Log intraday predictions for accuracy tracking.
    for name, info in buy_picks + sell_picks:
        ticker_key = next((t for t, n in WATCHLIST.items() if n == name), "")
        log_prediction(ticker_key, name, info["signal"], info["confidence"],
                       info["last_price"], info["target"], info["stop_loss"],
                       "intraday")

    stock_results.sort(key=lambda x: x["change_pct"], reverse=True)
    top_gainers = stock_results[:3]
    top_losers = stock_results[-3:][::-1]

    lines = [
        f"⚡ <b>BSE INTRADAY | {today_str} | {now}</b> "
        f"| {'Market Open 📈' if is_market_open() else 'Market Closed 🔒'}",
        "",
    ]
    if nifty_level > 0:
        lines.append(
            f"📊 NIFTY50: {nifty_level:,.0f} {arrow(nifty_pct)} ({nifty_pct:+.2f}%)"
        )
    if sensex_level > 0:
        lines.append(
            f"📊 SENSEX: {sensex_level:,.0f} {arrow(sensex_pct)} ({sensex_pct:+.2f}%)"
        )
    lines.append("")

    if buy_picks:
        lines.append("🟢 <b>BUY PICKS</b>")
        lines.append("━━━━━━━━━━━━")
        for i, (name, info) in enumerate(buy_picks[:5], 1):
            lines.append(
                f"{i}. <b>{name}</b> | {fmt_money(info['last_price'])} | "
                f"{signal_label(info['signal'])} | Conf {info['confidence']}% | "
                f"🎯 {fmt_money(info['target'])} | SL {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    if sell_picks:
        lines.append("🔴 <b>SELL / AVOID</b>")
        lines.append("━━━━━━━━━━━━")
        for i, (name, info) in enumerate(sell_picks[:4], 1):
            lines.append(
                f"{i}. <b>{name}</b> | {fmt_money(info['last_price'])} | "
                f"{signal_label(info['signal'])} | Conf {info['confidence']}% | "
                f"SL {fmt_money(info['stop_loss'])}"
            )
        lines.append("")

    if not buy_picks and not sell_picks:
        lines.append("⚪ No strong signals today — market neutral.")
        lines.append("")

    if top_gainers:
        lines.append("🟢 <b>TOP GAINERS</b>")
        lines.append("━━━━━━━━━━━━")
        for i, s in enumerate(top_gainers, 1):
            lines.append(
                f"{i}. <b>{s['name']}</b> +{s['change_pct']:.1f}% "
                f"| {fmt_money(s['close'])}"
            )
        lines.append("")

    if top_losers:
        lines.append("🔴 <b>TOP LOSERS</b>")
        lines.append("━━━━━━━━━━━━")
        for i, s in enumerate(top_losers, 1):
            lines.append(
                f"{i}. <b>{s['name']}</b> {s['change_pct']:.1f}% "
                f"| {fmt_money(s['close'])}"
            )
        lines.append("")

    lines += [
        "Not SEBI-registered advice.",
        "BSE Radar Bot | Intraday update",
    ]

    return "\n".join(lines)


def build_post_market_report() -> str:
    """Post-market report: real intraday results + accuracy of previous predictions."""
    today_str = ist_today().strftime("%d %b %Y")
    now = now_ist().strftime("%I:%M %p IST")

    nifty_level, nifty_pct = get_nifty_level()
    sensex_level, sensex_pct = get_sensex_level()

    stock_results = []
    intra_data = fetch_all(list(WATCHLIST.keys()), "5d", "5m")

    today = pd.Timestamp(ist_today())
    for ticker, name in WATCHLIST.items():
        try:
            df_intra, src = intra_data.get(ticker, (pd.DataFrame(), ""))
            if df_intra.empty:
                log.warning("Skipping %s post-market (no data, src=%s)", ticker, src)
                continue

            if df_intra.index.tz is None:
                today_intra = df_intra[df_intra.index.tz_localize("UTC").tz_convert(IST).normalize() == today]
            else:
                today_intra = df_intra[df_intra.index.tz_convert(IST).normalize() == today]

            if today_intra.empty:
                today_intra = df_intra.tail(78)

            close_col = _squeeze(today_intra["Close"])
            open_col = _squeeze(today_intra["Open"])
            high_col = _squeeze(today_intra["High"])
            low_col = _squeeze(today_intra["Low"])
            open_price = float(open_col.iloc[0])
            close_price = float(close_col.iloc[-1])
            day_change_pct = (close_price - open_price) / open_price * 100 if open_price else 0.0

            stock_results.append({
                "name": name,
                "ticker": ticker,
                "source": src,
                "open": open_price,
                "close": close_price,
                "high": float(high_col.max()) if len(high_col) else close_price,
                "low": float(low_col.min()) if len(low_col) else close_price,
                "change_pct": day_change_pct,
            })
        except Exception as exc:
            log.warning("Post-market calc failed for %s: %s", ticker, exc)
            continue

    gainers = sorted([s for s in stock_results if s["change_pct"] > 0],
                     key=lambda x: x["change_pct"], reverse=True)[:5]
    losers = sorted([s for s in stock_results if s["change_pct"] < 0],
                    key=lambda x: x["change_pct"])[:4]

    lines = [
        f"🔔 <b>BSE POST-MARKET | {today_str}</b>",
        "",
    ]
    if nifty_level > 0:
        pts = nifty_pct / 100 * nifty_level
        lines.append(
            f"📊 NIFTY50: {nifty_level:,.0f} "
            f"{arrow(nifty_pct)} {abs(pts):,.0f} pts ({nifty_pct:+.2f}%)"
        )
    if sensex_level > 0:
        pts = sensex_pct / 100 * sensex_level
        lines.append(
            f"📊 SENSEX: {sensex_level:,.0f} "
            f"{arrow(sensex_pct)} {abs(pts):,.0f} pts ({sensex_pct:+.2f}%)"
        )
    lines.append("")

    if gainers:
        lines.append("🏆 <b>GAINERS</b>")
        lines.append("━━━━━━━━━━━━")
        for i, s in enumerate(gainers, 1):
            lines.append(
                f"{i}. <b>{s['name']}</b> +{s['change_pct']:.1f}% "
                f"| {fmt_money(s['open'])} → {fmt_money(s['close'])}"
            )
        lines.append("")

    if losers:
        lines.append("📉 <b>LOSERS</b>")
        lines.append("━━━━━━━━━━━━")
        for i, s in enumerate(losers, 1):
            lines.append(
                f"{i}. <b>{s['name']}</b> {s['change_pct']:.1f}% "
                f"| {fmt_money(s['open'])} → {fmt_money(s['close'])}"
            )
        lines.append("")

    if not gainers and not losers:
        lines.append("⚪ No significant moves today.")
        lines.append("")

    total = len(stock_results)
    up_count = len([s for s in stock_results if s["change_pct"] > 0])
    down_count = total - up_count
    if total > 0:
        lines.append(f"📊 {up_count} up ↑ | {down_count} down ↓")
        lines.append("")

    # ── Accuracy section: how did previous predictions do? ──
    lines.append("📊 <b>ACCURACY — PREDICTION OUTCOMES</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    try:
        updated, wins, losses, still_open = evaluate_open_predictions()
        total_resolved = wins + losses
        acc = (wins / total_resolved * 100) if total_resolved > 0 else 0.0
        lines.append(
            f"🟢 <b>Wins: {wins}</b>   🔴 <b>Losses: {losses}</b>   ⏳ Open: {still_open}"
        )
        lines.append(f"🎯 Accuracy: {acc:.0f}% ({wins}/{total_resolved})")

        # Multi-day accuracy trend (last 7 days).
        all_records = load_predictions(days=7)
        resolved_week = [r for r in all_records if r.get("outcome") in ("WIN", "LOSS")]
        if resolved_week:
            week_wins = sum(1 for r in resolved_week if r["outcome"] == "WIN")
            week_acc = week_wins / len(resolved_week) * 100
            lines.append(
                f"📅 7-day: {week_acc:.0f}% ({week_wins}/{len(resolved_week)})"
            )

        resolved = [r for r in updated if r["outcome"] in ("WIN", "LOSS")]
        if resolved:
            lines.append("")
            lines.append("📋 <b>RESOLVED TODAY</b>")
            lines.append("━━━━━━━━━━━━")
            for r in resolved[:8]:
                emoji = "✅" if r["outcome"] == "WIN" else "❌"
                sig = signal_label(r["signal"])
                lines.append(
                    f"{emoji} <b>{r['name']}</b> | {sig} | "
                    f"Target ₹{r.get('target', 0):,.2f} / SL ₹{r.get('stop_loss', 0):,.2f} | "
                    f"Hit ₹{r.get('outcome_price', 0):,.2f}"
                )
    except Exception as exc:
        log.warning("Accuracy evaluation failed: %s", exc)
        lines.append("⏳ Could not evaluate prediction outcomes today.")
    lines.append("")
    lines += [
        "Not SEBI-registered advice.",
        "BSE Radar Bot | Post-market",
    ]

    return "\n".join(lines)


# ── Telegram sender ────────────────────────────────────────────────────────────

# Telegram message length limit (Telegram caps at 4096 chars per message).
TELEGRAM_MAX_CHARS = 4000


def split_message(message: str, max_chars: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split a long message into chunks that respect the Telegram limit,
    preferring to break at newlines so lines aren't cut mid-way."""
    if len(message) <= max_chars:
        return [message]
    chunks = []
    current = ""
    for line in message.split("\n"):
        # Account for the joining newline.
        add = line if not current else "\n" + line
        if len(current) + len(add) > max_chars:
            if current:
                chunks.append(current)
                current = line
            else:
                # Single line longer than limit — split it forcibly.
                chunks.append(line[:max_chars])
                current = line[max_chars:]
        else:
            current += add
    if current:
        chunks.append(current)
    return chunks


def send_telegram(message: str, bot_token: str, chat_ids: list[str]) -> None:
    """Send the same message to multiple Telegram chat IDs, splitting long
    messages into chunks that fit Telegram's 4096-char limit."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    chunks = split_message(message)
    for chat_id in chat_ids:
        for idx, chunk in enumerate(chunks, 1):
            text = chunk
            if len(chunks) > 1:
                text = f"({idx}/{len(chunks)})\n{chunk}"
            payload = {
                "chat_id": chat_id.strip(),
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            log.info("Message sent to chat %s (chunk %d/%d). ok=%s",
                     chat_id, idx, len(chunks), resp.json().get("ok"))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_ids_raw = os.environ.get("TELEGRAM_CHAT_IDS", "")
    chat_id_single = os.environ.get("TELEGRAM_CHAT_ID", "")
    report_type = os.environ.get("REPORT_TYPE", "pre").strip().lower()

    if not bot_token:
        log.error("TELEGRAM_BOT_TOKEN must be set.")
        sys.exit(1)

    chat_ids = []
    seen = set()
    if chat_ids_raw:
        for c in chat_ids_raw.split(","):
            c = c.strip()
            if c and c not in seen:
                chat_ids.append(c)
                seen.add(c)
    if chat_id_single:
        chat_id_single = chat_id_single.strip()
        if chat_id_single and chat_id_single not in seen:
            chat_ids.append(chat_id_single)
            seen.add(chat_id_single)

    if not chat_ids:
        log.error("TELEGRAM_CHAT_IDS (or TELEGRAM_CHAT_ID) must be set.")
        sys.exit(1)

    log.info("Building %s-market report for %d chat(s) (IST time: %s)...",
             report_type, len(chat_ids), now_ist().strftime("%Y-%m-%d %I:%M %p"))

    try:
        if report_type == "pre":
            message = build_pre_market_report()
        elif report_type == "intraday":
            message = build_intraday_report()
        elif report_type == "post":
            message = build_post_market_report()
        else:
            log.error("REPORT_TYPE must be 'pre', 'intraday', or 'post', got: %s", report_type)
            sys.exit(1)
    except Exception as exc:
        log.exception("Report build failed: %s", exc)
        error_msg = (
            f"⚠️ <b>BSE Radar Bot Error</b>\n"
            f"Could not build {report_type} report.\n"
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

    # Persist prediction log so accuracy carries across GitHub Actions runs.
    # Only commit on pre/post runs (not every intraday tick).
    if report_type in ("pre", "post"):
        commit_and_push_log()

    log.info("Done.")


if __name__ == "__main__":
    main()
