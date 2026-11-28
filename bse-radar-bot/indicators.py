"""
indicators.py — BSE Radar Bot v7.0
Professional technical indicator library.

Includes: EMA/SMA, RSI, MACD, ATR, Bollinger Bands, VWAP, Stochastic,
Williams %R, CCI, ROC, OBV, ADX, Parabolic SAR, Ichimoku, Supertrend,
Pivot Points, Donchian Channels, Keltner Channels, MFI, Chaikin Money Flow,
Elder Force Index, Hull EMA, Volatility (ATR %), and multi-timeframe bias.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("bse_radar")

# ── Re-export helpers used across modules ─────────────────────────────────────
__all__ = [
    "_squeeze", "compute_rsi", "compute_macd", "compute_atr", "ema_series",
    "volume_ratio", "price_momentum", "bb_position", "vwap_confirm",
    "support_resistance", "compute_stochastic", "compute_williams_r",
    "compute_cci", "compute_roc", "compute_obv", "compute_adx",
    "compute_parabolic_sar", "compute_ichimoku", "compute_supertrend",
    "compute_pivots", "compute_donchian", "compute_keltner",
    "compute_mfi", "compute_cmf", "compute_efi", "compute_hull_ema",
    "compute_volatility_pct",
]


def _squeeze(s) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        s = s.squeeze(axis="columns")
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def safe_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, timeout=25)
        return pd.DataFrame() if df is None or df.empty else df
    except Exception as exc:
        log.warning("yfinance %s %s/%s: %s", ticker, period, interval, exc)
        return pd.DataFrame()


def fetch_with_fallback(base: str, period: str, interval: str,
                        min_rows: int = 25) -> tuple:
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


def get_multi_timeframe_bias(ticker: str) -> tuple:
    return (0.0, "Neutral")


def get_market_regime() -> tuple:
    return (0.0, "NEUTRAL")


# ── Moving averages ───────────────────────────────────────────────────────────

def ema_series(series: pd.Series, span: int) -> pd.Series:
    return _squeeze(series).ewm(span=span, adjust=False).mean()


def sma_series(series: pd.Series, window: int) -> pd.Series:
    return _squeeze(series).rolling(window=window).mean()


def wma_series(series: pd.Series, window: int) -> pd.Series:
    """Weighted Moving Average."""
    s = _squeeze(series)
    weights = np.arange(1, window + 1, dtype=float)
    return s.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_ema(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average — reduces lag, smoother."""
    s = _squeeze(series)
    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma_half = wma_series(s, half)
    wma_full = wma_series(s, period)
    raw = 2 * wma_half - wma_full
    return wma_series(raw, sqrt)


# ── Oscillators ───────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> float:
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


def compute_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    series = _squeeze(series)
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series) -> tuple:
    series = _squeeze(series)
    if len(series) < 26:
        return 0.0, 0.0, 0.0
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


def compute_stochastic(df, period: int = 14, smooth: int = 3) -> tuple:
    if len(df) < period + smooth or not {"High", "Low", "Close"}.issubset(df.columns):
        return 50.0, 50.0
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    denom = highest_high - lowest_low
    k = 100 * ((close - lowest_low) / denom.replace(0, np.nan))
    d = k.rolling(window=smooth).mean()
    k_val = float(k.iloc[-1]); d_val = float(d.iloc[-1])
    if np.isnan(k_val): k_val = 50.0
    if np.isnan(d_val): d_val = 50.0
    return k_val, d_val


def compute_williams_r(df, period: int = 14) -> float:
    if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
        return -50.0
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low).replace(0, np.nan))
    val = float(wr.iloc[-1])
    return float(val) if not np.isnan(val) else -50.0


def compute_cci(df, period: int = 20) -> float:
    if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: float(np.abs(x - x.mean()).mean()), raw=True)
    cci = (tp - sma) / (0.015 * mean_dev.replace(0, np.nan))
    val = float(cci.iloc[-1])
    return float(val) if not np.isnan(val) else 0.0


def compute_roc(close: pd.Series, period: int = 10) -> float:
    close = _squeeze(close)
    if len(close) < period + 1:
        return 0.0
    old = close.iloc[-(period + 1)]
    new = close.iloc[-1]
    if old == 0 or np.isnan(old):
        return 0.0
    return float((new - old) / old * 100)


def compute_mfi(df, period: int = 14) -> float:
    """Money Flow Index — volume-weighted RSI."""
    if len(df) < period + 1 or not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return 50.0
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    volume = _squeeze(df["Volume"])
    tp = (high + low + close) / 3
    mf = tp * volume
    positive_flow = mf.where(tp > tp.shift(1), 0.0)
    negative_flow = mf.where(tp < tp.shift(1), 0.0)
    pos_mf = positive_flow.rolling(period).sum()
    neg_mf = negative_flow.rolling(period).sum()
    mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))
    val = float(mfi.iloc[-1])
    return float(val) if not np.isnan(val) else 50.0


# ── Trend / volatility ────────────────────────────────────────────────────────

def compute_atr(df, period: int = 14) -> float:
    if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    prev = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    val = atr.iloc[-1]
    return float(val) if not np.isnan(val) else 0.0


def compute_volatility_pct(df, period: int = 10) -> float:
    close = _squeeze(df["Close"])
    if len(close) < period + 1:
        return 0.0
    returns = close.pct_change().dropna()
    vol = returns.rolling(period).std() * np.sqrt(period)
    val = float(vol.iloc[-1]) * 100
    return float(val) if not np.isnan(val) else 0.0


def compute_supertrend(df, period: int = 10, multiplier: float = 3.0) -> tuple:
    """Supertrend — trend direction + trailing stop. Returns (value, direction)."""
    try:
        if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
            return float(_squeeze(df["Close"]).iloc[-1]), "Neutral"
        high = _squeeze(df["High"])
        low = _squeeze(df["Low"])
        close = _squeeze(df["Close"])
        atr = compute_atr(df, period)
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        st = [0.0] * len(df)
        direction = [1] * len(df)  # 1 = up, -1 = down

        for i in range(1, len(df)):
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            prev_close = close.iloc[i - 1]
            prev_st = st[i - 1]
            prev_dir = direction[i - 1]

            if prev_dir == 1:
                if curr_upper < prev_st or prev_close > curr_upper:
                    st[i] = curr_lower
                    direction[i] = -1 if prev_close < curr_lower else 1
                else:
                    st[i] = max(curr_upper, prev_st)
                    direction[i] = 1
            else:
                if curr_lower > prev_st or prev_close < curr_lower:
                    st[i] = curr_upper
                    direction[i] = 1 if prev_close > curr_upper else -1
                else:
                    st[i] = min(curr_lower, prev_st)
                    direction[i] = -1

        last_price = float(close.iloc[-1])
        last_st = st[-1]
        dir_label = "Bullish" if last_price > last_st else "Bearish"
        return round(last_st, 2), dir_label
    except Exception as exc:
        log.warning("Supertrend failed: %s", exc)
        return float(_squeeze(df["Close"]).iloc[-1]) if len(df) > 0 else 0.0, "Neutral"


def compute_pivots(df, lookback: int = 20) -> dict:
    """Classic floor-trader pivots from recent OHLC."""
    try:
        if len(df) < 2 or not {"High", "Low", "Close"}.issubset(df.columns):
            return {"pp": 0.0, "r1": 0.0, "r2": 0.0, "s1": 0.0, "s2": 0.0}
        high = float(_squeeze(df["High"]).iloc[-1])
        low = float(_squeeze(df["Low"]).iloc[-1])
        close = float(_squeeze(df["Close"]).iloc[-1])
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        return {"pp": round(pp, 2), "r1": round(r1, 2), "r2": round(r2, 2),
                "s1": round(s1, 2), "s2": round(s2, 2)}
    except Exception as exc:
        log.warning("Pivots failed: %s", exc)
        return {"pp": 0.0, "r1": 0.0, "r2": 0.0, "s1": 0.0, "s2": 0.0}


def compute_donchian(df, period: int = 20) -> tuple:
    """Donchian Channels — breakout levels."""
    try:
        if len(df) < period or not {"High", "Low"}.issubset(df.columns):
            return 0.0, 0.0, 0.0
        high = _squeeze(df["High"])
        low = _squeeze(df["Low"])
        upper = high.rolling(window=period).max().iloc[-1]
        lower = low.rolling(window=period).min().iloc[-1]
        mid = (upper + lower) / 2
        return float(upper), float(mid), float(lower)
    except Exception as exc:
        log.warning("Donchian failed: %s", exc)
        return 0.0, 0.0, 0.0


def compute_keltner(df, period: int = 20, multiplier: float = 2.0) -> tuple:
    """Keltner Channels — volatility envelope."""
    try:
        if len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
            return 0.0, 0.0, 0.0
        close = _squeeze(df["Close"])
        high = _squeeze(df["High"])
        low = _squeeze(df["Low"])
        typical = (high + low + close) / 3
        ema = typical.ewm(span=period, adjust=False).mean()
        atr = compute_atr(df, period)
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr
        return float(upper.iloc[-1]), float(ema.iloc[-1]), float(lower.iloc[-1])
    except Exception as exc:
        log.warning("Keltner failed: %s", exc)
        return 0.0, 0.0, 0.0


# ── Volume ────────────────────────────────────────────────────────────────────

def volume_ratio(df, lookback: int = 10) -> float:
    if "Volume" not in df.columns or len(df) < lookback + 1:
        return 1.0
    vol = _squeeze(df["Volume"])
    avg = float(vol.iloc[-(lookback + 1):-1].mean())
    last = float(vol.iloc[-1])
    if avg == 0 or np.isnan(avg):
        return 1.0
    return float(last / avg)


def compute_obv(df) -> tuple:
    """On-Balance Volume — returns (value, rising?)."""
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


def compute_cmf(df, period: int = 20) -> float:
    """Chaikin Money Flow — buying/selling pressure."""
    try:
        if len(df) < period + 1 or not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
            return 0.0
        high = _squeeze(df["High"])
        low = _squeeze(df["Low"])
        close = _squeeze(df["Close"])
        volume = _squeeze(df["Volume"])
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfv = mfm * volume
        cmf = mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)
        val = float(cmf.iloc[-1])
        return float(val) if not np.isnan(val) else 0.0
    except Exception as exc:
        log.warning("CMF failed: %s", exc)
        return 0.0


def compute_efi(df) -> float:
    """Elder Force Index — price change * volume."""
    try:
        if "Volume" not in df.columns or "Close" not in df.columns or len(df) < 13:
            return 0.0
        close = _squeeze(df["Close"])
        vol = _squeeze(df["Volume"])
        efi = (close.diff() * vol).ewm(span=13, adjust=False).mean()
        val = float(efi.iloc[-1])
        return float(val) if not np.isnan(val) else 0.0
    except Exception as exc:
        log.warning("EFI failed: %s", exc)
        return 0.0


# ── Bands / channels ──────────────────────────────────────────────────────────

def bb_position(df, period: int = 20) -> tuple:
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


def vwap_confirm(df) -> bool:
    if len(df) < 10 or not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return False
    close = _squeeze(df["Close"])
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    volume = _squeeze(df["Volume"])
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return bool(close.iloc[-1] > vwap.iloc[-1])


def support_resistance(df, lookback: int = 20) -> tuple:
    if len(df) < lookback:
        return 0.0, 0.0
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    recent_highs = high.tail(lookback)
    recent_lows = low.tail(lookback)
    last_price = float(close.iloc[-1])
    below = [l for l in recent_lows if l < last_price]
    above = [h for h in recent_highs if h > last_price]
    support = max(below) if below else float(recent_lows.min())
    resistance = min(above) if above else float(recent_highs.max())
    return support, resistance


def price_momentum(close: pd.Series, days: int = 5) -> float:
    close = _squeeze(close)
    if len(close) < days + 1:
        return 0.0
    old = close.iloc[-(days + 1)]
    new = close.iloc[-1]
    if old == 0 or np.isnan(old):
        return 0.0
    return float((new - old) / old * 100)


# ── ADX / SAR / Ichimoku ─────────────────────────────────────────────────────

def compute_adx(df, period: int = 14) -> tuple:
    if len(df) < period * 2 + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0, "Weak"
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_series = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=tr.index).ewm(span=period, adjust=False).mean() / atr_series.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=tr.index).ewm(span=period, adjust=False).mean() / atr_series.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    val = float(adx.iloc[-1])
    if np.isnan(val): val = 0.0
    label = "Strong" if val >= 25 else ("Moderate" if val >= 20 else "Weak")
    return round(val, 1), label


def compute_parabolic_sar(df, step: float = 0.02, max_step: float = 0.2) -> tuple:
    if len(df) < 10 or not {"High", "Low", "Close"}.issubset(df.columns):
        return float(_squeeze(df["Close"]).iloc[-1]), "Neutral"
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    sar = [float(close.iloc[0])]
    trend = 1
    ep = float(low.iloc[0])
    af = step
    for i in range(1, len(close)):
        prev_sar = sar[-1]
        if trend == 1:
            sar_val = prev_sar + af * (ep - prev_sar)
            sar_val = min(sar_val, float(low.iloc[i - 1]), float(low.iloc[i - 2]) if i >= 2 else float(low.iloc[i - 1]))
            if float(low.iloc[i]) < sar_val:
                trend = -1
                sar_val = ep
                ep = float(low.iloc[i])
                af = step
        else:
            sar_val = prev_sar + af * (ep - prev_sar)
            sar_val = max(sar_val, float(high.iloc[i - 1]), float(high.iloc[i - 2]) if i >= 2 else float(high.iloc[i - 1]))
            if float(high.iloc[i]) > sar_val:
                trend = 1
                sar_val = ep
                ep = float(high.iloc[i])
                af = step
        if trend == 1 and float(high.iloc[i]) > ep:
            ep = float(high.iloc[i]); af = min(af + step, max_step)
        if trend == -1 and float(low.iloc[i]) < ep:
            ep = float(low.iloc[i]); af = min(af + step, max_step)
        sar.append(sar_val)
    last_sar = sar[-1]
    last_price = float(close.iloc[-1])
    trend_label = "Bullish" if last_price > last_sar else "Bearish"
    return round(last_sar, 2), trend_label


def compute_ichimoku(df) -> tuple:
    if len(df) < 52 or not {"High", "Low", "Close"}.issubset(df.columns):
        return "Neutral", 0.0, 0.0
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    close = _squeeze(df["Close"])
    tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    last_close = float(close.iloc[-1])
    last_span_a = float(span_a.iloc[-1]) if not np.isnan(span_a.iloc[-1]) else last_close
    last_span_b = float(span_b.iloc[-1]) if not np.isnan(span_b.iloc[-1]) else last_close
    last_tenkan = float(tenkan.iloc[-1]) if not np.isnan(tenkan.iloc[-1]) else last_close
    last_kijun = float(kijun.iloc[-1]) if not np.isnan(kijun.iloc[-1]) else last_close
    cloud_top = max(last_span_a, last_span_b)
    cloud_bot = min(last_span_a, last_span_b)
    if last_close > cloud_top and last_tenkan > last_kijun:
        signal = "Bullish"
    elif last_close < cloud_bot and last_tenkan < last_kijun:
        signal = "Bearish"
    else:
        signal = "Neutral"
    return signal, round(cloud_top, 2), round(cloud_bot, 2)
