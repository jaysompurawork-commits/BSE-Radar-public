"""
reports.py — BSE Radar Bot v7.0
Professional HTML Telegram report formatters.
"""

from config import (
    MIN_CONFIDENCE_SHOW, INTRADAY_MAX_PICKS, PRE_MARKET_MAX_PICKS,
    POST_MARKET_SETUPS, TARGET1_PCT, TARGET2_PCT, STOPLOSS_PCT,
    MIN_RR_RATIO, next_market_open_str, ist_today, now_ist,
    is_trading_day, next_trading_day,
)
from indicators import (compute_rsi, compute_macd, compute_atr, volume_ratio,
                        bb_position, vwap_confirm, support_resistance,
                        compute_stochastic, compute_williams_r, compute_cci,
                        compute_roc, compute_obv, compute_adx,
                        compute_parabolic_sar, compute_ichimoku,
                        compute_supertrend, compute_pivots, compute_mfi,
                        compute_cmf, compute_efi, compute_volatility_pct,
                        hull_ema, price_momentum)
from sentiment import sentiment_score_for_ticker, market_wide_sentiment, fetch_fii_dii_flow

import backtest
import logging
import os

import pandas as pd
import numpy as np

log = logging.getLogger("bse_radar")

__all__ = [
    "build_pre_market_report", "build_intraday_report", "build_post_market_report",
    "build_closed_notice", "fmt", "conf_label", "sig_emoji", "sig_word",
]

# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt(v: float) -> str:
    return f"₹{v:,.2f}"


def conf_label(conf: int) -> str:
    if conf >= 90:
        return "🔥 Very Strong"
    if conf >= 80:
        return "💪 Strong"
    if conf >= 70:
        return "✅ Moderate"
    return "⚠️ Weak"


def sig_emoji(signal: str) -> str:
    return "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪")


def sig_word(signal: str) -> str:
    return {"BUY": "BUY", "SELL": "SELL"}.get(signal, "HOLD")


def holding_window(info) -> str:
    mins = info.get("hold_mins", 120)
    if mins < 60:
        return f"~{mins} min"
    hrs = mins / 60
    if hrs == int(hrs):
        return f"~{int(hrs)} hr"
    return f"~{hrs:.1f} hrs"


# ── Core signal engine ────────────────────────────────────────────────────────

def compute_score(df, ticker: str) -> dict:
    """Full multi-timeframe confluence scoring for one stock."""
    close = _squeeze(df["Close"])
    last_price = float(close.iloc[-1])
    atr = compute_atr(df)

    rsi = compute_rsi(close)
    macd_l, macd_s, macd_h = compute_macd(close)
    macd_bull = macd_l > macd_s

    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1]) if len(close) >= 9 else last_price
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1]) if len(close) >= 21 else last_price
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1]) if len(close) >= 20 else last_price
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else last_price
    hull20 = hull_ema(close, 20)
    hull_val = float(hull20.iloc[-1]) if len(hull20) and not pd.isna(hull20.iloc[-1]) else last_price

    trend_bull_9_21 = ema9 > ema21
    trend_bear_9_21 = ema9 < ema21
    trend_bull_20_50 = ema20 > ema50
    trend_bear_20_50 = ema20 < ema50

    mom = price_momentum(close, days=5)
    vol = volume_ratio(df)
    bb_pos, bb_width = bb_position(df)
    vwap_bull = vwap_confirm(df)
    support, resistance = support_resistance(df)

    stoch_k, stoch_d = compute_stochastic(df)
    stoch_bull = stoch_k > stoch_d
    willr = compute_williams_r(df)
    cci = compute_cci(df)
    roc = compute_roc(close, period=10)
    obv_val, obv_rising = compute_obv(df)
    mfi = compute_mfi(df)
    cmf = compute_cmf(df)
    efi = compute_efi(df)
    adx_val, adx_label = compute_adx(df)
    sar_val, sar_trend = compute_parabolic_sar(df)
    ichi_signal, ichi_top, ichi_bot = compute_ichimoku(df)
    super_val, super_dir = compute_supertrend(df)
    pivots = compute_pivots(df)

    # News sentiment
    news_score, news_label, news_conf = sentiment_score_for_ticker(ticker)
    fii_flow, fii_label = fetch_fii_dii_flow()

    score = 0
    confluence = []

    # 1. EMA alignment
    if trend_bull_9_21:
        score += 2
        confluence.append("EMA Bullish (9>21)")
    elif trend_bear_9_21:
        score -= 2
        confluence.append("EMA Bearish (9<21)")
    if trend_bull_20_50:
        score += 2
        confluence.append("Trend Bullish (20>50 EMA)")
    elif trend_bear_20_50:
        score -= 2
        confluence.append("Trend Bearish (20<50 EMA)")
    if last_price > hull_val:
        score += 1
        confluence.append("Price above Hull EMA(20)")
    else:
        score -= 1
        confluence.append("Price below Hull EMA(20)")

    # 2. RSI
    if rsi < 35:
        score += 2
        confluence.append(f"RSI Oversold ({rsi:.0f})")
    elif rsi > 65:
        score -= 2
        confluence.append(f"RSI Overbought ({rsi:.0f})")
    else:
        confluence.append(f"RSI {rsi:.0f}")

    # 3. MACD
    if macd_bull:
        score += 2
        confluence.append("MACD Bullish")
    else:
        score -= 2
        confluence.append("MACD Bearish")

    # 4. Momentum
    if mom > 1.0:
        score += 2
        confluence.append(f"Strong Momentum (+{mom:.1f}% 5d)")
    elif mom < -1.0:
        score -= 2
        confluence.append(f"Bearish Momentum ({mom:.1f}% 5d)")
    else:
        confluence.append(f"Momentum {mom:+.1f}% 5d")

    # 5. Volume
    if vol >= 1.2:
        score += 1
        confluence.append(f"High Volume ({vol:.1f}x)")
    else:
        confluence.append(f"Volume {vol:.1f}x")

    # 6. BB
    if bb_pos > 0.6:
        score += 1
        confluence.append("Upper BB Zone")
    elif bb_pos < 0.4:
        score -= 1
        confluence.append("Lower BB Zone")

    # 7. VWAP
    if vwap_bull:
        score += 1
        confluence.append("Above VWAP")
    else:
        score -= 1
        confluence.append("Below VWAP")

    # 8. MFI / CMF / EFI (volume-pressure confirmation)
    pressure_bull = (mfi > 50 and cmf > 0 and efi > 0)
    pressure_bear = (mfi < 50 and cmf < 0 and efi < 0)
    if pressure_bull:
        score += 1
        confluence.append("Volume Pressure Bullish")
    elif pressure_bear:
        score -= 1
        confluence.append("Volume Pressure Bearish")

    # 9. Stochastic
    if stoch_bull and stoch_k < 80:
        score += 1
        confluence.append(f"Stoch Bullish (K{stoch_k:.0f})")
    elif not stoch_bull and stoch_k > 20:
        score -= 1
        confluence.append(f"Stoch Bearish (K{stoch_k:.0f})")
    elif stoch_k <= 20:
        confluence.append(f"Stoch Oversold (K{stoch_k:.0f})")
    elif stoch_k >= 80:
        confluence.append(f"Stoch Overbought (K{stoch_k:.0f})")

    # 10. Williams %R / CCI / ROC
    if willr < -80:
        confluence.append(f"Williams %R Oversold ({willr:.0f})")
    elif willr > -20:
        confluence.append(f"Williams %R Overbought ({willr:.0f})")
    if cci < -100:
        confluence.append(f"CCI Oversold ({cci:.0f})")
    elif cci > 100:
        confluence.append(f"CCI Overbought ({cci:.0f})")
    if roc > 1.0:
        score += 1
        confluence.append(f"ROC Bullish (+{roc:.1f}%)")
    elif roc < -1.0:
        score -= 1
        confluence.append(f"ROC Bearish ({roc:.1f}%)")

    # 11. OBV
    if obv_rising:
        confluence.append("OBV Rising")
    else:
        confluence.append("OBV Falling")

    # 12. ADX
    if adx_val >= 25:
        confluence.append(f"ADX Strong ({adx_val})")
    elif adx_val >= 20:
        confluence.append(f"ADX Moderate ({adx_val})")

    # 13. Supertrend
    if super_dir == "Bullish" and last_price > super_val:
        score += 1
        confluence.append("Supertrend Bullish")
    elif super_dir == "Bearish" and last_price < super_val:
        score -= 1
        confluence.append("Supertrend Bearish")

    # 14. Parabolic SAR
    if last_price > sar_val and sar_trend == "Bullish":
        confluence.append("Parabolic SAR Bullish")
    elif last_price < sar_val and sar_trend == "Bearish":
        confluence.append("Parabolic SAR Bearish")

    # 15. Ichimoku
    if ichi_signal == "Bullish":
        score += 2
        confluence.append("Ichimoku Bullish Cloud")
    elif ichi_signal == "Bearish":
        score -= 2
        confluence.append("Ichimoku Bearish Cloud")

    # 16. Pivot context (informational only)
    px = last_price
    if pivots["r1"] and px > pivots["r1"]:
        confluence.append("Above Pivot R1")
    elif pivots["s1"] and px < pivots["s1"]:
        confluence.append("Below Pivot S1")

    # 17. News sentiment
    if news_score > 0.15:
        score += int(2 * news_conf)
        confluence.append(f"News: {news_label}")
    elif news_score < -0.15:
        score -= int(2 * news_conf)
        confluence.append(f"News: {news_label}")

    # 18. FII/DII
    if fii_label == "Bullish":
        score += 1
        confluence.append("FII-DII: Bullish inflow")
    elif fii_label == "Bearish":
        score -= 1
        confluence.append("FII-DII: Bearish outflow")

    signal = "BUY" if score >= 0 else "SELL"

    # Confidence
    factors_hit = sum([
        abs(rsi - 50) > 15,
        macd_l != 0,
        trend_bull_9_21 or trend_bear_9_21,
        abs(mom) > 0.5,
        vol >= 1.2,
        abs(bb_pos - 0.5) > 0.3,
        abs(stoch_k - 50) > 20,
        abs(willr) > 40,
        abs(cci) > 100,
        abs(roc) > 1.0,
        obv_rising is not None,
        adx_val >= 20,
        abs(last_price - sar_val) > 0.01,
        ichi_signal != "Neutral",
        vwap_bull or not vwap_bull,
        abs(news_score) > 0.1,
    ])
    confidence = min(94, 50 + factors_hit * 3)
    if abs(score) >= 5:
        confidence = min(95, confidence + 4)
    if abs(score) >= 9:
        confidence = min(98, confidence + 4)

    # Entry / targets / SL
    if signal == "BUY":
        entry = last_price
        target1 = round(entry * (1 + TARGET1_PCT / 100), 2)
        target2 = round(entry * (1 + TARGET2_PCT / 100), 2)
        stop_loss = round(entry * (1 - STOPLOSS_PCT / 100), 2)
    else:
        entry = last_price
        target1 = round(entry * (1 - TARGET1_PCT / 100), 2)
        target2 = round(entry * (1 - TARGET2_PCT / 100), 2)
        stop_loss = round(entry * (1 + STOPLOSS_PCT / 100), 2)

    risk = abs(entry - stop_loss)
    reward = abs(target2 - entry)
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

    # ATR-based hold window
    atr_pct = (atr / last_price * 100) if last_price > 0 else 1.0
    if atr_pct > 2.0:
        hold_mins = 90
    elif atr_pct > 1.0:
        hold_mins = 120
    else:
        hold_mins = 150

    return {
        "signal": signal,
        "confidence": confidence,
        "entry": entry,
        "target1": target1,
        "target2": target2,
        "stop_loss": stop_loss,
        "rr_ratio": rr_ratio,
        "hold_mins": hold_mins,
        "rsi": rsi,
        "score": score,
        "vol_ratio": vol,
        "bb_pos": round(bb_pos, 2),
        "vwap_bull": vwap_bull,
        "bias": "Neutral",
        "confluence": confluence,
        "atr": atr,
        "mom": mom,
        "stoch_k": round(stoch_k, 2),
        "willr": round(willr, 2),
        "cci": round(cci, 2),
        "roc": round(roc, 2),
        "obv_rising": obv_rising,
        "adx": adx_val,
        "sar": sar_val,
        "sar_trend": sar_trend,
        "ichimoku": ichi_signal,
        "supertrend": super_dir,
        "mfi": round(mfi, 1),
        "cmf": round(cmf, 3),
        "pivots": pivots,
        "news_score": news_score,
        "news_label": news_label,
        "fii_label": fii_label,
    }


def _squeeze(s):
    import pandas as pd
    import numpy as np
    if isinstance(s, pd.DataFrame):
        s = s.squeeze(axis="columns")
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def build_signal_block(rank, ticker, name, bse_code, sector, info, regime_emoji, regime_text) -> str:
    sig = info["signal"]
    emoji = sig_emoji(sig)
    label = conf_label(info["confidence"])
    t1_word = f"Target 1 (+{TARGET1_PCT}%)" if sig == "BUY" else f"Target 1 (-{TARGET1_PCT}%)"
    t2_word = f"Target 2 (+{TARGET2_PCT}%)" if sig == "BUY" else f"Target 2 (-{TARGET2_PCT}%)"
    sl_word = f"Stop Loss (-{STOPLOSS_PCT}%)" if sig == "BUY" else f"Stop Loss (+{STOPLOSS_PCT}%)"

    confluence_lines = "\n".join(f"  ✅ {c}" for c in info["confluence"][:12])

    return (
        f"⭐ <b>RANK #{rank} — {sig} PICK</b>\n"
        f"🎯 <b>BSE AI INTRADAY RADAR SIGNAL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏢 <b>Stock:</b> {name}\n"
        f"🏷️ <b>BSE Code:</b> {bse_code}  |  <b>Sector:</b> {sector}\n"
        f"{emoji} <b>Action:</b> {sig_word(sig)} ({info['confidence']}% Conf • {label})\n"
        f"\n"
        f"💵 <b>Entry:</b> {fmt(info['entry'])}\n"
        f"🎯 <b>{t1_word}:</b> {fmt(info['target1'])}\n"
        f"🚀 <b>{t2_word}:</b> {fmt(info['target2'])}\n"
        f"🛡️ <b>{sl_word}:</b> {fmt(info['stop_loss'])}\n"
        f"⚖️ <b>Risk:Reward:</b> 1 : {info['rr_ratio']}\n"
        f"⏱️ <b>Hold Window:</b> {holding_window(info)} (Intraday)\n"
        f"\n"
        f"🧠 <b>Confluence ({len(info['confluence'])} factors):</b>\n"
        f"{confluence_lines}\n"
        f"\n"
    )


def _scan_watchlist(fetch_kwargs, min_conf=MIN_CONFIDENCE_SHOW, limit=None):
    """Fetch + score every watchlist ticker; return ranked list."""
    from config import WATCHLIST
    import yfinance as yf

    ranked = []
    for ticker, info_dict in WATCHLIST.items():
        try:
            df, _ = _fetch(ticker, **fetch_kwargs)
            if df is None or df.empty or len(df) < 25:
                continue
            sig = compute_score(df, ticker)
            if sig["confidence"] < min_conf:
                continue
            ranked.append((ticker, info_dict["name"], info_dict["bse_code"],
                           info_dict.get("sector", "-"), sig))
        except Exception as exc:
            log.warning("Scan error %s: %s", ticker, exc)
    ranked.sort(key=lambda x: (x[4]["confidence"], abs(x[4]["score"])), reverse=True)
    if limit:
        ranked = ranked[:limit]
    return ranked


def _fetch(ticker, period="5d", interval="5m", min_rows=25):
    from indicators import fetch_with_fallback
    return fetch_with_fallback(ticker, period, interval, min_rows=min_rows)


# ── Report builders ───────────────────────────────────────────────────────────

def build_pre_market_report() -> str:
    """Morning brief — top conviction picks."""
    today = ist_today().strftime("%d %b %Y")
    now_str = now_ist().strftime("%I:%M:%S %p IST")
    regime_score, regime_text = _market_regime()
    regime_emoji = "📈" if regime_score > 0 else ("📉" if regime_score < 0 else "↔️")
    mkt_sent, mkt_label, mkt_conf = market_wide_sentiment()

    header = (
        f"📊 <b>BSE PRE-MARKET RADAR — {today}</b>\n"
        f"⏰ {now_str} IST\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏛️ Market Bias: {regime_emoji} {regime_text}\n"
        f"📰 Market Sentiment: {mkt_label}\n"
    )

    ranked = _scan_watchlist({"period": "5d", "interval": "5m", "min_rows": 40},
                             min_conf=MIN_CONFIDENCE_SHOW, limit=PRE_MARKET_MAX_PICKS)
    lines = [header]

    if not ranked:
        lines.append("⚪ No high-conviction setups today. Market is neutral — wait for clarity.\n")
        lines.append(f"🔔 Next market open: {next_market_open_str()}\n")
        lines.append("📌 <i>Not SEBI-registered investment advice. Trade at your own risk.</i>")
        return "\n".join(lines)

    # Top conviction pick first
    top = ranked[0]
    lines.append(build_signal_block(1, top[0], top[1], top[2], top[3], top[4], regime_emoji, regime_text))
    lines.append(f"⏰ Alert Time: {today} {now_str}\n\n")

    for i, (ticker, name, bse_code, sector, sig) in enumerate(ranked[1:], 2):
        lines.append(build_signal_block(i, ticker, name, bse_code, sector, sig, regime_emoji, regime_text))
        lines.append(f"⏰ Alert Time: {today} {now_str}\n\n")

    lines.append(
        f"📊 Data: Real NSE/BSE prices (yfinance) · Targets ±{TARGET1_PCT}%/{TARGET2_PCT}% · SL ±{STOPLOSS_PCT}%\n"
        f"📌 <i>Not SEBI-registered investment advice. Trade at your own risk.</i>\n"
        f"🤖 BSE Radar Bot v7.0 | Pre-market 9:00 AM IST"
    )
    return "\n".join(lines)


def build_intraday_report() -> str:
    """Live intraday scanner — top signals right now."""
    now_str = now_ist().strftime("%I:%M:%S %p IST")
    today = ist_today().strftime("%d %b %Y")
    regime_score, regime_text = _market_regime()
    regime_emoji = "📈" if regime_score > 0 else ("📉" if regime_score < 0 else "↔️")

    header = (
        f"⚡ <b>BSE INTRADAY SCANNER — {today} {now_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏛️ Market: {regime_emoji} {regime_text}\n"
    )

    ranked = _scan_watchlist({"period": "5d", "interval": "5m", "min_rows": 40},
                             min_conf=MIN_CONFIDENCE_SHOW, limit=INTRADAY_MAX_PICKS)
    lines = [header]

    if not ranked:
        lines.append("⚪ No high-confidence intraday signals right now.\n")
        lines.append("📌 <i>Not SEBI-registered investment advice.</i>")
        return "\n".join(lines)

    for i, (ticker, name, bse_code, sector, sig) in enumerate(ranked, 1):
        lines.append(build_signal_block(i, ticker, name, bse_code, sector, sig, regime_emoji, regime_text))
        lines.append(f"⏰ {today} {now_str}\n\n")

    lines.append(
        f"📊 Data: Real NSE/BSE 5-min prices · Scanner active 9:30 AM–3:00 PM IST\n"
        f"📌 <i>Not SEBI-registered investment advice.</i>\n"
        f"🤖 BSE Radar Bot v7.0 | Intraday"
    )
    return "\n".join(lines)


def build_post_market_report() -> str:
    """Day summary + next-session setups."""
    today = ist_today().strftime("%d %b %Y")
    now_str = now_ist().strftime("%I:%M %p IST")
    header = (
        f"🔔 <b>BSE POST-MARKET CLOSING REPORT — {today}</b>\n"
        f"⏰ Closing Bell: {now_str} IST\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )

    regime_score, regime_text = _market_regime()
    regime_emoji = "📈" if regime_score > 0 else ("📉" if regime_score < 0 else "↔️")
    mkt_sent, mkt_label, mkt_conf = market_wide_sentiment()

    # Top gainers/losers from daily closes
    from config import WATCHLIST
    from indicators import fetch_with_fallback, _squeeze
    results = []
    for ticker, info_dict in WATCHLIST.items():
        try:
            df, _ = fetch_with_fallback(ticker, "5d", "1d", min_rows=2)
            if df is None or df.empty or len(df) < 2:
                continue
            close = _squeeze(df["Close"])
            prev = float(close.iloc[-2])
            last = float(close.iloc[-1])
            chg = (last - prev) / prev * 100 if prev else 0.0
            results.append({
                "name": info_dict["name"], "ticker": ticker,
                "bse_code": info_dict["bse_code"],
                "last": last, "prev": prev, "chg": chg,
            })
        except Exception as exc:
            log.warning("Post-market error %s: %s", ticker, exc)

    results.sort(key=lambda x: x["chg"], reverse=True)
    gainers = [r for r in results if r["chg"] > 0]
    losers = [r for r in results if r["chg"] < 0]
    total = len(results)
    up_pct = (len(gainers) / total * 100) if total else 0
    down_pct = (len(losers) / total * 100) if total else 0

    lines = [header]
    lines.append(f"🏛️ Session Bias: {regime_emoji} {regime_text} · Market Sentiment: {mkt_label}\n")
    lines.append(f"• Advances: {len(gainers)} ({up_pct:.0f}%) | Declines: {len(losers)} ({down_pct:.0f}%)\n")

    if gainers:
        lines.append("🚀 Top Gainers:\n")
        for r in gainers[:5]:
            lines.append(f"  🟢 {r['name']}: {fmt(r['last'])} (+{r['chg']:.2f}%)\n")
        lines.append("")
    if losers:
        lines.append("🔻 Top Losers:\n")
        for r in losers[:5]:
            lines.append(f"  🔴 {r['name']}: {fmt(r['last'])} ({r['chg']:.2f}%)\n")
        lines.append("")

    # Key setups for next session
    ranked = _scan_watchlist({"period": "5d", "interval": "1d", "min_rows": 25},
                             min_conf=MIN_CONFIDENCE_SHOW, limit=POST_MARKET_SETUPS)
    if ranked:
        lines.append("🎯 Key Setups — Next Session:\n")
        for i, (ticker, name, bse_code, sector, sig) in enumerate(ranked, 1):
            arrow = "🟢" if sig["signal"] == "BUY" else "🔴"
            lines.append(
                f"  {arrow} {name}: {sig['signal']} ({sig['confidence']}% Conf)\n"
                f"     ↳ Close: {fmt(sig['entry'])} | Entry: {fmt(sig['entry'])} | "
                f"T1: {fmt(sig['target1'])} | SL: {fmt(sig['stop_loss'])}\n"
            )
        lines.append("")

    lines.append(
        f"📰 Media Sentiment: {mkt_label}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔔 Next market open: {next_market_open_str()}.\n"
        f"Pre-market brief arrives ~9:00 AM IST.\n"
        f"📌 <i>Not SEBI-registered investment advice.</i>"
    )
    return "\n".join(lines)


def _market_regime() -> tuple:
    """Return (score, text) for overall market regime using Nifty/Sensex."""
    try:
        from indicators import _squeeze, ema_series, safe_download
        score = 0
        for tk in ("^NSEI", "^BSESN"):
            from indicators import safe_download
            df = safe_download(tk, period="60d", interval="1d")
            if df is None or df.empty or len(df) < 50:
                continue
            close = _squeeze(df["Close"])
            ema50 = float(ema_series(close, 50).iloc[-1])
            if float(close.iloc[-1]) > ema50:
                score += 1
            else:
                score -= 1
        if score >= 1:
            return 1.0, "BULLISH"
        if score <= -1:
            return -1.0, "BEARISH"
        return 0.0, "NEUTRAL"
    except Exception as exc:
        log.warning("Regime calc failed: %s", exc)
        return 0.0, "NEUTRAL"


def build_closed_notice() -> str:
    today = ist_today().strftime("%d %b %Y (%A)")
    nd = next_trading_day()
    return (
        f"🔔 <b>BSE Radar Bot — Market Closed</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📅 {today}\n"
        f"🇮🇳 NSE/BSE is closed today (weekend/holiday).\n"
        f"No report sent.\n"
        f"🔔 Next market open: {nd.strftime('%A, %d %b %Y')} at 09:15 AM IST.\n"
        f"📌 <i>Not SEBI-registered investment advice.</i>"
    )
