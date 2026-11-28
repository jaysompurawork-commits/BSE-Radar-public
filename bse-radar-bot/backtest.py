"""
backtest.py — BSE Radar Bot v7.0
Signal performance logging and accuracy tracking (paper trading journal).

Every signal is appended to signals.jsonl with ticker, signal, confidence,
entry, targets, stop-loss, and timestamp. A follow-up process (or the next
session's post-market report) records actual outcomes so accuracy stats
(win rate, avg R/R, expectancy) can be computed per ticker and overall.
"""

import json
import logging
import os
from datetime import datetime, date, timedelta

from config import IST, DATA_DIR, SIGNAL_LOG, PERFORMANCE_LOG, PREDICTION_LOG, now_ist, ist_today

log = logging.getLogger("bse_radar")

__all__ = [
    "log_signal", "log_outcome", "compute_accuracy", "append_prediction",
    "ensure_data_dir",
]


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _append_jsonl(path: str, record: dict):
    ensure_data_dir()
    record["logged_at_ist"] = now_ist().isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_signal(ticker: str, name: str, bse_code: str, sector: str,
               signal: str, confidence: int, entry: float, target1: float,
               target2: float, stop_loss: float, rr_ratio: float,
               score: int, report_type: str, confluence: list,
               regime: str, sentiment_label: str) -> None:
    """Record a new signal to the paper-trading journal."""
    record = {
        "date": ist_today().isoformat(),
        "ticker": ticker,
        "name": name,
        "bse_code": bse_code,
        "sector": sector,
        "signal": signal,
        "confidence": confidence,
        "entry": entry,
        "target1": target1,
        "target2": target2,
        "stop_loss": stop_loss,
        "rr_ratio": rr_ratio,
        "score": score,
        "report_type": report_type,
        "confluence": confluence,
        "regime": regime,
        "sentiment": sentiment_label,
        "status": "open",          # open | hit_t1 | hit_t2 | hit_sl | expired
        "actual_return_pct": None,
        "outcome": None,
    }
    _append_jsonl(SIGNAL_LOG, record)
    log.info("Logged signal %s %s %s conf=%d entry=%.2f",
             ticker, signal, confidence, confidence, entry)


def log_outcome(ticker: str, exit_price: float, exit_reason: str,
                exit_date: date | None = None) -> dict | None:
    """Mark the most recent open signal for `ticker` as closed with outcome."""
    if not os.path.exists(SIGNAL_LOG):
        return None
    exit_date = exit_date or ist_today()
    records = []
    with open(SIGNAL_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    updated = None
    for rec in reversed(records):
        if rec.get("ticker") == ticker and rec.get("status") == "open":
            entry = rec.get("entry")
            signal = rec.get("signal")
            if entry:
                if signal == "BUY":
                    ret = (exit_price - entry) / entry * 100
                else:
                    ret = (entry - exit_price) / entry * 100
                rec["actual_return_pct"] = round(ret, 2)
            rec["exit_price"] = exit_price
            rec["exit_date"] = exit_date.isoformat()
            rec["exit_reason"] = exit_reason  # hit_t1 | hit_t2 | hit_sl | expired
            rec["status"] = "closed"
            rec["outcome"] = "win" if rec.get("actual_return_pct", 0) > 0 else ("loss" if rec.get("actual_return_pct", 0) < 0 else "breakeven")
            updated = rec
            break

    if updated is None:
        return None

    # Rewrite the whole log (small file) with updated statuses.
    with open(SIGNAL_LOG, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    log.info("Logged outcome %s %s -> %.2f%% (%s)",
             ticker, exit_reason, updated.get("actual_return_pct"), updated.get("outcome"))
    return updated


def compute_accuracy(days: int = 30) -> dict:
    """Compute win rate, avg return, expectancy over the last `days`."""
    if not os.path.exists(SIGNAL_LOG):
        return {"trades": 0, "wins": 0, "win_rate": 0.0,
                "avg_return": 0.0, "expectancy": 0.0, "rr_avg": 0.0}
    cutoff = ist_today() - timedelta(days=days)
    trades = []
    with open(SIGNAL_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "closed":
                continue
            try:
                d = date.fromisoformat(rec.get("date", ""))
            except (ValueError, TypeError):
                continue
            if d < cutoff:
                continue
            trades.append(rec)

    total = len(trades)
    wins = sum(1 for t in trades if t.get("outcome") == "win")
    losses = sum(1 for t in trades if t.get("outcome") == "loss")
    breakevens = sum(1 for t in trades if t.get("outcome") == "breakeven")
    win_rate = (wins / total * 100) if total else 0.0
    returns = [t.get("actual_return_pct") or 0 for t in trades]
    avg_return = sum(returns) / total if total else 0.0
    rr_vals = [t.get("rr_ratio") or 0 for t in trades]
    rr_avg = sum(rr_vals) / total if total else 0.0
    # Expectancy = win_rate * avg_win + (1-win_rate) * avg_loss
    win_returns = [r for r in returns if r > 0]
    loss_returns = [r for r in returns if r < 0]
    avg_win = sum(win_returns) / len(win_returns) if win_returns else 0.0
    avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else 0.0
    expectancy = (win_rate / 100) * avg_win + ((100 - win_rate) / 100) * avg_loss

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 2),
        "expectancy": round(expectancy, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "rr_avg": round(rr_avg, 2),
    }


def append_prediction(ticker: str, signal: str, confidence: int,
                      entry: float, report_type: str) -> None:
    """Lightweight prediction record for the daily predictions log."""
    record = {
        "date": ist_today().isoformat(),
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "entry": entry,
        "report_type": report_type,
    }
    _append_jsonl(PREDICTION_LOG, record)
