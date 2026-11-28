"""
bse_radar_bot.py — BSE Radar Bot v7.0 (Professional)
Entry point: pre / intraday / post reports via Telegram.

Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS, TELEGRAM_CHAT_ID, REPORT_TYPE,
DRY_RUN, SEND_CLOSED_NOTICE, PAPER_TRADING, NEWS_API_KEY
"""

import logging
import os
import sys
import traceback
from datetime import date

import requests

from config import (
    IST, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS_RAW, TELEGRAM_CHAT_ID,
    REPORT_TYPE, DRY_RUN, SEND_CLOSED_NOTICE, PAPER_TRADING,
    now_ist, ist_today, is_trading_day, next_trading_day, next_market_open_str,
)
import reports
from config import WATCHLIST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("bse_radar")

__all__ = ["send_telegram", "build_message", "main"]


def send_telegram(message: str, bot_token: str, chat_ids: list[str]) -> None:
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
        log.info("Sent to chat %s — ok=%s", chat_id, resp.json().get("ok"))


def build_message(report_type: str) -> str:
    if report_type == "pre":
        return reports.build_pre_market_report()
    if report_type == "intraday":
        return reports.build_intraday_report()
    if report_type == "post":
        return reports.build_post_market_report()
    raise ValueError(f"Unknown REPORT_TYPE: {report_type}")


def _chat_ids() -> list[str]:
    ids = [c.strip() for c in TELEGRAM_CHAT_IDS_RAW.split(",") if c.strip()]
    if TELEGRAM_CHAT_ID:
        ids.append(TELEGRAM_CHAT_ID.strip())
    # dedupe preserving order
    seen = set()
    out = []
    for c in ids:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main() -> None:
    report_type = REPORT_TYPE
    if report_type not in ("pre", "intraday", "post"):
        log.error("Unknown REPORT_TYPE: %s (must be pre/intraday/post)", report_type)
        sys.exit(1)

    log.info("IST time: %s | report_type=%s | dry_run=%s | paper=%s",
             now_ist().strftime("%Y-%m-%d %I:%M %p"), report_type, DRY_RUN, PAPER_TRADING)

    # ── Weekend / holiday guard ─────────────────────────────────────────────
    today = ist_today()
    if not is_trading_day():
        reason = "weekend" if today.weekday() >= 5 else "NSE/BSE holiday"
        nd = next_trading_day()
        log.info("Market closed today (%s — %s). No %s report. Next open: %s",
                 today.strftime("%A, %d %b %Y"), reason, report_type,
                 nd.strftime("%A, %d %b %Y"))
        if SEND_CLOSED_NOTICE and TELEGRAM_BOT_TOKEN:
            notice = reports.build_closed_notice()
            if DRY_RUN:
                print("---- DRY RUN: closed notice ----")
                print(notice)
            else:
                try:
                    ids = _chat_ids()
                    if ids:
                        send_telegram(notice, TELEGRAM_BOT_TOKEN, ids)
                except Exception as exc:
                    log.error("Closed-notice send failed: %s", exc)
        return

    # ── Intraday: only during prediction window (9:00 AM – 3:15 PM) ─────────
    from config import is_prediction_window, is_post_market_time
    if report_type == "intraday" and not is_prediction_window():
        log.info("Outside prediction window (9:00 AM–3:15 PM) — skipping intraday send.")
        return
    if report_type == "post" and not is_post_market_time():
        log.info("Not post-market time (3:20 PM+) — skipping post send.")
        return

    # ── Build message ───────────────────────────────────────────────────────
    try:
        message = build_message(report_type)
    except Exception as exc:
        log.error("Report build failed: %s\n%s", exc, traceback.format_exc())
        sys.exit(1)

    if not message:
        log.info("Empty report — nothing to send.")
        return

    # ── Dry run / paper trading ─────────────────────────────────────────────
    if DRY_RUN or PAPER_TRADING:
        print("---- DRY RUN / PAPER TRADING: report preview ----")
        print(message)
        log.info("Dry run complete — report printed above (not sent).")
        return

    # ── Send ────────────────────────────────────────────────────────────────
    if not TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)
    ids = _chat_ids()
    if not ids:
        log.error("TELEGRAM_CHAT_IDS (or TELEGRAM_CHAT_ID) is not set.")
        sys.exit(1)

    try:
        send_telegram(message, TELEGRAM_BOT_TOKEN, ids)
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        sys.exit(1)

    log.info("Done — %s report sent to %d chat(s).", report_type, len(ids))


if __name__ == "__main__":
    main()
