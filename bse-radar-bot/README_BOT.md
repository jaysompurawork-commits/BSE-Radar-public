# BSE Radar Bot — Setup Guide

Runs automatically on GitHub Actions. Your PC does NOT need to be on.

## What's new in v6.2

- **Weekend & holiday guard**: No pre-market or post-market report is sent on Saturdays, Sundays, or NSE/BSE holidays. The "next market open" line now resolves to the next **trading day** (e.g. Saturday → Monday 09:15 AM IST), never "tomorrow" on a weekend.
- **Optional closed-day notice**: Set `SEND_CLOSED_NOTICE=1` to send a short "market closed today" message on non-trading days.
- **Dry-run mode**: Set `DRY_RUN=1` locally to preview the report text instead of sending it to Telegram (great for testing without secrets).
- **Intraday job**: GitHub Actions now also runs an intraday scanner every 30 min during market hours (weekdays).

## 1. Add Secrets to GitHub

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather (e.g. `7123456789:AAFxxx...`) |
| `TELEGRAM_CHAT_IDS` | Comma-separated chat IDs (e.g. `-1001234567890`) |
| `TELEGRAM_CHAT_ID` | Single chat ID (legacy, also supported) |

### How to get your Chat ID

1. Add your bot to the group/channel as admin
2. Send any message in the group
3. Open in browser: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Look for `"chat":{"id":-100xxxxxxxxx}` — that number is your Chat ID

## 2. Schedule (IST, Mon–Fri only)

| Report | Time (IST) | Days |
|---|---|---|
| Pre-Market (BUY/SELL picks) | **9:00 AM** | Mon–Fri |
| Intraday scanner (live signals) | every 30 min, 9:30 AM–2:30 PM | Mon–Fri |
| Post-Market (day summary) | **4:00 PM** | Mon–Fri |

The cron schedules are weekday-only (`1-5`), so the bot never fires on a Saturday/Sunday. The weekend guard in the code is a second line of defence for manual/workflow_dispatch runs.

## 3. Run Manually

1. Go to your GitHub repo → **Actions** tab
2. Click **BSE Radar Bot** workflow
3. Click **Run workflow**
4. Choose `pre`, `intraday`, or `post`
5. Click green **Run workflow** button

### Local dry-run test

```powershell
# From the repo root
$env:DRY_RUN="1"
$env:REPORT_TYPE="pre"
python bse-radar-bot\bse_radar_bot.py
```
This prints the report to your terminal instead of sending it (no Telegram token needed).

## 4. What the Bot Sends

**Pre-market (9:00 AM IST):** BUY/SELL picks with confidence %, target price, stop-loss — based on real NSE closing data.

**Intraday:** Top live intraday signals refreshed every 30 min during market hours.

**Post-market (4:00 PM IST):** Top gainers/losers from today, advance/decline summary, key setups for next session, and the correct next market-open time.

## 5. Data Source

All prices are real NSE data fetched via `yfinance`. No mock/fake data.

## 6. Disclaimer

Not SEBI-registered investment advice. For educational/informational use only.
