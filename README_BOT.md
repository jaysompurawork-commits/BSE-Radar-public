# BSE Radar Bot — Setup Guide

Runs automatically on GitHub Actions. Your PC does NOT need to be on.

## 1. Add Secrets to GitHub

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these two secrets:

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather (e.g. `7123456789:AAFxxx...`) |
| `TELEGRAM_CHAT_ID` | Your group/channel ID (e.g. `-1001234567890`) |

### How to get your Chat ID

1. Add your bot to the group/channel as admin
2. Send any message in the group
3. Open in browser: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Look for `"chat":{"id":-100xxxxxxxxx}` — that number is your Chat ID

## 2. Schedule

The bot runs automatically:

| Report | Time (IST) | Days |
|---|---|---|
| Pre-Market (BUY/SELL picks) | **9:00 AM** | Mon–Fri |
| Post-Market (day summary) | **4:00 PM** | Mon–Fri |

## 3. Run Manually

1. Go to your GitHub repo → **Actions** tab
2. Click **BSE Radar Bot** workflow
3. Click **Run workflow**
4. Choose `pre` or `post`
5. Click green **Run workflow** button

## 4. What the Bot Sends

**Pre-market (9:00 AM IST):** BUY/SELL picks with confidence %, target price, stop-loss — based on real NSE closing data.

**Post-market (4:00 PM IST):** Top gainers/losers from today, advance/decline summary, Nifty close.

## 5. Data Source

All prices are real NSE data fetched via `yfinance`. No mock/fake data.

## 6. Disclaimer

Not SEBI-registered investment advice. For educational/informational use only.
