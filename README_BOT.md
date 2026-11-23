# BSE Radar Bot — Setup Guide

Runs automatically on GitHub Actions. Your PC does NOT need to be on.

## 1. Add Secrets to GitHub

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these two secrets:

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your group/channel ID (e.g. `-1001234567890`) |

Or use `TELEGRAM_CHAT_IDS` for multiple chat IDs separated by commas.

### How to get your Chat ID

1. Add your bot to the group/channel as admin
2. Send any message in the group
3. Open: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Look for `"chat":{"id":-100xxxxxxxxx}` — that number is your Chat ID

## 2. Schedule (IST)

| Report | Time (IST) | Days |
|---|---|---|
| Pre-Market (BUY/SELL picks) | **9:00 AM** | Mon–Fri |
| Intraday updates | Every **30 min**, 9:30 AM–3:00 PM | Mon–Fri |
| Post-Market (gainers/losers + accuracy) | **4:00 PM** | Mon–Fri |

## 3. Run Manually

1. Go to your GitHub repo → **Actions** tab
2. Click **BSE Radar Bot** workflow
3. Click **Run workflow**
4. Choose `pre`, `intraday`, or `post`
5. Click green **Run workflow** button

## 4. What the Bot Sends

**Pre-market (9:00 AM IST):** BUY/SELL picks with confidence %, target price, stop-loss.

**Intraday (every 30 min):** Nifty/Sensex, top gainers/losers, current prices.

**Post-market (4:00 PM):** Day summary + **accuracy update** showing how pre-market predictions performed (wins/losses/accuracy %).

## 5. Data Source

All prices are real NSE data fetched via `yfinance`. No mock/fake data.

## 6. Accuracy Tracking

Each pre-market prediction is logged and checked against intraday prices. The post-market report shows whether targets or stop-losses were hit and your overall accuracy %.

## 7. Disclaimer

Not SEBI-registered investment advice. For educational/informational use only.
