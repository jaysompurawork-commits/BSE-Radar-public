# BSE Radar Bot — v7.0 Professional

Professional-grade intraday prediction system for BSE/NSE equities. Runs automatically on GitHub Actions — your PC does NOT need to be on.

## What's new in v7.0

| Feature | Description |
|---|---|
| **30+ technical indicators** | EMA/SMA/Hull, RSI, MACD, ATR, Bollinger Bands, VWAP, Stochastic, Williams %R, CCI, ROC, OBV, ADX, Parabolic SAR, Ichimoku, Supertrend, Pivots, Donchian, Keltner, MFI, CMF, EFI |
| **Multi-timeframe confluence** | 9/21 EMA alignment, 20/50 trend, MTF bias, pivot context, ATR-based hold windows |
| **News sentiment** | Headline sentiment per stock + sector (requires `NEWS_API_KEY`; degrades gracefully without it) |
| **Institutional flow** | FII/DII direction hook (wired to neutral, ready for NSE/Alpha Vantage endpoint) |
| **Risk management** | ±0.5% T1, ±1.0% T2, ±0.5% SL, R/R ≥ 1.5 filter, ATR-based holding window |
| **Performance journal** | Every signal logged to `data/signals.jsonl`; win-rate / expectancy tracking in `backtest.py` |
| **Weekend/holiday guard** | No pre/post reports on Sat/Sun/NSE holidays; "next market open" always resolves to next trading day |
| **Dry-run / paper trading** | `DRY_RUN=1` previews reports without sending; `PAPER_TRADING=1` logs signals only |
| **Professional HTML Telegram reports** | Ranked picks with entry/T1/T2/SL, confluence factors, execution strategy, disclaimer |

## Schedule (IST, Mon–Fri only)

| Report | Time (IST) | Days |
|---|---|---|
| Pre-Market (ranked BUY/SELL picks) | **9:00 AM** | Mon–Fri |
| Intraday scanner (live signals) | every 10 min, 9:00 AM–3:15 PM | Mon–Fri |
| Post-Market (day summary + next-session setups) | **3:20 PM** | Mon–Fri |

Cron is weekday-only (`1-5`). The code guard is a second line of defence for manual/workflow_dispatch runs.

## 1. Add Secrets to GitHub

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Secret | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHAT_IDS` | Comma-separated chat IDs |
| `TELEGRAM_CHAT_ID` | Single chat ID (legacy) |
| `NEWS_API_KEY` | (optional) NewsAPI key for sentiment |
| `DRY_RUN` | `1` to preview locally without sending |
| `SEND_CLOSED_NOTICE` | `1` to send "market closed" notice on holidays |

### Get your Chat ID
1. Add your bot to the group/channel as admin
2. Send any message
3. Open: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Use the `"chat":{"id":-100xxxxxxxxx}` number

## 2. Run Manually (GitHub Actions)

1. Repo → **Actions** tab → **BSE Radar Bot v7.0**
2. **Run workflow** → choose `pre`, `intraday`, or `post`
3. Click **Run workflow**

## 3. Local dry-run test

From the repo root:

```powershell
$env:TELEGRAM_BOT_TOKEN="dummy"
$env:TELEGRAM_CHAT_ID="dummy"
$env:DRY_RUN="1"
$env:REPORT_TYPE="pre"
python bse-radar-bot\bse_radar_bot.py
```

Prints the report to your terminal instead of sending it (no real token needed).

## Project structure

```
bse-radar-bot/
├── bse_radar_bot.py      # entry point (pre/intraday/post + guards)
├── config.py             # watchlist, constants, time helpers, paths
├── indicators.py         # 30+ technical indicators
├── sentiment.py          # news sentiment + FII-DII flow
├── backtest.py           # signal journal + accuracy/win-rate tracking
├── reports.py            # signal engine + HTML Telegram formatters
├── requirements.txt      # Python deps
├── data/                 # signal/performance/prediction logs (created at runtime)
├── .github/workflows/bse_radar.yml
└── README_BOT.md
```

## Data Source

All prices are real NSE/BSE data via `yfinance`. News sentiment via NewsAPI (optional). No mock/fake data.

## Disclaimer

Technical analysis + news sentiment only. Not SEBI-registered investment advice. Trade at your own risk. Past performance does not guarantee future results.
