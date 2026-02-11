# ⚽ Football Streak Tracker — Telegram Bot

A Telegram bot that aggregates daily football **winning streaks** and **goal-scoring streaks** across Europe's top leagues.

## What It Tracks

| Streak Type | Description |
|---|---|
| 🔥 Winning Streak | Consecutive wins (most recent first) |
| 🛡️ Unbeaten Streak | Consecutive games without a loss |
| ⚽ Goal Streak | Consecutive games scoring at least 1 goal |
| 🧤 Clean Sheet Streak | Consecutive games without conceding |
| Form Guide | Last 5 results: 🟢 Win 🟡 Draw 🔴 Loss |

## Leagues Covered

- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League
- 🇪🇸 La Liga
- 🇩🇪 Bundesliga
- 🇮🇹 Serie A
- 🇫🇷 Ligue 1
- 🏆 Champions League

## Setup

### 1. Get API Keys

**Telegram Bot Token:**
1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the token you receive

**Football Data API Key:**
1. Register at [football-data.org](https://www.football-data.org/client/register)
2. The free tier gives you 10 requests/minute — enough for this bot
3. Copy your API key from the dashboard

### 2. Install & Run

```bash
# Clone or download the project
cd telegram-football-bot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export TELEGRAM_BOT_TOKEN="your-telegram-token"
export FOOTBALL_API_KEY="your-football-data-key"

# Optional: Enable daily summary to a specific chat
export DAILY_CHAT_ID="your-chat-id"

# Run the bot
python bot.py
```

### 3. Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message and instructions |
| `/streaks` | Top winning streaks across all leagues |
| `/goals` | Top goal-scoring streaks across all leagues |
| `/league` | Pick a specific league for full streak report |
| `/today` | Today's matches across tracked leagues |
| `/help` | Show available commands |

## Optional: Run with Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py .
CMD ["python", "bot.py"]
```

```bash
docker build -t football-streak-bot .
docker run -d \
  -e TELEGRAM_BOT_TOKEN="your-token" \
  -e FOOTBALL_API_KEY="your-key" \
  --name streak-bot \
  football-streak-bot
```

## Optional: Daily Summary

Set the `DAILY_CHAT_ID` environment variable to automatically receive a streak summary at **08:00 UTC** every day. To get your chat ID, message [@userinfobot](https://t.me/userinfobot) on Telegram.

## API Rate Limits

The free tier of football-data.org allows **10 requests/minute**. The bot handles this gracefully — each `/streaks` or `/goals` command makes ~6 requests (one per league). If you hit limits, wait a minute before trying again.

## Architecture

```
bot.py
├── FootballAPI          — HTTP client for football-data.org
├── calculate_streaks()  — Core streak calculation engine
├── format_*()           — Message formatting helpers
├── cmd_*()              — Telegram command handlers
├── league_callback()    — Inline keyboard handler
└── daily_summary()      — Scheduled job for daily reports
```
