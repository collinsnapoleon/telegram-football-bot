"""
Football Streak Tracker Telegram Bot
=====================================
Aggregates daily football winning streaks and goal-scoring streaks.
Uses the football-data.org API (free tier).

Commands:
  /start        - Welcome message & instructions
  /streaks      - Show top winning streaks across leagues
  /goals        - Show top goal-scoring streaks
  /league <code> - Show streaks for a specific league
  /today        - Show today's matches and streak implications
  /help         - Show all commands
"""

import os
import logging
import json
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

# ── Config ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8463379208:AAGNQgGvZT6qMVdPpEtJhW9xN99M5ma6Pzc")
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "bf1f95150ac74b3f804648682e0d5677")
FOOTBALL_API_BASE = "https://api.football-data.org/v4"

# ── Free-tier leagues (football-data.org) ─────────────────────────────────────
FREE_LEAGUES = {
    "PL": {"name": "Premier League", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "country": "England", "type": "LEAGUE"},
    "ELC": {"name": "Championship", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "country": "England", "type": "LEAGUE"},
    "PD": {"name": "La Liga", "flag": "🇪🇸", "country": "Spain", "type": "LEAGUE"},
    "BL1": {"name": "Bundesliga", "flag": "🇩🇪", "country": "Germany", "type": "LEAGUE"},
    "SA": {"name": "Serie A", "flag": "🇮🇹", "country": "Italy", "type": "LEAGUE"},
    "FL1": {"name": "Ligue 1", "flag": "🇫🇷", "country": "France", "type": "LEAGUE"},
    "DED": {"name": "Eredivisie", "flag": "🇳🇱", "country": "Netherlands", "type": "LEAGUE"},
    "PPL": {"name": "Primeira Liga", "flag": "🇵🇹", "country": "Portugal", "type": "LEAGUE"},
    "BSA": {"name": "Série A", "flag": "🇧🇷", "country": "Brazil", "type": "LEAGUE"},
    "CL": {"name": "Champions League", "flag": "🏆", "country": "Europe", "type": "CUP"},
    "WC": {"name": "FIFA World Cup", "flag": "🌍", "country": "World", "type": "CUP"},
    "EC": {"name": "European Championship", "flag": "🇪🇺", "country": "Europe", "type": "CUP"},
}

# ── Paid-tier leagues (require Tier 2+ subscription at €49+/month) ────────────
PAID_LEAGUES = {
    "EL": {"name": "Europa League", "flag": "🟠", "country": "Europe", "type": "CUP"},
    "CLI": {"name": "Conference League", "flag": "🟢", "country": "Europe", "type": "CUP"},
    "SD": {"name": "Segunda División", "flag": "🇪🇸", "country": "Spain", "type": "LEAGUE"},
    "BL2": {"name": "2. Bundesliga", "flag": "🇩🇪", "country": "Germany", "type": "LEAGUE"},
    "SB": {"name": "Serie B", "flag": "🇮🇹", "country": "Italy", "type": "LEAGUE"},
    "FL2": {"name": "Ligue 2", "flag": "🇫🇷", "country": "France", "type": "LEAGUE"},
    "BSB": {"name": "Série B", "flag": "🇧🇷", "country": "Brazil", "type": "LEAGUE"},
    "MLS": {"name": "MLS", "flag": "🇺🇸", "country": "USA", "type": "LEAGUE"},
    "ASL": {"name": "A-League", "flag": "🇦🇺", "country": "Australia", "type": "LEAGUE"},
    "JPL": {"name": "J. League", "flag": "🇯🇵", "country": "Japan", "type": "LEAGUE"},
    "LMX": {"name": "Liga MX", "flag": "🇲🇽", "country": "Mexico", "type": "LEAGUE"},
    "JPB": {"name": "Jupiler Pro League", "flag": "🇧🇪", "country": "Belgium", "type": "LEAGUE"},
    "SPL": {"name": "Scottish Premiership", "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "country": "Scotland", "type": "LEAGUE"},
    "DSL": {"name": "Superliga", "flag": "🇩🇰", "country": "Denmark", "type": "LEAGUE"},
    "SSL": {"name": "Super League", "flag": "🇨🇭", "country": "Switzerland", "type": "LEAGUE"},
    "TSL": {"name": "Süper Lig", "flag": "🇹🇷", "country": "Turkey", "type": "LEAGUE"},
    "GSL": {"name": "Super League", "flag": "🇬🇷", "country": "Greece", "type": "LEAGUE"},
}

# Combined dict — bot tries all, gracefully skips 403/404 on unpaid leagues
LEAGUES = {**FREE_LEAGUES, **PAID_LEAGUES}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── API Client ───────────────────────────────────────────────────────────────
class FootballAPI:
    """Wrapper around football-data.org API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"X-Auth-Token": api_key}

    async def _get(self, endpoint: str, params: dict = None) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{FOOTBALL_API_BASE}{endpoint}",
                headers=self.headers,
                params=params or {},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_matches(
        self, league_code: str, date_from: str = None, date_to: str = None
    ) -> list:
        """Fetch finished matches for a league within a date range."""
        params = {"status": "FINISHED"}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        data = await self._get(f"/competitions/{league_code}/matches", params)
        return data.get("matches", [])

    async def get_todays_matches(self, league_code: str) -> list:
        """Fetch today's scheduled and live matches."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        params = {"dateFrom": today, "dateTo": today}
        data = await self._get(f"/competitions/{league_code}/matches", params)
        return data.get("matches", [])

    async def get_standings(self, league_code: str) -> list:
        """Fetch current standings."""
        data = await self._get(f"/competitions/{league_code}/standings")
        standings = data.get("standings", [])
        for s in standings:
            if s.get("type") == "TOTAL":
                return s.get("table", [])
        return []


api = FootballAPI(FOOTBALL_API_KEY)


# ── Match stage filtering ─────────────────────────────────────────────────────
# Only these stages count toward league streaks. Everything else (friendlies,
# playoffs, penalty shootouts, super cups, etc.) is excluded.
VALID_STAGES = {
    "REGULAR_SEASON",
    "GROUP_STAGE",
    "LEAGUE_STAGE",        # UCL new format
    "ROUND_OF_16",
    "QUARTER_FINALS",
    "SEMI_FINALS",
    "FINAL",
    "LAST_16",
    "LAST_32",
    "LAST_64",
}

# Stages to always skip (friendlies, community shields, super cups, etc.)
SKIP_STAGES = {
    "FRIENDLY",
    "PRELIMINARY_ROUND",
    "QUALIFICATION",
    "QUALIFICATION_ROUND_1",
    "QUALIFICATION_ROUND_2",
    "QUALIFICATION_ROUND_3",
    "PLAYOFF_ROUND",
    "PLAYOFFS",
}


# ── Streak Calculation ───────────────────────────────────────────────────────
def calculate_streaks(matches: list, league_only: bool = True) -> dict:
    """
    Calculate winning streaks and goal-scoring streaks from match results.
    
    Args:
        matches: List of match dicts from the API.
        league_only: If True, skip friendlies, qualifiers, and playoff stages.
    
    Returns dict keyed by team name with streak data.
    """
    # Group matches by team, sorted by date
    team_matches = defaultdict(list)

    for match in matches:
        if match.get("status") != "FINISHED":
            continue

        # ── Stage filtering ──────────────────────────────────────────────
        stage = match.get("stage", "").upper()
        if league_only:
            # Skip if stage is explicitly in the skip list
            if stage in SKIP_STAGES:
                continue
            # If we know valid stages and the stage isn't one of them,
            # allow it only if VALID_STAGES is empty (unknown competition)
            if VALID_STAGES and stage and stage not in VALID_STAGES:
                continue

        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]
        match_date = match["utcDate"]

        # Home team record
        if home_goals > away_goals:
            result_home = "W"
        elif home_goals < away_goals:
            result_home = "L"
        else:
            result_home = "D"

        team_matches[home].append({
            "date": match_date,
            "goals_scored": home_goals,
            "goals_conceded": away_goals,
            "result": result_home,
            "opponent": away,
            "venue": "H",
        })

        # Away team record
        if away_goals > home_goals:
            result_away = "W"
        elif away_goals < home_goals:
            result_away = "L"
        else:
            result_away = "D"

        team_matches[away].append({
            "date": match_date,
            "goals_scored": away_goals,
            "goals_conceded": home_goals,
            "result": result_away,
            "opponent": home,
            "venue": "A",
        })

    # Calculate streaks per team
    streaks = {}
    for team, matches_list in team_matches.items():
        # Sort by date descending (most recent first)
        matches_list.sort(key=lambda x: x["date"], reverse=True)

        # Current winning streak (from most recent match backwards)
        win_streak = 0
        for m in matches_list:
            if m["result"] == "W":
                win_streak += 1
            else:
                break

        # Current unbeaten streak
        unbeaten_streak = 0
        for m in matches_list:
            if m["result"] in ("W", "D"):
                unbeaten_streak += 1
            else:
                break

        # Current goal-scoring streak (consecutive games scoring 1+)
        goal_streak = 0
        for m in matches_list:
            if m["goals_scored"] >= 1:
                goal_streak += 1
            else:
                break

        # Current clean sheet streak
        clean_sheet_streak = 0
        for m in matches_list:
            if m["goals_conceded"] == 0:
                clean_sheet_streak += 1
            else:
                break

        # Form guide (last 5 matches)
        form = "".join(
            {"W": "🟢", "D": "🟡", "L": "🔴"}[m["result"]]
            for m in matches_list[:5]
        )

        # Total goals in last 5
        recent_goals = sum(m["goals_scored"] for m in matches_list[:5])

        streaks[team] = {
            "win_streak": win_streak,
            "unbeaten_streak": unbeaten_streak,
            "goal_streak": goal_streak,
            "clean_sheet_streak": clean_sheet_streak,
            "form": form,
            "recent_goals": recent_goals,
            "matches_played": len(matches_list),
            "last_match": matches_list[0] if matches_list else None,
        }

    return streaks


# ── Formatting Helpers ────────────────────────────────────────────────────────
def format_winning_streaks(streaks: dict, league_name: str, top_n: int = 10) -> str:
    """Format top winning streaks as a readable message."""
    sorted_teams = sorted(
        streaks.items(), key=lambda x: x[1]["win_streak"], reverse=True
    )[:top_n]

    if not sorted_teams or sorted_teams[0][1]["win_streak"] == 0:
        return f"📊 *{league_name}*\nNo active winning streaks found."

    lines = [f"🔥 *{league_name} — Winning Streaks*\n"]
    medal = ["🥇", "🥈", "🥉"]

    for i, (team, data) in enumerate(sorted_teams):
        if data["win_streak"] == 0:
            break
        icon = medal[i] if i < 3 else f"  {i+1}."
        lines.append(
            f"{icon} *{team}*\n"
            f"    🏆 {data['win_streak']} wins in a row\n"
            f"    Form: {data['form']}"
        )

    return "\n".join(lines)


def format_goal_streaks(streaks: dict, league_name: str, top_n: int = 10) -> str:
    """Format top goal-scoring streaks as a readable message."""
    sorted_teams = sorted(
        streaks.items(), key=lambda x: x[1]["goal_streak"], reverse=True
    )[:top_n]

    if not sorted_teams or sorted_teams[0][1]["goal_streak"] == 0:
        return f"📊 *{league_name}*\nNo active goal-scoring streaks found."

    lines = [f"⚽ *{league_name} — Goal-Scoring Streaks*\n"]
    medal = ["🥇", "🥈", "🥉"]

    for i, (team, data) in enumerate(sorted_teams):
        if data["goal_streak"] == 0:
            break
        icon = medal[i] if i < 3 else f"  {i+1}."
        lines.append(
            f"{icon} *{team}*\n"
            f"    ⚽ Scored in {data['goal_streak']} consecutive games\n"
            f"    📈 {data['recent_goals']} goals in last 5 | Form: {data['form']}"
        )

    return "\n".join(lines)


def format_full_report(streaks: dict, league_name: str) -> str:
    """Full streak report for a league."""
    sorted_teams = sorted(
        streaks.items(),
        key=lambda x: (x[1]["win_streak"], x[1]["goal_streak"]),
        reverse=True,
    )[:15]

    lines = [f"📋 *{league_name} — Full Streak Report*\n"]

    for team, data in sorted_teams:
        streak_tags = []
        if data["win_streak"] >= 3:
            streak_tags.append(f"🔥{data['win_streak']}W")
        if data["unbeaten_streak"] >= 5:
            streak_tags.append(f"🛡️{data['unbeaten_streak']}U")
        if data["goal_streak"] >= 5:
            streak_tags.append(f"⚽{data['goal_streak']}G")
        if data["clean_sheet_streak"] >= 2:
            streak_tags.append(f"🧤{data['clean_sheet_streak']}CS")

        tags = " ".join(streak_tags) if streak_tags else "—"
        lines.append(f"▪️ *{team}* {data['form']}\n    {tags}")

    lines.append("\n_Legend: W=Win streak, U=Unbeaten, G=Goal streak, CS=Clean sheets_")
    return "\n".join(lines)


# ── Bot Command Handlers ─────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message."""
    free_count = len(FREE_LEAGUES)
    paid_count = len(PAID_LEAGUES)
    text = (
        "⚽ *Football Streak Tracker* ⚽\n\n"
        "I track winning streaks, goal-scoring streaks, and form "
        "across football leagues worldwide.\n\n"
        "*Commands:*\n"
        "/streaks — Top winning streaks (league matches only)\n"
        "/goals — Top goal-scoring streaks (league matches only)\n"
        "/league — Pick a specific league for detailed streaks\n"
        "/today — Today's matches\n"
        "/help — Show this message\n\n"
        f"📡 *{free_count} free leagues* + *{paid_count} paid leagues* available\n"
        f"_Free: {', '.join(l['name'] for l in FREE_LEAGUES.values())}_\n\n"
        "💡 _Paid leagues require a football-data.org subscription (€49+/mo)_\n"
        "🚫 _Cups, friendlies, qualifiers & playoffs are automatically filtered out_"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_streaks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show winning streaks across free-tier leagues."""
    await update.message.reply_text(
        "🔄 Fetching streak data (league matches only, skipping cups & friendlies)...\n"
        "_(This may take ~30 seconds due to API rate limits)_",
        parse_mode=ParseMode.MARKDOWN,
    )

    # Only scan free-tier leagues to stay within 10 req/min
    date_from = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")

    all_messages = []

    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        # Skip tournament-style comps (no active streaks between events)
        if info.get("type") == "CUP":
            continue
        try:
            # Respect rate limit: 10 req/min → ~6 sec between requests
            if i > 0:
                await asyncio.sleep(6)
            matches = await api.get_matches(code, date_from, date_to)
            if not matches:
                continue
            streaks = calculate_streaks(matches)
            msg = format_winning_streaks(streaks, f"{info['flag']} {info['name']}", top_n=5)
            all_messages.append(msg)
        except Exception as e:
            logger.error(f"Error fetching {code}: {e}")
            continue

    if all_messages:
        full = "\n\n".join(all_messages)
        if len(full) > 4000:
            for msg in all_messages:
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(full, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("❌ Could not fetch streak data. Try again later.")


async def cmd_goals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show goal-scoring streaks across free-tier leagues."""
    await update.message.reply_text(
        "🔄 Fetching goal streaks (league matches only, skipping cups & friendlies)...\n"
        "_(This may take ~30 seconds due to API rate limits)_",
        parse_mode=ParseMode.MARKDOWN,
    )

    date_from = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")

    all_messages = []

    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP":
            continue
        try:
            if i > 0:
                await asyncio.sleep(6)
            matches = await api.get_matches(code, date_from, date_to)
            if not matches:
                continue
            streaks = calculate_streaks(matches)
            msg = format_goal_streaks(streaks, f"{info['flag']} {info['name']}", top_n=5)
            all_messages.append(msg)
        except Exception as e:
            logger.error(f"Error fetching {code}: {e}")
            continue

    if all_messages:
        full = "\n\n".join(all_messages)
        if len(full) > 4000:
            for msg in all_messages:
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(full, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("❌ Could not fetch goal streak data.")


async def cmd_league(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show league selection by region."""
    keyboard = [
        [InlineKeyboardButton("🏴󠁧󠁢󠁥󠁮󠁧󠁿 England", callback_data="region_england")],
        [InlineKeyboardButton("🇪🇸 Spain", callback_data="region_spain")],
        [InlineKeyboardButton("🇩🇪 Germany", callback_data="region_germany")],
        [InlineKeyboardButton("🇮🇹 Italy", callback_data="region_italy")],
        [InlineKeyboardButton("🇫🇷 France", callback_data="region_france")],
        [
            InlineKeyboardButton("🇳🇱 Netherlands", callback_data="league_DED"),
            InlineKeyboardButton("🇵🇹 Portugal", callback_data="league_PPL"),
        ],
        [
            InlineKeyboardButton("🇧🇷 Brazil", callback_data="region_brazil"),
            InlineKeyboardButton("🇺🇸 USA", callback_data="league_MLS"),
        ],
        [InlineKeyboardButton("🏆 European Cups", callback_data="region_europe")],
        [InlineKeyboardButton("🌍 More Leagues", callback_data="region_more")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🏟️ *Select a region or league:*",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN,
    )


# ── Region sub-menus ─────────────────────────────────────────────────────────
REGION_MENUS = {
    "england": {
        "title": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Leagues",
        "leagues": ["PL", "ELC"],
    },
    "spain": {
        "title": "🇪🇸 Spanish Leagues",
        "leagues": ["PD", "SD"],
    },
    "germany": {
        "title": "🇩🇪 German Leagues",
        "leagues": ["BL1", "BL2"],
    },
    "italy": {
        "title": "🇮🇹 Italian Leagues",
        "leagues": ["SA", "SB"],
    },
    "france": {
        "title": "🇫🇷 French Leagues",
        "leagues": ["FL1", "FL2"],
    },
    "brazil": {
        "title": "🇧🇷 Brazilian Leagues",
        "leagues": ["BSA", "BSB"],
    },
    "europe": {
        "title": "🏆 European Competitions",
        "leagues": ["CL", "EL", "CLI", "WC", "EC"],
    },
    "more": {
        "title": "🌍 More Leagues",
        "leagues": ["MLS", "ASL", "JPL", "LMX", "JPB", "SPL", "DSL", "SSL", "TSL", "GSL"],
    },
}


async def region_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle region selection — show leagues within that region."""
    query = update.callback_query
    await query.answer()

    region_key = query.data.replace("region_", "")
    region = REGION_MENUS.get(region_key)
    if not region:
        await query.edit_message_text("❌ Unknown region.")
        return

    keyboard = []
    for code in region["leagues"]:
        info = LEAGUES.get(code)
        if not info:
            continue
        tier = "🆓" if code in FREE_LEAGUES else "💰"
        keyboard.append(
            [InlineKeyboardButton(
                f"{info['flag']} {info['name']} {tier}",
                callback_data=f"league_{code}",
            )]
        )
    keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_to_regions")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"*{region['title']}*\n🆓 = Free tier  💰 = Paid tier",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN,
    )


async def back_to_regions_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Go back to region selection."""
    query = update.callback_query
    await query.answer()

    keyboard = [
        [InlineKeyboardButton("🏴󠁧󠁢󠁥󠁮󠁧󠁿 England", callback_data="region_england")],
        [InlineKeyboardButton("🇪🇸 Spain", callback_data="region_spain")],
        [InlineKeyboardButton("🇩🇪 Germany", callback_data="region_germany")],
        [InlineKeyboardButton("🇮🇹 Italy", callback_data="region_italy")],
        [InlineKeyboardButton("🇫🇷 France", callback_data="region_france")],
        [
            InlineKeyboardButton("🇳🇱 Netherlands", callback_data="league_DED"),
            InlineKeyboardButton("🇵🇹 Portugal", callback_data="league_PPL"),
        ],
        [
            InlineKeyboardButton("🇧🇷 Brazil", callback_data="region_brazil"),
            InlineKeyboardButton("🇺🇸 USA", callback_data="league_MLS"),
        ],
        [InlineKeyboardButton("🏆 European Cups", callback_data="region_europe")],
        [InlineKeyboardButton("🌍 More Leagues", callback_data="region_more")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "🏟️ *Select a region or league:*",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN,
    )


async def league_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle league selection from inline keyboard."""
    query = update.callback_query
    await query.answer()

    code = query.data.replace("league_", "")
    info = LEAGUES.get(code)
    if not info:
        await query.edit_message_text("❌ Unknown league.")
        return

    is_paid = code in PAID_LEAGUES
    await query.edit_message_text(f"🔄 Fetching {info['name']} streak data...")

    date_from = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        matches = await api.get_matches(code, date_from, date_to)
        if not matches:
            await query.edit_message_text(f"No recent matches found for {info['name']}.")
            return

        streaks = calculate_streaks(matches)
        report = format_full_report(streaks, f"{info['flag']} {info['name']}")
        await query.edit_message_text(report, parse_mode=ParseMode.MARKDOWN)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and is_paid:
            await query.edit_message_text(
                f"🔒 *{info['name']}* requires a paid football-data.org subscription.\n\n"
                "Upgrade at: https://www.football-data.org/pricing\n\n"
                "Your free tier covers: PL, La Liga, Bundesliga, Serie A, "
                "Ligue 1, Eredivisie, Primeira Liga, Championship, "
                "Brazilian Série A, Champions League, World Cup, and Euros.",
                parse_mode=ParseMode.MARKDOWN,
            )
        elif e.response.status_code == 429:
            await query.edit_message_text(
                "⏱️ Rate limit hit (10 req/min on free tier). Wait a minute and try again."
            )
        else:
            await query.edit_message_text(f"❌ API error ({e.response.status_code}): {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        await query.edit_message_text(f"❌ Error fetching data: {e}")


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's matches across free-tier leagues."""
    await update.message.reply_text("🔄 Checking today's fixtures...")

    all_lines = ["📅 *Today's Matches*\n"]
    found_any = False

    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP":
            continue
        try:
            if i > 0:
                await asyncio.sleep(6)
            matches = await api.get_todays_matches(code)
            if not matches:
                continue

            found_any = True
            all_lines.append(f"\n{info['flag']} *{info['name']}*")

            for m in matches:
                home = m["homeTeam"]["name"]
                away = m["awayTeam"]["name"]
                status = m["status"]

                if status == "FINISHED":
                    h = m["score"]["fullTime"]["home"]
                    a = m["score"]["fullTime"]["away"]
                    all_lines.append(f"  ✅ {home} {h} - {a} {away}")
                elif status in ("IN_PLAY", "PAUSED"):
                    h = m["score"]["fullTime"]["home"] or 0
                    a = m["score"]["fullTime"]["away"] or 0
                    all_lines.append(f"  🔴 {home} {h} - {a} {away} (LIVE)")
                else:
                    time_str = m.get("utcDate", "")
                    if time_str:
                        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                        kick_off = dt.strftime("%H:%M UTC")
                    else:
                        kick_off = "TBD"
                    all_lines.append(f"  🕐 {home} vs {away} ({kick_off})")

        except Exception as e:
            logger.error(f"Error fetching today's {code}: {e}")
            continue

    if not found_any:
        all_lines.append("\nNo matches scheduled today across tracked leagues.")

    await update.message.reply_text(
        "\n".join(all_lines), parse_mode=ParseMode.MARKDOWN
    )


# ── Scheduled Daily Summary (optional) ───────────────────────────────────────
async def daily_summary(context: ContextTypes.DEFAULT_TYPE):
    """
    Send daily streak summary to subscribed chats.
    Configure DAILY_CHAT_ID env var with a chat ID to receive daily updates.
    """
    chat_id = os.environ.get("DAILY_CHAT_ID")
    if not chat_id:
        return

    date_from = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")

    summary_lines = ["🌅 *Daily Streak Summary*\n"]

    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP":
            continue
        try:
            if i > 0:
                await asyncio.sleep(6)
            matches = await api.get_matches(code, date_from, date_to)
            if not matches:
                continue

            streaks = calculate_streaks(matches)
            # Find notable streaks (3+ wins or 5+ goals)
            notable = {
                team: data
                for team, data in streaks.items()
                if data["win_streak"] >= 3 or data["goal_streak"] >= 5
            }

            if notable:
                summary_lines.append(f"\n{info['flag']} *{info['name']}*")
                for team, data in sorted(
                    notable.items(), key=lambda x: x[1]["win_streak"], reverse=True
                ):
                    parts = []
                    if data["win_streak"] >= 3:
                        parts.append(f"🔥 {data['win_streak']}W")
                    if data["goal_streak"] >= 5:
                        parts.append(f"⚽ {data['goal_streak']}G")
                    summary_lines.append(f"  {team}: {' | '.join(parts)} {data['form']}")

        except Exception as e:
            logger.error(f"Daily summary error {code}: {e}")

    if len(summary_lines) > 1:
        await context.bot.send_message(
            chat_id=int(chat_id),
            text="\n".join(summary_lines),
            parse_mode=ParseMode.MARKDOWN,
        )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    """Start the bot."""
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("⚠️  Set TELEGRAM_BOT_TOKEN environment variable!")
        print("   Get one from @BotFather on Telegram")
        return

    if FOOTBALL_API_KEY == "YOUR_FOOTBALL_DATA_API_KEY":
        print("⚠️  Set FOOTBALL_API_KEY environment variable!")
        print("   Get a free key at https://www.football-data.org/client/register")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("streaks", cmd_streaks))
    app.add_handler(CommandHandler("goals", cmd_goals))
    app.add_handler(CommandHandler("league", cmd_league))
    app.add_handler(CommandHandler("today", cmd_today))

    # Callback handler for inline keyboards
    app.add_handler(CallbackQueryHandler(league_callback, pattern=r"^league_"))
    app.add_handler(CallbackQueryHandler(region_callback, pattern=r"^region_"))
    app.add_handler(CallbackQueryHandler(back_to_regions_callback, pattern=r"^back_to_regions$"))

    # Optional: Schedule daily summary at 08:00 UTC
    job_queue = app.job_queue
    if job_queue and os.environ.get("DAILY_CHAT_ID"):
        from datetime import time as dt_time
        job_queue.run_daily(
            daily_summary,
            time=dt_time(hour=8, minute=0),
            name="daily_streak_summary",
        )
        logger.info("Daily summary scheduled for 08:00 UTC")

    print("🤖 Football Streak Tracker is running!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
