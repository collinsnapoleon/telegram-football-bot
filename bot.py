"""
Football Streak Tracker & Predictor — Telegram Bot
====================================================
Aggregates daily football winning streaks, goal-scoring streaks,
and uses statistical modelling (Poisson, form weighting, strength
ratings) to predict match outcomes.

Commands:
  /start         - Welcome message & instructions
  /streaks       - Top winning streaks across leagues
  /goals         - Top goal-scoring streaks
  /league        - Pick a league for detailed streaks
  /today         - Today's matches
  /predict       - AI predictions for today's matches
  /tips          - Best picks of the day (highest confidence)
  /help          - Show all commands
"""

import os
import logging
import json
import asyncio
import math
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

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "YOUR_FOOTBALL_DATA_API_KEY")
FOOTBALL_API_BASE = "https://api.football-data.org/v4"

FREE_LEAGUES = {
    "PL": {"name": "Premier League", "flag": "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f", "country": "England", "type": "LEAGUE"},
    "ELC": {"name": "Championship", "flag": "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f", "country": "England", "type": "LEAGUE"},
    "PD": {"name": "La Liga", "flag": "\U0001f1ea\U0001f1f8", "country": "Spain", "type": "LEAGUE"},
    "BL1": {"name": "Bundesliga", "flag": "\U0001f1e9\U0001f1ea", "country": "Germany", "type": "LEAGUE"},
    "SA": {"name": "Serie A", "flag": "\U0001f1ee\U0001f1f9", "country": "Italy", "type": "LEAGUE"},
    "FL1": {"name": "Ligue 1", "flag": "\U0001f1eb\U0001f1f7", "country": "France", "type": "LEAGUE"},
    "DED": {"name": "Eredivisie", "flag": "\U0001f1f3\U0001f1f1", "country": "Netherlands", "type": "LEAGUE"},
    "PPL": {"name": "Primeira Liga", "flag": "\U0001f1f5\U0001f1f9", "country": "Portugal", "type": "LEAGUE"},
    "BSA": {"name": "Serie A", "flag": "\U0001f1e7\U0001f1f7", "country": "Brazil", "type": "LEAGUE"},
    "CL": {"name": "Champions League", "flag": "\U0001f3c6", "country": "Europe", "type": "CUP"},
    "WC": {"name": "FIFA World Cup", "flag": "\U0001f30d", "country": "World", "type": "CUP"},
    "EC": {"name": "European Championship", "flag": "\U0001f1ea\U0001f1fa", "country": "Europe", "type": "CUP"},
}

PAID_LEAGUES = {
    "EL": {"name": "Europa League", "flag": "\U0001f7e0", "country": "Europe", "type": "CUP"},
    "CLI": {"name": "Conference League", "flag": "\U0001f7e2", "country": "Europe", "type": "CUP"},
    "SD": {"name": "Segunda Division", "flag": "\U0001f1ea\U0001f1f8", "country": "Spain", "type": "LEAGUE"},
    "BL2": {"name": "2. Bundesliga", "flag": "\U0001f1e9\U0001f1ea", "country": "Germany", "type": "LEAGUE"},
    "SB": {"name": "Serie B", "flag": "\U0001f1ee\U0001f1f9", "country": "Italy", "type": "LEAGUE"},
    "FL2": {"name": "Ligue 2", "flag": "\U0001f1eb\U0001f1f7", "country": "France", "type": "LEAGUE"},
    "BSB": {"name": "Serie B", "flag": "\U0001f1e7\U0001f1f7", "country": "Brazil", "type": "LEAGUE"},
    "MLS": {"name": "MLS", "flag": "\U0001f1fa\U0001f1f8", "country": "USA", "type": "LEAGUE"},
    "ASL": {"name": "A-League", "flag": "\U0001f1e6\U0001f1fa", "country": "Australia", "type": "LEAGUE"},
    "JPL": {"name": "J. League", "flag": "\U0001f1ef\U0001f1f5", "country": "Japan", "type": "LEAGUE"},
    "LMX": {"name": "Liga MX", "flag": "\U0001f1f2\U0001f1fd", "country": "Mexico", "type": "LEAGUE"},
    "JPB": {"name": "Jupiler Pro League", "flag": "\U0001f1e7\U0001f1ea", "country": "Belgium", "type": "LEAGUE"},
    "SPL": {"name": "Scottish Premiership", "flag": "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f", "country": "Scotland", "type": "LEAGUE"},
    "DSL": {"name": "Superliga", "flag": "\U0001f1e9\U0001f1f0", "country": "Denmark", "type": "LEAGUE"},
    "SSL": {"name": "Super League", "flag": "\U0001f1e8\U0001f1ed", "country": "Switzerland", "type": "LEAGUE"},
    "TSL": {"name": "Super Lig", "flag": "\U0001f1f9\U0001f1f7", "country": "Turkey", "type": "LEAGUE"},
    "GSL": {"name": "Super League", "flag": "\U0001f1ec\U0001f1f7", "country": "Greece", "type": "LEAGUE"},
}

LEAGUES = {**FREE_LEAGUES, **PAID_LEAGUES}

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"X-Auth-Token": api_key}

    async def _get(self, endpoint, params=None):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{FOOTBALL_API_BASE}{endpoint}",
                headers=self.headers, params=params or {}, timeout=15,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_matches(self, league_code, date_from=None, date_to=None):
        params = {"status": "FINISHED"}
        if date_from: params["dateFrom"] = date_from
        if date_to: params["dateTo"] = date_to
        data = await self._get(f"/competitions/{league_code}/matches", params)
        return data.get("matches", [])

    async def get_todays_matches(self, league_code):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        params = {"dateFrom": today, "dateTo": today}
        data = await self._get(f"/competitions/{league_code}/matches", params)
        return data.get("matches", [])

    async def get_standings(self, league_code):
        data = await self._get(f"/competitions/{league_code}/standings")
        for s in data.get("standings", []):
            if s.get("type") == "TOTAL":
                return s.get("table", [])
        return []


api = FootballAPI(FOOTBALL_API_KEY)

VALID_STAGES = {
    "REGULAR_SEASON", "GROUP_STAGE", "LEAGUE_STAGE",
    "ROUND_OF_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL",
    "LAST_16", "LAST_32", "LAST_64",
}
SKIP_STAGES = {
    "FRIENDLY", "PRELIMINARY_ROUND", "QUALIFICATION",
    "QUALIFICATION_ROUND_1", "QUALIFICATION_ROUND_2", "QUALIFICATION_ROUND_3",
    "PLAYOFF_ROUND", "PLAYOFFS",
}


def is_valid_stage(match):
    stage = match.get("stage", "").upper()
    if stage in SKIP_STAGES: return False
    if VALID_STAGES and stage and stage not in VALID_STAGES: return False
    return True


def calculate_streaks(matches, league_only=True):
    team_matches = defaultdict(list)
    for match in matches:
        if match.get("status") != "FINISHED": continue
        if league_only and not is_valid_stage(match): continue
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        hg = match["score"]["fullTime"]["home"]
        ag = match["score"]["fullTime"]["away"]
        if hg is None or ag is None: continue
        md = match["utcDate"]
        rh = "W" if hg > ag else ("L" if hg < ag else "D")
        ra = "W" if ag > hg else ("L" if ag < hg else "D")
        team_matches[home].append({"date": md, "goals_scored": hg, "goals_conceded": ag, "result": rh, "opponent": away, "venue": "H"})
        team_matches[away].append({"date": md, "goals_scored": ag, "goals_conceded": hg, "result": ra, "opponent": home, "venue": "A"})

    streaks = {}
    for team, ml in team_matches.items():
        ml.sort(key=lambda x: x["date"], reverse=True)
        ws = 0
        for m in ml:
            if m["result"] == "W": ws += 1
            else: break
        ub = 0
        for m in ml:
            if m["result"] in ("W", "D"): ub += 1
            else: break
        gs = 0
        for m in ml:
            if m["goals_scored"] >= 1: gs += 1
            else: break
        cs = 0
        for m in ml:
            if m["goals_conceded"] == 0: cs += 1
            else: break
        form = "".join({"W": chr(0x1F7E2), "D": chr(0x1F7E1), "L": chr(0x1F534)}[m["result"]] for m in ml[:5])
        rg = sum(m["goals_scored"] for m in ml[:5])
        streaks[team] = {"win_streak": ws, "unbeaten_streak": ub, "goal_streak": gs, "clean_sheet_streak": cs, "form": form, "recent_goals": rg, "matches_played": len(ml), "last_match": ml[0] if ml else None}
    return streaks


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE — Poisson Model + Form Weighting + Streaks
# ══════════════════════════════════════════════════════════════════════════════

def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def build_team_stats(matches):
    """Build per-team attack/defense strength ratings from historical matches."""
    team_data = defaultdict(lambda: {
        "home_scored": [], "home_conceded": [],
        "away_scored": [], "away_conceded": [],
        "all_scored": [], "all_conceded": [],
        "results": [],
        "h2h": defaultdict(lambda: {"scored": [], "conceded": [], "results": []}),
    })

    finished = [m for m in matches if m.get("status") == "FINISHED" and is_valid_stage(m)]

    for match in finished:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        hg = match["score"]["fullTime"]["home"]
        ag = match["score"]["fullTime"]["away"]
        date = match["utcDate"]
        if hg is None or ag is None: continue

        team_data[home]["home_scored"].append(hg)
        team_data[home]["home_conceded"].append(ag)
        team_data[home]["all_scored"].append(hg)
        team_data[home]["all_conceded"].append(ag)
        team_data[home]["results"].append((date, "W" if hg > ag else "L" if hg < ag else "D"))

        team_data[away]["away_scored"].append(ag)
        team_data[away]["away_conceded"].append(hg)
        team_data[away]["all_scored"].append(ag)
        team_data[away]["all_conceded"].append(hg)
        team_data[away]["results"].append((date, "W" if ag > hg else "L" if ag < hg else "D"))

        team_data[home]["h2h"][away]["scored"].append(hg)
        team_data[home]["h2h"][away]["conceded"].append(ag)
        team_data[home]["h2h"][away]["results"].append("W" if hg > ag else "L" if hg < ag else "D")
        team_data[away]["h2h"][home]["scored"].append(ag)
        team_data[away]["h2h"][home]["conceded"].append(hg)
        team_data[away]["h2h"][home]["results"].append("W" if ag > hg else "L" if ag < hg else "D")

    all_hg = [m["score"]["fullTime"]["home"] for m in finished if m["score"]["fullTime"]["home"] is not None]
    all_ag = [m["score"]["fullTime"]["away"] for m in finished if m["score"]["fullTime"]["away"] is not None]
    league_avg_home = sum(all_hg) / max(len(all_hg), 1)
    league_avg_away = sum(all_ag) / max(len(all_ag), 1)

    stats = {"_league_avg_home": league_avg_home, "_league_avg_away": league_avg_away}

    for team, data in team_data.items():
        n_all = max(len(data["all_scored"]), 1)
        n_home = max(len(data["home_scored"]), 1)
        n_away = max(len(data["away_scored"]), 1)

        avg_scored = sum(data["all_scored"]) / n_all
        avg_conceded = sum(data["all_conceded"]) / n_all
        home_avg_scored = sum(data["home_scored"]) / n_home
        away_avg_scored = sum(data["away_scored"]) / n_away

        attack_strength = avg_scored / max(league_avg_home, 0.5)
        defense_strength = avg_conceded / max(league_avg_away, 0.5)
        home_attack = home_avg_scored / max(league_avg_home, 0.5)
        home_defense = sum(data["home_conceded"]) / n_home / max(league_avg_away, 0.5)
        away_attack = away_avg_scored / max(league_avg_away, 0.5)
        away_defense = sum(data["away_conceded"]) / n_away / max(league_avg_home, 0.5)

        data["results"].sort(key=lambda x: x[0], reverse=True)
        form_points = 0
        form_weight_total = 0
        for i, (_, result) in enumerate(data["results"][:10]):
            weight = math.exp(-0.15 * i)
            form_weight_total += weight
            if result == "W": form_points += 3 * weight
            elif result == "D": form_points += 1 * weight
        form_rating = form_points / max(form_weight_total, 1) / 3.0

        results_only = [r for _, r in data["results"]]
        total_games = max(len(results_only), 1)

        btts_count = sum(1 for s, c in zip(data["all_scored"], data["all_conceded"]) if s >= 1 and c >= 1)
        over25_count = sum(1 for s, c in zip(data["all_scored"], data["all_conceded"]) if s + c >= 3)

        stats[team] = {
            "attack_strength": attack_strength, "defense_strength": defense_strength,
            "home_attack": home_attack, "home_defense": home_defense,
            "away_attack": away_attack, "away_defense": away_defense,
            "avg_scored": avg_scored, "avg_conceded": avg_conceded,
            "home_avg_scored": home_avg_scored, "away_avg_scored": away_avg_scored,
            "form_rating": form_rating,
            "win_rate": results_only.count("W") / total_games,
            "draw_rate": results_only.count("D") / total_games,
            "loss_rate": results_only.count("L") / total_games,
            "btts_rate": btts_count / n_all,
            "over25_rate": over25_count / n_all,
            "matches_played": len(data["results"]),
            "h2h": data["h2h"],
        }

    return stats


def predict_match(home_team, away_team, stats, streaks):
    """Generate full prediction using Poisson model + form + streaks + H2H."""
    hs = stats.get(home_team)
    aws = stats.get(away_team)
    lah = stats.get("_league_avg_home", 1.4)
    laa = stats.get("_league_avg_away", 1.1)

    if not hs or not aws:
        return {
            "home_team": home_team, "away_team": away_team,
            "home_xg": lah, "away_xg": laa,
            "home_win": 0.45, "draw": 0.25, "away_win": 0.30,
            "over25": 0.50, "btts": 0.50,
            "confidence": "\u26aa Low", "confidence_score": 0.3,
            "prediction": "Insufficient data", "goals_prediction": "~2-3 goals",
            "factors": ["Limited data available"], "score_line": "? - ?",
            "home_form": "", "away_form": "",
        }

    # Step 1: Base xG (Poisson)
    home_xg = hs["home_attack"] * aws["away_defense"] * lah
    away_xg = aws["away_attack"] * hs["home_defense"] * laa

    # Step 2: Home advantage (+8%)
    home_xg *= 1.08

    # Step 3: Form adjustment (+/- 15%)
    form_diff = hs["form_rating"] - aws["form_rating"]
    home_xg *= (1 + form_diff * 0.15)
    away_xg *= (1 - form_diff * 0.15)

    # Step 4: Streak momentum (+/- 10%)
    hstreak = streaks.get(home_team, {})
    astreak = streaks.get(away_team, {})
    home_xg *= (1 + min(hstreak.get("win_streak", 0), 5) * 0.02)
    away_xg *= (1 + min(astreak.get("win_streak", 0), 5) * 0.02)
    if hstreak.get("goal_streak", 0) >= 5: home_xg *= 1.05
    if astreak.get("goal_streak", 0) >= 5: away_xg *= 1.05

    # Step 5: H2H adjustment (+/- 8%)
    h2h = hs.get("h2h", {}).get(away_team, {})
    h2h_results = h2h.get("results", [])
    if len(h2h_results) >= 3:
        h2h_dom = (h2h_results.count("W") / len(h2h_results)) - 0.5
        home_xg *= (1 + h2h_dom * 0.16)
        away_xg *= (1 - h2h_dom * 0.16)

    home_xg = max(0.3, min(home_xg, 4.5))
    away_xg = max(0.2, min(away_xg, 4.0))

    # Step 6: Poisson probabilities
    hw, dr, aw = 0.0, 0.0, 0.0
    score_probs = {}
    for hg in range(9):
        for ag in range(9):
            p = poisson_pmf(hg, home_xg) * poisson_pmf(ag, away_xg)
            score_probs[(hg, ag)] = p
            if hg > ag: hw += p
            elif hg == ag: dr += p
            else: aw += p

    total = hw + dr + aw
    if total > 0: hw /= total; dr /= total; aw /= total

    # Step 7: Over/Under & BTTS
    o25 = sum(p for (hg, ag), p in score_probs.items() if hg + ag >= 3)
    btts = sum(p for (hg, ag), p in score_probs.items() if hg >= 1 and ag >= 1)
    hist_btts = (hs["btts_rate"] + aws["btts_rate"]) / 2
    hist_o25 = (hs["over25_rate"] + aws["over25_rate"]) / 2
    btts = btts * 0.6 + hist_btts * 0.4
    o25 = o25 * 0.6 + hist_o25 * 0.4

    # Step 8: Most likely score
    best_score = max(score_probs, key=score_probs.get)

    # Step 9: Confidence
    max_prob = max(hw, dr, aw)
    data_q = min(min(hs["matches_played"], aws["matches_played"]) / 15, 1.0)
    conf_score = min(max_prob * 0.6 + data_q * 0.25 + abs(form_diff) * 0.15, 1.0)
    if conf_score >= 0.7: conf = "\U0001f7e2 High"
    elif conf_score >= 0.5: conf = "\U0001f7e1 Medium"
    else: conf = "\u26aa Low"

    # Step 10: Prediction text
    if hw > aw and hw > dr:
        pred = f"\U0001f3e0 {home_team} Win" if hw >= 0.55 else f"\U0001f3e0 {home_team} Slight Edge"
    elif aw > hw and aw > dr:
        pred = f"\u2708\ufe0f {away_team} Win" if aw >= 0.55 else f"\u2708\ufe0f {away_team} Slight Edge"
    else:
        pred = "\U0001f91d Draw Likely"

    factors = []
    if hstreak.get("win_streak", 0) >= 3: factors.append(f"\U0001f525 {home_team} on {hstreak['win_streak']}-game win streak")
    if astreak.get("win_streak", 0) >= 3: factors.append(f"\U0001f525 {away_team} on {astreak['win_streak']}-game win streak")
    if hs["form_rating"] > 0.7: factors.append(f"\U0001f4c8 {home_team} in excellent form")
    if aws["form_rating"] > 0.7: factors.append(f"\U0001f4c8 {away_team} in excellent form")
    if hs["form_rating"] < 0.3: factors.append(f"\U0001f4c9 {home_team} in poor form")
    if aws["form_rating"] < 0.3: factors.append(f"\U0001f4c9 {away_team} in poor form")
    if o25 >= 0.65: factors.append("\u26a1 High-scoring match expected")
    if btts >= 0.65: factors.append("\u26bd Both teams likely to score")
    if hstreak.get("clean_sheet_streak", 0) >= 3: factors.append(f"\U0001f9e4 {home_team} {hstreak['clean_sheet_streak']} clean sheets in a row")
    if len(h2h_results) >= 3:
        h2hw = h2h_results.count("W")
        if h2hw >= len(h2h_results) * 0.6: factors.append(f"\U0001f4ca {home_team} dominates H2H ({h2hw}/{len(h2h_results)} wins)")

    txg = home_xg + away_xg
    if txg >= 3.5: gp = "High scoring (3+ goals expected)"
    elif txg >= 2.5: gp = "Moderate (2-3 goals expected)"
    elif txg >= 1.5: gp = "Low scoring (1-2 goals expected)"
    else: gp = "Very tight (0-1 goals expected)"

    return {
        "home_team": home_team, "away_team": away_team,
        "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2),
        "home_win": round(hw, 3), "draw": round(dr, 3), "away_win": round(aw, 3),
        "over25": round(o25, 3), "btts": round(btts, 3),
        "confidence": conf, "confidence_score": round(conf_score, 3),
        "prediction": pred, "goals_prediction": gp,
        "factors": factors[:4], "score_line": f"{best_score[0]} - {best_score[1]}",
        "home_form": hstreak.get("form", ""), "away_form": astreak.get("form", ""),
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def format_prediction(pred):
    lines = [
        f"\u26bd *{pred['home_team']}* vs *{pred['away_team']}*",
        f"",
        f"\U0001f3af *Prediction:* {pred['prediction']}",
        f"\U0001f4ca Most likely score: *{pred['score_line']}*",
        f"",
        f"\U0001f4c8 *Probabilities:*",
        f"   \U0001f3e0 Home: {pred['home_win']:.0%}  \U0001f91d Draw: {pred['draw']:.0%}  \u2708\ufe0f Away: {pred['away_win']:.0%}",
        f"",
        f"\u26bd *Goals:*",
        f"   xG: {pred['home_team'][:3].upper()} {pred['home_xg']} - {pred['away_xg']} {pred['away_team'][:3].upper()}",
        f"   Over 2.5: {pred['over25']:.0%} | BTTS: {pred['btts']:.0%}",
        f"   \U0001f4cb {pred['goals_prediction']}",
    ]
    if pred.get("home_form") or pred.get("away_form"):
        lines.append(f"")
        lines.append(f"\U0001f4ca *Form:*")
        if pred.get("home_form"): lines.append(f"   \U0001f3e0 {pred['home_team'][:15]}: {pred['home_form']}")
        if pred.get("away_form"): lines.append(f"   \u2708\ufe0f {pred['away_team'][:15]}: {pred['away_form']}")
    if pred["factors"]:
        lines.append(f"")
        lines.append(f"\U0001f4a1 *Key Factors:*")
        for f in pred["factors"]: lines.append(f"   {f}")
    lines.append(f"")
    lines.append(f"\U0001f3af Confidence: {pred['confidence']}")
    lines.append(f"\u2500" * 30)
    return "\n".join(lines)


def format_tip(pred, rank):
    medal = ["\U0001f947", "\U0001f948", "\U0001f949"]
    icon = medal[rank] if rank < 3 else f"  {rank + 1}."
    lines = [
        f"{icon} *{pred['home_team']}* vs *{pred['away_team']}*",
        f"   \U0001f3af {pred['prediction']}",
        f"   \U0001f4ca Score: {pred['score_line']} | xG: {pred['home_xg']} - {pred['away_xg']}",
        f"   \U0001f3e0 {pred['home_win']:.0%} | \U0001f91d {pred['draw']:.0%} | \u2708\ufe0f {pred['away_win']:.0%}",
    ]
    angles = []
    if pred["over25"] >= 0.60: angles.append(f"Over 2.5 ({pred['over25']:.0%})")
    if pred["btts"] >= 0.60: angles.append(f"BTTS ({pred['btts']:.0%})")
    if pred["home_win"] >= 0.55: angles.append(f"Home Win ({pred['home_win']:.0%})")
    if pred["away_win"] >= 0.55: angles.append(f"Away Win ({pred['away_win']:.0%})")
    if angles: lines.append(f"   \U0001f4b0 Best angles: {' | '.join(angles)}")
    lines.append(f"   Confidence: {pred['confidence']}")
    return "\n".join(lines)


def format_winning_streaks(streaks, league_name, top_n=10):
    sorted_teams = sorted(streaks.items(), key=lambda x: x[1]["win_streak"], reverse=True)[:top_n]
    if not sorted_teams or sorted_teams[0][1]["win_streak"] == 0:
        return f"\U0001f4ca *{league_name}*\nNo active winning streaks found."
    lines = [f"\U0001f525 *{league_name} \u2014 Winning Streaks*\n"]
    medal = ["\U0001f947", "\U0001f948", "\U0001f949"]
    for i, (team, data) in enumerate(sorted_teams):
        if data["win_streak"] == 0: break
        icon = medal[i] if i < 3 else f"  {i+1}."
        lines.append(f"{icon} *{team}*\n    \U0001f3c6 {data['win_streak']} wins in a row\n    Form: {data['form']}")
    return "\n".join(lines)


def format_goal_streaks(streaks, league_name, top_n=10):
    sorted_teams = sorted(streaks.items(), key=lambda x: x[1]["goal_streak"], reverse=True)[:top_n]
    if not sorted_teams or sorted_teams[0][1]["goal_streak"] == 0:
        return f"\U0001f4ca *{league_name}*\nNo active goal-scoring streaks found."
    lines = [f"\u26bd *{league_name} \u2014 Goal-Scoring Streaks*\n"]
    medal = ["\U0001f947", "\U0001f948", "\U0001f949"]
    for i, (team, data) in enumerate(sorted_teams):
        if data["goal_streak"] == 0: break
        icon = medal[i] if i < 3 else f"  {i+1}."
        lines.append(f"{icon} *{team}*\n    \u26bd Scored in {data['goal_streak']} consecutive games\n    \U0001f4c8 {data['recent_goals']} goals in last 5 | Form: {data['form']}")
    return "\n".join(lines)


def format_full_report(streaks, league_name):
    sorted_teams = sorted(streaks.items(), key=lambda x: (x[1]["win_streak"], x[1]["goal_streak"]), reverse=True)[:15]
    lines = [f"\U0001f4cb *{league_name} \u2014 Full Streak Report*\n"]
    for team, data in sorted_teams:
        tags = []
        if data["win_streak"] >= 3: tags.append(f"\U0001f525{data['win_streak']}W")
        if data["unbeaten_streak"] >= 5: tags.append(f"\U0001f6e1\ufe0f{data['unbeaten_streak']}U")
        if data["goal_streak"] >= 5: tags.append(f"\u26bd{data['goal_streak']}G")
        if data["clean_sheet_streak"] >= 2: tags.append(f"\U0001f9e4{data['clean_sheet_streak']}CS")
        t = " ".join(tags) if tags else "\u2014"
        lines.append(f"\u25aa\ufe0f *{team}* {data['form']}\n    {t}")
    lines.append("\n_Legend: W=Win streak, U=Unbeaten, G=Goal streak, CS=Clean sheets_")
    return "\n".join(lines)


# ── Command Handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "\u26bd *Football Streak Tracker & Predictor* \u26bd\n\n"
        "I track streaks, analyse form, and use statistical modelling "
        "to predict match outcomes.\n\n"
        "*\U0001f4ca Streak Commands:*\n"
        "/streaks \u2014 Top winning streaks\n"
        "/goals \u2014 Top goal-scoring streaks\n"
        "/league \u2014 Pick a league for detailed streaks\n"
        "/today \u2014 Today's matches\n\n"
        "*\U0001f916 Prediction Commands:*\n"
        "/predict \u2014 AI predictions for today's matches\n"
        "/tips \u2014 Best picks of the day (highest confidence)\n\n"
        "/help \u2014 Show this message\n\n"
        "\U0001f9e0 _Predictions use Poisson modelling, form weighting,_\n"
        "_streak momentum, and head-to-head analysis_\n\n"
        "\u26a0\ufe0f _Predictions are statistical estimates, not guarantees._\n"
        "_Please gamble responsibly._"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_streaks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f504 Fetching streak data (league matches only)...\n_(This may take ~60 seconds)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    date_from = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")
    all_messages = []
    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP": continue
        try:
            if i > 0: await asyncio.sleep(6)
            matches = await api.get_matches(code, date_from, date_to)
            if not matches: continue
            streaks = calculate_streaks(matches)
            msg = format_winning_streaks(streaks, f"{info['flag']} {info['name']}", top_n=5)
            all_messages.append(msg)
        except Exception as e:
            logger.error(f"Error fetching {code}: {e}")
    if all_messages:
        full = "\n\n".join(all_messages)
        if len(full) > 4000:
            for msg in all_messages:
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(full, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("\u274c Could not fetch streak data.")


async def cmd_goals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f504 Fetching goal streaks (league matches only)...\n_(This may take ~60 seconds)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    date_from = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")
    all_messages = []
    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP": continue
        try:
            if i > 0: await asyncio.sleep(6)
            matches = await api.get_matches(code, date_from, date_to)
            if not matches: continue
            streaks = calculate_streaks(matches)
            msg = format_goal_streaks(streaks, f"{info['flag']} {info['name']}", top_n=5)
            all_messages.append(msg)
        except Exception as e:
            logger.error(f"Error fetching {code}: {e}")
    if all_messages:
        full = "\n\n".join(all_messages)
        if len(full) > 4000:
            for msg in all_messages: await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(full, parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("\u274c Could not fetch goal streak data.")


async def cmd_league(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f England", callback_data="region_england")],
        [InlineKeyboardButton("\U0001f1ea\U0001f1f8 Spain", callback_data="region_spain")],
        [InlineKeyboardButton("\U0001f1e9\U0001f1ea Germany", callback_data="region_germany")],
        [InlineKeyboardButton("\U0001f1ee\U0001f1f9 Italy", callback_data="region_italy")],
        [InlineKeyboardButton("\U0001f1eb\U0001f1f7 France", callback_data="region_france")],
        [InlineKeyboardButton("\U0001f1f3\U0001f1f1 Netherlands", callback_data="league_DED"), InlineKeyboardButton("\U0001f1f5\U0001f1f9 Portugal", callback_data="league_PPL")],
        [InlineKeyboardButton("\U0001f1e7\U0001f1f7 Brazil", callback_data="region_brazil"), InlineKeyboardButton("\U0001f1fa\U0001f1f8 USA", callback_data="league_MLS")],
        [InlineKeyboardButton("\U0001f3c6 European Cups", callback_data="region_europe")],
        [InlineKeyboardButton("\U0001f30d More Leagues", callback_data="region_more")],
    ]
    await update.message.reply_text("\U0001f3df\ufe0f *Select a region or league:*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)


REGION_MENUS = {
    "england": {"title": "English Leagues", "leagues": ["PL", "ELC"]},
    "spain": {"title": "Spanish Leagues", "leagues": ["PD", "SD"]},
    "germany": {"title": "German Leagues", "leagues": ["BL1", "BL2"]},
    "italy": {"title": "Italian Leagues", "leagues": ["SA", "SB"]},
    "france": {"title": "French Leagues", "leagues": ["FL1", "FL2"]},
    "brazil": {"title": "Brazilian Leagues", "leagues": ["BSA", "BSB"]},
    "europe": {"title": "European Competitions", "leagues": ["CL", "EL", "CLI", "WC", "EC"]},
    "more": {"title": "More Leagues", "leagues": ["MLS", "ASL", "JPL", "LMX", "JPB", "SPL", "DSL", "SSL", "TSL", "GSL"]},
}


async def region_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    region = REGION_MENUS.get(query.data.replace("region_", ""))
    if not region: await query.edit_message_text("\u274c Unknown region."); return
    keyboard = []
    for code in region["leagues"]:
        info = LEAGUES.get(code)
        if not info: continue
        tier = "\U0001f193" if code in FREE_LEAGUES else "\U0001f4b0"
        keyboard.append([InlineKeyboardButton(f"{info['flag']} {info['name']} {tier}", callback_data=f"league_{code}")])
    keyboard.append([InlineKeyboardButton("\u2b05\ufe0f Back", callback_data="back_to_regions")])
    await query.edit_message_text(f"*{region['title']}*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)


async def back_to_regions_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f England", callback_data="region_england")],
        [InlineKeyboardButton("\U0001f1ea\U0001f1f8 Spain", callback_data="region_spain")],
        [InlineKeyboardButton("\U0001f1e9\U0001f1ea Germany", callback_data="region_germany")],
        [InlineKeyboardButton("\U0001f1ee\U0001f1f9 Italy", callback_data="region_italy")],
        [InlineKeyboardButton("\U0001f1eb\U0001f1f7 France", callback_data="region_france")],
        [InlineKeyboardButton("\U0001f1f3\U0001f1f1 Netherlands", callback_data="league_DED"), InlineKeyboardButton("\U0001f1f5\U0001f1f9 Portugal", callback_data="league_PPL")],
        [InlineKeyboardButton("\U0001f1e7\U0001f1f7 Brazil", callback_data="region_brazil"), InlineKeyboardButton("\U0001f1fa\U0001f1f8 USA", callback_data="league_MLS")],
        [InlineKeyboardButton("\U0001f3c6 European Cups", callback_data="region_europe")],
        [InlineKeyboardButton("\U0001f30d More Leagues", callback_data="region_more")],
    ]
    await query.edit_message_text("\U0001f3df\ufe0f *Select a region or league:*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)


async def league_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    code = query.data.replace("league_", "")
    info = LEAGUES.get(code)
    if not info: await query.edit_message_text("\u274c Unknown league."); return
    is_paid = code in PAID_LEAGUES
    await query.edit_message_text(f"\U0001f504 Fetching {info['name']} streak data...")
    date_from = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        matches = await api.get_matches(code, date_from, date_to)
        if not matches: await query.edit_message_text(f"No recent matches for {info['name']}."); return
        streaks = calculate_streaks(matches)
        await query.edit_message_text(format_full_report(streaks, f"{info['flag']} {info['name']}"), parse_mode=ParseMode.MARKDOWN)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and is_paid:
            await query.edit_message_text(f"\U0001f512 *{info['name']}* requires a paid subscription.\nUpgrade at: https://www.football-data.org/pricing", parse_mode=ParseMode.MARKDOWN)
        elif e.response.status_code == 429:
            await query.edit_message_text("\u23f1\ufe0f Rate limit hit. Wait a minute and try again.")
        else:
            await query.edit_message_text(f"\u274c API error: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        await query.edit_message_text(f"\u274c Error: {e}")


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("\U0001f504 Checking today's fixtures...")
    all_lines = ["\U0001f4c5 *Today's Matches*\n"]
    found_any = False
    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP": continue
        try:
            if i > 0: await asyncio.sleep(6)
            matches = await api.get_todays_matches(code)
            if not matches: continue
            found_any = True
            all_lines.append(f"\n{info['flag']} *{info['name']}*")
            for m in matches:
                home = m["homeTeam"]["name"]
                away = m["awayTeam"]["name"]
                status = m["status"]
                if status == "FINISHED":
                    all_lines.append(f"  \u2705 {home} {m['score']['fullTime']['home']} - {m['score']['fullTime']['away']} {away}")
                elif status in ("IN_PLAY", "PAUSED"):
                    all_lines.append(f"  \U0001f534 {home} {m['score']['fullTime']['home'] or 0} - {m['score']['fullTime']['away'] or 0} {away} (LIVE)")
                else:
                    ts = m.get("utcDate", "")
                    ko = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M UTC") if ts else "TBD"
                    all_lines.append(f"  \U0001f550 {home} vs {away} ({ko})")
        except Exception as e:
            logger.error(f"Error today {code}: {e}")
    if not found_any: all_lines.append("\nNo matches today.")
    await update.message.reply_text("\n".join(all_lines), parse_mode=ParseMode.MARKDOWN)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f916 *Generating predictions...*\n"
        "_(Analysing form, streaks, attack/defense ratings & H2H)_\n"
        "_(This may take ~90 seconds due to API rate limits)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    date_from = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    all_predictions = []
    req = 0

    for code, info in FREE_LEAGUES.items():
        if info.get("type") == "CUP": continue
        try:
            if req > 0: await asyncio.sleep(7)
            today_matches = await api.get_todays_matches(code)
            req += 1
            upcoming = [m for m in today_matches if m["status"] in ("SCHEDULED", "TIMED")]
            if not upcoming: continue

            await asyncio.sleep(7)
            historical = await api.get_matches(code, date_from, date_to)
            req += 1
            if not historical: continue

            team_stats = build_team_stats(historical)
            stks = calculate_streaks(historical)

            for match in upcoming:
                home = match["homeTeam"]["name"]
                away = match["awayTeam"]["name"]
                ts = match.get("utcDate", "")
                ko = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M UTC") if ts else ""
                pred = predict_match(home, away, team_stats, stks)
                pred["league"] = f"{info['flag']} {info['name']}"
                pred["kick_off"] = ko
                all_predictions.append(pred)
        except Exception as e:
            logger.error(f"Prediction error {code}: {e}")

    if not all_predictions:
        await update.message.reply_text("\u274c No upcoming matches found today.\nTry again on a match day.")
        return

    header = f"\U0001f916 *AI Match Predictions \u2014 {today}*\n\U0001f4ca _Poisson model + form + streaks + H2H_\n{'=' * 30}\n"
    current_league = ""
    current_msg = header

    for pred in all_predictions:
        league_hdr = ""
        if pred["league"] != current_league:
            current_league = pred["league"]
            league_hdr = f"\n\U0001f3df\ufe0f *{current_league}*\n"
        pred_text = league_hdr + format_prediction(pred)
        if len(current_msg) + len(pred_text) > 3800:
            await update.message.reply_text(current_msg, parse_mode=ParseMode.MARKDOWN)
            current_msg = pred_text
        else:
            current_msg += "\n" + pred_text

    if current_msg:
        current_msg += "\n\n\u26a0\ufe0f _Statistical estimates only. Not guarantees. Gamble responsibly._"
        await update.message.reply_text(current_msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_tips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f52e *Finding best picks...*\n_(Scanning all leagues for high-confidence predictions)_\n_(~90 seconds)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    date_from = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    all_predictions = []
    req = 0

    for code, info in FREE_LEAGUES.items():
        if info.get("type") == "CUP": continue
        try:
            if req > 0: await asyncio.sleep(7)
            today_matches = await api.get_todays_matches(code)
            req += 1
            upcoming = [m for m in today_matches if m["status"] in ("SCHEDULED", "TIMED")]
            if not upcoming: continue

            await asyncio.sleep(7)
            historical = await api.get_matches(code, date_from, date_to)
            req += 1
            if not historical: continue

            team_stats = build_team_stats(historical)
            stks = calculate_streaks(historical)

            for match in upcoming:
                home = match["homeTeam"]["name"]
                away = match["awayTeam"]["name"]
                ts = match.get("utcDate", "")
                ko = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M UTC") if ts else ""
                pred = predict_match(home, away, team_stats, stks)
                pred["league"] = f"{info['flag']} {info['name']}"
                pred["kick_off"] = ko
                all_predictions.append(pred)
        except Exception as e:
            logger.error(f"Tips error {code}: {e}")

    if not all_predictions:
        await update.message.reply_text("\u274c No upcoming matches today.\nTry again on a match day.")
        return

    all_predictions.sort(key=lambda x: x["confidence_score"], reverse=True)
    top = all_predictions[:8]

    lines = [f"\U0001f52e *Top Picks \u2014 {today}*", "_Ranked by prediction confidence_", "=" * 30, ""]
    for i, pred in enumerate(top):
        lines.append(f"*{pred['league']}* ({pred['kick_off']})")
        lines.append(format_tip(pred, i))
        lines.append("")

    high_conf = [p for p in all_predictions if p["confidence_score"] >= 0.7]
    o25_picks = [p for p in all_predictions if p["over25"] >= 0.65]
    btts_picks = [p for p in all_predictions if p["btts"] >= 0.65]

    lines.append("=" * 30)
    lines.append(f"\U0001f4ca *Today's Summary:*")
    lines.append(f"   {len(all_predictions)} matches analysed")
    lines.append(f"   \U0001f7e2 {len(high_conf)} high-confidence picks")
    lines.append(f"   \u26bd {len(o25_picks)} likely Over 2.5")
    lines.append(f"   \U0001f3af {len(btts_picks)} likely BTTS")
    lines.append(f"")
    lines.append(f"\u26a0\ufe0f _Statistical estimates only. Gamble responsibly._")

    text = "\n".join(lines)
    if len(text) > 4000:
        mid = len(top) // 2
        msg1 = lines[:4]
        for i, pred in enumerate(top[:mid]):
            msg1.append(f"*{pred['league']}* ({pred['kick_off']})")
            msg1.append(format_tip(pred, i))
            msg1.append("")
        await update.message.reply_text("\n".join(msg1), parse_mode=ParseMode.MARKDOWN)
        msg2 = []
        for i, pred in enumerate(top[mid:]):
            msg2.append(f"*{pred['league']}* ({pred['kick_off']})")
            msg2.append(format_tip(pred, i + mid))
            msg2.append("")
        msg2.extend(lines[-7:])
        await update.message.reply_text("\n".join(msg2), parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def daily_summary(context: ContextTypes.DEFAULT_TYPE):
    chat_id = os.environ.get("DAILY_CHAT_ID")
    if not chat_id: return
    date_from = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")

    summary = [f"\U0001f305 *Daily Streak & Prediction Summary \u2014 {today}*\n"]
    all_preds = []

    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP": continue
        try:
            if i > 0: await asyncio.sleep(7)
            historical = await api.get_matches(code, date_from, date_to)
            if not historical: continue
            stks = calculate_streaks(historical)
            notable = {t: d for t, d in stks.items() if d["win_streak"] >= 3 or d["goal_streak"] >= 5}
            if notable:
                summary.append(f"\n{info['flag']} *{info['name']}*")
                for t, d in sorted(notable.items(), key=lambda x: x[1]["win_streak"], reverse=True):
                    parts = []
                    if d["win_streak"] >= 3: parts.append(f"\U0001f525 {d['win_streak']}W")
                    if d["goal_streak"] >= 5: parts.append(f"\u26bd {d['goal_streak']}G")
                    summary.append(f"  {t}: {' | '.join(parts)} {d['form']}")

            await asyncio.sleep(7)
            today_m = await api.get_todays_matches(code)
            upcoming = [m for m in today_m if m["status"] in ("SCHEDULED", "TIMED")]
            if upcoming and historical:
                ts = build_team_stats(historical)
                for m in upcoming:
                    pred = predict_match(m["homeTeam"]["name"], m["awayTeam"]["name"], ts, stks)
                    pred["league"] = f"{info['flag']} {info['name']}"
                    all_preds.append(pred)
        except Exception as e:
            logger.error(f"Daily summary error {code}: {e}")

    if all_preds:
        all_preds.sort(key=lambda x: x["confidence_score"], reverse=True)
        summary.append(f"\n{'=' * 25}")
        summary.append(f"\U0001f52e *Top Predictions Today:*")
        for pred in all_preds[:5]:
            mp = max(pred["home_win"], pred["draw"], pred["away_win"])
            summary.append(f"  {pred['home_team']} vs {pred['away_team']}\n    \U0001f3af {pred['prediction']} ({mp:.0%})\n    \U0001f4ca xG: {pred['home_xg']} - {pred['away_xg']}")

    if len(summary) > 1:
        summary.append(f"\n\u26a0\ufe0f _Statistical estimates only._")
        await context.bot.send_message(chat_id=int(chat_id), text="\n".join(summary), parse_mode=ParseMode.MARKDOWN)


def main():
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("\u26a0\ufe0f  Set TELEGRAM_BOT_TOKEN environment variable!")
        return
    if FOOTBALL_API_KEY == "YOUR_FOOTBALL_DATA_API_KEY":
        print("\u26a0\ufe0f  Set FOOTBALL_API_KEY environment variable!")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("streaks", cmd_streaks))
    app.add_handler(CommandHandler("goals", cmd_goals))
    app.add_handler(CommandHandler("league", cmd_league))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(CommandHandler("tips", cmd_tips))

    app.add_handler(CallbackQueryHandler(league_callback, pattern=r"^league_"))
    app.add_handler(CallbackQueryHandler(region_callback, pattern=r"^region_"))
    app.add_handler(CallbackQueryHandler(back_to_regions_callback, pattern=r"^back_to_regions$"))

    job_queue = app.job_queue
    if job_queue and os.environ.get("DAILY_CHAT_ID"):
        from datetime import time as dt_time
        job_queue.run_daily(daily_summary, time=dt_time(hour=8, minute=0), name="daily_streak_summary")
        logger.info("Daily summary scheduled for 08:00 UTC")

    print("\U0001f916 Football Streak Tracker & Predictor is running!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
