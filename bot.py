"""
Football Streak Tracker & AI Predictor — Telegram Bot v2
=========================================================
Uses ELO ratings, Poisson modelling, form weighting, league position,
streak momentum, and head-to-head analysis to predict match outcomes.

Inspired by:
  - ProphitBet (feature engineering, ensemble approach)
  - jkrusina/SoccerPredictor (time-series, double chance)
  - ronyka77/SoccerPredictor (ELO ratings, stacked models, Poisson xG)

Commands:
  /start, /help   - Welcome & instructions
  /streaks         - Top winning streaks across leagues
  /goals           - Top goal-scoring streaks
  /league          - Pick a league for detailed streaks
  /today           - Today's matches
  /predict         - AI predictions for today's matches
  /tips            - Best picks of the day (highest confidence)
"""

import os, logging, asyncio, math
from datetime import datetime, timedelta
from collections import defaultdict

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
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
            resp = await client.get(f"{FOOTBALL_API_BASE}{endpoint}", headers=self.headers, params=params or {}, timeout=15)
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
        data = await self._get(f"/competitions/{league_code}/matches", {"dateFrom": today, "dateTo": today})
        return data.get("matches", [])

    async def get_standings(self, league_code):
        try:
            data = await self._get(f"/competitions/{league_code}/standings")
            for s in data.get("standings", []):
                if s.get("type") == "TOTAL":
                    return s.get("table", [])
        except Exception:
            pass
        return []

api = FootballAPI(FOOTBALL_API_KEY)

VALID_STAGES = {"REGULAR_SEASON","GROUP_STAGE","LEAGUE_STAGE","ROUND_OF_16","QUARTER_FINALS","SEMI_FINALS","FINAL","LAST_16","LAST_32","LAST_64"}
SKIP_STAGES = {"FRIENDLY","PRELIMINARY_ROUND","QUALIFICATION","QUALIFICATION_ROUND_1","QUALIFICATION_ROUND_2","QUALIFICATION_ROUND_3","PLAYOFF_ROUND","PLAYOFFS"}

def is_valid_stage(match):
    stage = match.get("stage", "").upper()
    if stage in SKIP_STAGES: return False
    if VALID_STAGES and stage and stage not in VALID_STAGES: return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# ELO RATING SYSTEM (inspired by ronyka77/SoccerPredictor)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_elo_ratings(matches, k_factor=32, home_advantage=65):
    """
    Calculate ELO ratings for all teams from historical match data.
    
    Uses the standard ELO formula adapted for football:
      - K-factor: how much a single result changes the rating (32 = moderate)
      - Home advantage: ELO points added to home team's expected score
      - Goal difference multiplier: bigger wins = bigger rating changes
    
    All teams start at 1500 (league average).
    """
    elo = defaultdict(lambda: 1500.0)
    
    # Sort matches chronologically
    sorted_matches = sorted(
        [m for m in matches if m.get("status") == "FINISHED" and is_valid_stage(m)],
        key=lambda m: m.get("utcDate", "")
    )
    
    for match in sorted_matches:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        hg = match["score"]["fullTime"]["home"]
        ag = match["score"]["fullTime"]["away"]
        if hg is None or ag is None:
            continue
        
        # Expected scores (with home advantage)
        dr = (elo[home] + home_advantage) - elo[away]
        exp_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        exp_away = 1.0 - exp_home
        
        # Actual scores
        if hg > ag:
            actual_home, actual_away = 1.0, 0.0
        elif hg < ag:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Goal difference multiplier (bigger wins = bigger ELO change)
        goal_diff = abs(hg - ag)
        if goal_diff <= 1:
            gd_mult = 1.0
        elif goal_diff == 2:
            gd_mult = 1.5
        elif goal_diff == 3:
            gd_mult = 1.75
        else:
            gd_mult = 1.75 + (goal_diff - 3) * 0.125
        
        # Update ratings
        change = k_factor * gd_mult
        elo[home] += change * (actual_home - exp_home)
        elo[away] += change * (actual_away - exp_away)
    
    return dict(elo)


def elo_win_probability(home_elo, away_elo, home_advantage=65):
    """Calculate expected win probability from ELO ratings."""
    dr = (home_elo + home_advantage) - away_elo
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


# ══════════════════════════════════════════════════════════════════════════════
# STREAK CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

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
        form = "".join({
            "W": "\U0001f7e2", "D": "\U0001f7e1", "L": "\U0001f534"
        }[m["result"]] for m in ml[:5])
        rg = sum(m["goals_scored"] for m in ml[:5])
        rc = sum(m["goals_conceded"] for m in ml[:5])
        
        # Goal difference momentum (last 5 vs previous 5)
        recent_gd = sum(m["goals_scored"] - m["goals_conceded"] for m in ml[:5])
        prev_gd = sum(m["goals_scored"] - m["goals_conceded"] for m in ml[5:10]) if len(ml) >= 10 else 0
        gd_momentum = recent_gd - prev_gd  # positive = improving
        
        streaks[team] = {
            "win_streak": ws, "unbeaten_streak": ub,
            "goal_streak": gs, "clean_sheet_streak": cs,
            "form": form, "recent_goals": rg, "recent_conceded": rc,
            "gd_momentum": gd_momentum,
            "matches_played": len(ml),
            "last_match": ml[0] if ml else None,
        }
    return streaks


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE v2 — ELO + Poisson + Form + Position + Streaks + H2H
# ══════════════════════════════════════════════════════════════════════════════

def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def build_team_stats(matches):
    """Build comprehensive per-team stats from historical matches."""
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
    lah = sum(all_hg) / max(len(all_hg), 1)
    laa = sum(all_ag) / max(len(all_ag), 1)

    stats = {"_league_avg_home": lah, "_league_avg_away": laa}
    for team, data in team_data.items():
        n_all = max(len(data["all_scored"]), 1)
        n_home = max(len(data["home_scored"]), 1)
        n_away = max(len(data["away_scored"]), 1)

        home_attack = (sum(data["home_scored"]) / n_home) / max(lah, 0.5)
        home_defense = (sum(data["home_conceded"]) / n_home) / max(laa, 0.5)
        away_attack = (sum(data["away_scored"]) / n_away) / max(laa, 0.5)
        away_defense = (sum(data["away_conceded"]) / n_away) / max(lah, 0.5)

        # Form rating with exponential decay (recent matches weighted higher)
        data["results"].sort(key=lambda x: x[0], reverse=True)
        fp, fw = 0, 0
        for i, (_, r) in enumerate(data["results"][:10]):
            w = math.exp(-0.15 * i)
            fw += w
            if r == "W": fp += 3 * w
            elif r == "D": fp += 1 * w
        form_rating = fp / max(fw, 1) / 3.0

        results_only = [r for _, r in data["results"]]
        total = max(len(results_only), 1)
        btts_count = sum(1 for s, c in zip(data["all_scored"], data["all_conceded"]) if s >= 1 and c >= 1)
        o15_count = sum(1 for s, c in zip(data["all_scored"], data["all_conceded"]) if s + c >= 2)
        o25_count = sum(1 for s, c in zip(data["all_scored"], data["all_conceded"]) if s + c >= 3)

        stats[team] = {
            "home_attack": home_attack, "home_defense": home_defense,
            "away_attack": away_attack, "away_defense": away_defense,
            "avg_scored": sum(data["all_scored"]) / n_all,
            "avg_conceded": sum(data["all_conceded"]) / n_all,
            "form_rating": form_rating,
            "win_rate": results_only.count("W") / total,
            "draw_rate": results_only.count("D") / total,
            "btts_rate": btts_count / n_all,
            "over15_rate": o15_count / n_all,
            "over25_rate": o25_count / n_all,
            "matches_played": len(data["results"]),
            "h2h": data["h2h"],
        }
    return stats


def predict_match(home_team, away_team, stats, streaks, elo_ratings, standings_map):
    """
    Full prediction using 6-layer model:
      1. Poisson base xG (attack/defense strength)
      2. ELO rating adjustment
      3. League position factor
      4. Form weighting
      5. Streak momentum
      6. Head-to-head adjustment
    """
    hs = stats.get(home_team)
    aws = stats.get(away_team)
    lah = stats.get("_league_avg_home", 1.4)
    laa = stats.get("_league_avg_away", 1.1)

    if not hs or not aws:
        return {
            "home_team": home_team, "away_team": away_team,
            "home_xg": round(lah, 2), "away_xg": round(laa, 2),
            "home_win": 0.45, "draw": 0.25, "away_win": 0.30,
            "over15": 0.65, "btts": 0.50, "double_home": 0.70, "double_away": 0.55,
            "confidence": "\u26aa Low", "confidence_score": 0.3,
            "prediction": "Insufficient data", "goals_prediction": "~2 goals",
            "factors": ["Limited data available"], "score_line": "? - ?",
            "home_form": "", "away_form": "", "home_elo": 1500, "away_elo": 1500,
            "home_pos": "?", "away_pos": "?",
        }

    # ── Layer 1: Base xG from Poisson model ──────────────────────────────
    home_xg = hs["home_attack"] * aws["away_defense"] * lah
    away_xg = aws["away_attack"] * hs["home_defense"] * laa

    # ── Layer 2: ELO rating adjustment (+/- 12%) ─────────────────────────
    home_elo = elo_ratings.get(home_team, 1500)
    away_elo = elo_ratings.get(away_team, 1500)
    elo_diff = (home_elo - away_elo) / 400.0  # normalised
    elo_factor = max(-0.12, min(0.12, elo_diff * 0.08))
    home_xg *= (1 + elo_factor)
    away_xg *= (1 - elo_factor)

    # ── Layer 3: League position factor (+/- 8%) ─────────────────────────
    home_pos = standings_map.get(home_team, 10)
    away_pos = standings_map.get(away_team, 10)
    total_teams = max(len(standings_map), 20)
    # Normalise: top of table = high, bottom = low (0 to 1 scale)
    home_pos_strength = 1.0 - (home_pos - 1) / max(total_teams - 1, 1)
    away_pos_strength = 1.0 - (away_pos - 1) / max(total_teams - 1, 1)
    pos_diff = home_pos_strength - away_pos_strength  # -1 to +1
    home_xg *= (1 + pos_diff * 0.08)
    away_xg *= (1 - pos_diff * 0.08)

    # ── Layer 4: Home advantage boost (+10%) ─────────────────────────────
    home_xg *= 1.10

    # ── Layer 5: Form adjustment (+/- 15%) ───────────────────────────────
    form_diff = hs["form_rating"] - aws["form_rating"]
    home_xg *= (1 + form_diff * 0.15)
    away_xg *= (1 - form_diff * 0.15)

    # ── Layer 6: Streak momentum (+/- 10%) ───────────────────────────────
    hsk = streaks.get(home_team, {})
    ask = streaks.get(away_team, {})
    home_xg *= (1 + min(hsk.get("win_streak", 0), 5) * 0.02)
    away_xg *= (1 + min(ask.get("win_streak", 0), 5) * 0.02)
    if hsk.get("goal_streak", 0) >= 5: home_xg *= 1.05
    if ask.get("goal_streak", 0) >= 5: away_xg *= 1.05
    
    # Goal difference momentum boost
    gd_mom_home = hsk.get("gd_momentum", 0)
    gd_mom_away = ask.get("gd_momentum", 0)
    if gd_mom_home > 3: home_xg *= 1.03
    elif gd_mom_home < -3: home_xg *= 0.97
    if gd_mom_away > 3: away_xg *= 1.03
    elif gd_mom_away < -3: away_xg *= 0.97

    # ── Layer 7: Head-to-head adjustment (+/- 8%) ────────────────────────
    h2h = hs.get("h2h", {}).get(away_team, {})
    h2h_results = h2h.get("results", [])
    if len(h2h_results) >= 3:
        h2h_dom = (h2h_results.count("W") / len(h2h_results)) - 0.5
        home_xg *= (1 + h2h_dom * 0.16)
        away_xg *= (1 - h2h_dom * 0.16)

    # Clamp
    home_xg = max(0.25, min(home_xg, 4.5))
    away_xg = max(0.2, min(away_xg, 4.0))

    # ── Poisson probability matrix ───────────────────────────────────────
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

    # ── Blend with ELO probabilities (30% ELO, 70% Poisson) ─────────────
    elo_hw = elo_win_probability(home_elo, away_elo)
    elo_aw = 1.0 - elo_win_probability(away_elo, home_elo, home_advantage=0)
    elo_dr = 1.0 - elo_hw - (1.0 - elo_aw)
    elo_dr = max(0.15, min(elo_dr, 0.35))  # clamp draw
    elo_aw_adj = 1.0 - elo_hw - elo_dr
    
    hw = hw * 0.70 + elo_hw * 0.30
    aw = aw * 0.70 + max(elo_aw_adj, 0.05) * 0.30
    dr = 1.0 - hw - aw
    dr = max(dr, 0.08)  # floor draw probability
    # Re-normalise
    t = hw + dr + aw
    hw /= t; dr /= t; aw /= t

    # ── Over 1.5, BTTS, Double Chance ────────────────────────────────────
    o15_poisson = sum(p for (hg, ag), p in score_probs.items() if hg + ag >= 2)
    btts_poisson = sum(p for (hg, ag), p in score_probs.items() if hg >= 1 and ag >= 1)
    o25_poisson = sum(p for (hg, ag), p in score_probs.items() if hg + ag >= 3)

    hist_btts = (hs["btts_rate"] + aws["btts_rate"]) / 2
    hist_o15 = (hs["over15_rate"] + aws["over15_rate"]) / 2
    hist_o25 = (hs["over25_rate"] + aws["over25_rate"]) / 2

    o15 = o15_poisson * 0.6 + hist_o15 * 0.4
    btts = btts_poisson * 0.6 + hist_btts * 0.4
    o25 = o25_poisson * 0.6 + hist_o25 * 0.4

    # Double chance (1X = home or draw, X2 = draw or away)
    dc_home = hw + dr  # 1X
    dc_away = dr + aw  # X2

    # Most likely score
    best_score = max(score_probs, key=score_probs.get)

    # ── Confidence score ─────────────────────────────────────────────────
    max_prob = max(hw, dr, aw)
    data_q = min(min(hs["matches_played"], aws["matches_played"]) / 15, 1.0)
    elo_agreement = 1.0 if (elo_hw > 0.5 and hw > aw) or (elo_hw < 0.5 and aw > hw) else 0.5
    conf_score = min(max_prob * 0.45 + data_q * 0.20 + abs(form_diff) * 0.10 + elo_agreement * 0.25, 1.0)
    
    if conf_score >= 0.72: conf = "\U0001f7e2 High"
    elif conf_score >= 0.52: conf = "\U0001f7e1 Medium"
    else: conf = "\u26aa Low"

    # ── Prediction text ──────────────────────────────────────────────────
    if hw > aw and hw > dr:
        pred = f"\U0001f3e0 {home_team} Win" if hw >= 0.55 else f"\U0001f3e0 {home_team} Slight Edge"
    elif aw > hw and aw > dr:
        pred = f"\u2708\ufe0f {away_team} Win" if aw >= 0.55 else f"\u2708\ufe0f {away_team} Slight Edge"
    else:
        pred = "\U0001f91d Draw Likely"

    # Key factors
    factors = []
    if abs(home_elo - away_elo) > 100:
        stronger = home_team if home_elo > away_elo else away_team
        diff = abs(int(home_elo - away_elo))
        factors.append(f"\U0001f4ca ELO: {stronger} rated +{diff} higher")
    if abs(home_pos - away_pos) >= 5:
        higher = home_team if home_pos < away_pos else away_team
        factors.append(f"\U0001f3c6 {higher} significantly higher in table")
    if hsk.get("win_streak", 0) >= 3: factors.append(f"\U0001f525 {home_team} on {hsk['win_streak']}-game win streak")
    if ask.get("win_streak", 0) >= 3: factors.append(f"\U0001f525 {away_team} on {ask['win_streak']}-game win streak")
    if hs["form_rating"] > 0.7: factors.append(f"\U0001f4c8 {home_team} in excellent form")
    if aws["form_rating"] > 0.7: factors.append(f"\U0001f4c8 {away_team} in excellent form")
    if hs["form_rating"] < 0.3: factors.append(f"\U0001f4c9 {home_team} in poor form")
    if aws["form_rating"] < 0.3: factors.append(f"\U0001f4c9 {away_team} in poor form")
    if gd_mom_home > 3: factors.append(f"\u2b06\ufe0f {home_team} GD improving sharply")
    if gd_mom_away > 3: factors.append(f"\u2b06\ufe0f {away_team} GD improving sharply")
    if o15 >= 0.80: factors.append("\u26a1 Goals very likely (O1.5)")
    if btts >= 0.65: factors.append("\u26bd Both teams likely to score")
    if hsk.get("clean_sheet_streak", 0) >= 3: factors.append(f"\U0001f9e4 {home_team} {hsk['clean_sheet_streak']} clean sheets running")
    if len(h2h_results) >= 3:
        h2hw = h2h_results.count("W")
        if h2hw >= len(h2h_results) * 0.6: factors.append(f"\U0001f4ca {home_team} dominates H2H ({h2hw}/{len(h2h_results)})")

    txg = home_xg + away_xg
    if txg >= 3.5: gp = "High scoring (3+ goals expected)"
    elif txg >= 2.5: gp = "Moderate (2-3 goals expected)"
    elif txg >= 1.5: gp = "Low scoring (1-2 goals expected)"
    else: gp = "Very tight (0-1 goals expected)"

    return {
        "home_team": home_team, "away_team": away_team,
        "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2),
        "home_win": round(hw, 3), "draw": round(dr, 3), "away_win": round(aw, 3),
        "over15": round(o15, 3), "over25": round(o25, 3), "btts": round(btts, 3),
        "double_home": round(dc_home, 3), "double_away": round(dc_away, 3),
        "confidence": conf, "confidence_score": round(conf_score, 3),
        "prediction": pred, "goals_prediction": gp,
        "factors": factors[:5], "score_line": f"{best_score[0]} - {best_score[1]}",
        "home_form": hsk.get("form", ""), "away_form": ask.get("form", ""),
        "home_elo": int(home_elo), "away_elo": int(away_elo),
        "home_pos": home_pos if standings_map else "?",
        "away_pos": away_pos if standings_map else "?",
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def format_prediction(pred):
    lines = [
        f"\u26bd *{pred['home_team']}* vs *{pred['away_team']}*",
        f"   \U0001f4ca ELO: {pred['home_elo']} vs {pred['away_elo']} | Pos: {pred['home_pos']} vs {pred['away_pos']}",
        f"",
        f"\U0001f3af *Prediction:* {pred['prediction']}",
        f"\U0001f4ca Most likely score: *{pred['score_line']}*",
        f"",
        f"\U0001f4c8 *Win Probabilities:*",
        f"   \U0001f3e0 Home: {pred['home_win']:.0%}  \U0001f91d Draw: {pred['draw']:.0%}  \u2708\ufe0f Away: {pred['away_win']:.0%}",
        f"",
        f"\u26bd *Goals:*",
        f"   xG: {pred['home_team'][:3].upper()} {pred['home_xg']} - {pred['away_xg']} {pred['away_team'][:3].upper()}",
        f"   Over 1.5: {pred['over15']:.0%} | BTTS: {pred['btts']:.0%}",
        f"   \U0001f4cb {pred['goals_prediction']}",
        f"",
        f"\U0001f91d *Double Chance:*",
        f"   1X (Home/Draw): {pred['double_home']:.0%} | X2 (Draw/Away): {pred['double_away']:.0%}",
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
        f"   ELO: {pred['home_elo']} vs {pred['away_elo']}",
    ]
    angles = []
    if pred["over15"] >= 0.75: angles.append(f"Over 1.5 ({pred['over15']:.0%})")
    if pred["btts"] >= 0.60: angles.append(f"BTTS ({pred['btts']:.0%})")
    if pred["home_win"] >= 0.55: angles.append(f"Home Win ({pred['home_win']:.0%})")
    if pred["away_win"] >= 0.55: angles.append(f"Away Win ({pred['away_win']:.0%})")
    if pred["double_home"] >= 0.75: angles.append(f"1X ({pred['double_home']:.0%})")
    if pred["double_away"] >= 0.75: angles.append(f"X2 ({pred['double_away']:.0%})")
    if angles: lines.append(f"   \U0001f4b0 Best angles: {' | '.join(angles[:3])}")
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
        "\u26bd *Football Streak Tracker & AI Predictor v2* \u26bd\n\n"
        "Track streaks and get data-driven match predictions.\n\n"
        "*\U0001f4ca Streak Commands:*\n"
        "/streaks \u2014 Top winning streaks\n"
        "/goals \u2014 Top goal-scoring streaks\n"
        "/league \u2014 Detailed league streaks\n"
        "/today \u2014 Today's matches\n\n"
        "*\U0001f916 AI Prediction Commands:*\n"
        "/predict \u2014 Full predictions for today\n"
        "/tips \u2014 Best picks (highest confidence)\n\n"
        "/help \u2014 Show this message\n\n"
        "\U0001f9e0 *Prediction Model (6 layers):*\n"
        "1\ufe0f\u20e3 Poisson goal model (attack/defense strength)\n"
        "2\ufe0f\u20e3 ELO rating system (dynamic team strength)\n"
        "3\ufe0f\u20e3 League position weighting\n"
        "4\ufe0f\u20e3 Form analysis (exponential decay)\n"
        "5\ufe0f\u20e3 Streak momentum + GD trajectory\n"
        "6\ufe0f\u20e3 Head-to-head historical record\n\n"
        "\u26a0\ufe0f _Statistical estimates, not guarantees. Gamble responsibly._"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)

async def cmd_streaks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("\U0001f504 Fetching streak data...\n_(~60 seconds)_", parse_mode=ParseMode.MARKDOWN)
    df = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    dt = datetime.utcnow().strftime("%Y-%m-%d")
    msgs = []
    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP": continue
        try:
            if i > 0: await asyncio.sleep(6)
            matches = await api.get_matches(code, df, dt)
            if not matches: continue
            msgs.append(format_winning_streaks(calculate_streaks(matches), f"{info['flag']} {info['name']}", top_n=5))
        except Exception as e: logger.error(f"Error {code}: {e}")
    if msgs:
        full = "\n\n".join(msgs)
        if len(full) > 4000:
            for m in msgs: await update.message.reply_text(m, parse_mode=ParseMode.MARKDOWN)
        else: await update.message.reply_text(full, parse_mode=ParseMode.MARKDOWN)
    else: await update.message.reply_text("\u274c Could not fetch streak data.")

async def cmd_goals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("\U0001f504 Fetching goal streaks...\n_(~60 seconds)_", parse_mode=ParseMode.MARKDOWN)
    df = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    dt = datetime.utcnow().strftime("%Y-%m-%d")
    msgs = []
    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP": continue
        try:
            if i > 0: await asyncio.sleep(6)
            matches = await api.get_matches(code, df, dt)
            if not matches: continue
            msgs.append(format_goal_streaks(calculate_streaks(matches), f"{info['flag']} {info['name']}", top_n=5))
        except Exception as e: logger.error(f"Error {code}: {e}")
    if msgs:
        full = "\n\n".join(msgs)
        if len(full) > 4000:
            for m in msgs: await update.message.reply_text(m, parse_mode=ParseMode.MARKDOWN)
        else: await update.message.reply_text(full, parse_mode=ParseMode.MARKDOWN)
    else: await update.message.reply_text("\u274c Could not fetch goal streak data.")

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
    query = update.callback_query; await query.answer()
    region = REGION_MENUS.get(query.data.replace("region_", ""))
    if not region: await query.edit_message_text("\u274c Unknown region."); return
    kb = []
    for code in region["leagues"]:
        info = LEAGUES.get(code)
        if not info: continue
        tier = "\U0001f193" if code in FREE_LEAGUES else "\U0001f4b0"
        kb.append([InlineKeyboardButton(f"{info['flag']} {info['name']} {tier}", callback_data=f"league_{code}")])
    kb.append([InlineKeyboardButton("\u2b05\ufe0f Back", callback_data="back_to_regions")])
    await query.edit_message_text(f"*{region['title']}*", reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.MARKDOWN)

async def back_to_regions_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
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
    query = update.callback_query; await query.answer()
    code = query.data.replace("league_", "")
    info = LEAGUES.get(code)
    if not info: await query.edit_message_text("\u274c Unknown league."); return
    is_paid = code in PAID_LEAGUES
    await query.edit_message_text(f"\U0001f504 Fetching {info['name']} streak data...")
    df = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    dt = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        matches = await api.get_matches(code, df, dt)
        if not matches: await query.edit_message_text(f"No recent matches for {info['name']}."); return
        await query.edit_message_text(format_full_report(calculate_streaks(matches), f"{info['flag']} {info['name']}"), parse_mode=ParseMode.MARKDOWN)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and is_paid:
            await query.edit_message_text(f"\U0001f512 *{info['name']}* requires a paid subscription.\nhttps://www.football-data.org/pricing", parse_mode=ParseMode.MARKDOWN)
        elif e.response.status_code == 429:
            await query.edit_message_text("\u23f1\ufe0f Rate limit hit. Wait a minute.")
        else: await query.edit_message_text(f"\u274c API error: {e}")
    except Exception as e: logger.error(f"Error: {e}"); await query.edit_message_text(f"\u274c Error: {e}")

async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("\U0001f504 Checking today's fixtures...")
    lines = ["\U0001f4c5 *Today's Matches*\n"]
    found = False
    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP": continue
        try:
            if i > 0: await asyncio.sleep(6)
            matches = await api.get_todays_matches(code)
            if not matches: continue
            found = True
            lines.append(f"\n{info['flag']} *{info['name']}*")
            for m in matches:
                home, away, status = m["homeTeam"]["name"], m["awayTeam"]["name"], m["status"]
                if status == "FINISHED":
                    lines.append(f"  \u2705 {home} {m['score']['fullTime']['home']} - {m['score']['fullTime']['away']} {away}")
                elif status in ("IN_PLAY", "PAUSED"):
                    lines.append(f"  \U0001f534 {home} {m['score']['fullTime']['home'] or 0} - {m['score']['fullTime']['away'] or 0} {away} (LIVE)")
                else:
                    ts = m.get("utcDate", "")
                    ko = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M UTC") if ts else "TBD"
                    lines.append(f"  \U0001f550 {home} vs {away} ({ko})")
        except Exception as e: logger.error(f"Error today {code}: {e}")
    if not found: lines.append("\nNo matches today.")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def _fetch_league_predictions(code, info, date_from, date_to):
    """Fetch data and generate predictions for one league. Returns list of predictions."""
    predictions = []
    try:
        today_matches = await api.get_todays_matches(code)
        upcoming = [m for m in today_matches if m["status"] in ("SCHEDULED", "TIMED")]
        if not upcoming:
            return predictions

        await asyncio.sleep(7)
        historical = await api.get_matches(code, date_from, date_to)
        if not historical:
            return predictions

        # Build all prediction layers
        team_stats = build_team_stats(historical)
        stks = calculate_streaks(historical)
        elo_ratings = calculate_elo_ratings(historical)

        # Try to get standings (extra API call)
        standings_map = {}
        try:
            await asyncio.sleep(7)
            standings = await api.get_standings(code)
            for entry in standings:
                team_name = entry.get("team", {}).get("name", "")
                position = entry.get("position", 99)
                if team_name:
                    standings_map[team_name] = position
        except Exception:
            pass  # standings are optional

        for match in upcoming:
            home = match["homeTeam"]["name"]
            away = match["awayTeam"]["name"]
            ts = match.get("utcDate", "")
            ko = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M UTC") if ts else ""
            pred = predict_match(home, away, team_stats, stks, elo_ratings, standings_map)
            pred["league"] = f"{info['flag']} {info['name']}"
            pred["kick_off"] = ko
            predictions.append(pred)
    except Exception as e:
        logger.error(f"Prediction error {code}: {e}")
    return predictions


async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f916 *Generating AI predictions...*\n"
        "_(ELO ratings + Poisson model + form + league position + streaks + H2H)_\n"
        "_(This may take ~2 minutes due to API rate limits)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    df = (datetime.utcnow() - timedelta(days=150)).strftime("%Y-%m-%d")
    dt = datetime.utcnow().strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    all_preds = []
    req = 0

    for code, info in FREE_LEAGUES.items():
        if info.get("type") == "CUP": continue
        if req > 0: await asyncio.sleep(7)
        preds = await _fetch_league_predictions(code, info, df, dt)
        all_preds.extend(preds)
        req += 1

    if not all_preds:
        await update.message.reply_text("\u274c No upcoming matches found today.\nTry again on a match day.")
        return

    header = (
        f"\U0001f916 *AI Match Predictions \u2014 {today}*\n"
        f"\U0001f4ca _6-layer model: Poisson + ELO + Position + Form + Streaks + H2H_\n"
        f"{'=' * 30}\n"
    )
    current_league = ""
    current_msg = header

    for pred in all_preds:
        lhdr = ""
        if pred["league"] != current_league:
            current_league = pred["league"]
            lhdr = f"\n\U0001f3df\ufe0f *{current_league}*\n"
        pt = lhdr + format_prediction(pred)
        if len(current_msg) + len(pt) > 3800:
            await update.message.reply_text(current_msg, parse_mode=ParseMode.MARKDOWN)
            current_msg = pt
        else:
            current_msg += "\n" + pt

    if current_msg:
        current_msg += "\n\n\u26a0\ufe0f _Statistical estimates only. Not guarantees. Gamble responsibly._"
        await update.message.reply_text(current_msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_tips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f52e *Finding best picks...*\n_(Scanning all leagues)_\n_(~2 minutes)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    df = (datetime.utcnow() - timedelta(days=150)).strftime("%Y-%m-%d")
    dt = datetime.utcnow().strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    all_preds = []
    req = 0

    for code, info in FREE_LEAGUES.items():
        if info.get("type") == "CUP": continue
        if req > 0: await asyncio.sleep(7)
        preds = await _fetch_league_predictions(code, info, df, dt)
        all_preds.extend(preds)
        req += 1

    if not all_preds:
        await update.message.reply_text("\u274c No upcoming matches today.\nTry again on a match day.")
        return

    all_preds.sort(key=lambda x: x["confidence_score"], reverse=True)
    top = all_preds[:8]

    lines = [f"\U0001f52e *Top Picks \u2014 {today}*", "_Ranked by AI confidence_", "=" * 30, ""]
    for i, pred in enumerate(top):
        lines.append(f"*{pred['league']}* ({pred['kick_off']})")
        lines.append(format_tip(pred, i))
        lines.append("")

    hc = [p for p in all_preds if p["confidence_score"] >= 0.72]
    o15p = [p for p in all_preds if p["over15"] >= 0.80]
    btts_p = [p for p in all_preds if p["btts"] >= 0.65]
    dc_p = [p for p in all_preds if p["double_home"] >= 0.80 or p["double_away"] >= 0.80]

    lines.append("=" * 30)
    lines.append(f"\U0001f4ca *Today's Summary:*")
    lines.append(f"   {len(all_preds)} matches analysed")
    lines.append(f"   \U0001f7e2 {len(hc)} high-confidence picks")
    lines.append(f"   \u26bd {len(o15p)} strong Over 1.5")
    lines.append(f"   \U0001f3af {len(btts_p)} likely BTTS")
    lines.append(f"   \U0001f91d {len(dc_p)} strong Double Chance")
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
        msg2.extend(lines[-8:])
        await update.message.reply_text("\n".join(msg2), parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def daily_summary(context: ContextTypes.DEFAULT_TYPE):
    chat_id = os.environ.get("DAILY_CHAT_ID")
    if not chat_id: return
    df = (datetime.utcnow() - timedelta(days=150)).strftime("%Y-%m-%d")
    dt = datetime.utcnow().strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    summary = [f"\U0001f305 *Daily Streak & AI Prediction Summary \u2014 {today}*\n"]
    all_preds = []

    for i, (code, info) in enumerate(FREE_LEAGUES.items()):
        if info.get("type") == "CUP": continue
        try:
            if i > 0: await asyncio.sleep(7)
            historical = await api.get_matches(code, df, dt)
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
            preds = await _fetch_league_predictions(code, info, df, dt)
            all_preds.extend(preds)
        except Exception as e: logger.error(f"Daily error {code}: {e}")

    if all_preds:
        all_preds.sort(key=lambda x: x["confidence_score"], reverse=True)
        summary.append(f"\n{'=' * 25}")
        summary.append(f"\U0001f52e *Top Predictions:*")
        for pred in all_preds[:5]:
            mp = max(pred["home_win"], pred["draw"], pred["away_win"])
            summary.append(f"  {pred['home_team']} vs {pred['away_team']}\n    \U0001f3af {pred['prediction']} ({mp:.0%})\n    ELO: {pred['home_elo']} vs {pred['away_elo']} | xG: {pred['home_xg']}-{pred['away_xg']}")

    if len(summary) > 1:
        summary.append(f"\n\u26a0\ufe0f _Statistical estimates only._")
        await context.bot.send_message(chat_id=int(chat_id), text="\n".join(summary), parse_mode=ParseMode.MARKDOWN)


def main():
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("\u26a0\ufe0f  Set TELEGRAM_BOT_TOKEN!"); return
    if FOOTBALL_API_KEY == "YOUR_FOOTBALL_DATA_API_KEY":
        print("\u26a0\ufe0f  Set FOOTBALL_API_KEY!"); return

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

    jq = app.job_queue
    if jq and os.environ.get("DAILY_CHAT_ID"):
        from datetime import time as dt_time
        jq.run_daily(daily_summary, time=dt_time(hour=8, minute=0), name="daily_summary")
        logger.info("Daily summary scheduled for 08:00 UTC")

    print("\U0001f916 Football Streak Tracker & AI Predictor v2 is running!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
