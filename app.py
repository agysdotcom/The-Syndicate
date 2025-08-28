# app.py
import os
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -------------------------
# Basic setup & keys
# -------------------------
nltk_available = True
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon")
    except Exception:
        nltk_available = False

if nltk_available:
    sia = SentimentIntensityAnalyzer()
else:
    # fallback dumb scorer
    class _DummySIA:
        def polarity_scores(self, t): return {"compound": 0.0}
    sia = _DummySIA()

st.set_page_config(page_title="THE SYNDICATE - AI Soccer Predictor", layout="wide")

# NOTE: keys are taken from your snippet. Consider storing these as env vars in production.
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "a6917f6db6a731e8b6cfa9f9f365a5ed")
THEODDSAPI_KEY = os.getenv("THEODDSAPI_KEY", "69bb2856e8ec4ad7b9a12f305147408")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "c7d0efc525bf48199ab229f8f70fbc01")

BASE_FOOTBALL = "https://v3.football.api-sports.io"
BASE_ODDS = "https://api.the-odds-api.com/v4"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

CAPE_TOWN_TZ = pytz.timezone("Africa/Johannesburg")

LEAGUES = {
    "All": 0, "Premier League": 39, "La Liga": 140, "Serie A": 135,
    "Bundesliga": 78, "Ligue 1": 61, "Eredivisie": 88, "Primeira Liga": 94,
    "Scottish Premiership": 179, "Belgian Pro League": 144,
    "Champions League": 2, "Europa League": 3, "Conference League": 848
}

# -------------------------
# Helpers: time / math / models
# -------------------------
def safe_parse_datetime_utc_to_capetown(date_utc: str) -> datetime:
    try:
        dt_utc = datetime.fromisoformat(date_utc.replace("Z", "")).replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(CAPE_TOWN_TZ)
    except Exception:
        return datetime.now(tz=CAPE_TOWN_TZ)

def sentiment_score(text_list: List[str]) -> float:
    if not text_list:
        return 0.0
    scores = []
    for t in text_list:
        try:
            scores.append(sia.polarity_scores(t)['compound'])
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))

def _odds_to_implied(odds: List[Optional[float]]) -> List[float]:
    probs = []
    for o in odds:
        try:
            if o is None:
                probs.append(0.0)
            else:
                o = float(o)
                probs.append(1.0 / o if o > 1e-9 else 0.0)
        except Exception:
            probs.append(0.0)
    return probs

def _proportional_devig(probs: List[float]) -> List[float]:
    s = sum(probs)
    if s > 0:
        return [p / s for p in probs]
    # fallback: equal
    if len(probs) == 0:
        return [1/3, 1/3, 1/3]
    return [1 / len(probs)] * len(probs)

def poisson_pmf(k, lam):
    # safe pmf
    return np.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf_over(total: float, lam_total: float) -> float:
    # probability total goals > total (e.g. over 2.5)
    # compute P(sum > floor(total + 0.5)) by summing up to that floor and subtracting from 1
    floor_needed = int(math.floor(total + 0.5))
    c = 0.0
    for k in range(0, floor_needed + 1):
        c += poisson_pmf(k, lam_total)
    return max(0.0, 1.0 - c)

def make_star_confidence(value: float) -> str:
    # value in [0,1] -> 1..5 stars
    stars = int(np.clip(1 + round(4 * value), 1, 5))
    return "⭐" * stars

# -------------------------
# API fetchers (cached)
# -------------------------
@st.cache_data(ttl=900)
def fetch_fixtures(league_id: int, date_iso: str) -> List[Dict]:
    try:
        params = {"season": int(date_iso[:4]), "date": date_iso}
        if league_id != 0:
            params["league"] = league_id
        r = requests.get(f"{BASE_FOOTBALL}/fixtures", headers=HEADERS_FOOTBALL, params=params, timeout=15)
        data = r.json()
        finished = {"FT", "AET", "PEN", "INT"}
        fixtures = []
        for f in data.get("response", []):
            status = f["fixture"]["status"]["short"]
            if status in finished:
                continue
            fixtures.append({
                "leagueName": f["league"]["name"],
                "home": f["teams"]["home"]["name"],
                "away": f["teams"]["away"]["name"],
                "utc": f["fixture"]["date"],
                "status": status,
                "fixture_id": f["fixture"]["id"]
            })
        return fixtures
    except Exception:
        return []

@st.cache_data(ttl=600)
def fetch_match_stats(fixture_id: int) -> Dict:
    # Placeholder stats; replace with real API call if available
    # Keep small values so filters like 'high corners' still work
    return {"corners": {"home": 3, "away": 4}, "yellow_cards": {"home": 1, "away": 2}}

@st.cache_data(ttl=1800)
def fetch_odds(home: str, away: str, date_iso: str) -> Dict:
    try:
        # This endpoint returns many events; we filter locally for the fixture.
        params = {"apiKey": THEODDSAPI_KEY, "regions": "uk,eu,us", "markets": "h2h,totals", "oddsFormat": "decimal", "dateFormat": "iso"}
        r = requests.get(f"{BASE_ODDS}/sports/soccer/odds", params=params, timeout=20)
        data = r.json() if r.ok else []
        target = date_iso[:10]
        for ev in data:
            if not ev.get("commence_time", "").startswith(target):
                continue
            ht, at = ev.get("home_team",""), ev.get("away_team","")
            if home.lower() in ht.lower() and away.lower() in at.lower():
                return ev
            if home.lower() in at.lower() and away.lower() in ht.lower():
                return ev
        return {}
    except Exception:
        return {}

def extract_match_odds(odds_obj: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        sites = odds_obj.get("bookmakers", [])
        if not sites:
            return None, None, None
        # prefer the first site; you could aggregate best odds across bookmakers in future
        market_list = sites[0].get("markets", [])
        # find h2h market
        for m in market_list:
            if m.get("key") in ("h2h", "h2h"):
                outcomes = m.get("outcomes", [])
                d = {p['name'].lower(): p.get('price') for p in outcomes if 'name' in p}
                return d.get("home"), d.get("draw"), d.get("away")
        # fallback: try outcomes in first market
        m = market_list[0] if market_list else {"outcomes": []}
        prices = m.get("outcomes", [])
        d = {p['name'].lower(): p.get('price') for p in prices if 'name' in p}
        return d.get("home"), d.get("draw"), d.get("away")
    except Exception:
        return None, None, None

@st.cache_data(ttl=900)
def fetch_news_snippets(team: str) -> List[str]:
    try:
        params = {"q": f"{team} injury press conference latest", "apiKey": NEWSAPI_KEY, "language": "en", "pageSize": 5, "sortBy": "publishedAt"}
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        data = r.json() if r.ok else {}
        return [a.get("title", "") for a in data.get("articles", []) if a.get("title")]
    except Exception:
        return []

# -------------------------
# Prediction functions
# -------------------------
def predict_win_odds(h_odds: Optional[float], d_odds: Optional[float], a_odds: Optional[float], news_score: float=0) -> Tuple[float, float, float]:
    probs = _odds_to_implied([h_odds, d_odds, a_odds])
    probs = _proportional_devig(probs)
    # nudging by news sentiment (small effect)
    probs = [min(max(p + 0.05 * news_score, 0), 1) for p in probs]
    s = sum(probs)
    if s == 0:
        return 1/3, 1/3, 1/3
    return tuple(p/s for p in probs)

def over_under_probs(xg_home: float, xg_away: float) -> Dict[str, float]:
    total = xg_home + xg_away
    res = {}
    for l in [0.5,1.5,2.5,3.5,4.5]:
        res[f"Over {l}"] = poisson_cdf_over(l, total)
        res[f"Under {l}"] = 1 - res[f"Over {l}"]
    return res

# -------------------------
# Enrichment pipeline
# -------------------------
def prepare_fixtures_with_stats(league_id: int, date_iso: str) -> List[Dict]:
    fixtures = fetch_fixtures(league_id, date_iso)
    enriched = []
    for f in fixtures:
        o = fetch_odds(f["home"], f["away"], date_iso)
        h_od, d_od, a_od = extract_match_odds(o)
        # gather news for both teams
        news = []
        news += fetch_news_snippets(f["home"])
        news += fetch_news_snippets(f["away"])
        news_s = sentiment_score(news)
        ph, pd, pa = predict_win_odds(h_od, d_od, a_od, news_s)
        # small heuristic xG baseline + model
        xg_home, xg_away = 1.3 + ph*1.5, 1.1 + pa*1.5
        ou = over_under_probs(xg_home, xg_away)
        stats = fetch_match_stats(f["fixture_id"])
        best_bet = "🏆 Strong Pick" if max(ph, pa) > 0.6 else ""
        confidence = make_star_confidence(max(ph, pa))
        enriched.append({
            **f,
            "home_prob": ph, "draw_prob": pd, "away_prob": pa,
            "news_sentiment": news_s,
            "over_under": ou,
            "corners": stats.get("corners", {"home":0,"away":0}),
            "yellow_cards": stats.get("yellow_cards", {"home":0,"away":0}),
            "best_bet": best_bet,
            "confidence": confidence,
            "book_odds": {"home": h_od, "draw": d_od, "away": a_od},
            "odds_raw_object": o
        })
    return enriched

# -------------------------
# Narrative & bets
# -------------------------
def generate_narrative(game: Dict) -> str:
    ph, pd, pa = game['home_prob'], game['draw_prob'], game['away_prob']
    outcome = "Home Win" if ph > pa and ph > pd else ("Away Win" if pa > ph and pa > pd else "Draw")
    confidence = make_star_confidence(max(ph, pd, pa))
    ou_total = game.get("over_under", {}).get("Over 2.5", 0)
    narrative = f"Prediction: {outcome} ({max(ph, pd, pa):.0%} probability, {confidence})\n"
    narrative += f"Expected total goals: {ou_total:.0%} chance of over 2.5 goals.\n"
    if game.get('news_sentiment', 0) > 0:
        narrative += "Recent news indicates positive sentiment for teams.\n"
    c = game.get("corners", {"home":0, "away":0})
    y = game.get("yellow_cards", {"home":0, "away":0})
    narrative += f"Expected Corners: {c['home']} - {c['away']}, Yellow Cards: {y['home']} - {y['away']}\n"
    return narrative

def generate_betslips(fixtures: List[Dict]) -> Dict[str, List[Dict]]:
    conf_thresh = 0.60
    winners, goals, yellow, corners = [], [], [], []
    for f in fixtures:
        max_win = max(f["home_prob"], f["draw_prob"], f["away_prob"])
        pick = None
        book_price = None
        if f["home_prob"] == max_win:
            pick = f"Home Win ({f['home']})"
            book_price = f["book_odds"].get("home")
        elif f["away_prob"] == max_win:
            pick = f"Away Win ({f['away']})"
            book_price = f["book_odds"].get("away")
        else:
            book_price = f["book_odds"].get("draw")
        if max_win > conf_thresh and pick:
            winners.append({"match": f"{f['home']} vs {f['away']}", "pick": pick, "prob": max_win, "confidence": f["confidence"], "game": f, "book_price": book_price})
        for k, v in f["over_under"].items():
            if k.lower().startswith("over") and v > conf_thresh:
                goals.append({"match": f"{f['home']} vs {f['away']}", "pick": k, "prob": v, "confidence": f["confidence"], "game": f, "book_price": None})
        if sum(f.get("yellow_cards", {"home":0,"away":0}).values()) >= 3:
            yellow.append({"match": f"{f['home']} vs {f['away']}", "pick": "High Yellow Cards", "prob": None, "confidence": "N/A", "game": f})
        if sum(f.get("corners", {"home":0,"away":0}).values()) >= 6:
            corners.append({"match": f"{f['home']} vs {f['away']}", "pick": "High Corners", "prob": None, "confidence": "N/A", "game": f})
    return {
        "Winner Betslip": winners,
        "Over Goals Betslip": goals,
        "Yellow Cards Betslip": yellow,
        "Corners Betslip": corners,
    }

# -------------------------
# Betslip state helpers
# -------------------------
def init_session_state():
    if "betslip" not in st.session_state:
        st.session_state["betslip"] = []
    if "bankroll" not in st.session_state:
        st.session_state["bankroll"] = 100.0  # default 100 units
    if "risk_profile" not in st.session_state:
        st.session_state["risk_profile"] = "Balanced"

def add_to_betslip(selection: Dict):
    st.session_state["betslip"].append(selection)

def remove_from_betslip(idx: int):
    if 0 <= idx < len(st.session_state["betslip"]):
        st.session_state["betslip"].pop(idx)

def clear_betslip():
    st.session_state["betslip"] = []

def compute_combined_fractional_odds(legs: List[Dict]) -> float:
    # multiply decimal odds; if odds missing treat as 1.0 (shouldn't happen)
    o = 1.0
    for leg in legs:
        od = leg.get("odds")
        if od is None:
            od = 1.0
        o *= float(od)
    return o

def implied_prob_from_odds(decimal_odds: float) -> float:
    return 1.0 / decimal_odds if decimal_odds > 0 else 0.0

def estimate_fair_odds_from_model_prob(p: float) -> float:
    return 1.0 / p if p > 0 else float("inf")

def edge_from_book_and_model(book_odds: float, model_p: float) -> Optional[float]:
    if book_odds is None or model_p is None or model_p <= 0:
        return None
    fair = estimate_fair_odds_from_model_prob(model_p)
    return (book_odds / fair) - 1.0

def kelly_fraction(edge: float, odds: float) -> float:
    # simplified Kelly: f* = edge / (odds - 1)
    if edge is None or odds is None or odds <= 1:
        return 0.0
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = edge / b
    if np.isnan(f) or f <= 0:
        return 0.0
    return float(np.clip(f, 0.0, 1.0))

# -------------------------
# Visual helpers
# -------------------------
def plot_total_goals_distribution(xg_total: float, cutoff: float = 2.5):
    # PMF for totals 0..8 (approx)
    k = list(range(0, 9))
    pmf = [poisson_pmf(i, xg_total) for i in k]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(k, pmf)
    ax.set_xlabel("Total goals")
    ax.set_ylabel("Probability")
    ax.set_xticks(k)
    # vertical line at cutoff (2.5 -> between 2 and 3)
    ax.axvline(x=2.5, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"Total goals distribution (λ={xg_total:.2f}) — cutoff {cutoff}")
    st.pyplot(fig)
    plt.close(fig)

# -------------------------
# UI Layout
# -------------------------
def main():
    init_session_state()

    st.markdown(
        """
    <style>
    /* Dark theme baseline */
    .stApp {
        background-color: #0f1115;
        color: #e6eef8;
    }
    .card {
        background: linear-gradient(
