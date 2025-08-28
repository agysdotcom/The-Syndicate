import os
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import requests
import streamlit as st
import pytz
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# NLP setup
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="THE SYNDICATE - AI Soccer Predictor", layout="wide")

# API keys (use env vars or hardcoded for demo)
API_FOOTBALL_KEY = "a6917f6db6a731e8b6cfa9f9f365a5ed"
THEODDSAPI_KEY = "69bb2856e8ec4ad7b9a12f305147408"
NEWSAPI_KEY = "c7d0efc525bf48199ab229f8f70fbc01"

BASE_FOOTBALL = "https://v3.football.api-sports.io"
BASE_ODDS = "https://api.the-odds-api.com/v4"

LEAGUES = {
    "All": 0, "Premier League": 39, "La Liga": 140, "Serie A": 135,
    "Bundesliga": 78, "Ligue 1": 61, "Eredivisie": 88, "Primeira Liga": 94,
    "Scottish Premiership": 179, "Belgian Pro League": 144,
    "Champions League": 2, "Europa League": 3, "Conference League": 848,
}

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}
CAPE_TOWN_TZ = pytz.timezone("Africa/Johannesburg")

# Custom CSS for card-based UI
st.markdown("""
<style>
.card {
    background: #15202b;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgb(32 34 44 / 0.5);
    color: #e1e8ed;
}
.league-header {
    font-weight: 700;
    font-size: 1.2rem;
    margin-bottom: 0.5rem;
    color: #1da1f2;
}
.teams {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 8px;
}
.pred-bar {
    height: 14px;
    border-radius: 7px;
    margin-bottom: 6px;
    background: #38444d;
    overflow: hidden;
}
.home-bar {
    background: #17bf63;
}
.away-bar {
    background: #e0245e;
}
.over-bar {
    background: #794bc4;
}
.prob-labels {
    font-size: 0.75rem;
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
}
.stats-box {
    padding: 6px 12px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-right: 6px;
    margin-bottom: 6px;
}
.corners-box {
    background-color: #1da1f2;
    color: white;
}
.yellow-box {
    background-color: #f5a623;
    color: black;
}
.strong-pick {
    background-color: #ffad1f;
    color: black;
    padding: 0 10px;
    border-radius: 12px;
    font-weight: 700;
    float: right;
}
.confidence-stars {
    float: right;
    font-size: 1rem;
    margin-left: 8px;
}
.add-betslip {
    text-align: right;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)


# Helper functions (safe parse date, probabilities, sentiment ...)
def safe_parse_datetime_utc_to_capetown(date_utc: str) -> datetime:
    try:
        dt_utc = datetime.fromisoformat(date_utc.replace("Z", "")).replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(CAPE_TOWN_TZ)
    except Exception:
        return datetime.now(tz=CAPE_TOWN_TZ)

def sentiment_score(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return np.mean([sia.polarity_scores(t)['compound'] for t in texts])

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
    if len(probs) == 0:
        return [1/3, 1/3, 1/3]
    return [1 / len(probs)] * len(probs)

def poisson_pmf(k, lam):
    return np.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf_over(total: float, lam_total: float) -> float:
    floor_needed = int(math.floor(total + 0.5))
    c = sum(poisson_pmf(k, lam_total) for k in range(floor_needed+1))
    return max(0.0, 1.0 - c)

def make_star_confidence(value: float) -> str:
    stars = int(np.clip(1 + round(4 * value), 1, 5))
    return "⭐" * stars


# API Fetchers -
@st.cache_data(ttl=900)
def fetch_fixtures(league_id: int, date_iso: str) -> List[Dict]:
    try:
        params = {"season": int(date_iso[:4]), "date": date_iso}
        if league_id:
            params["league"] = league_id
        r = requests.get(f"{BASE_FOOTBALL}/fixtures", headers=HEADERS_FOOTBALL, params=params, timeout=15)
        data = r.json()
        finished_status = {"FT", "AET", "PEN", "INT"}
        fixtures = []
        for f in data.get("response", []):
            status = f["fixture"]["status"]["short"]
            if status in finished_status:
                continue
            fixtures.append({
                "leagueName": f["league"]["name"],
                "home": f["teams"]["home"]["name"],
                "away": f["teams"]["away"]["name"],
                "utc": f["fixture"]["date"],
                "status": status,
                "fixture_id": f["fixture"]["id"],
            })
        return fixtures
    except:
        return []

@st.cache_data(ttl=600)
def fetch_match_stats(fixture_id: int) -> Dict:
    # Placeholder data
    return {"corners": {"home": 3, "away": 4}, "yellow_cards": {"home": 1, "away": 2}}

@st.cache_data(ttl=1800)
def fetch_odds(home: str, away: str, date_iso: str) -> Dict:
    try:
        params = {"apiKey": THEODDSAPI_KEY, "regions": "uk,eu,us", "markets": "h2h,totals", "oddsFormat": "decimal", "dateFormat": "iso"}
        r = requests.get(f"{BASE_ODDS}/sports/soccer/odds", params=params, timeout=20)
        data = r.json() if r.ok else []
        target = date_iso[:10]
        for ev in data:
            if not ev.get("commence_time", "").startswith(target):
                continue
            ht, at = ev.get("home_team", ""), ev.get("away_team", "")
            if home.lower() in ht.lower() and away.lower() in at.lower():
                return ev
            if home.lower() in at.lower() and away.lower() in ht.lower():
                return ev
        return {}
    except:
        return {}

def extract_match_odds(odds_obj: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        sites = odds_obj.get("bookmakers", [])
        if not sites:
            return None, None, None
        market_list = sites[0].get("markets", [])
        for m in market_list:
            if m.get("key") == "h2h":
                outcomes = m.get("outcomes", [])
                d = {p['name'].lower(): p.get('price') for p in outcomes if 'name' in p}
                return d.get("home"), d.get("draw"), d.get("away")
        m = market_list[0] if market_list else {"outcomes": []}
        prices = m.get("outcomes", [])
        d = {p['name'].lower(): p.get('price') for p in prices if 'name' in p}
        return d.get("home"), d.get("draw"), d.get("away")
    except:
        return None, None, None

@st.cache_data(ttl=900)
def fetch_news_snippets(team: str) -> List[str]:
    try:
        params = {"q": f"{team} injury press conference latest", "apiKey": NEWSAPI_KEY, "language": "en", "pageSize": 5, "sortBy": "publishedAt"}
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        data = r.json() if r.ok else {}
        return [a.get("title", "") for a in data.get("articles", []) if a.get("title")]
    except:
        return []


# Prediction functions
def predict_win_odds(h_odds: Optional[float], d_odds: Optional[float], a_odds: Optional[float], news_score: float=0) -> Tuple[float, float, float]:
    probs = _odds_to_implied([h_odds, d_odds, a_odds])
    probs = _proportional_devig(probs)
    probs = [min(max(p + 0.05 * news_score, 0), 1) for p in probs]
    s = sum(probs)
    if s == 0:
        return 1/3, 1/3, 1/3
    return tuple(p / s for p in probs)

def over_under_probs(xg_home: float, xg_away: float) -> Dict[str, float]:
    total = xg_home + xg_away
    res = {}
    for l in [0.5, 1.5, 2.5, 3.5, 4.5]:
        res[f"Over {l}"] = poisson_cdf_over(l, total)
        res[f"Under {l}"] = 1 - res[f"Over {l}"]
    return res


def prepare_fixtures_with_stats(league_id:int, date_iso:str)->List[Dict]:
    fixtures = fetch_fixtures(league_id, date_iso)
    enriched = []
    for f in fixtures:
        o = fetch_odds(f["home"], f["away"], date_iso)
        h_od, d_od, a_od = extract_match_odds(o)
        news = fetch_news_snippets(f["home"]) + fetch_news_snippets(f["away"])
        news_s = sentiment_score(news)
        ph, pd, pa = predict_win_odds(h_od, d_od, a_od, news_s)
        xg_home, xg_away = 1.3 + ph * 1.5, 1.1 + pa * 1.5
        ou = over_under_probs(xg_home, xg_away)
        stats = fetch_match_stats(f["fixture_id"])
        best_bet = "🏆 Strong Pick" if max(ph, pa) > 0.6 else ""
        confidence = make_star_confidence(max(ph, pa))
        enriched.append({
            **f,
            "home_prob": ph,
            "draw_prob": pd,
            "away_prob": pa,
            "news_sentiment": news_s,
            "over_under": ou,
            "corners": stats.get("corners", {"home": 0, "away": 0}),
            "yellow_cards": stats.get("yellow_cards", {"home": 0, "away": 0}),
            "best_bet": best_bet,
            "confidence": confidence
        })
    return enriched


def color_bar(width: float, css_class: str) -> str:
    width_pct = min(max(width*100, 0), 100)
    return f'<div class="pred-bar"><div class="{css_class}" style="width:{width_pct}%;"></div></div>'


def generate_card(game: Dict):
    dt_str = safe_parse_datetime_utc_to_capetown(game['utc']).strftime("%Y-%m-%d %H:%M")
    home_prob = game["home_prob"]
    away_prob = game["away_prob"]
    over25_prob = game["over_under"].get("Over 2.5", 0)
    corners = game.get("corners", {"home": 0, "away": 0})
    yellows = game.get("yellow_cards", {"home": 0, "away": 0})

    strong_pick_html = ""
    if game["best_bet"]:
        strong_pick_html = f"<span class='strong-pick'>{game['best_bet']}<span class='confidence-stars'>{game['confidence']}</span></span>"
    else:
        strong_pick_html = f"<span class='confidence-stars'>{game['confidence']}</span>"

    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='league-header'>{game['leagueName']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div><b>{game['home']} vs {game['away']}</b> <span style='float:right;'>{dt_str}</span> {strong_pick_html}</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='prob-labels'>
            <div>Home: {home_prob:.0%}</div>
            <div>Over 2.5: {over25_prob:.0%}</div>
            <div>Away: {away_prob:.0%}</div>
        </div>
        {color_bar(home_prob, 'home-bar')}
        {color_bar(over25_prob, 'over-bar')}
        {color_bar(away_prob, 'away-bar')}
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div>
            <span class='stats-box corners-box'>Corners: {corners['home']} - {corners['away']}</span>
            <span class='stats-box yellow-box'>Yellow Cards: {yellows['home']} - {yellows['away']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(f"Add to Betslip", key=f"add_{game['fixture_id']}"):
        if "betslip" not in st.session_state:
            st.session_state["betslip"] = []
        st.session_state["betslip"].append(game)
        st.success("Added to Betslip!")

    st.markdown("</div>", unsafe_allow_html=True)


league_selected = st.sidebar.selectbox("Select League", list(LEAGUES.keys()), 0)
date_today = datetime.utcnow().strftime("%Y-%m-%d")

fixtures = []
if league_selected == "All":
    for lid in LEAGUES.values():
        if lid:
            fixtures.extend(prepare_fixtures_with_stats(lid, date_today))
else:
    fixtures = prepare_fixtures_with_stats(LEAGUES[league_selected], date_today)

if not fixtures:
    st.info("No fixtures available for the selected league and date.")
else:
    # Layout fixtures in rows of two responsive columns
    for i in range(0, len(fixtures), 2):
        row = fixtures[i:i+2]
        cols = st.columns(len(row))
        for col, game in zip(cols, row):
            with col:
                generate_card(game)

# Betslip display
if "betslip" in st.session_state and st.session_state["betslip"]:
    st.markdown("---")
    st.header("Your Betslip")
    for bet in st.session_state["betslip"]:
        st.write(f"{bet['home']} vs {bet['away']}: {bet['best_bet']} with confidence {bet['confidence']}")
