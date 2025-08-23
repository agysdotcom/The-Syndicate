import os
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st
import nltk
import pytz

# NLP setup
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# Config
st.set_page_config(page_title="THE SYNDICATE - Soccer Predictor", layout="wide")

API_FOOTBALL_KEY = "a6917f6db6a731e8b6cfa9f9f365a5ed"
THEODDSAPI_KEY = "69bb2856e8ec4ad7b9a12f305147408"
NEWSAPI_KEY = "c7d0efc525bf48199ab229f8f70fbc01"

BASE_FOOTBALL = "https://v3.football.api-sports.io"
BASE_ODDS = "https://api.the-odds-api.com/v4"

LEAGUES = {
    "All": 0,
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
    "Eredivisie": 88,
    "Primeira Liga": 94,
    "Scottish Premiership": 179,
    "Belgian Pro League": 144,
    "Champions League": 2,
    "Europa League": 3,
    "Conference League": 848,
}

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}
CAPE_TOWN_TZ = pytz.timezone("Africa/Johannesburg")  # GMT+2 identical

# CSS Styling
st.markdown("""
<style>
body {background:#0b0d12;}
.stApp {background:#0b0d12; color:#e5e7eb;}
.card {background:#121420; border:1px solid #1e2233; border-radius:18px; padding:18px; margin-bottom:18px;}
.odds-box {background:#0e111a; border:1px solid #20263b; border-radius:12px; padding:10px; margin:5px 0;}
.chip { padding:3px 10px; border-radius:999px; font-size:0.75rem; }
.chip.live { background:#172a1f; color:#22c55e; border:1px solid #134e4a; }
.chip.value { background:#19242d; color:#22d3ee; border:1px solid #164e63; }
.bar { height:10px; background:#0f172a; border-radius:999px; overflow:hidden; margin-top:5px;}
.bar > div { height:100%; background:linear-gradient(90deg,#22d3ee,#60a5fa); }
.neon-green { color:#22c55e; font-weight:bold;}
.neon-red { color:#f87171; font-weight:bold;}
.neon-amber { color:#fbbf24; font-weight:bold;}
.neon-blue { color:#60a5fa; font-weight:bold;}
.bottom-nav {position:fixed; left:0; right:0; bottom:0; background:#0c0f16; border-top:1px solid #1f2940; display:flex; gap:14px; justify-content:space-around; padding:10px 10px; z-index:999;}
.bottom-nav a { color:#cbd5e1; text-decoration:none; font-size:0.9rem; }
.bottom-nav a strong { display:block; font-size:0.75rem; color:#93c5fd; }
</style>
""", unsafe_allow_html=True)

# Helper functions
def safe_parse_datetime_utc_to_capetown(date_utc: str) -> datetime:
    try:
        if date_utc.endswith("Z"):
            date_utc = date_utc[:-1]
        dt_utc = datetime.fromisoformat(date_utc).replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(CAPE_TOWN_TZ)
    except Exception:
        return datetime.now(tz=CAPE_TOWN_TZ)

def sentiment_score(text_list: List[str]) -> float:
    if not text_list:
        return 0.0
    return np.mean([sia.polarity_scores(t)['compound'] for t in text_list])

def _proportional_devig(probs: List[float]) -> List[float]:
    s = sum(probs)
    return probs if s == 0 else [p/s for p in probs]

def _odds_to_implied(odds: List[float]) -> List[float]:
    probs = []
    for o in odds:
        try:
            o = float(o)
            probs.append(1.0 / o if o > 1e-9 else 0.0)
        except:
            probs.append(0.0)
    return probs

def poisson_pmf(k, lam):
    return np.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf_over(total: float, lam_total: float) -> float:
    floor_needed = int(math.floor(total + 0.5))
    c = sum(poisson_pmf(k, lam_total) for k in range(floor_needed))
    return 1.0 - c

def make_star_confidence(value: float) -> str:
    score = max(0.0, min(1.0, value))
    stars = int(np.clip(1 + round(4*score), 1, 5))
    return "⭐"*stars

# Fetching data
@st.cache_data(ttl=900)
def fetch_fixtures(league_id: int, date_iso: str) -> List[Dict]:
    try:
        url = f"{BASE_FOOTBALL}/fixtures"
        headers = HEADERS_FOOTBALL
        params = {"season": int(date_iso[:4]), "date": date_iso}
        if league_id != 0: params["league"] = league_id
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if not r.ok: return []
        data = r.json()
        fixtures = []
        finished_states = {"FT", "AET", "PEN", "INT"}
        for f in data.get("response", []):
            status = f["fixture"]["status"]["short"]
            if status in finished_states:
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
    except:
        return []

@st.cache_data(ttl=600)
def fetch_match_stats(fixture_id: int) -> Dict:
    return {"corners": {"home": 3, "away": 4}, "yellow_cards": {"home": 1, "away": 2}}

@st.cache_data(ttl=1800)
def fetch_odds(home:str, away:str, date_iso:str) -> Dict:
    try:
        url = f"{BASE_ODDS}/sports/soccer/odds"
        params = {
            "apiKey": THEODDSAPI_KEY,
            "regions": "uk,eu,us",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        r = requests.get(url, params=params, timeout=20)
        if not r.ok: return {}
        data = r.json()
        match = None
        target_date = date_iso[:10]
        for ev in data:
            if not ev.get("commence_time","").startswith(target_date): continue
            ht = ev.get("home_team","")
            at = ev.get("away_team","")
            if home.lower() in ht.lower() and away.lower() in at.lower():
                match = ev
                break
            if home.lower() in at.lower() and away.lower() in ht.lower():
                match = ev
                break
        return match or {}
    except:
        return {}

def extract_match_odds(odds_obj: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        sites = odds_obj.get("bookmakers", [])
        if not sites:
            return None, None, None
        market = sites[0].get("markets", [])
        prices = market.get("outcomes", [])
        d = {p['name'].lower(): p['price'] for p in prices if 'name' in p and 'price' in p}
        return d.get("home"), d.get("draw"), d.get("away")
    except:
        return None, None, None

@st.cache_data(ttl=900)
def fetch_news_snippets(team: str) -> List[str]:
    try:
        params = {
            "q": f"{team} injury press conference latest",
            "apiKey": NEWSAPI_KEY,
            "language": "en",
            "pageSize": 5,
            "sortBy": "publishedAt"
        }
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        data = r.json() if r.ok else {}
        return [a.get("title","") for a in data.get("articles", []) if a.get("title")]
    except:
        return []

# Prediction
def predict_win_odds(h_odds: Optional[float], d_odds: Optional[float], a_odds: Optional[float], news_score: float=0.0) -> Tuple[float, float, float]:
    probs = _odds_to_implied([h_odds, d_odds, a_odds])
    probs = _proportional_devig(probs)
    probs = [min(max(p + 0.05 * news_score, 0.0), 1.0) for p in probs]
    s = sum(probs)
    if s == 0:
        return 1/3, 1/3, 1/3
    return tuple([p/s for p in probs])

def over_under_probs(xg_home: float, xg_away: float) -> Dict[str, float]:
    total_xg = xg_home + xg_away
    lines = [0.5,1.5,2.5,3.5,4.5]
    res = {}
    for l in lines:
        res[f"Over {l}"] = poisson_cdf_over(l, total_xg)
        res[f"Under {l}"] = 1 - res[f"Over {l}"]
    return res

def prepare_fixtures_with_stats(league_id: int, date_iso: str) -> List[Dict]:
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
            "corners": stats.get("corners", {"home":0,"away":0}),
            "yellow_cards": stats.get("yellow_cards", {"home":0,"away":0}),
            "best_bet": best_bet,
            "confidence": confidence
        })
    return enriched

# Render game tabs
def render_game_tabs(game: Dict):
    subtabs = st.tabs(["Summary", "Corners", "Yellow Cards", "Over/Under"])
    with subtabs[0]:
        st.markdown(f"**Status:** {game['status']}")
        st.markdown(f"**Betting Recommendation:** {game['best_bet']}")
    with subtabs[1]:
        c = game.get("corners", {"home":0,"away":0})
        st.markdown(f"Home Corners: {c['home']} | Away Corners: {c['away']}")
    with subtabs[2]:
        y = game.get("yellow_cards", {"home":0,"away":0})
        st.markdown(f"Home Yellow Cards: {y['home']} | Away Yellow Cards: {y['away']}")
    with subtabs[3]:
        ou = game.get("over_under", {})
        st.markdown("**Over/Under Probabilities**")
        for key in sorted(ou.keys()):
            if key.lower().startswith("over"):
                st.markdown(f"{key}: {ou[key]:.1%}")

# Generate full Betslip
def generate_full_betslips(fixtures: List[Dict]) -> List[Dict]:
    conf_thresh = 0.6
    bets = []
    for f in fixtures:
        max_win_prob = max(f["home_prob"], f["draw_prob"], f["away_prob"])
        pick = None
        if f["home_prob"] == max_win_prob:
            pick = f"Home Win ({f['home']})"
        elif f["away_prob"] == max_win_prob:
            pick = f"Away Win ({f['away']})"
        if pick and max_win_prob > conf_thresh:
            bets.append({"match": f"{f['home']} vs {f['away']}", "type":"Win","pick":pick,"probability":max_win_prob,"confidence":f["confidence"]})
        for k,v in f["over_under"].items():
            if k.lower().startswith("over") and v>conf_thresh:
                bets.append({"match": f"{f['home']} vs {f['away']}", "type":"Over Goals","pick":k,"probability":v,"confidence":f["confidence"]})
        total_yc = f.get("yellow_cards",{}).get("home",0)+f.get("yellow_cards",{}).get("away",0)
        if total_yc>=6:
            bets.append({"match": f"{f['home']} vs {f['away']}", "type":"High Yellow Cards","pick":f"{total_yc} Yellow Cards","probability":None,"confidence":"N/A"})
        total_corners = f.get("corners",{}).get("home",0)+f.get("corners",{}).get("away",0)
        if total_corners>=10:
            bets.append({"match": f"{f['home']} vs {f['away']}", "type":"High Corners","pick":f"{total_corners} Corners","probability":None,"confidence":"N/A"})
    return bets

# Sidebar league select
league_selected = st.sidebar.selectbox("Select League", list(LEAGUES.keys()), index=0)
date_today = datetime.utcnow().strftime("%Y-%m-%d")
fixtures = []
if league_selected=="All":
    for lid in LEAGUES.values():
        if lid:
            fixtures.extend(prepare_fixtures_with_stats(lid,date_today))
else:
    fixtures = prepare_fixtures_with_stats(LEAGUES[league_selected],date_today)

league_names = sorted(set(f["leagueName"] for f in fixtures))
if league_names:
    league_tabs = st.tabs(league_names)
    for idx,lname in enumerate(league_names):
        with league_tabs[idx]:
            league_fixtures = [f for f in fixtures if f["leagueName"]==lname]
            if not league_fixtures:
                st.info(f"No matches available for league {lname}")
            for game in league_fixtures:
                st.subheader(f"{game['home']} vs {game['away']} - {safe_parse_datetime_utc_to_capetown(game['utc']).strftime('%Y-%m-%d %H:%M')}")
                render_game_tabs(game)
else:
    st.info("No matches available")

# Betslip
st.markdown("---")
st.header("Betslip - Best Recommendations")
full_betslip = generate_full_betslips(fixtures)
if not full_betslip:
    st.info("No bets available at the moment.")
else:
    for b in full_betslip:
        st.markdown(
            f"**{b['match']}** | Type: {b['type']} | Bet: {b['pick']} | "
            f"Probability: {b['probability']:.1%}" if b['probability'] is not None else "N/A"
            f" | Confidence: {b['confidence']}"
        )

# Bottom navigation
st.markdown("""
<div class="bottom-nav">
<a href="#">🏠<strong>Home</strong></a>
<a href="#">💱<strong>Trade</strong></a>
<a href="#">📝<strong>Betslip</strong></a>
<a href="#">🎯<strong>Sportsbook</strong></a>
<a href="#">📦<strong>Orders</strong></a>
</div>
""", unsafe_allow_html=True)
