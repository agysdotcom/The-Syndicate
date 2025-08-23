# app.py
import os
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st
import nltk
import pytz

# ---------------- NLP Setup ----------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# ---------------- Config ----------------
st.set_page_config(page_title="THE SYNDICATE - Soccer Predictor", layout="wide")

API_FOOTBALL_KEY = "a6917f6db6a731e8b6cfa9f9f365a5ed"
THEODDSAPI_KEY = "69bb2856e8ec4ad7b9a12f305147b408"
NEWSAPI_KEY = "c7d0efc525bf48199ab229f8f70fbc01"

BASE_FOOTBALL = "https://v3.football.api-sports.io"
BASE_ODDS = "https://api.the-odds-api.com/v4"

LEAGUES = {
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
JOHANNESBURG_TZ = pytz.timezone("Africa/Johannesburg")

# ---------------- Style ----------------
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

# ---------------- Header ----------------
st.markdown('<div style="text-align:center; margin-bottom: 1rem;">'
            '<span style="font-size:2.2rem; font-weight:800; color:#60a5fa;">⚡ THE SYNDICATE</span><br>'
            '<span style="font-size:1rem; color:#94a3b8;">Football Predictions • 2025–2026</span>'
            '</div>', unsafe_allow_html=True)

# ---------------- Helpers ----------------
def safe_parse_datetime_utc_to_johannesburg(date_utc: str) -> datetime:
    try:
        if date_utc.endswith("Z"):
            date_utc = date_utc[:-1]
        dt_utc = datetime.fromisoformat(date_utc).replace(tzinfo=timezone.utc)
        dt_johannesburg = dt_utc.astimezone(JOHANNESBURG_TZ)
        return dt_johannesburg
    except Exception:
        return datetime.now(tz=JOHANNESBURG_TZ)

def _proportional_devig(prob_list: List[float]) -> List[float]:
    s = sum(prob_list)
    return prob_list if s == 0 else [p / s for p in prob_list]

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

def poisson_cdf_over(total_line_half: float, lam_total: float) -> float:
    floor_needed = int(math.floor(total_line_half + 0.5))
    c = sum(poisson_pmf(k, lam_total) for k in range(floor_needed))
    return 1.0 - c

def make_star_confidence(value: float) -> str:
    score = max(0.0, min(1.0, value))
    stars = int(np.clip(1 + round(4*score), 1, 5))
    return "⭐"*stars

# ---------------- Fetch Fixtures ----------------
@st.cache_data(ttl=60*15)
def fetch_api_football_fixtures(league_id: int, date_iso: str) -> List[Dict]:
    try:
        url = f"{BASE_FOOTBALL}/fixtures"
        headers = {"x-apisports-key": API_FOOTBALL_KEY}
        params = {"league": league_id, "season": int(date_iso[:4]), "date": date_iso}
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if not r.ok: return []
        data = r.json()
        fixtures = []
        for f in data.get("response", []):
            fixture = f["fixture"]
            teams = f["teams"]
            fixtures.append({
                "leagueName": f["league"]["name"],
                "home": teams["home"]["name"],
                "away": teams["away"]["name"],
                "utc": fixture["date"],
                "status": fixture["status"]["short"],
            })
        return fixtures
    except Exception:
        return []

@st.cache_data(ttl=60*30)
def fetch_odds(home: str, away: str, date_iso: str) -> Dict:
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
            if not ev.get("commence_time", "").startswith(target_date): continue
            ht = ev["home_team"]; at = ev["away_team"]
            if home.lower() in ht.lower() and away.lower() in at.lower(): match = ev; break
            if home.lower() in at.lower() and away.lower() in ht.lower(): match = ev; break
        return match or {}
    except: return {}

def extract_match_odds(odds_obj: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        sites = odds_obj.get("bookmakers", [])
        if not sites: return None, None, None
        market = sites[0].get("markets", [])
        prices = market.get("outcomes", [])
        d = {p['name'].lower(): p['price'] for p in prices if 'name' in p and 'price' in p}
        return d.get("home"), d.get("draw"), d.get("away")
    except: return None, None, None

@st.cache_data(ttl=60*30)
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
        articles = data.get("articles", [])
        return [a.get("title","") for a in articles if a.get("title")]
    except: return []

def sentiment_score(text_list: List[str]) -> float:
    if not text_list: return 0.0
    scores = [sia.polarity_scores(t)['compound'] for t in text_list]
    return np.mean(scores)

# ---------------- Predictions ----------------
def predict_win_odds(h_odds: Optional[float], d_odds: Optional[float], a_odds: Optional[float], news_score: float=0.0) -> Tuple[float, float, float]:
    probs = _odds_to_implied([h_odds, d_odds, a_odds])
    probs = _proportional_devig(probs)
    probs = [min(max(p + 0.05 * news_score, 0.0), 1.0) for p in probs]
    s = sum(probs)
    if s==0: return 1/3,1/3,1/3
    return tuple([p/s for p in probs])

def over_under_probs(xg_home: float, xg_away: float) -> Dict[str, float]:
    total_xg = xg_home + xg_away
    lines = [0.5,1.5,2.5,3.5,4.5]
    res = {}
    for l in lines:
        res[f"Over {l}"] = poisson_cdf_over(l, total_xg)
        res[f"Under {l}"] = 1 - res[f"Over {l}"]
    return res

# ---------------- Fixtures Preparation ----------------
def prepare_fixtures_for_date(date_iso: str) -> List[Dict]:
    all_fixtures = []
    for league_id in LEAGUES.values():
        for f in fetch_api_football_fixtures(league_id, date_iso):
            o = fetch_odds(f['home'], f['away'], date_iso)
            h_od, d_od, a_od = extract_match_odds(o)
            news = fetch_news_snippets(f['home']) + fetch_news_snippets(f['away'])
            news_s = sentiment_score(news)
            ph, pd, pa = predict_win_odds(h_od, d_od, a_od, news_s)
            xg_home, xg_away = 1.3 + ph*1.5, 1.1 + pa*1.5
            ou = over_under_probs(xg_home, xg_away)
            best_bet = "🏆 Strong Pick" if max(ph,pa) > 0.6 else ""
            confidence = make_star_confidence(max(ph,pa))
            all_fixtures.append({
                **f,
                "home_prob": ph,
                "draw_prob": pd,
                "away_prob": pa,
                "news": news,
                "over_under": ou,
                "best_bet": best_bet,
                "confidence": confidence,
            })
    return all_fixtures

# ---------------- Render Fixture ----------------
def render_fixture(f):
    with st.expander(f"{f['home']} vs {f['away']} | {f['leagueName']} | {safe_parse_datetime_utc_to_johannesburg(f['utc']).strftime('%H:%M SAST')} | Status: {f['status']}"):
        col1,col2,col3 = st.columns([2,3,2])
        col1.markdown(f"**Teams:** {f['home']} vs {f['away']}")
        col2.markdown(f"**Win Probabilities:** Home: {f['home_prob']:.0%} | Draw: {f['draw_prob']:.0%} | Away: {f['away_prob']:.0%}")
        col2.progress(f['home_prob'])
        col3.markdown(f"**Confidence:** {f['confidence']}")
        if f['best_bet']:
            col3.markdown(f"<span style='background:#22d3ee;color:#0b0d12;padding:4px 8px;border-radius:10px;'>{f['best_bet']}</span>", unsafe_allow_html=True)
        st.markdown("**Over/Under Probabilities**")
        ou_table = pd.DataFrame(list(f['over_under'].items()), columns=["Market","Probability"])
        st.table(ou_table.style.format({"Probability":"{:.0%}"}))
        st.markdown("**Recent News:**")
        for n in f['news']: st.markdown(f"- {n}")

# ---------------- Tabs ----------------
date_today = datetime.utcnow().strftime("%Y-%m-%d")
date_tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
tabs = st.tabs(["Favourites","Live","Today","Tomorrow","Betslips"])

fixtures_today = prepare_fixtures_for_date(date_today)
fixtures_tomorrow = prepare_fixtures_for_date(date_tomorrow)

# ---------------- Betslip Generators ----------------
CONFIDENCE_THRESHOLD = 0.6

def generate_winners_betslip(fixtures): return [{"match":f"{f['home']} vs {f['away']}", "pick": f"{f['home']} Win" if f['home_prob']>f['away_prob'] else f"{f['away']} Win", "prob": max(f['home_prob'],f['away_prob'])} for f in fixtures if max(f['home_prob'],f['away_prob'])>=CONFIDENCE_THRESHOLD]

def generate_goals_betslip(fixtures): return [{"match": f"{f['home']} vs {f['away']}", "pick": max(f['over_under'].items(), key=lambda x: abs(x[1]-0.65))[0], "prob": max(f['over_under'].values())} for f in fixtures]

def generate_mixed_betslip(fixtures): return generate_winners_betslip(fixtures) + generate_goals_betslip(fixtures)

def generate_draws_betslip(fixtures): return [{"match":f"{f['home']} vs {f['away']}", "pick":"Draw","prob":f['draw_prob']} for f in fixtures]

def generate_value_betslip(fixtures): return [{"match":f"{f['home']} vs {f['away']}", "pick": "Home" if f['home_prob']>0.6 else "Away","prob": max(f['home_prob'],f['away_prob'])} for f in fixtures]

def render_betslip_selection(title:str, bets:List[Dict], betslip_state:List):
    st.markdown(f"### {title}")
    if not bets: st.info("No strong bets for this category."); return
    for b in bets:
        if st.checkbox(f"{b['match']} | Pick: {b['pick']} | Prob: {b['prob']:.0%}", key=f"{title}_{b['match']}"):
            betslip_state.append(b)

# ---------------- Render Tabs ----------------
with tabs[0]:
    st.info("Favourites tab content goes here.")
with tabs[1]:
    st.info("Live tab content goes here.")
with tabs[2]:
    for fx in fixtures_today: render_fixture(fx)
with tabs[3]:
    for fx in fixtures_tomorrow: render_fixture(fx)
with tabs[4]:
    # Betslips interactive
    st.info("Select bets to add to your personal Betslip")
    user_betslip = []
    render_betslip_selection("🏆 Winners Betslip", generate_winners_betslip(fixtures_today), user_betslip)
    render_betslip_selection("⚽ Goals Betslip", generate_goals_betslip(fixtures_today), user_betslip)
    render_betslip_selection("🔀 Mixed Betslip", generate_mixed_betslip(fixtures_today), user_betslip)
    render_betslip_selection("🎯 Draws Betslip", generate_draws_betslip(fixtures_today), user_betslip)
    render_betslip_selection("💰 Value Betslip", generate_value_betslip(fixtures_today), user_betslip)
    if user_betslip:
        st.markdown("## Your Virtual Betslip")
        df = pd.DataFrame(user_betslip)
        st.table(df)

# ---------------- Bottom Nav ----------------
st.markdown("""
<div class="bottom-nav">
<a href="#">🏠<strong>Home</strong></a>
<a href="#">💱<strong>Trade</strong></a>
<a href="#">📝<strong>Betslip</strong></a>
<a href="#">🎯<strong>Sportsbook</strong></a>
<a href="#">📦<strong>Orders</strong></a>
</div>
""", unsafe_allow_html=True)
