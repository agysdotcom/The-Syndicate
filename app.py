import os
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st
import nltk

# NLP setup for SentimentIntensityAnalyzer
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
BASE_OPENLIGA = "https://api.openligadb.de"
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

# ---------------- Style ----------------
st.markdown(
    """
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
    .neon-green { color:#22c55e; }
    .neon-red { color:#f87171; }
    .neon-amber { color:#fbbf24; }
    .neon-blue { color:#60a5fa; }
    .bottom-nav {position:fixed; left:0; right:0; bottom:0; background:#0c0f16; border-top:1px solid #1f2940; display:flex; gap:14px; justify-content:space-around; padding:10px 10px; z-index:999;}
    .bottom-nav a { color:#cbd5e1; text-decoration:none; font-size:0.9rem; }
    .bottom-nav a strong { display:block; font-size:0.75rem; color:#93c5fd; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center; margin-bottom: 0.8rem;">
    <span style="font-size:2.2rem; font-weight:800; color:#60a5fa;">⚡ THE SYNDICATE</span><br>
    <span style="font-size:1rem; color:#94a3b8;">Football Predictions • 2025–2026</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
def safe_parse_datetime(date_utc: str) -> datetime:
    try:
        if date_utc.endswith("Z"):
            date_utc = date_utc[:-1] + "+00:00"
        return datetime.fromisoformat(date_utc)
    except Exception:
        return datetime.utcnow()


def _proportional_devig(prob_list: List[float]) -> List[float]:
    s = sum(prob_list)
    if s == 0:
        return prob_list
    return [p / s for p in prob_list]


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

# ---------------- Fetch Data ----------------
@st.cache_data(ttl=60*15)
def fetch_openligadb_fixtures(date_iso: str) -> List[Dict]:
    try:
        r = requests.get(f"{BASE_OPENLIGA}/getmatchdata/{date_iso}", timeout=15)
        data = r.json() if r.ok else []
        fixtures = []
        for m in data:
            try:
                utc_kick = m.get("MatchDateTimeUTC") or m.get("MatchDateTime")
                utc_kick = utc_kick.replace(" ","T")+"Z" if " " in utc_kick else utc_kick
                fixtures.append({
                    "leagueName": m.get("LeagueName","Unknown"),
                    "home": m["Team1"]["TeamName"],
                    "away": m["Team2"]["TeamName"],
                    "utc": utc_kick,
                    "status": "NS" if not m.get("MatchIsFinished") else "FT",
                })
            except:
                continue
        return fixtures
    except:
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
        if not r.ok:
            return {}
        data = r.json()
        match = None
        target_date = date_iso[:10]
        for ev in data:
            try:
                if not ev.get("commence_time", "").startswith(target_date):
                    continue
                ht = ev["home_team"]
                at = ev["away_team"]
                if home.lower() in ht.lower() and away.lower() in at.lower():
                    match = ev
                    break
                if home.lower() in at.lower() and away.lower() in ht.lower():
                    match = ev
                    break
            except:
                continue
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
        home = d.get("home")
        draw = d.get("draw")
        away = d.get("away")
        return home, draw, away
    except:
        return None, None, None

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
        snippets = [a.get("title","") for a in articles if a.get("title")]
        return snippets
    except:
        return []

def sentiment_score(text_list: List[str]) -> float:
    if not text_list:
        return 0.0
    scores = [sia.polarity_scores(t)['compound'] for t in text_list]
    return np.mean(scores)

# ---------------- Prediction ----------------
def predict_win_odds(h_odds: Optional[float], d_odds: Optional[float], a_odds: Optional[float], news_score: float=0.0) -> Tuple[float, float, float]:
    probs = _odds_to_implied([h_odds, d_odds, a_odds])
    probs = _proportional_devig(probs)
    probs = [min(max(p + 0.05 * news_score, 0.0), 1.0) for p in probs]
    s = sum(probs)
    probs = [p / s for p in probs]
    return tuple(probs)

def over_under_probs(xg_home: float, xg_away: float) -> Dict[str, float]:
    total_xg = xg_home + xg_away
    lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    res = {}
    for l in lines:
        res[f"Over {l}"] = poisson_cdf_over(l, total_xg)
        res[f"Under {l}"] = 1 - res[f"Over {l}"]
    return res

# ---------------- UI ----------------
tab = st.tabs(["Favourites", "Live", "Today", "Tomorrow"])

date_today = datetime.utcnow().strftime("%Y-%m-%d")
date_tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

def render_fixtures(fixtures: List[Dict], date_iso: str):
    if not fixtures:
        st.warning("No fixtures found.")
        return
    for f in fixtures:
        home, away = f["home"], f["away"]
        utc = f["utc"]
        status = f["status"]
        # fetch odds
        o = fetch_odds(home, away, date_iso)
        h_od, d_od, a_od = extract_match_odds(o)
        # fetch news
        news = fetch_news_snippets(home) + fetch_news_snippets(away)
        news_s = sentiment_score(news)
        # predict
        ph, pd, pa = predict_win_odds(h_od, d_od, a_od, news_s)
        # dummy xG for Poisson
        xg_home, xg_away = 1.3 + ph * 1.5, 1.1 + pa * 1.5
        ou = over_under_probs(xg_home, xg_away)
        best_bet = "🏆 Strong Pick" if max(ph, pa) > 0.6 else ""
        conf = make_star_confidence(max(ph, pa))
        # display
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.markdown(f'**{f["leagueName"]}** | {safe_parse_datetime(utc).strftime("%H:%M UTC")} | Status: {status}')
        st.markdown(f'**{home}** vs **{away}** {f"<span class=\'chip value\'>{best_bet}</span>" if best_bet else ""}', unsafe_allow_html=True)
        st.markdown(f'<div class="odds-box">Home: {ph:.2%} | Draw: {pd:.2%} | Away: {pa:.2%}</div>', unsafe_allow_html=True)
        st.markdown(f'Confidence: {conf} | News sentiment: {news_s:.2f}')
        ou_table = "<br>".join([f"{k}: {v:.2%}" for k, v in ou.items()])
        st.markdown(f'<div class="odds-box">{ou_table}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab[2]:
    fixtures = fetch_openligadb_fixtures(date_today)
    render_fixtures(fixtures, date_today)

with tab:
    fixtures = fetch_openligadb_fixtures(date_tomorrow)
    render_fixtures(fixtures, date_tomorrow)

# ---------------- Bottom Nav ----------------
st.markdown(
    """
    <div class="bottom-nav">
      <a href="#">🏠<strong>Home</strong></a>
      <a href="#">💱<strong>Trade</strong></a>
      <a href="#">📝<strong>Betslip</strong></a>
      <a href="#">🎯<strong>Sportsbook</strong></a>
      <a href="#">📦<strong>Orders</strong></a>
    </div>
    """,
    unsafe_allow_html=True,
)
