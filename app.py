# app.py
import os
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st
import nltk
import pytz

# NLP Setup
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
    "All": None,
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

CAPE_TOWN_TZ = pytz.timezone("Africa/Johannesburg")  # GMT+2

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

# ---------------- Helpers ----------------
def safe_parse_datetime_utc_to_cape(date_utc: str) -> datetime:
    try:
        if date_utc.endswith("Z"): date_utc = date_utc[:-1]
        dt_utc = datetime.fromisoformat(date_utc).replace(tzinfo=timezone.utc)
        dt_cape = dt_utc.astimezone(CAPE_TOWN_TZ)
        return dt_cape
    except:
        return datetime.now(tz=CAPE_TOWN_TZ)

def _proportional_devig(prob_list: List[float]) -> List[float]:
    s = sum(prob_list)
    return prob_list if s==0 else [p/s for p in prob_list]

def _odds_to_implied(odds: List[float]) -> List[float]:
    probs=[]
    for o in odds:
        try: probs.append(1.0/float(o) if o>1e-9 else 0.0)
        except: probs.append(0.0)
    return probs

def poisson_pmf(k, lam): return np.exp(-lam)*(lam**k)/math.factorial(k)

def poisson_cdf_over(line, lam): return 1.0 - sum(poisson_pmf(k, lam) for k in range(int(line+0.5)))

def make_star_confidence(value: float) -> str:
    stars = int(np.clip(1+round(4*max(0,min(1,value))),1,5))
    return "⭐"*stars

def sentiment_score(texts: List[str]) -> float:
    if not texts: return 0.0
    return np.mean([sia.polarity_scores(t)['compound'] for t in texts])

# ---------------- Fetching ----------------
@st.cache_data(ttl=60*15)
def fetch_fixtures(league_id: Optional[int], date_iso:str) -> List[Dict]:
    fixtures=[]
    leagues = [league_id] if league_id else [v for k,v in LEAGUES.items() if v]
    for lid in leagues:
        try:
            url=f"{BASE_FOOTBALL}/fixtures"
            params={"league":lid,"season":int(date_iso[:4]),"date":date_iso}
            r = requests.get(url, headers={"x-apisports-key":API_FOOTBALL_KEY}, params=params, timeout=15)
            if r.ok:
                for f in r.json().get("response", []):
                    fixtures.append({
                        "league":f["league"]["name"],
                        "home":f["teams"]["home"]["name"],
                        "away":f["teams"]["away"]["name"],
                        "utc":f["fixture"]["date"],
                        "status":f["fixture"]["status"]["short"]
                    })
        except: continue
    return fixtures

# ---------------- Odds ----------------
@st.cache_data(ttl=60*30)
def fetch_odds(home:str,away:str,date_iso:str)->Dict:
    try:
        r=requests.get(f"{BASE_ODDS}/sports/soccer/odds", params={
            "apiKey": THEODDSAPI_KEY, "regions":"uk,eu,us", "markets":"h2h,totals", "oddsFormat":"decimal", "dateFormat":"iso"
        },timeout=20)
        if not r.ok: return {}
        target = date_iso[:10]
        for ev in r.json():
            if not ev.get("commence_time","").startswith(target): continue
            ht=ev["home_team"]; at=ev["away_team"]
            if home.lower() in ht.lower() and away.lower() in at.lower(): return ev
            if home.lower() in at.lower() and away.lower() in ht.lower(): return ev
        return {}
    except: return {}

def extract_match_odds(o:Dict)->Tuple[Optional[float],Optional[float],Optional[float]]:
    try:
        sites=o.get("bookmakers",[])
        if not sites: return None,None,None
        prices=sites[0].get("markets",[]).get("outcomes",[])
        d={p['name'].lower():p['price'] for p in prices if 'name' in p and 'price' in p}
        return d.get("home"),d.get("draw"),d.get("away")
    except: return None,None,None

# ---------------- Prediction ----------------
def predict_win_odds(h_od, d_od, a_od, news_score=0.0):
    probs=_odds_to_implied([h_od,d_od,a_od])
    probs=_proportional_devig(probs)
    probs=[min(max(p+0.05*news_score,0.0),1.0) for p in probs]
    s=sum(probs)
    return tuple([p/s for p in probs]) if s>0 else (1/3,1/3,1/3)

def over_under_probs(xg_home,xg_away):
    total=xg_home+xg_away
    lines=[0.5,1.5,2.5,3.5,4.5]
    return {f"Over {l}":poisson_cdf_over(l,total) for l in lines}

# ---------------- Prepare fixtures ----------------
def prepare_fixtures(fixtures:List[Dict])->List[Dict]:
    result=[]
    for f in fixtures:
        if f['status']=="FT": continue
        o=fetch_odds(f['home'],f['away'],f['utc'])
        h_od,d_od,a_od=extract_match_odds(o)
        news_score=sentiment_score([]) # can extend with fetch_news_snippets
        ph,pd,pa=predict_win_odds(h_od,d_od,a_od,news_score)
        xg_home, xg_away=1.3+ph*1.5, 1.1+pa*1.5
        ou=over_under_probs(xg_home,xg_away)
        recommendation = "Over 2.5 Goals" if max(ou.values())>0.6 else "Check Other Markets"
        result.append({**f,"home_prob":ph,"draw_prob":pd,"away_prob":pa,"over_under":ou,"recommendation":recommendation})
    return result

# ---------------- Render Fixtures ----------------
def render_fixture(f):
    dt = safe_parse_datetime_utc_to_cape(f['utc']).strftime("%H:%M GMT+2")
    with st.expander(f"{f['home']} vs {f['away']} | {f['league']} | {dt} | Status: {f['status']}"):
        st.markdown(f"**Recommendation:** {f['recommendation']}")
        tabs_inner = st.tabs(["Over/Under","Corners","Yellow Cards"])
        with tabs_inner[0]:
            df=pd.DataFrame(list(f['over_under'].items()),columns=["Market","Prob"])
            st.table(df.style.format({"Prob":"{:.0%}"}))
        with tabs_inner[1]: st.info("Corners stats here (placeholder)")
        with tabs_inner[2]: st.info("Yellow cards stats here (placeholder)")

# ---------------- UI ----------------
league_select=st.selectbox("Select League:",list(LEAGUES.keys()))
date_today=(datetime.utcnow()+timedelta(hours=2)).strftime("%Y-%m-%d")
fixtures = prepare_fixtures(fetch_fixtures(LEAGUES[league_select],date_today))

# Tabs per league
if league_select=="All":
    league_groups={}
    for f in fixtures:
        league_groups.setdefault(f['league'],[]).append(f)
    for league_name, games in league_groups.items():
        with st.expander(f"League: {league_name}"):
            for g in games: render_fixture(g)
else:
    for g in fixtures: render_fixture(g)

# ---------------- Betslip ----------------
st.markdown("## Betslip (Recommended Over Goals)")
betslip=[{"match":f"{f['home']} vs {f['away']}","pick":max(f['over_under'], key=f['over_under'].get),"prob":max(f['over_under'].values())} for f in fixtures if "Over" in f['recommendation']]
if betslip: st.table(pd.DataFrame(betslip))

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
