import os
import math
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import numpy as np
import requests
import streamlit as st
import nltk
import pytz
from nltk.sentiment import SentimentIntensityAnalyzer

# ------------------- NLP Setup -------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="THE SYNDICATE - AI Soccer Predictor", layout="wide")

# ------------------- API Keys -------------------
API_FOOTBALL_KEY = "a6917f6db6a731e8b6cfa9f9f365a5ed"
THEODDSAPI_KEY = "69bb2856e8ec4ad7b9a12f305147b408"
NEWSAPI_KEY = "c7d0efc525bf48199ab229f8f70fbc01"

BASE_FOOTBALL = "https://v3.football.api-sports.io"
BASE_ODDS = "https://api.the-odds-api.com/v4"

LEAGUES = {
    "All": 0, "Premier League": 39, "La Liga": 140, "Serie A": 135,
    "Bundesliga": 78, "Ligue 1": 61, "Eredivisie": 88, "Primeira Liga": 94,
    "Scottish Premiership": 179, "Belgian Pro League": 144,
    "Champions League": 2, "Europa League": 3, "Conference League": 848
}

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}
CAPE_TOWN_TZ = pytz.timezone("Africa/Johannesburg")

# ------------------- Utility Functions -------------------
def safe_parse_datetime_utc_to_capetown(date_utc: str) -> datetime:
    try:
        dt_utc = datetime.fromisoformat(date_utc.replace("Z", "")).replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(CAPE_TOWN_TZ)
    except Exception:
        return datetime.now(tz=CAPE_TOWN_TZ)

def sentiment_score(text_list: List[str]) -> float:
    if not text_list:
        return 0.0
    return np.mean([sia.polarity_scores(t)['compound'] for t in text_list])

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
    return [p / s for p in probs] if s > 0 else [1 / len(probs)] * len(probs)

def poisson_pmf(k, lam):
    return np.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf_over(total: float, lam_total: float) -> float:
    floor_needed = int(math.floor(total + 0.5))
    c = sum(poisson_pmf(k, lam_total) for k in range(floor_needed))
    return 1 - c

def make_star_confidence(value: float) -> str:
    stars = int(np.clip(1 + round(4 * value), 1, 5))
    return "⭐" * stars

# ------------------- Fetch Data -------------------
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
    # Use your API to fetch real match stats if available
    # Example placeholders
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
        market = sites[0].get("markets", [])
        prices = market.get("outcomes", [])
        d = {p['name'].lower(): p['price'] for p in prices if 'name' in p and 'price' in p}
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

# ------------------- Prediction Engine -------------------
def predict_win_odds(h_odds: Optional[float], d_odds: Optional[float], a_odds: Optional[float], news_score: float=0) -> Tuple[float,float,float]:
    probs = _odds_to_implied([h_odds,d_odds,a_odds])
    probs = _proportional_devig(probs)
    probs = [min(max(p + 0.05*news_score,0),1) for p in probs]
    s = sum(probs)
    if s==0:
        return 1/3,1/3,1/3
    return tuple(p/s for p in probs)

def over_under_probs(xg_home: float, xg_away: float) -> Dict[str,float]:
    total = xg_home + xg_away
    res = {}
    for l in [0.5,1.5,2.5,3.5,4.5]:
        res[f"Over {l}"] = poisson_cdf_over(l, total)
        res[f"Under {l}"] = 1 - res[f"Over {l}"]
    return res

def prepare_fixtures_with_stats(league_id:int,date_iso:str)->List[Dict]:
    fixtures = fetch_fixtures(league_id,date_iso)
    enriched=[]
    for f in fixtures:
        o = fetch_odds(f["home"],f["away"],date_iso)
        h_od,d_od,a_od = extract_match_odds(o)
        news = fetch_news_snippets(f["home"]) + fetch_news_snippets(f["away"])
        news_s = sentiment_score(news)
        ph,pd,pa = predict_win_odds(h_od,d_od,a_od,news_s)
        xg_home,xg_away = 1.3 + ph*1.5, 1.1 + pa*1.5
        ou = over_under_probs(xg_home,xg_away)
        stats = fetch_match_stats(f["fixture_id"])
        best_bet = "🏆 Strong Pick" if max(ph,pa)>0.6 else ""
        confidence = make_star_confidence(max(ph,pa))
        enriched.append({
            **f,
            "home_prob":ph,"draw_prob":pd,"away_prob":pa,
            "news_sentiment":news_s,
            "over_under":ou,
            "corners":stats.get("corners",{"home":0,"away":0}),
            "yellow_cards":stats.get("yellow_cards",{"home":0,"away":0}),
            "best_bet":best_bet,
            "confidence":confidence
        })
    return enriched

def generate_fixture_card(game:Dict):
    st.markdown(f"### {game['home']} vs {game['away']} ({safe_parse_datetime_utc_to_capetown(game['utc']).strftime('%Y-%m-%d %H:%M')})")
    st.markdown(f"**Recommendation:** {game['best_bet']}")
    st.markdown(f"**Confidence:** {game['confidence']}")
    st.text(f"Win probs: H:{game['home_prob']:.2f}, D:{game['draw_prob']:.2f}, A:{game['away_prob']:.2f}")
    st.text(f"Over/Under 2.5: {game['over_under'].get('Over 2.5',0):.2f}")
    c,y = game['corners'],game['yellow_cards']
    st.text(f"Corners: {c['home']} - {c['away']}, Yellow Cards: {y['home']} - {y['away']}")
    st.button("Add to Betslip", key=f"{game['fixture_id']}_add")

# ------------------- Auto Betslip Engine -------------------
def build_auto_betslips(fixtures:List[Dict], min_total_prob=0.7, max_legs=4) -> List[Dict]:
    bets=[]
    # Build candidate pool
    pool=[]
    for f in fixtures:
        if f["home_prob"]>0.4:
            pool.append({"match":f"{f['home']} vs {f['away']}","pick":f"Home Win ({f['home']})","prob":f["home_prob"]})
        if f["away_prob"]>0.4:
            pool.append({"match":f"{f['home']} vs {f['away']}","pick":f"Away Win ({f['away']})","prob":f["away_prob"]})
        if f["over_under"].get("Over 2.5",0)>0.4:
            pool.append({"match":f"{f['home']} vs {f['away']}","pick":"Over 2.5 Goals","prob":f["over_under"]["Over 2.5"]})
        if sum(f["corners"].values())>=6:
            pool.append({"match":f"{f['home']} vs {f['away']}","pick":"High Corners","prob":0.5})
        if sum(f["yellow_cards"].values())>=3:
            pool.append({"match":f"{f['home']} vs {f['away']}","pick":"High Yellow Cards","prob":0.5})
    # Generate combinations
    for n in range(1,max_legs+1):
        for combo in combinations(pool,n):
            total_prob=np.prod([c["prob"] for c in combo])
            if total_prob>=min_total_prob:
                bets.append({"legs":combo,"total_prob":total_prob})
    # Ensure at least 2 bets
    if len(bets)<2:
        for n in range(1,max_legs+1):
            for combo in combinations(pool,n):
                total_prob=np.prod([c["prob"] for c in combo])
                bets.append({"legs":combo,"total_prob":total_prob})
                if len(bets)>=2:
                    break
            if len(bets)>=2:
                break
    bets=sorted(bets,key=lambda x:-x["total_prob"])
    return bets[:5]  # top 5 bets

# ------------------- App Layout -------------------
league_selected = st.sidebar.selectbox("Select League", list(LEAGUES.keys()),0)
date_today = datetime.utcnow().strftime("%Y-%m-%d")

# Prepare fixtures
fixtures=[]
if league_selected=="All":
    for lid in LEAGUES.values():
        if lid:
            fixtures.extend(prepare_fixtures_with_stats(lid,date_today))
else:
    fixtures=prepare_fixtures_with_stats(LEAGUES[league_selected],date_today)

# Tabs
tab1,tab2 = st.tabs(["Fixtures","Auto Betslips"])

with tab1:
    league_names = sorted(set(f["leagueName"] for f in fixtures))
    if league_names:
        league_tabs = st.tabs(league_names)
        for idx,lname in enumerate(league_names):
            with league_tabs[idx]:
                league_fixtures=[f for f in fixtures if f["leagueName"]==lname]
                if not league_fixtures:
                    st.info(f"No matches available for {lname}")
                for game in league_fixtures:
                    generate_fixture_card(game)
    else:
        st.info("No fixtures to display.")

with tab2:
    for lname in sorted(set(f["leagueName"] for f in fixtures)):
        st.subheader(f"Auto Betslips - {lname}")
        league_fixtures=[f for f in fixtures if f["leagueName"]==lname]
        bets=build_auto_betslips(league_fixtures)
        for b in bets:
            st.markdown(f"**Combined Prob:** {b['total_prob']:.2f}")
            for leg in b["legs"]:
                st.markdown(f"- {leg['match']} | Bet: {leg['pick']} | Prob: {leg['prob']:.2f}")
            st.button("Add to Betslip", key=f"auto_{lname}_{bets.index(b)}")
