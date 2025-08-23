# app.py - Streamlit Neon Sportsbook
import os, math, requests, streamlit as st
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from transformers import pipeline

# ---------- Setup ----------
st.set_page_config(page_title="⚡ THE SYNDICATE", layout="wide", page_icon="⚽")
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ---------- API Keys ----------
API_FOOTBALL_KEY = st.secrets.get("API_FOOTBALL_KEY")
THEODDSAPI_KEY = st.secrets.get("THEODDSAPI_KEY")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY")
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY} if API_FOOTBALL_KEY else {}

# ---------- Leagues ----------
LEAGUES = {
    "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78,
    "Ligue 1": 61, "Eredivisie": 88, "Primeira Liga": 94, "Scottish Premiership": 179,
    "Belgian Pro League": 144, "Champions League": 2, "Europa League": 3, "Conference League": 848
}

# ---------- Utils ----------
def safe_float(x):
    try: return float(x)
    except: return None

def odds_to_implied(odds):
    return [1/o if o and o>0 else 0 for o in odds]

def devig(probs):
    s=sum(probs)
    return [p/s for p in probs] if s>0 else probs

def poisson_probs(lam, max_goals=5):
    return [math.exp(-lam)*(lam**k)/math.factorial(k) for k in range(max_goals+1)]

def poisson_over(lam, line):
    return 1 - sum(poisson_probs(lam)[:int(line+1)])

def confidence_stars(score):
    stars = int(np.clip(1 + round(4*score),1,5))
    return "⭐"*stars

# ---------- Fetch Fixtures ----------
@st.cache_data(ttl=600)
def fetch_fixtures(league_id, date_iso):
    url=f"https://v3.football.api-sports.io/fixtures?league={league_id}&season=2025&date={date_iso}"
    r=requests.get(url, headers=HEADERS_FOOTBALL, timeout=10)
    if r.ok: return r.json().get("response",[])
    return []

# ---------- Fetch Odds ----------
@st.cache_data(ttl=300)
def fetch_odds(home, away):
    url="https://api.the-odds-api.com/v4/sports/soccer/odds"
    params={"apiKey":THEODDSAPI_KEY,"regions":"uk,eu,us","markets":"h2h,totals","oddsFormat":"decimal"}
    r=requests.get(url,params=params,timeout=10)
    if not r.ok: return {}
    for ev in r.json():
        if home.lower() in ev.get("home_team","").lower() and away.lower() in ev.get("away_team","").lower():
            return ev
    return {}

def extract_h2h(odds_obj):
    try:
        for bk in odds_obj.get("bookmakers",[]):
            for mk in bk.get("markets",[]):
                if mk.get("key")=="h2h":
                    outcomes=mk.get("outcomes",[])
                    home=draw=away=None
                    for o in outcomes:
                        n=o.get("name","").lower()
                        p=safe_float(o.get("price"))
                        if "draw" in n: draw=p
                        elif "home" in n: home=p
                        else: away=p
                    return home,draw,away
    except: return None,None,None
    return None,None,None

# ---------- Fetch News ----------
def fetch_news(team):
    if not NEWSAPI_KEY: return []
    url="https://newsapi.org/v2/everything"
    params={"q":f"{team} injury OR press conference","from":(datetime.utcnow()-timedelta(days=2)).isoformat(),"sortBy":"publishedAt","apiKey":NEWSAPI_KEY,"language":"en"}
    r=requests.get(url, params=params, timeout=5)
    if r.ok: return r.json().get("articles",[])
    return []

def summarize_sentiment(news_articles):
    pos=neg=0
    for a in news_articles:
        text=a.get("title","") + ". " + a.get("description","")
        summary = summarizer(text,max_length=50,min_length=25,do_sample=False)[0]['summary_text'] if summarizer else text
        score = sia.polarity_scores(summary)["compound"] if sia else 0
        if score>0.1: pos+=1
        elif score<-0.1: neg+=1
    total=max(pos+neg,1)
    return {"positive":pos/total,"negative":neg/total,"neutral":1-(pos+neg)/total}

# ---------- Neon Styles ----------
st.markdown("""
<style>
body {background-color:#0d0d0d; color:#39FF14;}
h1 {text-align:center; color:#39FF14;}
.expanderHeader {color:#39FF14; font-weight:bold;}
.win-bar {height:20px; border-radius:5px; margin-bottom:3px;}
.win-green {background:#00FF7F;}
.win-amber {background:#FFD700;}
.win-red {background:#FF3131;}
.strong-pick {color:#39FF14; font-weight:bold; text-shadow:0 0 8px #39FF14;}
.live-badge {color:#FF3131; font-weight:bold; text-shadow:0 0 8px #FF3131;}
</style>
""",unsafe_allow_html=True)

# ---------- UI ----------
st.markdown("<h1>⚡ THE SYNDICATE - Football Predictor</h1>",unsafe_allow_html=True)
selected_date=st.date_input("Select date",datetime.utcnow())
date_iso=selected_date.isoformat()

# Bottom Bar
st.markdown("""
<style>
.footer {position:fixed; bottom:0; width:100%; background-color:#0d0d0d; color:#39FF14; text-align:center; padding:10px; font-size:18px;}
.footer a {color:#39FF14; margin:20px; text-decoration:none; font-weight:bold;}
</style>
<div class="footer">
<a href='#'>Home</a> | <a href='#'>Trade</a> | <a href='#'>Betslip</a> | <a href='#'>Sportsbook</a> | <a href='#'>My Orders</a>
</div>
""", unsafe_allow_html=True)

tabs=st.tabs(["Today","Tomorrow"])
for tab,label in zip(tabs,[0,1]):
    with tab:
        date_use=(selected_date+timedelta(days=label)).isoformat()
        for lname,lid in LEAGUES.items():
            fixtures=fetch_fixtures(lid,date_use)
            for f in fixtures:
                home=f["teams"]["home"]["name"]
                away=f["teams"]["away"]["name"]
                kickoff=f['fixture']['date'][:16]
                live = f.get("fixture",{}).get("status",{}).get("short","NS")=="LIVE"
                live_badge = "<span class='live-badge'>LIVE</span>" if live else ""
                
                st.markdown(f"### {lname} | {home} vs {away} | {kickoff} {live_badge}",unsafe_allow_html=True)
                
                odds_obj=fetch_odds(home,away)
                h_odd,d_odd,a_odd=extract_h2h(odds_obj)
                probs=devig(odds_to_implied([h_odd,d_odd,a_odd]))
                
                # News & sentiment
                news=fetch_news(home)+fetch_news(away)
                sentiment=summarize_sentiment(news)
                adj_factor=sentiment["positive"]-sentiment["negative"]
                adj_probs=[min(max(p+adj_factor*0.1,0),1) for p in probs]
                adj_probs=devig(adj_probs)
                
                # Poisson Over/Under
                lam_home,lam_away=1.2,1.0 # placeholder
                ou_probs={line:poisson_over(lam_home+lam_away,line) for line in [0.5,1.5,2.5,3.5,4.5]}
                
                # Strong Pick
                strong_pick = "<span class='strong-pick'>💚 Strong Pick!</span>" if adj_probs[0]>0.55 or adj_probs[2]>0.55 else ""
                
                # Expandable card
                with st.expander(f"{home} vs {away} {strong_pick}", expanded=False):
                    # Win % bars
                    colors = ["win-green" if p>0.55 else "win-amber" if 0.45<=p<=0.55 else "win-red" for p in adj_probs]
                    for t,p,c in zip([home,"Draw",away],adj_probs,colors):
                        st.markdown(f"{t}: <div class='win-bar {c}' style='width:{int(p*100)}%'></div>",unsafe_allow_html=True)
                    # Odds
                    st.markdown(f"**Odds H/D/A:** {h_odd}/{d_odd}/{a_odd}")
                    # Poisson
                    st.markdown("**Poisson Over Probabilities:**")
                    st.table(pd.DataFrame.from_dict(ou_probs,orient='index',columns=["Over Probability"]).style.format("{:.1%}"))
                    # Confidence
                    st.markdown(f"**Confidence:** {confidence_stars(sum(adj_probs)/3)}")
                    # News highlights
                    st.markdown("**News Highlights:**")
                    for n in news[:3]: st.markdown(f"- {n['title']}")
                    # Best Bet
                    st.markdown(f"**Best Bet:** {'H' if adj_probs[0]>0.55 else 'A' if adj_probs[2]>0.55 else 'D'}")
                st.markdown("---")
