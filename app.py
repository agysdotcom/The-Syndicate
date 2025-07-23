import streamlit as st
import requests
from datetime import datetime, date
from functools import lru_cache

API_KEY = st.secrets["THE_ODDS_API_KEY"]

LEAGUES = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Eredivisie": "soccer_netherlands_eredivisie",
    "Primeira Liga": "soccer_portugal_primeira_liga",
    "Scottish Premiership": "soccer_scotland_premiership",
    "Belgian Pro League": "soccer_belgium_first_div",
    "Champions League": "soccer_uefa_champs_league",
    "Europa League": "soccer_uefa_europa_league",
    "Conference League": "soccer_uefa_conference_league"
}

st.set_page_config(page_title="The Syndicate’s Predictions", layout="wide")
st.title("🔮 The Syndicate’s Predictions")

selected_date = st.sidebar.date_input("Pick a date", value=date.today())
st.sidebar.markdown("---")

def convert_odds(o): return round(100/o, 1) if o else 0
def classify(p):
    return "Very Likely" if p > 75 else "Likely" if p > 60 else "Edge" if p > 45 else "Unlikely"
def stars(p):
    return "⭐⭐⭐⭐⭐" if p > 75 else "⭐⭐⭐⭐" if p > 60 else "⭐⭐⭐" if p > 45 else "⭐⭐" if p > 30 else "⭐"

@lru_cache()
def load_matches():
    results = []
    for league_name, sport in LEAGUES.items():
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
            params={"apiKey": API_KEY, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal", "dateFormat": "iso"}
        )
        if r.status_code != 200:
            continue
        for ev in r.json():
            dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if dt.date() < selected_date:
                continue
            outcomes = next(m for m in ev["bookmakers"][0]["markets"] if m["key"]=="h2h")["outcomes"]
            home = outcomes[0]["name"]; away = outcomes[1]["name"]
            home_od = outcomes[0]["price"]; away_od = outcomes[1]["price"]
            home_p = convert_odds(home_od); away_p = convert_odds(away_od)
            pred = home if home_p > away_p else away
            conf = int((max(home_p, away_p)/100)*5)
            over = next((m for m in ev["bookmakers"][0]["markets"] if m["key"]=="totals"), None)
            ou = "Over 2.5" if over else "N/A"
            results.append({
                "league": league_name,
                "time": dt.strftime("%Y-%m-%d %H:%M"),
                "home": home,
                "away": away,
                "home_p": home_p,
                "away_p": away_p,
                "pred": pred,
                "class": classify(max(home_p, away_p)),
                "stars": stars(max(home_p, away_p)),
                "conf": conf,
                "ou": ou
            })
    return sorted(results, key=lambda x: x["time"])

matches = load_matches()

tab1, tab2 = st.tabs(["📊 All Predictions", "🔥 Strong Picks"])

with tab1:
    if not matches:
        st.warning("No upcoming matches found from selected date.")
    for m in matches:
        st.write(f"**{m['time']} – {m['league']}**")
        st.write(f"{m['home']} vs {m['away']}")
        st.write(f"Probabilities: {m['home_p']}% vs {m['away_p']}%")
        st.write(f"Prediction: {m['pred']} — {m['class']} {m['stars']}")
        st.write(f"Over/Under: {m['ou']}")
        st.divider()

with tab2:
    strong = [m for m in matches if m["conf"] >= 4]
    if not strong:
        st.info("No strong picks yet.")
    for m in strong:
        st.write(f"🔥 {m['time']} – {m['league']} — {m['home']} vs {m['away']}")
        st.write(f"Pick: **{m['pred']}** with {m['stars']} confidence")
        st.write("---")
