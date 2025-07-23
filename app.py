import streamlit as st
import requests
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="The Syndicate's Predictions", layout="wide")

# Load API key
API_KEY = os.getenv("THE_ODDS_API_KEY")

# Title
st.title("The Syndicate's Predictions")
selected_date = st.date_input("Select a date", datetime.today())
st.markdown("#### Upcoming Fixtures with Predictions")

# League IDs (examples)
leagues = {
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

def implied_prob(decimal_odds):
    return round(100 / decimal_odds, 2) if decimal_odds else 0

def classify_probability(prob):
    if prob > 75:
        return "Highly Likely"
    elif prob > 60:
        return "Likely"
    elif prob > 45:
        return "Possible"
    else:
        return "Unlikely"

def get_confidence_rating(prob):
    if prob > 75:
        return "⭐⭐⭐⭐⭐"
    elif prob > 60:
        return "⭐⭐⭐⭐"
    elif prob > 45:
        return "⭐⭐⭐"
    elif prob > 30:
        return "⭐⭐"
    else:
        return "⭐"

def fetch_data(league_id):
    url = f"https://api.the-odds-api.com/v4/sports/{league_id}/odds/?regions=eu&markets=h2h,totals,spreads&oddsFormat=decimal&apiKey={API_KEY}"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    return res.json()

# Loop through leagues
all_predictions = []
for league_name, league_id in leagues.items():
    matches = fetch_data(league_id)
    for match in matches:
        commence = datetime.fromisoformat(match["commence_time"].replace("Z", "+00:00"))
        if commence.date() >= selected_date:
            match_block = {
                "league": league_name,
                "time": commence.strftime('%Y-%m-%d %H:%M'),
                "teams": match["teams"],
                "bookmakers": match.get("bookmakers", [])
            }
            all_predictions.append(match_block)

# Display matches
if all_predictions:
    for m in all_predictions:
        st.markdown(f"### {m['teams'][0]} vs {m['teams'][1]}")
        st.caption(f"{m['league']} | {m['time']}")
        if m["bookmakers"]:
            for bookmaker in m["bookmakers"]:
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        st.subheader("🔮 Win Probabilities")
                        for outcome in market["outcomes"]:
                            prob = implied_prob(outcome['price'])
                            st.write(f"**{outcome['name']}**: {prob}% chance — {classify_probability(prob)} | Confidence: {get_confidence_rating(prob)}")
                    elif market["key"] == "totals":
                        st.subheader("📊 Over/Under Goals")
                        for o in market["outcomes"]:
                            st.write(f"{o['name']} {o['point']} Goals @ {o['price']} odds")
        st.markdown("---")
else:
    st.info("No matches found for the selected date or leagues.")
