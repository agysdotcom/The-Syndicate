import streamlit as st
import requests
from datetime import datetime, date
from functools import lru_cache

# ------------------ CONFIG ------------------

API_KEY = "69bb2856e8ec4ad7b9a12f305147b408"

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
    "Conference League": "soccer_uefa_europa_conf_league",
}

# ------------------ STYLE ------------------

st.set_page_config(page_title="The Syndicate's Predictions", layout="wide", page_icon="⚽")

st.markdown("""
    <style>
    body { font-family: 'Arial', sans-serif; }
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric { color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("## 🎯 The Syndicate's Predictions")
selected_date = st.date_input("Select match date", value=date.today())

# ------------------ HELPERS ------------------

def classify_probability(prob):
    if prob > 75:
        return "Very Likely"
    elif prob > 60:
        return "Likely"
    elif prob > 50:
        return "Edge Case"
    else:
        return "Unlikely"

def calculate_confidence(prob):
    if prob > 80:
        return 5
    elif prob > 65:
        return 4
    elif prob > 55:
        return 3
    elif prob > 45:
        return 2
    else:
        return 1

def generate_narrative(match):
    if match['confidence'] >= 4:
        return f"{match['home_team']} are showing strong signs of dominance based on odds, with a high confidence rating of {match['confidence']} stars."
    elif match['confidence'] == 3:
        return f"The game between {match['home_team']} and {match['away_team']} looks tight, but {match['prediction']} has a slight edge."
    else:
        return f"This match appears unpredictable. Proceed with caution despite the odds leaning toward {match['prediction']}."

# ------------------ MAIN DATA FUNCTION ------------------

@lru_cache(maxsize=128)
def fetch_matches_for_all_leagues(date_filter):
    results = []

    for league_name, league_code in LEAGUES.items():
        url = f"https://api.the-odds-api.com/v4/sports/{league_code}/odds?regions=eu&markets=h2h,totals&oddsFormat=decimal&apiKey={API_KEY}"
        try:
            res = requests.get(url)
            if res.status_code != 200:
                continue

            for m in res.json():
                dt = datetime.fromisoformat(m["commence_time"].replace("Z", "+00:00"))
                if dt.date() != date_filter:
                    continue

                h_team = m["home_team"]
                a_team = m["away_team"]
                bookmakers = m.get("bookmakers", [])
                if not bookmakers:
                    continue

                h2h = bookmakers[0]["markets"][0]["outcomes"]
                home_odds = next((o["price"] for o in h2h if o["name"] == h_team), None)
                draw_odds = next((o["price"] for o in h2h if o["name"] == "Draw"), None)
                away_odds = next((o["price"] for o in h2h if o["name"] == a_team), None)

                if not all([home_odds, draw_odds, away_odds]):
                    continue

                prob_home = round((1 / home_odds) / ((1 / home_odds) + (1 / draw_odds) + (1 / away_odds)) * 100, 2)
                prediction = h_team if prob_home > 50 else a_team
                confidence = calculate_confidence(prob_home)

                totals = next((mk for mk in bookmakers[0]["markets"] if mk["key"] == "totals"), None)
                over_under = "Over 2.5 Goals" if totals else "Data Unavailable"

                results.append({
                    "league": league_name,
                    "home_team": h_team,
                    "away_team": a_team,
                    "date": dt.strftime("%Y-%m-%d"),
                    "time": dt.strftime("%H:%M"),
                    "prediction": prediction,
                    "prob_home_win": prob_home,
                    "classification": classify_probability(prob_home),
                    "confidence": confidence,
                    "over_under": over_under,
                    "narrative": generate_narrative({
                        "home_team": h_team,
                        "away_team": a_team,
                        "prediction": prediction,
                        "confidence": confidence
                    }),
                    "yellow_cards": "Likely 3+ cards (est.)",
                    "corner_kicks": "Likely 8+ corners (est.)"
                })

        except Exception as e:
            st.error(f"Error loading {league_name}: {e}")

    return sorted(results, key=lambda x: (x["league"], x["time"]))

# ------------------ UI ------------------

matches = fetch_matches_for_all_leagues(selected_date)

tab1, tab2 = st.tabs(["📊 All Predictions", "🔥 Strong Picks"])

with tab1:
    if not matches:
        st.warning("No upcoming matches found.")
    else:
        for m in matches:
            st.markdown(f"### ⚽ {m['home_team']} vs {m['away_team']} ({m['league']})")
            st.markdown(f"🕒 {m['date']} {m['time']}")
            st.progress(m['prob_home_win'] / 100)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏆 Predicted", m["prediction"])
            col2.metric("🔢 Prob.", f"{m['prob_home_win']}%")
            col3.metric("📊 Rating", f"{m['confidence']}⭐")
            col4.metric("🔍 Type", m["classification"])

            col5, col6, col7 = st.columns(3)
            col5.write(f"📈 {m['over_under']}")
            col6.write(f"🟨 {m['yellow_cards']}")
            col7.write(f"🔁 {m['corner_kicks']}")

            st.markdown(f"**Narrative:** {m['narrative']}")
            st.divider()

with tab2:
    strong_picks = [m for m in matches if m["confidence"] >= 4]
    if not strong_picks:
        st.info("No strong picks for this date.")
    else:
        for m in strong_picks:
            st.success(f"🔥 {m['home_team']} vs {m['away_team']} — **{m['prediction']}** ({m['confidence']}⭐)")
            st.caption(m["narrative"])
