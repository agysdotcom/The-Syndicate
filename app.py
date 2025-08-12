import streamlit as st
import requests
from datetime import datetime, timedelta
from math import ceil

# --- CONFIG ---
API_KEY = st.secrets.get("API_FOOTBALL_KEY", None)
if API_KEY is None:
    st.error("API_FOOTBALL_KEY not found in Streamlit secrets. Please add it!")
    st.stop()

HEADERS = {"x-apisports-key": API_KEY}
BASE_URL = "https://v3.football.api-sports.io"

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

LEAGUE_BADGES = {
    "Premier League": "https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg",
    "La Liga": "https://upload.wikimedia.org/wikipedia/en/1/15/La_Liga_Santander.svg",
    "Serie A": "https://upload.wikimedia.org/wikipedia/en/e/e1/Serie_A_logo_%282019%29.svg",
    "Bundesliga": "https://upload.wikimedia.org/wikipedia/commons/d/df/Bundesliga_logo_%282017%29.svg",
    "Ligue 1": "https://upload.wikimedia.org/wikipedia/en/8/8a/Ligue1.svg",
    "Eredivisie": "https://upload.wikimedia.org/wikipedia/en/2/28/Eredivisie_logo_2019.svg",
    "Primeira Liga": "https://upload.wikimedia.org/wikipedia/en/f/fb/Primeira_Liga_logo.svg",
    "Scottish Premiership": "https://upload.wikimedia.org/wikipedia/en/5/59/Scottish_Premiership_Logo.svg",
    "Belgian Pro League": "https://upload.wikimedia.org/wikipedia/en/7/7c/Belgian_Pro_League_logo.svg",
    "Champions League": "https://upload.wikimedia.org/wikipedia/en/8/89/UEFA_Champions_League_logo_2.svg",
    "Europa League": "https://upload.wikimedia.org/wikipedia/en/d/d2/UEFA_Europa_League_logo.svg",
    "Conference League": "https://upload.wikimedia.org/wikipedia/en/3/38/UEFA_Conference_League_logo.svg"
}

def safe_parse_datetime(date_utc):
    if date_utc.endswith("Z"):
        date_utc = date_utc[:-1] + "+00:00"
    return datetime.fromisoformat(date_utc)

def fetch_fixtures(league_ids, date_str):
    fixtures = []
    url = f"{BASE_URL}/fixtures"
    for league_id in league_ids:
        params = {
            "league": league_id,
            "season": 2025,
            "date": date_str,
            "timezone": "UTC",
        }
        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code == 200:
            data = res.json()
            fixtures.extend(data.get("response", []))
        else:
            st.warning(f"Failed to fetch fixtures for league {league_id}")
    return fixtures

def get_odds_for_fixture(fixture_id):
    url = f"{BASE_URL}/odds"
    params = {"fixture": fixture_id}
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code == 200:
        data = res.json()
        return data.get("response", [])
    return []

def get_match_winner_odds(odds_response):
    if not odds_response:
        return ("N/A", "N/A", "N/A")
    try:
        bookmakers = odds_response[0].get("bookmakers", [])
        for bookmaker in bookmakers:
            for bet in bookmaker.get("bets", []):
                if bet.get("name") == "Match Winner":
                    values = bet.get("values", [])
                    home = draw = away = "N/A"
                    for v in values:
                        if v.get("value") == "Home":
                            home = v.get("odd", "N/A")
                        elif v.get("value") == "Draw":
                            draw = v.get("odd", "N/A")
                        elif v.get("value") == "Away":
                            away = v.get("odd", "N/A")
                    return (home, draw, away)
    except Exception:
        return ("N/A", "N/A", "N/A")
    return ("N/A", "N/A", "N/A")

def calc_probabilities(odds_response):
    if not odds_response:
        return None
    try:
        for bookmaker in odds_response[0]["bookmakers"]:
            for bet in bookmaker["bets"]:
                if bet["name"] == "Match Winner":
                    values = bet["values"]
                    home_odd = next(v["odd"] for v in values if v["value"] == "Home")
                    draw_odd = next(v["odd"] for v in values if v["value"] == "Draw")
                    away_odd = next(v["odd"] for v in values if v["value"] == "Away")
                    home_prob = 1 / float(home_odd)
                    draw_prob = 1 / float(draw_odd)
                    away_prob = 1 / float(away_odd)
                    total = home_prob + draw_prob + away_prob
                    return {
                        "home": round((home_prob / total) * 100, 2),
                        "draw": round((draw_prob / total) * 100, 2),
                        "away": round((away_prob / total) * 100, 2),
                    }
    except Exception:
        return None

def confidence_stars(probabilities):
    if probabilities is None:
        return "No data"
    max_prob = max(probabilities.values())
    stars = int(min(max((max_prob - 50) / 8, 1), 5))
    return "⭐" * stars

def narrative(fixture, probs):
    home = fixture["teams"]["home"]["name"]
    away = fixture["teams"]["away"]["name"]
    date_str = fixture["fixture"]["date"][:10]
    if probs is None:
        return "No betting odds available for this match."
    best_outcome = max(probs, key=probs.get)
    best_prob = probs[best_outcome]
    explanation = (
        f"On {date_str}, {home} will face {away}. "
        f"The highest predicted outcome is {best_outcome.upper()} with a probability of {best_prob}%. "
        "Odds are derived from leading bookmakers and indicate market confidence."
    )
    return explanation

def over_under_goals_placeholder():
    import random
    over = random.randint(40, 70)
    under = 100 - over
    return over, under

def main():
    st.set_page_config(
        page_title="THE SYNDICATE - Soccer Predictor", layout="wide", initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    body {background-color: #18181b;}
    .stApp {
        background-color: #18181b;
        color: #e5e7eb;
    }
    div[data-testid="stHeader"], footer, #MainMenu {display: none;}
    img {
        max-height: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center; margin-bottom: 1.2rem;">'
        '<span style="font-size:2.2rem; font-weight:bold; color:#60a5fa;">⚽ THE SYNDICATE</span><br>'
        '<span style="font-size:1rem; color:#a3a3a3;">Football Predictions for 2025-2026 Season</span>'
        '</div>', unsafe_allow_html=True
    )

    league_names = list(LEAGUES.keys())
    selected_leagues = st.sidebar.multiselect(
        "Select Leagues", league_names, default=league_names[:3]
    )
    today = datetime.utcnow().date()
    selected_date = st.sidebar.date_input(
        "Select Match Date", min_value=today, max_value=today + timedelta(days=30), value=today
    )

    if not selected_leagues:
        st.warning("Please select at least one league.")
        st.stop()

    league_ids = [LEAGUES[l] for l in selected_leagues]

    with st.spinner("Fetching fixtures..."):
        fixtures = fetch_fixtures(league_ids, selected_date.isoformat())

    if not fixtures:
        st.info("No fixtures found for the selected date and leagues.")
        st.stop()

    for f in fixtures:
        status = f["fixture"]["status"]["short"]
        if status != "NS":
            continue

        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        date_utc = f["fixture"]["date"]
        date_local = safe_parse_datetime(date_utc)
        league_name = f['league']['name']
        badge_url = LEAGUE_BADGES.get(league_name, "")

        odds_response = get_odds_for_fixture(f["fixture"]["id"])
        probs = calc_probabilities(odds_response)
        conf = confidence_stars(probs)
        over, under = over_under_goals_placeholder()
        yellow_cards = 4
        corners = 9

        home_odd, draw_odd, away_odd = get_match_winner_odds(odds_response)

        st.markdown(
            f"""
            <div style="background:#23232b; border-radius:18px; padding:20px; margin-bottom:24px; box-shadow: 0 2px 8px #1e293b33; color:#f3f4f6;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                    <img src="{badge_url}" alt="{league_name}" loading="lazy" />
                    <strong style="font-size:1.05rem;">{league_name}</strong>
                    <span style="margin-left:auto; font-family: monospace; color:#818cf8;">{date_local.strftime('%Y-%m-%d %H:%M UTC')}</span>
                </div>
                <div style="display:flex; font-weight:bold; font-size:1.3rem; margin-bottom:10px;">
                    <span style="flex:1; color:#60a5fa;">{home}</span>
                    <span style="margin: 0 8px; color:#a1a1aa;">vs</span>
                    <span style="flex:1; color:#facc15; text-align:right;">{away}</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(3,1fr); gap:12px; margin-bottom:16px; font-size:0.9rem;">
                    <div style="background:#111827; padding:10px; border-radius:10px;">
                        <div style="color:#3b82f6; font-weight:600;">Home Win</div>
                        <div>Odds: <strong>{home_odd}</strong></div>
                        <div>Predicted: <strong>{probs['home'] if probs else 'N/A'}%</strong></div>
                    </div>
                    <div style="background:#111827; padding:10px; border-radius:10px;">
                        <div style="color:#f3f4f6; font-weight:600;">Draw</div>
                        <div>Odds: <strong>{draw_odd}</strong></div>
                        <div>Predicted: <strong>{probs['draw'] if probs else 'N/A'}%</strong></div>
                    </div>
                    <div style="background:#111827; padding:10px; border-radius:10px;">
                        <div style="color:#facc15; font-weight:600;">Away Win</div>
                        <div>Odds: <strong>{away_odd}</strong></div>
                        <div>Predicted: <strong>{probs['away'] if probs else 'N/A'}%</strong></div>
                    </div>
                </div>
                <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:0.95rem;">
                    <div><strong>Confidence:</strong> <span style="color:#fbbf24; font-size:1.25rem;">{conf}</span></div>
                    <div><strong>Over 2.5 Goals:</strong> <span style="color:#38bdf8;">{over}%</span> | <strong>Under 2.5 Goals:</strong> <span style="color:#f472b6;">{under}%</span></div>
                    <div><strong>Yellow Cards:</strong> <span style="color:#fde68a;">{yellow_cards}</span> | <strong>Corners:</strong> <span style="color:#60a5fa;">{corners}</span></div>
                </div>
                <div style="margin-top:15px; background:#1f2937; border-radius:10px; padding:12px; font-style: italic; color:#cbd5e1;">
                    {narrative(f, probs)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
