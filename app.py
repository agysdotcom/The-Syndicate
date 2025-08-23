# app.py
import os
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- OPTIONAL NLP IMPORTS (robust fallbacks) ----------
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except:  # noqa: E722
        nltk.download("vader_lexicon")
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False
    SentimentIntensityAnalyzer = None

try:
    from transformers import pipeline
    _SUMMARIZER = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    _HAS_SUM = True
except Exception:
    _HAS_SUM = False
    _SUMMARIZER = None

# ---------- CONFIG ----------
st.set_page_config(
    page_title="THE SYNDICATE - Soccer Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_FOOTBALL_KEY = st.secrets.get("API_FOOTBALL_KEY", None)
THEODDSAPI_KEY = st.secrets.get("THEODDSAPI_KEY", None)
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", None)  # Preferred
BING_SEARCH_KEY = st.secrets.get("BING_SEARCH_KEY", None)  # optional alternative

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY} if API_FOOTBALL_KEY else {}
BASE_FOOTBALL = "https://v3.football.api-sports.io"

# OpenLigaDB (fixtures baseline)
BASE_OPENLIGA = "https://api.openligadb.de"  # Public; no key

# TheOddsAPI
BASE_ODDS = "https://api.the-odds-api.com/v4"

# Supported leagues (API-Football IDs) – extend as you wish
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
    "Conference League": "https://upload.wikimedia.org/wikipedia/en/3/38/UEFA_Conference_League_logo.svg",
}

# ---------- STYLE ----------
st.markdown(
    """
    <style>
    body {background:#0b0d12;}
    .stApp {background:#0b0d12; color:#e5e7eb;}
    .league-badge img {max-height: 22px;}
    .card {
        background:#121420; border:1px solid #1e2233; border-radius:18px;
        padding:18px; box-shadow:0 2px 12px rgba(0,0,0,0.35); margin-bottom:18px;
    }
    .odds-box {
        background:#0e111a; border:1px solid #20263b; border-radius:12px; padding:10px;
    }
    .chip { padding:3px 10px; border-radius:999px; font-size:0.75rem; }
    .chip.live { background:#172a1f; color:#22c55e; border:1px solid #134e4a; }
    .chip.value { background:#19242d; color:#22d3ee; border:1px solid #164e63; }
    .bar { height:10px; background:#0f172a; border-radius:999px; overflow:hidden; }
    .bar > div { height:100%; background:linear-gradient(90deg,#22d3ee,#60a5fa); }
    .neon-green { color:#22c55e; }
    .neon-red { color:#f87171; }
    .neon-amber { color:#fbbf24; }
    .neon-blue { color:#60a5fa; }
    .bottom-nav {
        position:fixed; left:0; right:0; bottom:0; background:#0c0f16; border-top:1px solid #1f2940;
        display:flex; gap:14px; justify-content:space-around; padding:10px 10px; z-index:999;
    }
    .bottom-nav a { color:#cbd5e1; text-decoration:none; font-size:0.9rem; }
    .bottom-nav a strong { display:block; font-size:0.75rem; color:#93c5fd; }
    .badge-strong { color:#22c55e; border:1px solid #14532d; background:#062a12; padding:2px 8px; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="text-align:center; margin-bottom: 0.8rem;">'
    '<span style="font-size:2.2rem; font-weight:800; color:#60a5fa;">⚡ THE SYNDICATE</span><br>'
    '<span style="font-size:1rem; color:#94a3b8;">Football Predictions • 2025–2026</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ---------- UTIL ----------
def safe_parse_datetime(date_utc: str) -> datetime:
    try:
        if date_utc.endswith("Z"):
            date_utc = date_utc[:-1] + "+00:00"
        return datetime.fromisoformat(date_utc)
    except Exception:
        # fallback
        return datetime.utcnow()

def _proportional_devig(prob_list: List[float]) -> List[float]:
    """Proportional de-vig on probabilities (not odds)."""
    s = sum(prob_list)
    if s == 0:
        return prob_list
    return [p / s for p in prob_list]

def _odds_to_implied(odds: List[float]) -> List[float]:
    probs = []
    for o in odds:
        try:
            o = float(o)
            if o <= 1e-9:
                probs.append(0.0)
            else:
                probs.append(1.0 / o)
        except Exception:
            probs.append(0.0)
    return probs

def _american_or_decimal_to_float(o) -> Optional[float]:
    """Accepts string/float decimal; also handles American odds like +150/-120."""
    try:
        s = str(o).strip()
        if s.startswith("+") or s.startswith("-"):
            a = int(s)
            if a > 0:
                return 1 + 100.0 / a
            else:
                return 1 + abs(a) / 100.0
        return float(s)
    except Exception:
        return None

def poisson_pmf(k, lam):
    return np.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf_over(total_line_half: float, lam_total: float) -> float:
    """
    P(Total goals > L) for L in {0.5,1.5,2.5,3.5,4.5}
    If L = 2.5, means sum_{k>=3} P(k; lam_total).
    """
    floor_needed = int(math.floor(total_line_half + 0.5))  # 2.5 -> 3; 1.5 -> 2
    c = 0.0
    # sum from 0..(floor_needed-1) then 1 - that
    for k in range(floor_needed):
        c += poisson_pmf(k, lam_total)
    return float(1.0 - c)

def make_star_confidence(value: float) -> str:
    """
    Value 0..1 => 1..5 stars roughly.
    """
    score = max(0.0, min(1.0, value))
    stars = int(np.clip(1 + round(4 * score), 1, 5))
    return "⭐" * stars

# ---------- NEWS ----------
@st.cache_data(show_spinner=False, ttl=60*15)
def fetch_news_snippets(team_name: str) -> List[Dict]:
    """
    Use NewsAPI (preferred) or Bing News Search if provided.
    Returns list of dicts: {title, url, source, publishedAt, description}
    """
    snippets = []
    q = f'{team_name} injury OR "press conference" OR suspension OR coach OR manager'
    from_time = (datetime.utcnow() - timedelta(hours=48)).isoformat(timespec="seconds") + "Z"

    if NEWSAPI_KEY:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": q,
                "from": from_time,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 10,
                "apiKey": NEWSAPI_KEY,
            }
            r = requests.get(url, params=params, timeout=15)
            if r.ok:
                data = r.json()
                for a in data.get("articles", []):
                    snippets.append({
                        "title": a.get("title"),
                        "url": a.get("url"),
                        "source": a.get("source", {}).get("name"),
                        "publishedAt": a.get("publishedAt"),
                        "description": a.get("description"),
                        "content": a.get("content"),
                    })
        except Exception:
            pass

    elif BING_SEARCH_KEY:
        try:
            url = "https://api.bing.microsoft.com/v7.0/news/search"
            params = {"q": q, "mkt": "en-US", "freshness": "Day", "count": 10, "sortBy": "Date"}
            headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY}
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.ok:
                data = r.json()
                for v in data.get("value", []):
                    snippets.append({
                        "title": v.get("name"),
                        "url": v.get("url"),
                        "source": v.get("provider", [{}])[0].get("name"),
                        "publishedAt": v.get("datePublished"),
                        "description": v.get("description"),
                        "content": None,
                    })
        except Exception:
            pass
    return snippets

def summarize_and_score_news(snips: List[Dict]) -> Tuple[str, float]:
    """
    Returns (short_summary, sentiment_score in [-1,1]).
    Sentiment via VADER if available; fallback neutral.
    """
    if not snips:
        return ("No notable recent news in last 48h.", 0.0)

    text_chunks = []
    for s in snips:
        t = s.get("title") or ""
        d = s.get("description") or ""
        c = s.get("content") or ""
        text_chunks.append(" ".join([t, d, c]).strip())
    big_text = "\n".join(text_chunks)[:8000]  # keep reasonable

    # Summarize
    summary = None
    if _HAS_SUM:
        try:
            res = _SUMMARIZER(big_text[:3000], max_length=120, min_length=40, do_sample=False)
            summary = res[0]["summary_text"]
        except Exception:
            pass
    if not summary:
        # simple heuristic summary
        summary = text_chunks[0][:240] + ("..." if len(text_chunks[0]) > 240 else "")

    # Sentiment
    sentiment = 0.0
    if _HAS_VADER:
        try:
            sia = SentimentIntensityAnalyzer()
            ss = sia.polarity_scores(big_text)
            sentiment = float(ss["compound"])
        except Exception:
            sentiment = 0.0

    return (summary, sentiment)

def news_adjustment(sentiment: float) -> float:
    """
    Map sentiment [-1,1] -> adjustment factor for xG totals.
    Small, controlled nudge: -7%..+7%
    """
    return 1.0 + 0.07 * float(sentiment)

# ---------- FIXTURES (OpenLigaDB baseline) ----------
@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_openligadb_fixtures(date_iso: str, league_hint: Optional[str] = None) -> List[Dict]:
    """
    OpenLigaDB does not map 1:1 with all leagues above; we'll still pull a general list
    for the date and later match with API-Football if possible.
    """
    try:
        # OpenLigaDB provides "getmatchdata" by league & season; for a general baseline,
        # we use "getmatchdata/<date>" which returns a broad set.
        url = f"{BASE_OPENLIGA}/getmatchdata/{date_iso}"
        r = requests.get(url, timeout=15)
        if not r.ok:
            return []
        data = r.json()
        # Normalize to our structure
        fixtures = []
        for m in data:
            try:
                utc_kick = m.get("MatchDateTimeUTC") or m.get("MatchDateTime")
                utc_kick = utc_kick.replace(" ", "T") + "Z" if " " in utc_kick else utc_kick
                fixtures.append({
                    "leagueName": (m.get("LeagueName") or m.get("LeagueShortcut") or "Unknown"),
                    "home": m["Team1"]["TeamName"],
                    "away": m["Team2"]["TeamName"],
                    "utc": utc_kick,
                    "status": "NS" if not m.get("MatchIsFinished") else "FT",
                })
            except Exception:
                continue
        return fixtures
    except Exception:
        return []

# ---------- API-FOOTBALL STATS ----------
@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_team_form_stats(league_id: int, team_id: int) -> Dict:
    """
    Example pulls: last 5, home/away split, goals for/against, xG if available.
    """
    if not API_FOOTBALL_KEY:
        return {}
    try:
        url = f"{BASE_FOOTBALL}/teams/statistics"
        params = {"league": league_id, "season": 2025, "team": team_id}
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=20)
        if not r.ok:
            return {}
        data = r.json().get("response", {})
        return data
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=60*30)
def search_team_id(team_name: str) -> Optional[int]:
    if not API_FOOTBALL_KEY:
        return None
    try:
        url = f"{BASE_FOOTBALL}/teams"
        params = {"search": team_name}
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=15)
        if not r.ok:
            return None
        res = r.json().get("response", [])
        if not res:
            return None
        return res[0]["team"]["id"]
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_odds_theoddsapi(home: str, away: str, date_iso: str) -> Dict:
    """
    Pull head market odds via TheOddsAPI. We match by team names heuristically.
    """
    if not THEODDSAPI_KEY:
        return {}
    try:
        # Fetch soccer odds market (h2h) across regions
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
                comp = ev.get("commence_time", "")
                if not comp.startswith(target_date):
                    continue
                ht = ev["home_team"]
                at = ev["away_team"]
                if home.lower() in ht.lower() and away.lower() in at.lower():
                    match = ev
                    break
                # also swap (some feeds flip home/away)
                if home.lower() in at.lower() and away.lower() in ht.lower():
                    match = ev
                    break
            except Exception:
                continue
        return match or {}
    except Exception:
        return {}

def extract_match_odds(odds_obj: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns decimal odds (home, draw, away) from bookmakers (pick first available).
    """
    if not odds_obj:
        return (None, None, None)
    try:
        for bk in odds_obj.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") == "h2h":
                    outcomes = mk.get("outcomes", [])
                    home = draw = away = None
                    for o in outcomes:
                        label = o.get("name", "").lower()
                        if "draw" in label:
                            draw = _american_or_decimal_to_float(o.get("price"))
                        elif odds_obj.get("home_team", "").lower() in label:
                            home = _american_or_decimal_to_float(o.get("price"))
                        elif odds_obj.get("away_team", "").lower() in label:
                            away = _american_or_decimal_to_float(o.get("price"))
                    # If labels are generic, outcomes usually in order [home, draw, away]
                    if home is None or away is None:
                        if len(outcomes) >= 3:
                            home = home or _american_or_decimal_to_float(outcomes[0].get("price"))
                            draw = draw or _american_or_decimal_to_float(outcomes[1].get("price"))
                            away = away or _american_or_decimal_to_float(outcomes[2].get("price"))
                    return (home, draw, away)
    except Exception:
        pass
    return (None, None, None)

def implied_probs_devig_from_odds(home: Optional[float], draw: Optional[float], away: Optional[float]) -> Optional[Dict[str, float]]:
    if any(o is None for o in (home, draw, away)):
        return None
    probs = _odds_to_implied([home, draw, away])
    devig = _proportional_devig(probs)
    return {"home": round(devig[0] * 100, 2), "draw": round(devig[1] * 100, 2), "away": round(devig[2] * 100, 2)}

# ---------- XG & POISSON ----------
def estimate_team_xg(team_stats: Dict, home: bool) -> float:
    """
    Very lightweight estimator using API-Football stats object:
    - goals for per match (home/away split if available)
    - shots, xG if provided by plan
    Fallback to 1.3 (home) / 1.1 (away).
    """
    try:
        goals = team_stats["goals"]
        if home:
            per = goals["for"]["average"]["home"]
        else:
            per = goals["for"]["average"]["away"]
        if per is None:
            raise ValueError
        return float(per)
    except Exception:
        return 1.3 if home else 1.1

def poisson_over_under_table(lam_home: float, lam_away: float, news_factor: float) -> Dict[str, float]:
    total_lambda = (lam_home + lam_away) * news_factor
    lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    res = {}
    for L in lines:
        over_p = poisson_cdf_over(L, total_lambda)
        res[f"Over {L}"] = round(over_p * 100, 1)
        res[f"Under {L}"] = round((1 - over_p) * 100, 1)
    return res

def pick_best_value(model_probs: Dict[str, float], market_odds: Tuple[Optional[float], Optional[float], Optional[float]]) -> Optional[str]:
    home, draw, away = market_odds
    if any(o is None for o in (home, draw, away)) or model_probs is None:
        return None
    # Fair odds from model:
    fair_home = 1.0 / (model_probs["home"]/100.0)
    fair_draw = 1.0 / (model_probs["draw"]/100.0)
    fair_away = 1.0 / (model_probs["away"]/100.0)
    edges = {
        "Home": (home - fair_home) / fair_home,
        "Draw": (draw - fair_draw) / fair_draw,
        "Away": (away - fair_away) / fair_away,
    }
    best = max(edges.items(), key=lambda kv: kv[1])
    return f"{best[0]} ({best[1]*100:.1f}% edge)" if best[1] > 0.03 else None  # 3%+ edge

def narrative_block(home: str, away: str, date_str: str, probs: Optional[Dict[str, float]], ou_25: Tuple[float, float], news_summary: str, conf_stars: str) -> str:
    if not probs:
        return f"No reliable betting odds available. {news_summary}"
    best_key = max(probs, key=probs.get)
    pct = probs[best_key]
    over25 = ou_25[0]
    conf = conf_stars
    return (
        f"{home} vs {away} on {date_str}. "
        f"{home}–{away} market leans {best_key.upper()} at {pct:.0f}%. "
        f"Poisson projects ~{over25:.0f}% chance for Over 2.5 goals. "
        f"News: {news_summary} "
        f"Confidence: {conf}."
    )

# ---------- UI HELPERS ----------
def league_header(league: str, kickoff: datetime) -> str:
    badge = LEAGUE_BADGES.get(league, "")
    return (
        f'<div class="league-badge" style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">'
        f'<img src="{badge}" alt="{league}" />'
        f'<strong style="font-size:1.05rem;">{league}</strong>'
        f'<span style="margin-left:auto; font-family:monospace; color:#818cf8;">{kickoff.strftime("%Y-%m-%d %H:%M UTC")}</span>'
        f"</div>"
    )

def odds_row(home_odd, draw_odd, away_odd, probs):
    h = f"{home_odd:.2f}" if home_odd else "—"
    d = f"{draw_odd:.2f}" if draw_odd else "—"
    a = f"{away_odd:.2f}" if away_odd else "—"
    ph = f"{probs['home']:.1f}%" if probs else "N/A"
    pd = f"{probs['draw']:.1f}%" if probs else "N/A"
    pa = f"{probs['away']:.1f}%" if probs else "N/A"
    return f"""
    <div style="display:grid; grid-template-columns: repeat(3,1fr); gap:12px; margin-bottom:14px; font-size:0.9rem;">
        <div class="odds-box">
            <div style="color:#3b82f6; font-weight:600;">Home</div>
            <div>Odds: <strong>{h}</strong></div>
            <div>Model: <strong>{ph}</strong></div>
        </div>
        <div class="odds-box">
            <div style="color:#e5e7eb; font-weight:600;">Draw</div>
            <div>Odds: <strong>{d}</strong></div>
            <div>Model: <strong>{pd}</strong></div>
        </div>
        <div class="odds-box">
            <div style="color:#facc15; font-weight:600;">Away</div>
            <div>Odds: <strong>{a}</strong></div>
            <div>Model: <strong>{pa}</strong></div>
        </div>
    </div>
    """

# ---------- APP ----------
def sidebar_controls():
    league_names = list(LEAGUES.keys())
    st.sidebar.markdown("### Filters")
    selected_leagues = st.sidebar.multiselect("Leagues", league_names, default=league_names[:5])

    today = datetime.utcnow().date()
    tabs = st.sidebar.radio("When", ["Live", "Today", "Tomorrow"], index=1)
    if tabs == "Live":
        date_val = today
    elif tabs == "Today":
        date_val = today
    else:
        date_val = today + timedelta(days=1)

    favs = st.sidebar.text_input("Favourites (comma-separated team names)", value="")
    st.sidebar.caption("Example: Liverpool, Barcelona, Bayern")

    return selected_leagues, date_val, tabs, [x.strip() for x in favs.split(",") if x.strip()]

def main():
    selected_leagues, selected_date, when_tab, favourites = sidebar_controls()

    if not selected_leagues:
        st.warning("Please select at least one league.")
        st.stop()

    # Top tabs (content groups)
    t_fav, t_live, t_today, t_tom = st.tabs(["⭐ Favourites", "🟢 Live", "📅 Today", "🗓 Tomorrow"])

    # Fetch fixtures baseline (OpenLigaDB) + we’ll enrich
    with st.spinner("Fetching fixtures (baseline: OpenLigaDB)…"):
        fixtures_base = fetch_openligadb_fixtures(selected_date.isoformat())

    # If no baseline, continue empty
    if not fixtures_base:
        st.info("No fixtures found on OpenLigaDB for the selected date. You can still switch leagues/dates.")
        fixtures_base = []

    # Build a lightweight list filtered by selected leagues if name appears
    # (OpenLiga naming may vary – keep loose filter)
    filtered = []
    for f in fixtures_base:
        league_name = f.get("leagueName", "Unknown")
        if any(l.lower() in league_name.lower() for l in selected_leagues) or not selected_leagues:
            filtered.append(f)

    # If empty after filter, keep some to display
    fixtures = filtered if filtered else fixtures_base

    # Prepare a tidy DataFrame for quick filtering & display
    df = pd.DataFrame(fixtures)
    if not df.empty:
        df["kickoff"] = df["utc"].apply(safe_parse_datetime)
    else:
        df = pd.DataFrame(columns=["leagueName", "home", "away", "utc", "status", "kickoff"])

    # Helper to render a single match card
    def render_match_row(row, container):
        league_name = row["leagueName"]
        home = row["home"]
        away = row["away"]
        kickoff = row["kickoff"]
        utc_iso = row["utc"]

        # Odds via TheOddsAPI
        odds_obj = fetch_odds_theoddsapi(home, away, utc_iso)
        home_odd, draw_odd, away_odd = extract_match_odds(odds_obj)
        probs = implied_probs_devig_from_odds(home_odd, draw_odd, away_odd)

        # Stats/xG via API-Football
        home_id = search_team_id(home) if API_FOOTBALL_KEY else None
        away_id = search_team_id(away) if API_FOOTBALL_KEY else None
        lam_home = lam_away = None
        if API_FOOTBALL_KEY and home_id and away_id:
            # find plausible league id match (best effort – choose from user selection)
            league_id = next((LEAGUES[l] for l in LEAGUES if l.lower() in league_name.lower()), None)
            if league_id:
                home_stats = fetch_team_form_stats(league_id, home_id)
                away_stats = fetch_team_form_stats(league_id, away_id)
                lam_home = estimate_team_xg(home_stats, home=True)
                lam_away = estimate_team_xg(away_stats, home=False)

        if lam_home is None:  # fallbacks
            lam_home = 1.35
        if lam_away is None:
            lam_away = 1.15

        # Unstructured news → summary & sentiment
        news_snips_home = fetch_news_snippets(home)
        news_snips_away = fetch_news_snippets(away)
        sum_home, sent_home = summarize_and_score_news(news_snips_home)
        sum_away, sent_away = summarize_and_score_news(news_snips_away)
        combined_sent = np.tanh((sent_home + sent_away) / 2.0)  # mild squashing
        adj = news_adjustment(combined_sent)

        # Poisson multi-line O/U
        ou_table = poisson_over_under_table(lam_home, lam_away, adj)
        over25 = (ou_table["Over 2.5"], ou_table["Under 2.5"])

        # Confidence from max prob
        if probs:
            maxp = max(probs.values()) / 100.0
            conf_stars = make_star_confidence(maxp)
        else:
            conf_stars = "No data"

        best_value = pick_best_value(probs, (home_odd, draw_odd, away_odd))

        with container:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(league_header(league_name, kickoff), unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <span style="flex:1; font-size:1.25rem; font-weight:800; color:#60a5fa;">{home}</span>
                    <span style="color:#94a3b8;">vs</span>
                    <span style="flex:1; font-size:1.25rem; font-weight:800; text-align:right; color:#facc15;">{away}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(odds_row(home_odd, draw_odd, away_odd, probs), unsafe_allow_html=True)

            # Value & meta row
            meta_cols = st.columns([1, 1, 1, 1])
            with meta_cols[0]:
                st.markdown(f"**Confidence:** {conf_stars}")
            with meta_cols[1]:
                st.markdown(f"**Over 2.5**: {over25[0]}%  |  **Under 2.5**: {over25[1]}%")
            with meta_cols[2]:
                if best_value:
                    st.markdown(f"**Best Bet:** :green[{best_value}]")
                else:
                    st.markdown("**Best Bet:** —")
            with meta_cols[3]:
                chip = '<span class="chip value">Strong Pick</span>' if best_value else '<span class="chip">—</span>'
                st.markdown(chip, unsafe_allow_html=True)

            with st.expander("Show details (xG, O/U, Cards, Corners, H2H, News)"):
                # xG table
                xg_cols = st.columns(3)
                with xg_cols[0]:
                    st.metric(label="xG (Home est.)", value=f"{lam_home:.2f}")
                with xg_cols[1]:
                    st.metric(label="xG (Away est.)", value=f"{lam_away:.2f}")
                with xg_cols[2]:
                    st.metric(label="News Factor", value=f"{adj:.3f}")

                # O/U table 0.5→4.5
                ou_df = pd.DataFrame(
                    {
                        "Line": ["0.5", "1.5", "2.5", "3.5", "4.5"],
                        "Over %": [
                            ou_table["Over 0.5"],
                            ou_table["Over 1.5"],
                            ou_table["Over 2.5"],
                            ou_table["Over 3.5"],
                            ou_table["Over 4.5"],
                        ],
                        "Under %": [
                            ou_table["Under 0.5"],
                            ou_table["Under 1.5"],
                            ou_table["Under 2.5"],
                            ou_table["Under 3.5"],
                            ou_table["Under 4.5"],
                        ],
                    }
                )
                st.markdown("**Over/Under Probabilities (Poisson)**")
                st.dataframe(ou_df, use_container_width=True, hide_index=True)

                # Simple yellow cards & corners projections
                # (Referee/league factors could be drawn from API-Football plan if available)
                cards_est = np.clip(3.0 + 0.8 * (lam_home + lam_away), 3.0, 7.0)
                corners_est = np.clip(7.0 + 0.9 * (lam_home + lam_away), 6.0, 14.0)
                kc = st.columns(2)
                with kc[0]:
                    st.metric("Projected Yellow Cards", f"{cards_est:.1f}")
                with kc[1]:
                    st.metric("Projected Corners", f"{corners_est:.1f}")

                # News highlights
                st.markdown("**News Highlights (last 48h)**")
                st.write(f"**{home}:** {sum_home}")
                st.write(f"**{away}:** {sum_away}")

            # Narrative
            story = narrative_block(
                home, away, kickoff.strftime("%Y-%m-%d"),
                probs,
                over25,
                f"{sum_home} / {sum_away}",
                conf_stars,
            )
            st.markdown(
                f"""<div style="margin-top:12px; background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px; font-style:italic; color:#cbd5e1;">
                {story}
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Render per tab
    def render_table(container, subset_df):
        if subset_df.empty:
            with container:
                st.info("No fixtures to display.")
                return
        # Sort by kickoff
        subset_df = subset_df.sort_values("kickoff")
        for _, r in subset_df.iterrows():
            render_match_row(r, container)

    # Favourites
    with t_fav:
        if favourites:
            fav_mask = df["home"].fillna("").str.contains("|".join([f for f in favourites]), case=False) | \
                       df["away"].fillna("").str.contains("|".join([f for f in favourites]), case=False)
            render_table(t_fav, df[fav_mask])
        else:
            st.caption("Add favourites in the sidebar to see them here.")
            render_table(t_fav, df.head(6))

    # Live (simple heuristic by time proximity)
    with t_live:
        now = datetime.utcnow()
        live_mask = (df["kickoff"] <= now + timedelta(minutes=5)) & (df["kickoff"] >= now - timedelta(hours=2))
        render_table(t_live, df[live_mask])

    # Today
    with t_today:
        day_mask = df["kickoff"].dt.date == selected_date
        render_table(t_today, df[day_mask])

    # Tomorrow
    with t_tom:
        tm = selected_date + timedelta(days=1)
        day2_mask = df["kickoff"].dt.date == tm
        render_table(t_tom, df[day2_mask])

    # Bottom nav
    st.markdown(
        """
        <div class="bottom-nav">
            <a href="#"><span>🏠 Home</span><strong>Browse</strong></a>
            <a href="#"><span>💹 Trade</span><strong>Markets</strong></a>
            <a href="#"><span>🧾 Betslip</span><strong>Slip</strong></a>
            <a href="#"><span>🎰 Sportsbook</span><strong>Odds</strong></a>
            <a href="#"><span>🧑 My</span><strong>Orders</strong></a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------- ENTRY ----------
if __name__ == "__main__":
    main()
