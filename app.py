"""Streamlit UI. Run: streamlit run app.py"""
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

import data
import pipeline

st.set_page_config(page_title="Football Oracle", page_icon="⚽", layout="wide")
PRED = data.DATA / "predictions.pkl"
NAMES = {k: v[0] for k, v in data.LEAGUES.items()}
ALERT = re.compile(r"injur|ruled out|doubt|suspen|sack|ban|illness|hamstring|surgery|fitness|leaves|resign", re.I)
FORM_LABELS = {
    "elo": "Elo rating", "pts_5": "Points / game (last 5)", "pts_10": "Points / game (last 10)",
    "gf_5": "Goals scored / game (last 5)", "ga_5": "Goals conceded / game (last 5)",
    "sf_5": "Shots / game (last 5)", "stf_5": "Shots on target / game (last 5)",
    "sta_5": "Shots on target conceded (last 5)", "xg_5": "Expected goals / game (last 5)",
    "xga_5": "Expected goals conceded / game (last 5)", "cf_5": "Corners / game (last 5)",
    "yc_10": "Yellow cards / game (last 10)", "rc_10": "Red cards / game (last 10)",
    "v_pts_5": "Home/away form: points / game (last 5 at this venue type)",
    "v_gf_5": "Home/away form: goals scored", "v_ga_5": "Home/away form: goals conceded",
    "rest": "Days since last league match", "played": "League games played this season",
}
SQUAD_LABELS = {
    "coach_name": "Manager", "mgr_games": "Manager: league games in data", "mgr_ppg": "Manager: points / game",
    "mgr_team_games": "Manager: games at this club", "mgr_days": "Manager: days in the job",
    "lu_missing": "Regular starters missing (of 11)", "lu_rating": "Line-up form (player ratings, 0 = average)",
    "lu_value": "Line-up market value (typical player, €m)",
    "lu_def": "Formation: defenders", "lu_fwd": "Formation: forwards", "lu_out": "Regulars out",
}


@st.cache_data
def load(_mtime):
    stadiums = pd.read_csv(data.STADIUMS).set_index("team")["stadium"] if data.STADIUMS.exists() else pd.Series()
    out = data.DATA / "unavailable.pkl"
    return (pd.read_pickle(PRED), pd.read_pickle(data.DATA / "history.pkl"),
            json.loads((data.DATA / "metrics.json").read_text()), stadiums,
            pd.read_pickle(out) if out.exists() else pd.DataFrame(columns=["team", "name", "reason"]))


@st.cache_data(ttl=3600, show_spinner=False)
def news(team, league):
    url = f"https://news.google.com/rss/search?q={quote(f'"{team}" {league}')}&hl=en-GB&gl=GB&ceid=GB:en"
    try:
        root = ET.fromstring(requests.get(url, timeout=10).content)
    except (requests.RequestException, ET.ParseError):
        return []
    return [(i.findtext("title"), i.findtext("link"), i.findtext("pubDate", "")[:16]) for i in root.iter("item")][:8]


def fmt(v):
    if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def recent(hist, team, n=6):
    h = hist[(hist.HomeTeam == team) | (hist.AwayTeam == team)].tail(n).iloc[::-1]
    home = h.HomeTeam == team
    gf, ga = h.FTHG.where(home, h.FTAG), h.FTAG.where(home, h.FTHG)
    return pd.DataFrame({"Date": h.Date.dt.strftime("%d %b %y"), "": home.map({True: "H", False: "A"}),
                         "Opponent": h.AwayTeam.where(home, h.HomeTeam),
                         "Score": gf.astype(int).astype(str) + "-" + ga.astype(int).astype(str),
                         "Res": [("W" if f > a else "D" if f == a else "L") for f, a in zip(gf, ga)]})


st.sidebar.title("⚽ Football Oracle")
if st.sidebar.button("Refresh data & retrain", width="stretch") or not PRED.exists():
    with st.spinner("Downloading data, training and predicting (first run takes several minutes)…"):
        pipeline.run()
    st.cache_data.clear()
preds, hist, metrics, stadiums, unavailable = load(PRED.stat().st_mtime)
st.sidebar.caption(f"Model trained {metrics['trained_at']} on {metrics['trained_on']:,} matches")
af = metrics.get("api_football", {})
if not af.get("key_set"):
    st.sidebar.warning("API-Football key not set: line-ups, managers and injuries outside the Premier League are "
                       "missing. Get a free key at dashboard.api-football.com and set APIFOOTBALL_KEY.")
else:
    st.sidebar.caption(f"API-Football: {af['used_today']}/{af['daily_budget']} requests used today, "
                       f"{af['fixtures_with_lineups']:,} matches with line-ups cached"
                       + (f", seasons {af['allowed_seasons'][0]}–{af['allowed_seasons'][1]} on this plan" if af.get("allowed_seasons") else ""))
ss = metrics.get("sofascore")
if ss:
    st.sidebar.caption(f"Sofascore: {ss['matches_scraped']:,} of {ss['finished_matches']:,} matches scraped "
                       f"({ss['requests_this_run']} requests last refresh"
                       + (", stopped: Sofascore is refusing requests, try again later)" if ss.get("blocked") else ")"))
leagues = st.sidebar.multiselect("Leagues", list(NAMES), default=list(NAMES), format_func=NAMES.get)
days = st.sidebar.slider("Days ahead", 1, 180, 14)

up = preds[preds.Div.isin(leagues) & (preds.Date <= pd.Timestamp.today() + pd.Timedelta(days=days))]
up = up.sort_values(["Date", "Time", "Div"]).reset_index(drop=True)
probs = up[["pH", "pD", "pA"]].to_numpy()
mkt = up[["mkt_H", "mkt_D", "mkt_A"]].to_numpy()
pick = up.pick.to_numpy()

st.header("Upcoming matches")
table = pd.DataFrame({
    "Date": up.Date.dt.strftime("%a %d %b"), "Time": up.Time, "League": up.Div.map(NAMES),
    "Home": up.HomeTeam, "Away": up.AwayTeam,
    "Home %": up.pH * 100, "Draw %": up.pD * 100, "Away %": up.pA * 100,
    "Prediction": [[h, "Draw", a][p] for h, a, p in zip(up.HomeTeam, up.AwayTeam, pick)],
    "Score": up.score,
    "Market %": [mkt[i, p] * 100 for i, p in enumerate(pick)],
    "Edge pp": [(probs[i, p] - mkt[i, p]) * 100 for i, p in enumerate(pick)],
})
pct = st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)
event = st.dataframe(
    table, key="matches", hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    column_config={"Home %": pct, "Draw %": pct, "Away %": pct,
                   "Market %": st.column_config.NumberColumn(format="%.0f%%", help="Bookmaker implied probability of the predicted outcome"),
                   "Edge pp": st.column_config.NumberColumn(format="%+.1f", help="Model minus bookmaker probability, percentage points")})
st.caption(f"{len(up)} matches. Select a row for details. Matches without published odds use the no-odds model.")

if event.selection.rows:
    m = up.iloc[event.selection.rows[0]]
    st.divider()
    st.subheader(f"{m.HomeTeam} vs {m.AwayTeam}")
    where = stadiums.get(m.HomeTeam, "")
    st.caption(" · ".join(str(x) for x in [NAMES[m.Div], m.Date.strftime("%A %d %B %Y"), m.Time, where,
                                           f"Referee: {m.Referee}" if pd.notna(m.Referee) else ""] if x and pd.notna(x)))

    c = st.columns(5)
    for col, label, p, mk in zip(c, [f"{m.HomeTeam} win", "Draw", f"{m.AwayTeam} win"],
                                 [m.pH, m.pD, m.pA], [m.mkt_H, m.mkt_D, m.mkt_A]):
        col.metric(label, f"{p:.0%}", None if pd.isna(mk) else f"{(p - mk) * 100:+.1f} pp vs market", delta_color="off")
    c[3].metric("Most likely score", m.score)
    c[4].metric("Expected goals", f"{m.xg_h:.2f} – {m.xg_a:.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Team form & strength**")
        st.dataframe(pd.DataFrame({m.HomeTeam: [m[f"h_{k}"] for k in FORM_LABELS],
                                   m.AwayTeam: [m[f"a_{k}"] for k in FORM_LABELS]},
                                  index=list(FORM_LABELS.values())).round(2), width="stretch")
    with right:
        st.markdown("**Context**")
        ctx = {
            "Previous meetings (last 6)": f"{m.h2h_n:.0f}",
            f"{m.HomeTeam} win rate in those": f"{m.h2h_win:.0%}" if pd.notna(m.h2h_win) else "–",
            "Draw rate in those": f"{m.h2h_draw:.0%}" if pd.notna(m.h2h_draw) else "–",
            f"Avg goal difference for {m.HomeTeam}": f"{m.h2h_gd:+.2f}" if pd.notna(m.h2h_gd) else "–",
            "Referee matches in data": f"{m.ref_n:.0f}" if pd.notna(m.ref_n) else "unknown",
            "Referee home-win rate": f"{m.ref_hwin:.0%}" if pd.notna(m.ref_hwin) else "–",
            "Referee cards / game": f"{m.ref_cards:.1f}" if pd.notna(m.ref_cards) else "–",
            "Forecast max temperature": f"{m.temp:.0f} °C" if pd.notna(m.temp) else "not yet available",
            "Forecast precipitation": f"{m.rain:.1f} mm" if pd.notna(m.rain) else "–",
            "Forecast max wind": f"{m.wind:.0f} km/h" if pd.notna(m.wind) else "–",
            "Bookmaker odds H / D / A": f"{m.OddsH:.2f} / {m.OddsD:.2f} / {m.OddsA:.2f}" if pd.notna(m.OddsH) else "not published yet",
        }
        st.dataframe(pd.Series(ctx, name="").to_frame(), width="stretch")

    st.markdown("**Squad & manager**")
    squad = {s: {k: m.get(f"{s}_{k}") for k in SQUAD_LABELS} for s in "ha"}
    for s in squad:  # stored as mean log10 EUR
        squad[s]["lu_value"] = 10 ** squad[s]["lu_value"] / 1e6 if pd.notna(squad[s]["lu_value"]) else None
    st.dataframe(pd.DataFrame({m.HomeTeam: [fmt(squad["h"][k]) for k in SQUAD_LABELS],
                               m.AwayTeam: [fmt(squad["a"][k]) for k in SQUAD_LABELS]},
                              index=list(SQUAD_LABELS.values())), width="stretch")
    left, right = st.columns(2)
    for col, team in ((left, m.HomeTeam), (right, m.AwayTeam)):
        with col:
            out = unavailable[unavailable.team == team]
            st.markdown(f"**{team}: injured / suspended / doubtful** ({len(out)})")
            if len(out):
                st.dataframe(out[["name", "reason"]].rename(columns={"name": "Player", "reason": "Reason"}),
                             hide_index=True, width="stretch")
            else:
                st.caption("None reported (or no availability data for this league).")

    left, right = st.columns(2)
    for col, team in ((left, m.HomeTeam), (right, m.AwayTeam)):
        with col:
            st.markdown(f"**{team}: recent results**")
            st.dataframe(recent(hist, team), hide_index=True, width="stretch")

    st.markdown("**Head to head (last 8 league meetings)**")
    pair = {m.HomeTeam, m.AwayTeam}
    h2h = hist[hist.HomeTeam.isin(pair) & hist.AwayTeam.isin(pair)].tail(8).iloc[::-1]
    st.dataframe(pd.DataFrame({"Date": h2h.Date.dt.strftime("%d %b %Y"), "Home": h2h.HomeTeam,
                               "Score": h2h.FTHG.astype(int).astype(str) + "-" + h2h.FTAG.astype(int).astype(str),
                               "Away": h2h.AwayTeam}), hide_index=True, width="stretch")

    st.markdown("**Latest news** (⚠️ = injury / suspension / manager keywords; shown for context, not used by the model)")
    left, right = st.columns(2)
    for col, team in ((left, m.HomeTeam), (right, m.AwayTeam)):
        with col:
            st.markdown(f"*{team}*")
            items = news(team, NAMES[m.Div])
            for title, link, when in items:
                title = title.replace("[", "(").replace("]", ")")
                st.markdown(f"{'⚠️ ' if ALERT.search(title) else ''}[{title}]({link}) · *{when}*")
            if not items:
                st.caption("No news found.")

with st.expander("Model accuracy (backtest)"):
    rows = {k: v for k, v in metrics.items() if isinstance(v, dict) and "accuracy" in v}
    st.dataframe(pd.DataFrame(rows).T.rename(columns={"log_loss": "Log loss (lower is better)", "accuracy": "Accuracy"}),
                 width="stretch")
    st.caption(f"Trained on seasons before {metrics['test_from_season']}, tested on {metrics['test_matches']:,} matches "
               "from then on. 'bookmakers' = market average odds converted to probabilities.")
    if "coverage" in metrics:
        st.markdown("Share of matches since last season with player / manager data")
        st.dataframe(pd.DataFrame(metrics["coverage"]).T.map(lambda v: f"{v:.0%}"), width="stretch")
