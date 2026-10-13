"""Data sources (all free, no API keys):
- football-data.co.uk: historical results, match stats, referees, bookmaker odds, near-term fixtures with odds
- fixturedownload.com: full current-season schedule incl. stadium names
- OpenStreetMap Nominatim: stadium coordinates (cached, editable in data/stadiums.csv)
- open-meteo.com: historical + forecast weather at each stadium
"""
import difflib
import json
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
CACHE = DATA / "cache"
STADIUMS = DATA / "stadiums.csv"

# football-data code -> (display name, fixturedownload slug, country code for geocoding)
LEAGUES = {
    "E0": ("Premier League", "epl", "gb"),
    "SP1": ("La Liga", "la-liga", "es"),
    "I1": ("Serie A", "serie-a", "it"),
    "D1": ("Bundesliga", "bundesliga", "de"),
    "F1": ("Ligue 1", "ligue-1", "fr"),
    "N1": ("Eredivisie", "eredivisie", "nl"),
    "T1": ("Süper Lig", "super-lig", "tr"),
}
FIRST_SEASON = 2015  # 2015/16

KEEP = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Referee",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]

# fixturedownload name -> football-data name, only where fuzzy matching gets it wrong
ALIASES = {
    "Manchester United": "Man United", "Manchester City": "Man City", "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest", "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Atlético de Madrid": "Ath Madrid", "Atlético Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "Real Betis": "Betis", "RCD Espanyol de Barcelona": "Espanol", "Celta Vigo": "Celta", "RC Celta": "Celta",
    "Rayo Vallecano": "Vallecano", "Real Sociedad": "Sociedad", "FC Bayern München": "Bayern Munich",
    "Borussia Mönchengladbach": "M'gladbach", "Bayer 04 Leverkusen": "Leverkusen", "1. FSV Mainz 05": "Mainz",
    "Eintracht Frankfurt": "Ein Frankfurt", "1. FC Köln": "FC Koln", "1. FC Heidenheim 1846": "Heidenheim",
    "FC St. Pauli": "St Pauli", "Internazionale": "Inter", "AC Milan": "Milan", "Paris Saint-Germain": "Paris SG",
    "PSV": "PSV Eindhoven", "PSV Eindhoven": "PSV Eindhoven", "Fortuna Sittard": "For Sittard",
    "Go Ahead Eagles": "Go Ahead Eagles", "Sparta Rotterdam": "Sparta Rotterdam", "NEC": "Nijmegen",
    "RC Deportivo": "La Coruna", "R. Racing Club": "Santander", "Olympique Lyonnais": "Lyon",
    "Excelsior Rotterdam": "Excelsior", "AZ": "AZ Alkmaar", "N.E.C. Nijmegen": "Nijmegen",
}


def current_season(today=None):
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _get(url, path, refresh=False, **kw):
    if refresh or not path.exists():
        r = requests.get(url, timeout=60, headers={"User-Agent": "football-oracle/0.1"}, **kw)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path


def _clean(df):
    """football-data CSV -> standard columns + market implied probabilities."""
    df = df.rename(columns=lambda c: c.replace("\xef\xbb\xbf", "").strip()).dropna(subset=["HomeTeam", "AwayTeam"])
    out = df.reindex(columns=KEEP).copy()
    out["Date"] = pd.to_datetime(out["Date"], format="mixed", dayfirst=True)
    for c in KEEP[5:]:
        if c != "Referee":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # market average price; older seasons call it BbAv, fall back to single books
    odds = pd.DataFrame(np.nan, index=df.index, columns=list("HDA"))
    for pre in ["Avg", "BbAv", "B365", "PS"]:
        for s in "HDA":
            if pre + s in df:
                odds[s] = odds[s].fillna(pd.to_numeric(df[pre + s], errors="coerce"))
    inv = 1 / odds
    for s in "HDA":
        out["Odds" + s] = odds[s]
        out["mkt_" + s] = inv[s] / inv.sum(axis=1, min_count=3)
    return out


def load_results():
    cur = current_season()
    frames = []
    for y in range(FIRST_SEASON, cur + 1):
        code = f"{y % 100:02d}{(y + 1) % 100:02d}"
        for div in LEAGUES:
            p = _get(f"https://www.football-data.co.uk/mmz4281/{code}/{div}.csv",
                     CACHE / "fd" / f"{code}_{div}.csv", refresh=y == cur)
            df = _clean(pd.read_csv(p, encoding="latin-1", on_bad_lines="skip"))
            df["Season"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True).dropna(subset=["FTHG", "FTAG"])


def _norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    drop = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "us", "ogc", "rc", "cd", "ud", "sv", "vfb", "vfl",
            "tsg", "bv", "sk", "fk", "rcd", "de", "1.", "04", "05", "1846", "1899", "1900", "1907", "1910"}
    return " ".join(t for t in s.replace("-", " ").split() if t not in drop)


def map_names(fx_names, fd_names):
    fd_names = list(fd_names)
    out = {}
    for n in fx_names:
        if n in ALIASES:
            out[n] = ALIASES[n]
        else:
            out[n] = max(fd_names, key=lambda f: difflib.SequenceMatcher(None, _norm(n), _norm(f)).ratio())
    s = pd.Series(out)
    if s.duplicated().any():
        print(f"WARNING: ambiguous team names, add to ALIASES in data.py: {s[s.duplicated(keep=False)].to_dict()}")
    return out


def load_upcoming(results):
    """Unplayed fixtures for the current season in football-data team naming, plus team -> stadium."""
    cur = current_season()
    near = _clean(pd.read_csv(_get("https://www.football-data.co.uk/fixtures.csv", CACHE / "fixtures.csv", True),
                              encoding="latin-1"))
    near = near[near["Div"].isin(LEAGUES)]
    season = []
    stadium = {}
    for div, (_, slug, country) in LEAGUES.items():
        try:
            fx = pd.read_json(_get(f"https://fixturedownload.com/feed/json/{slug}-{cur}",
                                   CACHE / f"fx_{slug}_{cur}.json", True))
        except Exception as e:  # schedule source is optional: near-term fixtures still come from football-data
            print(f"fixturedownload {slug}: {e}")
            continue
        teams = set(results.loc[(results.Div == div) & (results.Season == cur), "HomeTeam"]) | \
            set(near.loc[near.Div == div, "HomeTeam"])
        names = map_names(set(fx.HomeTeam) | set(fx.AwayTeam), teams)
        fx["HomeTeam"], fx["AwayTeam"] = fx.HomeTeam.map(names), fx.AwayTeam.map(names)
        for team, loc in fx.groupby("HomeTeam")["Location"].agg(lambda s: s.mode().iat[0]).items():
            stadium[team] = (loc, country)
        when = pd.to_datetime(fx["DateUtc"], utc=True).dt.tz_convert("Europe/Istanbul" if div == "T1" else "Europe/Paris")
        fx = fx[fx.HomeTeamScore.isna()]
        when = when[fx.index]
        season.append(pd.DataFrame({"Div": div, "Date": when.dt.tz_localize(None).dt.normalize(),
                                    "Time": when.dt.strftime("%H:%M"), "HomeTeam": fx.HomeTeam, "AwayTeam": fx.AwayTeam}))
    up = pd.concat([near, *season], ignore_index=True).drop_duplicates(["Div", "HomeTeam", "AwayTeam"])
    up["Season"] = cur
    played = results[results.Season == cur].set_index(["Div", "HomeTeam", "AwayTeam"]).index
    up = up[~up.set_index(["Div", "HomeTeam", "AwayTeam"]).index.isin(played)]
    up = up[up.Date >= pd.Timestamp(date.today() - timedelta(days=1))]
    return up.reset_index(drop=True), stadium


def load_stadiums(stadium):
    """team -> lat/lon. Geocodes only teams missing from data/stadiums.csv, so manual fixes there are kept."""
    known = pd.read_csv(STADIUMS) if STADIUMS.exists() else pd.DataFrame(columns=["team", "stadium", "lat", "lon"])
    new = []
    for team, (name, country) in stadium.items():
        if team in set(known.team):
            continue
        lat = lon = np.nan
        for q in (name, f"{team} stadium"):
            r = requests.get("https://nominatim.openstreetmap.org/search", timeout=30,
                             params={"q": q, "format": "json", "limit": 1, "countrycodes": country},
                             headers={"User-Agent": "football-oracle/0.1"})
            time.sleep(1.1)  # Nominatim usage policy: max 1 request/second
            if r.ok and r.json():
                lat, lon = float(r.json()[0]["lat"]), float(r.json()[0]["lon"])
                break
        new.append({"team": team, "stadium": name, "lat": lat, "lon": lon})
    if new:
        known = pd.concat([known, pd.DataFrame(new)], ignore_index=True)
        DATA.mkdir(exist_ok=True)
        known.to_csv(STADIUMS, index=False)
    return known


def load_weather(stadiums):
    """Daily weather at each home stadium: max temp (C), precipitation (mm), max wind (km/h)."""
    daily = "temperature_2m_max,precipitation_sum,wind_speed_10m_max"
    frames, archive_ok = [], True
    for team, lat, lon in stadiums[["team", "lat", "lon"]].dropna().itertuples(index=False):
        safe = "".join(c if c.isalnum() else "_" for c in team)
        arch = CACHE / "weather" / f"{safe}.json"
        stale = arch.exists() and time.time() - arch.stat().st_mtime > 60 * 86400  # recent API covers 92 past days
        files = []
        # an 11-year archive call is "heavy" for open-meteo's free quota (~10k/day); missing ones fill in on later runs
        if archive_ok and (stale or not arch.exists()):
            try:
                _get("https://archive-api.open-meteo.com/v1/archive", arch, True, params={
                    "latitude": lat, "longitude": lon, "start_date": f"{FIRST_SEASON}-07-01",
                    "end_date": str(date.today() - timedelta(days=7)), "daily": daily})
            except requests.RequestException as e:
                print(f"weather history paused (rerun later to continue): {e}")
                archive_ok = False
        if arch.exists():
            files.append(arch)
        try:
            files.append(_get("https://api.open-meteo.com/v1/forecast", CACHE / "weather" / f"{safe}_fc.json", True,
                              params={"latitude": lat, "longitude": lon, "past_days": 92, "forecast_days": 16,
                                      "daily": daily}))
        except requests.RequestException as e:
            print(f"weather forecast {team}: {e}")
        for p in files:
            try:
                d = pd.DataFrame(json.loads(p.read_text())["daily"])
            except (ValueError, KeyError):  # truncated download: drop it so the next run refetches
                p.unlink()
                continue
            frames.append(pd.DataFrame({"HomeTeam": team, "Date": pd.to_datetime(d["time"]),
                                        "temp": d["temperature_2m_max"], "rain": d["precipitation_sum"],
                                        "wind": d["wind_speed_10m_max"]}))
    if not frames:
        return pd.DataFrame(columns=["HomeTeam", "Date", "temp", "rain", "wind"])
    return pd.concat(frames).drop_duplicates(["HomeTeam", "Date"], keep="last")
