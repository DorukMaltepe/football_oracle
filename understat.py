"""Understat (understat.com, no key): expected goals for every Top 5 league match since 2014/15.
One request per league season, the JSON its league page loads. Past seasons are cached for good."""
import json

import pandas as pd
import requests

import data

URL = "https://understat.com/getLeagueData"
LEAGUES = {"E0": "EPL", "SP1": "La_liga", "I1": "Serie_A", "D1": "Bundesliga", "F1": "Ligue_1"}
CACHE = data.CACHE / "understat"


def load(results):
    """Returns DataFrame: Div, Season, HomeTeam, AwayTeam, HxG, AxG for every played match."""
    cur, rows = data.current_season(), []
    for div, league in LEAGUES.items():
        for season in range(data.FIRST_SEASON, cur + 1):
            path = CACHE / f"{league}_{season}.json"
            try:
                data._get(f"{URL}/{league}/{season}", path, refresh=season == cur,
                          headers={"X-Requested-With": "XMLHttpRequest"})  # without it the endpoint answers 404
            except requests.RequestException as e:
                print(f"Understat {league} {season}: {e}")  # keep the cached copy, if any
            if not path.exists():
                continue
            games = [g for g in json.loads(path.read_text(encoding="utf-8"))["dates"] if g["isResult"]]
            fd = results[(results.Div == div) & (results.Season == season)]
            if not games or fd.empty:
                continue
            names = data.map_names({g[s]["title"] for g in games for s in "ha"}, set(fd.HomeTeam) | set(fd.AwayTeam))
            rows += [{"Div": div, "Season": season, "HomeTeam": names[g["h"]["title"]], "AwayTeam": names[g["a"]["title"]],
                      "HxG": float(g["xG"]["h"]), "AxG": float(g["xG"]["a"])} for g in games]
    return pd.DataFrame(rows, columns=["Div", "Season", "HomeTeam", "AwayTeam", "HxG", "AxG"])
