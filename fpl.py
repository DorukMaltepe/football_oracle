"""Premier League player data from Fantasy Premier League (unofficial, no key needed).
- past seasons: per-gameweek player rows from github.com/vaastav/Fantasy-Premier-League
- current season: the live FPL API (finished gameweeks), plus injuries/suspensions and news for every player
Player performance is FPL's BPS (bonus points system), a per-match score built from ~30 Opta stats."""
import json
import re
import unicodedata

import pandas as pd

import data

VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
API = "https://fantasy.premierleague.com/api"
FIRST = 2016  # first season in the vaastav archive
CACHE = data.CACHE / "fpl"


def _name(s):
    """'Aaron_Cresswell_402' (older seasons) and 'Aaron Cresswell' -> 'aaron cresswell'."""
    s = re.sub(r"_\d+$", "", str(s)).replace("_", " ")
    return " ".join(unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().split())


def _json(url, path, refresh):
    return json.loads(data._get(url, CACHE / path, refresh).read_text(encoding="utf-8"))


def _season(results, y):
    """One past season: link each player row to its football-data match via date + opponent + home/away."""
    s = f"{y}-{(y + 1) % 100:02d}"
    gw = pd.read_csv(data._get(f"{VAASTAV}/{s}/gws/merged_gw.csv", CACHE / f"{s}_merged_gw.csv"),
                     encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", low_memory=False)
    teams = pd.read_csv(data._get(f"{VAASTAV}/master_team_list.csv", CACHE / "master_team_list.csv"))
    teams = teams[teams.season == s].set_index("team")["team_name"]
    if teams.empty:  # the master list stops at 2023-24; newer seasons ship their own teams.csv
        teams = pd.read_csv(data._get(f"{VAASTAV}/{s}/teams.csv", CACHE / f"{s}_teams.csv")).set_index("id")["name"]
    res = results[(results.Div == "E0") & (results.Season == y)][["Date", "HomeTeam", "AwayTeam"]]
    names = data.map_names(set(teams), set(res.HomeTeam) | set(res.AwayTeam))
    gw = gw[pd.to_numeric(gw.minutes, errors="coerce") > 0].copy()
    gw["opp"] = gw.opponent_team.map(teams).map(names)
    gw["Date"] = pd.to_datetime(gw.kickoff_time, utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
    home = gw.was_home.astype(str) == "True"
    h = gw[home].merge(res, left_on=["Date", "opp"], right_on=["Date", "AwayTeam"]).assign(team=lambda d: d.HomeTeam)
    a = gw[~home].merge(res, left_on=["Date", "opp"], right_on=["Date", "HomeTeam"]).assign(team=lambda d: d.AwayTeam)
    gw = pd.concat([h, a])
    # shortcut: FPL records "starts" only from gameweek 16 of 2022/23; before that 60+ minutes stands in for starting
    started = gw.minutes >= 60
    if "starts" in gw:
        recorded = gw.groupby("fixture").starts.transform("sum") > 0
        started = started.where(~recorded, gw.starts == 1)
    return pd.DataFrame({"Div": "E0", "Season": y, "HomeTeam": gw.HomeTeam, "AwayTeam": gw.AwayTeam, "team": gw.team,
                         "player": "fpl:" + gw["name"].map(_name), "name": gw["name"].map(_name).str.title(),
                         "started": started, "minutes": gw.minutes, "rating": pd.to_numeric(gw.bps, errors="coerce")})


def load(results):
    """Returns (appearances, unavailable) for the Premier League."""
    cur = data.current_season()
    frames = [_season(results, y) for y in range(FIRST, cur)]

    boot = _json(f"{API}/bootstrap-static/", "bootstrap.json", True)
    fixtures = {f["id"]: f for f in _json(f"{API}/fixtures/", "fixtures.json", True)}
    fpl_team = {t["id"]: t["name"] for t in boot["teams"]}
    res = results[(results.Div == "E0") & (results.Season == cur)]
    names = data.map_names(set(fpl_team.values()), set(res.HomeTeam) | set(res.AwayTeam))
    team = {i: names[n] for i, n in fpl_team.items()}
    elements = {e["id"]: e for e in boot["elements"]}

    rows = []
    for ev in (e for e in boot["events"] if e["finished"]):
        live = _json(f"{API}/event/{ev['id']}/live/", f"live_{cur}_{ev['id']}.json", False)  # finished: never changes
        for el in live["elements"]:
            fx = {x["fixture"] for x in el["explain"]}
            e, st = elements.get(el["id"]), el["stats"]
            # shortcut: players with two fixtures in one gameweek are skipped; split explain stats per fixture to keep them
            if len(fx) != 1 or not e or not st["minutes"]:
                continue
            f = fixtures[fx.pop()]
            if e["team"] not in (f["team_h"], f["team_a"]):
                continue
            full = f"{e['first_name']} {e['second_name']}"
            rows.append({"Div": "E0", "Season": cur, "HomeTeam": team[f["team_h"]], "AwayTeam": team[f["team_a"]],
                         "team": team[e["team"]], "player": "fpl:" + _name(full), "name": full,
                         "started": st["starts"] == 1, "minutes": st["minutes"], "rating": st["bps"]})
    apps = pd.concat([*frames, pd.DataFrame(rows)], ignore_index=True)

    out = [{"team": team[e["team"]], "player": "fpl:" + _name(f"{e['first_name']} {e['second_name']}"),
            "name": e["web_name"], "reason": e["news"] or {"i": "injured", "s": "suspended"}.get(e["status"], "unavailable")}
           for e in boot["elements"]
           if e["status"] != "u"  # "u" = left the club (transfer or loan), not a squad absence
           and (e["status"] in ("i", "s", "n") or (e["chance_of_playing_next_round"] is not None
                                                    and e["chance_of_playing_next_round"] < 75))]
    return apps, pd.DataFrame(out, columns=["team", "player", "name", "reason"])
