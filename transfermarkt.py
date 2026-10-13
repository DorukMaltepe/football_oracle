"""Transfermarkt history from github.com/dcaribou/transfermarkt-datasets (CC0, downloaded once, ~140 MB):
managers, referees, formations and starting XIs for every match since 2015/16, plus player market values.
Its updates stopped on 6 July 2026, so it covers seasons up to 2025/26; Sofascore covers the current season."""
import numpy as np
import pandas as pd

import data

URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
COMPETITIONS = {"E0": "GB1", "SP1": "ES1", "I1": "IT1", "D1": "L1", "F1": "FR1", "N1": "NL1", "T1": "TR1"}
CACHE = data.CACHE / "transfermarkt"
KEYS = ["Div", "Season", "HomeTeam", "AwayTeam"]


def _csv(name, **kw):
    return pd.read_csv(data._get(f"{URL}/{name}.csv.gz", CACHE / f"{name}.csv.gz"), **kw)


def load(results):
    """Returns dict of DataFrames: matches (Referee), apps (starting XIs, all leagues), formations,
    managers (one row per team per match) and values (player market value history)."""
    div_of = {v: k for k, v in COMPETITIONS.items()}
    g = _csv("games")
    g = g[g.competition_id.isin(div_of) & (g.season >= data.FIRST_SEASON)].rename(columns={"season": "Season"})
    g["Div"] = g.competition_id.map(div_of)
    names = {}
    for (div, season), x in g.groupby(["Div", "Season"]):
        fd = results[(results.Div == div) & (results.Season == season)]
        if len(fd):
            tm = set(x.home_club_name) | set(x.away_club_name)
            names.update({(div, season, n): f for n, f in data.map_names(tm, set(fd.HomeTeam) | set(fd.AwayTeam)).items()})
    for side in ("home", "away"):
        g[f"{side.title()}Team"] = [names.get(k) for k in zip(g.Div, g.Season, g[f"{side}_club_name"])]
    g = g.dropna(subset=["HomeTeam", "AwayTeam"]).drop_duplicates(KEYS)

    sides = pd.concat([pd.DataFrame({**{k: g[k] for k in KEYS}, "game_id": g.game_id, "club_id": g[f"{s}_club_id"],
                                     "date": g.date, "team": g[f"{s.title()}Team"],
                                     "formation": g[f"{s}_club_formation"].str.extract(r"^(\d[\d-]*\d)")[0],  # "4-3-3 Attacking"
                                     "manager": g[f"{s}_club_manager_name"]}) for s in ("home", "away")])
    mgr = sides.dropna(subset=["manager"])

    cols = ["game_id", "player_id", "club_id", "player_name", "type"]
    lu = pd.concat(c[c.game_id.isin(g.game_id) & (c.type == "starting_lineup")]
                   for c in _csv("game_lineups", usecols=cols, chunksize=500_000))
    lu = lu.merge(sides[[*KEYS, "game_id", "club_id", "team"]], on=["game_id", "club_id"])
    apps = pd.DataFrame({**{k: lu[k] for k in KEYS}, "team": lu.team, "player": "p:" + lu.player_name.map(data.person),
                         "name": lu.player_name, "started": True, "minutes": np.nan, "rating": np.nan,
                         "tm_id": lu.player_id})
    return {
        "matches": g[[*KEYS, "referee"]].rename(columns={"referee": "Referee"}),
        "apps": apps,
        "formations": sides[[*KEYS, "team", "formation"]],
        "managers": pd.DataFrame({"team": mgr.team, "date": mgr.date, "coach": "c:" + mgr.manager.map(data.person),
                                  "coach_name": mgr.manager}),
        "values": _csv("player_valuations", usecols=["player_id", "date", "market_value_in_eur"]),
    }


def market_values(apps, tm):
    """Add each player's latest Transfermarkt market value before the match (column "value", EUR).
    Players from other sources are found by club + name, or by name alone when that is unique.
    apps needs a Date column."""
    known = tm["apps"].assign(key=tm["apps"].player.str.split(":", n=1).str[1])
    by_team = known.drop_duplicates(["team", "key"], keep="last").set_index(["team", "key"]).tm_id
    ids = known.drop_duplicates(["key", "tm_id"]).drop_duplicates("key", keep=False).set_index("key").tm_id
    key = apps.player.str.split(":", n=1).str[1]
    tm_id = apps.get("tm_id", pd.Series(np.nan, index=apps.index))
    tm_id = tm_id.fillna(pd.Series(by_team.reindex(pd.MultiIndex.from_arrays([apps.team, key])).to_numpy(), index=apps.index))
    tm_id = tm_id.fillna(key.map(ids))
    v = tm["values"].assign(date=pd.to_datetime(tm["values"].date)).dropna(subset=["date"]).sort_values("date")
    a = apps.assign(tm_id=tm_id.astype(float), _row=np.arange(len(apps))).sort_values("Date")
    a = pd.merge_asof(a.dropna(subset=["tm_id"]), v.assign(tm_id=v.player_id.astype(float))[["tm_id", "date", "market_value_in_eur"]],
                      left_on="Date", right_on="date", by="tm_id", direction="backward")
    out = apps.drop(columns=["tm_id"], errors="ignore").copy()
    out["value"] = pd.Series(a.market_value_in_eur.to_numpy(), index=a._row.to_numpy()).reindex(range(len(apps))).to_numpy()
    return out
