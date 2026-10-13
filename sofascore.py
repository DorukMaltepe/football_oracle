"""Sofascore, scraped from the JSON API behind sofascore.com (no key): line-ups, player ratings, formations,
managers, referees, injured/suspended players and match statistics (xG etc.) for every league since 2015/16.
The site rejects ordinary HTTP clients (403), so curl_cffi impersonates Chrome's TLS fingerprint.
Only the fields used are kept, in one JSON file per league-season. Finished matches are fetched once; season lists
and upcoming line-ups refresh on each run. A pipeline run scrapes match details for the current season only
(Transfermarkt and Understat cover the history); RUN_BUDGET caps requests per run, newest matches first.
Full history backfill: SOFASCORE_RUN_BUDGET=200000 python sofascore.py"""
import json
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
from curl_cffi import requests

import data

URL = "https://www.sofascore.com/api/v1"
TOURNAMENTS = {"E0": 17, "SP1": 8, "I1": 23, "D1": 35, "F1": 34, "N1": 37, "T1": 52}
RUN_BUDGET = int(os.environ.get("SOFASCORE_RUN_BUDGET", 1000))  # ~4 requests per match, ~20 min per run
DELAY = 1.0  # seconds between requests; ~2,000 quick requests triggered a bot challenge
UPCOMING_DAYS = 7  # Sofascore lists missing players a few days before kick-off
CACHE = data.CACHE / "sofascore"
_session, _used, _blocked = None, 0, False


def fetch(path):
    """GET URL/path -> JSON. {} when Sofascore has no such data (404), None when over budget, blocked or offline."""
    global _session, _used, _blocked
    if _used >= RUN_BUDGET or _blocked:
        return None
    _session = _session or requests.Session(impersonate="chrome")
    time.sleep(DELAY)
    _used += 1
    try:
        r = _session.get(f"{URL}/{path}", timeout=30)
    except requests.RequestsError as e:
        print(f"Sofascore {path}: {e}")
        return None
    if r.status_code == 404:
        return {}
    if r.status_code != 200:
        print(f"Sofascore {path}: HTTP {r.status_code}")
        _blocked = r.status_code in (403, 429)  # bot challenge or rate limit: stop for this run
        return None
    return r.json()


def _lineups(js):
    """{side: {"formation", "players": [[id, name, substitute, minutes, rating]], "missing": [[id, name, type, why, back]]}}"""
    return {side: {"formation": t.get("formation"),
                   "players": [[p["player"]["id"], p["player"]["name"], p.get("substitute", False),
                                p.get("statistics", {}).get("minutesPlayed"), p.get("statistics", {}).get("rating")]
                               for p in t.get("players", [])],
                   "missing": [[m["player"]["id"], m["player"]["name"], m.get("type"), m.get("description"),
                                m.get("expectedEndDate")] for m in t.get("missingPlayers", [])]}
            for side, t in js.items() if side in ("home", "away")}


def _managers(js):
    return {side: [m["id"], m["name"]] for side, m in (("home", js.get("homeManager")), ("away", js.get("awayManager"))) if m}


def _statistics(js):
    """Whole-match stats: {key: [home, away]}, e.g. expectedGoals, ballPossession, bigChanceCreated."""
    full = next((p for p in js.get("statistics", []) if p["period"] == "ALL"), {"groups": []})
    return {i["key"]: [i.get("homeValue"), i.get("awayValue")] for g in full["groups"] for i in g["statisticsItems"]}


def _event(js):
    return {"referee": (js.get("event", {}).get("referee") or {}).get("name")}


PARTS = {"lineups": _lineups, "managers": _managers, "statistics": _statistics, "": _event}  # fetch order per match


def _file(div, season):
    return CACHE / f"{div}_{season}.json"


def _read(div, season):
    f = _file(div, season)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"events": {}, "details": {}, "complete": False}


def _list(tid, sid, store, refresh):
    """Season's past matches, newest page first. Once the list is complete, a refresh stops at the first page
    with nothing new. Returns False if a request failed."""
    page = 0
    while True:
        js = fetch(f"unique-tournament/{tid}/season/{sid}/events/last/{page}")
        if js is None:
            return False
        new = False
        for e in js.get("events", []):
            # [kick-off, home, away, status, stage]; stage tells regular season from play-offs in the same tournament
            rec = [e["startTimestamp"], e["homeTeam"]["name"], e["awayTeam"]["name"], e["status"]["type"],
                   e.get("tournament", {}).get("name")]
            new |= store["events"].get(str(e["id"])) != rec
            store["events"][str(e["id"])] = rec
        if not js.get("hasNextPage") or (refresh and store["complete"] and not new):
            return True
        page += 1


def _details(stores, first_season):
    """Fetch the missing parts of finished matches, newest first, until a request fails or the budget is spent."""
    todo = sorted(((rec[0], key, eid) for key, st in stores.items() if key[1] >= first_season
                   for eid, rec in st["events"].items()
                   if rec[3] == "finished" and len(st["details"].get(eid, {})) < len(PARTS)), reverse=True)
    for _, key, eid in todo:
        d = stores[key]["details"].setdefault(eid, {})
        for part, parse in PARTS.items():
            if part not in d:
                js = fetch(f"event/{eid}/{part}".rstrip("/"))
                if js is None:
                    return
                d[part] = parse(js) if js else None  # None: Sofascore has no such data for this match


def sync(seasons=1):
    """Lists first, then line-ups of the next few days' matches, then details of finished matches of the last
    `seasons` seasons, newest first.
    Returns (stores {(div, season): store}, upcoming [(div, event id, kick-off, home, away, lineups)])."""
    cur, stores, upcoming, sids = data.current_season(), {}, [], {}
    for div, tid in TOURNAMENTS.items():  # Sofascore season ids, e.g. "26/27" -> 96668
        listed = (fetch(f"unique-tournament/{tid}/seasons") or {}).get("seasons", [])
        # recent seasons are labelled "26/27", old ones "1969/1970"
        sids[div] = {2000 + int(s["year"][:2]): s["id"] for s in listed if re.fullmatch(r"\d\d/\d\d", s["year"])}
    for season in range(cur, data.FIRST_SEASON - 1, -1):  # season-outer so a small budget covers every league
        for div, tid in TOURNAMENTS.items():
            store = stores[(div, season)] = _read(div, season)
            if season in sids[div] and (season == cur or not store["complete"]):
                if _list(tid, sids[div][season], store, refresh=season == cur):
                    store["complete"] = True
        if season != cur:
            continue
        soon = time.time() + UPCOMING_DAYS * 86400
        for div, tid in TOURNAMENTS.items():
            if cur not in sids[div]:
                continue
            for e in (fetch(f"unique-tournament/{tid}/season/{sids[div][cur]}/events/next/0") or {}).get("events", []):
                if e["startTimestamp"] < soon:
                    lu = fetch(f"event/{e['id']}/lineups")
                    upcoming.append((div, e["id"], e["startTimestamp"], e["homeTeam"]["name"],
                                     e["awayTeam"]["name"], _lineups(lu) if lu else {}))

    try:
        _details(stores, cur - seasons + 1)
    finally:
        CACHE.mkdir(parents=True, exist_ok=True)
        for (div, season), st in stores.items():
            if st["events"]:
                _file(div, season).write_text(json.dumps(st), encoding="utf-8")
    return stores, upcoming


def load(results):
    """Returns dict of DataFrames: matches (xG, referee), apps, formations, managers (one row per team per match),
    unavailable, plus status."""
    stores, upcoming = sync()
    keys = ["Div", "Season", "HomeTeam", "AwayTeam"]
    matches, apps, forms, mgrs, latest = [], [], [], [], {}
    for (div, season), st in sorted(stores.items(), key=lambda kv: kv[0][1]):
        fd = results[(results.Div == div) & (results.Season == season)]
        fd_names = set(fd.HomeTeam) | set(fd.AwayTeam)
        # the most common stage is the regular season; play-offs can involve teams from other divisions
        stages = pd.Series([r[4] for r in st["events"].values()]).value_counts()
        events = {eid: r for eid, r in st["events"].items() if r[3] == "finished" and r[4] == stages.index[0]} \
            if len(stages) else {}
        if not fd_names or not events:
            continue
        names = data.map_names({n for r in events.values() for n in r[1:3]}, fd_names)
        for n, fd_n in names.items():
            latest[(div, n)] = fd_n
        for eid, (ts, h, a, *_) in events.items():
            d = st["details"].get(eid) or {}
            match = {"Div": div, "Season": season, "HomeTeam": names[h], "AwayTeam": names[a]}
            side_team = {"home": names[h], "away": names[a]}
            stats, ev = d.get("statistics") or {}, d.get("") or {}
            if stats or ev:
                xg = stats.get("expectedGoals") or [None, None]
                matches.append({**match, "HxG": xg[0], "AxG": xg[1], "Referee": ev.get("referee")})
            date = datetime.fromtimestamp(ts, timezone.utc).date()
            for side, m in (d.get("managers") or {}).items():
                mgrs.append({"team": side_team[side], "date": date, "coach": "c:" + data.person(m[1]), "coach_name": m[1]})
            for side, t in (d.get("lineups") or {}).items():
                forms.append({**match, "team": side_team[side], "formation": t["formation"]})
                if div == "E0":
                    continue  # Premier League players come from FPL, matching its injury list
                for pid, name, sub, mins, rating in t["players"]:
                    if mins:
                        apps.append({**match, "team": side_team[side], "player": "p:" + data.person(name), "name": name,
                                     "started": not sub, "minutes": mins, "rating": rating})

    out, seen = [], set()
    for div, _, ts, h, a, lu in sorted(upcoming, key=lambda u: u[2]):  # each team's next match only
        for side, name in (("home", h), ("away", a)):
            team = latest.get((div, name))
            if div == "E0" or not team or team in seen:
                continue
            seen.add(team)
            for pid, pname, kind, why, back in lu.get(side, {}).get("missing", []):
                reason = why or kind or "unavailable"
                if kind == "doubtful":
                    reason = f"doubtful: {reason}"
                if pd.notna(back := pd.to_datetime(back, errors="coerce", utc=True)):
                    reason += f" - expected back {back:%d %b}"
                out.append({"team": team, "player": "p:" + data.person(pname), "name": pname, "reason": reason})

    matches = pd.DataFrame(matches, columns=[*keys, "HxG", "AxG", "Referee"]).drop_duplicates(keys)
    return {
        "matches": matches,
        "apps": pd.DataFrame(apps, columns=[*keys, "team", "player", "name", "started", "minutes", "rating"]),
        "formations": pd.DataFrame(forms, columns=[*keys, "team", "formation"]),
        "managers": pd.DataFrame(mgrs, columns=["team", "date", "coach", "coach_name"]),
        "unavailable": pd.DataFrame(out, columns=["team", "player", "name", "reason"]),
        "status": {"requests_this_run": _used, "run_budget": RUN_BUDGET, "blocked": _blocked,
                   "finished_matches": sum(r[3] == "finished" for st in stores.values() for r in st["events"].values()),
                   "matches_scraped": sum(len(d) == len(PARTS) for st in stores.values() for d in st["details"].values())},
    }


if __name__ == "__main__":
    sync(seasons=99)
