"""API-Football (api-sports.io) free tier: line-ups, formations, player ratings, coaches and injuries for all leagues.
Free plan: 100 requests/day (resets 00:00 UTC), 10/minute, and only some seasons. Every response is cached on disk
and a daily counter stops at DAILY_BUDGET, so history backfills a little on each run.
Key: free at dashboard.api-football.com, read from the APIFOOTBALL_KEY environment variable."""
import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests

import data

URL = "https://v3.football.api-sports.io"
LEAGUE_IDS = {"E0": 39, "SP1": 140, "I1": 135, "D1": 78, "F1": 61, "N1": 88, "T1": 203}
DAILY_BUDGET = int(os.environ.get("APIFOOTBALL_DAILY_BUDGET", 95))  # keep a few spare under the 100/day cap
CACHE = data.CACHE / "apifootball"
STATE = CACHE / "state.json"
FINISHED = {"FT", "AET", "PEN"}
_last_call = 0.0


def state():
    """{"day", "used", "blocked": [seasons], "allowed": [from, to], "checked": iso date}"""
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    today = datetime.now(timezone.utc).date().isoformat()
    if st.get("day") != today:
        st.update(day=today, used=0)
    if st.get("checked", "") < str(date.fromordinal(date.today().toordinal() - 30)):  # re-probe plan limits monthly
        st.update(blocked=[], allowed=None, checked=today)
    return st


def _save(st):
    CACHE.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st))


def season_ok(st, season):
    a = st.get("allowed")
    return season not in st.get("blocked", []) and (not a or a[0] <= season <= a[1])


def get(endpoint, max_age_days=None, **params):
    """Cached GET returning the response list, or None when it isn't cached and can't be fetched
    (no key, daily budget spent, plan restriction). max_age_days=None caches forever."""
    global _last_call
    name = endpoint + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    if len(name) > 80:  # batched ids would make Windows paths too long
        name = endpoint + "_" + hashlib.md5(name.encode()).hexdigest()
    path = CACHE / f"{name}.json"
    stale = json.loads(path.read_text(encoding="utf-8"))["response"] if path.exists() else None
    if stale is not None and (max_age_days is None or time.time() - path.stat().st_mtime < max_age_days * 86400):
        return stale
    key, st = os.environ.get("APIFOOTBALL_KEY"), state()
    if not key or st["used"] >= DAILY_BUDGET:
        return stale
    time.sleep(max(0.0, 6.5 - (time.time() - _last_call)))  # free plan: 10 requests/minute
    _last_call = time.time()
    try:
        r = requests.get(f"{URL}/{endpoint}", params=params, headers={"x-apisports-key": key}, timeout=30)
        body = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"API-Football {endpoint} {params}: {e}")
        return stale
    st["used"] += 1
    limit, left = r.headers.get("x-ratelimit-requests-limit"), r.headers.get("x-ratelimit-requests-remaining")
    if limit and left:  # the server's count also includes requests made elsewhere today
        st["used"] = max(st["used"], int(limit) - int(left))
    errors = body.get("errors")
    if errors:
        msg = " ".join(map(str, errors.values() if isinstance(errors, dict) else errors))
        if "season" in msg.lower() and "season" in params:  # e.g. "Free plans do not have access to this season, try from 2022 to 2024."
            st["blocked"] = sorted(set(st.get("blocked", [])) | {params["season"]})
            if m := re.search(r"from (\d{4}) to (\d{4})", msg):
                st["allowed"] = [int(m[1]), int(m[2])]
        elif "limit" in msg.lower() or "token" in msg.lower() or "key" in msg.lower():
            st["used"] = DAILY_BUDGET
        print(f"API-Football {endpoint} {params}: {msg}")
        _save(st)
        return stale
    _save(st)
    path.write_bytes(r.content)
    return body["response"]


def _cached(prefix):
    for p in CACHE.glob(f"{prefix}_*.json"):
        yield p, json.loads(p.read_text(encoding="utf-8"))["response"]


def sync():
    """Spend today's budget in priority order: fixture lists -> coaches -> injuries -> match details."""
    if not os.environ.get("APIFOOTBALL_KEY"):
        return
    cur = data.current_season()
    lists = []
    for season in range(cur, data.FIRST_SEASON - 1, -1):
        for lid in LEAGUE_IDS.values():
            if not season_ok(state(), season):
                break  # one blocked request per season is enough to learn the plan's limits
            res = get("fixtures", 1 if season == cur else None, league=lid, season=season)
            lists += res or []
    teams = dict.fromkeys(t["id"] for f in sorted(lists, key=lambda f: -f["league"]["season"])
                          for t in f["teams"].values())
    for team in teams:
        get("coachs", 14, team=team)
    if season_ok(state(), cur):
        for lid in LEAGUE_IDS.values():
            get("injuries", 1, league=lid, season=cur)
    have = {f["fixture"]["id"] for _, res in _cached("fixtures") for f in res if "lineups" in f}
    todo = [f["fixture"]["id"] for f in sorted(lists, key=lambda f: -f["fixture"]["timestamp"])
            if f["fixture"]["status"]["short"] in FINISHED and f["fixture"]["id"] not in have]
    for i in range(0, len(todo), 20):  # one request returns line-ups and player stats for up to 20 fixtures
        if state()["used"] >= DAILY_BUDGET:
            break
        get("fixtures", ids="-".join(map(str, todo[i:i + 20])))


def load(results):
    """Parse everything cached so far. Returns dict of DataFrames: apps, formations, coaches, unavailable."""
    sync()
    league_div = {v: k for k, v in LEAGUE_IDS.items()}
    fixtures = {}
    for _, res in _cached("fixtures"):  # season lists and detail batches; keep the detailed copy
        for f in res:
            if "lineups" in f or f["fixture"]["id"] not in fixtures:
                fixtures[f["fixture"]["id"]] = f

    # API team id -> football-data name, per league and season
    api_names = {}
    for f in fixtures.values():
        key = (league_div.get(f["league"]["id"]), f["league"]["season"])
        for t in f["teams"].values():
            api_names.setdefault(key, {})[t["id"]] = t["name"]
    team_map, latest = {}, {}
    for (div, season), ids in sorted(api_names.items(), key=lambda kv: kv[0][1]):
        fd = results[(results.Div == div) & (results.Season == season)]
        fd = set(fd.HomeTeam) | set(fd.AwayTeam)
        if div is None or not fd:
            continue
        names = data.map_names(set(ids.values()), fd)
        for tid, n in ids.items():
            team_map[(div, season, tid)] = latest[tid] = names[n]

    apps, forms = [], []
    for f in fixtures.values():
        if "lineups" not in f:
            continue
        div, season = league_div.get(f["league"]["id"]), f["league"]["season"]
        side = {t["id"]: team_map.get((div, season, t["id"])) for t in f["teams"].values()}
        home, away = side[f["teams"]["home"]["id"]], side[f["teams"]["away"]["id"]]
        if not home or not away:
            continue
        match = {"Div": div, "Season": season, "HomeTeam": home, "AwayTeam": away}
        for lu in f["lineups"]:
            forms.append({**match, "team": side.get(lu["team"]["id"]), "formation": lu.get("formation")})
        if div == "E0":
            continue  # Premier League players come from FPL, which has a longer history
        for tp in f.get("players", []):
            for p in tp["players"]:
                g = p["statistics"][0]["games"]
                if g.get("minutes"):
                    apps.append({**match, "team": side.get(tp["team"]["id"]), "player": f"af:{p['player']['id']}",
                                 "name": p["player"]["name"], "started": not g.get("substitute"),
                                 "minutes": g["minutes"], "rating": pd.to_numeric(g.get("rating"), errors="coerce")})

    coaches = []
    for p, res in _cached("coachs"):
        tid = int(re.search(r"team-(\d+)", p.name)[1])
        for c in res:
            for job in c.get("career") or []:
                if job["team"]["id"] == tid and tid in latest and job.get("start"):
                    coaches.append({"team": latest[tid], "coach": c["id"], "coach_name": c["name"],
                                    "start": job["start"], "end": job.get("end")})

    out, today = [], str(date.today())
    for _, res in _cached("injuries"):
        for i in res:
            if (league_div.get(i["league"]["id"]) != "E0" and i["player"].get("type") == "Missing Fixture"
                    and i["fixture"]["date"][:10] >= today and i["team"]["id"] in latest):
                out.append({"team": latest[i["team"]["id"]], "player": f"af:{i['player']['id']}",
                            "name": i["player"]["name"], "reason": i["player"].get("reason") or "unavailable"})

    st = state()
    return {
        "apps": pd.DataFrame(apps, columns=["Div", "Season", "HomeTeam", "AwayTeam", "team", "player", "name",
                                            "started", "minutes", "rating"]),
        "formations": pd.DataFrame(forms, columns=["Div", "Season", "HomeTeam", "AwayTeam", "team", "formation"]),
        "coaches": pd.DataFrame(coaches, columns=["team", "coach", "coach_name", "start", "end"]).drop_duplicates(),
        "unavailable": pd.DataFrame(out, columns=["team", "player", "name", "reason"]).drop_duplicates(["team", "player"]),
        "status": {"key_set": bool(os.environ.get("APIFOOTBALL_KEY")), "used_today": st["used"],
                   "daily_budget": DAILY_BUDGET, "allowed_seasons": st.get("allowed"), "blocked_seasons": st.get("blocked"),
                   "fixtures_with_lineups": sum("lineups" in f for f in fixtures.values()),
                   "coach_records": len(coaches)},
    }
