import time

import pandas as pd
import pytest

import sofascore as ss

DAY = 86400
T0 = 1788220800  # 2026-09-01


def ev(eid, ts, home, away, status="finished", stage="LaLiga"):
    return {"id": eid, "startTimestamp": ts, "homeTeam": {"name": home}, "awayTeam": {"name": away},
            "status": {"type": status}, "tournament": {"name": stage}}


def player(pid, name, sub, mins, rating):
    return {"player": {"id": pid, "name": name}, "substitute": sub,
            "statistics": {"minutesPlayed": mins, "rating": rating} if mins else {}}


def manager(mid, name):
    return {"id": mid, "name": name}


SITE = {
    "unique-tournament/8/seasons": {"seasons": [{"year": "26/27", "id": 1}, {"year": "1926/1927", "id": 9}]},
    "unique-tournament/8/season/1/events/last/0": {"hasNextPage": False, "events": [
        ev(102, T0 + 7 * DAY, "FC Barcelona", "Real Madrid"), ev(101, T0, "Real Madrid", "FC Barcelona"),
        ev(103, T0 + 8 * DAY, "Real Madrid", "Second Division FC", stage="LaLiga, Playoffs")]},
    "event/101/lineups": {"home": {"formation": "4-3-3", "players": [player(10, "Vini", False, 90, 7.4),
                                                                     player(11, "Bench", True, None, None)]},
                          "away": {"formation": "4-2-3-1", "players": [player(20, "Pedri", False, 80, 6.9)]}},
    "event/101/managers": {"homeManager": manager(1, "Carlo"), "awayManager": manager(2, "Xavi")},
    "event/101/statistics": {"statistics": [{"period": "ALL", "groups": [{"statisticsItems": [
        {"key": "expectedGoals", "homeValue": 1.5, "awayValue": 0.7}]}]}]},
    "event/101": {"event": {"referee": {"name": "Gil Manzano"}}},
    "event/102/managers": {"homeManager": manager(3, "Flick"), "awayManager": manager(1, "Carlo")},
}


@pytest.fixture
def site(tmp_path, monkeypatch):
    now = int(time.time())
    pages = {**SITE, "unique-tournament/8/season/1/events/next/0": {"events": [ev(201, now + DAY, "Real Madrid", "FC Barcelona")]},
             "event/201/lineups": {"home": {"players": [], "missingPlayers": [
                 {"player": {"id": 10, "name": "Vini"}, "type": "missing", "description": "Knee Injury",
                  "expectedEndDate": "2026-10-20T00:00:00+00:00"}]}}}
    calls = []

    def fake_fetch(path):
        calls.append(path)
        return pages.get(path, {})  # {} = Sofascore has no such data

    monkeypatch.setattr(ss, "CACHE", tmp_path)
    monkeypatch.setattr(ss, "fetch", fake_fetch)
    monkeypatch.setattr(ss.data, "current_season", lambda: 2026)
    return calls


def test_scrape_and_parse(site):
    results = pd.DataFrame({"Div": "SP1", "Season": 2026, "HomeTeam": ["Real Madrid", "Barcelona"],
                            "AwayTeam": ["Barcelona", "Real Madrid"]})
    out = ss.load(results)
    m = out["matches"].set_index("HomeTeam")  # match 102 has no stats or referee: no row
    assert len(m) == 1 and m.loc["Real Madrid", "HxG"] == 1.5 and m.loc["Real Madrid", "Referee"] == "Gil Manzano"
    assert "unique-tournament/8/season/9/events/last/0" not in site  # "1926/1927" is not read as 2019
    apps = out["apps"]
    assert apps.player.tolist() == ["p:vini", "p:pedri"] and apps.iloc[0].started and apps.iloc[0].rating == 7.4
    assert set(out["formations"].formation) == {"4-3-3", "4-2-3-1"}
    c = ss.data.manager_stints(out["managers"]).sort_values(["team", "start"]).to_dict("records")
    assert [(r["team"], r["coach_name"], None if pd.isna(r["end"]) else str(r["end"])) for r in c] == [
        ("Barcelona", "Xavi", "2026-09-01 00:00:00"), ("Barcelona", "Flick", None), ("Real Madrid", "Carlo", None)]
    assert out["unavailable"][["team", "name", "reason"]].values.tolist() == [
        ["Real Madrid", "Vini", "Knee Injury - expected back 20 Oct"]]

    n = len(site)
    ss.load(results)  # finished matches come from disk; only lists and upcoming line-ups are fetched again
    assert not [p for p in site[n:] if p.startswith(("event/101", "event/102"))]


def test_fetch_stops_when_blocked(monkeypatch):
    class Session:
        def get(self, url, timeout):
            return type("R", (), {"status_code": 403})()

    monkeypatch.setattr(ss, "_session", Session())
    monkeypatch.setattr(ss, "_used", 0)
    monkeypatch.setattr(ss, "_blocked", False)
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)
    assert ss.fetch("event/1/lineups") is None
    assert ss.fetch("event/2/lineups") is None and ss._used == 1 and ss._blocked  # no second request after a block
