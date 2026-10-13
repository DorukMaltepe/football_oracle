import json

import pandas as pd
import pytest

import apifootball as af


class FakeResponse:
    def __init__(self, body, left=90):
        self.content = json.dumps(body).encode()
        self.headers = {"x-ratelimit-requests-limit": "100", "x-ratelimit-requests-remaining": str(left)}

    def json(self):
        return json.loads(self.content)


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(af, "CACHE", tmp_path)
    monkeypatch.setattr(af, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(af.time, "sleep", lambda s: None)
    monkeypatch.setenv("APIFOOTBALL_KEY", "test")
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append((url, params))
        if params.get("season") == 2026:
            return FakeResponse({"errors": {"plan": "Free plans do not have access to this season, try from 2022 to 2024."},
                                 "response": []})
        return FakeResponse({"errors": [], "response": [{"id": 1}]}, left=100 - len(calls))

    monkeypatch.setattr(af.requests, "get", fake_get)
    return calls


def test_cache_budget_and_plan_limits(api, monkeypatch):
    assert af.get("fixtures", league=39, season=2026) is None
    st = af.state()
    assert st["allowed"] == [2022, 2024] and not af.season_ok(st, 2026) and af.season_ok(st, 2023)
    assert af.get("coachs", 14, team=33) == [{"id": 1}]
    assert af.get("coachs", 14, team=33) == [{"id": 1}] and len(api) == 2  # second read comes from disk
    monkeypatch.setattr(af, "DAILY_BUDGET", 2)
    assert af.get("coachs", team=34) is None and len(api) == 2  # budget spent: no request


def test_parse_lineups_coaches_injuries(api, tmp_path, monkeypatch):
    fixture = {"fixture": {"id": 7, "timestamp": 1, "date": "2024-03-01T19:00:00+00:00", "status": {"short": "FT"}},
               "league": {"id": 140, "season": 2023},
               "teams": {"home": {"id": 1, "name": "Real Madrid"}, "away": {"id": 2, "name": "FC Barcelona"}},
               "lineups": [{"team": {"id": 1}, "formation": "4-3-3"}, {"team": {"id": 2}, "formation": "4-2-3-1"}],
               "players": [{"team": {"id": 1}, "players": [
                   {"player": {"id": 10, "name": "Vini"}, "statistics": [{"games": {"minutes": 90, "rating": "7.4", "substitute": False}}]},
                   {"player": {"id": 11, "name": "Bench"}, "statistics": [{"games": {"minutes": None, "rating": None, "substitute": True}}]}]}]}
    (tmp_path / "fixtures_abc.json").write_text(json.dumps({"response": [fixture]}))
    (tmp_path / "coachs_team-1.json").write_text(json.dumps({"response": [
        {"id": 5, "name": "Carlo", "career": [{"team": {"id": 1}, "start": "2021-06-01", "end": None},
                                              {"team": {"id": 99}, "start": "2019-01-01", "end": "2021-05-31"}]}]}))
    (tmp_path / "injuries_league-140_season-2026.json").write_text(json.dumps({"response": [
        {"player": {"id": 10, "name": "Vini", "type": "Missing Fixture", "reason": "Knee"}, "team": {"id": 1},
         "fixture": {"date": "2099-01-01T19:00:00+00:00"}, "league": {"id": 140}}]}))
    results = pd.DataFrame({"Div": "SP1", "Season": 2023, "HomeTeam": ["Real Madrid"], "AwayTeam": ["Barcelona"]})
    monkeypatch.delenv("APIFOOTBALL_KEY")  # parse the cache only, no network
    out = af.load(results)
    apps = out["apps"]
    assert len(apps) == 1 and apps.iloc[0].player == "af:10" and apps.iloc[0].HomeTeam == "Real Madrid"
    assert apps.iloc[0].AwayTeam == "Barcelona" and apps.iloc[0].started and apps.iloc[0].rating == 7.4
    assert set(out["formations"].formation) == {"4-3-3", "4-2-3-1"}
    assert out["coaches"].to_dict("records") == [{"team": "Real Madrid", "coach": 5, "coach_name": "Carlo",
                                                  "start": "2021-06-01", "end": None}]
    assert out["unavailable"].iloc[0][["team", "name", "reason"]].tolist() == ["Real Madrid", "Vini", "Knee"]
