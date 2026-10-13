import numpy as np
import pandas as pd

import data
import features as F
from pipeline import likely_score


def _matches():
    rows = [("A", "B", 3, 0), ("B", "C", 1, 1), ("C", "A", 0, 2), ("B", "A", 2, 1), ("A", "B", np.nan, np.nan)]
    return pd.DataFrame({"Div": "E0", "Season": [2020] * 4 + [2021], "Date": pd.date_range("2020-08-01", periods=5, freq="7D"),
                         "HomeTeam": [r[0] for r in rows], "AwayTeam": [r[1] for r in rows],
                         "FTHG": [r[2] for r in rows], "FTAG": [r[3] for r in rows], "Referee": "X"}
                        ).reindex(columns=data.KEEP + ["Season"])


def test_no_leakage_and_upcoming_rows():
    df = F.build(_matches())
    assert np.isnan(df.loc[0, "h_pts_5"])  # first match: no history
    assert df.loc[3, "a_pts_5"] == 3.0  # A before match 4: won 3-0, won 2-0
    assert df.loc[4, "h_pts_5"] == 2.0  # A before upcoming: W, W, L
    assert df.loc[4, "h_v_pts_5"] == 3.0  # A's only previous home match
    assert df.loc[4, "h2h_n"] == 2 and df.loc[4, "h2h_win"] == 0.5 and df.loc[4, "h2h_gd"] == 1.0
    assert df.loc[4, "ref_n"] == 4 and df.loc[4, "ref_hwin"] == 0.5
    assert df.loc[0, "h_elo"] == 1500 and df.loc[4, "h_elo"] > 1500


def test_likely_score_matches_outcome():
    assert likely_score(1.0, 1.0, 0) == "1-0"
    assert likely_score(1.0, 1.0, 1) == "1-1" or likely_score(1.0, 1.0, 1) == "0-0"
    assert likely_score(0.5, 2.5, 2) in {"0-2", "0-1"}


def test_squad_lineups_and_injuries():
    m = _matches()
    xi = {0: range(1, 12), 2: range(1, 12), 3: range(2, 13)}  # team A's starters; player 1 dropped in match 3
    apps = pd.DataFrame([{**m.loc[i, ["Div", "Season", "HomeTeam", "AwayTeam"]].to_dict(), "team": "A",
                          "player": f"fpl:p{k}", "name": f"P{k}", "started": True, "minutes": 90, "rating": k + i}
                         for i, players in xi.items() for k in players])
    out = pd.DataFrame({"team": ["A"], "player": ["fpl:p2"], "name": ["P2"], "reason": ["knee"]})
    forms = pd.DataFrame({**m.loc[3, ["Div", "Season", "HomeTeam", "AwayTeam"]].to_dict(), "team": "A", "formation": "4-2-3-1"}, index=[0])
    df = F.build(m, apps=apps, unavailable=out, formations=forms)
    assert np.isnan(df.loc[0, "h_lu_missing"])  # no previous line-ups yet
    assert df.loc[2, "a_lu_missing"] == 0 and df.loc[3, "a_lu_missing"] == 1
    assert df.loc[3, "a_lu_def"] == 4 and df.loc[3, "a_lu_fwd"] == 1
    assert df.loc[4, "h_lu_missing"] == 1 and df.loc[4, "h_lu_out"] == "P2 (knee)"  # upcoming: regular P2 injured
    assert df.loc[4, "h_lu_def"] == 4 and not np.isnan(df.loc[4, "h_lu_rating"])
    assert np.isnan(df.loc[4, "a_lu_missing"])  # no line-up data for B


def test_manager_record():
    coaches = pd.DataFrame({"team": ["A", "B"], "coach": [1, 2], "coach_name": ["Ann", "Bob"],
                            "start": ["2020-01-01", "2020-08-10"], "end": [None, None]})
    df = F.build(_matches(), coaches=coaches)
    assert df.loc[0, "h_mgr_games"] == 0 and np.isnan(df.loc[0, "h_mgr_ppg"]) and df.loc[0, "h_coach_name"] == "Ann"
    assert np.isnan(df.loc[0, "a_mgr_games"]) and np.isnan(df.loc[1, "h_mgr_games"])  # B's coach not yet appointed
    assert df.loc[4, "h_mgr_games"] == 3 and df.loc[4, "h_mgr_ppg"] == 2.0  # A: W, W, L under Ann
    assert df.loc[4, "a_mgr_games"] == 1 and df.loc[4, "a_mgr_ppg"] == 3.0
    assert df.loc[4, "a_mgr_days"] == (pd.Timestamp("2020-08-29") - pd.Timestamp("2020-08-10")).days