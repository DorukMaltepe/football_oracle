import numpy as np
import pandas as pd

import data
import features as F
from pipeline import likely_score


def _matches():
    rows = [("A", "B", 3, 0), ("B", "C", 1, 1), ("C", "A", 0, 2), ("B", "A", 2, 1), ("A", "B", np.nan, np.nan)]
    return pd.DataFrame({"Div": "E0", "Season": 2020, "Date": pd.date_range("2020-08-01", periods=5, freq="7D"),
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
