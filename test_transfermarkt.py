import numpy as np
import pandas as pd

import transfermarkt


def test_market_values_match_players_across_sources():
    tm = {"apps": pd.DataFrame({"team": ["Real Madrid", "Real Madrid", "Getafe"], "player": ["p:vini", "p:danilo", "p:danilo"],
                                "tm_id": [1, 2, 3]}),
          "values": pd.DataFrame({"player_id": [1, 1, 2, 3], "date": ["2025-01-01", "2026-06-01", "2025-01-01", "2025-01-01"],
                                  "market_value_in_eur": [100e6, 150e6, 5e6, 1e6]})}
    apps = pd.DataFrame({"team": ["Real Madrid", "Real Madrid", "Getafe", "Arsenal", "Arsenal"],
                         "player": ["p:vini", "p:vini", "p:danilo", "fpl:vini", "fpl:danilo"],
                         "Date": pd.to_datetime(["2026-01-01", "2026-09-01", "2026-09-01", "2026-09-01", "2026-09-01"])})
    v = transfermarkt.market_values(apps, tm)["value"].tolist()
    assert v[:4] == [100e6, 150e6, 1e6, 150e6]  # as of match date; club + name first, then a unique name
    assert np.isnan(v[4])  # "danilo" is ambiguous outside his known clubs
