"""Pre-match features. Every feature for a match uses only matches strictly before it (shift(1)),
so played and upcoming fixtures go through the same code without leakage."""
import numpy as np
import pandas as pd

# stat -> (column seen from home team, column seen from away team)
STATS = {"gf": ("FTHG", "FTAG"), "ga": ("FTAG", "FTHG"), "sf": ("HS", "AS"), "sa": ("AS", "HS"),
         "stf": ("HST", "AST"), "sta": ("AST", "HST"), "cf": ("HC", "AC"), "ca": ("AC", "HC"),
         "fo": ("HF", "AF"), "yc": ("HY", "AY"), "rc": ("HR", "AR")}
FORM = ["pts", *STATS]
WINDOWS = (5, 10)

TEAM_FEATS = ["elo", "rest", "played", *[f"{c}_{w}" for w in WINDOWS for c in FORM], "v_pts_5", "v_gf_5", "v_ga_5"]
MARKET = ["mkt_H", "mkt_D", "mkt_A"]
FEATURES = ["league", "elo_diff", *[f"{s}_{f}" for s in "ha" for f in TEAM_FEATS],
            "h2h_n", "h2h_win", "h2h_draw", "h2h_gd", "ref_n", "ref_hwin", "ref_cards",
            "temp", "rain", "wind", *MARKET]
NO_MARKET = [f for f in FEATURES if f not in MARKET]


def elo(df, k=20, home_adv=60):
    """Pre-match Elo for home and away. Promoted teams (first seen after season one) start below average."""
    first = df.Season.min()
    r, eh, ea = {}, [], []
    for h, a, hg, ag, season in zip(df.HomeTeam, df.AwayTeam, df.FTHG, df.FTAG, df.Season):
        start = 1500 if season == first else 1400
        rh, ra = r.setdefault(h, start), r.setdefault(a, start)
        eh.append(rh)
        ea.append(ra)
        if np.isnan(hg):
            continue
        exp = 1 / (1 + 10 ** ((ra - rh - home_adv) / 400))
        d = k * (np.sign(hg - ag) / 2 + 0.5 - exp) * np.sqrt(1 + abs(hg - ag))
        r[h], r[a] = rh + d, ra - d
    return np.array(eh), np.array(ea)


def _rolling(df, keys, cols, w):
    prev = df.groupby(keys)[cols].shift(1)
    prev[keys] = df[keys]
    out = prev.groupby(keys)[cols].rolling(w, min_periods=1).mean()
    return out.reset_index(level=list(range(len(keys))), drop=True)


def team_form(df):
    sides = []
    for venue, i in (("H", 0), ("A", 1)):
        side = pd.DataFrame({"mid": df.index, "team": df.HomeTeam if i == 0 else df.AwayTeam,
                             "date": df.Date, "season": df.Season, "venue": venue,
                             **{k: df[cols[i]] for k, cols in STATS.items()}})
        sides.append(side)
    t = pd.concat(sides, ignore_index=True).sort_values(["date", "mid"], kind="stable").reset_index(drop=True)
    t["pts"] = np.where(t.gf > t.ga, 3, np.where(t.gf == t.ga, 1, 0)).astype(float)
    t.loc[t.gf.isna(), "pts"] = np.nan
    for w in WINDOWS:
        t = t.join(_rolling(t, ["team"], FORM, w).add_suffix(f"_{w}"))
    t = t.join(_rolling(t, ["team", "venue"], ["pts", "gf", "ga"], 5).add_prefix("v_").add_suffix("_5"))
    t["rest"] = t.groupby("team")["date"].diff().dt.days.clip(upper=30)
    t["played"] = t.groupby(["team", "season"]).cumcount()
    return t


def head_to_head(df, n=6):
    first_home = df.HomeTeam < df.AwayTeam
    key = np.where(first_home, df.HomeTeam + "|" + df.AwayTeam, df.AwayTeam + "|" + df.HomeTeam)
    gd = (df.FTHG - df.FTAG) * np.where(first_home, 1, -1)  # from the alphabetically-first team's side
    h = pd.DataFrame({"key": key, "gd": gd, "win": (gd > 0).astype(float), "draw": (gd == 0).astype(float),
                      "loss": (gd < 0).astype(float)}, index=df.index)
    h.loc[gd.isna(), ["win", "draw", "loss"]] = np.nan
    prev = h.groupby("key")[["gd", "win", "draw", "loss"]].shift(1)
    prev["key"] = key
    roll = prev.groupby("key").rolling(n, min_periods=1)
    m = roll[["gd", "win", "draw", "loss"]].mean().reset_index(level=0, drop=True).sort_index()
    cnt = roll["gd"].count().reset_index(level=0, drop=True).sort_index()
    sign = np.where(first_home, 1, -1)
    return pd.DataFrame({"h2h_n": cnt, "h2h_win": np.where(first_home, m.win, m.loss),
                         "h2h_draw": m.draw, "h2h_gd": m.gd * sign}, index=df.index)


def referee(df):
    r = pd.DataFrame({"ref": df.Referee, "hwin": (df.FTHG > df.FTAG).astype(float),
                      "cards": df[["HY", "AY"]].sum(axis=1, min_count=1) + 2 * df[["HR", "AR"]].sum(axis=1, min_count=1)},
                     index=df.index)
    r.loc[df.FTHG.isna(), "hwin"] = np.nan
    prev = r.groupby("ref")[["hwin", "cards"]].shift(1)
    prev["ref"] = r.ref
    g = prev.groupby("ref")
    m = g[["hwin", "cards"]].expanding().mean().reset_index(level=0, drop=True)
    n = g["hwin"].expanding().count().reset_index(level=0, drop=True)
    return pd.DataFrame({"ref_n": n, "ref_hwin": m.hwin, "ref_cards": m.cards}).reindex(df.index)


def build(matches, weather=None):
    """matches: played + upcoming (NaN goals) rows. Returns one row per match with FEATURES and targets."""
    df = matches.sort_values(["Date", "Div"], kind="stable").reset_index(drop=True)
    df["league"] = df.Div.astype("category")
    df["h_elo"], df["a_elo"] = elo(df)
    df["elo_diff"] = df.h_elo - df.a_elo
    t = team_form(df)
    cols = [c for c in TEAM_FEATS if c != "elo"]
    for venue, p in (("H", "h_"), ("A", "a_")):
        side = t[t.venue == venue].set_index("mid")[cols].add_prefix(p)
        df = df.join(side)
    df = df.join(head_to_head(df)).join(referee(df))
    if weather is not None and len(weather):
        df = df.merge(weather, on=["HomeTeam", "Date"], how="left")
    for c in FEATURES:
        if c not in df:
            df[c] = np.nan
    df["result"] = np.select([df.FTHG > df.FTAG, df.FTHG == df.FTAG], [0, 1], 2)  # 0=H 1=D 2=A
    return df
