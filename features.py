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

LINEUP = ["lu_missing", "lu_rating", "lu_def", "lu_fwd"]
MANAGER = ["mgr_games", "mgr_ppg", "mgr_team_games", "mgr_days"]
TEAM_FEATS = ["elo", "rest", "played", *[f"{c}_{w}" for w in WINDOWS for c in FORM], "v_pts_5", "v_gf_5", "v_ga_5",
              *LINEUP, *MANAGER]
RECENT_DAYS = 45  # line-up data older than this says nothing about the next match
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


def managers(t, coaches):
    """Coach on match day (from career records) and his record in earlier matches of this dataset."""
    if coaches is None or coaches.empty:
        return t.assign(coach_name=None, **{c: np.nan for c in MANAGER})
    c = coaches.assign(mstart=pd.to_datetime(coaches.start, errors="coerce"),
                       mend=pd.to_datetime(coaches.end, errors="coerce")).dropna(subset=["mstart"])
    m = pd.merge_asof(t.reset_index().sort_values("date", kind="stable"),
                      c[["team", "coach", "coach_name", "mstart", "mend"]].sort_values("mstart"),
                      left_on="date", right_on="mstart", by="team", direction="backward")
    m.loc[m.mend < m.date, ["coach", "coach_name", "mstart"]] = np.nan  # last job here had already ended
    m = m.set_index("index").sort_values(["date", "mid"], kind="stable")
    has = m.coach.notna()
    m["mgr_games"] = m.groupby("coach").cumcount().where(has)
    m["mgr_ppg"] = m.groupby("coach")["pts"].transform(lambda s: s.shift(1).expanding().mean())
    m["mgr_team_games"] = m.groupby(["coach", "team"]).cumcount().where(has)
    m["mgr_days"] = (m.date - m.mstart).dt.days
    return m.drop(columns=["mend"]).sort_index()


def squad(df, apps, unavailable=None, formations=None):
    """Line-up features per side.
    lu_missing: how many of the team's 11 most-used starters (previous 10 matches) are not in the XI.
    lu_rating:  mean form of the XI (player rating z-scored per source, averaged over his previous 5 games).
    lu_def / lu_fwd: defenders / forwards in the formation.
    Upcoming matches have no XI yet: the regulars minus injured/suspended players stand in for it, and
    lu_out lists who is missing."""
    cols = [f"{p}{c}" for p in ("h_", "a_") for c in [*LINEUP, "lu_out"]]
    out = pd.DataFrame(index=df.index, columns=cols, dtype=object)
    if apps is None or apps.empty:
        return out.astype({c: float for c in cols if not c.endswith("lu_out")})
    keys = ["Div", "Season", "HomeTeam", "AwayTeam"]
    mid = pd.Series(df.index, index=pd.MultiIndex.from_frame(df[keys])).groupby(level=list(range(4))).first()

    def with_mid(x):
        x = x.assign(mid=mid.reindex(pd.MultiIndex.from_frame(x[keys])).to_numpy()).dropna(subset=["mid"])
        return x.assign(mid=x.mid.astype(int), date=df.Date.to_numpy()[x.mid.astype(int)])

    a = with_mid(apps)
    src = a.player.str.split(":").str[0]
    a["z"] = a.groupby(src).rating.transform(lambda r: (r - r.mean()) / r.std())
    a = a.sort_values("date", kind="stable")
    a["form"] = a.groupby("player").z.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    now_form = a.groupby("player").z.apply(lambda s: s.tail(5).mean())
    starters = a[a.started].groupby(["team", "mid"])
    xi_players, xi_rating = starters.player.agg(frozenset).to_dict(), starters.form.mean().to_dict()
    form_of = {}
    if formations is not None and not formations.empty:
        f = with_mid(formations).dropna(subset=["formation"])
        parts = f.formation.str.split("-")
        f = f.assign(d=pd.to_numeric(parts.str[0], errors="coerce"), w=pd.to_numeric(parts.str[-1], errors="coerce"))
        form_of = {(t, m): (d, w) for t, m, d, w in zip(f.team, f.mid, f.d, f.w)}
    gone = {}
    if unavailable is not None:
        for t, p, n, r in unavailable[["team", "player", "name", "reason"]].itertuples(index=False):
            gone.setdefault(t, {})[p] = f"{n} ({r})"

    sides = pd.concat([pd.DataFrame({"mid": df.index, "team": df.HomeTeam, "p": "h_", "date": df.Date, "played": df.FTHG.notna()}),
                       pd.DataFrame({"mid": df.index, "team": df.AwayTeam, "p": "a_", "date": df.Date, "played": df.FTHG.notna()})])
    rows = {}
    for team, g in sides.sort_values(["date", "mid"], kind="stable").groupby("team", sort=False):
        hist, last_form = [], None  # hist: (date, XI) of matches with line-up data
        for m, p, d, played in g[["mid", "p", "date", "played"]].itertuples(index=False):
            counts = {}
            for _, players in hist[-10:]:
                for pl in players:
                    counts[pl] = counts.get(pl, 0) + 1
            regulars = sorted(counts, key=counts.get, reverse=True)[:11]
            players = xi_players.get((team, m))
            if players is not None:
                rows[(m, p)] = (len(set(regulars) - players) if regulars else np.nan, xi_rating[(team, m)],
                                *form_of.get((team, m), (np.nan, np.nan)), None)
                hist.append((d, players))
                last_form = form_of.get((team, m), last_form)
            elif not played and regulars and (d - hist[-1][0]).days <= RECENT_DAYS:
                missing = [gone[team][pl] for pl in regulars if pl in gone.get(team, {})]
                avail = [now_form.get(pl, np.nan) for pl in regulars if pl not in gone.get(team, {})]
                rows[(m, p)] = (len(missing), np.nanmean(avail) if avail else np.nan,
                                *(last_form or (np.nan, np.nan)), "; ".join(missing))
    for (m, p), vals in rows.items():
        out.loc[m, [f"{p}{c}" for c in [*LINEUP, "lu_out"]]] = vals
    return out.astype({c: float for c in cols if not c.endswith("lu_out")})


def build(matches, weather=None, apps=None, unavailable=None, formations=None, coaches=None):
    """matches: played + upcoming (NaN goals) rows. Returns one row per match with FEATURES and targets."""
    df = matches.sort_values(["Date", "Div"], kind="stable").reset_index(drop=True)
    df["league"] = df.Div.astype("category")
    df["h_elo"], df["a_elo"] = elo(df)
    df["elo_diff"] = df.h_elo - df.a_elo
    t = managers(team_form(df), coaches)
    cols = [c for c in TEAM_FEATS if c != "elo" and c not in LINEUP] + ["coach_name"]
    for venue, p in (("H", "h_"), ("A", "a_")):
        side = t[t.venue == venue].set_index("mid")[cols].add_prefix(p)
        df = df.join(side)
    df = df.join(head_to_head(df)).join(referee(df)).join(squad(df, apps, unavailable, formations))
    if weather is not None and len(weather):
        df = df.merge(weather, on=["HomeTeam", "Date"], how="left")
    for c in FEATURES:
        if c not in df:
            df[c] = np.nan
    df["result"] = np.select([df.FTHG > df.FTAG, df.FTHG == df.FTAG], [0, 1], 2)  # 0=H 1=D 2=A
    return df
