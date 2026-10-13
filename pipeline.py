"""Download -> features -> backtest -> train on all history -> predict upcoming. Run: python pipeline.py"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss

import apifootball
import data
import features as F
import fpl
import sofascore
import transfermarkt
import understat


def classifier():
    return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=400, max_leaf_nodes=15, min_samples_leaf=80,
                                          l2_regularization=1.0, categorical_features="from_dtype", random_state=0)


def goals_model():
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
                                         min_samples_leaf=80, categorical_features="from_dtype", random_state=0)


def likely_score(lh, la, outcome):
    """Most likely scoreline among those that agree with the predicted outcome (0=H, 1=D, 2=A)."""
    g = np.arange(10)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    ok = [np.greater, np.equal, np.less][outcome](g[:, None], g[None, :])
    i, j = np.unravel_index(np.where(ok, m, -1).argmax(), m.shape)
    return f"{i}-{j}"


def backtest(played, season):
    """Train on seasons before `season`, test on `season` onward. Compare with bookmaker implied probabilities."""
    tr, te = played[played.Season < season], played[played.Season >= season]
    te = te[te[F.MARKET].notna().all(axis=1)]
    out = {"test_matches": len(te), "test_from_season": f"{season}/{season + 1}"}
    no_squad = [c for c in F.FEATURES if not any(c.endswith(s) for s in F.LINEUP + F.MANAGER)]
    for name, cols in (("model", F.FEATURES), ("model_no_odds", F.NO_MARKET), ("model_no_squad_manager", no_squad)):
        p = classifier().fit(tr[cols], tr.result).predict_proba(te[cols])
        out[name] = {"log_loss": log_loss(te.result, p), "accuracy": accuracy_score(te.result, p.argmax(1))}
    p = te[F.MARKET].to_numpy()
    out["bookmakers"] = {"log_loss": log_loss(te.result, p), "accuracy": accuracy_score(te.result, p.argmax(1))}
    out["always_home"] = {"accuracy": float((te.result == 0).mean())}
    return out


def run():
    results = data.load_results()
    upcoming, stadium = data.load_upcoming(results)
    print(f"{len(results)} played matches, {len(upcoming)} upcoming")
    weather = data.load_weather(data.load_stadiums(stadium))
    try:
        epl_apps, epl_out = fpl.load(results)
    except requests.RequestException as e:  # player data is optional: the model handles missing features
        print(f"FPL: {e}")
        epl_apps = epl_out = None
    af, ss = apifootball.load(results), sofascore.load(results)
    tm, us = transfermarkt.load(results), understat.load(results)
    keys = ["Div", "Season", "HomeTeam", "AwayTeam"]
    # first non-empty value wins: xG from Understat (Top 5), then Sofascore; referees from football-data (England),
    # then Sofascore, then Transfermarkt
    extra = pd.concat([us, ss["matches"], tm["matches"]]).groupby(keys, sort=False).first().reset_index()
    results = results.merge(extra, on=keys, how="left", suffixes=("", "_x"))
    results["Referee"] = results.Referee.fillna(results.pop("Referee_x"))
    # Sofascore and Transfermarkt key players by name, so their line-ups form one history per team (Transfermarkt up
    # to last season, Sofascore since; Sofascore wins where both have a match, it has ratings). The Premier League
    # uses FPL, which matches its injury list; API-Football only fills leagues neither covers.
    tm_apps = tm["apps"][(tm["apps"].Div != "E0")
                         & ~pd.MultiIndex.from_frame(tm["apps"][keys]).isin(pd.MultiIndex.from_frame(ss["apps"][keys]))]
    named = pd.concat([ss["apps"], tm_apps])
    apps = pd.concat([epl_apps, named, af["apps"][~af["apps"].Div.isin(named.Div)]], ignore_index=True)
    apps = transfermarkt.market_values(apps.merge(results[[*keys, "Date"]], on=keys), tm)
    unavailable = pd.concat([epl_out, ss["unavailable"], af["unavailable"][~af["unavailable"].team.isin(named.team)]],
                            ignore_index=True)
    formations = pd.concat([ss["formations"], tm["formations"], af["formations"]]).dropna(subset=["formation"])
    formations = formations.drop_duplicates([*keys, "team"])
    managers = pd.concat([ss["managers"], tm["managers"]]).assign(date=lambda d: pd.to_datetime(d.date))
    coaches = data.manager_stints(managers.drop_duplicates(["team", "date"]))
    # API-Football careers only fill in before a club's first match with a known manager
    first = coaches.groupby("team").start.min()
    old = pd.to_datetime(af["coaches"].start, errors="coerce") < af["coaches"].team.map(first)
    coaches = pd.concat([coaches, af["coaches"][old | ~af["coaches"].team.isin(first.index)]], ignore_index=True)
    df = F.build(pd.concat([results, upcoming], ignore_index=True), weather,
                 apps, unavailable, formations, coaches)
    played, future = df[df.FTHG.notna()], df[df.FTHG.isna()].copy()

    metrics = backtest(played, data.current_season() - 1)
    recent = played[played.Season >= data.current_season() - 1]
    metrics["coverage"] = {data.LEAGUES[d][0]: {"line-ups": float(g.h_lu_missing.notna().mean()),
                                                "player ratings": float(g.h_lu_rating.notna().mean()),
                                                "market values": float(g.h_lu_value.notna().mean()),
                                                "managers": float(g.h_mgr_games.notna().mean()),
                                                "xG": float(g.HxG.notna().mean()),
                                                "referees": float(g.ref_n.notna().mean())}
                           for d, g in recent.groupby("Div")}
    metrics["api_football"] = af["status"]
    metrics["sofascore"] = ss["status"]
    print(json.dumps(metrics, indent=1))

    # two models: one using bookmaker odds, one without for fixtures whose odds aren't published yet
    has_odds = future[F.MARKET].notna().all(axis=1)
    for cols, rows in ((F.FEATURES, has_odds), (F.NO_MARKET, ~has_odds)):
        if rows.any():
            future.loc[rows, ["pH", "pD", "pA"]] = classifier().fit(played[cols], played.result).predict_proba(
                future.loc[rows, cols])
    future["xg_h"] = goals_model().fit(played[F.NO_MARKET], played.FTHG).predict(future[F.NO_MARKET])
    future["xg_a"] = goals_model().fit(played[F.NO_MARKET], played.FTAG).predict(future[F.NO_MARKET])
    future["pick"] = future[["pH", "pD", "pA"]].to_numpy().argmax(1)
    future["score"] = [likely_score(h, a, o) for h, a, o in zip(future.xg_h, future.xg_a, future.pick)]

    metrics["trained_at"] = datetime.now().isoformat(timespec="minutes")
    metrics["trained_on"] = len(played)
    data.DATA.mkdir(exist_ok=True)
    future.to_pickle(data.DATA / "predictions.pkl")
    played.to_pickle(data.DATA / "history.pkl")
    unavailable.to_pickle(data.DATA / "unavailable.pkl")
    (data.DATA / "metrics.json").write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    run()
