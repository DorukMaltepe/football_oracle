"""Download -> features -> backtest -> train on all history -> predict upcoming. Run: python pipeline.py"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss

import data
import features as F


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
    for name, cols in (("model", F.FEATURES), ("model_no_odds", F.NO_MARKET)):
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
    df = F.build(pd.concat([results, upcoming], ignore_index=True), weather)
    played, future = df[df.FTHG.notna()], df[df.FTHG.isna()].copy()

    metrics = backtest(played, data.current_season() - 1)
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
    (data.DATA / "metrics.json").write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    run()
