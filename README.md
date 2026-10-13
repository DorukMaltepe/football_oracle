# Football Oracle

Predicts football match results for seven European leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie and Süper Lig. It uses machine learning trained on every league match since 2015/16.

For each upcoming fixture it gives home / draw / away probabilities and the most likely score, and a local web UI shows the reasoning behind each prediction: form, head-to-head, referee, weather, odds and the latest news.

## Purpose

This is a personal, educational project. It explores how far a match predictor can get using **only free, public data with no API keys**, so anyone can clone it and run it.

It's built to be honest about its accuracy. Each training run is checked against the bookmakers' odds on matches the model has never seen, and the result is shown in the UI. It is not betting advice.

## Quick start

Requires Python 3.12+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
# macOS / Linux
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The first launch downloads about 11 seasons of data, geocodes the stadiums and trains the models. This takes several minutes. After that, click **Refresh data & retrain** in the sidebar (or run `python pipeline.py`) to pull new results and fixtures. Run the tests with `pytest`.

## What you get

- **Fixture list:** every upcoming fixture in the chosen leagues and date range, with model probabilities, predicted winner, most likely score, the bookmakers' implied probability, and the model-minus-market *edge*.
- **Match view** (click a row):
  - probabilities against the market, and expected goals
  - side-by-side team form: Elo, points, goals, shots, corners, cards, home or away form, rest days
  - head-to-head record, recent results, referee tendencies, weather forecast and odds
  - latest news for both teams, with injury, suspension and manager stories flagged ⚠️
- **Model accuracy:** the backtest table described below.

## Architecture

```mermaid
flowchart LR
    subgraph Sources["Free data sources"]
        FD["football-data.co.uk<br/>results, match stats,<br/>referees, odds, near-term fixtures"]
        FX["fixturedownload.com<br/>full season schedule, stadiums"]
        OSM["OpenStreetMap Nominatim<br/>stadium coordinates"]
        OM["Open-Meteo<br/>historical and forecast weather"]
        GN["Google News RSS<br/>team headlines"]
    end

    subgraph Pipeline["pipeline.py (offline, on refresh)"]
        D["data.py<br/>download and cache,<br/>team-name mapping,<br/>stadiums, weather"]
        F["features.py<br/>leak-free pre-match features"]
        B["Backtest<br/>time-based split vs bookmakers"]
        T["Train<br/>2 classifiers for H/D/A<br/>2 Poisson goal regressors"]
        P["Predict upcoming fixtures"]
    end

    subgraph Store["data/"]
        C[("cache/<br/>raw downloads")]
        S[("stadiums.csv<br/>editable")]
        A[("predictions.pkl<br/>history.pkl<br/>metrics.json")]
    end

    UI["app.py<br/>Streamlit UI"]

    FD --> D
    FX --> D
    OSM --> D
    OM --> D
    D <--> C
    D <--> S
    D --> F --> B
    F --> T --> P
    B --> A
    P --> A
    A --> UI
    GN -- "live, cached 1h" --> UI
    UI -- "Refresh button" --> D
```

The work happens offline in `pipeline.py`. The UI reads only the saved predictions and history, so it loads instantly; the only thing it fetches live is news. The data flow:

1. **`data.py`** downloads results and odds for every season and league, plus the near-term fixtures with odds (`fixtures.csv`).
2. It also downloads the full season schedule from fixturedownload.com and maps its team names onto football-data's naming.
3. It geocodes each home stadium once, then fetches daily weather at each one.
4. **`features.py`** adds the upcoming fixtures, with blank results, to the played matches and computes every feature from earlier matches only.
5. **`pipeline.py`** backtests on the most recent seasons, retrains on all history and writes predictions for every unplayed fixture.
6. **`app.py`** displays them.

## Features used by the model

| Factor | Feature | Source |
|---|---|---|
| Club strength | Elo rating, updated after every match and weighted by goal margin | derived from results |
| Recent form | Points, goals, shots, shots on target, corners, fouls and cards, averaged over the last 5 and 10 matches | football-data.co.uk |
| Home / away form | Each team's last 5 matches at the same venue type (home team at home, away team away) | football-data.co.uk |
| Head-to-head | Last 6 league meetings: win rate, draw rate, average goal difference | football-data.co.uk |
| Referee | Home-win rate, cards per game, number of matches refereed | football-data.co.uk (England only) |
| Betting market | Average bookmaker odds converted to probabilities, with the bookmaker margin removed | football-data.co.uk |
| Weather | Max temperature, precipitation and max wind at the home stadium on match day | Open-Meteo |
| Fatigue / season stage | Days since last match, league games played this season | derived |
| League | League code, so each league can behave differently | |

### Not in the model

| Factor | Why not | Partial stand-in |
|---|---|---|
| Player form, line-ups, formations, injuries | No free historical source; needs a paid API such as API-Football or Sportmonks | Bookmaker odds already price these in |
| Manager records | No free historical source | Team form and Elo |
| News | No historical news archive to train on | Shown in the UI, keywords flagged |
| Pitch quality | No dataset exists | Each team's home form |

## Design decisions

**Free, keyless data only.** Anyone can clone the repo and run it without signing up for anything. The cost is the missing player-level data listed above. It's the first thing to add if a paid feed becomes available.

**Gradient-boosted trees (scikit-learn `HistGradientBoosting*`).** The data is about 27k rows of mixed numbers, a good fit for boosted trees. They handle missing values natively, which matters here. Referees are only published for England, shot statistics are missing for the 2015/16 and 2016/17 Eredivisie and Süper Lig seasons, and weather is unavailable for some stadiums. Missing values go straight to the model; nothing is filled in. The league is used as a native categorical feature.

**One feature pipeline for past and future matches.** Upcoming fixtures are added to the match history with blank results, and every feature is computed with `shift(1)`. A match's features therefore come only from earlier matches, whether it was played years ago or kicks off tomorrow. This prevents the most common bug in sports models, leaking the result into the features. It also guarantees training and prediction use identical code. `test_features.py` asserts the no-leakage property.

**Bookmaker odds are both a feature and the yardstick.** The betting market is the strongest single predictor available, so the model uses it as a feature. Accuracy is also reported against it: matching the market is a good result, and beating it consistently is rare.

**Two classifiers: with and without odds.** Bookmakers publish odds only a few days ahead. Fixtures further out are scored by a second model trained without the odds features. The UI notes this, and the backtest reports both models.

**Time-based backtest, not a random split.** The models train on seasons before last season and are tested on last season plus the current one. A random split would let the model learn from matches played after the ones it is tested on.

**Scoreline from Poisson goal models, constrained to the predicted outcome.** Two Poisson regressors predict expected goals for each side. The displayed score is the most likely scoreline among those that agree with the predicted winner. This avoids showing "Home win, 1-1".

**Elo rating with a goal-margin multiplier.** Elo is a cheap, proven measure of team strength (K = 20, home advantage 60 points). Promoted teams start at 1400 rather than 1500, because newly promoted sides are usually weaker than average.

**Flat files instead of a database or server.** This is a single-user local tool, so cached CSVs plus pickled outputs are enough. Streamlit provides the UI with no front-end build step.

**Team-name mapping: fuzzy matching plus aliases.** The two fixture sources spell clubs differently ("Manchester United" vs "Man United"). Names are fuzzy-matched within each league, and an `ALIASES` table in `data.py` overrides the known failures. If two names map to the same club, a warning tells you which alias to add.

**Stadiums are geocoded once and stored in an editable CSV.** Lookups are limited to the league's country, which prevents results such as a "Johan Cruyff Arena" in Malta. Manual corrections in `data/stadiums.csv` are never overwritten.

**Weather history fills in gradually.** Open-Meteo's free quota counts an 11-year daily history as many calls. When the limit is reached, the pipeline pauses history downloads and still fetches forecasts. Each refresh completes more stadiums. Until then, those matches have no weather data, and the model handles that like any other missing value.

## Results

Backtest from 13 Oct 2026: trained on 2015/16–2024/25, tested on 2,796 matches from 2025/26 and 2026/27 that have odds.

| | Accuracy | Log loss (lower is better) |
|---|---|---|
| Model (with odds) | 53.0% | 0.980 |
| Model (no odds) | 52.1% | 0.991 |
| Bookmakers | 53.3% | 0.977 |
| Always pick home win | 43.6% | – |

The model is roughly level with the bookmakers. That is typical: football has a lot of randomness, and the market already includes information this model can't see, such as line-ups and injuries. Each retrain refreshes these numbers in the UI.

## Project layout

```
app.py             Streamlit UI
pipeline.py        backtest, training, prediction (entry point: python pipeline.py)
features.py        pre-match feature engineering
data.py            data sources, caching, team-name mapping, stadiums, weather
test_features.py   no-leakage and scoreline tests
data/stadiums.csv  geocoded stadiums (committed, hand-editable)
data/cache/        raw downloads (git-ignored)
```

**Adding a league:** add a line to `LEAGUES` in `data.py`. It needs the football-data.co.uk code, a display name, the fixturedownload.com slug and a country code, for example `"P1": ("Primeira Liga", "primeira-liga", "pt")`. Then retrain.

## Notes

- `scikit-learn` is pinned to 1.8.0 because Windows Smart App Control blocks a compiled file in 1.9.1.
- Data comes from [football-data.co.uk](https://www.football-data.co.uk), [fixturedownload.com](https://fixturedownload.com), [OpenStreetMap](https://www.openstreetmap.org/copyright), [Open-Meteo](https://open-meteo.com) and Google News. Please respect their terms and rate limits.
- Predictions are probabilities, not certainties. Don't use them for gambling decisions.
