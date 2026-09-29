# Premier League Match Outcome Predictor

A reproducible machine-learning pipeline that predicts **Home Win / Draw / Away Win** probabilities for English Premier League matches, using only information that was available before kickoff. It compares a **Random Forest** and an **XGBoost** classifier against simple baselines, evaluated chronologically on seasons the models never saw.

> **Status: Phase 3, plus model improvements.** Working so far: the training pipeline (Phase 1); live predictions, the prediction history and the live tracker (Phase 2); the bookmaker-odds benchmark, betting simulation, EDA and final report (Phase 3); and a round of improvements scored on past seasons only: Elo ratings, weighted form, 20 more seasons of history, a draw rule and stricter betting rules. The Streamlit dashboard (Phase 4) is next.
>
> 📄 **[Read the final report](reports/final_report.md)**, a full walkthrough from data collection to simulated betting.

## Problem statement

Given a fixture such as *Liverpool vs Arsenal*, estimate the probability of each outcome. This is a 3-class classification problem with the target encoded as:

| Value | Meaning  |
|------:|----------|
| 0     | Away Win |
| 1     | Draw     |
| 2     | Home Win |

The model trains on **all** Premier League matches in the training period, not just previous meetings of the two clubs. For each historical match, the features describe both teams **as they were before that match kicked off**.

## Architecture

```
Football-Data.co.uk CSVs ─► src/data/collect.py      download + cache in data/raw/
                          ─► src/data/clean.py        normalize names, dates, target; validate
                          ─► src/features/build_features.py   leakage-safe pre-match features (+ elo.py)
                          ─► src/models/split.py      season-based train / validation / test + walk-forward folds
                          ─► src/models/train_random_forest.py, train_xgboost.py   tuning + fitting
                          ─► src/models/evaluate.py   metrics, baselines, calibration, importance, plots
                          ─► src/models/decision.py   draw rule for the predicted outcome
                          ─► models/  +  reports/model_comparison.md
```

`src/pipeline.py` runs every step end to end.

Live predictions (Phase 2):

```
openfootball fixtures + results ─► src/data/current_season.py   season in progress (Football-Data file preferred when reachable)
historical seasons + current     ─► src/data/update.py           one clean table of every finished match
saved models + that table        ─► src/models/predict.py        fixture features (same code as training) -> RF and XGBoost probabilities
                                 ─► src/tracking/history.py      data/predictions/prediction_history.csv (locked at kickoff)
                                 ─► src/tracking/performance.py  live accuracy, per-class hit rates, accuracy over time
src/live.py: predict | update | report | match
```

Analysis and report (Phase 3):

```
Bet365 odds (football-data.co.uk) ─► src/data/odds.py              join to matches, dates cross-checked
out-of-sample predictions         ─► src/models/benchmark.py       bookmaker implied probabilities scored like a model
                                  ─► src/models/betting.py         $100 flat-stake simulation, ROI with bootstrap CI
                                  ─► src/models/feature_selection.py  walk-forward permutation-importance selection
                                  ─► src/analysis/eda.py           charts
                                  ─► src/final_report.py           reports/final_report.md
src/experiments.py                ─► reports/experiments.md        every candidate improvement scored on walk-forward seasons
src/analysis/in_play_reference.py ─► reports/in_play_reference.json   why some projects report ~70% (half-time inputs)
```

```
├── config.yaml                 seasons, sources, feature windows, split, tuning budget
├── data/raw/                   cached season CSVs (committed so runs are reproducible)
├── data/processed/             matches.csv, features.csv (generated)
├── data/predictions/           prediction_history.csv (the live forward test)
├── models/                     random_forest.pkl, xgboost.json, model_metadata.json
├── reports/                    final_report.md, model_comparison.md, metrics.json, figures/
├── src/
│   ├── analysis/               eda.py, in_play_reference.py
│   ├── data/                   collect.py, clean.py, current_season.py, update.py, odds.py
│   ├── features/               build_features.py, elo.py
│   ├── models/                 split.py, tuning.py, train_random_forest.py, train_xgboost.py, evaluate.py,
│   │                           predict.py, decision.py, benchmark.py, betting.py, feature_selection.py
│   ├── tracking/               history.py, performance.py
│   ├── utils/                  config.py, team_names.py, plotting.py
│   ├── pipeline.py             training
│   ├── experiments.py          improvement experiments (walk-forward seasons only)
│   ├── live.py                 live predictions and tracking
│   └── final_report.py         benchmark, betting simulation, EDA -> reports/final_report.md
└── tests/                      leakage, feature, cleaning, split and team-name tests
```

## Data source

**[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)** Premier League season files (`E0.csv`). They are free and stable, with one row per match: date, teams, full-time goals and result, shots, shots on target, corners, fouls and cards.

* **Seasons used:** 2000/01 to 2025/26. 2000/01 to 2004/05 only warm up the rolling features and Elo ratings, so the first training season doesn't start with every club rated the same. Model rows start in 2005/06. Every season has all 380 matches with no missing shot data.
* **Mirror:** the loader tries the official URL first and falls back to the [`datasets/football-datasets`](https://github.com/datasets/football-datasets) GitHub mirror, which republishes the same E0 files. The CSVs in `data/raw/` came from the mirror because the build environment couldn't reach the official site.
* **Team names** are mapped to one canonical spelling in `src/utils/team_names.py` (for example, `Man United`, `Man Utd` and `Manchester Utd` all become `Manchester United`). An unknown spelling raises an error instead of silently creating a new "team".
* **Season in progress:** football-data.co.uk wasn't reachable from the build environment and the mirror doesn't publish a season until it ends. So fixtures, kickoff times and results for 2026/27 come from [openfootball/football.json](https://github.com/openfootball/football.json). I checked it against all 380 matches of 2025/26 and every score and date matched football-data exactly. openfootball has **goals only, no shots**. When the Football-Data current-season file can be downloaded, it is used instead and cross-checked against openfootball. Each saved prediction records whether shot data was available.
* **Not available:** possession and expected goals aren't in these files. Following the project rule, **possession features are excluded rather than imputed**. A second source such as FBref could add them later; `collect.py` is organized so that another source can be plugged in.

## Features

Every feature is computed from matches that finished **before** the match being described. The production models use **14 inputs**:

| Group | Features |
|---|---|
| Elo ratings | `home_elo`, `away_elo`, `elo_diff` |
| Weighted form (per team, home_ and away_ versions) | `ewm_points`, `ewm_goal_diff`, `ewm_sot_for`, `ewm_sot_against` |
| Weighted-form differences (home minus away) | `ewm_points_diff`, `ewm_goal_diff_diff`, `ewm_sot_balance_diff` |

* **Elo** (`src/features/elo.py`) gives each club one strength number that carries across seasons. After every match the winner takes rating points from the loser: more for an upset, more for a bigger margin. The home side gets 60 points of advantage, ratings are pulled 15% back to the mean each summer, and a promoted club starts at the average rating of the three clubs that went down the season before. K, home advantage and the summer pull-back were chosen on training seasons only.
* **Weighted form** is an exponentially weighted average of earlier matches (half-life 5 matches), so the latest game counts most and older ones fade out instead of dropping off a 5-match cliff. It includes shots on target *conceded*, the only defensive shot statistic in the data.

These were picked by `python -m src.experiments`, which scores every candidate on the walk-forward seasons 2019/20–2023/24 only ([reports/experiments.md](reports/experiments.md)):

| Change | Features | RF log loss | XGB log loss |
|---|---|---|---|
| Phase 3 setup (trained from 2015/16) | 42 | 0.9924 | 0.9939 |
| + more history (trained from 2005/06) | 42 | 0.9914 | 0.9903 |
| + Elo ratings | 45 | 0.9791 | 0.9791 |
| + weighted form | 56 | 0.9766 | 0.9769 |
| + match context (rest days, runs, last season's finish, head-to-head) | 66 | 0.9765 | 0.9766 |
| **Elo + weighted form only** | **14** | **0.9757** | **0.9764** |
| + other top-5 leagues in training | 56 | 0.9766 | 0.9760 |
| + bookmaker odds as inputs (can't be used live) | 59 | 0.9621 | 0.9627 |
| *Bookmaker (Bet365, margin removed)* | | *0.9586* | *0.9586* |

Elo ratings made the biggest difference. With Elo and weighted form in, the 42 Phase 3 features (last-5 form, season-to-date stats, league position, venue form and their differences) added nothing, so they were dropped. They are still built, tested and selectable in `config.yaml` (`features.base_features`, `features.groups`), along with the unused `context` group.

Design choices:

* **Home advantage** is Elo's home bonus plus each team's own form, not a constant `home_advantage = 1` column.
* **Promoted teams:** weighted form restarts when a club returns after at least one season away, and its Elo rating starts from the relegated clubs' average rather than its stale old rating.
* **Missing values are left as `NaN`, not filled.** XGBoost and scikit-learn ≥ 1.4 Random Forests route missing values natively, so no arbitrary numbers are invented.
* **No scaling.** Tree models don't need it.
* **League position** for a match on date D (a Phase 3 feature, still built for the report) is rebuilt from that season's matches played on dates strictly before D.

### Leakage prevention

* All rolling, weighted and cumulative statistics are computed after `shift(1)` within each team's own history, so a match never contributes to its own features.
* Elo ratings are read before each match's update is applied. A club's rating only changes after its own matches, and a club plays at most once per date.
* Season-to-date features use a running total minus the current match, grouped by team and season, so they reset each season and never include later games.
* Goals, shots, the result and the target are listed in `POST_MATCH_COLUMNS` and are never in `feature_columns()`.
* `tests/test_features_leakage.py` checks all of this directly:
  * changing a match's score does not change that match's features, but does change the team's next match;
  * rewriting every match after a cutoff date leaves all earlier features identical;
  * last-5 values equal a hand computation from exactly the previous five matches;
  * season features and league positions reset at each season's opening round;
  * for 25 real matches, every feature in every group (Elo, weighted form, context and the Phase 3 set) rebuilt from **only earlier history plus the match as an unplayed fixture** equals the training-table value exactly, which is the same path upcoming-fixture predictions use;
  * `tests/test_elo.py` checks that ratings start equal, only move between the two clubs in a match, and that a promoted club starts from the relegated clubs' ratings.

  I also checked the tests themselves: removing `shift(1)`, or including the current match in season totals, makes four of them fail.

## Models

Only the two models the project is about, plus trivial baselines for context:

* **Random Forest:** `sklearn.ensemble.RandomForestClassifier`, tuning `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` and `class_weight`.
* **XGBoost:** `xgboost.XGBClassifier` with `objective="multi:softprob"` (a probability for each class), tuning `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha` and `reg_lambda`.
* **Baselines:** *Always Home Win*, and *Class frequencies*, which predicts the training outcome rates for every match. Home Win is the most common outcome in every season, so this also stands in for "always predict the most common outcome", and it has a meaningful log loss.

Hyperparameters come from a small random search (15 RF and 30 XGBoost candidates over sensible ranges), scored by **log loss**, because the goal is good probabilities. The random seed is fixed at 42 in `config.yaml`.

## Validation methodology

Matches are never shuffled across time.

1. **Walk-forward tuning** inside the training seasons: for each season N from 2019/20 to 2023/24, train on every earlier season (from 2005/06) and validate on N. Hyperparameters are chosen by mean log loss across these five folds. The feature set, the draw rule and the betting rule were chosen on these seasons too, and the Elo settings on training seasons only.
2. **Validation season 2024/25:** fit on 2005/06–2023/24, compare the models and feature sets.
3. **Test season 2025/26:** refit on 2005/06–2024/25 and evaluate **once**. The test season played no part in any decision. All the test numbers below come from this step.
4. **Production models:** after evaluation, both models are refit with the same hyperparameters on every completed season (2005/06–2025/26) and saved. These are the models that make live predictions, so 2026/27 predictions learn from 2025/26 too.

Metrics: accuracy, macro F1, per-class precision/recall/F1, confusion matrices, log loss, multiclass Brier score, expected calibration error and calibration curves.

## Results

Full reports: [`reports/final_report.md`](reports/final_report.md) and [`reports/model_comparison.md`](reports/model_comparison.md). The raw numbers are in `reports/metrics.json`.

**Test season 2025/26** (380 matches, never used for any decision):

| Model | Accuracy | Macro F1 | Log Loss | Brier |
|---|---|---|---|---|
| Always Home Win | 0.426 | 0.199 | n/a | n/a |
| Class frequencies | 0.426 | 0.199 | 1.082 | 0.655 |
| Random Forest | **0.484** | 0.358 | **1.026** | **0.617** |
| XGBoost | 0.476 | 0.354 | 1.031 | 0.620 |
| Random Forest + draw rule | 0.455 | 0.412 | 1.026 | 0.617 |
| XGBoost + draw rule | 0.463 | **0.423** | 1.031 | 0.620 |
| *Bookmaker (Bet365, margin removed)* | *0.489* | *0.367* | *1.019* | *0.612* |

**Before and after the improvements** (Phase 3 numbers from the Phase 3 commit):

| | Phase 3 | Now | Bookmaker |
|---|---|---|---|
| Test 2025/26, Random Forest log loss | 1.036 | **1.026** | 1.019 |
| Test 2025/26, Random Forest accuracy | 46.6% | **48.4%** | 48.9% |
| Validation 2024/25, Random Forest log loss | 1.006 | **0.989** | 0.971 |
| Validation 2024/25, Random Forest accuracy | **52.9%** | 51.3% | 54.2% |
| Walk-forward 2019/20–2023/24, log loss (Phase 3 hyperparameters) | 0.992 | **0.976** | 0.959 |

Log loss improved in every evaluation block, and the models closed about half of their gap to the bookmaker. Accuracy went up on test and down on validation; with 380 matches a season, a 2-point accuracy swing is within noise, which is why log loss is the main measure.

What this means:

* Both models clearly beat the baselines on every probability metric and on accuracy, in every season tested.
* **The two models are statistically tied.** XGBoost has the lower validation log loss (by 0.0007), so it is the selected model, but the paired bootstrap 95% interval for the per-match log-loss gap contains zero on both validation and test. Live predictions show both.
* **Draws:** with plain argmax neither model ever predicts a draw, and neither does the bookmaker. The draw rule (threshold 0.28 for Random Forest, 0.27 for XGBoost, chosen on walk-forward seasons) predicts a draw in about 20% of matches and catches about a fifth of real draws. Those draw calls are right 26–30% of the time, only a little above the draw rate, and cost up to 3 points of accuracy. Live predictions use the rule; set `decision.draw_rule: false` in `config.yaml` to go back to argmax.
* Probabilities are well calibrated. The most important features, by permutation importance on validation, are the Elo gap, the weighted shots-on-target balance, the home side's Elo and the weighted goal-difference gap.

![Confusion matrices](reports/confusion_matrices_test.png)
![Calibration](reports/calibration_test.png)

### Against the bookmaker, and simulated betting

The strongest benchmark is the betting market itself. Bet365's odds, with the margin removed, still have the lower log loss in all seven out-of-sample seasons, but the gap is smaller than in Phase 3: 0.007 on the test season, down from 0.017.

Staking $100 per bet on the 2025/26 test season:

* backing Random Forest's pick in every match loses $3,952 (−10.4% ROI; Phase 3: −13.1%);
* value betting (model probability × odds − 1 > 5%) makes $768 on 216 bets (+3.6%, 95% CI −18.1% to +25.8%; Phase 3: −5.8%), but the same rule loses 24.1% on the validation season;
* a stricter rule searched on the walk-forward seasons (edge above 30%, best price across bookmakers) loses on both validation (−45.8%) and test (−13.1%): the rule that looked best on past seasons was noise.

So a negative ROI is not a bug in the simulation. The bookmaker's margin (5.5% on the test season) is built into every price, and a model that is slightly less accurate than the market loses about that much on average. Taking the best available price instead of Bet365's shrinks every loss, but no strategy is reliably profitable. Details, charts and the edge-threshold sensitivity are in the [final report](reports/final_report.md).

### Why some projects report 60–70%

[stogaja/Football-Match-Outcome-Predictor](https://github.com/stogaja/Football-Match-Outcome-Predictor) reports about 70% with Random Forest and logistic regression, but its inputs are the half-time score and the full-match shots, shots on target and red cards of the match being predicted. None of that is known before kickoff. Rerunning its setup on our data (`python -m src.analysis.in_play_reference`) gives 64% (RF) and 67% (LR); the half-time score alone gives 61%; the team identities alone, the only pre-match input in that set, give 48%. For a pre-match model the realistic ceiling is the bookmaker, whose favourite wins 49–59% of the time depending on the season.

![Log loss by season](reports/figures/log_loss_by_season.png)

## How predictions work

For an upcoming fixture, `src/models/predict.py`:

1. takes every Premier League match finished before the fixture's date, including this season's;
2. builds both teams' features with `features_for_fixtures`, the **same code** that built the training table;
3. checks the columns are exactly the saved models' feature list (stored in `models/model_metadata.json`);
4. runs Random Forest and XGBoost and reports each model's Home/Draw/Away probabilities, the predicted outcome from the draw rule, and whether the models agree.

```
Chelsea vs Bournemouth  (2026-10-10)

Random Forest
  Chelsea Win:       38.5%
  Draw:              29.7%
  Bournemouth Win:   31.8%
  Predicted: Draw

XGBoost
  Chelsea Win:       36.4%
  Draw:              28.0%
  Bournemouth Win:   35.6%
  Predicted: Draw

Models agree
```

Chelsea are the most likely winners in both models, but the draw probability is above each model's draw threshold, so the predicted outcome is a draw. With plain argmax it would be a Chelsea win.

The tests prove that features built this way for a known match equal the training-table features, and so do the probabilities.

**Which fixtures get predicted:** those kicking off within the next 14 days (`live.horizon_days`) that are the *next* match for both teams. A team's match after next waits until the match in between has a result.

## Prediction tracking (live forward test)

`python -m src.live predict` saves predictions to [`data/predictions/prediction_history.csv`](data/predictions/prediction_history.csv). The file has the columns requested in the project brief, plus a few for auditability: `prediction_id`, `kickoff`, `round`, `models_agree`, `model_trained_at`, `results_source`, `data_notes`, the actual score and `result_recorded_at`.

Rules that keep the test honest:

* A prediction can only be saved **before kickoff**; trying afterwards raises an error.
* The **first prediction for a fixture stands**. Re-running never changes saved probabilities.
* Filling in results (`python -m src.live update`, also run automatically by `predict`) only writes the actual-result and correct/incorrect columns.
* `data_notes` flags predictions made with incomplete inputs, such as `no_shot_data` or `missing_earlier_result`.
* The CSV is committed to git, so the commit history independently shows when each prediction was made.

`python -m src.live report` shows total predictions, correct predictions and live accuracy per model, log loss and Brier score, the hit rate for each predicted outcome (Home/Draw/Away), how often the models agree, and weekly and cumulative accuracy over time.

**First live predictions:** the 10 matchday 6 fixtures of 2026/27 (10–12 October 2026) were saved on 29 September 2026, by the Phase 3 models. They stay as saved (the first prediction stands); later fixtures are predicted by the improved models, and `model_trained_at` shows which model made each prediction.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required. The pipeline was developed on Python 3.11 with scikit-learn 1.9, XGBoost 3.2 and pandas 3.0.

## Usage

```bash
python -m src.pipeline                  # full run: data -> features -> tuning -> evaluation -> saved models (~5 min)
python -m src.pipeline --quick          # 3 tuning candidates per model, for a fast smoke test
python -m src.pipeline --force-download # re-download the season CSVs
python -m pytest                        # run the tests

python -m src.experiments              # score candidate improvements on walk-forward seasons (~4 min)
python -m src.analysis.in_play_reference  # reproduce a half-time "70% accuracy" setup on our data
python -m src.models.feature_selection  # walk-forward feature selection (after the pipeline)
python -m src.final_report              # odds benchmark, betting simulation, EDA -> reports/final_report.md

python -m src.live predict              # refresh data, record results, predict + save upcoming fixtures
python -m src.live predict --dry-run    # show predictions without saving
python -m src.live update               # record results of finished matches
python -m src.live report               # live accuracy so far
python -m src.live match "Liverpool" "Man City" --date 2026-10-11   # one-off prediction, not saved
```

To keep the forward test running, schedule `python -m src.live predict` a couple of times a week (for example with cron) and commit the history file.

To change seasons, windows, feature groups, the split or the tuning budget, edit `config.yaml`. Try new features in `src/experiments.py` first: it scores them on the walk-forward seasons without touching validation or test.

## Limitations

* **Football is noisy, and draws are the hardest part.** Draws make up about a quarter of matches but are almost never the single most likely outcome, for the models or for the bookmaker. The draw rule makes the models predict some draws, at a small cost in accuracy; those draw calls are right only a little more often than draws happen anyway. Probability-based metrics (log loss, Brier, calibration) are the fairer measure.
* **One test season is a small sample** (380 matches). Accuracy differences of 1–2 points between the models are within noise; the report includes a bootstrap interval on the RF–XGBoost gap.
* **No possession, xG, lineups, injuries or transfers.** Team strength changes within and between seasons in ways these features can't see.
* **The market is ahead.** Bookmaker odds beat both models on log loss in every out-of-sample season, and none of the betting strategies shows a reliable profit. The odds are football-data.co.uk's, republished by [premier-league-data](https://github.com/AnishKhetani/premier-league-data), and they are used only for benchmarking, never as model inputs.
* **Live predictions for 2026/27 have no shot statistics** because the openfootball feed has goals only. Replaying the 2025/26 test season with shot data removed changed log loss from 1.036 to 1.040 (RF) and 1.038 to 1.042 (XGBoost), with no loss of accuracy, so the effect is small but real.
* Early-season rows rely on thin season-to-date statistics.

## Future improvements

* Phase 4: Streamlit dashboard.
* Inputs the market uses and these sources don't have: expected goals, lineups, injuries, transfers.
* Shot statistics for the season in progress (the live feed has goals only).
* A live odds feed, which would allow the market-informed variant from the experiments.
* SHAP explanations for individual predictions.
