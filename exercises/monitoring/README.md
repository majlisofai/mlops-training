# Lesson 06: monitor drift, skew and delayed labels

In lesson 05 the champion went into production behind an API, and you proved that it is fast enough. Fast is not the same as right. After the release, the world keeps changing: a storm is forecast, a data feed changes its units, a new season changes the sport, and the race results arrive hours after the predictions. In this lesson the API writes every prediction to a log, and you build a monitor that reads the log and raises an alert, with a reason, when something is wrong. Then a trainer breaks production on purpose and you find out what broke.

Time: about 2 hours.

| Part | What you do | What it shows |
|---|---|---|
| Prediction log | Start the API with a log, replay a window of races | Every answer is kept: the inputs, the prediction, the model version, the time |
| Monitoring checks | Complete three checks in `monitor.py` | Each check measures one thing and says pass or alert, and why |
| Drift, skew, delayed labels | Replay a storm, a broken feed and the 2026 season | Three different problems, three different checks |
| Thresholds | Read `monitoring.yaml`, make one threshold stricter | Every threshold is a trade-off between missed problems and false alarms |
| Drift Detective | Name the fault that the trainer injected | Triage: from an alert to a cause |
| GitHub Actions | Run the monitor in your fork, on demand and every day | The monitor fails the job when it raises an alert |

Run every command from the repository root, `/workspaces/mlops-training`.

## What you will have at the end

- An API that logs every request to `logs/predictions.jsonl`, including the requests it rejected.
- A monitor with five checks, three of them written by you, and its thresholds in `monitoring.yaml` with the measurement behind each number.
- An alert for each of the three problems: input drift in the storm, training-serving skew in the new feed, and an accuracy drop in 2026 that only the delayed labels show.
- A fault that you named from the monitor output alone.
- A GitHub Actions run where the monitor passes, and a run where it fails on an alert.

## Three things that go wrong after a release

| Problem | What changes | Example in this lesson | How you see it |
|---|---|---|---|
| **Input drift** | The inputs move away from what the model was trained on. The data is correct, the world changed. | A storm: `rain_prob` 0.92 for every car, against 0.10 to 0.29 in the reference races | Compare the inputs the model saw with a reference window |
| **Training-serving skew** | The API receives a different value than the training pipeline computes for the same car. The data is wrong. | A new timing feed sends `quali_gap_ms` in seconds: 0.825 instead of 825 | Compare served features with the features recomputed by the training pipeline, request by request |
| **Delayed labels** | The truth arrives later than the prediction: provisional results after the race, final results after the stewards, 2 to 26 hours later | In 2026 a backmarker team leads and 27.5 % of the cars do not finish. The inputs barely move, but the model is right much less often. | Join the predictions with the labels when they arrive, and measure accuracy |

Drift and skew can be checked as soon as the requests arrive. Accuracy needs the labels, so it always comes late.

## The monitoring architecture

```
replay.py / inject.py ---- POST /v1/predict, X-Request-ID ---->  API (lesson 05 app.py, port 8000)
        |                                                          |
        | logs/replay_manifest.json                                | PREDICTION_LOG: one JSON line per entry
        | (request ids, race/driver keys,                          v
        |  recomputed features)                            logs/predictions.jsonl
        |                                                          |
        +------------------------->  monitor.py  <-----------------+  <--- labels CSV (when they arrive)
                                         |        <--- monitoring.yaml (thresholds), reference window
                                         v
                    results/monitoring/alerts.json, summary.md, drift_report.html (Evidently)
                    exit 0 = no alert, exit 1 = alert
```

| File | What it does |
|---|---|
| [`../serve-model/app.py`](../serve-model/app.py) | The lesson 05 API. It now writes the prediction log when `PREDICTION_LOG` is set (see step 3). |
| [`replay.py`](replay.py) | Sends a window to the API, one request per race (20 cars), and writes `logs/replay_manifest.json` |
| [`monitor.py`](monitor.py) | The monitor. **You complete TODO 1 to TODO 3**; TODO 4 is optional. |
| [`monitoring.yaml`](monitoring.yaml) | The reference window and the thresholds of every check |
| [`inject.py`](inject.py) | Drift Detective: sends a window with a hidden fault (step 10) |
| [`requirements-ci.txt`](requirements-ci.txt) | The packages for GitHub Actions, with the same versions as the codespace |
| [`.github/workflows/monitor.yml`](../../.github/workflows/monitor.yml) | Runs the whole chain on GitHub Actions |

The windows come from [`f1/data/`](../../f1/data/README.md):

| Window | Entries | What it is |
|---|---:|---|
| `reference` | 80 | The last four races of 2025. The monitoring reference: what normal looks like. |
| `wet_race` | 20 | One 2025 race with a storm forecast: `rain_prob` set to 0.92 |
| `serving_skew` | 80 | The reference races, as a new timing feed sends them: `quali_gap_ms` in seconds. The file also holds the recomputed values. |
| `season_2026` | 160 | The first eight races of 2026, new regulations. Labels in `season_2026_final.csv`. |

The monitor runs five checks:

| Check | It compares | Alert when |
|---|---|---|
| Service health | Answered against rejected entries | More than 1 % of the entries were rejected |
| Input drift | The inputs the model saw against the reference window, feature by feature. Also an Evidently drift report. | A mean moved more than 2 reference standard deviations, or a spread shrank below x0.5 or grew above x2 |
| Training-serving skew | Each served value against the recomputed value for the same entry | More than 5 % of the values of one feature differ by more than 1 % |
| Delayed labels | Predictions against the labels that arrived | Fewer than 90 % of the final labels arrived, or accuracy fell more than 0.05 below 0.842 |
| Prediction drift | The share of cars predicted to score points | It is more than 0.15 away from 0.5 |

## Step 1: start the MLflow server

Open a terminal and run the same command as in lessons 02 to 05:

```bash
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --allowed-hosts "*.app.github.dev,localhost:*,127.0.0.1:*" \
  --cors-allowed-origins "https://*.app.github.dev"
```

Leave this terminal running. Open the **Ports** tab and click the globe icon next to **MLflow UI**. Open a second terminal (**Terminal → New Terminal**) for the next step.

## Step 2: make sure there is a champion

This codespace is new, so its registry is empty. Run the lesson 04 pipeline with the good candidate and the reference gates:

```bash
GATES=exercises/quality-gates/solution/test_gates.py bash exercises/quality-gates/run_pipeline.sh
```

> **Check:** seven `PASSED` gates (accuracy 0.842, roc_auc 0.907, rookie_accuracy 0.646) and
> `PROMOTED: f1-points-finish version 1 is now @champion (previous champion: none).`

The version numbers in this lesson are for a new codespace. If you registered more versions in this codespace already, your numbers are higher. That is fine.

## Step 3: start the API with a prediction log

You cannot monitor what you did not record. The lesson 05 API now has a prediction log: when the environment variable `PREDICTION_LOG` names a file, every request to `/v1/predict` adds one JSON line per entry to it. Without `PREDICTION_LOG` the API works exactly as in lesson 05, so the lesson 05 contract tests and Docker image do not change. The change is small, about 50 lines in [`app.py`](../serve-model/app.py): the function `log_request`, one call in `predict` and one in the error handler. We changed the real API instead of a copy because the log must come from the service that runs in production.

Start the API in the second terminal:

```bash
PREDICTION_LOG=logs/predictions.jsonl uvicorn app:app --app-dir exercises/serve-model --port 8000
```

> **Check:** the log says `Serving f1-points-finish version 1 (models:/f1-points-finish@champion)`, then `Application startup complete.`

Leave it running. Open a third terminal for the next commands.

What goes into each line, and why:

| Field | Why the monitor needs it |
|---|---|
| `ts` | When it happened. A monitor looks at a time window. |
| `request_id` | The client's `X-Request-ID` header, or a random id. It joins the log with other data about the same request. |
| `status` | 200 or 422. Rejected requests are logged too, with the reason for each entry: an API that rejects everything is not healthy. |
| `model_name`, `model_version` | Which model answered. After a promotion, you compare the versions. |
| `entry`, `inputs` | The seven inputs of one car, as the model saw them, or as the client sent them when they were rejected |
| `prediction`, `probability` | The answer. Later joined with the label to measure accuracy. |

## Step 4: replay the reference window

[`replay.py`](replay.py) plays the pit wall. It sends a window to the API, one request per race, 20 cars per request, each with an `X-Request-ID`:

```bash
python exercises/monitoring/replay.py reference
```

> **Check:**
>
> ```
> Replaying reference to http://127.0.0.1:8000 (serving f1-points-finish version 1)
>   reference-...-1: 20 entries -> 200, version 1, 12 predicted in the points
>   reference-...-2: 20 entries -> 200, version 1, 11 predicted in the points
>   reference-...-3: 20 entries -> 200, version 1, 11 predicted in the points
>   reference-...-4: 20 entries -> 200, version 1, 12 predicted in the points
> Replayed reference: 4 requests, 80 entries: 80 answered, 0 rejected.
> Manifest: logs/replay_manifest.json
> ```

Look at the first line of the log:

```bash
head -1 logs/predictions.jsonl
```

> **Check:** `{"ts": "...", "request_id": "reference-...-1", "status": 200, "model_name": "f1-points-finish", "model_version": "1", "entry": 0, "inputs": {"grid": 1, "quali_gap_ms": 0, "driver_form": 8.0, ...}, "prediction": 1, "probability": 0.956...}`

`replay.py` also wrote `logs/replay_manifest.json`. It tells the monitor which requests belong to this window, the race and driver of each entry when the window has them, and the features that the training pipeline computes for the same entries (`recomputed`). In production these come from the feature pipeline, joined by request id.

## Step 5: write the monitoring checks

Run the monitor as it is:

```bash
python exercises/monitoring/monitor.py
```

> **Check:** `PASS service health` with `80 of 80 entries answered`, `SKIP delayed labels`, three checks that say `TODO`, then
> `WARNING: 3 checks are not written yet and checked nothing: input drift, training-serving skew, prediction drift` and `MONITOR OK: no alerts`.

A monitor that says OK because nothing was checked is worse than no monitor: it makes people trust it. Open [`monitor.py`](monitor.py). Read `check_service_health`: it is done, and it shows the pattern. Each check returns a `Check` with a status (`pass` or `alert`), a one-line reason, details and the measured numbers. Complete **TODO 1 to TODO 3**:

| TODO | Function | What it measures |
|---|---|---|
| 1 | `numeric_drift` | For one feature: how far its mean moved, in reference standard deviations, and how its spread changed (standard deviation / reference standard deviation) |
| 2 | `skew_mismatches` | For one feature: which served values differ from the recomputed ones by more than 1 % |
| 3 | `label_performance` | Accuracy of the predictions on the provisional labels and on the final labels, and the alert when it is too far below the baseline |
| 4 (optional) | `check_prediction_drift` | The share of cars predicted to score points |

Each TODO raises `NotImplementedError` until you delete that line. The comments in each function say what to compute. The thresholds come from `cfg`, the content of `monitoring.yaml`. The reference window has no labels, so the delayed labels check says `SKIP`: you test TODO 3 in step 8. Run the monitor again on the reference window:

```bash
python exercises/monitoring/monitor.py
```

> **Check:** every feature has `shift 0.00 sd` and spread `x1`, `Evidently flags (p < 0.05): none`, `served = recomputed for all 7 features (80 entries)`, `0.57 of the cars predicted to score points` (if you wrote TODO 4), and `MONITOR OK: no alerts`. The command exits with 0 (`echo $?`).

The reference window against itself must be quiet. That is the first test of any monitor: it does not alert on normal data.

Open the Evidently report: right-click `results/monitoring/drift_report.html` and choose **Download**, then open it in your browser. Evidently compares each feature with the reference using a statistical test (with these small windows, Kolmogorov-Smirnov or chi-square) and draws both distributions. A feature is flagged when the test's p-value is below 0.05. It is the same comparison as your per-feature check, with a different test.

## Step 6: a storm is forecast (input drift)

```bash
python exercises/monitoring/replay.py wet_race
python exercises/monitoring/monitor.py
```

> **Check:** `ALERT input drift  rain_prob: mean 0.92 vs 0.185 in the reference, shift 10.41 sd (max 2.0)`. No other feature moves more than 0.25 sd. The other checks pass, and the output ends with `MONITOR ALERT: input drift`. `echo $?` prints 1.

The data is correct: it really will rain. The model has seen rain in training (forecasts up to 0.55), but never 0.92. Nothing is broken, and still you want to know: the model's answers for this race rest on a part of the input space where it has little experience. Whether you act on it (warn the race engineers, hold back the predictions) is a decision for people, not for the monitor.

Read the Evidently line: it flags `rain_prob` **and** `circuit_type`. A single race has one circuit type, so all 20 entries are `permanent`, while the reference has a mix. That is not drift: the window is too small. Your check does not compare category shares for this reason, and it does not check the spread of `rain_prob`, because all 20 cars in a race have the same forecast. Knowing your features matters as much as the statistics.

## Step 7: a new timing feed (training-serving skew)

The timing team switched to a new feed. Replay what it sends:

```bash
python exercises/monitoring/replay.py serving_skew
```

> **Check:** all four requests get `422: entries.1.quali_gap_ms: Input should be a valid integer, got a number with a fractional part (and 18 more problems)`, and
> `Replayed serving_skew: 4 requests, 80 entries: 0 answered, 80 rejected.`

```bash
python exercises/monitoring/monitor.py
```

> **Check:**
>
> ```
>   ALERT  service health         0 of 80 entries answered, 80 rejected (100.0%, max 1%)
>                                   76 x quali_gap_ms: Input should be a valid integer, got a number with a fractional part
>   SKIP   input drift            no answered entries: no input reached the model
>   ALERT  training-serving skew  quali_gap_ms: 76 of 80 served values differ from the recomputed ones;
>                                 served / recomputed: median x0.001 (95%, max 5%)
> ```
>
> and `MONITOR ALERT: service health, training-serving skew`.

Two checks, two views of one incident. Service health says **that** the API fails: the pit wall gets no predictions. The skew check says **why**: the feed sends values a thousand times smaller than the training pipeline computes, so seconds instead of milliseconds. The 4 values that match are the pole-sitters, whose gap is 0 in any unit.

This time the lesson 05 contract caught the bug, because 0.825 is not an integer. That was luck. A feed that rounds to whole seconds (1 instead of 825) passes the contract, and the model answers with wrong inputs, silently. You meet that version in step 10.

## Step 8: the 2026 season (delayed labels)

New regulations, new power units. Replay the first eight races of 2026:

```bash
python exercises/monitoring/replay.py season_2026
python exercises/monitoring/monitor.py
```

> **Check:** `MONITOR OK: no alerts`. Input drift passes: the largest move is `team_dnf_rate`, shift 1.03 sd and spread x1.27. `SKIP delayed labels  no labels for this window yet: the results are not known`. Prediction drift: `0.61 of the cars predicted to score points`.

The races are over, but the labels have not arrived. As far as the inputs go, 2026 looks like 2025. The provisional results come right after each race. The stewards then hand out penalties (see [`stewards_2026.csv`](../../f1/data/stewards_2026.csv)), published 2 to 26 hours after the race, and only then is the classification final. [`season_2026_final.csv`](../../f1/data/season_2026_final.csv) holds both labels. Give the monitor the labels:

```bash
python exercises/monitoring/monitor.py --labels f1/data/season_2026_final.csv
```

> **Check:**
>
> ```
>   ALERT  delayed labels         accuracy 0.706 on the final labels, baseline 0.842 (alert below 0.792)
>                                   labels from f1/data/season_2026_final.csv: 160 provisional, 160 final for 160 predictions
>                                   accuracy on provisional labels 0.706, on final labels 0.706 (baseline 0.842)
>                                   the stewards changed 6 labels: 3 predictions went from right to wrong, 3 from wrong to right
> ```
>
> and `MONITOR ALERT: delayed labels`.

The champion was right for 0.842 of the 2025 entries. In 2026 it is right for 0.706. The cause is not in the inputs, it is in the sport: 27.5 % of the cars do not finish (10.9 % before), and Cobalt Racing, a backmarker team, leads the championship. The relation between the inputs and the result changed. That is called **concept drift**, and no input check can see it. Only the labels show it, and they come late.

The provisional and final accuracy are the same here, 0.706, but 6 of the 160 labels changed: the model lost 3 right answers and won 3. With other penalties the provisional number could be too high or too low. Report accuracy on final labels, and say so.

What if you only had the provisional results? Try it:

```bash
python exercises/monitoring/monitor.py --labels f1/data/season_2026_provisional.csv
```

> **Check:** `ALERT delayed labels  labels late: 0 of 160 predictions have a final label (0%, need 90%)`. The details still show `accuracy on provisional labels 0.706`.

The monitor refuses to give a final verdict on provisional labels. Who decides when labels are "late" is a business rule: here, final labels for 90 % of the predictions.

## Step 9: why these thresholds

Every number in [`monitoring.yaml`](monitoring.yaml) was chosen from measurements of the windows in this lesson. A threshold too tight raises false alarms, and people learn to ignore the monitor. A threshold too loose misses real problems.

| Threshold | Value | Measured on normal windows | Measured on problems | Why this number |
|---|---:|---|---|---|
| `max_rejected_share` | 1 % | 0 % in every normal window | 100 % (new feed), 25 % (renamed category, step 10) | The lesson 05 SLO: at most 1 % failed requests |
| `max_mean_shift` | 2.0 sd | at most 1.03 sd: `rain_prob` in 2025 rounds 17 to 20, `team_dnf_rate` in 2026 | 10.41 sd: `rain_prob` in the storm | Twice the largest normal move, and five times smaller than the storm |
| `min_spread_ratio`, `max_spread_ratio` | x0.5 to x2.0 | x0.76 to x1.27 | x0.0011 (seconds instead of ms), x0 (stuck feature, step 10) | A fault that makes all values (nearly) equal moves the mean very little. The spread catches it. |
| `per_race_features` | `rain_prob` | spread 0 in every single-race window | | All 20 cars in a race have the same forecast |
| `relative_tolerance` | 1 % | 0 differences in every normal window | x0.001 | Allows rounding, not a change of unit |
| `max_mismatch_share` | 5 % | 0 % | 25 % to 98 % | One wrong car in a race of 20 is 5 %: a single glitch does not page anyone |
| `baseline_accuracy` | 0.842 | | | The champion on the 2025 holdout, measured by the lesson 04 gates |
| `max_accuracy_drop` | 0.05 | | 0.136 in 2026 (0.706) | With 160 predictions, accuracy moves by about ±0.03 by chance alone (one standard error). 0.05 is beyond that, and 0.792 is below the grid rule (0.804 on 2025). |
| `min_final_label_coverage` | 90 % | 100 % when the stewards have published | 38 % (label delay, step 10), 0 % with provisional results only | Up to 26 hours of delay is normal. Judge the model when nearly all races are final. |
| `expected_share`, `max_share_change` | 0.5 ± 0.15 | 0.50 to 0.61 | | Exactly 10 of 20 cars score points in every race |

Prediction drift did not alert in any window of this lesson. It is a cheap early warning that needs no labels, so keep it, but do not rely on it alone.

Now make one threshold stricter and see what it costs. Copy the file, change `max_mean_shift` from `2.0` to `1.0`, and run the monitor on the 2026 window again:

```bash
sed 's/max_mean_shift: 2.0/max_mean_shift: 1.0/' exercises/monitoring/monitoring.yaml > /tmp/strict.yaml
python exercises/monitoring/replay.py season_2026
python exercises/monitoring/monitor.py --config /tmp/strict.yaml
```

> **Check:** `ALERT input drift  team_dnf_rate: mean 0.2074 vs 0.1208 in the reference, shift 1.03 sd (max 1.0)`.

The strict monitor sees 2026 coming, days before the labels. But the same threshold also alerts on four ordinary races of 2025 (`rain_prob`, shift 1.03 sd). Try it: `python exercises/monitoring/inject.py normal_weekend`, then the monitor with `--config /tmp/strict.yaml`. Which one is worth more to your team, the early warning or the quiet nights? Write the answer next to the number, so the next person knows.

## Step 10: Drift Detective

The trainer (or your partner) breaks production. [`inject.py`](inject.py) sends a window with a hidden fault to your API, under the window name `incident`. There are five cases; one of them has no fault at all.

The trainer runs one of:

```bash
python exercises/monitoring/inject.py --list
python exercises/monitoring/inject.py --random
```

`--random` keeps the case secret in `logs/incident_answer.txt`. The detective then runs the monitor and reads the evidence:

```bash
python exercises/monitoring/monitor.py
```

> **Check:** the injection prints only `Incident sent: 4 requests, 80 entries. Detective: run the monitor.` (or 8 requests and 160 entries).

Name the fault with three answers:

1. **What** changed: which check alerts, which feature, and how (moved, shrank, renamed, missing)?
2. **Where** did it most likely break: the feed, the feature store, a reference list, the label pipeline, or nowhere?
3. **What** would you do now?

Use the reason lines first. Then confirm in the log: `grep incident logs/predictions.jsonl | head`. When you have named it, compare with `logs/incident_answer.txt`, and play another round. The trainer has the answer key.

## Step 11: run the monitor on GitHub Actions

The workflow [`monitor.yml`](../../.github/workflows/monitor.yml) runs the whole chain in one job:

| Step | What it does |
|---|---|
| Start the MLflow server | An MLflow server for this run only |
| 1. Promote a champion | The lesson 04 pipeline with the reference gates. The champion is version 1. |
| 2. Start the API with the prediction log | `uvicorn` with `PREDICTION_LOG=logs/predictions.jsonl` |
| 3. Replay the window | `replay.py` for a window, `inject.py` for a Drift Detective case |
| 4. Run the monitor | **Your** `monitor.py` (or the reference solution, if you choose it). With `season_2026` it adds `--labels f1/data/season_2026_final.csv`. The job fails when the monitor raises an alert. |

It writes the summary table to the run page and keeps `results/monitoring/` (with the Evidently report) and `logs/` as the artifact `monitoring-report`. It runs in three cases:

| Trigger | Window |
|---|---|
| A push that changes `exercises/monitoring/` or `exercises/serve-model/app.py` | `reference` |
| **Run workflow** | The one you choose under **scenario** |
| Every day at 05:23 UTC | `reference`. A daily run on a known window proves that the monitoring chain itself still works. GitHub runs schedules only on the default branch. |

1. **Commit your work.**

   ```bash
   git add exercises/monitoring
   git commit -m "Write the monitoring checks"
   ```

2. **Push to your fork.** Open **Source Control** and click **Sync Changes** (or **Publish Branch**). If you made a fork in an earlier lesson, the codespace pushes there; otherwise click **Create Fork**. If this is a new fork, open its **Actions** tab and enable the workflows.
3. **Make `06-monitoring` the default branch of your fork** (**Settings → General → Default branch**), so that GitHub shows the **Run workflow** button and runs the daily schedule.
4. Your push started the workflow **monitor** with the reference window.

> **Check:** the job **Replay and monitor** succeeds in about two minutes (we measured 1 min 28 s to 1 min 54 s). The run page shows `Monitoring: OK, no alerts` and a table with five checks, the same results as in step 5.

Now a storm. Open **Actions → monitor → Run workflow**, set **scenario** to `wet_race`, and click **Run workflow**.

> **Check:** step **4. Run the monitor** fails with `MONITOR ALERT: input drift`, and the job fails. The run page shows `Monitoring: ALERT: input drift` and `rain_prob: mean 0.92 vs 0.185 in the reference, shift 10.41 sd (max 2.0)`, the same numbers as in your codespace. Download the artifact `monitoring-report` to see the Evidently report, the prediction log and the alerts.

Try `serving_skew` and `season_2026` too: they fail with `service health, training-serving skew` and with `delayed labels` (accuracy 0.706).

When a run fails, GitHub notifies the person who started it; for the schedule, the person who last changed the schedule in the workflow file (if their Actions notifications are on). That is the simplest alert channel there is.

## Discuss

1. In step 7 the contract turned skew into a loud failure. In the `units_change` case of step 10, a feed that rounds to whole seconds passes silently. Should the API reject more (tighter ranges for `quali_gap_ms`), or should the monitor catch more? What does each cost?
2. 2026 showed no input drift above the threshold, yet the model got much worse. If your labels arrived only once a month, what would you watch in the meantime?
3. The reference window is four races. Evidently flags `rain_prob` on four ordinary races, because four forecasts are a tiny sample. Would you rather use the whole 2021-2024 training data as the reference? What would you lose?
4. Who should receive each alert: the team that runs the API, the data engineers who own the feeds, or the data scientists who own the model? Pick one for each of the five checks.

## Stretch

- **Saudi league.** The monitor's functions work on any data. Try your TODO 1 and TODO 2 on the [`spl/data/scenarios`](../../spl/data/) windows, in a Python shell from the repository root:

  ```python
  import json, sys, pandas as pd, yaml
  sys.path.insert(0, "exercises/monitoring")
  import monitor
  cfg = yaml.safe_load(open("exercises/monitoring/monitoring.yaml"))
  load = lambda name, key="rows": pd.DataFrame(json.load(open(f"spl/data/scenarios/{name}.json"))[key])
  ref, heat = load("reference_window"), load("heatwave_window")
  served, recomputed = load("serving_skew_window", "served"), load("serving_skew_window", "recomputed")
  print(monitor.numeric_drift("kickoff_temp_c", heat["kickoff_temp_c"], ref["kickoff_temp_c"], cfg, per_race=False))
  print(monitor.numeric_drift("travel_km", served["travel_km"], ref["travel_km"], cfg, per_race=False))
  print(monitor.skew_mismatches(served["travel_km"], recomputed["travel_km"], True, cfg).sum())
  ```

  We measured: the heatwave alerts (`kickoff_temp_c` shift 3.48 sd, spread x0). The feed that sends miles instead of kilometres does **not** alert on drift (shift 0.29 sd, spread x0.62), but 69 of 144 values are skewed, with a median ratio of x0.621: a mile is 1.609 km. The 75 that match are home games, 0 km.
- Add a latency check to service health: log the time each request took in `log_request`, and alert when the 95th percentile is above the lesson 05 SLO (500 ms).
- Make the reference a whole season: let `reference:` in `monitoring.yaml` also accept `f1/data/races.csv` with a season filter. Measure the five Drift Detective cases again. Which thresholds would you change?
- Compare model versions: promote the lesson 05 huge forest, reload the API, replay `season_2026` again and extend `check_delayed_labels` to report accuracy per `model_version`.

## Stuck?

There is a reference solution in [`solution/monitor.py`](solution/monitor.py). Try the TODOs first. To run it:

```bash
python exercises/monitoring/solution/monitor.py
```

The answer key for the Drift Detective is in `solution/ANSWER_KEY.md`. It is for the trainer: do not open it before you play.
