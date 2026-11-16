# mlops-training · Lesson 06: monitor drift, skew and delayed labels

Synthetic datasets for hands-on MLOps training: data validation, leakage, drift, training-serving skew and delayed labels.

**In this lesson** you watch the F1 points-finish model in production. The lesson 05 API now writes every prediction to a log: the inputs, the answer, the model version and the time. You replay race windows against it and build a monitor that reads the log and raises an alert, with a reason, for three different problems: input drift (a storm forecast), training-serving skew (a timing feed that sends seconds instead of milliseconds) and an accuracy drop in the 2026 season that only the delayed race results reveal. Then the trainer breaks production on purpose and you name the fault from the monitor output. At the end the monitor runs on GitHub Actions in your fork, on demand and every day, and fails the job on an alert.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/majlisofai/mlops-training?ref=06-monitoring)

## Trainees: start here

1. **Open your codespace.** Click the badge above. Check that the branch is **`06-monitoring`**, then click **Create codespace**. A copy of VS Code opens in your browser. You need a GitHub account. The codespace runs on your account and uses your free monthly Codespaces hours.
2. **Wait for the setup.** The first start takes a few minutes, because the codespace installs Docker and the Python packages. Wait until the terminal says the setup has finished before you run any command.
3. **Open the lesson.** In the file explorer, open [`exercises/monitoring/README.md`](exercises/monitoring/README.md). Right-click it and choose **Open Preview** to read it formatted.
4. **Do the eleven steps** in the lesson, in order:

   | Step | What you do | You are done when |
   |---|---|---|
   | 1 | Start the MLflow server | The MLflow UI opens in a browser tab |
   | 2 | Run the lesson 04 pipeline with the good candidate | `PROMOTED: f1-points-finish version 1 is now @champion`, with 7 gates passed |
   | 3 | Start the API with `PREDICTION_LOG=logs/predictions.jsonl` | The log says `Serving f1-points-finish version 1` |
   | 4 | Replay the reference window | `Replayed reference: 4 requests, 80 entries: 80 answered, 0 rejected.` |
   | 5 | Complete TODO 1 to TODO 3 in `exercises/monitoring/monitor.py` | The reference window gives `MONITOR OK: no alerts`, with input drift and skew `PASS` |
   | 6 | Replay the storm forecast | `ALERT input drift  rain_prob: ... shift 10.41 sd` |
   | 7 | Replay the new timing feed | `ALERT service health` (80 rejected) and `ALERT training-serving skew` (`quali_gap_ms`, `median x0.001`) |
   | 8 | Replay 2026, then give the monitor the final labels | `ALERT delayed labels  accuracy 0.706 on the final labels, baseline 0.842` |
   | 9 | Read the thresholds, make one stricter | The strict monitor alerts on 2026 `team_dnf_rate` |
   | 10 | Drift Detective: name the fault the trainer injected | Your answer matches `logs/incident_answer.txt` |
   | 11 | Run the workflow **monitor** in your fork, with `reference` and `wet_race` | The reference run succeeds; the wet race run fails at **4. Run the monitor** |

5. **Discuss** the four questions at the end of the lesson with your group.

**Rules for this lesson**

- Run every command from the repository root, `/workspaces/mlops-training`.
- Keep the terminals with the MLflow server and the API open while you work. Use another terminal for the commands (**Terminal → New Terminal**).
- In this lesson you **push to your own fork** (step 11). You cannot push to `majlisofai/mlops-training`.
- Do not change `monitoring.yaml` to make an alert go away. Change a threshold only with a measurement and a reason, and write both next to the number. Do not change `.github/workflows/monitor.yml`.
- Try the TODOs before you open the reference solution in `exercises/monitoring/solution/`. Do not open `solution/ANSWER_KEY.md`: it is for the trainer.

**If something goes wrong**

| Problem | Fix |
|---|---|
| `mlflow: command not found` or `uvicorn: command not found` | The setup has not finished. Wait, or run `pip install -r .devcontainer/requirements.txt`. |
| The MLflow UI shows **403** | Start the server with the full command from step 1, including `--allowed-hosts` and `--cors-allowed-origins`. |
| A port does not open | Open the **Ports** tab, find **MLflow UI** (5000) or **f1 API** (8000) and click the globe icon. If the port is not listed, click **Forward a Port** and type the number. |
| The API stops with `Registered Model with name=f1-points-finish not found` or `Registered model alias champion not found` | There is no champion yet. Do step 2. |
| The API stops with `Max retries exceeded` or `Connection refused` on port 5000 | The MLflow server is not running. Start it (step 1), then start the API again. |
| `[Errno 98] address already in use` on port 8000 | Something else uses port 8000: an API from an earlier start, or the lesson 05 container. Stop uvicorn with **Ctrl+C**, or run `docker stop f1-api`. |
| `replay.py` says `The API does not answer at http://127.0.0.1:8000` | Start the API (step 3) and wait for `Application startup complete.` |
| The monitor says `the log has no entry of this window: is PREDICTION_LOG set...` | The API runs without the log. Stop it with **Ctrl+C**, start it again with `PREDICTION_LOG=logs/predictions.jsonl` in front (step 3), and replay the window again. |
| `logs/replay_manifest.json not found` | Replay a window first (step 4). The monitor always checks the last window you sent. |
| The monitor says `TODO` for a check you wrote | You left the `raise NotImplementedError(...)` line in the function. Delete it. |
| The monitor stops with `NameError` or `KeyError` in your TODO | Read the last lines of the error: they name the line in `monitor.py`. Use the names in the comments (`ref_mean`, `ref_sd`, `t`, `has_prov`, ...). |
| `MONITOR OK` but you expected an alert | Check that you replayed the right window: the first line of the monitor output names it. |
| You get other numbers than the lesson | Your champion is a different model. Run step 2 with the good candidate and the reference gates. If you have several versions, `@champion` must be the 300-tree forest. |
| You pushed to your fork, but no workflow run started | Actions are off in a new fork. Open the **Actions** tab of your fork and click **I understand my workflows, go ahead and enable them**. Then start a run with **Run workflow** (step 11). |
| There is no **Run workflow** button | Make `06-monitoring` the default branch of your fork (**Settings → General → Default branch**), then reload the **Actions** page. |
| The push is rejected with `refusing to allow ... to create or update workflow` | You changed a file in `.github/workflows/`. Undo your change to that file, commit, and push again. |
| The daily run does not start | GitHub runs schedules only on the default branch, and pauses them in a repository with no activity for 60 days. Make `06-monitoring` the default branch of your fork. |
| You closed the browser tab | Go to [github.com/codespaces](https://github.com/codespaces) and open your codespace again. Your work is still there. Start the MLflow server and the API again. |

When you finish, stop the codespace (**Codespaces** menu at the bottom left → **Stop Current Codespace**) so it does not use your hours.

## Tools in the codespace

The codespace runs Python 3.12 and installs the tools from [`.devcontainer/requirements.txt`](.devcontainer/requirements.txt):

| Task | Tools |
|---|---|
| Data exploration | pandas, DuckDB, JupyterLab, matplotlib, seaborn, ydata-profiling, and the Data Wrangler and Rainbow CSV extensions (right-click a CSV → **Open in Data Wrangler**) |
| Data versioning | DVC and the DVC extension |
| Data quality | pandera and Great Expectations for validation, Evidently for drift and training-serving skew reports |
| Model training | scikit-learn |
| Experiment tracking | MLflow tracking and Model Registry (the codespace forwards port 5000 for the MLflow UI) |
| Quality gates and CI | pytest for the gates, GitHub Actions for the release pipeline ([`.github/workflows/release.yml`](.github/workflows/release.yml)) |
| Model serving | FastAPI and uvicorn for the API (port 8000), Docker (docker-in-docker) for the container image, pytest with FastAPI's TestClient for the contract tests |
| Load testing and CI | Locust (web UI on port 8089), GitHub Actions for the serving check ([`.github/workflows/serve.yml`](.github/workflows/serve.yml)) |
| Monitoring and CI | The API's prediction log, Evidently for the data drift report, GitHub Actions for the monitoring run, on demand and daily ([`.github/workflows/monitor.yml`](.github/workflows/monitor.yml)) |

## Datasets

| Folder | Dataset | Target |
|---|---|---|
| [`f1/data/`](f1/data/) | Synthetic Grand Prix championship: 10 fictional teams, 24 circuits, five seasons, 2,400 car-race rows | `points_finish` |
| [`spl/data/`](spl/data/) | Synthetic Saudi league: 18 clubs per season, 34 rounds, five seasons, 3,060 club-match rows | `win` |

**All results, players, teams and drivers are synthetic.** They were generated from seed 2026 and describe no real season or person. They are not a basis for betting, club or team decisions, or claims about anyone real.

Each dataset folder has a `README.md` with the file list, the column dictionary, which columns are known before versus after the event (leakage), and the facts you can check. Besides the main training and holdout CSV, each folder holds the 2026 provisional and final results, a corrections file, and `scenarios/*.json` windows for monitoring exercises (a reference window, an input-drift window and a training-serving-skew window).
