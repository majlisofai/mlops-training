# mlops-training · Lesson 08: fairness, transparency and the model card

Synthetic datasets for hands-on MLOps training: data validation, leakage, drift, training-serving skew and delayed labels.

**In this lesson** you ask for whom the F1 points-finish model works. You measure the champion per subgroup with fairlearn (accuracy, selection rate, false positive and false negative rate for rookies, team tiers and circuit types), write fairness and privacy requirements as gates, and watch the champion that passed every lesson 04 gate get blocked: it is far too optimistic about rookies. You try two mitigations and measure what they cost, choose a fairer model, check that no personal data reaches the model or the prediction log, and generate a model card from MLflow that records the evidence and the human decision. At the end the same check runs on GitHub Actions in your fork, once blocked and once passing.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/majlisofai/mlops-training?ref=08-fairness)

## Trainees: start here

1. **Open your codespace.** Click the badge above. Check that the branch is **`08-fairness`**, then click **Create codespace**. A copy of VS Code opens in your browser. You need a GitHub account. The codespace runs on your account and uses your free monthly Codespaces hours.
2. **Wait for the setup.** The first start takes a few minutes, because the codespace installs Docker and the Python packages. Wait until the terminal says the setup has finished before you run any command.
3. **Open the lesson.** In the file explorer, open [`exercises/fairness/README.md`](exercises/fairness/README.md). Right-click it and choose **Open Preview** to read it formatted.
4. **Do the ten steps** in the lesson, in order:

   | Step | What you do | You are done when |
   |---|---|---|
   | 1 | Start the MLflow server and open the MLflow UI | The MLflow UI opens in a browser tab |
   | 2 | Run the lesson 04 pipeline with the good candidate | `PROMOTED: f1-points-finish version 1 is now @champion` |
   | 3 | Complete TODO 1 to TODO 3 in `exercises/fairness/fairness_report.py` and run it | The rookie row shows `0.646  0.875  0.762  0.037`, and the report is logged to the run |
   | 4 | Start the API with a prediction log and send the 2025 season | `Sent 24 races (480 entries)`; with `--with-driver-code`, `24 races were refused` |
   | 5 | Run the fairness and privacy gates, complete TODO 4 in `test_fairness.py` | `1 failed, 5 passed`: `rookie false positive rate gap +0.620 is above 0.55` |
   | 6 | Run the mitigation test | Eight runs in the MLflow experiment `f1-points-finish-fairness` |
   | 7 | Compare six candidates, promote the logistic regression, check it again | The gates end with `6 passed, 3 warnings` for version 8 |
   | 8 | Generate the model card, fill in `model_card.yaml`, generate it again | `Placeholders still to fill in model_card.yaml: 0.` |
   | 9 | Run the privacy gate on a leaky prediction log | `6 personal identifiers in exercises/fairness/examples/prediction_log_v0.jsonl` |
   | 10 | Run the workflow **fairness** in your fork, with the default and the logreg candidate | The default run fails at **4. Fairness and privacy gates**; the logreg run succeeds |

5. **Discuss** the five questions at the end of the lesson with your group.

**Rules for this lesson**

- Run every command from the repository root, `/workspaces/mlops-training`.
- Keep the terminal with the MLflow server open while you work. Use other terminals for the API and the commands (**Terminal → New Terminal**).
- The subgroups (`rookie`, `team_tier`, `circuit_type`) are for measurement practice. They are **not protected characteristics**, and they are never model inputs. Treat `driver_code` and `driver_name` as personal data.
- Do not change `fairness.yaml` to make a model pass. Do not change `.github/workflows/fairness.yml`.
- In this lesson you **push to your own fork** (step 10). You cannot push to `majlisofai/mlops-training`.
- Try the TODOs before you open the reference solutions in `exercises/fairness/solution/`.

**If something goes wrong**

| Problem | Fix |
|---|---|
| `mlflow: command not found` or `ModuleNotFoundError: No module named 'fairlearn'` | The setup has not finished. Wait, or run `pip install -r .devcontainer/requirements.txt`. |
| The MLflow UI shows **403** | Start the server with the full command from step 1, including `--allowed-hosts` and `--cors-allowed-origins`. |
| A port does not open | Open the **Ports** tab, find **MLflow UI** (5000) or **f1 API** (8000) and click the globe icon. If the port is not listed, click **Forward a Port** and type the number. |
| `Registered model alias champion not found` or `Could not find models:/f1-points-finish@champion` | There is no champion yet. Do step 2. |
| `Max retries exceeded` or `Connection refused` on port 5000 | The MLflow server is not running. Start it (step 1). |
| `TODO 1: subgroup_frame() is not written yet` | Expected in step 3. Complete TODO 1 to TODO 3 in `fairness_report.py` and delete the `raise` lines. |
| `[Errno 98] address already in use` on port 8000 | The lesson 05 API or its container still runs. Stop uvicorn with **Ctrl+C**, or run `docker stop f1-api`. |
| `send_season.py` says `Connection refused` | The logged API is not running. Start it (step 4). |
| `no_personal_data_in_prediction_log` is **NOT CHECKED** | There is no `logs/predictions.jsonl` yet. Do step 4. |
| The gates check the wrong model version | The gates and the report check `@champion`. To check another version, set `MODEL_URI`, for example `MODEL_URI=models:/f1-points-finish@candidate`. |
| The model card says `fairness_gate_report.json` is for version 1, not version 8 | Run the gates again (step 7) before `model_card.py`. |
| The model card says `No fairness report in the run` | Run `fairness_report.py` for that version first: the card reads the report from the model's run. |
| pytest says `import file mismatch` | You collected `test_fairness.py` and `solution/test_fairness.py` in one run. Run one of them at a time. |
| You pushed to your fork, but no workflow run started | Actions are off in a new fork. Open the **Actions** tab of your fork and click **I understand my workflows, go ahead and enable them**. Then start a run with **Run workflow** (step 10). |
| There is no **Run workflow** button | Make `08-fairness` the default branch of your fork (**Settings → General → Default branch**), then reload the **Actions** page. |
| The push is rejected with `refusing to allow ... to create or update workflow` | You changed a file in `.github/workflows/`. Undo your change to that file, commit, and push again. |
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
| Fairness, privacy and transparency | fairlearn (`MetricFrame`) for per-subgroup metrics, pytest for the fairness and privacy gates, a model card generated from MLflow, GitHub Actions for the fairness check ([`.github/workflows/fairness.yml`](.github/workflows/fairness.yml)) |

## Datasets

| Folder | Dataset | Target |
|---|---|---|
| [`f1/data/`](f1/data/) | Synthetic Grand Prix championship: 10 fictional teams, 24 circuits, five seasons, 2,400 car-race rows | `points_finish` |
| [`spl/data/`](spl/data/) | Synthetic Saudi league: 18 clubs per season, 34 rounds, five seasons, 3,060 club-match rows | `win` |

**All results, players, teams and drivers are synthetic.** They were generated from seed 2026 and describe no real season or person. They are not a basis for betting, club or team decisions, or claims about anyone real.

Each dataset folder has a `README.md` with the file list, the column dictionary, which columns are known before versus after the event (leakage), and the facts you can check. Besides the main training and holdout CSV, each folder holds the 2026 provisional and final results, a corrections file, and `scenarios/*.json` windows for monitoring exercises (a reference window, an input-drift window and a training-serving-skew window).
