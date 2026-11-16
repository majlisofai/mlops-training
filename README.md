# mlops-training · Lesson 07: recover the service and train a controlled challenger

Synthetic datasets for hands-on MLOps training: data validation, leakage, drift, training-serving skew and delayed labels.

**In this lesson** the F1 points-finish API from lesson 05 runs as a service, and on a Sunday night somebody breaks it: a hotfix becomes the production model without the quality gates, and at the nightly restart the API cannot start. You collect the evidence with a runbook (health, version, logs, inputs, accuracy on the delayed final labels), ask an AI assistant for a second opinion, decide, roll back, and prove the recovery with the lesson 05 checks. Then you do properly what the hotfix tried to do: you train a challenger on newer data, in time order, compare it with the champion on 2026 races that neither model saw, and promote it only through the lesson 04 gates and an approval. At the end the comparison runs on GitHub Actions in your fork, once with a challenger that wins and once with one that is blocked.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/majlisofai/mlops-training?ref=07-recovery)

## Trainees: start here

1. **Open your codespace.** Click the badge above. Check that the branch is **`07-recovery`**, then click **Create codespace**. A copy of VS Code opens in your browser. You need a GitHub account. The codespace runs on your account and uses your free monthly Codespaces hours.
2. **Wait for the setup.** The first start takes a few minutes, because the codespace installs Docker and the Python packages. Wait until the terminal says the setup has finished before you run any command.
3. **Open the lesson.** In the file explorer, open [`exercises/recovery/README.md`](exercises/recovery/README.md). Right-click it and choose **Open Preview** to read it formatted. Keep [`exercises/recovery/RUNBOOK.md`](exercises/recovery/RUNBOOK.md) open next to it.
4. **Do the ten steps** in the lesson, in order:

   | Step | What you do | You are done when |
   |---|---|---|
   | 1 | Start MLflow, make the champion, run the API container with a restart policy | `/health` answers `"model_version":"1"` from the container `f1-api` |
   | 2 | Collect the evidence while the service is healthy | `triage.py all` shows `UP`, version 1, and `2026 overall: accuracy 0.706` |
   | 3 | Your trainer starts the incident | The pager message `f1-api health check FAILING` |
   | 4 | Triage: answer the five runbook questions | Your report names the version, the time, the log error and the reason the API cannot start |
   | 5 | Give the evidence to an AI assistant and check its hypotheses | Each hypothesis is confirmed or rejected by a line of evidence |
   | 6 | Roll back and prove the recovery | `/health` says version 1, `8 passed`, `SLO MET`, accuracy 0.706 again |
   | 7 | Complete TODO 1 to TODO 3 in `exercises/recovery/train_challenger.py` | `Registered f1-points-finish version 3 as @challenger` |
   | 8 | Compare champion and challenger on 2026 rounds 5 to 8 | `RECOMMENDATION: PROMOTE`, and you can say why that is a close call |
   | 9 | Promote through `promote.py`, check the release, watch a weaker proposal be refused | `PROMOTED: ... version 3`, then `REFUSED` for `all_history.yaml` |
   | 10 | Run the workflow **challenger** in your fork, with a winning and a blocked challenger | The first run succeeds; the second fails at **3. Compare champion and challenger** |

5. **Discuss** the four questions at the end of the lesson with your group, and read the trade-offs table before you do.

**Rules for this lesson**

- Run every command from the repository root, `/workspaces/mlops-training`.
- Keep the terminal with the MLflow server open while you work. Use other terminals for the commands (**Terminal → New Terminal**).
- **Do not open `exercises/recovery/incident.py` before step 4 is done.** It holds the answer.
- Collect evidence before you change anything, and write down the time of every observation and action.
- In this lesson you **push to your own fork** (step 10). You cannot push to `majlisofai/mlops-training`.
- Do not change `comparison.yaml` to make a challenger pass. Do not change `.github/workflows/challenger.yml`.
- Try the TODOs before you open the reference solution in `exercises/recovery/solution/`.

**If something goes wrong**

| Problem | Fix |
|---|---|
| `mlflow: command not found` or `locust: command not found` | The setup has not finished. Wait, or run `pip install -r .devcontainer/requirements.txt`. |
| The MLflow UI shows **403** | Start the server with the full command from step 1, including `--allowed-hosts` and `--cors-allowed-origins`. |
| A port does not open | Open the **Ports** tab, find **MLflow UI** (5000), **f1 API** (8000) or **Locust UI** (8089) and click the globe icon. If the port is not listed, click **Forward a Port** and type the number. |
| `docker: Error response from daemon: Conflict. The container name "/f1-api" is already in use` | An old container with that name exists. Run `docker rm -f f1-api`, then run the container again. |
| `docker: conflicting options: cannot specify both --restart and --rm` | Leave out `--rm`. In this lesson the container runs in the background with `-d --restart on-failure`. |
| The container starts, then stops with `Connection refused` to `127.0.0.1:5000` | You left out `--network host`, or the MLflow server is not running. |
| `incident.py: nothing was changed. ... has no @champion` | Do step 1 first: the incident needs a champion and a running container. |
| `incident.py: nothing was changed. The container f1-api is not running` | Start the container as in step 1. The incident restarts it, so it must run under Docker, not uvicorn. |
| `triage.py` says `Container f1-api: not found` | The API does not run as a container. Start it as in step 1, or set `CONTAINER=<name>`. |
| After the rollback, `/health` still gives `Connection refused` | Docker waits longer after each failed start. Wait a minute, or run `docker restart f1-api`. If it still fails, read `docker logs --tail 20 f1-api`: `@champion` must be a version that the API can serve. |
| `promote.py --rollback` says `REFUSED: version 1 has no previous champion to go back to` | `@champion` is already version 1: the rollback is done. Check with `triage.py registry`. |
| `train_challenger.py` says `STOPPED at TODO 1` (or 2, 3) | That TODO is not written yet, or its `raise NotImplementedError` line is still there. |
| `compare.py` says `@champion and @challenger are both version N` | Train a new challenger first (step 7). |
| `compare.py` ends with `KEEP the champion` and exit code 1 | That is the result, not an error: the challenger did not meet a rule. The list below the recommendation says which. |
| `promote.py` says `REFUSED: the gate report is for version N, but @candidate is version M` | Something moved `@candidate` after `compare.py`. Run `compare.py` again, then `promote.py`. |
| Your numbers differ from the lesson in the last digit of roc_auc | Forests trained on different kinds of computers can differ by a hair. Accuracy should match exactly. |
| You pushed to your fork, but no workflow run started | Actions are off in a new fork. Open the **Actions** tab of your fork and click **I understand my workflows, go ahead and enable them**. Then start a run with **Run workflow** (step 10). |
| There is no **Run workflow** button | Make `07-recovery` the default branch of your fork (**Settings → General → Default branch**), then reload the **Actions** page. |
| The push is rejected with `refusing to allow ... to create or update workflow` | You changed a file in `.github/workflows/`. Undo your change to that file, commit, and push again. |
| You closed the browser tab | Go to [github.com/codespaces](https://github.com/codespaces) and open your codespace again. Your work is still there. Start the MLflow server and run `docker start f1-api`. |

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
| Incident response | [`RUNBOOK.md`](exercises/recovery/RUNBOOK.md) and `triage.py` (health, registry, logs, inputs, delayed labels), Docker's restart policy, `promote.py --rollback`, an AI assistant such as GitHub Copilot Chat |
| Challenger comparison and CI | `train_challenger.py` and `compare.py` with the MLflow Model Registry aliases `@champion`, `@challenger` and `@candidate`, GitHub Actions for the comparison ([`.github/workflows/challenger.yml`](.github/workflows/challenger.yml)) |

## Datasets

| Folder | Dataset | Target |
|---|---|---|
| [`f1/data/`](f1/data/) | Synthetic Grand Prix championship: 10 fictional teams, 24 circuits, five seasons, 2,400 car-race rows | `points_finish` |
| [`spl/data/`](spl/data/) | Synthetic Saudi league: 18 clubs per season, 34 rounds, five seasons, 3,060 club-match rows | `win` |

**All results, players, teams and drivers are synthetic.** They were generated from seed 2026 and describe no real season or person. They are not a basis for betting, club or team decisions, or claims about anyone real.

Each dataset folder has a `README.md` with the file list, the column dictionary, which columns are known before versus after the event (leakage), and the facts you can check. Besides the main training and holdout CSV, each folder holds the 2026 provisional and final results, a corrections file, and `scenarios/*.json` windows for monitoring exercises (a reference window, an input-drift window and a training-serving-skew window).
