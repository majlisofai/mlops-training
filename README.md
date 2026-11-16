# mlops-training · Lesson 09: security gates and production-readiness acceptance

Synthetic datasets for hands-on MLOps training: data validation, leakage, drift, training-serving skew and delayed labels.

**In this lesson** you make the F1 points-finish API ready for production, and you prove it. You protect the API with a client key and an admin key, so that nobody anonymous can ask for predictions or reload the model. You fingerprint the release with a SHA-256 manifest signed off in MLflow, and you audit the serving image for known vulnerabilities. Then one command runs every gate of the course, from data validation to the load test, and writes a release dossier with the evidence, the owners and a decision that only a named person can set to ship. Your trainer tampers with the model and deploys the unsecured image, and the suite blocks both. At the end the same acceptance runs on GitHub Actions in your fork, where your approval is the release decision, and you decide eight release scenarios on your own.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/majlisofai/mlops-training?ref=09-acceptance)

## Trainees: start here

1. **Open your codespace.** Click the badge above. Check that the branch is **`09-acceptance`**, then click **Create codespace**. A copy of VS Code opens in your browser. You need a GitHub account. The codespace runs on your account and uses your free monthly Codespaces hours.
2. **Wait for the setup.** The first start takes a few minutes, because the codespace installs Docker and the Python packages. Wait until the terminal says the setup has finished before you run any command.
3. **Open the lesson.** In the file explorer, open [`exercises/acceptance/README.md`](exercises/acceptance/README.md). Right-click it and choose **Open Preview** to read it formatted.
4. **Do the eleven steps** in the lesson, in order:

   | Step | What you do | You are done when |
   |---|---|---|
   | 1 | Start MLflow, promote the good candidate, write your name into `release.yaml` | `PROMOTED: f1-points-finish version 1 is now @champion` |
   | 2 | Make the API keys with `keys.py` and load them | `.secrets/api.env` exists, and `git check-ignore` says git ignores it |
   | 3 | Complete TODO 1 and 2 in `app.py` and TODO 3 and 4 in `test_access.py` | `8 passed` |
   | 4 | Build and run the secured container, call it with and without keys | 401 without a key, 403 with the wrong role, 200 with the right one |
   | 5 | Sign the release manifest | `INTEGRITY OK: 10 files match the manifest` |
   | 6 | Audit the dependencies of the image | `AUDIT OK` for `lab09-api:1`, 6 findings for the lesson 05 image |
   | 7 | Measure the subgroups and write the model card | `SUBGROUPS OK`, and no `TODO` left in `evidence/model_card.md` |
   | 8 | Run the acceptance suite and decide | Ten gates PASS, then `DECISION: SHIP ... set by <your name>` |
   | 9 | Tamper with the model, deploy the unsecured image | `DO NOT SHIP` for each fault, then `INTEGRITY OK` again |
   | 10 | Run the workflow **acceptance** in your fork, approve, then inject a fault | The approved run ships; the fault run is blocked at **7. Acceptance suite** |
   | 11 | Decide the eight cards in `scenarios.md` alone | You wrote a decision, the evidence and a next action for each card |

5. **Discuss** the five questions at the end of the lesson with your group.

**Rules for this lesson**

- Run every command from the repository root, `/workspaces/mlops-training`.
- Keep the terminal with the MLflow server open while you work. Use other terminals for the container and the commands (**Terminal → New Terminal**), and load the keys in each new terminal: `set -a; source .secrets/api.env; set +a`.
- **Never commit, paste or share `.secrets/api.env` or a key.** If a key leaks, make new ones with `python exercises/acceptance/keys.py --rotate` and restart the API.
- In this lesson you **push to your own fork** (step 10). You cannot push to `majlisofai/mlops-training`.
- Do not change `criteria.yaml`, `slo.yaml` or `.github/workflows/acceptance.yml` to make a release pass, and do not sign a manifest again to hide a change.
- Try the TODOs before you open the reference solution in `exercises/acceptance/solution/`. Do not open `solution/scenarios.md` before the scenario check.

**If something goes wrong**

| Problem | Fix |
|---|---|
| `mlflow: command not found`, `uvicorn: command not found` or `No module named pip_audit` | The setup has not finished. Wait, or run `pip install -r .devcontainer/requirements.txt`. |
| The MLflow UI shows **403** | Start the server with the full command from step 1, including `--allowed-hosts` and `--cors-allowed-origins`. |
| A port does not open | Open the **Ports** tab, find **MLflow UI** (5000) or **f1 API** (8000) and click the globe icon. If the port is not listed, click **Forward a Port** and type the number. |
| `Registered Model with name=f1-points-finish not found` or `Registered model alias champion not found` | There is no champion yet. Do step 1. |
| `ERROR: API_CLIENT_KEY is not set` from `accept.py`, or `KeyError: 'API_CLIENT_KEY'` | This terminal has no keys. Run `set -a; source .secrets/api.env; set +a`. If the file does not exist, do step 2. |
| `.secrets/api.env exists. Use --rotate` | You already have keys. Use them, or make new ones with `--rotate` and restart the container. |
| The container stops with `RuntimeError: API_CLIENT_KEY is not set or shorter than 32 characters` | You left out `--env-file .secrets/api.env`. The API does not start without its keys. |
| Every call answers `unknown API key`, also with the right variable | The container runs with other keys than your terminal: you ran `keys.py --rotate` after you started it. Stop the container and start it again. |
| Access tests fail with `status 200: ... answers without a key` against the container | The container runs an image built before you did TODO 1 and 2. Build `lab09-api:1` again and restart the container. |
| `[Errno 98] address already in use` or `bind: address already in use` on port 8000 | Another API uses port 8000: the lesson 05 container `f1-api`, uvicorn, or `lab09-unsecured`. Stop it with **Ctrl+C** or `docker stop <name>`. |
| `docker: Error response from daemon: Conflict. The container name "/lab09-api" is already in use` | An old container with that name exists. Run `docker rm -f lab09-api`, then start it again. |
| `Cannot connect to the Docker daemon` | Docker is still starting. Wait a minute and try again. If it does not help, rebuild the codespace (**Ctrl+Shift+P** → **Codespaces: Rebuild Container**). |
| `NOT SIGNED: f1-points-finish version N has no tag integrity.manifest_sha256` | Do step 5. Each new champion version needs its own sign-off. |
| `REFUSED: f1-points-finish version 1 was already signed` | That is the protection working. If a file changed, find out why before you think of `--force`. |
| `INTEGRITY FAILED` and you did not run `tamper.py` | Something changed after the sign-off. Run `git status` for the data and criteria files, and `python exercises/acceptance/tamper.py --restore` if the trainer tampered. |
| `TAMPERED ... is already tampered with. Run --restore first.` | Run `python exercises/acceptance/tamper.py --restore`. |
| `AUDIT ERROR` or the dependency gate says `EVIDENCE MISSING` | pip-audit could not reach PyPI. Check the internet connection of the codespace and run it again. |
| The model card gate says `it describes version N, not M` | Your card names another version. Write the version from `curl -s http://127.0.0.1:8000/health` into `**Model version:**`. |
| `REFUSED: dossier.md was changed after accept.py wrote it` | Run `accept.py` again; do not edit the dossier by hand. The decision goes in with `decide.py`. |
| The good model breaks the SLO in `accept.py` | The codespace was busy. Close other load tests, wait a minute and run `accept.py` again. |
| pytest says `import file mismatch` | You collected `test_access.py` and `solution/test_access.py` in one run. Run one file at a time. |
| You pushed to your fork, but no workflow run started | Actions are off in a new fork. Open the **Actions** tab of your fork and enable them. Then start a run with **Run workflow** (step 10). |
| There is no **Run workflow** button | Make `09-acceptance` the default branch of your fork (**Settings → General → Default branch**), then reload the **Actions** page. |
| **Ship decision** ran without asking you | Your fork's environment `production` has no required reviewer. Add yourself (step 10), then run the workflow again. |
| The push is rejected with `refusing to allow ... to create or update workflow` | You changed a file in `.github/workflows/`. Undo your change to that file, commit, and push again. |
| You closed the browser tab | Go to [github.com/codespaces](https://github.com/codespaces) and open your codespace again. Your work is still there. Start the MLflow server and the container again, and load the keys. |

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
| Security and acceptance | FastAPI security dependencies (API keys), SHA-256 manifests, skops trusted types, pip-audit for known vulnerabilities, GitHub Actions for the acceptance and the release decision ([`.github/workflows/acceptance.yml`](.github/workflows/acceptance.yml)) |

## Datasets

| Folder | Dataset | Target |
|---|---|---|
| [`f1/data/`](f1/data/) | Synthetic Grand Prix championship: 10 fictional teams, 24 circuits, five seasons, 2,400 car-race rows | `points_finish` |
| [`spl/data/`](spl/data/) | Synthetic Saudi league: 18 clubs per season, 34 rounds, five seasons, 3,060 club-match rows | `win` |

**All results, players, teams and drivers are synthetic.** They were generated from seed 2026 and describe no real season or person. They are not a basis for betting, club or team decisions, or claims about anyone real.

Each dataset folder has a `README.md` with the file list, the column dictionary, which columns are known before versus after the event (leakage), and the facts you can check. Besides the main training and holdout CSV, each folder holds the 2026 provisional and final results, a corrections file, and `scenarios/*.json` windows for monitoring exercises (a reference window, an input-drift window and a training-serving-skew window).
