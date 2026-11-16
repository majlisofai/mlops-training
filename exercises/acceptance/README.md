# Lesson 09: security gates and production-readiness acceptance

In lessons 02 to 05 you built a gate for each risk: data validation, quality gates, contract tests and a load test. Each one ran on its own, and in lesson 05 anyone who could reach the API could reload its model. In this lesson you close the last gaps and put everything together. You protect the API with keys and roles, fingerprint the release so that a changed file cannot slip through, and check the image for known vulnerabilities. Then one command runs every gate of the course and writes a **release dossier**: the evidence, the owners and a decision field that only a named person can set to **ship**. You finish with a trainer who tampers with the release, and an individual scenario check.

Time: about 2 hours.

| Part | What you do | What it shows |
|---|---|---|
| Access control | Protect the API with a client key and an admin key, and write the access tests | 401 without a key, 403 with the wrong role, 200 with the right one |
| Integrity | Sign a SHA-256 manifest of the model, the data and the release criteria | A changed file fails the release, even when the model still works |
| Dependencies | Audit the serving image with pip-audit | The image holds more than your requirements file |
| Acceptance | Run every gate in one command and write the release dossier | Missing evidence means do not ship. Only a person ships. |
| Faults | The trainer tampers with the model and deploys the unsecured image | Each fault is caught by a different gate |
| GitHub Actions | Run the acceptance in your fork, approve the release | The approval of the environment is the human decision |
| Scenario check | Decide eight release cases alone | You can read a dossier and defend a decision |

Run every command from the repository root, `/workspaces/mlops-training`.

## What you will have at the end

- An API that refuses anonymous callers: a client key for predictions, an admin key for reloads, keys only in environment variables.
- Eight access tests in `test_access.py`, two of them written by you, and two TODOs in `app.py` that protect the endpoints.
- A signed manifest: the SHA-256 of every model file, the data and the release criteria, logged in MLflow.
- A release dossier with ten gates and your decision, and a run where the same suite blocks a tampered model.
- A GitHub Actions run that you approved, and one that blocked a fault.

## How the pieces fit

```
             lesson 04 pipeline                 lesson 09
candidate --> validate, train, gates, promote --> manifest.py sign --> @champion version N, signed
                                                                            |
       keys.py --> .secrets/api.env --> docker run --env-file  -->  API (app.py + security.py)
                                                                            |
                    accept.py: 10 gates against MLflow, the files and the running API
                                                                            |
                              dossier.md: evidence, owners, decision = PENDING or do not ship
                                                                            |
                                         decide.py --approver "a named person" --> ship
```

| File | What it does |
|---|---|
| [`security.py`](security.py) | API keys with two roles: `client` and `admin`. You do not need to change it. |
| [`app.py`](app.py) | The lesson 05 API, plus keys. **You complete TODO 1 and TODO 2.** |
| [`test_access.py`](test_access.py), [`conftest.py`](conftest.py) | The access tests. **You complete TODO 3 and TODO 4.** |
| [`keys.py`](keys.py) | Makes a client key and an admin key in `.secrets/api.env`, which git ignores |
| [`Dockerfile`](Dockerfile) | The image of the secured API. It holds no keys and no pip. |
| [`manifest.py`](manifest.py) | `sign`: SHA-256 manifest of the release, logged and tagged in MLflow. `verify`: hash again and compare. |
| [`tamper.py`](tamper.py) | For the trainer: changes the model file or the data after the sign-off |
| [`audit_deps.py`](audit_deps.py) | pip-audit of the packages inside the image |
| [`subgroup_check.py`](subgroup_check.py) | Accuracy for each reporting subgroup, written to `evidence/fairness_report.json` |
| [`model_card_template.md`](model_card_template.md) | The model card you fill in, to `evidence/model_card.md` |
| [`release.yaml`](release.yaml) | The release: owners, approvers, files to fingerprint, accepted findings, evidence paths |
| [`accept.py`](accept.py) | Runs the ten gates and writes `dossier.md` and `dossier.json` |
| [`decide.py`](decide.py) | Sets the decision. Only a named approver can set **ship**, and only when every gate passed. |
| [`scenarios.md`](scenarios.md) | The individual scenario check |
| [`.github/workflows/acceptance.yml`](../../.github/workflows/acceptance.yml) | The same acceptance on GitHub Actions, with a fault to inject and an approval |

## Step 1: start MLflow, promote a champion, name the owners

Start the MLflow server in a first terminal, with the same command as in lessons 02 to 05:

```bash
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --allowed-hosts "*.app.github.dev,localhost:*,127.0.0.1:*" \
  --cors-allowed-origins "https://*.app.github.dev"
```

In a second terminal, promote the good candidate with the lesson 04 pipeline and the reference gates:

```bash
GATES=exercises/quality-gates/solution/test_gates.py bash exercises/quality-gates/run_pipeline.sh
```

> **Check:** seven `PASSED` gates (accuracy 0.842, roc_auc 0.907, margin 0.037, rookie_accuracy 0.646), then
> `PROMOTED: f1-points-finish version 1 is now @champion (previous champion: none).`

Open [`release.yaml`](release.yaml) and replace every `your-name` under `owners` and `approvers` with your name, exactly as you will type it later. In a real team these are four or five different people. The version numbers in this lesson are for a new codespace; if yours are higher, that is fine.

## Step 2: make the API keys

An API key is a long random secret that a caller sends with each request. The API has two: a **client** key for applications that ask for predictions, and an **admin** key for the people who reload the model. Make them:

```bash
python exercises/acceptance/keys.py
ls -l .secrets/api.env
git check-ignore -v .secrets/api.env
```

> **Check:** `Wrote a client key and an admin key to .secrets/api.env (git ignores it).`, the file is `-rw-------` (only you can read it), and git names the rule that ignores it: `.gitignore:18:.secrets/	.secrets/api.env`.

Load the keys into this terminal. Do this again in every new terminal you open in this lesson:

```bash
set -a; source .secrets/api.env; set +a
```

Some rules, and why:

- **Keys come from the environment, never from the code, the image or git.** Anything in git stays in its history and in every fork. `keys.py` refuses to write a file that git does not ignore.
- **The API fails closed.** Without both keys, or with keys shorter than 32 characters, it does not start. An API that starts without keys would be open to everyone, and nobody would notice.
- **The API keeps only the SHA-256 of each key** and compares with `hmac.compare_digest`, which takes the same time whether the first or the last character is wrong.
- **The logs name a key by its fingerprint**, the first 12 hex digits of its SHA-256, never by the key itself.

## Step 3: protect the endpoints and write the access tests

Read [`security.py`](security.py). `require_role("client")` is a FastAPI *dependency*: a function that FastAPI runs before the endpoint. It reads the `X-API-Key` header and answers:

| The request has | Answer | Meaning |
|---|---|---|
| no key, or a key the API does not know | **401** Unauthorized | "Who are you?" |
| a known key with the wrong role | **403** Forbidden | "I know you, but you may not do this." |
| a key with the right role | the endpoint runs | |

Run the access tests as they are. Like the lesson 05 tests, they start the API inside pytest, with throwaway keys:

```bash
pytest exercises/acceptance/test_access.py -rs --tb=line
```

> **Check:** `3 failed, 3 passed, 2 skipped`. The failures say `status 200: /admin/reload answers without a key`, `status 200: /v1/predict answers without a key` and `status 200: /v1/predict accepts an unknown key`. That is the lesson 05 API: open to everyone.

Complete **TODO 1 and TODO 2** in [`app.py`](app.py): add the dependency to `predict()` and to `reload()`, and add the key's fingerprint to the reload log line. Run the tests again.

> **Check:** `6 passed, 2 skipped`.

Complete **TODO 3 and TODO 4** in [`test_access.py`](test_access.py):

| TODO | Test | What it checks |
|---|---|---|
| 3 | `test_wrong_role_is_403` | A client key on `/admin/reload` and an admin key on `/v1/predict` both get 403, with a message about the role |
| 4 | `test_admin_key_can_reload` | An admin key reloads (200), and the version in the answer is the version `/health` names |

> **Check:** `8 passed`.

To see that TODO 3 protects something, break the role check: in `security.py`, change `if caller.role != role:` to `if False:` and run the tests. TODO 3 fails with `status 200: a client key reloaded the model`. Change it back.

Why does the admin key not get predictions too? Least privilege: each key can do one job. If the client key leaks from an application, nobody can change the model with it. If the admin key leaks, nobody can read predictions with it.

## Step 4: build and run the secured container

Build the image from your `app.py`:

```bash
docker build -t lab09-api:1 exercises/acceptance
```

> **Check:** `docker image ls lab09-api` shows `lab09-api:1`, about 830 MB.

Start it once **without** keys:

```bash
docker run --rm --network host lab09-api:1
```

> **Check:** it stops at once with `RuntimeError: API_CLIENT_KEY is not set or shorter than 32 characters.` and `Application startup failed.` It fails closed.

Now start it with the keys. `--env-file` passes the variables in the file to the container when it starts; the image itself never sees them:

```bash
docker run --rm --name lab09-api --network host --env-file .secrets/api.env lab09-api:1
```

> **Check:** `Serving f1-points-finish version 1 (models:/f1-points-finish@champion)`.

In another terminal (load the keys there first), try every combination:

```bash
BODY='{"entries": [
  {"grid": 1, "quali_gap_ms": 0, "driver_form": 2.33, "team_form": 27.0, "team_dnf_rate": 0.083, "rain_prob": 0.24, "circuit_type": "permanent"},
  {"grid": 20, "quali_gap_ms": 1981, "driver_form": 17.0, "team_form": 0.0, "team_dnf_rate": 0.0, "rain_prob": 0.24, "circuit_type": "permanent"}]}'
URL=http://127.0.0.1:8000
curl -s -w '  -> %{http_code}\n' -X POST $URL/v1/predict -H 'Content-Type: application/json' -d "$BODY"
curl -s -w '  -> %{http_code}\n' -X POST $URL/v1/predict -H 'Content-Type: application/json' -H "X-API-Key: $API_CLIENT_KEY" -d "$BODY"
curl -s -w '  -> %{http_code}\n' -X POST $URL/v1/predict -H 'Content-Type: application/json' -H "X-API-Key: $API_ADMIN_KEY" -d "$BODY"
curl -s -w '  -> %{http_code}\n' -X POST $URL/admin/reload
curl -s -w '  -> %{http_code}\n' -X POST $URL/admin/reload -H "X-API-Key: $API_CLIENT_KEY"
curl -s -w '  -> %{http_code}\n' -X POST $URL/admin/reload -H "X-API-Key: wrong-key"
curl -s -w '  -> %{http_code}\n' -X POST $URL/admin/reload -H "X-API-Key: $API_ADMIN_KEY"
```

> **Check:**
>
> ```
> {"detail":"missing API key: send it in the X-API-Key header"}  -> 401
> {"model_name":"f1-points-finish","model_version":"1","predictions":[1,0],"probabilities":[0.958...,0.009...]}  -> 200
> {"detail":"this key has the role 'admin'; /v1/predict needs the role 'client'"}  -> 403
> {"detail":"missing API key: send it in the X-API-Key header"}  -> 401
> {"detail":"this key has the role 'client'; /admin/reload needs the role 'admin'"}  -> 403
> {"detail":"unknown API key"}  -> 401
> {"model_name":"f1-points-finish","previous_version":"1","model_version":"1",...}  -> 200
> ```
>
> The container log says `Reloaded models:/f1-points-finish@champion: version 1 -> 1 by admin key ` and 12 hex digits. Compare them with `python -c "import hashlib,os; print(hashlib.sha256(os.environ['API_ADMIN_KEY'].encode()).hexdigest()[:12])"`.

The same access tests can call the container instead of an API inside pytest. That is how the acceptance suite uses them:

```bash
API_URL=http://127.0.0.1:8000 pytest exercises/acceptance/test_access.py -q --tb=line
```

> **Check:** `8 passed`.

Open the **f1 API** port and add `/docs`. The page now has an **Authorize** button: FastAPI made it from the `X-API-Key` header in `security.py`. `/health` and `/docs` stay public: a load balancer must check the API, and the contract is not a secret.

Leave the container running.

## Step 5: sign the release

The quality gates checked a model **file**. If somebody changes that file after the gates, nothing you built so far notices: the API loads whatever is in the artifact store. A **SHA-256 manifest** is a list of files with the SHA-256 of each one. Change one byte of a file, and its SHA-256 changes completely.

```bash
python exercises/acceptance/manifest.py sign --signer "<your name>"
```

> **Check:** ten files: the seven files of the model version (`model/model.skops`, about 10.6 MB, `model/MLmodel` and five small files), `f1/data/races.csv`, `criteria.yaml` and `slo.yaml`. Then
> `SIGNED: manifest sha256 ... by <your name>. Written to manifest.json, logged to run ... as integrity/manifest-v1.json, tagged integrity.manifest_sha256.`

In the MLflow UI, open **Models → f1-points-finish → Version 1**: it has the tags `integrity.manifest_sha256`, `integrity.signed_by`, `integrity.signed_at` and `integrity.files = 10`. The run of version 1 has the manifest under **Artifacts → integrity**. Run the check:

```bash
python exercises/acceptance/manifest.py verify
```

> **Check:** ten lines `OK`, `skops types in model.skops: sklearn.tree._tree.Tree`, then `INTEGRITY OK: 10 files match the manifest signed by <your name> at ...`.

What the check covers, and why:

- **The model files, downloaded again from the registry.** That is what the API loads.
- **The data and the release criteria.** A model gated against changed data, or against a lowered `min_accuracy` in `criteria.yaml`, was not gated against what was signed.
- **The manifest itself.** The SHA-256 of the manifest is in a model version tag. A manifest that was edited to match a changed file no longer matches the tag.
- **The types inside the model file.** MLflow saved the model with skops. Unlike pickle, which can run any code while it loads, skops builds only types that are trusted. `sklearn.tree._tree.Tree` is not on skops' built-in list, so lesson 03 trusted it. That list is stored in the model's `MLmodel` file, where anyone who can change the model can also change the list. So the allowed list lives in git, in `release.yaml` (`allowed_skops_types`), and `verify` fails on any other type.

`sign` refuses to sign a version again. Signing again would hide a change; `--force` exists for when you know why the files changed.

A SHA-256 manifest is a fingerprint, not a signature: anybody who can write to MLflow could replace both the manifest and the tag. Discuss question 1 is about that.

## Step 6: audit the dependencies

The API runs with about 55 Python packages that you did not write. [pip-audit](https://github.com/pypa/pip-audit) looks each package and version up in PyPI's vulnerability data. Audit the requirements file first, then the image you built:

```bash
python exercises/acceptance/audit_deps.py
python exercises/acceptance/audit_deps.py --image lab09-api:1
```

> **Check:** `Dependency audit of exercises/acceptance/requirements-serve.txt: 55 packages, 0 known vulnerabilities`, and `Dependency audit of image lab09-api:1: 56 packages, 0 known vulnerabilities`. Both say `AUDIT OK`. The two lists differ by one package, `packaging`: the image audit checks exactly what is installed, not what pip-audit resolved from the file.

Now audit the image of lesson 05. Build it under a lesson 09 name:

```bash
docker build -t lab09-unsecured:1 exercises/serve-model
python exercises/acceptance/audit_deps.py --image lab09-unsecured:1
```

> **Check:** `57 packages, 6 known vulnerabilities`, all six in `pip 25.0.1` (fixed in versions 25.3 to 26.2), and `AUDIT FAILED: 6 open findings.` We measured this in November 2026; the base image `python:3.12-slim` changes over time, so your list can differ.

The requirements file was clean, but the image was not: the base image brought its own pip. The API never runs pip, but anyone who gets a shell in the container could. The lesson 09 [`Dockerfile`](Dockerfile) removes it after installing the packages (`pip uninstall --yes pip`), and `audit_deps.py` lists the packages with Python's `importlib.metadata`, so it works without pip.

**Does the audit gate the release?** In this course: yes, for the serving image, because that is what runs in production. A finding fails the gate unless somebody accepts it in `release.yaml` with a reason, an owner and an expiry date. The tools of the codespace are not deployed, so they are reported, not gated. You can audit them too; it takes about 90 seconds:

```bash
python exercises/acceptance/audit_deps.py -r .devcontainer/requirements.txt
```

> **Check:** three distinct advisories in two packages: two in `setuptools 80.10.2` (fixed in 83.0.0 and 84.1.0) and one in `diskcache 5.6.3` (no fix yet), which DVC uses. `setuptools` is pinned below 81 because ydata-profiling needs it. The `diskcache` advisory is about pickle: anybody who can write to its cache folder can run code. That is the same risk that skops protects the model from.

## Step 7: attach the evidence

A release needs more than passing tests: people must be able to see what the model is for, where it fails and who is responsible. Measure the subgroups:

```bash
python exercises/acceptance/subgroup_check.py
```

> **Check:**
>
> ```
>   PASS  rookie=false             432     0.863     0.500  0.869
>   PASS  rookie=true               48     0.646     0.875  0.963
>   PASS  team_tier=backmarker     144     0.979     0.000  0.000
>   PASS  team_tier=midfield       192     0.714     0.599  0.773
>   PASS  team_tier=top            144     0.875     0.993  0.992
> SUBGROUPS OK: accuracy of every subgroup of rookie, team_tier >= 0.6. Written to exercises/acceptance/evidence/fairness_report.json.
> ```

The columns are accuracy, the share predicted to score points, and the share of the cars that did score that the model found (true positive rate). Every subgroup passes the lesson 04 floor of 0.60, but look at the backmarkers: 0.979 accuracy, because the model never predicts points for them. It found none of the three backmarker entries that scored. Lesson 08 goes deeper into fairness; its report can take this file's place (see [`evidence/README.md`](evidence/README.md)).

Write the model card. Copy the template and replace every `TODO`, using the numbers from steps 1 and 7 and from [`f1/data/README.md`](../../f1/data/README.md):

```bash
cp exercises/acceptance/model_card_template.md exercises/acceptance/evidence/model_card.md
```

If you made a model card in lesson 08, use it instead, and add the lines `**Model name:**` and `**Model version:**`.

## Step 8: run the acceptance suite and decide

One command runs every gate of the course against the release: the version behind `@champion` and the API on port 8000. It does not stop at the first failure: the dossier must show all the evidence.

```bash
python exercises/acceptance/accept.py --image lab09-api:1
```

| # | Gate | From lesson | Passes when |
|---|---|---|---|
| 1 | Data validation and lineage | 02, 04 | 0 failure cases, and the data's md5 is the md5 the model was trained on |
| 2 | Integrity | 09 | Every file matches the signed manifest; only allowed skops types |
| 3 | Quality gates | 04 | All seven reference gates pass on `@champion`, measured again |
| 4 | Deployed version | 05 | The API serves the version under acceptance |
| 5 | API contract tests | 05 | The lesson 05 reference tests pass against the container, with a client key |
| 6 | Access tests | 09 | **Your** access tests pass against the container, none skipped |
| 7 | Load test and SLO | 05 | `SLO MET`, with the client key |
| 8 | Dependency audit | 09 | No open finding in the image |
| 9 | Model card | 08, 09 | A card for this version, without `TODO` |
| 10 | Subgroup and fairness evidence | 08 | A report for this version that passed |

A gate has three results: PASS, FAIL, or **EVIDENCE MISSING**: not signed, a skipped test, no model card. In lesson 04 a skipped gate let a cheating model through. Here, a check that did not run is never a pass.

If you have not finished your model card, you see this first:

> **Check:** `MISSING  exercises/acceptance/evidence/model_card.md: 11 TODO left; it describes version TODO, not 1` and `DECISION: DO NOT SHIP (Model card: EVIDENCE MISSING)`.

With the card done, run it again. It takes about 40 seconds, most of it the load test.

> **Check:** ten `PASS` lines, for example:
>
> ```
>    PASS     0 failure cases; md5 199d7bda0a7b, the data the model was trained on
>    PASS     10 files match the manifest signed by <your name>; skops types: sklearn.tree._tree.Tree
>    PASS     7 of 7 gates passed on version 1: accuracy 0.842, roc_auc 0.907, margin 0.037, rookie_accuracy 0.646
>    PASS     http://127.0.0.1:8000 serves f1-points-finish version 1
>    PASS     8 passed, 0 failed, 0 not written (exercises/serve-model/solution/test_api.py)
>    PASS     8 passed, 0 failed, 0 not written (exercises/acceptance/test_access.py)
>    PASS     SLO MET: requests 247, p95 latency 75 ms, failure rate 0.00%
>    PASS     pip-audit, image lab09-api:1: 56 packages, 0 open findings, 0 accepted
>    PASS     exercises/acceptance/evidence/model_card.md, for version 1
>    PASS     5 of 5 subgroup checks passed; lowest: rookie=true accuracy 0.646
> DECISION: PENDING: every gate passed. A named approver decides with decide.py.
> ```

Open `dossier.md` (right-click → **Open Preview**). It has the decision block, the release (model version, links to the MLflow run and model version, the data md5, the manifest, who proposed and promoted the model, the git commit and the image ID), the owners, and the gate table with the evidence and the published controls each gate supports. The logs of every gate are in `results/acceptance/`.

The suite never writes **ship**. Try to ship as somebody who is not an approver:

```bash
python exercises/acceptance/decide.py --decision ship --approver "Somebody Else"
```

> **Check:** `REFUSED: 'Somebody Else' is not an approver in release.yaml (approvers: <your name>).`

Now decide as yourself:

```bash
python exercises/acceptance/decide.py --decision ship --approver "<your name>"
```

> **Check:** `WARNING: the approver is also the model owner: no separation of duties`, then
> `DECISION: SHIP for f1-points-finish version 1, set by <your name> via decide.py.` The decision block in `dossier.md` names you, and the model version in MLflow has the tags `release.decision = ship`, `release.decided_by` and `acceptance.result = passed`.

`decide.py` also refuses to ship when any gate did not pass, when a bot is named, and when `dossier.md` was edited after `accept.py` wrote it (try changing a number in it). Anyone named can set **do not ship**, with `--reason`.

## Step 9: the trainer breaks the release

Your trainer now plays an attacker and a careless colleague. If you work alone, run the commands yourself.

**Fault 1: the model file is replaced.** Somebody with write access to the artifact store swaps the champion's model file:

```bash
python exercises/acceptance/tamper.py --target model
curl -s -X POST http://127.0.0.1:8000/admin/reload -H "X-API-Key: $API_ADMIN_KEY"
```

> **Check:** `TAMPERED: mlartifacts/1/models/m-.../artifacts/model.skops now holds a forest of 30 trees instead of 300. It still loads and answers.` The reload answers 200. The two-car `curl` of step 4 still answers `"predictions":[1,0]`, with probability 0.933 for the pole-sitter instead of 0.958.

Run the acceptance suite:

```bash
python exercises/acceptance/accept.py --image lab09-api:1
```

> **Check:** nine gates pass, and one fails:
>
> ```
>    FAIL     model/model.skops: CHANGED
>    PASS     7 of 7 gates passed on version 1: accuracy 0.842, roc_auc 0.905, margin 0.037, rookie_accuracy 0.646
>    PASS     SLO MET: requests 254, p95 latency 33 ms, failure rate 0.00%
> DECISION: DO NOT SHIP (Integrity: signed SHA-256 manifest: FAILED)
> ```
>
> `decide.py --decision ship` now answers `REFUSED: the suite says do not ship: ... Nobody can ship this release.`

The tampered model passes every quality gate, the contract tests and the SLO. It is even faster. Only the fingerprint shows that it is not the model that was gated and signed. Put the original back and reload:

```bash
python exercises/acceptance/tamper.py --restore
curl -s -X POST http://127.0.0.1:8000/admin/reload -H "X-API-Key: $API_ADMIN_KEY"
python exercises/acceptance/manifest.py verify
```

> **Check:** `RESTORED: ...model.skops`, then `INTEGRITY OK: 10 files match the manifest ...`.

**Fault 2: the old image is deployed.** A colleague redeploys the lesson 05 image. Stop your container with **Ctrl+C** and start the old one in its place:

```bash
docker run --rm --name lab09-unsecured --network host lab09-unsecured:1
```

Run `accept.py --image lab09-unsecured:1` in the other terminal.

> **Check:**
>
> ```
>    PASS     8 passed, 0 failed, 0 not written (exercises/serve-model/solution/test_api.py)
>    FAIL     4 passed, 4 failed, 0 not written (exercises/acceptance/test_access.py); AssertionError: status 200:
>             /v1/predict answers without a key; ...
>    PASS     SLO MET: requests 261, p95 latency 97 ms, failure rate 0.00%
>    FAIL     pip-audit, image lab09-unsecured:1: 57 packages, 6 open findings, 0 accepted; ...
> DECISION: DO NOT SHIP (Access tests: FAILED; Dependency audit: FAILED)
> ```

The contract tests and the load test pass: they send a client key, and the old API ignores it. Only the access tests ask what happens **without** a key. Stop the old container (**Ctrl+C**) and start `lab09-api:1` again, as in step 4.

## Step 10: run the acceptance on GitHub Actions

The workflow [`acceptance.yml`](../../.github/workflows/acceptance.yml) runs the whole lesson in two jobs:

| Job | Step | What it does |
|---|---|---|
| **Acceptance suite** | Make throwaway API keys | `keys.py` in the runner. The keys are masked in every log and die with the runner. |
| | 1. Promote the candidate | The lesson 04 pipeline, version 1 becomes `@champion` |
| | 2. Sign the manifest | `manifest.py sign`, signed by `github-actions (run <id>)` |
| | 3. Attach the evidence | `subgroup_check.py`; with the reference files, the reference model card. Your own card if you committed `evidence/model_card.md`. |
| | 4. Inject the fault | `tamper` only: `tamper.py --target model` |
| | 5. Build the Docker image | Your `app.py` (or the reference); `no-auth`: the lesson 05 image |
| | 6. Run the container | `--network host --env-file .secrets/api.env` |
| | 7. Acceptance suite | `accept.py`. The gate table goes to the run page; the job fails when the decision is do not ship. |
| **Ship decision** | 8. Record the decision | Runs only when every gate passed. It uses the environment `production`, so it waits for a reviewer. `decide.py` records who approved. |

The dossier, the logs, the manifest and the evidence are kept as the artifact `release-dossier`; after the approval, `release-dossier-decided` holds the dossier with the decision.

Why throwaway keys? The keys protect the API inside this one job, for a few minutes. A repository secret would add nothing, and a key that is never stored cannot leak. A real deployment reads its keys from a secret store, for example a GitHub environment secret, instead of `keys.py`.

1. **Commit your work.** Commit your model card too: it is part of the release.

   ```bash
   git add exercises/acceptance
   git commit -m "Secure the API and write the access tests"
   ```

2. **Push to your fork**, as in lesson 05 step 9 (**Sync Changes**, **Create Fork** if you have none, enable Actions in a new fork).
3. **Set up the approval.** In your fork, open **Settings → Environments → production** (create it as in lesson 04 step 5 if it does not exist), and add yourself as a required reviewer.
4. **Make `09-acceptance` the default branch of your fork**, so that GitHub shows the **Run workflow** button.

Your push started the workflow **acceptance** with your files and no fault.

> **Check:** the job **Acceptance suite** passes and the run page shows `Acceptance: f1-points-finish version 1: ready for a human decision` with ten gates PASS. The job **Ship decision** waits: **Waiting for review: production needs approval**. Read the gate table, then click **Review deployments → production → Approve and deploy**. The job prints `DECISION: SHIP for f1-points-finish version 1, set by <your-user> via github-environment production (approved by a required reviewer).`

Now inject a fault. Open **Actions → acceptance → Run workflow**, choose the fault `tamper` (or `no-auth`), and click **Run workflow**.

> **Check:** step **7. Acceptance suite** fails with `DECISION: DO NOT SHIP (Integrity: signed SHA-256 manifest: FAILED)`, and the job **Ship decision** is skipped: there is nothing to approve.

The workflow input `files` chooses whose `app.py` and access tests run: yours (`trainee`, the default) or the reference solution. Before you finish TODO 1 to TODO 4, a run with your files says do not ship at the access tests: that is the gate doing its job.

## Step 11: scenario check

Open [`scenarios.md`](scenarios.md). For each of the eight cards, write down alone: the decision, the evidence that decides it, and your next action. Then compare with your group and your trainer.

## Discuss

1. The manifest's fingerprint is a tag in MLflow, next to the files it protects. Who could change both the model file and the tag? Where would you keep the fingerprint, or a real signature, so that the person who can change the model cannot also change the proof?
2. Each role has one shared key. Compare that with one key per client application, and with OAuth2 tokens from an identity provider. What happens in each case when one application's key leaks?
3. The serving image has no known vulnerability, the codespace tools have three. One of them, in `diskcache`, has no fix. Should a finding without a fix block a release? Who may accept it, and for how long?
4. In the course repository, the environment `production` has no required reviewers, so the **Ship decision** job runs without anybody approving it, and `decide.py` writes `NO REVIEW RECORDED`. Is that still a human decision? What would an auditor say?
5. Which gate would you drop if the suite took too long, and what could then slip through? Use the faults of step 9 to argue.

## Stretch

- Sign the manifest for real: in `manifest.py`, compute an HMAC-SHA256 of the manifest with a key that only the release manager has (`hmac.new(key, data, "sha256")`), store the HMAC instead of the plain SHA-256, and check it in `verify`. Who keeps the key?
- Bind the evidence to the release: make `subgroup_check.py` write the manifest SHA-256 into its report, and make `accept.py` refuse a report for another manifest. Then run `tamper.py --target model` again: which gates fail now?
- Run `tamper.py --target data`. Which two gates fail, and why does data validation still find 0 failure cases?
- Give each client application its own key: replace `API_CLIENT_KEY` with a file of named keys, and log the name of the client with each prediction.
- Check the running image: add a gate to `accept.py` that compares the image ID of the running container (`docker inspect lab09-api`) with the image the dossier names.

## Stuck?

There are reference solutions in [`solution/`](solution/): `app.py` (TODO 1 and 2), `test_access.py` (TODO 3 and 4), `model_card.md`, and the answer key for the scenarios, `scenarios.md`. Try them yourself first. To run the reference access tests, and to build and accept the reference image:

```bash
pytest exercises/acceptance/solution/test_access.py
docker build -t lab09-api:ref --build-arg APP=solution/app.py exercises/acceptance
python exercises/acceptance/accept.py --reference --image lab09-api:ref
```

Run one test file at a time: pytest cannot collect `test_access.py` and `solution/test_access.py` in one run, because the two files have the same name.

## Sources

- NIST, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, January 2023, <https://doi.org/10.6028/NIST.AI.100-1>. Section 3.3 (secure and resilient), Table 1 (GOVERN 2.1), Table 2 (MAP 2.3), Table 3 (MEASURE 1.3, 2.3, 2.5, 2.7, 2.8, 2.9, 2.11) and Table 4 (MANAGE 1.1, 3.1). The dossier maps each gate to these subcategories.
- SDAIA, *AI Ethics Principles* (September 2023), <https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf>. The dossier uses five of its seven principles by name: Fairness; Privacy & Security; Reliability & Safety; Transparency & Explainability; Accountability & Responsibility. We took the names from the summary at <https://regulations.ai/regulations/RAI-SA-NA-PCAESXX-2023>, because the SDAIA site refused our automated download; read the original before you quote it.
- The NCA Essential Cybersecurity Controls (ECC-2:2024) are not mapped here: whether they apply to this system is still to be confirmed.
- skops, "Secure persistence with skops", <https://skops.readthedocs.io/en/stable/persistence.html>: why pickle is unsafe, trusted types and `get_untrusted_types`.
- pip-audit, <https://github.com/pypa/pip-audit>: what it audits, its vulnerability services, `--disable-pip` and `--no-deps`, and its exit codes.
- FastAPI, "Security tools" reference, <https://fastapi.tiangolo.com/reference/security/>: `APIKeyHeader` and `auto_error`.
- GitHub Docs, "Workflow commands for GitHub Actions", <https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands>: masking a value with `add-mask`, `GITHUB_ENV` and `GITHUB_STEP_SUMMARY`.
